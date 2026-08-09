#!/usr/bin/env python3
"""Run one clean episode through the shared CanonicalExecutionCore.

This is a diagnostic runner only.  It has no intervention branch and refuses
to start unless ``--enable-runtime`` is supplied explicitly.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
from scripts.detector_v5.run_stage_v_m1_v2_8gpu import canonical_gpu_uuid  # noqa: E402
CANONICAL_ACTION_DECODE_CONTRACT = (
    "OfficialOpenVLAActionAdapter.predict_action_with_scores_single_generation;"
    "attention_implementation=eager;predict_action_attention_mask_append=one_if_input_ids_appended"
)

from gripper_attack.stage_v_canonical_execution_core import (  # noqa: E402
    CANONICAL_INIT_STATE_HASH_ALGORITHM,
    CANONICAL_INIT_STATE_SCHEMA,
    CONTRACT_FIELDS,
    CanonicalExecutionCore,
    CanonicalExecutionError,
    PolicyStep,
    sha256_file,
    write_diagnostic_trace_artifacts,
    write_trace_artifacts,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CanonicalExecutionError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _load_raw_capture_plan(path: Path, identity: Mapping[str, Any], horizon: int, gpu: int | None = None) -> tuple[dict[str, Any], frozenset[int]]:
    plan = _load(path)
    if plan.get("schema") not in {"STAGE_V_M1_RAW_CAPTURE_PLAN_V1", "STAGE_V_M1_V2_RAW_CAPTURE_PLAN_V1", "STAGE_V_M1_V2_1_RAW_CAPTURE_PLAN_V1", "STAGE_V_M1_V2_1_1_RAW_CAPTURE_PLAN_V1"}:
        raise CanonicalExecutionError("RAW_CAPTURE_PLAN_SCHEMA_INVALID")
    if plan.get("status") != "FROZEN_BEFORE_RAW_CAPTURE_RUN":
        raise CanonicalExecutionError("RAW_CAPTURE_PLAN_NOT_FROZEN")
    if plan.get("identity") != identity.get("canonical_parent_key"):
        raise CanonicalExecutionError("RAW_CAPTURE_PLAN_IDENTITY_MISMATCH")
    if plan.get("schema") in {"STAGE_V_M1_V2_RAW_CAPTURE_PLAN_V1", "STAGE_V_M1_V2_1_RAW_CAPTURE_PLAN_V1", "STAGE_V_M1_V2_1_1_RAW_CAPTURE_PLAN_V1"}:
        if gpu is None or str(gpu) not in plan.get("capture_steps_by_gpu", {}):
            raise CanonicalExecutionError("RAW_CAPTURE_PLAN_GPU_MISSING")
        source_steps = plan["capture_steps_by_gpu"][str(gpu)]
    else:
        source_steps = plan.get("capture_steps", [])
    steps = frozenset(int(step) for step in source_steps)
    if not steps or min(steps) < 0 or max(steps) >= horizon:
        raise CanonicalExecutionError("RAW_CAPTURE_PLAN_STEP_INVALID")
    return plan, steps


def _write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_external_modules(snapshot_root: Path, upstream_root: Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    official_src = (snapshot_root / "src").resolve()
    if not official_src.is_dir() or official_src.is_symlink():
        raise CanonicalExecutionError("OFFICIAL_SNAPSHOT_SRC_INVALID")
    if not upstream_root.is_dir() or upstream_root.is_symlink():
        raise CanonicalExecutionError("UPSTREAM_ROOT_INVALID")
    import gripper_attack
    external_package = official_src / "gripper_attack"
    if not external_package.is_dir():
        raise CanonicalExecutionError("OFFICIAL_SNAPSHOT_PACKAGE_MISSING")
    if str(external_package) not in [str(item) for item in gripper_attack.__path__]:
        gripper_attack.__path__.append(str(external_package))
    sys.path.insert(0, str(upstream_root))
    from experiments.robot.libero.libero_utils import get_libero_image
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model
    from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    return get_libero_image, get_processor, get_model, OfficialOpenVLAActionAdapter, benchmark, (get_libero_path, OffScreenRenderEnv)


def _load_policy(args: argparse.Namespace, get_processor: Any, get_model: Any, adapter_type: Any) -> tuple[Any, Any, Any, str]:
    """Load the pinned model without inheriting an unavailable FlashAttention dependency."""
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        # The upstream imports may already have registered the same classes.
        pass
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(args.model_path),
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        load_in_4bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    processor = get_processor(cfg)
    model.eval()

    # The pinned Prismatic predict_action appends the empty action token to input_ids
    # but leaves attention_mask unchanged; eager attention rejects that one-token
    # mismatch, while the legacy FlashAttention path did not expose it.
    original_predict_action = model.predict_action
    model._stage_v_last_model_inputs = None

    def predict_action_with_consistent_mask(*call_args: Any, **call_kwargs: Any) -> Any:
        input_ids = call_kwargs.get("input_ids")
        if input_ids is None and call_args:
            input_ids = call_args[0]
        attention_mask = call_kwargs.get("attention_mask")
        recorded_input_ids = input_ids
        recorded_attention_mask = attention_mask
        if input_ids is not None and attention_mask is not None:
            if int(input_ids.shape[1]) == int(attention_mask.shape[1]) and not torch.all(input_ids[:, -1] == 29871):
                recorded_input_ids = torch.cat([input_ids, torch.full_like(input_ids[:, :1], 29871)], dim=1)
                call_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
                )
                recorded_attention_mask = call_kwargs["attention_mask"]
        model._stage_v_last_model_inputs = {
            key: value.detach().cpu()
            for key, value in {
                "pixel_values": call_kwargs.get("pixel_values"),
                "input_ids": recorded_input_ids,
                "attention_mask": recorded_attention_mask,
            }.items()
            if value is not None
        }
        return original_predict_action(*call_args, **call_kwargs)

    model.predict_action = predict_action_with_consistent_mask
    stats = getattr(model, "norm_stats", {})
    key = args.suite
    if key not in stats and f"{key}_no_noops" in stats:
        key = f"{key}_no_noops"
    if key not in stats:
        raise CanonicalExecutionError(f"UNNORM_KEY_MISSING:{args.suite}")
    adapter = adapter_type(model, processor, device, key, center_crop=True, base_vla_name=str(args.model_path))
    return adapter, model, processor, key


def _render_binding_ids(env: Any) -> tuple[int | None, int | None]:
    observed: int | None = None
    context_observed: int | None = None
    context = getattr(getattr(env, "sim", None), "render_context", None)
    for obj in (env, getattr(env, "sim", None), context):
        for name in ("render_gpu_device_id", "gpu_device_id", "device_id"):
            value = getattr(obj, name, None) if obj is not None else None
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if observed is None:
                observed = value
            if obj is context and context_observed is None:
                context_observed = value
    return observed, context_observed


def _write_runtime_binding_receipt(args: argparse.Namespace, env: Any, output_dir: Path) -> None:
    import torch

    observed, context_observed = _render_binding_ids(env)
    env_device_attribute = getattr(env, "render_gpu_device_id", None)
    env_device = env_device_attribute
    try:
        env_device = int(env_device)
    except (TypeError, ValueError):
        env_device = observed
    if observed != int(args.gpu) or env_device != int(args.gpu):
        raise CanonicalExecutionError("RUNTIME_EGL_DEVICE_BINDING_UNVERIFIED")
    if not torch.cuda.is_available() or int(torch.cuda.current_device()) != 0:
        raise CanonicalExecutionError("RUNTIME_CUDA_LOGICAL_DEVICE_UNVERIFIED")
    properties = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(properties, "uuid", "")).strip()
    if not device_uuid:
        raise CanonicalExecutionError("RUNTIME_GPU_UUID_UNAVAILABLE")
    receipt = {
        "schema": "STAGE_V_M1_V2_1_1_RUNTIME_BINDING_RECEIPT_V1",
        "status": "PASS",
        "logical_worker_id": f"worker_{int(args.gpu)}",
        "requested_physical_gpu": int(args.gpu),
        "physical_gpu_index": int(args.gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_uuid": device_uuid,
        "torch_device_uuid_canonical": canonical_gpu_uuid(device_uuid),
        "torch_device_name": torch.cuda.get_device_name(0),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "env_render_gpu_device_id": env_device,
        "env_render_gpu_device_id_attribute_present": env_device_attribute is not None,
        "egl_binding_source": "env.render_gpu_device_id" if env_device_attribute is not None else "renderer_device_observation",
        "render_context_observed_device_id": context_observed,
        "renderer_device_information": {
            "env_class": type(env).__name__,
            "sim_class": type(getattr(env, "sim", None)).__name__,
            "render_context_class": type(getattr(getattr(env, "sim", None), "render_context", None)).__name__,
            "observed_device_id": observed,
        },
        "source_commit": str(args.source_commit),
        "source_tree": str(args.source_tree),
        "run_label": str(args.run_label or "UNSPECIFIED"),
        "run_set": str(args.run_set or "UNSPECIFIED"),
        "pid": os.getpid(),
        "episode_started": False,
        "receipt_written_before_step_0": True,
    }
    _write(output_dir / "M1_V2_RUNTIME_BINDING_RECEIPT.json", receipt)


def run(args: argparse.Namespace) -> int:
    if not args.enable_runtime:
        raise CanonicalExecutionError("RUNTIME_DISABLED_UNTIL_EXPLICIT_RB1_CANARY_AUTHORIZATION")
    if args.mode not in {"CLEAN_QUALIFICATION", "COUNTERFACTUAL_CLEAN_PREFIX"}:
        raise CanonicalExecutionError("INTERVENTION_MODE_NOT_SUPPORTED_BY_CLEAN_RUNNER")
    candidate = _load(args.candidate)
    contract = _load(args.contract)
    missing = [field for field in CONTRACT_FIELDS if field not in contract]
    extra = sorted(set(contract) - set(CONTRACT_FIELDS))
    if missing or extra:
        raise CanonicalExecutionError(f"CONTRACT_FIELDS_INVALID:missing={missing}:extra={extra}")
    if contract["clean_core_sha256"] != sha256_file(REPO_ROOT / "src/gripper_attack/stage_v_canonical_execution_core.py"):
        raise CanonicalExecutionError("CORE_SHA256_MISMATCH")
    if contract["runner_sha256"] != sha256_file(Path(__file__)):
        raise CanonicalExecutionError("RUNNER_SHA256_MISMATCH")
    if contract["initial_state_hash_algorithm"] != CANONICAL_INIT_STATE_HASH_ALGORITHM or contract["initial_state_identity_schema"] != CANONICAL_INIT_STATE_SCHEMA:
        raise CanonicalExecutionError("INITIAL_STATE_CONTRACT_MISMATCH")
    if contract["action_decode_contract"] != CANONICAL_ACTION_DECODE_CONTRACT:
        raise CanonicalExecutionError("ACTION_DECODE_CONTRACT_MISMATCH")
    if str(contract["source_commit"]) != str(args.source_commit) or str(contract["source_tree"]) != str(args.source_tree):
        raise CanonicalExecutionError("SOURCE_BINDING_MISMATCH")
    if int(contract["seed"]) != int(args.seed) or int(contract["num_steps_wait"]) < 0:
        raise CanonicalExecutionError("SEED_OR_WARMUP_CONTRACT_MISMATCH")
    if str(candidate.get("suite")) != args.suite:
        raise CanonicalExecutionError("CANDIDATE_SUITE_MISMATCH")
    suite = args.suite
    task_index = int(candidate["task_index"])
    state_index = int(candidate["state_index"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    get_libero_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_instance = benchmark.get_benchmark_dict()[suite]()
    task = suite_instance.get_task(task_index)
    initial_state = copy.deepcopy(suite_instance.get_task_init_states(task_index)[state_index])
    identity = {
        "canonical_parent_key": str(candidate["canonical_parent_key"]),
        "suite": suite,
        "task_index": task_index,
        "state_index": state_index,
    }

    adapter, model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    horizon = int(contract["suite_horizon"])
    num_steps_wait = int(contract["num_steps_wait"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    def env_factory() -> Any:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=int(args.gpu),
        )
        try:
            _write_runtime_binding_receipt(args, env, output_dir)
        except BaseException:
            env.close()
            raise
        return env

    def policy(step: int, obs: Any, task_label: str) -> PolicyStep:
        image = get_libero_image(obs, 224)
        action, _generation, meta = adapter.predict_action_with_scores(image, task_label)
        tokens = tuple(int(item) for item in meta["captured_action_token_ids"])
        if len(tokens) != 7 or int(meta["captured_score_count"]) != 7 or meta.get("single_generation_parity_pass") is not True:
            raise CanonicalExecutionError("OFFICIAL_SINGLE_GENERATION_CONTRACT_FAIL")
        policy_capture["policy_rgb_224"] = image.copy()
        policy_capture["model_inputs"] = getattr(model, "_stage_v_last_model_inputs", None)
        if policy_capture["model_inputs"] is None:
            raise CanonicalExecutionError("MODEL_INPUT_DIAGNOSTIC_CAPTURE_MISSING")
        return PolicyStep(raw_action=action, token_ids=tokens, metadata=meta)

    def physical_state(env: Any, _obs: Any, _step: int) -> dict[str, Any]:
        data = env.sim.data
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "time": float(data.time),
        }

    def full_sim_state(env: Any) -> dict[str, Any]:
        sim = env.sim
        data = sim.data
        state: dict[str, Any] = {"schema": "STAGE_V_FULL_SIM_STATE_DIAGNOSTIC_V1", "data": {}}
        state_fields = (
            "qpos", "qvel", "qacc", "qacc_warmstart", "act", "ctrl",
            "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
        )
        for field in state_fields:
            if hasattr(data, field):
                value = getattr(data, field)
                state["data"][field] = value.copy() if hasattr(value, "copy") else value
        if hasattr(sim, "get_state"):
            sim_state = sim.get_state()
            state["sim_state"] = {
                field: getattr(sim_state, field)
                for field in ("time", "qpos", "qvel", "act", "udd_state")
                if hasattr(sim_state, field)
            }
        return state

    policy_capture: dict[str, Any] = {}
    raw_capture_plan: dict[str, Any] | None = None
    raw_capture_steps: frozenset[int] = frozenset()
    if args.raw_capture_plan is not None:
        raw_capture_plan, raw_capture_steps = _load_raw_capture_plan(args.raw_capture_plan.resolve(), identity, horizon, args.gpu)

    def diagnostic(env: Any, _obs: Any, _step: int, _policy_step: PolicyStep) -> dict[str, Any]:
        return {
            "full_sim_state": full_sim_state(env),
            "policy_rgb_224": policy_capture["policy_rgb_224"],
            "model_inputs": policy_capture["model_inputs"],
        }

    def raw_capture(_env: Any, obs: Any, _step: int, _policy_step: PolicyStep) -> dict[str, Any]:
        return {
            "raw_observation": obs,
            "policy_rgb_224": policy_capture["policy_rgb_224"],
            "model_inputs": policy_capture["model_inputs"],
        }

    def success(env: Any, _obs: Any, _info: Mapping[str, Any], done: bool, step: int) -> bool:
        if done:
            return True
        return bool(step == horizon - 1 and hasattr(env, "check_success") and env.check_success())

    core = CanonicalExecutionCore(
        env_factory=env_factory,
        policy=policy,
        action_postprocess=adapter.postprocess,
        initial_state=initial_state,
        identity=identity,
        task_label=str(task.language),
        seed=int(args.seed),
        num_steps_wait=num_steps_wait,
        suite_horizon=horizon,
        observation_getter=lambda _env, obs, _step: obs,
        physical_state_getter=physical_state,
        diagnostic_getter=diagnostic,
        raw_capture_getter=raw_capture if raw_capture_plan is not None else None,
        raw_capture_steps=raw_capture_steps,
        success_predicate=success,
        termination_predicate=lambda _env, _obs, _info, done, _step: bool(done),
    )
    supplied = candidate.get("initial_state_sha256")
    if supplied and str(supplied) != core.initial_state_sha256:
        raise CanonicalExecutionError("CANDIDATE_CANONICAL_INITIAL_STATE_SHA256_MISMATCH")
    trace = core.run_clean_episode(mode=args.mode)
    trace_root = output_dir / "trace"
    artifacts, hashes = write_trace_artifacts(trace_root, trace)
    diagnostic_artifacts, diagnostic_hashes = write_diagnostic_trace_artifacts(trace_root, trace)
    if raw_capture_plan is not None:
        from gripper_attack.stage_v_canonical_execution_core import write_raw_capture_artifacts
        raw_manifest = write_raw_capture_artifacts(trace_root / "raw_capture", trace)
        raw_manifest.update({
            "plan_sha256": sha256_file(args.raw_capture_plan.resolve()),
            "source_commit": str(args.source_commit),
            "source_tree": str(args.source_tree),
            "identity": str(candidate["canonical_parent_key"]),
        })
        _write(output_dir / "M1_RAW_CAPTURE_MANIFEST.json", raw_manifest)
    receipt = core.build_receipt(
        trace=trace,
        mode=args.mode,
        comparison_scope="CLEAN_PATH",
        contract=contract,
        trace_artifacts=artifacts,
        trace_hashes=hashes,
        diagnostic_trace_artifacts=diagnostic_artifacts,
        diagnostic_trace_hashes=diagnostic_hashes,
    )
    receipt["source_commit"] = args.source_commit
    receipt["source_tree"] = args.source_tree
    receipt["runner_sha256"] = sha256_file(Path(__file__))
    _write(output_dir / "RB1_PRODUCER_RECEIPT.json", receipt)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--suite", required=True, choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", required=True, choices=["CLEAN_QUALIFICATION", "COUNTERFACTUAL_CLEAN_PREFIX"])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--run-set")
    parser.add_argument("--enable-runtime", action="store_true")
    parser.add_argument("--raw-capture-plan", type=Path)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, KeyError, ValueError, CanonicalExecutionError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
