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
    write_trace_artifacts,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CanonicalExecutionError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


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

    def predict_action_with_consistent_mask(*call_args: Any, **call_kwargs: Any) -> Any:
        input_ids = call_kwargs.get("input_ids")
        if input_ids is None and call_args:
            input_ids = call_args[0]
        attention_mask = call_kwargs.get("attention_mask")
        if input_ids is not None and attention_mask is not None:
            if int(input_ids.shape[1]) == int(attention_mask.shape[1]) and not torch.all(input_ids[:, -1] == 29871):
                call_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
                )
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
    os.environ.setdefault("MUJOCO_GL", "egl")
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

    adapter, _model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    horizon = int(contract["suite_horizon"])
    num_steps_wait = int(contract["num_steps_wait"])

    def env_factory() -> Any:
        return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

    def policy(step: int, obs: Any, task_label: str) -> PolicyStep:
        image = get_libero_image(obs, 224)
        action, _generation, meta = adapter.predict_action_with_scores(image, task_label)
        tokens = tuple(int(item) for item in meta["captured_action_token_ids"])
        if len(tokens) != 7 or int(meta["captured_score_count"]) != 7 or meta.get("single_generation_parity_pass") is not True:
            raise CanonicalExecutionError("OFFICIAL_SINGLE_GENERATION_CONTRACT_FAIL")
        return PolicyStep(raw_action=action, token_ids=tokens, metadata=meta)

    def physical_state(env: Any, _obs: Any, _step: int) -> dict[str, Any]:
        data = env.sim.data
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "time": float(data.time),
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
        success_predicate=success,
        termination_predicate=lambda _env, _obs, _info, done, _step: bool(done),
    )
    supplied = candidate.get("initial_state_sha256")
    if supplied and str(supplied) != core.initial_state_sha256:
        raise CanonicalExecutionError("CANDIDATE_CANONICAL_INITIAL_STATE_SHA256_MISMATCH")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace = core.run_clean_episode(mode=args.mode)
    trace_root = output_dir / "trace"
    artifacts, hashes = write_trace_artifacts(trace_root, trace)
    receipt = core.build_receipt(
        trace=trace,
        mode=args.mode,
        comparison_scope="CLEAN_PATH",
        contract=contract,
        trace_artifacts=artifacts,
        trace_hashes=hashes,
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
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, KeyError, ValueError, CanonicalExecutionError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
