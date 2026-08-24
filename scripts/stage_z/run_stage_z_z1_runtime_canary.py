#!/usr/bin/env python3
"""One-cell Stage-Z Z1 engineering runtime canary.

This file deliberately keeps the real runtime behind the existing
ExecutionAuthorization guard.  It never selects a scientific identity and
never reads task success, attack, or physical-outcome signals.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import ModuleSpec
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
MODEL_FAMILIES = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
ACTION_DIM = 7
FREE_MEMORY_MIN_MIB = 20_480
PHASE = "Z1"


class OFTUnnormKeyResolutionError(RuntimeError):
    """Dedicated engineering-invalid failure for missing OFT norm statistics."""


def resolve_official_unnorm_key(norm_stats: dict[str, Any], suite: str) -> tuple[str, str]:
    """Match frozen OpenVLA-OFT ``check_unnorm_key`` semantics exactly."""
    candidate = suite
    if candidate in norm_stats:
        return candidate, "EXACT_SUITE_KEY"
    fallback = f"{candidate}_no_noops"
    if fallback in norm_stats:
        return fallback, "OFFICIAL_NO_NOOPS_FALLBACK"
    raise OFTUnnormKeyResolutionError(f"MODEL_UNNORM_KEY_MISSING:{suite}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_value(path: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", path, *args], text=True).strip()


def validate_canary(ledger: dict[str, Any], parent_key: str, suite: str, role: str) -> dict[str, Any]:
    if ledger.get("status") != "STAGE_Z_Z1_ENGINEERING_CANARY_LEDGER_FROZEN":
        raise RuntimeError("CANARY_LEDGER_NOT_FROZEN")
    rows = [row for row in ledger.get("selected", []) if row.get("canonical_parent_key") == parent_key]
    if len(rows) != 1 or rows[0].get("suite") != suite or rows[0].get("role") != role:
        raise RuntimeError("CANARY_IDENTITY_NOT_PREDECLARED")
    row = rows[0]
    if not row.get("permanent_exclusion") or row.get("scientific_use") or row.get("outcome_read"):
        raise RuntimeError("CANARY_EXCLUSION_FIREWALL_INVALID")
    return row


def validate_action(action: Any, *, clip: bool = False) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size != ACTION_DIM or not np.isfinite(values).all():
        raise RuntimeError("FINAL_ACTION_NOT_EXACTLY_SEVEN_FINITE_VALUES")
    if np.any(values < -1.000001) or np.any(values > 1.000001):
        if not clip:
            raise RuntimeError("FINAL_ACTION_OUTSIDE_LIBERO_RANGE")
        # The official robosuite controller clips the input at this boundary.
        values = np.clip(values, -1.0, 1.0).astype(np.float32)
    return values


def gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"GPU_QUERY_INVALID:{query}")
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    processes = []
    for line in apps.splitlines():
        if line.strip():
            parts = [item.strip() for item in line.split(",")]
            processes.append({"pid": parts[0], "name": parts[1], "used_memory_mib": parts[2]})
    result = {
        "index": int(fields[0]),
        "free_memory_mib": int(fields[1]),
        "used_memory_mib": int(fields[2]),
        "utilization_gpu_percent": int(fields[3]),
        "compute_processes": processes,
    }
    if result["free_memory_mib"] <= FREE_MEMORY_MIN_MIB:
        raise RuntimeError(f"GPU_NOT_ELIGIBLE_FREE_MEMORY_MIB:{result['free_memory_mib']}")
    return result


def require_single_visible_gpu(gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(gpu_id):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES_MUST_BE_SINGLE_PHYSICAL_GPU:{gpu_id}:{visible}")


def verify_m1_materialization(
    manifest_path: Path,
    checkpoint: Path,
    suite: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError("M1_MANIFEST_SHA256_MISMATCH")
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS_SEALED_FOUR_OFT_BYTE_MANIFESTS":
        raise RuntimeError("M1_MANIFEST_NOT_SEALED")
    suite_spec = manifest.get("suites", {}).get(suite)
    if not isinstance(suite_spec, dict) or not isinstance(suite_spec.get("rows"), list):
        raise RuntimeError(f"M1_MANIFEST_SUITE_MISSING:{suite}")
    expected: dict[str, dict[str, Any]] = {}
    for row in suite_spec["rows"]:
        relative = str(row["path"]).replace("\\", "/")
        candidate = (checkpoint / relative).resolve()
        if checkpoint.resolve() not in candidate.parents:
            raise RuntimeError(f"M1_MANIFEST_PATH_ESCAPE:{relative}")
        expected[relative] = {"sha256": str(row["sha256"]), "size": int(row["size"])}
    actual_paths = {
        path.relative_to(checkpoint).as_posix(): path
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if set(actual_paths) != set(expected):
        raise RuntimeError("M1_CHECKPOINT_FILE_SET_MISMATCH")
    total_bytes = 0
    for relative, spec in expected.items():
        path = actual_paths[relative]
        size = path.stat().st_size
        if size != spec["size"] or sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"M1_CHECKPOINT_MANIFEST_MISMATCH:{relative}")
        total_bytes += size
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "suite": suite,
        "files": len(expected),
        "bytes": total_bytes,
        "verified": True,
    }


def static_authority(config: dict[str, Any], ledger: dict[str, Any], *, parent_key: str, suite: str, role: str) -> dict[str, Any]:
    if config["status"] != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_RUNTIME_SOURCE_NOT_FROZEN")
    if config["z0r2"]["root_seal_sha256"] != "e6e3db5a9f5e7641d2c09c0b0ca225ca99763494430cdda22830edab1853053d":
        raise RuntimeError("Z0R2_ROOT_BINDING_INVALID")
    row = validate_canary(ledger, parent_key, suite, role)
    common = config["environment"]["common_libero_checkout"]
    if git_value(common, "rev-parse", "HEAD") != config["environment"]["common_libero_commit"]:
        raise RuntimeError("COMMON_LIBERO_COMMIT_MISMATCH")
    if git_value(common, "rev-parse", "HEAD^{tree}") != config["environment"]["common_libero_tree"]:
        raise RuntimeError("COMMON_LIBERO_TREE_MISMATCH")
    if git_value(common, "status", "--short"):
        raise RuntimeError("COMMON_LIBERO_SOURCE_DIRTY")
    for family in ("M1_OPENVLA_OFT", "M2_PI05_LIBERO"):
        spec = config["model_families"][family]
        if git_value(spec["source_checkout"], "rev-parse", "HEAD") != spec["source_commit"]:
            raise RuntimeError(f"{family}_SOURCE_COMMIT_MISMATCH")
        if git_value(spec["source_checkout"], "rev-parse", "HEAD^{tree}") != spec["source_tree"]:
            raise RuntimeError(f"{family}_SOURCE_TREE_MISMATCH")
        if git_value(spec["source_checkout"], "status", "--short"):
            raise RuntimeError(f"{family}_SOURCE_DIRTY")
    return row


def configure_libero(config: dict[str, Any]) -> None:
    os.environ["LIBERO_CONFIG_PATH"] = config["environment"]["libero_config_path"]
    common_root = config["environment"]["common_libero_checkout"]
    common_python_root = str(Path(common_root) / "libero")
    if common_python_root not in sys.path:
        sys.path.insert(0, common_python_root)
    import importlib
    import importlib.util

    official_init = (Path(common_python_root) / "libero" / "__init__.py").resolve()
    parent = importlib.import_module("libero")
    loaded = sys.modules.get("libero.libero")
    if loaded is None or Path(str(getattr(loaded, "__file__", ""))).resolve() != official_init:
        for name in list(sys.modules):
            if name == "libero.libero" or name.startswith("libero.libero."):
                del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            "libero.libero",
            official_init,
            submodule_search_locations=[str(official_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("OFFICIAL_LIBERO_SPEC_UNAVAILABLE")
        module = importlib.util.module_from_spec(spec)
        sys.modules["libero.libero"] = module
        setattr(parent, "libero", module)
        spec.loader.exec_module(module)
    if Path(str(getattr(sys.modules["libero.libero"], "__file__", ""))).resolve() != official_init:
        raise RuntimeError("OFFICIAL_LIBERO_MODULE_PATH_MISMATCH")


def make_libero_env(config: dict[str, Any], suite: str, task_idx: int):
    configure_libero(config)
    common_root = config["environment"]["common_libero_checkout"]
    from libero.libero import benchmark, get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_idx)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=config["environment"]["camera_height"],
        camera_widths=config["environment"]["camera_width"],
        control_freq=config["environment"]["control_freq"],
        render_gpu_device_id=-1,
    )
    env.seed(config["environment"]["env_seed"])
    return env, task_suite, task


def snapshot_state(env) -> np.ndarray:
    return np.asarray(env.get_sim_state(), dtype=np.float64).copy()


def restore_state(env, state: np.ndarray) -> None:
    env.set_state(np.asarray(state, dtype=np.float64))
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def model_observation(obs: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    def image(value: Any) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(value)[::-1, ::-1])
        resized = Image.fromarray(array).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)

    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - float(quat[3] ** 2), 0.0))
    axis = np.zeros(3, dtype=np.float32) if den == 0 else (quat[:3] * (2.0 * np.arccos(quat[3]))) / den
    state = np.concatenate((obs["robot0_eef_pos"], axis, obs["robot0_gripper_qpos"])).astype(np.float32)
    return {"full_image": image(obs["agentview_image"]), "wrist_image": image(obs["robot0_eye_in_hand_image"]), "state": state}


def _install_optional_import_shims() -> None:
    # The official dynamic model module imports these optional packages before
    # the evaluator import below; install the no-op Z1 shims first.
    try:
        import json_numpy  # type: ignore  # noqa: F401
    except ImportError:
        json_numpy = ModuleType("json_numpy")
        json_numpy.__spec__ = ModuleSpec("json_numpy", loader=None)
        json_numpy.patch = lambda: None  # type: ignore[attr-defined]
        sys.modules["json_numpy"] = json_numpy
    try:
        import wandb  # type: ignore  # noqa: F401
    except ImportError:
        wandb = ModuleType("wandb")
        wandb.__spec__ = ModuleSpec("wandb", loader=None)
        sys.modules["wandb"] = wandb


def load_openvla(checkpoint: str, *, oft: bool, suite: str, return_chunk: bool = False):
    source = "/mnt/sdc/dty_user/openvla_attack/repos/openvla-oft-stage-z-e4287e9_20260823"
    sys.path.insert(0, source)
    _install_optional_import_shims()

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, use_fast=False)
    model = ModelClass.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    if oft:
        # Match frozen upstream get_vla: configure the fused backbone for the
        # primary plus wrist image pair before get_vla_action concatenates them.
        model.vision_backbone.set_num_images_in_input(2)
        # Match frozen upstream OpenVLA-OFT: always load the checkpoint's
        # dataset statistics, even when Transformers supplied a config-level
        # norm_stats attribute during from_pretrained().
        from experiments.robot.openvla_utils import _load_dataset_stats  # type: ignore

        _load_dataset_stats(model, checkpoint)
    elif not hasattr(model, "norm_stats"):
        stats = Path(checkpoint) / "dataset_statistics.json"
        model.norm_stats = json.loads(stats.read_text(encoding="utf-8"))
    resolved_unnorm_key, resolution_mode = resolve_official_unnorm_key(model.norm_stats, suite)
    dataset_statistics_path = Path(checkpoint) / "dataset_statistics.json"
    normalization_metadata = {
        "requested_task_suite": suite,
        "available_norm_stats_keys": sorted(str(key) for key in model.norm_stats),
        "resolved_unnorm_key": resolved_unnorm_key,
        "resolution_mode": resolution_mode,
        "dataset_statistics_path": str(dataset_statistics_path),
        "dataset_statistics_sha256": sha256_file(dataset_statistics_path) if dataset_statistics_path.is_file() else None,
        "stats_loader": "official_openvla_utils._load_dataset_stats" if oft else "checkpoint_embedded_or_existing_norm_stats",
        "component_state_dict_loader": "official_openvla_utils.load_component_state_dict" if oft else None,
        "vision_num_images_in_input": 2 if oft else 1,
        "checkpoint_mutated": False,
    }

    if not oft:
        # M0's checkpoint API returns one 7-D action; the official evaluator
        # consumes the OFT-style (action_chunk, auxiliary) pair.
        native_predict_action = model.predict_action

        def predict_action_compat(**kwargs: Any):
            result = native_predict_action(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            action = np.asarray(result, dtype=np.float32).reshape(-1)
            if action.size != ACTION_DIM:
                raise RuntimeError("M0_PREDICT_ACTION_SHAPE_INVALID")
            return action[None, :], None

        model.predict_action = predict_action_compat

    action_head = proprio_projector = None
    if oft:
        from prismatic.models.action_heads import L1RegressionActionHead  # type: ignore
        from prismatic.models.projectors import ProprioProjector  # type: ignore
        from experiments.robot.openvla_utils import load_component_state_dict  # type: ignore

        action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
        action_file = next(Path(checkpoint).glob("action_head-*checkpoint.pt"))
        action_head.load_state_dict(load_component_state_dict(str(action_file)))
        action_head = action_head.to(dtype=dtype, device="cuda:0").eval()
        proprio_projector = ProprioProjector(model.llm_dim, proprio_dim=8)
        proprio_file = next(Path(checkpoint).glob("proprio_projector-*checkpoint.pt"))
        proprio_projector.load_state_dict(load_component_state_dict(str(proprio_file)))
        proprio_projector = proprio_projector.to(dtype=dtype, device="cuda:0").eval()
    from experiments.robot.libero.run_libero_eval import process_action  # type: ignore
    from experiments.robot.openvla_utils import get_vla_action  # type: ignore

    cfg = SimpleNamespace(
        model_family="openvla",
        # M0's fused DINO+SigLIP processor emits 6 channels for one image;
        # only OFT checkpoints are configured for the additional wrist image.
        num_images_in_input=2 if oft else 1,
        use_proprio=oft,
        center_crop=True,
        unnorm_key=resolved_unnorm_key,
    )

    def infer(obs: dict[str, Any], instruction: str) -> tuple[np.ndarray, dict[str, Any]]:
        policy_obs = model_observation(obs)
        raw = get_vla_action(
            cfg,
            model,
            processor,
            policy_obs,
            instruction,
            action_head=action_head,
            proprio_projector=proprio_projector,
        )
        if return_chunk:
            raw_chunk = np.asarray(raw, dtype=np.float32)
            if raw_chunk.ndim != 2 or raw_chunk.shape[1] != ACTION_DIM or raw_chunk.shape[0] < 1:
                raise RuntimeError(f"OPENVLA_ACTION_CHUNK_INVALID:{raw_chunk.shape}")
            env_chunk = np.stack([validate_action(process_action(row.copy(), "openvla")) for row in raw_chunk])
            return env_chunk, {
                "raw_action_chunk": raw_chunk.tolist(),
                "chunk_length": int(raw_chunk.shape[0]),
                "postprocess": "official_openvla",
                "fresh_boundary": "FRESH_OFT_ACTION_QUEUE" if oft else "FRESH_PER_STEP",
            }
        raw_first = validate_action(raw[0])
        env_action = validate_action(process_action(raw_first.copy(), "openvla"))
        return env_action, {"raw_action": raw_first.tolist(), "chunk_length": len(raw), "postprocess": "official_openvla"}

    return infer, model, normalization_metadata


def load_pi05(checkpoint: str, *, return_chunk: bool = False):
    source = "/mnt/sdc/dty_user/openvla_attack/repos/openpi-stage-z-15a9616a_20260822"
    # The frozen official checkout uses src-layout packages; bind those exact
    # package roots instead of relying on the repository root being importable.
    sys.path.insert(0, str(Path(source) / "src"))
    sys.path.insert(0, str(Path(source) / "packages/openpi-client/src"))
    # The mandated A800 runtime is Python 3.10 while the frozen OpenPI source
    # uses the Python 3.11 datetime.UTC spelling. Preserve the source checkout
    # and supply only this equivalent standard-library compatibility alias.
    import datetime

    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc
    from openpi.policies import policy_config  # type: ignore
    from openpi.training import config as openpi_config  # type: ignore

    policy = policy_config.create_trained_policy(openpi_config.get_config("pi05_libero"), checkpoint)

    def infer(obs: dict[str, Any], instruction: str) -> tuple[np.ndarray, dict[str, Any]]:
        from openpi_client import image_tools  # type: ignore

        base = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
        wrist = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])
        element = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(base, 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
            "observation/state": np.asarray(model_observation(obs)["state"], dtype=np.float32),
            "prompt": str(instruction),
        }
        chunk = np.asarray(policy.infer(element)["actions"])
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < 1:
            raise RuntimeError(f"PI05_ACTION_CHUNK_INVALID:{chunk.shape}")
        if return_chunk:
            raw_chunk = np.asarray(chunk, dtype=np.float32)
            clipped_chunk = np.stack([validate_action(row, clip=True) for row in raw_chunk])
            return clipped_chunk, {
                "chunk_length": int(raw_chunk.shape[0]),
                "replan": "fresh_policy_infer",
                "raw_action_chunk": raw_chunk.tolist(),
                "action_after_libero_clip_chunk": clipped_chunk.tolist(),
                "action_was_clipped": bool(not np.array_equal(raw_chunk, clipped_chunk)),
                "fresh_boundary": "FRESH_PI05_REPLAN",
            }
        raw_action = np.asarray(chunk[0], dtype=np.float32).reshape(-1)
        action = validate_action(raw_action, clip=True)
        return action, {
            "chunk_length": int(chunk.shape[0]),
            "replan": "fresh_policy_infer",
            "raw_action": raw_action.tolist(),
            "action_after_libero_clip": action.tolist(),
            "action_was_clipped": bool(not np.array_equal(raw_action, action)),
        }

    return infer, policy


def run_cell(config: dict[str, Any], ledger: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    require_single_visible_gpu(args.gpu_id)
    gpu = gpu_snapshot(args.gpu_id)
    canary = static_authority(config, ledger, parent_key=args.parent_key, suite=args.suite, role=args.role)
    if args.model_family not in MODEL_FAMILIES or args.suite not in SUITES:
        raise RuntimeError("CELL_ARGUMENT_INVALID")
    model_spec = config["model_families"][args.model_family]
    configure_libero(config)
    if args.model_family == "M0_OPENVLA":
        checkpoint = model_spec["paths"][args.suite]
    elif args.model_family == "M1_OPENVLA_OFT":
        checkpoint = str(Path(model_spec["checkpoint_root"]) / args.suite)
    else:
        checkpoint = model_spec["checkpoint"]

    checkpoint_manifest = None
    if args.model_family == "M1_OPENVLA_OFT":
        if args.m1_manifest is None:
            raise RuntimeError("M1_MANIFEST_REQUIRED")
        checkpoint_manifest = verify_m1_materialization(
            args.m1_manifest,
            Path(checkpoint),
            args.suite,
            str(model_spec["checkpoint_manifests_sha256"]),
        )

    counters = {"model_inference_calls": 0, "env_step_calls": 0, "physical_interventions": 0, "pgd_calls": 0, "attacked_env_steps": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "stage_z_scientific_parent_exposure": 0, "eval160_reads": 0, "protected_reads": 0}

    def authorized_runtime() -> dict[str, Any]:
        from stage_z_preparation.contract import ExecutionAuthorization, ProtectedCounters, Z0R2_PASS
        from stage_z_preparation.runner import run_authorized_callback

        authorization = ExecutionAuthorization(
            execution_enabled=True,
            z0r2_status=Z0R2_PASS,
            root_seal_sha256=config["z0r2"]["root_seal_sha256"],
            expected_root_seal_sha256=config["z0r2"]["root_seal_sha256"],
            model_authority_sha256=config["z0r2"]["model_authority_sha256"],
            expected_model_authority_sha256=config["z0r2"]["model_authority_sha256"],
            common_libero_sha256=config["z0r2"]["common_libero_sha256"],
            expected_common_libero_sha256=config["z0r2"]["common_libero_sha256"],
            panel_sha256=config["z0r2"]["panel_sha256"],
            expected_panel_sha256=config["z0r2"]["panel_sha256"],
            frozen_parent_keys=frozenset([args.parent_key]),
            phase=PHASE,
            authorized_phases=frozenset([PHASE]),
            counters=ProtectedCounters(),
        )

        def execute() -> dict[str, Any]:
            if args.model_family == "M0_OPENVLA":
                infer, model, normalization_metadata = load_openvla(checkpoint, oft=False, suite=args.suite)
            elif args.model_family == "M1_OPENVLA_OFT":
                infer, model, normalization_metadata = load_openvla(checkpoint, oft=True, suite=args.suite)
            else:
                infer, model = load_pi05(checkpoint)
                normalization_metadata = {"checkpoint_mutated": False}
            env, task_suite, task = make_libero_env(config, args.suite, args.task_idx)
            try:
                env.reset()
                initial_states = task_suite.get_task_init_states(args.task_idx)
                obs = env.set_init_state(initial_states[args.state_id])
                counters["env_step_calls"] += 0
                dummy = [0.0] * 6 + [-1.0]
                for _ in range(int(config["environment"]["dummy_wait_steps"])):
                    obs = env.step(dummy)[0]
                    counters["env_step_calls"] += 1
                pre_state = snapshot_state(env)
                action, action_meta = infer(obs, str(task.language))
                counters["model_inference_calls"] += 1
                action_before_intervention = action.copy()
                clean_result_a = env.step(action.tolist())
                counters["env_step_calls"] += 1
                post_state_a = snapshot_state(env)
                del clean_result_a
                restore_state(env, pre_state)
                restored_pre_state = snapshot_state(env)
                if not np.array_equal(pre_state, restored_pre_state):
                    raise RuntimeError("BRANCH_REPLAY_PRE_STATE_NOT_EXACT")
                # The official robosuite binding restores qpos/qvel/time but
                # retains MuJoCo solver internals; replaying in that same
                # instance is not byte-deterministic. Reinstantiate the same
                # frozen canary environment for the clean branch instead.
                branch_env, _, _ = make_libero_env(config, args.suite, args.task_idx)
                try:
                    branch_env.reset()
                    branch_env.set_init_state(initial_states[args.state_id])
                    for _ in range(int(config["environment"]["dummy_wait_steps"])):
                        branch_env.step(dummy)
                        counters["env_step_calls"] += 1
                    branch_pre_state = snapshot_state(branch_env)
                    clean_result_b = branch_env.step(action.tolist())
                    counters["env_step_calls"] += 1
                    post_state_b = snapshot_state(branch_env)
                    del clean_result_b
                finally:
                    branch_env.close()
                if not np.array_equal(pre_state, branch_pre_state):
                    raise RuntimeError("BRANCH_REPLAY_FRESH_PRE_STATE_NOT_EXACT")
                if not np.array_equal(post_state_a, post_state_b):
                    raise RuntimeError("BRANCH_REPLAY_POST_STATE_NOT_EXACT")
                return {
                    "status": "PASS",
                    "task_language": str(task.language),
                    "action_before_intervention": action_before_intervention.tolist(),
                    "action_meta": action_meta,
                    "branch_replay": {
                        "mode": "FRESH_OFFICIAL_ENV_AFTER_SNAPSHOT_RESTORE_PROBE",
                        "pre_state_sha256": hashlib.sha256(pre_state.tobytes()).hexdigest(),
                        "post_state_a_sha256": hashlib.sha256(post_state_a.tobytes()).hexdigest(),
                        "post_state_b_sha256": hashlib.sha256(post_state_b.tobytes()).hexdigest(),
                        "pre_state_exact": True,
                        "post_state_exact": True,
                    },
                    "normalization": normalization_metadata,
                }
            finally:
                env.close()

        return run_authorized_callback(
            authorization=authorization,
            parent_key=args.parent_key,
            phase=PHASE,
            callback=execute,
        )

    result = authorized_runtime()
    result.update(
        {
            "schema": "STAGE_Z_Z1_RUNTIME_CELL_RECEIPT_V1",
            "model_family": args.model_family,
            "suite": args.suite,
            "role": args.role,
            "canonical_parent_key": args.parent_key,
            "task_idx": args.task_idx,
            "state_id": args.state_id,
            "selection_rank_sha256": canary["selection_rank_sha256"],
            "checkpoint": checkpoint,
            "checkpoint_manifest": checkpoint_manifest,
            "gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "runtime_counters": counters,
            "scientific_claim": "NONE_ENGINEERING_ONLY",
            "task_success_read": False,
            "physical_outcome_read": False,
            "intervention": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--role", choices=("PRIMARY", "BACKUP"), required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--task-idx", type=int, required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--m1-manifest", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    ledger = load_json(args.ledger)
    result = run_cell(config, ledger, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "model_family": args.model_family, "suite": args.suite, "output": str(args.output)}))


if __name__ == "__main__":
    main()
