#!/usr/bin/env python3
"""Collect one hash-bound R8W suite-local clean shard.

This is a generalized, resumable version of the validated R8T collector.  It
accepts every frozen R7 cohort without changing cohort semantics, records
post-step canonical success evidence, and never reads or creates attack data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.collect_c2g_clean_window_rollouts import (
    contact_pairs,
    entity_position,
    git_provenance,
    normalized_rgb,
    policy_features,
    read_manifest,
    save_rgb,
    sha256_file,
    target_support_contact,
)
from scripts.stageb.collect_c2g_clean_window_rollouts_event_v2 import (
    _binding_by_index,
    _event_with_binding,
    _single_binding_event,
    entity_joint_scalar_with_hint,
)
from gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from gripper_attack.c2g_clean_event_tracking import (
    goal_event_bindings,
    joint_hint_from_interaction_site,
    select_active_goal_event,
)
from gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from gripper_attack.c2g_clean_window_runtime import derive_gripper_token_semantics
from gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets

COLLECTION_SCHEMA = "c2g.r8w.teacher_v2_full_clean_collection.2026-07-12.v1"
POST_STEP_SCHEMA = "c2g.r8w.post_step_outcome.2026-07-12.v1"
EPISODE_RECEIPT_SCHEMA = "c2g.r8w.episode_receipt.2026-07-12.v1"
TEACHER_SCHEMA = "c2g.teacher_v2.raw_privileged_evidence.2026-07-11.v1"
EVENT_TRACKING_SCHEMA = "c2g.clean_goal_event_tracking.2026-07-11.v1"
ACTION_ORDER = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
COHORT_TO_SPLIT = {
    "DETECTOR_TRAIN": "train",
    "DETECTOR_VAL": "val",
    "DETECTOR_TEST_WITHIN_TASK": "test",
    "ATTACK_EVAL_PREREGISTERED": "attack_eval",
}
STATUS_PHASES = (
    "CREATED",
    "WAITING_MODEL_LOAD_LOCK",
    "LOADING_PROCESSOR",
    "LOADING_MODEL",
    "MODEL_READY",
    "CREATING_ENVIRONMENT",
    "RUNNING_EPISODES",
    "FINALIZING",
    "PASS",
    "FAILED",
)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_memory_snapshot(physical_gpu: int) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        values = [value.strip() for value in output.split(",")]
        if len(values) != 6:
            raise ValueError(output)
        return {
            "physical_gpu": int(values[0]),
            "memory_total_mib": int(values[1]),
            "memory_used_mib": int(values[2]),
            "memory_free_mib": int(values[3]),
            "utilization_percent": int(values[4]),
            "temperature_c": int(values[5]),
        }
    except Exception as exc:
        return {"physical_gpu": physical_gpu, "error": f"{type(exc).__name__}: {exc}"}


def write_worker_status(
    path: Path,
    *,
    worker_id: str,
    physical_gpu: int,
    suite: str,
    shard_id: str,
    phase: str,
    completed: int,
    failed: int,
    current_parent_key: str | None,
) -> None:
    if phase not in STATUS_PHASES:
        raise ValueError(f"invalid worker phase: {phase}")
    value = {
        "worker_id": worker_id,
        "physical_gpu": physical_gpu,
        "suite": suite,
        "shard_id": shard_id,
        "pid": os.getpid(),
        "phase": phase,
        "timestamp": utc_now(),
        "completed_episode_count": completed,
        "failed_episode_count": failed,
        "current_parent_key": current_parent_key,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_memory_snapshot": gpu_memory_snapshot(physical_gpu),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


@contextmanager
def model_load_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must contain an object")
        rows.append(value)
    return rows


def build_rgb_manifest(rgb_dir: Path, manifest_path: Path) -> tuple[int, str]:
    entries = []
    for path in sorted(rgb_dir.glob("frame_*.png")):
        entries.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in entries),
        encoding="utf-8",
    )
    return len(entries), sha256_file(manifest_path)


def verify_rgb_manifest(rgb_dir: Path, manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        entries = read_jsonl(manifest_path)
        if not entries:
            return False
        for row in entries:
            path = rgb_dir / str(row["path"])
            if not path.is_file() or path.stat().st_size != int(row["bytes"]):
                return False
            if sha256_file(path) != str(row["sha256"]):
                return False
        return True
    except Exception:
        return False


def validate_episode_receipt(
    episode_dir: Path,
    *,
    expected_parent_key: str,
    expected_worker_id: str,
    expected_shard_id: str,
    expected_git_head: str,
    expected_manifest_sha: str,
) -> tuple[bool, str]:
    receipt_path = episode_dir / "episode_receipt.json"
    metadata_path = episode_dir / "episode_metadata.json"
    steps_path = episode_dir / "step_records.jsonl"
    rgb_manifest = episode_dir / "rgb_manifest.jsonl"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema": EPISODE_RECEIPT_SCHEMA,
            "parent_key": expected_parent_key,
            "worker_id": expected_worker_id,
            "shard_id": expected_shard_id,
            "git_head": expected_git_head,
            "manifest_sha256": expected_manifest_sha,
            "runtime_valid": True,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                return False, f"receipt {key} mismatch"
        if metadata.get("runtime_valid") is not True or metadata.get("clean_success_observed") not in (True, False):
            return False, "metadata is not runtime-valid with boolean success"
        if not steps_path.is_file() or steps_path.stat().st_size == 0:
            return False, "step records missing or empty"
        if sha256_file(metadata_path) != receipt.get("metadata_sha256"):
            return False, "metadata SHA mismatch"
        if sha256_file(steps_path) != receipt.get("step_records_sha256"):
            return False, "step records SHA mismatch"
        if sha256_file(rgb_manifest) != receipt.get("rgb_manifest_sha256"):
            return False, "RGB manifest SHA mismatch"
        if not verify_rgb_manifest(episode_dir / "rgb", rgb_manifest):
            return False, "RGB artifact verification failed"
        return True, "PASS"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def stable_episode_seed(base_seed: int, parent_key: str) -> int:
    digest = hashlib.sha256(f"R8T|{base_seed}|{parent_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def array_sha256(value: Any) -> tuple[str, list[int], str]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest(), list(array.shape), str(array.dtype)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        try:
            module = importlib.import_module(name)
            return str(getattr(module, "__version__", "LOCAL_SOURCE"))
        except Exception:
            return "NOT_INSTALLED"


def git_commit_for_path(path: Path) -> str:
    current = path.resolve()
    for root in (current, *current.parents):
        if (root / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip()
            except Exception:
                return "UNRESOLVED"
    return "UNRESOLVED"


def runtime_provenance() -> dict[str, Any]:
    values = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "libero": package_version("libero"),
        "robosuite": package_version("robosuite"),
        "mujoco": package_version("mujoco"),
        "mujoco_py": package_version("mujoco_py"),
    }
    try:
        import libero
        values["libero_source_path"] = str(Path(libero.__file__).resolve())
        values["libero_git_commit"] = git_commit_for_path(Path(libero.__file__).resolve())
    except Exception:
        values["libero_source_path"] = "UNRESOLVED"
        values["libero_git_commit"] = "UNRESOLVED"
    return values


def jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(child, depth + 1) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child, depth + 1) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def controller_provenance(env: Any) -> dict[str, Any]:
    roots = [env, getattr(env, "env", None), getattr(env, "unwrapped", None)]
    for root in roots:
        if root is None:
            continue
        robots = getattr(root, "robots", None)
        if not robots:
            continue
        robot = robots[0]
        controller = getattr(robot, "controller", None)
        output: dict[str, Any] = {
            "robot_class": type(robot).__name__,
            "controller_class": type(controller).__name__ if controller is not None else "UNRESOLVED",
        }
        for name in ("control_freq", "controller_config", "controller_configs", "control_dim", "action_dim"):
            value = getattr(root, name, None)
            if value is None:
                value = getattr(robot, name, None)
            if value is None and controller is not None:
                value = getattr(controller, name, None)
            if value is not None:
                output[name] = jsonable(value)
        if controller is not None:
            for name in ("input_type", "output_min", "output_max", "kp", "damping_ratio"):
                value = getattr(controller, name, None)
                if value is not None:
                    output[name] = jsonable(value)
        return output
    return {"controller_class": "UNRESOLVED"}


def finite_action(value: Any, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise RuntimeError(f"{name} must be a finite 7D vector, got {action.shape}")
    return action


def termination_after_step(check_success: bool, done: bool) -> str | None:
    if check_success:
        return "ENV_CHECK_SUCCESS"
    if done:
        return "DONE_WITHOUT_SUCCESS"
    return None


def canonical_clean_success(any_check_success: bool, final_check_success: bool) -> bool:
    return bool(any_check_success or final_check_success)


def validate_post_step_outcome(row: Mapping[str, Any]) -> bool:
    required = (
        "reward_after_step",
        "done_after_step",
        "env_check_success_after_step",
        "info_success_after_step",
        "info_task_success_after_step",
        "info_is_success_after_step",
    )
    if any(name not in row for name in required):
        return False
    try:
        return (
            np.isfinite(float(row["reward_after_step"]))
            and type(row["done_after_step"]) is bool
            and type(row["env_check_success_after_step"]) is bool
        )
    except (TypeError, ValueError):
        return False


def frozen_manifest_max_steps(episodes: Sequence[Mapping[str, Any]]) -> int:
    values = {row.get("max_steps") for row in episodes}
    if len(values) != 1:
        raise ValueError(f"R8W shard contains mixed max_steps: {sorted(values, key=str)}")
    value = next(iter(values))
    if type(value) is not int or value <= 0:
        raise ValueError(f"invalid manifest max_steps: {value!r}")
    return value


def selected_model_hashes(model_path: Path) -> dict[str, str]:
    allowed_prefixes = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    return {
        path.name: sha256_file(path)
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.name in allowed_prefixes
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--model-load-lock-file", type=Path, required=True)
    parser.add_argument("--worker-status-file", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260711)
    parser.add_argument("--near-target-threshold", type=float, default=0.08)
    parser.add_argument("--relative-lift-threshold", type=float, default=0.015)
    parser.add_argument("--progress-threshold", type=float, default=0.01)
    parser.add_argument("--fixture-motion-threshold", type=float, default=0.005)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise ValueError("R8W shard manifest SHA mismatch")
    provenance = git_provenance(args.expected_git_commit)
    if args.device != "cuda:0":
        raise ValueError("R8W worker model device is frozen to cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.physical_gpu):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES must equal physical GPU {args.physical_gpu}, got {visible!r}"
        )
    if os.environ.get("C2G_PHYSICAL_GPU") != str(args.physical_gpu):
        raise ValueError("C2G_PHYSICAL_GPU mismatch")
    for path in (
        args.suite_model_map,
        args.suite_model_report,
        args.goal_model_manifest,
        args.model_verification_report,
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    frozen_verification = json.loads(args.model_verification_report.read_text(encoding="utf-8"))
    if not isinstance(frozen_verification, Mapping) or not str(frozen_verification.get("status", "")).startswith("PASS"):
        raise ValueError("frozen model verification report is not PASS")
    model_map = json.loads(args.suite_model_map.read_text(encoding="utf-8"))
    episodes = read_manifest(manifest_path)
    if not episodes:
        raise RuntimeError("empty R8W shard manifest")
    suites = {str(row.get("suite", "")) for row in episodes}
    if len(suites) != 1 or not suites.issubset(SUITES):
        raise ValueError("R8W shard must contain exactly one valid suite")
    for row in episodes:
        cohort = str(row.get("cohort", ""))
        if cohort not in COHORT_TO_SPLIT or row.get("split") != COHORT_TO_SPLIT[cohort]:
            raise ValueError("R8W shard contains an invalid cohort/split binding")
        if row.get("assigned_worker_id") != args.worker_id:
            raise ValueError("R8W shard worker assignment mismatch")
        if row.get("assigned_shard_id") != args.shard_id:
            raise ValueError("R8W shard ID mismatch")
        if row.get("assigned_physical_gpu") != args.physical_gpu:
            raise ValueError("R8W shard physical GPU mismatch")
        if row.get("collection_purpose") not in {"FULL_CLEAN_2000", "FRESH_SHADOW_CANARY"} or row.get("materializable") is not False:
            raise ValueError("R8W collection-purpose boundary mismatch")
    purposes = {str(row["collection_purpose"]) for row in episodes}
    if len(purposes) != 1:
        raise ValueError("R8W shard contains mixed collection purpose")
    collection_purpose = next(iter(purposes))
    max_steps = frozen_manifest_max_steps(episodes)

    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=args.resume)

    suite = next(iter(suites))
    completed = 0
    failed = 0
    write_worker_status(
        args.worker_status_file.resolve(), worker_id=args.worker_id,
        physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
        phase="CREATED", completed=completed, failed=failed, current_parent_key=None,
    )
    pending: list[dict[str, Any]] = []
    resumed: list[dict[str, Any]] = []
    for episode in episodes:
        parent_key = str(episode["parent_key"])
        episode_dir = output_root / "episodes" / suite / parent_key
        if not episode_dir.exists():
            pending.append(episode)
            continue
        valid, reason = validate_episode_receipt(
            episode_dir,
            expected_parent_key=parent_key,
            expected_worker_id=args.worker_id,
            expected_shard_id=args.shard_id,
            expected_git_head=args.expected_git_commit,
            expected_manifest_sha=args.manifest_sha256,
        )
        if valid:
            resumed.append({"parent_key": parent_key, "reason": reason})
            completed += 1
        else:
            failed += 1
            write_worker_status(
                args.worker_status_file.resolve(), worker_id=args.worker_id,
                physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
                phase="FAILED", completed=completed, failed=failed, current_parent_key=parent_key,
            )
            raise RuntimeError(
                f"existing incomplete or provenance-mismatched episode is preserved: {episode_dir}: {reason}"
            )

    from scripts.stageb.c2f_libero_openvla_adapter import _resolve_task_language
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelClass
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from v4_run_eval_openvla import (
        decode_with_scores,
        physical_gripper_state,
        postprocess_openvla_action_for_libero,
    )

    runtime = runtime_provenance()
    model_path = Path(str(model_map[suite])).resolve()
    write_worker_status(
        args.worker_status_file.resolve(), worker_id=args.worker_id,
        physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
        phase="WAITING_MODEL_LOAD_LOCK", completed=completed, failed=failed, current_parent_key=None,
    )
    with model_load_lock(args.model_load_lock_file.resolve()):
        write_worker_status(
            args.worker_status_file.resolve(), worker_id=args.worker_id,
            physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
            phase="LOADING_PROCESSOR", completed=completed, failed=failed, current_parent_key=None,
        )
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        write_worker_status(
            args.worker_status_file.resolve(), worker_id=args.worker_id,
            physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
            phase="LOADING_MODEL", completed=completed, failed=failed, current_parent_key=None,
        )
        model = AutoModelClass.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=args.device,
        ).eval()
        torch.cuda.synchronize()
        write_worker_status(
            args.worker_status_file.resolve(), worker_id=args.worker_id,
            physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
            phase="MODEL_READY", completed=completed, failed=failed, current_parent_key=None,
        )
    unnorm_key = suite if suite in getattr(model, "norm_stats", {}) else f"{suite}_no_noops"
    if unnorm_key not in getattr(model, "norm_stats", {}):
        raise RuntimeError(f"cannot resolve unnorm key for {suite}")
    semantics = derive_gripper_token_semantics(model, unnorm_key)

    results: list[dict[str, Any]] = [
        {"parent_key": row["parent_key"], "status": "RESUMED_VALID_RECEIPT"}
        for row in resumed
    ]
    artifact_entries: list[dict[str, Any]] = []
    suite_obj = benchmark.get_benchmark_dict()[suite]()

    for episode_index, episode in enumerate(pending):
        task_index = int(episode["task_index"])
        state_id = int(episode["state_id"])
        parent_key = str(episode["parent_key"])
        seed = stable_episode_seed(args.base_seed, parent_key)
        set_deterministic_seeds(seed)
        episode_dir = output_root / "episodes" / suite / parent_key
        if episode_dir.exists():
            raise FileExistsError(episode_dir)
        episode_dir.mkdir(parents=True)
        rgb_dir = episode_dir / "rgb"
        step_path = episode_dir / "step_records.jsonl"
        metadata_path = episode_dir / "episode_metadata.json"
        env = None
        rows_written = 0
        started = time.time()
        last_info: dict[str, Any] = {}
        any_check_success = False
        first_success_step: int | None = None
        final_check_success = False
        done_first_step: int | None = None
        termination_reason = "MAX_POLICY_STEPS"
        reward_sum = 0.0
        reward_max: float | None = None
        reward_nonzero_step_count = 0
        write_worker_status(
            args.worker_status_file.resolve(), worker_id=args.worker_id,
            physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
            phase="CREATING_ENVIRONMENT", completed=completed, failed=failed,
            current_parent_key=parent_key,
        )
        try:
            task = suite_obj.get_task(task_index)
            states = suite_obj.get_task_init_states(task_index)
            if state_id < 0 or state_id >= len(states):
                raise IndexError("state_id outside official init-state range")
            init_state = states[state_id]
            init_sha, init_shape, init_dtype = array_sha256(init_state)
            task_language, language_source = _resolve_task_language(task, episode)
            bddl_path = (Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).resolve()
            structured = parse_bddl_task_metadata(bddl_path)
            task_metadata = {
                **structured,
                "task_language": task_language,
                "suite": suite,
                "task_index": task_index,
                "episode_key": parent_key,
                "parent_key": parent_key,
                "gripper_command_semantics": "raw_openvla_threshold_0p5_close_below",
            }
            resolution = resolve_task_targets(task_metadata)
            mechanism = infer_clean_mechanism_type(task_metadata, resolution=resolution)
            bindings = goal_event_bindings(resolution)
            if mechanism != "unsupported_or_unknown" and not bindings:
                raise RuntimeError("eligible mechanism has no structured goal-event binding")
            binding_by_index = _binding_by_index(bindings)
            task_metadata.update(
                mechanism_type=mechanism,
                structured_goal_metadata={
                    "target_objects": list(resolution.resolved_target_objects),
                    "target_receptacles": list(resolution.resolved_receptacles),
                    "target_sites": list(resolution.resolved_sites),
                    "target_fixtures": list(resolution.resolved_manipulable_entities),
                    "target_destinations": list(resolution.resolved_destination_entities),
                    "goal_bindings": [list(value) for value in resolution.goal_bindings],
                },
                goal_event_bindings=[binding.to_dict() for binding in bindings],
                event_tracking_schema=EVENT_TRACKING_SCHEMA,
                teacher_schema_version=TEACHER_SCHEMA,
            )

            env, obs = build_v4_exact_env(
                str(bddl_path), args.physical_gpu, max_steps, args.dummy_wait
            )
            obs = env.set_init_state(init_state)
            env, obs = apply_dummy_wait(env, obs, args.dummy_wait)
            write_worker_status(
                args.worker_status_file.resolve(), worker_id=args.worker_id,
                physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
                phase="RUNNING_EPISODES", completed=completed, failed=failed,
                current_parent_key=parent_key,
            )
            controller = controller_provenance(env)
            streamer = SC5StreamingFeatureAdapterV2()
            eef_site = env.sim.model.site_name2id("gripper0_grip_site")
            previous_eef = None
            target_objects = set(resolution.resolved_target_objects)
            target_manipulable = set(resolution.resolved_manipulable_entities)
            region_owner_by_site = dict(task_metadata.get("region_owner_by_site", {}))
            baseline_z: dict[int, float] = {}
            initial_distance: dict[int, float] = {}
            initial_joint: dict[int, float] = {}
            last_active_index: int | None = None
            for binding in bindings:
                target_position = entity_position(env, binding.target_entity)
                destination_position = entity_position(env, binding.destination_entity) if binding.destination_entity else None
                if target_position is not None and destination_position is not None:
                    initial_distance[binding.subgoal_index] = float(np.linalg.norm(target_position - destination_position))
                hint = joint_hint_from_interaction_site(binding.target_entity, binding.interaction_site)
                joint = entity_joint_scalar_with_hint(env, binding.target_entity, hint)
                if joint is not None:
                    initial_joint[binding.subgoal_index] = joint

            with step_path.open("w", encoding="utf-8") as handle:
                for step in range(max_steps):
                    rgb = normalized_rgb(obs, step)
                    gripper_state = physical_gripper_state(env, obs)
                    qpos = np.asarray(gripper_state.get("qpos", []), dtype=np.float32).reshape(-1)
                    qpos_sum = float(qpos[:2].sum()) if qpos.size >= 2 else 0.0
                    opening = float(np.abs(qpos[:2]).sum()) if qpos.size >= 2 else 0.0
                    decoded, _, _, generation = decode_with_scores(
                        model, processor, args.device, rgb, task_language, unnorm_key, 8,
                        libero_official_preprocess=False,
                        libero_preprocess_backend="official_pil_lanczos",
                        center_crop=True,
                        resize_size=224,
                        drop_attention_mask=True,
                    )
                    if not getattr(generation, "scores", None):
                        raise RuntimeError("clean generation lacks score tensors")
                    raw_action = finite_action(decoded, "clean_action_raw_7d")
                    applied_action = finite_action(
                        postprocess_openvla_action_for_libero(raw_action, enabled=True),
                        "applied_action_7d",
                    )
                    logits = generation.scores[-1][0].detach()
                    policy = policy_features(logits, semantics)
                    top_k = min(16, int(logits.numel()))
                    top_values, top_ids = torch.topk(logits.float(), k=top_k)
                    eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32).copy()
                    velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
                    previous_eef = eef.copy()
                    stream = streamer.update(
                        step_id=step,
                        raw_gripper=float(raw_action[-1]),
                        env_gripper=float(applied_action[-1]),
                        gripper_qpos=qpos_sum,
                        gripper_opening_proxy=opening,
                        eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                        eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                        action_dx=float(applied_action[0]), action_dy=float(applied_action[1]),
                        action_dz=float(applied_action[2]), action_gripper=float(raw_action[-1]),
                    )
                    features = list(stream["features"].values())
                    if len(features) != 25 or not np.isfinite(np.asarray(features, dtype=np.float32)).all():
                        raise RuntimeError(f"invalid 25D features at step {step}")

                    pairs = contact_pairs(env)
                    event = select_active_goal_event(
                        pairs,
                        bindings,
                        manipulable_targets=sorted(target_manipulable),
                        finger_aliases=task_metadata.get("finger_aliases"),
                    )
                    if event["active_target_known"]:
                        last_active_index = int(event["active_subgoal_index"])
                    elif len(bindings) == 1:
                        event = _single_binding_event(bindings[0])
                    elif last_active_index is not None:
                        event = _event_with_binding(event, binding_by_index[last_active_index])
                        event["active_target_reason"] = "RETAINED_LAST_CONTACTED_SUBGOAL"
                    active_binding = (
                        binding_by_index.get(int(event["active_subgoal_index"]))
                        if event.get("active_target_known") and event.get("active_subgoal_index") is not None
                        else None
                    )
                    clean_close = float(raw_action[-1]) < 0.5
                    target_position = destination_position = None
                    relative_lift = progress = fixture_motion = None
                    target_contact = bilateral_contact = False
                    near_target = supported = release_safe = None
                    manipulation_active = constrained_active = None
                    if active_binding is not None:
                        target_contact = active_binding.target_entity in set(event.get("contacted_goal_targets", []))
                        bilateral_contact = active_binding.target_entity in set(event.get("bilateral_goal_targets", []))
                        target_position = entity_position(env, active_binding.target_entity)
                        destination_position = entity_position(env, active_binding.destination_entity) if active_binding.destination_entity else None
                        if target_position is not None and destination_position is not None:
                            current_distance = float(np.linalg.norm(target_position - destination_position))
                            initial_distance.setdefault(active_binding.subgoal_index, current_distance)
                            progress = initial_distance[active_binding.subgoal_index] - current_distance
                            near_target = current_distance <= args.near_target_threshold
                        if clean_close and bilateral_contact and target_position is not None and active_binding.target_entity in target_objects:
                            baseline_z.setdefault(active_binding.subgoal_index, float(target_position[2]))
                        if target_position is not None and active_binding.subgoal_index in baseline_z:
                            relative_lift = float(target_position[2]) - baseline_z[active_binding.subgoal_index]
                        hint = joint_hint_from_interaction_site(active_binding.target_entity, active_binding.interaction_site)
                        joint = entity_joint_scalar_with_hint(env, active_binding.target_entity, hint)
                        if joint is not None:
                            initial_joint.setdefault(active_binding.subgoal_index, joint)
                            fixture_motion = abs(joint - initial_joint[active_binding.subgoal_index])
                            constrained_active = fixture_motion >= args.fixture_motion_threshold
                        if any(value is not None for value in (relative_lift, progress, fixture_motion)):
                            manipulation_active = bool(
                                (relative_lift is not None and relative_lift >= args.relative_lift_threshold)
                                or (progress is not None and progress >= args.progress_threshold)
                                or (fixture_motion is not None and fixture_motion >= args.fixture_motion_threshold)
                            )
                        if active_binding.destination_entity:
                            support_entities = [active_binding.destination_entity]
                            owner = region_owner_by_site.get(active_binding.destination_entity)
                            if owner:
                                support_entities.append(owner)
                            supported = target_support_contact(pairs, active_binding.target_entity, support_entities)
                            if near_target is not None:
                                release_safe = bool(near_target and supported)
                        elif fixture_motion is not None:
                            release_safe = bool(
                                fixture_motion >= args.fixture_motion_threshold and not target_contact
                            )

                    row = {
                        "step": step,
                        "teacher_schema_version": TEACHER_SCHEMA,
                        "rgb_path": f"rgb/frame_{step:06d}.png",
                        "task_language": task_language,
                        "features_25d": [float(value) for value in features],
                        "clean_policy_intent_9d": policy,
                        **{name: policy[index] for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)},
                        "clean_action_raw_7d": raw_action.astype(float).tolist(),
                        "applied_action_7d": applied_action.astype(float).tolist(),
                        "action_order": list(ACTION_ORDER),
                        "clean_action_token_top_ids": top_ids.detach().cpu().tolist(),
                        "clean_action_token_top_logits": top_values.detach().cpu().tolist(),
                        "clean_gripper_command": float(raw_action[-1]),
                        "clean_close_intent": clean_close,
                        "mujoco_contact_pairs": pairs,
                        **event,
                        "object_relative_lift": relative_lift,
                        "target_distance_decrease": progress,
                        "constrained_manipulation_active": constrained_active,
                        "manipulation_progress_active": manipulation_active,
                        "near_target": near_target,
                        "supported_at_target": supported,
                        "release_safe": release_safe,
                        "target_object_position": None if target_position is None else target_position.astype(float).tolist(),
                        "target_destination_position": None if destination_position is None else destination_position.astype(float).tolist(),
                        "fixture_joint_motion": fixture_motion,
                        "active_target_contact": target_contact,
                        "active_target_bilateral_contact": bilateral_contact,
                    }
                    save_rgb(rgb_dir / f"frame_{step:06d}.png", rgb)
                    obs, reward, done, info = env.step(applied_action)
                    check_success = bool(env.check_success())
                    if check_success and not any_check_success:
                        any_check_success = True
                        first_success_step = step
                    last_info = dict(info or {})
                    # Record per-step outcomes on this step's row
                    row["reward_after_step"] = float(reward)
                    row["done_after_step"] = bool(done)
                    row["env_check_success_after_step"] = bool(check_success)
                    row["info_success_after_step"] = info.get("success")
                    row["info_task_success_after_step"] = info.get("task_success")
                    row["info_is_success_after_step"] = info.get("is_success")
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                    rows_written += 1
                    reward_value = float(reward)
                    reward_sum += reward_value
                    reward_max = reward_value if reward_max is None else max(reward_max, reward_value)
                    if reward_value != 0.0:
                        reward_nonzero_step_count += 1
                    if done and done_first_step is None:
                        done_first_step = step
                    if release_safe:
                        last_active_index = None
                    termination = termination_after_step(check_success, bool(done))
                    if termination is not None:
                        termination_reason = termination
                        break

            # Final success check after policy loop
            try:
                final_check_success = bool(env.check_success())
            except Exception:
                final_check_success = False

            clean_success_observed = canonical_clean_success(any_check_success, final_check_success)
            metadata = {
                "schema": COLLECTION_SCHEMA,
                **task_metadata,
                "state_id": state_id,
                "cohort": episode["cohort"],
                "split": episode["split"],
                "task_language_source": language_source,
                "resolution": resolution.to_dict(),
                "model_path": str(model_path),
                "model_selected_hashes": selected_model_hashes(model_path),
                "suite_model_map": str(args.suite_model_map.resolve()),
                "suite_model_map_sha256": sha256_file(args.suite_model_map.resolve()),
                "suite_model_report": str(args.suite_model_report.resolve()),
                "suite_model_report_sha256": sha256_file(args.suite_model_report.resolve()),
                "goal_model_manifest": str(args.goal_model_manifest.resolve()),
                "goal_model_manifest_sha256": sha256_file(args.goal_model_manifest.resolve()),
                "model_verification_report": str(args.model_verification_report.resolve()),
                "model_verification_report_sha256": sha256_file(args.model_verification_report.resolve()),
                "unnorm_key": unnorm_key,
                "token_semantics_sha256": semantics["token_semantics_sha256"],
                "open_token_ids": list(semantics["open_token_ids"]),
                "close_token_ids": list(semantics["close_token_ids"]),
                "raw_action_order": list(ACTION_ORDER),
                "applied_action_order": list(ACTION_ORDER),
                "action_semantics": {
                    "raw": "OpenVLA unnormalized 7D delta-pose plus gripper",
                    "applied": "postprocess_openvla_action_for_libero(raw_action, enabled=True)",
                    "order": list(ACTION_ORDER),
                },
                "controller_config": controller,
                "runtime_versions": runtime,
                "bddl_file": str(bddl_path),
                "bddl_sha256": sha256_file(bddl_path),
                "official_init_state_sha256": init_sha,
                "official_init_state_shape": init_shape,
                "official_init_state_dtype": init_dtype,
                "replay_seed": seed,
                "base_seed": args.base_seed,
                "max_steps": max_steps,
                "dummy_wait": args.dummy_wait,
                "thresholds": {
                    "near_target": args.near_target_threshold,
                    "relative_lift": args.relative_lift_threshold,
                    "progress": args.progress_threshold,
                    "fixture_motion": args.fixture_motion_threshold,
                },
                "runtime_valid": True,
                "n_steps": rows_written,
                "clean_success_metric": "LIBERO_ENV_CHECK_SUCCESS",
                "clean_success_observed": clean_success_observed,
                "clean_success_first_step": first_success_step,
                "final_env_check_success": final_check_success,
                "termination_reason": termination_reason,
                "done_first_step": done_first_step,
                "reward_sum": reward_sum,
                "reward_max": 0.0 if reward_max is None else reward_max,
                "reward_nonzero_step_count": reward_nonzero_step_count,
                "post_step_outcome_complete": True,
                "post_step_outcome_schema_version": POST_STEP_SCHEMA,
                "runtime_seconds": time.time() - started,
                "condition": "CLEAN",
                "worker_id": args.worker_id,
                "shard_id": args.shard_id,
                "physical_gpu": args.physical_gpu,
                "CUDA_VISIBLE_DEVICES": visible,
                "model_device": args.device,
                "render_gpu_device_id": args.physical_gpu,
                "shard_manifest": str(manifest_path),
                "shard_manifest_sha256": args.manifest_sha256,
                "collection_purpose": collection_purpose,
                "materializable": False,
                "eligible_for_detector_fit": episode["eligible_for_detector_fit"],
                "eligible_for_checkpoint_selection": episode["eligible_for_checkpoint_selection"],
                "eligible_for_threshold_calibration": episode["eligible_for_threshold_calibration"],
                "eligible_for_clean_test": episode["eligible_for_clean_test"],
                "eligible_for_attack_evaluation": episode["eligible_for_attack_evaluation"],
                "student_allowed_modalities": [
                    "rgb", "task_language", "features_25d", "clean_policy_intent_9d"
                ],
                "student_forbidden_modalities": [
                    "mujoco_contact_pairs", "active_target_entity", "active_subgoal_index",
                    "object_relative_lift", "target_distance_decrease",
                    "target_object_position", "target_destination_position", "release_safe",
                    "attack_outcome", "post_intervention",
                ],
                **provenance,
            }
            write_json(metadata_path, metadata)
            rgb_manifest_path = episode_dir / "rgb_manifest.jsonl"
            rgb_count, rgb_manifest_sha = build_rgb_manifest(rgb_dir, rgb_manifest_path)
            if rgb_count != rows_written:
                raise RuntimeError(f"RGB/step count mismatch: {rgb_count} != {rows_written}")
            receipt = {
                "schema": EPISODE_RECEIPT_SCHEMA,
                "parent_key": parent_key,
                "worker_id": args.worker_id,
                "shard_id": args.shard_id,
                "git_head": args.expected_git_commit,
                "manifest_sha256": args.manifest_sha256,
                "metadata_sha256": sha256_file(metadata_path),
                "step_records_sha256": sha256_file(step_path),
                "rgb_manifest_sha256": rgb_manifest_sha,
                "runtime_valid": True,
                "completion_timestamp": utc_now(),
            }
            receipt_path = episode_dir / "episode_receipt.json"
            write_json(receipt_path, receipt)
            valid_receipt, receipt_reason = validate_episode_receipt(
                episode_dir,
                expected_parent_key=parent_key,
                expected_worker_id=args.worker_id,
                expected_shard_id=args.shard_id,
                expected_git_head=args.expected_git_commit,
                expected_manifest_sha=args.manifest_sha256,
            )
            if not valid_receipt:
                raise RuntimeError(f"new episode receipt failed self-verification: {receipt_reason}")
            completed += 1
            results.append({
                "parent_key": parent_key,
                "suite": suite,
                "task_index": task_index,
                "state_id": state_id,
                "n_steps": rows_written,
                "clean_success_observed": metadata["clean_success_observed"],
                "status": "PASS",
            })
            for artifact in (metadata_path, step_path, rgb_manifest_path, receipt_path):
                artifact_entries.append({
                    "path": artifact.relative_to(output_root).as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                })
        except Exception as exc:
            failed += 1
            failure = {
                "schema": COLLECTION_SCHEMA,
                "parent_key": parent_key,
                "suite": suite,
                "task_index": task_index,
                "state_id": state_id,
                "runtime_valid": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "n_steps": rows_written,
                "worker_id": args.worker_id,
                "shard_id": args.shard_id,
                "physical_gpu": args.physical_gpu,
                "shard_manifest_sha256": args.manifest_sha256,
                "clean_success_observed": None,
                **provenance,
            }
            write_json(metadata_path, failure)
            write_worker_status(
                args.worker_status_file.resolve(), worker_id=args.worker_id,
                physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
                phase="FAILED", completed=completed, failed=failed,
                current_parent_key=parent_key,
            )
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        write_worker_status(
            args.worker_status_file.resolve(), worker_id=args.worker_id,
            physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
            phase="RUNNING_EPISODES", completed=completed, failed=failed,
            current_parent_key=None,
        )
        print(
            f"[{completed}/{len(episodes)}] {parent_key}: {rows_written} rows",
            flush=True,
        )

    write_worker_status(
        args.worker_status_file.resolve(), worker_id=args.worker_id,
        physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
        phase="FINALIZING", completed=completed, failed=failed, current_parent_key=None,
    )
    artifact_entries = []
    for episode in episodes:
        episode_dir = output_root / "episodes" / suite / str(episode["parent_key"])
        for artifact in (
            episode_dir / "episode_metadata.json",
            episode_dir / "step_records.jsonl",
            episode_dir / "rgb_manifest.jsonl",
            episode_dir / "episode_receipt.json",
        ):
            if not artifact.is_file():
                raise RuntimeError(f"missing finalized episode artifact: {artifact}")
            artifact_entries.append({
                "path": artifact.relative_to(output_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })
    artifact_entries.sort(key=lambda row: row["path"])
    artifact_manifest = output_root / "c2g_r8w_collection_artifacts.jsonl"
    artifact_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifact_entries),
        encoding="utf-8",
    )
    report = {
        "schema": COLLECTION_SCHEMA,
        "status": "PASS_C2G_R8W_TEACHER_V2_CLEAN_SHARD_COLLECTION",
        "worker_id": args.worker_id,
        "shard_id": args.shard_id,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": visible,
        "model_device": args.device,
        "render_gpu_device_id": args.physical_gpu,
        "suite": suite,
        "collection_purpose": collection_purpose,
        "episode_count": len(results),
        "runtime_valid_episode_count": len(results),
        "new_episode_count": len(pending),
        "resumed_episode_count": len(resumed),
        "failed_episode_count": failed,
        "results": results,
        "shard_manifest": str(manifest_path),
        "shard_manifest_sha256": args.manifest_sha256,
        "artifact_manifest": str(artifact_manifest),
        "artifact_manifest_sha256": sha256_file(artifact_manifest),
        "model_verification_report": str(args.model_verification_report.resolve()),
        "model_verification_report_sha256": sha256_file(args.model_verification_report.resolve()),
        "openvla_model_loads": 1,
        "openvla_clean_inference_runs": len(pending),
        "libero_environments_created": len(pending),
        "libero_clean_rollouts": len(pending),
        "attacks_launched": 0,
        "attack_outcomes_read": False,
        **provenance,
    }
    report_path = output_root / "c2g_r8w_collection_report.json"
    write_json(report_path, report)
    write_worker_status(
        args.worker_status_file.resolve(), worker_id=args.worker_id,
        physical_gpu=args.physical_gpu, suite=suite, shard_id=args.shard_id,
        phase="PASS", completed=completed, failed=failed, current_parent_key=None,
    )
    print(json.dumps({**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
