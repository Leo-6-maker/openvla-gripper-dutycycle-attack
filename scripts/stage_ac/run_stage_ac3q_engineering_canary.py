#!/usr/bin/env python3
"""Qualify AC3 branch/endpoint machinery on consumed AC0 canaries only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
CANARY_KEYS = (
    "libero_10/task_04/state_20",
    "libero_object/task_02/state_42",
    "libero_spatial/task_05/state_34",
)
DOSES = (3, 5, 10)
ACTION_DIM = 7
ARM_TOLERANCE = 1e-7
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {"M0_OPENVLA": "FRESH_PER_STEP", "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE", "M2_PI05_LIBERO": "FRESH_PI05_REPLAN"}
GATE = "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"AC3Q_MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def step_unpack(result: Any) -> tuple[Any, bool]:
    require(isinstance(result, (tuple, list)) and len(result) >= 3, "AC3Q_ENV_STEP_RETURN_INVALID")
    return result[0], bool(result[2]) or bool(result[3]) if len(result) >= 5 else bool(result[2])


def counters() -> dict[str, int]:
    return {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "engineering_endpoint_evaluations": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "ac2_exposure": 0,
    }


def frame_from(obs: Any, env: Any) -> np.ndarray | None:
    import numpy as np

    if isinstance(obs, dict):
        for key in ("agentview_image", "agentview_rgb", "frontview_image"):
            value = obs.get(key)
            if value is not None:
                array = np.asarray(value)
                if array.ndim == 3 and array.shape[-1] in (3, 4):
                    return np.asarray(array[..., :3], dtype=np.uint8).copy()
    try:
        value = env.render()
    except Exception:
        return None
    array = np.asarray(value) if value is not None else None
    return np.asarray(array[..., :3], dtype=np.uint8).copy() if array is not None and array.ndim == 3 and array.shape[-1] >= 3 else None


def write_video(path: Path, frames: list[np.ndarray]) -> dict[str, Any]:
    require(frames, "AC3Q_VIDEO_NO_FRAMES")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=10, codec="libx264")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data), "frames": len(frames), "fps": 10}


def telemetry(aa1: ModuleType, env: Any, binding: dict[str, Any], target: str, counts: dict[str, int]) -> dict[str, Any]:
    value = aa1.telemetry(env, binding, target, counts)
    return value


def endpoint_trace(aa1: ModuleType, clean_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], dose: int, valid: bool, counts: dict[str, int]) -> str:
    counts["engineering_endpoint_evaluations"] += 1
    return str(aa1.endpoint_label(clean_rows, branch_rows, int(dose), valid))


def run_branch(
    aa1r1: ModuleType,
    aa1: ModuleType,
    config: dict[str, Any],
    family: str,
    canary: dict[str, Any],
    infer: Any,
    clean: dict[str, Any],
    point: dict[str, Any],
    dose: int | None,
    route: str,
    counts: dict[str, int],
    video_path: Path | None,
) -> dict[str, Any]:
    import numpy as np

    anchor_step = int(point["step"])
    required = 20 if dose is None else int(dose) + 10
    require(anchor_step + required <= len(clean["actions"]), "AC3Q_BRANCH_HORIZON_CENSORED")
    expected_state = clean["boundary_states"].get(anchor_step)
    require(expected_state is not None, "AC3Q_BRANCH_ANCHOR_STATE_MISSING")
    env, _task, obs, episode_done = aa1r1.replay_prefix(config, canary, clean["actions"], anchor_step, expected_state, counts)
    queue: list[tuple[np.ndarray, np.ndarray]] = []
    queue_boundaries: list[int] = []
    rows: list[dict[str, Any]] = []
    action_receipts: list[dict[str, Any]] = []
    policy_clean_arm_deltas: list[float] = []
    intervention_arm_deltas: list[float] = []
    frames: list[np.ndarray] = []
    first_frame = frame_from(obs, env)
    if first_frame is not None:
        frames.append(first_frame)
    try:
        for offset in range(required):
            step = anchor_step + offset
            require(not episode_done, "AC3Q_TERMINAL_BEFORE_REQUIRED_HORIZON")
            pre = telemetry(aa1, env, clean["binding"], str(clean["target_object"]), counts)
            is_clean = dose is None
            if is_clean:
                raw_action = np.asarray(clean["actions"][step]["raw"], dtype=np.float32)
                final_action = np.asarray(clean["actions"][step]["final"], dtype=np.float32)
            else:
                if not queue:
                    queue_boundaries.append(step)
                    queue = aa1.model_pairs(infer, obs, clean["language"], family, counts)
                raw_action, final_action = queue.pop(0)
                reference = np.asarray(clean["actions"][step]["final"], dtype=np.float32)
                policy_clean_arm_deltas.append(float(np.max(np.abs(reference[:6] - final_action[:6]))))
            require(raw_action.size == ACTION_DIM and final_action.size == ACTION_DIM, "AC3Q_ACTION_DIM_NOT_SEVEN")
            require(np.isfinite(raw_action).all() and np.isfinite(final_action).all(), "AC3Q_ACTION_NONFINITE")
            opened = False
            if not is_clean and offset < int(dose):
                opened_raw, opened_final = aa1.command_open_action(family, raw_action.tolist(), final_action.tolist(), int(dose))
                opened_raw = np.asarray(opened_raw, dtype=np.float32)
                opened_final = np.asarray(opened_final, dtype=np.float32)
                require(opened_raw.size == 7 and opened_final.size == 7, "AC3Q_OPENED_ACTION_DIM_INVALID")
                require(np.array_equal(opened_final[:6], final_action[:6]) and float(opened_final[-1]) == -1.0, "AC3Q_OPEN_ACTION_CONTRACT_INVALID")
                action = opened_final
                opened = True
                intervention_delta = float(np.max(np.abs(opened_final[:6] - final_action[:6])))
                intervention_arm_deltas.append(intervention_delta)
                action_receipts.append({
                    "step": step,
                    "raw_policy_action": raw_action.tolist(),
                    "opened_raw_action": opened_raw.tolist(),
                    "env_action": opened_final.tolist(),
                    "arm_delta_linf": intervention_delta,
                    "native_open_raw": float(opened_raw[-1]),
                    "native_open_final": float(opened_final[-1]),
                })
                counts["open_intervention_steps"] += 1
            else:
                action = final_action
            obs, episode_done = step_unpack(env.step(action.tolist()))
            counts["env_step_calls"] += 1
            post = telemetry(aa1, env, clean["binding"], str(clean["target_object"]), counts)
            next_frame = frame_from(obs, env)
            if next_frame is not None:
                frames.append(next_frame)
            rows.append({"step": step, "pre": pre, "post": post, "action": action.tolist(), "opened": opened, "terminal_after": episode_done})
        expected_boundaries = [] if dose is None else list(range(anchor_step, anchor_step + required, QUEUE_LENGTH[family]))
        queue_reset = dose is None or queue_boundaries == expected_boundaries
        telemetry_aligned = len(rows) == required and all(isinstance(row["pre"], dict) and isinstance(row["post"], dict) for row in rows)
        arm_preserved = all(delta <= ARM_TOLERANCE for delta in intervention_arm_deltas)
        exact_open = dose is None or (len(action_receipts) == int(dose) and [x["step"] for x in action_receipts] == list(range(anchor_step, anchor_step + int(dose))))
        valid = queue_reset and telemetry_aligned and arm_preserved and exact_open
        endpoint = "AC3Q_CLEAN_REFERENCE_ENDPOINT_NOT_RUN" if dose is None else endpoint_trace(aa1, clean.get("engineering_clean_rows", []), rows, int(dose), valid, counts)
        video = write_video(video_path, frames) if video_path is not None else None
        return {
            "status": "PASS_AC3Q_ENGINEERING_BRANCH" if valid else "AC3Q_ENGINEERING_BRANCH_INVALID",
            "route": route,
            "condition": "CLEAN_REFERENCE" if dose is None else f"OPEN_T{dose}",
            "dose": dose,
            "point": {k: point.get(k) for k in ("step", "point_label", "selection_rank_sha256", "boundary", "clean_boundary_state_sha256")},
            "required_horizon_steps": required,
            "state_restore_exact": True,
            "exact_open_delivery": exact_open,
            "arm_preserved": arm_preserved,
            "arm_preservation_basis": "OPENED_ACTION_VS_SAME_STEP_POLICY_ACTION",
            "max_arm_delta_linf": max(intervention_arm_deltas, default=0.0),
            "max_policy_vs_clean_arm_delta_linf": max(policy_clean_arm_deltas, default=0.0),
            "queue_reset_verified": queue_reset,
            "queue_boundary_steps": queue_boundaries,
            "expected_queue_boundary_steps": expected_boundaries,
            "telemetry_aligned": telemetry_aligned,
            "open_intervention_steps": len(action_receipts),
            "action_receipts": action_receipts,
            "engineering_endpoint_output": endpoint,
            "video": video,
            "rows": rows,
            "trace_digest": canonical_hash(rows),
        }
    finally:
        env.close()


def validate_static(args: argparse.Namespace, aa1: ModuleType) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    g0_root = load_json(args.g0_root)
    protocol = load_json(args.protocol)
    ac0 = load_json(args.ac0)
    aa1_protocol = load_json(args.aa1_protocol)
    plan = load_json(args.aa1_plan)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    z1_config = load_json(args.z1_config)
    require(g0_root.get("status") == "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE", "AC3Q_G0_NOT_FROZEN")
    require(protocol.get("status") == "STAGE_AC_AC3_G0_STATIC_FREEZE_AUTHORIZED", "AC3Q_PROTOCOL_NOT_FROZEN")
    require(protocol.get("gate") == GATE, "AC3Q_GATE_MISMATCH")
    require(ac0.get("fresh_science_authorized") is False, "AC3Q_AC0_FRESH_SCIENCE_NOT_FALSE")
    cells = [row for row in ac0.get("calibration_population", {}).get("cells", []) if row.get("model_family") == args.model_family and row.get("parent_key") == args.canonical_parent_key]
    require(len(cells) == 1 and args.canonical_parent_key in CANARY_KEYS, "AC3Q_CANARY_NOT_AC0_CONSUMED_CELL")
    canary = dict(cells[0])
    legacy_args = SimpleNamespace(model_family=args.model_family, gpu_id=args.gpu_id)
    static = aa1.static_validate(legacy_args, aa1_protocol, plan, aa0, capacity, z1_config, args.canonical_parent_key)
    legacy_canary = static["canary"]
    for key in ("suite", "task_idx", "state_id"):
        require(canary.get(key) == legacy_canary.get(key), f"AC3Q_CANARY_BINDING_MISMATCH:{key}")
    require(legacy_canary.get("permanent_exclusion") is True and legacy_canary.get("scientific_use") is False, "AC3Q_CANARY_EXCLUSION_INVALID")
    return canary, static, z1_config


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RuntimeError(f"AC3Q_OUTPUT_ALREADY_EXISTS:{args.output}")
    aa1r1 = load_module(ROOT / "scripts/stage_aa/run_stage_aa1r1_engineering_branch.py", "ac3q_aa1r1_runtime")
    aa1 = aa1r1.AA1
    canary, static, z1_config = validate_static(args, aa1)
    counts = counters()
    receipt: dict[str, Any] = {
        "schema": "STAGE_AC_AC3Q_ENGINEERING_CANARY_RECEIPT_V1",
        "status": "RUNNING",
        "gate": GATE,
        "model_family": args.model_family,
        "canonical_parent_key": args.canonical_parent_key,
        "canary": canary,
        "gpu_id": args.gpu_id,
        "permanent_exclusion": True,
        "scientific_use": False,
        "scientific_claim": "NONE_ENGINEERING_ONLY",
        "g0_root_sha256": sha256_file(args.g0_root),
        "runtime_counters": counts,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(args.output, receipt)
    aa1.require_single_gpu(args.gpu_id)
    gpu = aa1.gpu_snapshot(args.gpu_id)
    receipt["gpu_admission_snapshot"] = gpu
    aa1.Z1.configure_libero(z1_config)
    if args.model_family == "M2_PI05_LIBERO":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    receipt["runtime_environment"] = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET")}
    infer = model = None
    try:
        checkpoint = static["checkpoint"]
        if args.model_family == "M0_OPENVLA":
            infer, model, normalization = aa1.Z1.load_openvla(checkpoint, oft=False, suite=canary["suite"], return_chunk=True)
        elif args.model_family == "M1_OPENVLA_OFT":
            infer, model, normalization = aa1.Z1.load_openvla(checkpoint, oft=True, suite=canary["suite"], return_chunk=True)
        else:
            infer, model = aa1.Z1.load_pi05(checkpoint, return_chunk=True)
            normalization = {"checkpoint_mutated": False}
        clean = aa1r1.capture_clean(z1_config, args.model_family, {"suite": canary["suite"], "task_idx": canary["task_idx"], "state_id": canary["state_id"], "canonical_parent_key": args.canonical_parent_key}, infer, counts)
        receipt.update({"checkpoint": checkpoint, "normalization": normalization, "clean_status": clean.get("status"), "clean_trajectory_digest": clean.get("clean_trajectory_digest"), "engineering_point_selection": clean.get("points"), "runtime_counters": counts})
        point_ledger = {
            "schema": "STAGE_AC_AC3Q_ENGINEERING_POINT_LEDGER_V1",
            "status": "STAGE_AC_AC3Q_ENGINEERING_POINT_A_SEALED_BEFORE_OPEN",
            "gate": GATE,
            "model_family": args.model_family,
            "canonical_parent_key": args.canonical_parent_key,
            "selection_salt": aa1r1.AA1R1_SALT,
            "scientific_anchor_evaluation": "NOT_EVALUATED_BY_DESIGN",
            "engineering_points": clean.get("points"),
            "clean_trajectory_digest": clean.get("clean_trajectory_digest"),
            "checkpoint": checkpoint,
        }
        atomic_write(args.point_ledger, point_ledger)
        receipt["point_ledger_sealed"] = True
        selected = clean.get("points") or {}
        point = selected.get("point_a")
        if point is None:
            receipt.update({"status": "AC3Q_ENGINEERING_HOLD_NO_POINT", "branches": [], "branch_count": 0, "next_legal_action": "STOP_FOR_PI", "runtime_counters": counts})
            atomic_write(args.output, receipt)
            return receipt
        clean_video = args.video_dir / f"{args.model_family}_{args.canonical_parent_key.replace('/', '__')}_CLEAN_REFERENCE.mp4"
        first_clean = run_branch(aa1r1, aa1, z1_config, args.model_family, canary, infer, clean, point, None, "ENGINEERING_CLEAN_REFERENCE_A", counts, clean_video)
        second_clean = run_branch(aa1r1, aa1, z1_config, args.model_family, canary, infer, clean, point, None, "ENGINEERING_CLEAN_REPLAY_A_2", counts, None)
        clean["engineering_clean_rows"] = first_clean.get("rows", [])
        deterministic = first_clean.get("status") == "PASS_AC3Q_ENGINEERING_BRANCH" and second_clean.get("status") == "PASS_AC3Q_ENGINEERING_BRANCH" and first_clean.get("trace_digest") == second_clean.get("trace_digest")
        determinism = {"status": "PASS_AC3Q_CLEAN_REPLAY_DETERMINISTIC" if deterministic else "AC3Q_CLEAN_REPLAY_NONDETERMINISTIC", "first": first_clean, "second": second_clean, "trace_equal": first_clean.get("trace_digest") == second_clean.get("trace_digest")}
        branches = [first_clean]
        for dose in DOSES:
            video = args.video_dir / f"{args.model_family}_{args.canonical_parent_key.replace('/', '__')}_OPEN_T{dose}.mp4"
            branches.append(run_branch(aa1r1, aa1, z1_config, args.model_family, canary, infer, clean, point, dose, f"ENGINEERING_OPEN_T{dose}_A", counts, video))
        point_b = selected.get("point_b")
        if point_b is not None:
            video = args.video_dir / f"{args.model_family}_{args.canonical_parent_key.replace('/', '__')}_OPEN_T5_B.mp4"
            branches.append(run_branch(aa1r1, aa1, z1_config, args.model_family, canary, infer, clean, point_b, 5, "ENGINEERING_OPEN_T5_B", counts, video))
        all_pass = deterministic and all(branch.get("status") == "PASS_AC3Q_ENGINEERING_BRANCH" for branch in branches)
        receipt.update({
            "status": "PASS_AC3Q_ENGINEERING_BRANCH_CELL" if all_pass else "AC3Q_ENGINEERING_HOLD_BRANCH_INVALID",
            "clean_replay": determinism,
            "branches": branches,
            "branch_count": len(branches),
            "branch_pass_count": sum(branch.get("status") == "PASS_AC3Q_ENGINEERING_BRANCH" for branch in branches),
            "next_legal_action": "EXECUTE_G1_NEXT_CONSUMED_CANARY" if all_pass else "STOP_FOR_PI",
            "claim_boundary": "AC3Q consumed-only branch/endpoint qualification; no scientific outcome",
            "runtime_counters": counts,
        })
        atomic_write(args.output, receipt)
        return receipt
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def self_test() -> None:
    assert len(CANARY_KEYS) == 3
    assert {"OPEN_T3", "OPEN_T5", "OPEN_T10"} == {f"OPEN_T{dose}" for dose in DOSES}
    assert QUEUE_LENGTH["M2_PI05_LIBERO"] == 5
    print(json.dumps({"status": "AC3Q_STATIC_SELF_TEST_PASS", "canaries": 3, "routes_per_cell": 4}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path, required=False, default=ROOT / "configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json")
    parser.add_argument("--g0-root", type=Path, required=False, default=ROOT / "reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json")
    parser.add_argument("--ac0", type=Path, required=False, default=ROOT / "configs/STAGE_AC_AC0_CONSTRUCT_VALIDATION_PROTOCOL_V1.json")
    parser.add_argument("--aa1-protocol", type=Path, required=False, default=ROOT / "configs/STAGE_AA_AA1_ENGINEERING_RUNTIME_PROTOCOL_V1.json")
    parser.add_argument("--aa1-plan", type=Path, required=False, default=ROOT / "reports/STAGE_AA_AA1_ENGINEERING_CANARY_PLAN_V1.json")
    parser.add_argument("--aa0", type=Path, required=False, default=ROOT / "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json")
    parser.add_argument("--capacity", type=Path, required=False, default=ROOT / "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json")
    parser.add_argument("--z1-config", type=Path, required=False, default=ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json")
    parser.add_argument("--model-family", choices=MODELS)
    parser.add_argument("--canonical-parent-key")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--point-ledger", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ("model_family", "canonical_parent_key", "gpu_id", "output", "point_ledger", "video_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-') } is required unless --self-test is used")
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "branch_count": result.get("branch_count", 0), "open_steps": result.get("runtime_counters", {}).get("open_intervention_steps", 0)}, sort_keys=True))
        return 0 if result["status"].startswith("PASS_") else 1
    except Exception as exc:
        failure = {
            "schema": "STAGE_AC_AC3Q_ENGINEERING_CANARY_RECEIPT_V1",
            "status": "AC3Q_ENGINEERING_HOLD_RUNTIME_ERROR",
            "gate": GATE,
            "model_family": args.model_family,
            "canonical_parent_key": args.canonical_parent_key,
            "gpu_id": args.gpu_id,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
            "next_legal_action": "STOP_FOR_PI",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if args.output is not None:
            atomic_write(args.output, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
