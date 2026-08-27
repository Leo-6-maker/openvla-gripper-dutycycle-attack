#!/usr/bin/env python3
"""Run one consumed-only Stage AC0 clean calibration cell.

AC0 is a measurement calibration gate.  It never runs OPEN, PGD, endpoint
scoring, protected evaluation, or any fresh scientific identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_ac.eligibility_v2 import (
    ABSOLUTE_OBJECT_EEF_DISTANCE_MAX_M,
    CLEAN_CONTINUATION_STEPS,
    DISTANCE_CONSISTENCY_TOLERANCE_M,
    MIN_LIFT_M,
    RELATIVE_CARRY_DISPLACEMENT_MAX_M,
    STABLE_GRASP_WINDOW_STEPS,
    classify_calibration_control,
    scan_candidates,
    telemetry_valid,
)


MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
MIN_FREE_MIB = 20_480


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA2R2 = load_module(ROOT / "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py", "ac0_aa2r2_runtime")
RUNTIME = AA2R2.AA1
SEMANTICS = AA2R2.SEMANTICS


def load_json(path: Path) -> Dict[str, Any]:
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


def atomic_write(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_scalar(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, int, np.floating, np.integer)):
        number = float(value)
        if np.isfinite(number):
            return number
        if np.isnan(number):
            return "NaN"
        return "Infinity" if number > 0 else "-Infinity"
    return value


def safe_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return safe_array(value.tolist())
    if isinstance(value, (list, tuple)):
        return [safe_array(item) for item in value]
    return safe_scalar(value)


def set_clean_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def step_unpack(result: Any) -> Tuple[Any, bool]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise RuntimeError("AC0_ENV_STEP_RETURN_INVALID")
    return result[0], bool(result[2]) if len(result) == 3 else bool(result[2] or result[3])


def frame_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    frame = np.asarray(obs.get("agentview_image"))
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise RuntimeError(f"AC0_AGENTVIEW_FRAME_INVALID:{frame.shape}")
    frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[:2] != (256, 256):
        from PIL import Image

        frame = np.asarray(Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS), dtype=np.uint8)
    return np.ascontiguousarray(frame)


def open_video_writer(path: Path) -> Any:
    try:
        import imageio.v2 as imageio

        return imageio.get_writer(str(path), fps=10, codec="libx264", macro_block_size=1)
    except Exception as exc:
        raise RuntimeError(f"AC0_VIDEO_WRITER_UNAVAILABLE:{type(exc).__name__}:{exc}")


def gpu_snapshot(gpu_id: int) -> Dict[str, Any]:
    snapshot = RUNTIME.gpu_snapshot(gpu_id)
    if int(snapshot["free_memory_mib"]) <= MIN_FREE_MIB:
        raise RuntimeError(f"AC0_GPU_NOT_ELIGIBLE:{snapshot['free_memory_mib']}")
    return snapshot


def find_canary(plan: Dict[str, Any], family: str, parent_key: str) -> Dict[str, Any]:
    rows = [row for row in plan.get("canaries", []) if row.get("model_family") == family and row.get("canonical_parent_key") == parent_key]
    if len(rows) != 1:
        raise RuntimeError(f"AC0_CANARY_NOT_UNIQUE:{family}:{parent_key}")
    canary = rows[0]
    if canary.get("permanent_exclusion") is not True or canary.get("scientific_use") is not False:
        raise RuntimeError("AC0_CANARY_EXCLUSION_FIREWALL_INVALID")
    return canary


def validate_static(args: argparse.Namespace, protocol: Dict[str, Any], plan: Dict[str, Any], z1_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
    if protocol.get("status") != "STAGE_AC_AC0_PROVISIONAL_ENGINEERING_CALIBRATION_ONLY":
        raise RuntimeError("AC0_PROTOCOL_STATUS_INVALID")
    if protocol.get("fresh_science_authorized") is not False:
        raise RuntimeError("AC0_FRESH_SCIENCE_FIREWALL_INVALID")
    if protocol.get("calibration_population", {}).get("cell_count") != 9:
        raise RuntimeError("AC0_CALIBRATION_CELL_COUNT_INVALID")
    if plan.get("status") != "STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_FROZEN" or plan.get("cell_count") != 9:
        raise RuntimeError("AC0_CANARY_PLAN_NOT_FROZEN")
    eligibility = protocol.get("eligibility_v2", {})
    expected_eligibility = {
        "stable_grasp_window_steps": STABLE_GRASP_WINDOW_STEPS,
        "clean_continuation_steps": CLEAN_CONTINUATION_STEPS,
        "absolute_object_eef_distance_max_m": ABSOLUTE_OBJECT_EEF_DISTANCE_MAX_M,
        "relative_carry_displacement_max_m": RELATIVE_CARRY_DISPLACEMENT_MAX_M,
        "minimum_lift_m": MIN_LIFT_M,
        "full_episode_horizon_required": False,
        "distance_field_consistency_tolerance_m": DISTANCE_CONSISTENCY_TOLERANCE_M,
    }
    if any(eligibility.get(key) != value for key, value in expected_eligibility.items()):
        raise RuntimeError("AC0_ELIGIBILITY_PROTOCOL_BINDING_INVALID")
    variants = protocol.get("contact_flicker_calibration", {}).get("variants", [])
    if [(row.get("id"), row.get("max_false_contact_rows_after_initial_window")) for row in variants] != [("STRICT_NO_FLICKER", 0), ("ONE_ROW_FLICKER", 1)]:
        raise RuntimeError("AC0_FLICKER_VARIANTS_INVALID")
    canary = find_canary(plan, args.model_family, args.canonical_parent_key)
    protocol_cells = [row for row in protocol["calibration_population"]["cells"] if row.get("model_family") == args.model_family and row.get("parent_key") == args.canonical_parent_key]
    if len(protocol_cells) != 1:
        raise RuntimeError("AC0_PARENT_NOT_IN_PROTOCOL")
    protocol_cell = protocol_cells[0]
    for field in ("suite", "task_idx", "state_id", "seed"):
        if canary.get(field) != protocol_cell.get(field):
            raise RuntimeError(f"AC0_CANARY_PROTOCOL_BINDING_MISMATCH:{field}")
    if args.canonical_parent_key in set(load_json(ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json").get("selected_parent_keys", [])):
        raise RuntimeError("AC0_CANARY_OVERLAPS_STAGE_Z")
    checkpoint = AA2R2.checkpoint_path(z1_config, args.model_family, str(canary["suite"]))
    if not checkpoint.is_dir():
        raise RuntimeError(f"AC0_CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
    if args.model_family == "M1_OPENVLA_OFT":
        RUNTIME.Z1.verify_m1_materialization(
            Path(args.m1_manifest),
            checkpoint,
            str(canary["suite"]),
            str(z1_config["model_families"][args.model_family]["checkpoint_manifests_sha256"]),
        )
    output = args.output.resolve()
    video = args.video.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or video.exists() or video.with_suffix(video.suffix + ".partial").exists():
        raise RuntimeError("AC0_APPEND_ONLY_OUTPUT_EXISTS")
    if shutil.disk_usage(str(output.parent)).free < int(args.min_free_gib * 1024**3):
        raise RuntimeError(f"AC0_STORAGE_RESERVE_TOO_LOW:{shutil.disk_usage(str(output.parent)).free}")
    return canary, checkpoint


def capture_clean(
    config: Dict[str, Any],
    family: str,
    canary: Dict[str, Any],
    infer: Any,
    counters: Dict[str, int],
    context: Dict[str, Any],
    video_path: Path,
    eligibility_spec: Dict[str, Any],
    flicker_variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    suite = str(canary["suite"])
    env, _task_suite, task, obs, _initial_states = RUNTIME.make_env(config, suite, int(canary["task_idx"]), int(canary["state_id"]), counters)
    partial_video = video_path.with_suffix(video_path.suffix + ".partial")
    writer = open_video_writer(partial_video)
    try:
        binding = RUNTIME.TAXONOMY.bind_object_taxonomy(env, RUNTIME.bddl_path(env, task))
        if binding.get("status") != "PASS":
            raise RuntimeError(f"AC0_OBJECT_BINDING_INVALID:{binding}")
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: List[Tuple[np.ndarray, np.ndarray]] = []
        rows: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []
        boundary_states: Dict[str, Dict[str, Any]] = {}
        eligibility_rows: List[Dict[str, Any]] = []
        done = False
        baseline_z: Optional[float] = None
        horizon = HORIZONS[suite]
        for step in range(horizon):
            if done:
                break
            fresh = not queue
            if fresh:
                context["current_step"] = step
                snapshot = AA2R2.Z1.snapshot_state(env)
                boundary_states[str(step)] = {"sha256": sha256_bytes(np.asarray(snapshot).tobytes()), "state": safe_array(snapshot)}
                queue = AA2R2.model_pairs_v2(infer, obs, language, family, counters, context)
            raw_action, final_action = queue.pop(0)
            pre = RUNTIME.telemetry(env, binding, target, counters)
            if baseline_z is None and isinstance(pre.get("object_position"), list) and len(pre["object_position"]) == 3:
                baseline_z = float(pre["object_position"][2])
            semantics = SEMANTICS.validate_action_pair(
                family,
                raw_action.tolist(),
                final_action.tolist(),
                raw_gripper=float(raw_action[-1]),
                final_gripper=float(final_action[-1]),
            )
            if not semantics.get("accepted"):
                raise RuntimeError(f"AC0_ACTION_SEMANTICS_INVALID:{semantics.get('reason')}")
            writer.append_data(frame_from_obs(obs))
            obs, done = step_unpack(env.step(final_action.tolist()))
            counters["env_step_calls"] += 1
            post = RUNTIME.telemetry(env, binding, target, counters)
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "model_boundary": fresh,
                "terminal_before": False,
                "terminal_after": done,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                "action_semantics": semantics,
                "pre": pre,
                "post": post,
                "boundary_state_sha256": boundary_states[str(step)]["sha256"] if fresh else None,
            }
            rows.append(row)
            actions.append({"step": step, "boundary": fresh, "raw": raw_action.tolist(), "final": final_action.tolist()})
            eligibility_rows.append({**post, "step": step, "model_boundary": fresh, "terminal_before": False, "terminal_after": done})
        writer.close()
        os.replace(partial_video, video_path)
        critical = {}
        controls = {}
        for variant in flicker_variants:
            variant_id = str(variant["id"])
            max_false = int(variant["max_false_contact_rows_after_initial_window"])
            critical_candidates, critical_reasons = scan_candidates(
                eligibility_rows,
                actions,
                family,
                str(canary["canonical_parent_key"]),
                baseline_z,
                str(eligibility_spec["critical_selection_salt"]),
                "CRITICAL",
                max_false,
            )
            noncritical_candidates, noncritical_reasons = scan_candidates(
                eligibility_rows,
                actions,
                family,
                str(canary["canonical_parent_key"]),
                baseline_z,
                str(eligibility_spec["noncritical_selection_salt"]),
                "NONCRITICAL",
                max_false,
            )
            critical[variant_id] = {"max_contact_false_rows": max_false, "candidates": critical_candidates, "reason_counts": critical_reasons}
            controls[variant_id] = {"candidates": noncritical_candidates, "reason_counts": noncritical_reasons}
        control_labels = [
            {"step": int(row["step"]), "label": classify_calibration_control(eligibility_rows, int(row["step"]), baseline_z)}
            for row in eligibility_rows
            if row.get("model_boundary") is True
        ]
        return {
            "status": "AC0_CLEAN_CAPTURE_COMPLETE",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "steps_captured": len(rows),
            "complete_trajectory": len(rows) == horizon,
            "baseline_z_m": baseline_z,
            "telemetry_valid_rows": sum(telemetry_valid(row) for row in eligibility_rows),
            "clean_trajectory_digest": canonical_hash({"rows": rows, "actions": actions, "boundary_states": boundary_states}),
            "rows": rows,
            "actions": actions,
            "boundary_states": boundary_states,
            "eligibility_rows": eligibility_rows,
            "eligibility_diagnostics": {"critical": critical, "noncritical": controls, "control_labels_at_boundaries": control_labels},
            "video": {"path": str(video_path), "bytes": video_path.stat().st_size, "sha256": sha256_file(video_path), "fps": 10, "frames": len(rows), "width": 256, "height": 256},
        }
    finally:
        try:
            writer.close()
        except Exception:
            pass
        env.close()


def run_cell(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = load_json(args.protocol)
    plan = load_json(args.canary_plan)
    z1_config = load_json(args.z1_config)
    canary, checkpoint = validate_static(args, protocol, plan, z1_config)
    counters = {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "aa_v_phys_reads": 0,
        "v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "aa2_exposure": 0,
    }
    receipt: Dict[str, Any] = {
        "schema": "STAGE_AC_AC0_CALIBRATION_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "model_family": args.model_family,
        "canonical_parent_key": args.canonical_parent_key,
        "suite": canary["suite"],
        "task_idx": canary["task_idx"],
        "state_id": canary["state_id"],
        "seed": canary["seed"],
        "gpu_id": args.gpu_id,
        "checkpoint": str(checkpoint),
        "permanent_exclusion": True,
        "scientific_use": False,
        "fresh_scientific_exposure": 0,
        "runtime_counters": counters,
        "video_path": str(args.video.resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(args.output, receipt)
    context = {
        "receipt": receipt,
        "output": args.output,
        "counters": counters,
        "pair_audits": [],
        "family": args.model_family,
        "current_step": 0,
        "failure_persisted": False,
    }
    model = None
    try:
        RUNTIME.require_single_gpu(args.gpu_id)
        receipt["gpu"] = gpu_snapshot(args.gpu_id)
        set_clean_seed(int(canary["seed"]))
        RUNTIME.Z1.configure_libero(z1_config)
        if args.model_family == "M2_PI05_LIBERO":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if args.model_family == "M0_OPENVLA":
            infer, model, normalization = RUNTIME.Z1.load_openvla(str(checkpoint), oft=False, suite=str(canary["suite"]), return_chunk=True)
        elif args.model_family == "M1_OPENVLA_OFT":
            infer, model, normalization = RUNTIME.Z1.load_openvla(str(checkpoint), oft=True, suite=str(canary["suite"]), return_chunk=True)
        else:
            infer, model = RUNTIME.Z1.load_pi05(str(checkpoint), return_chunk=True)
            normalization = {"checkpoint_mutated": False}
        clean = capture_clean(
            z1_config,
            args.model_family,
            canary,
            infer,
            counters,
            context,
            args.video.resolve(),
            protocol["eligibility_v2"],
            protocol["contact_flicker_calibration"]["variants"],
        )
        receipt.update(
            {
                "status": "PASS_AC0_CALIBRATION_CELL",
                "normalization": normalization,
                "clean": clean,
                "action_pair_audit": context["pair_audits"],
                "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
                "runtime_counters": counters,
                "scientific_claim": "NONE_ENGINEERING_CALIBRATION_ONLY",
                "claim_boundary": "AC0 consumed-only construct validation; no fresh scientific denominator and no treatment.",
                "next_legal_action": "STOP_FOR_PI_AFTER_AC0",
            }
        )
        atomic_write(args.output, receipt)
        return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "AC0_ENGINEERING_HOLD_RUNTIME_ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc), "action_pair_audit_count": len(context["pair_audits"])},
                "runtime_counters": counters,
                "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
                "next_legal_action": "STOP_FOR_PI",
            }
        )
        atomic_write(args.output, receipt)
        raise
    finally:
        if model is not None:
            del model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODELS, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AC0_ENGINEERING_HOLD_RUNTIME_ERROR", "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
