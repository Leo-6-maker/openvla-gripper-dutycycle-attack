#!/usr/bin/env python3
"""AA2R2 Phase-A clean-only action-semantics requalification.

This runner reuses the frozen model/LIBERO loaders but does not call the AA2
scientific eligibility scanner or any treatment/endpoint path.  The three
canaries are permanently excluded from the AA2--AA5 scientific population.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_aa import action_semantics_v2 as SEMANTICS


MODELS = (SEMANTICS.MODEL_M0, SEMANTICS.MODEL_M1, SEMANTICS.MODEL_M2)
SUITES = ("libero_10", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
QUEUE_LENGTH = {SEMANTICS.MODEL_M0: 1, SEMANTICS.MODEL_M1: 8, SEMANTICS.MODEL_M2: 5}
BOUNDARY = {
    SEMANTICS.MODEL_M0: "FRESH_PER_STEP",
    SEMANTICS.MODEL_M1: "FRESH_OFT_ACTION_QUEUE",
    SEMANTICS.MODEL_M2: "FRESH_PI05_REPLAN",
}
ACTION_DIM = SEMANTICS.ACTION_DIM
MIN_FREE_MIB = 20_480


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA2 = load_module(ROOT / "scripts/stage_aa/run_stage_aa2_clean_screen.py", "aa2r2_aa2_runtime")
AA1 = AA2.AA1
Z1 = AA1.Z1


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
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_scalar(value: Any) -> float | int | str | Any:
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


def gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    snapshot = AA1.gpu_snapshot(gpu_id)
    if snapshot["free_memory_mib"] <= MIN_FREE_MIB:
        raise RuntimeError(f"GPU_NOT_ELIGIBLE_FREE_MEMORY_MIB:{snapshot['free_memory_mib']}")
    return snapshot


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


def checkpoint_path(config: dict[str, Any], family: str, suite: str) -> Path:
    spec = config["model_families"][family]
    if family == SEMANTICS.MODEL_M0:
        return Path(spec["paths"][suite])
    if family == SEMANTICS.MODEL_M1:
        return Path(spec["checkpoint_root"]) / suite
    return Path(spec["checkpoint"])


def find_canary(plan: dict[str, Any], family: str, parent_key: str) -> dict[str, Any]:
    rows = [
        row
        for row in plan.get("canaries", [])
        if row.get("model_family") == family and row.get("canonical_parent_key") == parent_key
    ]
    if len(rows) != 1:
        raise RuntimeError(f"AA2R2_CANARY_CELL_NOT_UNIQUE:{family}:{parent_key}")
    row = rows[0]
    if row.get("permanent_exclusion") is not True or row.get("scientific_use") is not False:
        raise RuntimeError("AA2R2_CANARY_EXCLUSION_FIREWALL_INVALID")
    return row


def validate_static(
    protocol: dict[str, Any],
    plan: dict[str, Any],
    aa0: dict[str, Any],
    capacity: dict[str, Any],
    z1_config: dict[str, Any],
    family: str,
    parent_key: str,
) -> tuple[dict[str, Any], Path]:
    if protocol.get("status") != "STAGE_AA_AA2R2_ACTION_SEMANTICS_AMENDMENT_AUTHORIZED":
        raise RuntimeError("AA2R2_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("phase_a", {}).get("authorized") is not True:
        raise RuntimeError("AA2R2_PHASE_A_NOT_AUTHORIZED")
    if protocol.get("scientific_firewall", {}).get("aa2_scientific_parent_exposure") != 0:
        raise RuntimeError("AA2R2_SCIENTIFIC_FIREWALL_INVALID")
    if plan.get("status") != "STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_FROZEN":
        raise RuntimeError("AA2R2_CANARY_PLAN_NOT_FROZEN")
    if plan.get("cell_count") != 9 or len(plan.get("canaries", [])) != 9:
        raise RuntimeError("AA2R2_CANARY_PLAN_CELL_COUNT_INVALID")
    if family not in MODELS or family not in plan.get("model_families", []):
        raise RuntimeError("AA2R2_MODEL_FAMILY_INVALID")
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    if z1_config.get("status") != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_RUNTIME_AUTHORITY_NOT_FROZEN")
    canary = find_canary(plan, family, parent_key)
    capacity_rows = capacity.get("aa1_engineering_canary_reservation", {}).get("reserved_rows", [])
    matches = [row for row in capacity_rows if row.get("canonical_parent_key") == parent_key]
    if len(matches) != 1 or matches[0].get("selection_rank_sha256") != canary.get("source_selection_rank_sha256"):
        raise RuntimeError("AA2R2_CANARY_CAPACITY_BINDING_MISMATCH")
    pool = set(capacity.get("analysis_pool_after_aa1_reservation", {}).get("keys", []))
    if parent_key in pool:
        raise RuntimeError("AA2R2_CANARY_REMAINS_IN_SCIENTIFIC_POOL")
    stage_z_panel = load_json(ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json")
    if parent_key in set(stage_z_panel.get("selected_parent_keys", [])):
        raise RuntimeError("AA2R2_CANARY_OVERLAPS_STAGE_Z")
    checkpoint = checkpoint_path(z1_config, family, str(canary["suite"]))
    if not checkpoint.exists():
        raise RuntimeError(f"CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
    if family == SEMANTICS.MODEL_M1:
        manifest = ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
        Z1.verify_m1_materialization(
            manifest,
            checkpoint,
            str(canary["suite"]),
            str(z1_config["model_families"][family]["checkpoint_manifests_sha256"]),
        )
    return canary, checkpoint


def _active_failure(
    context: dict[str, Any],
    *,
    status: str,
    error_type: str,
    message: str,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    receipt = context["receipt"]
    receipt.update(
        {
            "status": status,
            "error": {
                "type": error_type,
                "message": message,
                "diagnostics": diagnostics,
            },
            "action_pair_audit_count": len(context["pair_audits"]),
            "runtime_counters": dict(context["counters"]),
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
            "next_legal_action": "STOP_FOR_PI",
        }
    )
    atomic_write(context["output"], receipt)
    context["failure_persisted"] = True


def _record_pair_failure(context: dict[str, Any], check: dict[str, Any], raw_action: np.ndarray, final_action: np.ndarray, *, step: int, boundary_step: int, queue_index: int, meta: dict[str, Any]) -> None:
    record = {
        "cell_id": context["receipt"]["cell_id"] if "cell_id" in context["receipt"] else None,
        "model_family": context["family"],
        "canonical_parent_key": context["receipt"].get("canonical_parent_key"),
        "seed": context["receipt"].get("seed"),
        "step": step,
        "boundary_step": boundary_step,
        "queue_index": queue_index,
        "model_boundary": BOUNDARY[context["family"]],
        "queue_or_replan_boundary": BOUNDARY[context["family"]],
        "reported_fresh_boundary": meta.get("fresh_boundary"),
        "raw_action_7d": safe_array(raw_action),
        "final_action_7d": safe_array(final_action),
        "raw_gripper": safe_scalar(raw_action[-1]),
        "final_gripper": safe_scalar(final_action[-1]),
        "expected_final_gripper": check.get("expected_final_gripper"),
        "expected_final_action": check.get("expected_final_action"),
        "validator_version": check.get("validator_version"),
        "rule": check.get("rule"),
        "reason": check.get("reason"),
        "semantic_state": check.get("semantic_state"),
        "metadata": {"fresh_boundary": meta.get("fresh_boundary")},
    }
    record["first_offending_row_digest"] = canonical_hash({key: value for key, value in record.items() if key != "first_offending_row_digest"})
    _active_failure(
        context,
        status="AA2R2_ENGINEERING_INVALID_ACTION_SEMANTICS",
        error_type="RuntimeError",
        message=f"ACTION_SEMANTICS_INVALID:{context['family']}:{check.get('reason')}",
        diagnostics={"offending_row": record},
    )


def model_pairs_v2(infer: Any, obs: dict[str, Any], language: str, family: str, counters: dict[str, int], context: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    chunk, meta = infer(obs, language)
    counters["model_inference_calls"] += 1
    meta = meta if isinstance(meta, dict) else {}
    raw = np.asarray(meta.get("raw_action_chunk"), dtype=np.float32)
    final = np.asarray(chunk, dtype=np.float32)
    boundary_step = int(context["current_step"])
    if raw.ndim != 2 or final.ndim != 2 or raw.shape != final.shape or raw.shape[1] != ACTION_DIM:
        _active_failure(
            context,
            status="AA2R2_ENGINEERING_INVALID_ACTION_SHAPE",
            error_type="RuntimeError",
            message=f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{final.shape}",
            diagnostics={
                "step": boundary_step,
                "model_boundary": BOUNDARY[family],
                "raw_action_chunk": safe_array(raw) if raw.ndim == 2 else None,
                "final_action_chunk": safe_array(final) if final.ndim == 2 else None,
            },
        )
        raise RuntimeError(f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{final.shape}")
    if meta.get("fresh_boundary") != BOUNDARY[family]:
        _active_failure(
            context,
            status="AA2R2_ENGINEERING_INVALID_MODEL_BOUNDARY",
            error_type="RuntimeError",
            message=f"MODEL_BOUNDARY_INVALID:{family}:{meta.get('fresh_boundary')}",
            diagnostics={"step": boundary_step, "expected": BOUNDARY[family], "observed": meta.get("fresh_boundary")},
        )
        raise RuntimeError(f"MODEL_BOUNDARY_INVALID:{family}:{meta.get('fresh_boundary')}")
    length = QUEUE_LENGTH[family]
    if raw.shape[0] < length:
        _active_failure(
            context,
            status="AA2R2_ENGINEERING_INVALID_ACTION_QUEUE",
            error_type="RuntimeError",
            message=f"ACTION_CHUNK_TOO_SHORT:{family}:{raw.shape[0]}:{length}",
            diagnostics={"step": boundary_step, "raw_rows": int(raw.shape[0]), "required_rows": length},
        )
        raise RuntimeError(f"ACTION_CHUNK_TOO_SHORT:{family}:{raw.shape[0]}:{length}")
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(length):
        raw_action = raw[index].copy()
        final_action = final[index].copy()
        check = SEMANTICS.validate_action_pair(
            family,
            raw_action.tolist(),
            final_action.tolist(),
            raw_gripper=float(raw_action[-1]),
            final_gripper=float(final_action[-1]),
        )
        step = boundary_step + index
        audit = {
            "step": step,
            "boundary_step": boundary_step,
            "queue_index": index,
            "model_boundary": BOUNDARY[family],
            "reported_fresh_boundary": meta.get("fresh_boundary"),
            "semantics": check,
        }
        context["pair_audits"].append(audit)
        if not check.get("accepted"):
            _record_pair_failure(context, check, raw_action, final_action, step=step, boundary_step=boundary_step, queue_index=index, meta=meta)
            raise RuntimeError(f"ACTION_SEMANTICS_INVALID:{family}:{check.get('reason')}")
        pairs.append((raw_action, final_action))
    return pairs


def capture_engineering_clean(config: dict[str, Any], family: str, canary: dict[str, Any], infer: Any, counters: dict[str, int], context: dict[str, Any]) -> dict[str, Any]:
    suite = str(canary["suite"])
    task_idx = int(canary["task_idx"])
    state_id = int(canary["state_id"])
    parent_key = str(canary["canonical_parent_key"])
    env, _task_suite, task, obs, _initial_states = AA1.make_env(config, suite, task_idx, state_id, counters)
    try:
        binding = AA1.TAXONOMY.bind_object_taxonomy(env, AA1.bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {"status": "AA2R2_ENGINEERING_OBJECT_BINDING_INVALID", "binding": binding, "rows": [], "boundary_state_sha256": {}}
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        boundary_states: dict[int, str] = {}
        done = False
        horizon = HORIZONS[suite]
        for step in range(horizon):
            if done:
                break
            fresh = not queue
            if fresh:
                context["current_step"] = step
                snapshot = Z1.snapshot_state(env)
                boundary_states[step] = sha256_bytes(np.asarray(snapshot).tobytes())
                queue = model_pairs_v2(infer, obs, language, family, counters, context)
            raw_action, final_action = queue.pop(0)
            current = AA1.telemetry(env, binding, target, counters)
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "model_boundary": fresh,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                **current,
            }
            rows.append(row)
            obs, done = AA2.step_unpack(env.step(final_action.tolist()))
            counters["env_step_calls"] += 1
            row["terminal_after"] = done
        complete = len(rows) == horizon
        return {
            "status": "PASS_AA2R2_ENGINEERING_CLEAN_TRAJECTORY" if complete else "AA2R2_ENGINEERING_INCOMPLETE_TRAJECTORY",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "steps_captured": len(rows),
            "complete_trajectory": complete,
            "boundary_count": len(boundary_states),
            "boundary_state_sha256": {str(step): digest for step, digest in boundary_states.items()},
            "clean_trajectory_digest": canonical_hash({"rows": rows, "boundary_state_sha256": boundary_states}),
            "rows": rows,
        }
    finally:
        env.close()


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    plan = load_json(args.canary_plan)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    z1_config = load_json(args.z1_config)
    canary = find_canary(plan, args.model_family, args.canonical_parent_key)
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
    receipt: dict[str, Any] = {
        "schema": "STAGE_AA_AA2R2_ENGINEERING_CANARY_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "phase": "A",
        "model_family": args.model_family,
        "canonical_parent_key": args.canonical_parent_key,
        "suite": canary["suite"],
        "task_idx": canary["task_idx"],
        "state_id": canary["state_id"],
        "seed": canary["seed"],
        "gpu_id": args.gpu_id,
        "canary_permanent_exclusion": True,
        "scientific_use": False,
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    context: dict[str, Any] = {
        "receipt": receipt,
        "output": args.output,
        "counters": counters,
        "pair_audits": [],
        "family": args.model_family,
        "current_step": 0,
        "failure_persisted": False,
    }
    atomic_write(args.output, receipt)
    model = None
    try:
        canary, checkpoint = validate_static(protocol, plan, aa0, capacity, z1_config, args.model_family, args.canonical_parent_key)
        AA1.require_single_gpu(args.gpu_id)
        receipt["gpu"] = gpu_snapshot(args.gpu_id)
        set_clean_seed(int(canary["seed"]))
        Z1.configure_libero(z1_config)
        if args.model_family == SEMANTICS.MODEL_M2:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        receipt["checkpoint"] = str(checkpoint)
        receipt["runtime_environment"] = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET"),
        }
        if args.model_family == SEMANTICS.MODEL_M0:
            infer, model, normalization = Z1.load_openvla(str(checkpoint), oft=False, suite=str(canary["suite"]), return_chunk=True)
        elif args.model_family == SEMANTICS.MODEL_M1:
            infer, model, normalization = Z1.load_openvla(str(checkpoint), oft=True, suite=str(canary["suite"]), return_chunk=True)
        else:
            infer, model = Z1.load_pi05(str(checkpoint), return_chunk=True)
            normalization = {"checkpoint_mutated": False}
        clean = capture_engineering_clean(z1_config, args.model_family, canary, infer, counters, context)
        if clean.get("status") != "PASS_AA2R2_ENGINEERING_CLEAN_TRAJECTORY":
            raise RuntimeError(clean.get("status", "AA2R2_ENGINEERING_CLEAN_TRAJECTORY_INVALID"))
        if len(clean["rows"]) != counters["env_step_calls"]:
            raise RuntimeError("AA2R2_TELEMETRY_ACTION_ONE_TO_ONE_INVALID")
        expected_pairs = len(clean["rows"])
        if len(context["pair_audits"]) < expected_pairs:
            raise RuntimeError("AA2R2_ACTION_AUDIT_INCOMPLETE")
        receipt.update(
            {
                "status": "PASS_AA2R2_ENGINEERING_CANARY_CELL",
                "normalization": normalization,
                "clean_runtime": {key: value for key, value in clean.items() if key != "rows"},
                "clean_rows": clean["rows"],
                "action_pair_audit": context["pair_audits"],
                "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
                "runtime_counters": counters,
                "scientific_claim": "NONE_ENGINEERING_ONLY",
                "claim_boundary": "AA2R2 Phase-A action-semantics/clean-runtime qualification only; canary permanently excluded from AA2-AA5.",
                "next_legal_action": "STOP_FOR_PI_AFTER_PHASE_A",
            }
        )
        atomic_write(args.output, receipt)
        return receipt
    except Exception as exc:
        if not context["failure_persisted"]:
            _active_failure(
                context,
                status="AA2R2_ENGINEERING_HOLD_RUNTIME_ERROR",
                error_type=type(exc).__name__,
                message=str(exc),
                diagnostics={"action_pair_audit_count": len(context["pair_audits"])},
            )
        raise
    finally:
        if model is not None:
            del model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--aa0", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--model-family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AA2R2_ENGINEERING_HOLD_RUNTIME_ERROR", "model_family": args.model_family, "canonical_parent_key": args.canonical_parent_key, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
