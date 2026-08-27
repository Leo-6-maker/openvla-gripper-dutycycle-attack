#!/usr/bin/env python3
"""Run one AA2R2 Phase-B clean-only cell.

This is an append-only replacement for the historical AA2 runner.  It keeps
the frozen AA0 eligibility scan and 324-cell manifest, but validates every
model action with the AA2R2 official three-state/PI05 adapter.  It never
executes treatment, attack, endpoint, task-success, or protected reads.
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


# Reuse the frozen loader, LIBERO setup, telemetry, and AA0 scanner.  The old
# runner itself is never executed; only its read-only helpers are imported.
AA2 = load_module(ROOT / "scripts/stage_aa/run_stage_aa2_clean_screen.py", "aa2r2_phase_b_legacy_helpers")
AA1 = AA2.AA1


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


def safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return safe_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else ("NaN" if np.isnan(number) else ("Infinity" if number > 0 else "-Infinity"))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): safe_value(item) for key, item in value.items()}
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(safe_value(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def find_cell(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("cells", []) if row.get("cell_id") == cell_id]
    if len(rows) != 1:
        raise RuntimeError(f"AA2R2_CELL_ID_NOT_UNIQUE:{cell_id}")
    return rows[0]


def validate_static(
    protocol: dict[str, Any],
    source: dict[str, Any],
    manifest: dict[str, Any],
    cell: dict[str, Any],
    aa0: dict[str, Any],
    capacity: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if protocol.get("status") != "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE":
        raise RuntimeError("AA2R2_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("clean_only") is not True or protocol.get("open_intervention_allowed") is not False or protocol.get("attack_or_pgd_allowed") is not False:
        raise RuntimeError("AA2R2_CLEAN_ONLY_FIREWALL_INVALID")
    if source.get("status") != "STAGE_AA_AA2R2_PHASE_B_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("AA2R2_PHASE_B_SOURCE_NOT_FROZEN")
    if source.get("phase_b", {}).get("authorized") is not True or source.get("phase_b", {}).get("scientific_only_clean") is not True:
        raise RuntimeError("AA2R2_PHASE_B_AUTHORITY_INVALID")
    for name, binding in source.get("versioned_runtime_files", {}).items():
        path = ROOT / str(binding["path"])
        if not path.is_file() or path.stat().st_size != int(binding["bytes"]) or sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"AA2R2_PHASE_B_SOURCE_BINDING_MISMATCH:{name}")
    if manifest.get("status") != "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE" or manifest.get("cell_count") != 324 or len(manifest.get("cells", [])) != 324:
        raise RuntimeError("AA2R2_ORIGINAL_MANIFEST_INVALID")
    if source.get("original_manifest_sha256") != sha256_file(ROOT / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"):
        raise RuntimeError("AA2R2_ORIGINAL_MANIFEST_BINDING_INVALID")
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    if cell.get("model_family") not in MODELS or cell.get("suite") not in SUITES:
        raise RuntimeError("AA2R2_CELL_MODEL_OR_SUITE_INVALID")
    pool = set(capacity["analysis_pool_after_aa1_reservation"]["keys"])
    if cell.get("canonical_parent_key") not in pool:
        raise RuntimeError("AA2R2_PARENT_NOT_IN_FROZEN_POOL")
    canaries = {row["canonical_parent_key"] for row in capacity["aa1_engineering_canary_reservation"]["reserved_rows"]}
    if cell.get("canonical_parent_key") in canaries:
        raise RuntimeError("AA2R2_CANARY_EXPOSURE_FORBIDDEN")
    if cell.get("clean_only_authorization") != "AA2_CLEAN_ONLY_NO_OPEN_NO_ATTACK_NO_PROTECTED":
        raise RuntimeError("AA2R2_CELL_AUTHORIZATION_INVALID")
    old_runner_sha = source.get("historical_authorities", {}).get("aa2_runner", {}).get("sha256")
    if cell.get("eligibility_implementation_sha256") != old_runner_sha:
        raise RuntimeError("AA2R2_ORIGINAL_CELL_BINDING_CHANGED")
    if cell.get("seed") is None or config.get("environment", {}).get("dummy_wait_steps") != 10:
        raise RuntimeError("AA2R2_CELL_OR_ENVIRONMENT_BINDING_INVALID")


def _persist_failure(context: dict[str, Any], status: str, message: str, diagnostics: dict[str, Any] | None = None) -> None:
    receipt = context["receipt"]
    receipt.update(
        {
            "status": status,
            "error": {"type": "RuntimeError", "message": message, "diagnostics": safe_value(diagnostics)},
            "action_pair_audit_count": len(context["pair_audits"]),
            "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
            "runtime_counters": dict(context["counters"]),
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
            "next_legal_action": "STAGE_AA_AUTONOMOUS_AA2_ENGINEERING_HOLD_STOP_FOR_PI",
        }
    )
    atomic_write(context["output"], receipt)
    context["failure_persisted"] = True


def _record_action_failure(
    context: dict[str, Any],
    check: dict[str, Any],
    raw_action: np.ndarray,
    final_action: np.ndarray,
    *,
    step: int,
    boundary_step: int,
    queue_index: int,
    meta: dict[str, Any],
) -> None:
    record = {
        "cell_id": context["receipt"]["cell_id"],
        "model_family": context["family"],
        "canonical_parent_key": context["receipt"]["canonical_parent_key"],
        "seed": context["receipt"]["seed"],
        "step": step,
        "boundary_step": boundary_step,
        "queue_index": queue_index,
        "queue_or_replan_boundary": BOUNDARY[context["family"]],
        "reported_fresh_boundary": meta.get("fresh_boundary"),
        "raw_action_7d": safe_value(raw_action),
        "final_action_7d": safe_value(final_action),
        "raw_gripper": safe_value(raw_action[-1]),
        "final_gripper": safe_value(final_action[-1]),
        "expected_final_gripper": check.get("expected_final_gripper"),
        "expected_final_action": check.get("expected_final_action"),
        "validator_version": check.get("validator_version"),
        "rule": check.get("rule"),
        "reason": check.get("reason"),
        "semantic_state": check.get("semantic_state"),
    }
    record["first_offending_row_digest"] = canonical_hash(record)
    _persist_failure(
        context,
        "AA2R2_ENGINEERING_INVALID_ACTION_SEMANTICS",
        f"ACTION_SEMANTICS_INVALID:{context['family']}:{check.get('reason')}",
        {"offending_row": record},
    )


def model_pairs_v2(infer: Any, obs: dict[str, Any], language: str, family: str, counters: dict[str, int], context: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    chunk, meta = infer(obs, language)
    counters["model_inference_calls"] += 1
    meta = meta if isinstance(meta, dict) else {}
    raw = np.asarray(meta.get("raw_action_chunk"), dtype=np.float32)
    final = np.asarray(chunk, dtype=np.float32)
    boundary_step = int(context["current_step"])
    if raw.ndim != 2 or final.ndim != 2 or raw.shape != final.shape or raw.shape[1] != ACTION_DIM:
        _persist_failure(context, "AA2R2_ENGINEERING_INVALID_ACTION_SHAPE", f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{final.shape}", {"step": boundary_step, "raw_action_chunk": safe_value(raw), "final_action_chunk": safe_value(final)})
        raise RuntimeError(f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{final.shape}")
    if meta.get("fresh_boundary") != BOUNDARY[family]:
        _persist_failure(context, "AA2R2_ENGINEERING_INVALID_MODEL_BOUNDARY", f"MODEL_BOUNDARY_INVALID:{family}:{meta.get('fresh_boundary')}", {"step": boundary_step, "expected": BOUNDARY[family], "observed": meta.get("fresh_boundary")})
        raise RuntimeError(f"MODEL_BOUNDARY_INVALID:{family}:{meta.get('fresh_boundary')}")
    length = QUEUE_LENGTH[family]
    if raw.shape[0] < length:
        _persist_failure(context, "AA2R2_ENGINEERING_INVALID_ACTION_QUEUE", f"ACTION_CHUNK_TOO_SHORT:{family}:{raw.shape[0]}:{length}", {"step": boundary_step, "raw_rows": int(raw.shape[0]), "required_rows": length})
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
        audit = {
            "step": boundary_step + index,
            "boundary_step": boundary_step,
            "queue_index": index,
            "model_boundary": BOUNDARY[family],
            "reported_fresh_boundary": meta.get("fresh_boundary"),
            "semantics": check,
        }
        context["pair_audits"].append(audit)
        if not check.get("accepted"):
            _record_action_failure(context, check, raw_action, final_action, step=boundary_step + index, boundary_step=boundary_step, queue_index=index, meta=meta)
            raise RuntimeError(f"ACTION_SEMANTICS_INVALID:{family}:{check.get('reason')}")
        pairs.append((raw_action, final_action))
    return pairs


def capture_clean(config: dict[str, Any], cell: dict[str, Any], infer: Any, counters: dict[str, int], context: dict[str, Any]) -> dict[str, Any]:
    family = str(cell["model_family"])
    suite = str(cell["suite"])
    parent_key = str(cell["canonical_parent_key"])
    task_idx = int(cell["source_task_idx"])
    state_id = int(str(cell["state"]).split("_")[-1])
    env, _task_suite, task, obs, _initial_states = AA1.make_env(config, suite, task_idx, state_id, counters)
    try:
        binding = AA1.TAXONOMY.bind_object_taxonomy(env, AA1.bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {"status": "INELIGIBLE_CLEAN_OBJECT_BINDING", "reason_codes": ["OBJECT_BINDING_INVALID"], "binding": binding, "rows": [], "actions": []}
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        boundary_states: dict[int, np.ndarray] = {}
        done = False
        baseline_z: float | None = None
        horizon = HORIZONS[suite]
        for step in range(horizon):
            if done:
                break
            fresh = not queue
            if fresh:
                context["current_step"] = step
                boundary_states[step] = AA1.Z1.snapshot_state(env)
                queue = model_pairs_v2(infer, obs, language, family, counters, context)
            raw_action, final_action = queue.pop(0)
            current = AA1.telemetry(env, binding, target, counters)
            if baseline_z is None and AA2.finite_vector(current.get("object_position"), 3):
                baseline_z = float(current["object_position"][2])
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "terminal_before": done,
                "model_boundary": fresh,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                **current,
            }
            rows.append(row)
            actions.append({"boundary": fresh, "raw": raw_action.tolist(), "final": final_action.tolist()})
            obs, done = AA2.step_unpack(env.step(final_action.tolist()))
            counters["env_step_calls"] += 1
            row["terminal_after"] = done
        complete = len(rows) == horizon
        critical, critical_reasons = AA2.scan_candidates(rows, actions, family, parent_key, baseline_z, complete, "CRITICAL")
        noncritical, noncritical_reasons = AA2.scan_candidates(rows, actions, family, parent_key, baseline_z, complete, "NONCRITICAL")
        public = lambda values: [{key: value for key, value in item.items() if key != "evidence_rows"} for item in values]
        trajectory_digest = canonical_hash(
            {
                "rows": [AA2.compact_row(row) for row in rows],
                "actions": [{"boundary": row["boundary"], "raw": row["raw"], "final": row["final"]} for row in actions],
                "boundary_state_sha256": {str(step): sha256_bytes(state.tobytes()) for step, state in boundary_states.items()},
            }
        )
        return {
            "status": "PASS_AA2R2_CLEAN_TRAJECTORY_CAPTURED",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "steps_captured": len(rows),
            "complete_trajectory": complete,
            "baseline_z_m": baseline_z,
            "telemetry_valid_rows": sum(AA2.telemetry_valid(row) for row in rows),
            "clean_trajectory_digest": trajectory_digest,
            "boundary_count": len(boundary_states),
            "critical_candidates": public(critical),
            "noncritical_candidates": public(noncritical),
            "critical_reason_counts": critical_reasons,
            "noncritical_reason_counts": noncritical_reasons,
            "selected_critical": critical[0] if critical else None,
            "selected_noncritical": noncritical[0] if noncritical else None,
            "rows": rows,
            "actions": actions,
        }
    finally:
        env.close()


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    source = load_json(args.source_authority)
    manifest = load_json(args.launch_manifest)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    config = load_json(args.z1_config)
    cell = find_cell(manifest, args.cell_id)
    validate_static(protocol, source, manifest, cell, aa0, capacity, config)
    counters = {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "aa_v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "aa2_exposure": 0,
    }
    receipt: dict[str, Any] = {
        "schema": "STAGE_AA_AA2R2_PHASE_B_CLEAN_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": source["gate"],
        "phase": "B",
        "attempt_kind": args.attempt_kind,
        "recovery_of": args.recovery_of,
        "cell_id": args.cell_id,
        "model_family": cell["model_family"],
        "canonical_parent_key": cell["canonical_parent_key"],
        "suite": cell["suite"],
        "source_task_idx": cell["source_task_idx"],
        "state": cell["state"],
        "seed": cell["seed"],
        "clean_only": True,
        "scientific_parent_exposure": "AA2_CLEAN_ONLY_CENSUS_ALLOWED",
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(args.output, receipt)
    context: dict[str, Any] = {"receipt": receipt, "output": args.output, "counters": counters, "pair_audits": [], "family": str(cell["model_family"]), "current_step": 0, "failure_persisted": False}
    model = None
    try:
        AA1.require_single_gpu(args.gpu_id)
        receipt["gpu"] = AA1.gpu_snapshot(args.gpu_id)
        set_clean_seed(int(cell["seed"]))
        AA1.Z1.configure_libero(config)
        checkpoint = checkpoint_path(config, str(cell["model_family"]), str(cell["suite"]))
        if not checkpoint.exists():
            raise RuntimeError(f"CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
        checkpoint_manifest = None
        if cell["model_family"] == SEMANTICS.MODEL_M1:
            checkpoint_manifest = AA1.Z1.verify_m1_materialization(Path(args.m1_manifest), checkpoint, str(cell["suite"]), str(config["model_families"][SEMANTICS.MODEL_M1]["checkpoint_manifests_sha256"]))
        receipt.update({"checkpoint": str(checkpoint), "checkpoint_manifest": checkpoint_manifest})
        if cell["model_family"] == SEMANTICS.MODEL_M2:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if cell["model_family"] == SEMANTICS.MODEL_M0:
            infer, model, normalization = AA1.Z1.load_openvla(str(checkpoint), oft=False, suite=str(cell["suite"]), return_chunk=True)
        elif cell["model_family"] == SEMANTICS.MODEL_M1:
            infer, model, normalization = AA1.Z1.load_openvla(str(checkpoint), oft=True, suite=str(cell["suite"]), return_chunk=True)
        else:
            infer, model = AA1.Z1.load_pi05(str(checkpoint), return_chunk=True)
            normalization = {"checkpoint_mutated": False}
        counters["aa2_exposure"] = 1
        counters["scientific_parent_exposure"] = 1
        clean = capture_clean(config, cell, infer, counters, context)
        selected_critical = clean.get("selected_critical")
        selected_noncritical = clean.get("selected_noncritical")
        receipt.update(
            {
                "status": "AA2R2_PHASE_B_CLEAN_CELL_COMPLETE",
                "normalization": normalization,
                "clean": {key: value for key, value in clean.items() if key not in {"rows", "actions"}},
                "eligibility": {
                    "critical": selected_critical is not None,
                    "noncritical": selected_noncritical is not None,
                    "critical_anchor": selected_critical,
                    "noncritical_anchor": selected_noncritical,
                    "noncritical_affects_primary_denominator": False,
                },
                "action_pair_audit": context["pair_audits"],
                "action_pair_audit_count": len(context["pair_audits"]),
                "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
                "runtime_counters": counters,
                "scientific_claim": "NONE_AA2R2_CLEAN_DENOMINATOR_ONLY",
                "claim_boundary": "AA2R2 Phase-B clean-only census under the frozen AA0 eligibility contract; no treatment, endpoint, or promotion claim.",
                "next_legal_action": "STOP_FOR_PI_AFTER_FULL_CENSUS",
            }
        )
        atomic_write(args.output, receipt)
        return receipt
    except Exception as exc:
        if not context["failure_persisted"]:
            _persist_failure(context, "AA2R2_ENGINEERING_HOLD_RUNTIME_ERROR", f"{type(exc).__name__}:{exc}")
        raise
    finally:
        if model is not None:
            del model


def self_test() -> None:
    close = SEMANTICS.validate_action_pair(SEMANTICS.MODEL_M0, [0.0] * 6 + [0.4999999], [0.0] * 6 + [1.0])
    neutral = SEMANTICS.validate_action_pair(SEMANTICS.MODEL_M1, [0.0] * 6 + [0.5], [0.0] * 6 + [0.0])
    open_value = SEMANTICS.validate_action_pair(SEMANTICS.MODEL_M1, [0.0] * 6 + [0.5000001], [0.0] * 6 + [-1.0])
    assert close["accepted"] and neutral["accepted"] and open_value["accepted"]
    assert SEMANTICS.validate_action_pair(SEMANTICS.MODEL_M0, [0.0] * 6 + [0.5], [0.0] * 6 + [1.0])["accepted"] is False
    assert SEMANTICS.validate_action_pair(SEMANTICS.MODEL_M2, [2.0] * 6 + [-2.0], [1.0] * 6 + [-1.0])["accepted"]
    print(json.dumps({"status": "AA2R2_PHASE_B_STATIC_MOCK_PASS"}, sort_keys=True))


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--aa0", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-kind", choices=("NORMAL", "RECOVERY"), default="NORMAL")
    parser.add_argument("--recovery-of", default=None)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "cell_id": args.cell_id, "attempt_kind": args.attempt_kind}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AA2R2_ENGINEERING_HOLD_RUNTIME_ERROR", "cell_id": args.cell_id, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
