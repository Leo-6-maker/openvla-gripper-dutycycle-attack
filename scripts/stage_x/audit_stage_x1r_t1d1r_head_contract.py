#!/usr/bin/env python3
"""CPU-only D1R-A head-contract and historical replay regression audit."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"
CONTRACT = REPO / "configs/STAGE_X_X1R_T1D1R_STUDENT_HEAD_CONTRACT_V1.json"
REPORT = REPO / "reports/STAGE_X_X1R_T1D1R_HEAD_CONTRACT_AUDIT_V1.json"
BASE_RUNNER = REPO / "scripts/stage_x/run_stage_x1r_t1d1_screening_clean.py"
CANARY_HOLD = REPO / "reports/STAGE_X_X1R_T1D1_CANARY_RUNTIME_HOLD_V1.json"
CONTINUATION = REPO / "reports/STAGE_X_X1R_T1D1R_CONTINUATION_LEDGER_V1.json"
REPLAY_REPORT = REPO / "reports/STAGE_X_X1R_T1D0R2_STUDENT_REPLAY_PARITY_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_receipt() -> dict[str, Any]:
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")}


def replay_one(d1: Any, model: Any, mean: np.ndarray, std: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
    replay_path = Path(str(row["replay_path"]))
    if not replay_path.is_file():
        raise RuntimeError(f"HISTORICAL_REPLAY_INPUT_MISSING:{replay_path}")
    payload = load_json(replay_path)
    if payload.get("schema") != "STAGE_X_M4_CLEAN_REPLAY_STUDENT_INPUTS_V1" or payload.get("intervention_executed") or payload.get("outcomes_read") or payload.get("v_phys_read"):
        raise RuntimeError(f"HISTORICAL_REPLAY_SCOPE_INVALID:{row['canonical_parent_key']}")
    steps = [{"step": int(item["step"]), "raw_action_7d": item["raw_action_7d"], "action_env_7d": item["env_action_7d"]} for item in payload["replay_rows"]]
    telemetry = [{"step": int(item["step"]), "robot0_gripper_qpos": item["robot0_gripper_qpos"], "robot0_eef_pos": item["robot0_eef_pos"]} for item in payload["replay_rows"]]
    sys.path.insert(0, str(REPO / "src"))
    from gripper_attack.v5_r3_features import materialize_fit670_features

    materialized = materialize_fit670_features({"steps": steps, "telemetry": telemetry})
    features = [item["features_25d"] for item in materialized]
    candidates = [bool(item["candidate_close"]) for item in materialized]
    predictions = d1.student_trace(model, features, mean, std)
    suite = str(row["canonical_parent_key"]).split("/", 1)[0]
    predicted = d1.schedule(predictions, candidates, int(d1.HORIZONS[suite]), 0.55, 0.8)["first_emit_step"]
    expected = row.get("sealed_summary_first_emit_step")
    if predicted != expected:
        raise RuntimeError(f"HISTORICAL_REPLAY_EMIT_MISMATCH:{row['canonical_parent_key']}:{predicted}!={expected}")
    return {"canonical_parent_key": row["canonical_parent_key"], "input_path": str(replay_path), "input_sha256": sha256_file(replay_path), "row_count": len(payload["replay_rows"]), "expected_first_emit_step": expected, "replayed_first_emit_step": predicted, "match": True, "scope": "CPU_ONLY_SUMMARY_REGRESSION_NOT_PARITY_PROMOTION"}


def main() -> int:
    protocol = load_json(PROTOCOL)
    contract = load_json(CONTRACT)
    canary_hold = load_json(CANARY_HOLD)
    continuation = load_json(CONTINUATION)
    replay_report = load_json(REPLAY_REPORT)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    expected_contract_sha = protocol["student"]["head_contract_sha256"]
    checks["head_contract_sha256"] = {"expected": expected_contract_sha, "actual": sha256_file(CONTRACT), "pass": sha256_file(CONTRACT) == expected_contract_sha}
    checks["runtime_output_keys"] = {"actual": contract.get("runtime_output_keys"), "pass": contract.get("runtime_output_keys") == ["physical_criticality", "k10_feasible", "safe_release", "instability", "gripper_closing_state"]}
    checks["historical_semantic_alias"] = {"actual": contract.get("historical_semantic_aliases"), "pass": contract.get("historical_semantic_aliases") == {"k10_feasibility": "k10_feasible"}}
    checks["d1_canary_hold_binding"] = {"status": canary_hold.get("status"), "ordinals": sorted(int(row["ordinal"]) for row in canary_hold.get("canaries", [])), "pass": canary_hold.get("status") == "HOLD_RUNTIME_INVALID_AFTER_FIRST_POLICY_DECISION" and sorted(int(row["ordinal"]) for row in canary_hold.get("canaries", [])) == [1, 11, 20, 30] and all(row.get("retry_eligible") is False for row in canary_hold.get("canaries", []))}
    continuation_rows = continuation.get("rows", [])
    checks["continuation_ledger"] = {"count": len(continuation_rows), "ordinals": [int(row["ordinal"]) for row in continuation_rows], "suite_counts": continuation.get("suite_counts"), "pass": len(continuation_rows) == 35 and not ({1, 11, 20, 30} & {int(row["ordinal"]) for row in continuation_rows}) and int(continuation.get("repair_canary_ordinal")) == 2 and continuation.get("replacement") is False and continuation.get("rerank") is False}
    if not all(item["pass"] for item in checks.values()):
        errors.append("STATIC_CONTRACT_OR_POPULATION_CHECK_FAILED")

    d1 = import_module(BASE_RUNNER, "stage_x_t1d1_base_for_head_audit")
    paths = d1.student_paths(protocol)
    model, mean, std, physical, closing = d1.load_student(protocol, paths)
    from gripper_attack.stage_x_x1r_student_head_contract import runtime_head_names

    actual_names = list(runtime_head_names(model))
    checks["tracked_model_head_names"] = {"actual": actual_names, "pass": actual_names == contract["runtime_output_keys"]}
    synthetic = [[float((step + column) % 7) / 7.0 for column in range(25)] for step in range(6)]
    predictions = d1.student_trace(model, synthetic, mean, std)
    checks["synthetic_student_trace"] = {"rows": len(predictions), "keys": sorted(predictions[0]) if predictions else [], "pass": len(predictions) == 6 and all(sorted(row) == sorted(contract["runtime_output_keys"]) for row in predictions)}
    schedule_source = inspect.getsource(d1.schedule)
    checks["scheduler_gate_scope"] = {"forbidden_head_references_absent": all(name not in schedule_source for name in ("k10_feasible", "k10_feasibility", "safe_release", "instability")), "required_inputs_present": all(name in schedule_source for name in ("candidate_close", "legal", "physical_criticality", "gripper_closing_state", "emitted"))}
    if not checks["tracked_model_head_names"]["pass"] or not checks["synthetic_student_trace"]["pass"] or not all(checks["scheduler_gate_scope"].values()):
        errors.append("EXECUTABLE_HEAD_CONTRACT_CHECK_FAILED")

    replay_results = []
    for row in replay_report["replays"].values():
        replay_results.append(replay_one(d1, model, mean, std, row))
    checks["historical_t1c_replay_summary_regression"] = {"count": len(replay_results), "results": replay_results, "pass": len(replay_results) == 4 and all(item["match"] for item in replay_results)}
    if not checks["historical_t1c_replay_summary_regression"]["pass"]:
        errors.append("HISTORICAL_T1C_REPLAY_REGRESSION_FAILED")

    status = "STAGE_X_X1R_T1D1R_HEAD_CONTRACT_PASS" if not errors else "STAGE_X_X1R_T1D1R_HOLD_HEAD_CONTRACT_REPAIR"
    report = {
        "schema": "STAGE_X_X1R_T1D1R_HEAD_CONTRACT_AUDIT_V1",
        "status": status,
        "source": source_receipt(),
        "reviewed_pr129": {"commit": "4b0ceb65f8f7babdd29163e032c56fed3ba57526", "tree": "d7b688e82bf0b9c5e91c08b3ad15c3a6d94b89ad"},
        "d1_canary_hold_report": {"path": str(CANARY_HOLD), "sha256": sha256_file(CANARY_HOLD), "status": canary_hold.get("status")},
        "student": {"checkpoint_sha256": protocol["student"]["checkpoint_sha256"], "source_sha256": protocol["student"]["source_raw_sha256"], "runtime_output_keys": actual_names, "historical_semantic_aliases": contract["historical_semantic_aliases"], "physical_threshold": physical, "closing_threshold": closing},
        "checks": checks,
        "errors": errors,
        "protected_boundary": {"pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        "prospective_rollout_authorized_by_this_report": False,
        "next_gate": "D1R_REPAIR_CANARY_ORDINAL_2_ONLY" if not errors else "STAGE_X_X1R_T1D1R_HOLD_HEAD_CONTRACT_REPAIR"
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "errors": errors, "report": str(REPORT)}, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
