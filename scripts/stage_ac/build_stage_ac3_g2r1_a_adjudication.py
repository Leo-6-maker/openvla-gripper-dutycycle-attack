#!/usr/bin/env python3
"""Build the read-only G2R1-A censoring adjudication from sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
CONDITIONS = ("CLEAN_REFERENCE", "OPEN_T3", "OPEN_T5", "OPEN_T10")
OPEN_DOSE = {"OPEN_T3": 3, "OPEN_T5": 5, "OPEN_T10": 10}
INVALID = "ENGINEERING_INVALID_OR_HORIZON_CENSORED"
G2_STATUS = "STAGE_AC_AC3_G2_ENGINEERING_OR_HORIZON_HOLD_STOP_FOR_PI"
HORIZON_RE = re.compile(r"^AC3_G2_TERMINAL_BEFORE_HORIZON:(AC3-[^:]+):(\d+)$")
SEMANTICS_ERROR = "ACTION_SEMANTICS_INVALID:M1_OPENVLA_OFT:RAW_GRIPPER_AT_THRESHOLD"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical(value))


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"AC3_G2R1_A_APPEND_ONLY_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compare_remote_record(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("path", "bytes", "sha256"):
        require(str(actual.get(key)) == str(expected.get(key)), f"AC3_G2R1_A_REMOTE_RECORD:{label}:{key}")


def local_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def verify_record(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    require(path.is_file(), f"AC3_G2R1_A_MISSING:{label}:{path}")
    actual = local_record(path)
    require(actual["bytes"] == int(expected["bytes"]), f"AC3_G2R1_A_BYTES:{label}")
    require(actual["sha256"] == str(expected["sha256"]), f"AC3_G2R1_A_SHA:{label}")
    return actual


def classify(branch: dict[str, Any], job: dict[str, Any], indexed: dict[str, Any]) -> dict[str, Any]:
    branch_id = str(branch["branch_id"])
    error = str(branch["selected_error"])
    require(indexed.get("error", {}).get("message") == error, f"AC3_G2R1_A_ERROR_MISMATCH:{branch_id}")
    require(branch.get("model_family") == job.get("model_family"), f"AC3_G2R1_A_MODEL:{branch_id}")
    require(branch.get("suite") == job.get("suite"), f"AC3_G2R1_A_SUITE:{branch_id}")
    require(branch.get("canonical_parent_key") == job.get("canonical_parent_key"), f"AC3_G2R1_A_PARENT:{branch_id}")
    require(branch.get("condition") == job.get("condition"), f"AC3_G2R1_A_CONDITION:{branch_id}")
    require(int(branch.get("dose")) == int(job.get("dose")), f"AC3_G2R1_A_DOSE:{branch_id}")

    base = branch.get("base") or {}
    sidecar = branch.get("sidecar")
    if sidecar is not None:
        require(indexed.get("authority_source") == "failure_sidecar", f"AC3_G2R1_A_AUTHORITY:{branch_id}")
        compare_remote_record(base, indexed["history"]["base_receipt"], f"{branch_id}:base")
        compare_remote_record(sidecar, indexed["history"]["failure_sidecar"], f"{branch_id}:sidecar")
        require(sidecar.get("error") == error, f"AC3_G2R1_A_SIDECAR_ERROR:{branch_id}")
        require(sidecar.get("status") == INVALID and sidecar.get("next_legal_action") == "STOP_FOR_PI", f"AC3_G2R1_A_SIDECAR_STATUS:{branch_id}")
    else:
        require(indexed.get("authority_source") == "base_receipt", f"AC3_G2R1_A_BASE_AUTHORITY:{branch_id}")
        compare_remote_record(base, indexed["history"]["base_receipt"], f"{branch_id}:base")
        require(base.get("error") == error and base.get("status") == INVALID, f"AC3_G2R1_A_BASE_ERROR:{branch_id}")

    horizon_match = HORIZON_RE.match(error)
    if horizon_match:
        require(horizon_match.group(1) == branch_id, f"AC3_G2R1_A_ERROR_BRANCH:{branch_id}")
        terminal_step = int(horizon_match.group(2))
        required = int(branch["dose"]) + 10
        rows_len = int(base.get("rows_len", 0))
        if rows_len:
            require(base.get("last_terminal_after") is True, f"AC3_G2R1_A_TERMINAL_MARKER:{branch_id}")
            require(int(base.get("last_step")) + 1 == terminal_step, f"AC3_G2R1_A_TERMINAL_STEP:{branch_id}")
            require(int(base.get("available_horizon_steps")) == rows_len, f"AC3_G2R1_A_HORIZON_COUNT:{branch_id}")
        return {
            "classification": "TRUE_SIMULATOR_TERMINAL_HORIZON_CENSOR",
            "terminal_step": terminal_step,
            "required_local_steps": required,
            "available_local_steps": rows_len,
            "direct_terminal_marker_persisted": bool(rows_len),
            "recovery_authorized": False,
            "physical_outcome_read": False,
        }

    if error == SEMANTICS_ERROR:
        require(int(base.get("rows_len", 0)) == 0, f"AC3_G2R1_A_SEMANTICS_ROWS:{branch_id}")
        return {
            "classification": "ACTION_SEMANTICS_VALIDATOR_FAILURE_UNRESOLVED",
            "semantic_mismatch": True,
            "recovery_authorized": False,
            "physical_outcome_read": False,
        }
    raise RuntimeError(f"AC3_G2R1_A_UNCLASSIFIED_ERROR:{branch_id}:{error}")


def model_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        key = (str(row["model_family"]), str(row["canonical_parent_key"]))
        grouped[key][str(row["condition"])] = str(row["status"])
    result: dict[str, Any] = {}
    for model in MODELS:
        parent_rows = {key: value for key, value in grouped.items() if key[0] == model}
        pairs = sum(value.get("OPEN_T3") == "PASS" and value.get("OPEN_T10") == "PASS" for value in parent_rows.values())
        triplets = sum(all(value.get(condition) == "PASS" for condition in OPEN_DOSE) for value in parent_rows.values())
        unknown_by_dose = Counter()
        for value in parent_rows.values():
            for condition, dose in OPEN_DOSE.items():
                if value.get(condition) != "PASS":
                    unknown_by_dose[str(dose)] += 1
        result[model] = {
            "parent_count": len(parent_rows),
            "complete_t3_t10_pairs": pairs,
            "complete_t3_t5_t10_triplets": triplets,
            "unknown_open_parents_by_dose": dict(sorted(unknown_by_dose.items())),
        }
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    index, index_record = read_json(args.g2_index)
    terminal, terminal_record = read_json(args.g2_terminal)
    root, root_record = read_json(args.g2_root)
    manifest, manifest_record = read_json(args.manifest)
    snapshot, snapshot_record = read_json(args.snapshot)

    require(root.get("status") == G2_STATUS, "AC3_G2R1_A_G2_ROOT_STATUS")
    require(terminal.get("status") == G2_STATUS, "AC3_G2R1_A_G2_TERMINAL_STATUS")
    require(index.get("status") == "HOLD_AC3_G2_ENGINEERING_OR_HORIZON", "AC3_G2R1_A_G2_INDEX_STATUS")
    require(canonical_hash(root.get("root_payload")) == root.get("root_payload_sha256"), "AC3_G2R1_A_G2_ROOT_PAYLOAD_SHA")
    verify_record(args.g2_index, root["artifacts"]["receipt_index"], "g2_index")
    verify_record(args.g2_terminal, root["artifacts"]["terminal"], "g2_terminal")
    verify_record(args.manifest, root["root_payload"]["manifest"], "g0_manifest")
    require(index.get("counts", {}).get("manifest_branches") == 384, "AC3_G2R1_A_MANIFEST_COUNT")
    require(index.get("counts", {}).get("invalid_or_horizon_censored_branches") == 12, "AC3_G2R1_A_INVALID_COUNT")
    require(snapshot.get("schema") == "STAGE_AC_AC3_G2R1_A_REMOTE_STRUCTURAL_EVIDENCE_V1", "AC3_G2R1_A_SNAPSHOT_SCHEMA")
    require(snapshot.get("firewall", {}).get("physical_outcome_fields_read") is False, "AC3_G2R1_A_OUTCOME_FIREWALL")

    jobs = {str(job["branch_id"]): job for job in manifest["branches"]}
    invalid = {str(row["branch_id"]): row for row in index["invalid_or_horizon_censored"]}
    branches = {str(row["branch_id"]): row for row in snapshot["branches"]}
    require(set(invalid) == set(branches) and len(branches) == 12, "AC3_G2R1_A_BRANCH_SET")

    classified: list[dict[str, Any]] = []
    class_counts = Counter()
    model_counts = Counter()
    for branch_id in sorted(branches):
        require(branch_id in jobs, f"AC3_G2R1_A_BRANCH_NOT_IN_MANIFEST:{branch_id}")
        detail = classify(branches[branch_id], jobs[branch_id], invalid[branch_id])
        branch = branches[branch_id]
        row = {
            "branch_id": branch_id,
            "model_family": branch["model_family"],
            "suite": branch["suite"],
            "canonical_parent_key": branch["canonical_parent_key"],
            "condition": branch["condition"],
            "dose": int(branch["dose"]),
            "branch_seed": jobs[branch_id]["branch_seed"],
            "anchor_step": jobs[branch_id]["selected_anchor"]["step"],
            "anchor_state_sha256": jobs[branch_id]["selected_anchor"]["boundary_state_sha256"],
            "selection_rank_sha256": jobs[branch_id]["selected_anchor"]["selection_rank_sha256"],
            "source_receipt": jobs[branch_id]["source_receipt"],
            "error": branch["selected_error"],
            "detail": detail,
            "base_receipt": {key: value for key, value in branch["base"].items() if key in ("path", "bytes", "sha256", "status", "rows_len", "available_horizon_steps", "last_terminal_after", "last_step", "action_receipts_len", "env_step_calls", "model_inference_calls", "open_intervention_steps")},
            "sidecar_receipt": ({key: value for key, value in branch["sidecar"].items() if key in ("path", "bytes", "sha256", "status", "error", "next_legal_action")} if branch.get("sidecar") else None),
        }
        classified.append(row)
        class_counts[detail["classification"]] += 1
        model_counts[branch["model_family"]] += 1

    coverage = model_coverage(index["rows"])
    semantic_hold = class_counts.get("ACTION_SEMANTICS_VALIDATOR_FAILURE_UNRESOLVED", 0) > 0
    status = "STAGE_AC_AC3_G2R1_A_ACTION_SEMANTICS_DISCREPANCY_HOLD_STOP_FOR_PI" if semantic_hold else "STAGE_AC_AC3_G2R1_A_CENSORING_ADJUDICATION_PASS_CONTINUE"
    next_action = "STOP_FOR_PI" if semantic_hold else "FREEZE_G2R1_C_CENSORING_AWARE_ANALYSIS"
    adjudication = {
        "schema": "STAGE_AC_AC3_G2R1_A_CENSOR_ADJUDICATION_V1",
        "status": status,
        "gate": "STAGE_AC_AC3_G2R1_FAILURE_PATH_CENSORING_ADJUDICATION_V1",
        "claim_boundary": "G2R1-A structural censoring adjudication only; no physical outcome interpretation, no G3 statistics",
        "source_authority": {
            "g2_root": root_record,
            "g2_terminal": terminal_record,
            "g2_index": index_record,
            "g0_manifest": manifest_record,
            "remote_structural_snapshot": snapshot_record,
        },
        "outcome_firewall": snapshot["firewall"],
        "counts": {
            "frozen_invalid_branches": len(classified),
            "true_simulator_terminal_horizon_censor": class_counts.get("TRUE_SIMULATOR_TERMINAL_HORIZON_CENSOR", 0),
            "action_semantics_validator_failure_unresolved": class_counts.get("ACTION_SEMANTICS_VALIDATOR_FAILURE_UNRESOLVED", 0),
            "unclassified": class_counts.get("UNCLASSIFIED", 0),
        },
        "model_invalid_counts": dict(sorted(model_counts.items())),
        "model_coverage": coverage,
        "censoring_amendment": {
            "activated": not semantic_hold,
            "unknown_is_not_zero": True,
            "partial_identification_bounds": "p_lower=observed_events/32; p_upper=(observed_events+censored_dose)/32",
            "paired_t3_t10_requires_at_least_24_complete_parents": True,
            "triplet_t3_t5_t10_requires_at_least_24_complete_parents": True,
        },
        "branches": classified,
        "next_legal_action": next_action,
    }
    artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_A_CENSOR_ADJUDICATION_V1.json", adjudication)
    payload = {
        "gate": adjudication["gate"],
        "status": status,
        "g2_root": root_record,
        "g2_terminal": terminal_record,
        "g2_index": index_record,
        "g0_manifest": manifest_record,
        "remote_structural_snapshot": snapshot_record,
        "adjudication": artifact,
        "counts": adjudication["counts"],
        "model_coverage": coverage,
        "outcome_firewall": snapshot["firewall"],
        "historical_evidence_preserved": True,
    }
    root_out = {
        "schema": "STAGE_AC_AC3_G2R1_A_ROOT_SEAL_V1",
        "status": status,
        "root_payload": payload,
        "root_payload_sha256": canonical_hash(payload),
        "artifacts": {"adjudication": artifact},
        "claim_boundary": adjudication["claim_boundary"],
        "next_legal_action": next_action,
    }
    root_artifact = write_new(args.output_dir / "STAGE_AC_AC3_G2R1_A_ROOT_SEAL_V1.json", root_out)
    return {"status": status, "counts": adjudication["counts"], "model_coverage": coverage, "artifacts": {"adjudication": artifact, "root": root_artifact}, "root_payload_sha256": root_out["root_payload_sha256"]}


def self_test() -> None:
    assert HORIZON_RE.match("AC3_G2_TERMINAL_BEFORE_HORIZON:AC3-demo:145")
    assert not HORIZON_RE.match(SEMANTICS_ERROR)
    assert OPEN_DOSE["OPEN_T10"] == 10
    print(json.dumps({"status": "AC3_G2R1_A_STATIC_SELF_TEST_PASS", "outcome_firewall": True}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--g2-index", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json")
    parser.add_argument("--g2-terminal", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_TERMINAL_V1.json")
    parser.add_argument("--g2-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2_ROOT_SEAL_V1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_A_REMOTE_STRUCTURAL_EVIDENCE_V1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
