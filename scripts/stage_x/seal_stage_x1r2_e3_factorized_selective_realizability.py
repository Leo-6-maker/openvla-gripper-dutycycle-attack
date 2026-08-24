#!/usr/bin/env python3
"""Aggregate immutable E3 parent receipts and write the final root seal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.name.startswith(".d_probe")):
        if path.name in {"E3_ROOT_SEAL_V1.json", "E3_ROOT_SEAL_V1.sha256"}:
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return rows


def parent_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    fixture_root = root / str(row["suite"]) / str(row["fixture_id"])
    paths = sorted(path for path in fixture_root.rglob("parent_receipt.json") if path.is_file())
    if not paths:
        return {"suite": row["suite"], "fixture_id": row["fixture_id"], "canonical_parent_key": row["canonical_parent_key"], "status": "HOLD_E3_PARENT_RECEIPT_MISSING", "probe_available": False, "true_invocation_reached": False, "candidate_evidence_complete": False, "strict_valid_candidate": False}
    # Prefer the deepest attempt receipt; the legacy root receipt is only a
    # pointer/history record and must not override a completed repair attempt.
    path = max(paths, key=lambda item: (len(item.relative_to(fixture_root).parts), item.as_posix()))
    receipt = load_json(path)
    clean = receipt.get("clean_probe") or {}
    true = receipt.get("true_receipt") or {}
    probe = clean.get("probe")
    evidence_complete = bool(probe is None or true.get("candidate_audit_complete") is True)
    return {"suite": row["suite"], "fixture_id": row["fixture_id"], "canonical_parent_key": row["canonical_parent_key"], "status": receipt.get("status"), "clean_runtime_status": clean.get("status"), "probe_available": probe is not None, "probe_step": None if probe is None else int(probe["step"]), "true_invocation_reached": int((true.get("counters") or {}).get("true_invocation_reached", 0)) == 1, "candidate_evidence_complete": evidence_complete, "candidate_count": int(true.get("observed_candidate_count", len(true.get("candidate_audit") or []))), "strict_valid_candidate": receipt.get("status") == "PASS_E3_VALID_CANDIDATE", "selected_candidate_index": true.get("selected_candidate_index"), "selected_candidate_source": true.get("selected_candidate_source"), "counters": receipt.get("counters", {}), "protected_boundary": receipt.get("protected_boundary", {}), "receipt_path": str(path), "attempt_history": [{"path": str(item), "status": load_json(item).get("status")} for item in paths]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    root = Path(str(protocol["runtime"]["durable_output_root"]))
    population = load_json(ROOT / str(protocol["population"]["path"]))
    rows = [parent_row(root, row) for row in population["selected"]]
    probe_rows = [row for row in rows if row["probe_available"]]
    valid_rows = [row for row in rows if row["strict_valid_candidate"]]
    incomplete = [row for row in rows if row["probe_available"] and not row["candidate_evidence_complete"]]
    runtime_holds = [row for row in rows if str(row.get("status", "")).startswith("HOLD_") or row.get("status") == "HOLD_E3_PARENT_RECEIPT_MISSING"]
    if incomplete or runtime_holds:
        decision = "HOLD_E3_EXECUTABLE_EVIDENCE_INSUFFICIENT"
    elif len({row["suite"] for row in valid_rows}) == 4:
        decision = "E3_STRICT_SELECTIVE_REALIZABILITY_FOUR_SUITE_EXISTENCE_ESTABLISHED"
    elif valid_rows:
        decision = "E3_STRICT_SELECTIVE_REALIZABILITY_PARTIAL_SUITE_DEPENDENT"
    else:
        decision = "E3_STRICT_SELECTIVE_REALIZABILITY_NOT_ESTABLISHED_UNDER_FROZEN_METHOD"
    decision_report = {
        "schema": "STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_DECISION_V1",
        "status": decision,
        "claim_boundary": "Model-side timing-decoupled selective realizability only; not physical efficacy, timing utility, task outcome, or protected evaluation.",
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "fixed_denominator": len(rows),
        "counts": {"clean_runtime_valid": sum(row["clean_runtime_status"] == "PASS_E3_CLEAN_RUNTIME" for row in rows), "probe_available": len(probe_rows), "true_invocation_reached": sum(row["true_invocation_reached"] for row in rows), "six_candidate_evidence_complete": sum(row["candidate_evidence_complete"] for row in probe_rows), "strict_valid_candidate": len(valid_rows)},
        "by_suite": {suite: {"parents": sum(row["suite"] == suite for row in rows), "probes": sum(row["suite"] == suite and row["probe_available"] for row in rows), "valid": sum(row["suite"] == suite and row["strict_valid_candidate"] for row in rows)} for suite in SUITES},
        "parent_rows": rows,
        "incomplete_evidence_rows": incomplete,
        "runtime_hold_rows": runtime_holds,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "protected_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0},
        "mandatory_stop": "OWNER_PI_REVIEW_REQUIRED; no R0/R1/R2, physical follow-up, RAND, SHUFFLED, tuning, or protected reads",
    }
    decision_path = root / "E3_DECISION_TABLE_V1.json"
    write_json(decision_path, decision_report)
    entries = manifest(root)
    seal = {"schema": "STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_ROOT_SEAL_V1", "status": "SEALED_" + decision, "protocol_sha256": sha256_file(args.protocol), "decision_path": decision_path.name, "decision_sha256": sha256_file(decision_path), "artifact_count": len(entries), "artifact_manifest": entries, "protected_boundary": decision_report["protected_boundary"]}
    seal_path = root / "E3_ROOT_SEAL_V1.json"
    write_json(seal_path, seal)
    (root / "E3_ROOT_SEAL_V1.sha256").write_text(f"{sha256_file(seal_path)}  E3_ROOT_SEAL_V1.json\n", encoding="utf-8")
    print(json.dumps({"status": decision, "root": str(root), "parents": len(rows), "probes": len(probe_rows), "valid": len(valid_rows), "artifact_count": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
