"""Independently audit the Q2 clean A/B qualification and emit Manifest A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SALT = "STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807"
VALID_STATUSES = {"PASS", "DONE", "QUALIFIED", "TASK_FAILURE"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("Q2 row is not an object")
            output.append(dict(value))
    return output


def _ranked(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for raw in rows:
        row = normalize_parent(raw)
        key = str(row["canonical_parent_key"])
        row["qualification_rank_sha256"] = hashlib.sha256(f"{SALT}::{key}".encode()).hexdigest()
        ranked.append(row)
    return sorted(ranked, key=lambda item: (str(item["qualification_rank_sha256"]), str(item["canonical_parent_key"])))


def _result(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "CONTROL_RESULT.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else None


def engineering_valid(row: Mapping[str, Any], result: Mapping[str, Any], process_exit_code: Any,
                      source_commit: str, source_tree: str) -> tuple[bool, list[str]]:
    """Independent copy of Q2's hard engineering contract."""
    errors: list[str] = []
    if process_exit_code != 0 or result.get("exit_code") != 0:
        errors.append("PROCESS_EXIT_NONZERO")
    if result.get("status") not in VALID_STATUSES:
        errors.append("RESULT_STATUS_INVALID")
    for field in ("snapshot_restore_valid", "task_identity_valid", "runtime_valid", "metrics_finite", "artifact_validation_pass"):
        if result.get(field) is not True:
            errors.append(f"{field.upper()}_FALSE")
    if result.get("old_artifacts_reused") is not False:
        errors.append("OLD_ARTIFACT_REUSE")
    if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
        errors.append("SOURCE_PROVENANCE_MISMATCH")
    if result.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("PARENT_IDENTITY_MISMATCH")
    if not result.get("key_state_identity_sha256"):
        errors.append("INITIAL_STATE_IDENTITY_MISSING")
    for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts"):
        if result.get(field, 0) != 0:
            errors.append(f"BOUNDARY_VIOLATION:{field}")
    return not errors, sorted(set(errors))


def _pair(row: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], valid: Mapping[str, bool]) -> tuple[bool, str, list[str]]:
    errors = []
    if not valid.get("A"):
        errors.append("A_ENGINEERING_INVALID")
    if not valid.get("B"):
        errors.append("B_ENGINEERING_INVALID")
    if errors:
        return False, "ENGINEERING_INVALID", errors
    for replicate in ("A", "B"):
        if results[replicate].get("canonical_parent_key") != row.get("canonical_parent_key"):
            errors.append(f"{replicate}_PARENT_IDENTITY_MISMATCH")
    a_identity = results["A"].get("key_state_identity_sha256")
    b_identity = results["B"].get("key_state_identity_sha256")
    if not a_identity or not b_identity or a_identity != b_identity:
        errors.append("AB_INITIAL_STATE_IDENTITY_MISMATCH")
    if errors:
        return False, "CLEAN_REPEATABILITY_FAIL_IDENTITY", sorted(set(errors))
    a_success = results["A"].get("clean_success") is True
    b_success = results["B"].get("clean_success") is True
    if a_success and b_success:
        return True, "QUALIFIED", []
    if a_success and not b_success:
        return False, "CLEAN_REPEATABILITY_FAIL_A_SUCCESS_B_FAIL", ["B_CLEAN_SUCCESS_FALSE"]
    if not a_success and b_success:
        return False, "CLEAN_REPEATABILITY_FAIL_A_FAIL_B_SUCCESS", ["A_CLEAN_SUCCESS_FALSE"]
    return False, "CLEAN_REPEATABILITY_FAIL_BOTH_FAIL", ["A_CLEAN_SUCCESS_FALSE", "B_CLEAN_SUCCESS_FALSE"]


def _expected_prefixes(universe: list[dict[str, Any]], rows: list[dict[str, Any]], target: int = 10) -> tuple[dict[str, list[str]], list[str]]:
    by_suite = {suite: [row for row in _ranked(universe) if row["suite"] == suite] for suite in SUITES}
    actual = {suite: [str(row["canonical_parent_key"]) for row in rows if row.get("suite") == suite] for suite in SUITES}
    errors: list[str] = []
    expected: dict[str, list[str]] = {}
    for suite in SUITES:
        qualified_first = sum(bool(row.get("qualified")) for row in rows if row.get("suite") == suite and str(row.get("canonical_parent_key")) in {str(item["canonical_parent_key"]) for item in by_suite[suite][:20]})
        expected_count = 20 if qualified_first >= target else min(30, len(by_suite[suite]))
        expected[suite] = [str(item["canonical_parent_key"]) for item in by_suite[suite][:expected_count]]
        if actual[suite] != expected[suite][:len(actual[suite])]:
            errors.append(f"NON_PREFIX_EVALUATION:{suite}")
        if len(actual[suite]) not in {0, expected_count} and len(actual[suite]) < expected_count:
            errors.append(f"INCOMPLETE_SUITE_EVALUATION:{suite}")
    return expected, errors


def audit(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _json(args.protocol)
    report = _json(args.report)
    rows = _rows(args.rows)
    universe = _json(args.candidate_universe)
    errors: list[str] = []
    if protocol.get("schema") != "STAGE_Q2_PROTOCOL_V1" or protocol.get("status") != "FROZEN":
        errors.append("PROTOCOL_NOT_FROZEN")
    if protocol.get("salt") != SALT:
        errors.append("SALT_MISMATCH")
    if protocol.get("candidate_universe_sha256") != sha256_file(args.candidate_universe):
        errors.append("CANDIDATE_UNIVERSE_SHA256_MISMATCH")
    if report.get("protocol_sha256") != sha256_file(args.protocol):
        errors.append("REPORT_PROTOCOL_SHA256_MISMATCH")
    if report.get("candidate_universe_sha256") != sha256_file(args.candidate_universe):
        errors.append("REPORT_CANDIDATE_SHA256_MISMATCH")
    if report.get("source_commit") != args.source_commit or report.get("source_tree") != args.source_tree:
        errors.append("REPORT_SOURCE_MISMATCH")
    candidates = universe.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != int(universe.get("candidate_count", -1)):
        errors.append("CANDIDATE_UNIVERSE_SCHEMA_INVALID")
        candidates = []
    universe_rows = [normalize_parent(row) for row in candidates if isinstance(row, Mapping)]
    universe_keys = {str(row["canonical_parent_key"]) for row in universe_rows}
    seen: set[str] = set()
    valid_count = 0
    qualified: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    classifications: dict[str, int] = {}
    terminal_equal = 0
    horizon_equal = 0
    engineering_invalid = 0
    for row in rows:
        key = str(row.get("canonical_parent_key"))
        if key in seen:
            errors.append(f"DUPLICATE_ROW:{key}")
        seen.add(key)
        candidate = next((item for item in universe_rows if str(item["canonical_parent_key"]) == key), None)
        if candidate is None:
            errors.append(f"ROW_NOT_IN_UNIVERSE:{key}")
            continue
        if row.get("qualification_rank_sha256") != hashlib.sha256(f"{SALT}::{key}".encode()).hexdigest():
            errors.append(f"RANK_MISMATCH:{key}")
        actual_results: dict[str, dict[str, Any]] = {}
        valid: dict[str, bool] = {}
        for replicate in ("A", "B"):
            attempts = (row.get("replicate_attempts") or {}).get(replicate) or []
            if not isinstance(attempts, list) or len(attempts) < 1 or len(attempts) > 2:
                errors.append(f"INVALID_RETRY_COUNT:{key}:{replicate}")
            elif len(attempts) == 2:
                first_dir = Path(str(attempts[0].get("output_dir", "")))
                if not first_dir.is_dir():
                    errors.append(f"MISSING_RETRY_ATTEMPT:{key}:{replicate}")
            output_value = (row.get("replicate_output_dirs") or {}).get(replicate)
            output_dir = Path(str(output_value)) if output_value else None
            if output_dir is None or not output_dir.is_dir():
                errors.append(f"MISSING_OUTPUT:{key}:{replicate}")
                actual_results[replicate] = {"status": "FAIL", "exit_code": 1}
                valid[replicate] = False
                continue
            try:
                output_dir.resolve().relative_to(args.output_dir.resolve() / "qualification")
            except ValueError:
                errors.append(f"OUTPUT_OUTSIDE_Q2_ROOT:{key}:{replicate}")
            actual = _result(output_dir)
            if actual is None:
                errors.append(f"MISSING_CONTROL_RESULT:{key}:{replicate}")
                actual = {"status": "FAIL", "exit_code": 1}
                valid[replicate] = False
            else:
                stored = (row.get("replicates") or {}).get(replicate) or {}
                process_exit = stored.get("process_exit_code")
                valid[replicate], hard_errors = engineering_valid(candidate, actual, process_exit, args.source_commit, args.source_tree)
                errors.extend(f"{key}:{replicate}:{item}" for item in hard_errors)
            actual_results[replicate] = actual
            engineering_invalid += int(not valid[replicate])
        if valid["A"] and valid["B"]:
            valid_count += 1
        pair_ok, classification, pair_errors = _pair(candidate, actual_results, valid)
        classifications[classification] = classifications.get(classification, 0) + 1
        errors.extend(f"{key}:{item}" for item in pair_errors if classification == "ENGINEERING_INVALID" or item != "")
        row_qualified = row.get("qualified") is True
        if row_qualified != pair_ok:
            errors.append(f"ROW_DECISION_MISMATCH:{key}")
        if pair_ok:
            qualified[str(candidate["suite"])].append(dict(candidate))
        a_hash = actual_results["A"].get("terminal_state_sha256")
        b_hash = actual_results["B"].get("terminal_state_sha256")
        terminal_equal += int(bool(a_hash and b_hash and a_hash == b_hash))
        horizon_equal += int(actual_results["A"].get("remaining_horizon_complete") == actual_results["B"].get("remaining_horizon_complete"))
    if seen - universe_keys:
        errors.append("ROWS_CONTAIN_UNKNOWN_KEYS")
    if len(rows) != len(seen):
        errors.append("ROW_COUNT_NOT_UNIQUE")
    _, prefix_errors = _expected_prefixes(universe_rows, rows)
    errors.extend(prefix_errors)
    selected_by_suite = {suite: qualified[suite][:10] for suite in SUITES}
    if any(len(selected_by_suite[suite]) < 10 for suite in SUITES):
        errors.append("QUOTA_UNDERFILLED")
    selected_keys = [str(row["canonical_parent_key"]) for suite in SUITES for row in selected_by_suite[suite]]
    if len(selected_keys) != len(set(selected_keys)):
        errors.append("SELECTED_DUPLICATE_PARENT_KEYS")
    boundaries = {key: report.get(key, 0) for key in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts")}
    if any(value != 0 for value in boundaries.values()):
        errors.append("REPORT_BOUNDARY_NONZERO")
    verdict = "PASS" if report.get("status") == "PASS" and not errors and all(len(selected_by_suite[suite]) == 10 for suite in SUITES) and engineering_invalid == 0 else "FAIL"
    audit_payload = {
        "schema": "STAGE_Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT_V1", "verdict": verdict,
        "source_commit": args.source_commit, "source_tree": args.source_tree,
        "protocol_sha256": sha256_file(args.protocol), "candidate_universe_sha256": sha256_file(args.candidate_universe),
        "evaluated_rows": len(rows), "evaluated_by_suite": {suite: sum(row.get("suite") == suite for row in rows) for suite in SUITES},
        "qualified_by_suite": {suite: len(qualified[suite]) for suite in SUITES},
        "selected_by_suite": {suite: [row["canonical_parent_key"] for row in selected_by_suite[suite]] for suite in SUITES},
        "engineering_invalid_result_count": engineering_invalid, "valid_ab_pair_count": valid_count,
        "classifications": classifications, "terminal_state_sha256_equal_count_descriptive": terminal_equal,
        "remaining_horizon_complete_equal_count_descriptive": horizon_equal,
        "terminal_state_sha256_gate_used": False, "remaining_horizon_complete_gate_used": False,
        "selection_rule_verified": not any(item.startswith("NON_PREFIX") for item in errors),
        "boundaries": boundaries, "errors": sorted(set(errors)), "audited_utc": utc_now(),
    }
    atomic_write_json(args.output_dir / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", audit_payload)
    if verdict == "PASS":
        audit_sha = sha256_file(args.output_dir / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json")
        manifest = {
            "schema": "STAGE_Q2_PARENT_MANIFEST_A_V1", "status": "PASS", "source_commit": args.source_commit,
            "source_tree": args.source_tree, "salt": SALT, "protocol_sha256": sha256_file(args.protocol),
            "candidate_universe_sha256": sha256_file(args.candidate_universe),
            "q2_control_qualification_report_sha256": sha256_file(args.report),
            "q2_control_qualification_rows_sha256": sha256_file(args.rows),
            "q2_control_qualification_independent_audit_sha256": audit_sha,
            "selected_parents": [
                {**row, "selection_role": "q2_qualified_clean_control_parent", "qualification_mode": "FRESH_CLEAN_AB_REPLAY", "source_artifact_read": False, "old_artifacts_reused": False}
                for suite in SUITES for row in selected_by_suite[suite]
            ],
            "selected_count": len(selected_keys), "selected_by_suite": {suite: len(selected_by_suite[suite]) for suite in SUITES},
            "old_artifacts_reused": False, "source_artifacts_modified": False,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
            "generated_utc": utc_now(),
        }
        atomic_write_json(args.output_dir / "Q2_PARENT_MANIFEST_A.json", manifest)
        (args.output_dir / "Q2_PARENT_MANIFEST_A.sha256").write_text(
            f"{sha256_file(args.output_dir / 'Q2_PARENT_MANIFEST_A.json')}  Q2_PARENT_MANIFEST_A.json\n", encoding="utf-8",
        )
        science_manifest = {
            "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V1", "status": "FROZEN",
            "source_commit": args.source_commit, "source_tree": args.source_tree,
            "q2_parent_manifest_a_sha256": sha256_file(args.output_dir / "Q2_PARENT_MANIFEST_A.json"),
            "candidate_universe_sha256": sha256_file(args.candidate_universe),
            "selected_parents": manifest["selected_parents"], "selected_count": len(selected_keys),
            "old_artifacts_reused": False, "source_artifacts_modified": False,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
            "generated_utc": utc_now(),
        }
        atomic_write_json(args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.json", science_manifest)
        (args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.sha256").write_text(
            f"{sha256_file(args.output_dir / 'STAGE_V_FORMAL_PARENT_MANIFEST_V1.json')}  STAGE_V_FORMAL_PARENT_MANIFEST_V1.json\n", encoding="utf-8",
        )
    return audit_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    payload = audit(args)
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
