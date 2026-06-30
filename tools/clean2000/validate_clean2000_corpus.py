#!/usr/bin/env python3
"""Validate CLEAN2000 canonical episode index against the full contract.

This is a fail-closed validator. Every check failure produces a rejection entry.
Only a 100% clean report permits downstream freezing.

Usage:
  python validate_clean2000_corpus.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --output_dir /path/to/output

Output:
  CLEAN2000_VALIDATION_REPORT.json   — structured pass/fail report
  CLEAN2000_REJECTION_LEDGER.jsonl   — one entry per rejection
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_schema import (
    CANONICAL_FIELDS, EXPECTED_PER_SUITE, EXPECTED_TOTAL,
    VALID_SUITES, REQUIRED_CONDITION,
    TASK_ID_RANGE, STATE_ID_RANGE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Validate CLEAN2000 corpus index")
    p.add_argument("--index", required=True,
                   help="Path to CLEAN2000_INDEX_DRAFT.jsonl")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output files")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.index):
        print("ERROR: index file not found: {}".format(args.index))
        sys.exit(1)

    # Load index
    print("Loading index: {}".format(args.index))
    rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("  Loaded {} rows".format(len(rows)))

    rejections = []
    checks = {}

    # ── Schema checks ──
    schema_issues = _check_schema(rows)
    checks["schema"] = {"passed": len(schema_issues) == 0, "issues": len(schema_issues)}
    rejections.extend(schema_issues)
    print("  Schema: {} issues".format(len(schema_issues)))

    # ── Count checks ──
    count_issues = _check_counts(rows)
    checks["counts"] = {"passed": len(count_issues) == 0, "issues": len(count_issues)}
    rejections.extend(count_issues)
    print("  Counts: {} issues".format(len(count_issues)))

    # ── Suite distribution ──
    suite_issues = _check_suite_distribution(rows)
    checks["suite_distribution"] = {"passed": len(suite_issues) == 0, "issues": len(suite_issues)}
    rejections.extend(suite_issues)
    print("  Suite distribution: {} issues".format(len(suite_issues)))

    # ── Identity uniqueness ──
    identity_issues = _check_identity(rows)
    checks["identity"] = {"passed": len(identity_issues) == 0, "issues": len(identity_issues)}
    rejections.extend(identity_issues)
    print("  Identity: {} issues".format(len(identity_issues)))

    # ── Condition gate ──
    condition_issues = _check_condition(rows)
    checks["condition"] = {"passed": len(condition_issues) == 0, "issues": len(condition_issues)}
    rejections.extend(condition_issues)
    print("  Condition: {} issues".format(len(condition_issues)))

    # ── Telemetry completeness ──
    telemetry_issues = _check_telemetry(rows)
    checks["telemetry"] = {"passed": len(telemetry_issues) == 0, "issues": len(telemetry_issues)}
    rejections.extend(telemetry_issues)
    print("  Telemetry: {} issues".format(len(telemetry_issues)))

    # ── Artifact SHA checks ──
    artifact_issues = _check_artifacts(rows)
    checks["artifacts"] = {"passed": len(artifact_issues) == 0, "issues": len(artifact_issues)}
    rejections.extend(artifact_issues)
    print("  Artifacts: {} issues".format(len(artifact_issues)))

    # ── Source root integrity ──
    source_issues = _check_source_roots(rows)
    checks["source_roots"] = {"passed": len(source_issues) == 0, "issues": len(source_issues)}
    rejections.extend(source_issues)
    print("  Source roots: {} issues".format(len(source_issues)))

    # ── Task/state ranges ──
    range_issues = _check_ranges(rows)
    checks["ranges"] = {"passed": len(range_issues) == 0, "issues": len(range_issues)}
    rejections.extend(range_issues)
    print("  Ranges: {} issues".format(len(range_issues)))

    # ── COMPLETE status ──
    complete_issues = _check_complete_status(rows)
    checks["complete"] = {"passed": len(complete_issues) == 0, "issues": len(complete_issues)}
    rejections.extend(complete_issues)
    print("  Complete: {} issues".format(len(complete_issues)))

    total_rejections = len(rejections)
    passed = total_rejections == 0

    print()
    print("RESULT: {} ({})".format(
        "PASS" if passed else "FAIL",
        "{} issues".format(total_rejections) if total_rejections else "clean",
    ))

    # ── Write outputs ──
    report = {
        "gate": "CLEAN2000_VALIDATION_REPORT_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_path": os.path.abspath(args.index),
        "total_rows": len(rows),
        "passed": passed,
        "total_rejections": total_rejections,
        "checks": checks,
        "rejection_summary": _summarize_rejections(rejections),
    }

    report_path = os.path.join(args.output_dir, "CLEAN2000_VALIDATION_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  {}".format(report_path))

    if rejections:
        ledger_path = os.path.join(args.output_dir, "CLEAN2000_REJECTION_LEDGER.jsonl")
        with open(ledger_path, "w") as f:
            for r in rejections:
                f.write(json.dumps(r) + "\n")
        print("  {}".format(ledger_path))

    sys.exit(0 if passed else 1)


def _reject(episode_key, check, detail, severity="ERROR"):
    return {
        "episode_key": episode_key,
        "check": check,
        "detail": detail,
        "severity": severity,
    }


def _check_schema(rows):
    issues = []
    required_fields = list(CANONICAL_FIELDS.keys())
    for i, row in enumerate(rows):
        ek = row.get("episode_key", "row_{}".format(i))
        for field in required_fields:
            if field not in row:
                issues.append(_reject(ek, "schema/missing_field",
                    "missing required field: {}".format(field)))
        # Check no extra unknown fields
        for k in row:
            if k not in CANONICAL_FIELDS:
                issues.append(_reject(ek, "schema/unknown_field",
                    "unknown field: {}".format(k), "WARNING"))
    return issues


def _check_counts(rows):
    issues = []
    total = len(rows)
    if total != EXPECTED_TOTAL:
        issues.append(_reject("GLOBAL", "counts/total",
            "expected {} episodes, got {}".format(EXPECTED_TOTAL, total)))
    return issues


def _check_suite_distribution(rows):
    issues = []
    per_suite = {}
    for r in rows:
        s = r.get("suite", "")
        per_suite[s] = per_suite.get(s, 0) + 1

    for suite, expected in EXPECTED_PER_SUITE.items():
        actual = per_suite.get(suite, 0)
        if actual != expected:
            issues.append(_reject("GLOBAL", "suite_distribution/{}".format(suite),
                "expected {} episodes for {}, got {}".format(expected, suite, actual)))
    return issues


def _check_identity(rows):
    issues = []
    episode_keys = {}
    parent_keys = {}

    for r in rows:
        ek = r.get("episode_key", "")
        pk = r.get("parent_key", "")

        if not ek:
            issues.append(_reject("UNKNOWN", "identity/empty_episode_key",
                "episode_key is empty"))
            continue
        if not pk:
            issues.append(_reject(ek, "identity/empty_parent_key",
                "parent_key is empty"))
            continue

        # episode_key uniqueness
        if ek in episode_keys:
            issues.append(_reject(ek, "identity/duplicate_episode_key",
                "duplicate episode_key, first seen at {}".format(episode_keys[ek])))
        else:
            episode_keys[ek] = r.get("source_root", "")

        # parent_key: track for uniqueness (within clean collection)
        if pk in parent_keys:
            issues.append(_reject(ek, "identity/duplicate_parent_key",
                "duplicate parent_key, first seen at {}".format(parent_keys[pk])))
        else:
            parent_keys[pk] = r.get("source_root", "")

        # Check parent_key format
        expected_pk = "{}/task_{:02d}/state_{:03d}".format(
            r.get("suite", ""), r.get("task_id", -1), r.get("state_id", -1))
        if pk != expected_pk:
            issues.append(_reject(ek, "identity/parent_key_format",
                "expected '{}' got '{}'".format(expected_pk, pk)))

    return issues


def _check_condition(rows):
    issues = []
    for r in rows:
        ek = r.get("episode_key", "?")
        cond = r.get("condition", "")
        if cond != REQUIRED_CONDITION:
            issues.append(_reject(ek, "condition/not_clean",
                "condition='{}' must be '{}'".format(cond, REQUIRED_CONDITION)))
    return issues


def _check_telemetry(rows):
    issues = []
    for r in rows:
        ek = r.get("episode_key", "?")
        n_steps = r.get("n_steps", 0)
        n_tele = r.get("n_telemetry_rows", 0)
        n_valid = r.get("n_valid_steps", 0)
        n_invalid = r.get("invalid_feature_steps", 0)
        contiguous = r.get("step_index_contiguous", False)
        first_valid = r.get("first_valid_step", -1)

        if n_steps <= 0:
            issues.append(_reject(ek, "telemetry/n_steps_zero",
                "n_steps={} must be >= 1".format(n_steps)))

        if n_tele <= 0:
            issues.append(_reject(ek, "telemetry/no_telemetry_rows",
                "n_telemetry_rows={}".format(n_tele)))

        if n_valid + n_invalid != n_tele:
            issues.append(_reject(ek, "telemetry/valid_invalid_mismatch",
                "n_valid({}) + n_invalid({}) != n_telemetry_rows({})".format(
                    n_valid, n_invalid, n_tele), "WARNING"))

        if n_invalid < 0:
            issues.append(_reject(ek, "telemetry/negative_invalid",
                "invalid_feature_steps={} must be >= 0".format(n_invalid)))

        if not contiguous:
            issues.append(_reject(ek, "telemetry/step_not_contiguous",
                "step indices are not contiguous ({} duplicates, {} missing)".format(
                    r.get("duplicate_step_count", 0), r.get("missing_step_count", 0)),
                "WARNING"))

        # first_valid_step should be >= 0 if any valid steps exist
        if n_valid > 0 and first_valid < 0:
            issues.append(_reject(ek, "telemetry/first_valid_inconsistent",
                "n_valid={} but first_valid_step={}".format(n_valid, first_valid)))

    return issues


def _check_artifacts(rows):
    issues = []
    for r in rows:
        ek = r.get("episode_key", "?")
        for field in ["episode_summary_sha256", "step_telemetry_sha256",
                      "complete_marker_sha256", "artifact_inventory_sha256"]:
            val = r.get(field, "")
            if not val:
                issues.append(_reject(ek, "artifacts/missing_sha",
                    "{} is empty".format(field)))
            elif len(val) != 64:
                issues.append(_reject(ek, "artifacts/sha_length",
                    "{} length={} expected 64".format(field, len(val))))
            elif not all(c in "0123456789abcdef" for c in val):
                issues.append(_reject(ek, "artifacts/sha_hex",
                    "{} is not hex".format(field)))
    return issues


def _check_source_roots(rows):
    issues = []
    sources = set()
    for r in rows:
        ek = r.get("episode_key", "?")
        src = r.get("source_root", "")
        fmt = r.get("source_format", "")

        if not src or not os.path.isdir(src):
            issues.append(_reject(ek, "source/invalid_root",
                "source_root does not exist: {}".format(src)))
            continue

        # Check COMPLETE marker
        if not os.path.exists(os.path.join(src, "COMPLETE.json")):
            issues.append(_reject(ek, "source/no_complete_marker",
                "source_root missing COMPLETE.json"))

        # Check episode_summary
        if not os.path.exists(os.path.join(src, "episode_summary.json")):
            issues.append(_reject(ek, "source/no_episode_summary",
                "source_root missing episode_summary.json"))

        # Check step_telemetry
        if not os.path.exists(os.path.join(src, "step_telemetry.csv")):
            issues.append(_reject(ek, "source/no_telemetry",
                "source_root missing step_telemetry.csv"))

        # Check source_format
        if fmt not in ("clean1500_v1", "object500_v1"):
            issues.append(_reject(ek, "source/unknown_format",
                "unknown source_format: {}".format(fmt)))

        # Check no attack artifacts (object500 only)
        if fmt == "object500_v1":
            for attack_file in ["RUN_COMMAND.txt"]:
                fp = os.path.join(src, attack_file)
                if os.path.exists(fp):
                    with open(fp) as f:
                        content = f.read()
                    if "TRUE_T10" in content or "RANDOM" in content:
                        issues.append(_reject(ek, "source/attack_detected",
                            "attack condition found in RUN_COMMAND.txt"))

        sources.add(src)

    return issues


def _check_ranges(rows):
    issues = []
    for r in rows:
        ek = r.get("episode_key", "?")
        tid = r.get("task_id", -1)
        sid = r.get("state_id", -1)
        suite = r.get("suite", "")

        if not (TASK_ID_RANGE[0] <= tid <= TASK_ID_RANGE[1]):
            issues.append(_reject(ek, "range/task_id",
                "task_id={} out of range [{},{}]".format(
                    tid, TASK_ID_RANGE[0], TASK_ID_RANGE[1])))

        if not (STATE_ID_RANGE[0] <= sid <= STATE_ID_RANGE[1]):
            issues.append(_reject(ek, "range/state_id",
                "state_id={} out of range [{},{}]".format(
                    sid, STATE_ID_RANGE[0], STATE_ID_RANGE[1])))

        if suite not in VALID_SUITES:
            issues.append(_reject(ek, "range/suite",
                "suite='{}' not in valid suites".format(suite)))

    return issues


def _check_complete_status(rows):
    issues = []
    for r in rows:
        ek = r.get("episode_key", "?")
        if not r.get("complete", False):
            issues.append(_reject(ek, "complete/not_complete",
                "episode is not marked complete"))
    return issues


def _summarize_rejections(rejections):
    from collections import Counter
    by_check = Counter(r["check"] for r in rejections)
    by_severity = Counter(r["severity"] for r in rejections)
    return {
        "by_check": dict(by_check.most_common()),
        "by_severity": dict(by_severity),
    }


if __name__ == "__main__":
    main()
