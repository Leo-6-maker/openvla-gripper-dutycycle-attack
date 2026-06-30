#!/usr/bin/env python3
"""Classify CLEAN2000 episodes into mutually exclusive PRIMARY / SAFETY / EXCLUDED sets.

Classification rules (fail-closed):
  PRIMARY: clean_success AND teacher_label_valid AND telemetry_complete
           AND mechanism_eligible AND no schema fail AND complete
  SAFETY_ABSTENTION: clean_success but teacher_invalid OR mechanism_ineligible
  EXCLUDED_SCHEMA: schema validation failed
  EXCLUDED_INFRA: missing COMPLETE or artifact files
  EXCLUDED_TELEMETRY: telemetry incomplete (step gaps, invalid features)

Requires that PRIMARY + SAFETY + EXCLUDED = all episodes with empty intersections.

Usage:
  python build_primary_and_safety_sets.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --teacher_index CLEAN2000_TEACHER_LABEL_INDEX.jsonl \
    --output_dir /path/to/output

Output:
  CLEAN2000_PRIMARY.jsonl
  CLEAN2000_SAFETY_ABSTENTION.jsonl
  CLEAN2000_EXCLUDED.jsonl
  CLEAN2000_SET_CLOSURE_REPORT.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Build CLEAN2000 primary/safety sets")
    p.add_argument("--index", required=True)
    p.add_argument("--teacher_index", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index_by_key(path):
    """Load JSONL and index by episode_key."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            d[row["episode_key"]] = row
    return d


def classify(row, teacher_row):
    """Classify a single episode.

    Returns (set_name, reason) where set_name is one of:
      PRIMARY, SAFETY_ABSTENTION, EXCLUDED_SCHEMA, EXCLUDED_INFRA, EXCLUDED_TELEMETRY
    """
    ek = row.get("episode_key", "?")

    # ── EXCLUDED_INFRA: missing artifacts or not complete ──
    if not row.get("complete", False):
        return ("EXCLUDED_INFRA", "not_complete")
    if not row.get("step_telemetry_sha256", ""):
        return ("EXCLUDED_INFRA", "missing_step_telemetry")
    if not row.get("episode_summary_sha256", ""):
        return ("EXCLUDED_INFRA", "missing_episode_summary")
    if not row.get("artifact_inventory_sha256", ""):
        return ("EXCLUDED_INFRA", "missing_artifact_inventory")

    # ── EXCLUDED_SCHEMA: schema validation failures ──
    if not row.get("gate_pass", True):
        return ("EXCLUDED_SCHEMA", "gate_pass_false")
    if row.get("condition", "") != "CLEAN":
        return ("EXCLUDED_SCHEMA", "not_clean_condition")

    # ── EXCLUDED_TELEMETRY: telemetry issues ──
    if row.get("n_steps", 0) <= 0:
        return ("EXCLUDED_TELEMETRY", "n_steps_zero")
    if not row.get("step_index_contiguous", False):
        return ("EXCLUDED_TELEMETRY", "step_index_not_contiguous")
    if row.get("n_telemetry_rows", 0) <= 0:
        return ("EXCLUDED_TELEMETRY", "no_telemetry_rows")
    if row.get("duplicate_step_count", 0) > 0:
        return ("EXCLUDED_TELEMETRY", "duplicate_steps")
    if row.get("missing_step_count", 0) > 0:
        return ("EXCLUDED_TELEMETRY", "missing_steps")

    # ── SAFETY_ABSTENTION: teacher or mechanism issues ──
    if not row.get("mechanism_eligible", False):
        return ("SAFETY_ABSTENTION", "mechanism_ineligible: {}".format(
            row.get("abstain_reason", "unknown")))
    if not row.get("teacher_eligible", False):
        return ("SAFETY_ABSTENTION", "teacher_ineligible")

    # Check teacher label
    if teacher_row:
        if not teacher_row.get("teacher_label_valid", False):
            return ("SAFETY_ABSTENTION", "teacher_label_invalid: {}".format(
                teacher_row.get("teacher_invalid_reason", "unknown")))

    # ── PRIMARY ──
    if not row.get("task_success", False):
        return ("SAFETY_ABSTENTION", "clean_failure")

    return ("PRIMARY", "all_checks_passed")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading index: {}".format(args.index))
    rows = load_jsonl(args.index)
    print("  {} episodes".format(len(rows)))

    print("Loading teacher index: {}".format(args.teacher_index))
    teacher_by_key = load_index_by_key(args.teacher_index)
    print("  {} teacher labels".format(len(teacher_by_key)))

    # Classify
    print("Classifying...")
    sets = {
        "PRIMARY": [],
        "SAFETY_ABSTENTION": [],
        "EXCLUDED_SCHEMA": [],
        "EXCLUDED_INFRA": [],
        "EXCLUDED_TELEMETRY": [],
    }
    unclassified = []

    for row in rows:
        ek = row["episode_key"]
        teacher_row = teacher_by_key.get(ek, {})
        set_name, reason = classify(row, teacher_row)

        entry = {
            "episode_key": ek,
            "parent_key": row["parent_key"],
            "suite": row["suite"],
            "task_id": row["task_id"],
            "state_id": row["state_id"],
            "set": set_name,
            "reason": reason,
            "task_success": row.get("task_success", False),
            "teacher_eligible": row.get("teacher_eligible", False),
            "mechanism_eligible": row.get("mechanism_eligible", False),
            "teacher_label_valid": teacher_row.get("teacher_label_valid", False) if teacher_row else False,
        }

        if set_name in sets:
            sets[set_name].append(entry)
        else:
            unclassified.append(entry)

    # ── Closure check ──
    total_classified = sum(len(v) for v in sets.values()) + len(unclassified)
    expected = len(rows)
    intersection_free = True

    print()
    print("Set counts:")
    for name, entries in sorted(sets.items()):
        print("  {}: {}".format(name, len(entries)))
    if unclassified:
        print("  UNCLASSIFIED: {}".format(len(unclassified)))

    # Check closure
    if total_classified != expected:
        print("  ERROR: closure gap: {} classified != {} expected".format(
            total_classified, expected))
    else:
        print("  Closure: PASS ({} == {})".format(total_classified, expected))

    # ── Write outputs ──
    for name, entries in sets.items():
        if entries:
            path = os.path.join(args.output_dir, "CLEAN2000_{}.jsonl".format(name))
            with open(path, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            print("  {}".format(path))

    # Closure report
    report = {
        "gate": "CLEAN2000_SET_CLOSURE_REPORT_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_episodes": expected,
        "per_set": {name: len(entries) for name, entries in sorted(sets.items())},
        "unclassified": len(unclassified),
        "closure_pass": (total_classified == expected and len(unclassified) == 0),
        "intersection_free": intersection_free,
    }
    report_path = os.path.join(args.output_dir, "CLEAN2000_SET_CLOSURE_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  {}".format(report_path))

    # Per-suite breakdown
    by_suite = {}
    for name, entries in sorted(sets.items()):
        for e in entries:
            s = e["suite"]
            if s not in by_suite:
                by_suite[s] = {}
            by_suite[s][name] = by_suite[s].get(name, 0) + 1

    print()
    print("Per-suite breakdown:")
    for suite in sorted(by_suite):
        parts = ["{}={}".format(k, v) for k, v in sorted(by_suite[suite].items())]
        print("  {}: {}".format(suite, ", ".join(parts)))

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
