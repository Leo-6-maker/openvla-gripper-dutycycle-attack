#!/usr/bin/env python3
"""Build unified teacher label index from privileged_step_records.

Applies a single teacher extraction contract to both Object500 and CLEAN1500 sources.
For Object500, existing teacher labels are read for consistency cross-check only
— they do NOT replace the unified extraction.

Usage:
  python build_teacher_label_index.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --object_teacher_labels /path/to/FOLD00_teacher_labels_heldout.jsonl \
    --output_dir /path/to/output

Output:
  CLEAN2000_TEACHER_LABEL_INDEX.jsonl
  CLEAN2000_TEACHER_CROSS_VALIDATION.json  (Object500 only)
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_schema import VALID_SUITES


def parse_args():
    p = argparse.ArgumentParser(description="Build CLEAN2000 teacher label index")
    p.add_argument("--index", required=True,
                   help="Path to CLEAN2000_INDEX_DRAFT.jsonl")
    p.add_argument("--object_teacher_labels", default=None,
                   help="Path to FOLD00_teacher_labels_heldout.jsonl (Object500 only)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def load_privileged_records(ep_dir):
    """Load all privileged_step_records as a list of dicts, keyed by step_idx."""
    p = os.path.join(ep_dir, "privileged_step_records.jsonl")
    if not os.path.exists(p):
        return {}
    records = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step_idx", rec.get("policy_step_idx", -1))
            records[step] = rec
    return records


def extract_teacher_label(records, n_steps, teacher_eligible):
    """Unified teacher extraction from privileged_step_records.

    Args:
      records: dict of step_idx -> privileged record
      n_steps: total episode steps
      teacher_eligible: from episode_summary (gate on mechanism eligibility)

    Returns dict with teacher_label_valid, teacher_anchor_step, etc.
    """
    if not records:
        return {
            "teacher_label_valid": False,
            "teacher_anchor_step": -1,
            "teacher_window_start": -1,
            "teacher_window_end": -1,
            "teacher_confidence": 0.0,
            "teacher_invalid_reason": "no_privileged_records",
        }

    # Primary gate: episode-level teacher_eligible flag
    if not teacher_eligible:
        return {
            "teacher_label_valid": False,
            "teacher_anchor_step": -1,
            "teacher_window_start": -1,
            "teacher_window_end": -1,
            "teacher_confidence": 0.0,
            "teacher_invalid_reason": "teacher_ineligible",
        }

    # Find gripper opening event: where gripper_opening_proxy crosses threshold
    # This is the canonical "gripper release" anchor
    sorted_steps = sorted(records.keys())
    anchor_step = -1
    gripper_threshold = 0.05  # opening proxy threshold

    for step in sorted_steps:
        rec = records[step]
        opening = rec.get("gripper_opening_proxy", 0.0)
        if opening > gripper_threshold:
            anchor_step = step
            break

    if anchor_step < 0:
        return {
            "teacher_label_valid": False,
            "teacher_anchor_step": -1,
            "teacher_window_start": -1,
            "teacher_window_end": -1,
            "teacher_confidence": 0.0,
            "teacher_invalid_reason": "no_gripper_opening_detected",
        }

    # Window: K steps around anchor (K=10 per runtime config)
    K = 10
    window_start = max(0, anchor_step - K)
    window_end = min(n_steps - 1, anchor_step + K)

    # Confidence: 1.0 if anchor within valid range, lower if at extremes
    confidence = 1.0
    if anchor_step < K or anchor_step > n_steps - K:
        confidence = 0.5

    return {
        "teacher_label_valid": True,
        "teacher_anchor_step": anchor_step,
        "teacher_window_start": window_start,
        "teacher_window_end": window_end,
        "teacher_confidence": confidence,
        "teacher_invalid_reason": "",
    }


def load_existing_object_labels(label_path):
    """Load existing Object500 teacher labels for cross-validation.

    Returns dict: (task_idx, state_id) -> list of per-step label dicts.
    """
    if not label_path or not os.path.exists(label_path):
        return {}
    labels = {}
    with open(label_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            key = (rec.get("task_idx", -1), rec.get("state_id", -1))
            if key not in labels:
                labels[key] = []
            labels[key].append(rec)
    return labels


def cross_validate(episode_key, task_id, state_id, our_label, existing_labels):
    """Compare our extracted label against existing Object500 labels.

    Returns list of mismatch entries (empty if consistent).
    """
    key = (task_id, state_id)
    if key not in existing_labels:
        return [{
            "episode_key": episode_key,
            "check": "teacher_crossval/missing_existing",
            "detail": "no existing teacher label found for task={} state={}".format(
                task_id, state_id),
            "severity": "WARNING",
        }]

    existing = existing_labels[key]
    mismatches = []

    # Check anchor consistency
    our_anchor = our_label.get("teacher_anchor_step", -1)
    # Find the existing label's anchor from the per-step records
    # (anchor is typically where phase transitions to "release" or similar)
    existing_anchors = []
    for rec in existing:
        phase = rec.get("phase", "")
        if phase in ("release", "grasp") and rec.get("gripper_close", False):
            e_anchor = rec.get("step_idx", rec.get("policy_step_idx", -1))
            if e_anchor >= 0:
                existing_anchors.append(e_anchor)

    if existing_anchors and our_anchor >= 0:
        # Check if our anchor is within reasonable range of any existing anchor
        closest = min(existing_anchors, key=lambda x: abs(x - our_anchor))
        if abs(closest - our_anchor) > 10:  # tolerance: 10 steps
            mismatches.append({
                "episode_key": episode_key,
                "check": "teacher_crossval/anchor_mismatch",
                "detail": "our_anchor={} vs closest_existing={} (diff={})".format(
                    our_anchor, closest, abs(closest - our_anchor)),
                "severity": "ERROR",
            })

    # Check label validity consistency
    our_valid = our_label.get("teacher_label_valid", False)
    # If existing labels have meaningful data, our extraction should agree
    if len(existing) > 0 and not our_valid:
        mismatches.append({
            "episode_key": episode_key,
            "check": "teacher_crossval/validity_mismatch",
            "detail": "our extraction says invalid but existing labels exist",
            "severity": "WARNING",
        })

    return mismatches


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load episode index
    print("Loading index: {}".format(args.index))
    rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("  {} episodes".format(len(rows)))

    # Load existing Object500 labels
    existing_labels = {}
    if args.object_teacher_labels:
        existing_labels = load_existing_object_labels(args.object_teacher_labels)
        print("  {} existing Object500 teacher labels".format(len(existing_labels)))

    # Process each episode
    print("Extracting teacher labels...")
    teacher_index = []
    crossval_issues = []
    stats = {
        "total": len(rows),
        "teacher_valid": 0,
        "teacher_invalid": 0,
        "by_reason": {},
        "crossval_errors": 0,
        "crossval_warnings": 0,
    }

    for i, row in enumerate(rows):
        ek = row["episode_key"]
        ep_dir = row["source_root"]
        n_steps = row["n_steps"]
        task_id = row["task_id"]
        state_id = row["state_id"]
        source_fmt = row["source_format"]

        # Load records
        records = load_privileged_records(ep_dir)
        # Extract teacher label (gate on episode-level teacher_eligible)
        label = extract_teacher_label(records, n_steps, row.get("teacher_eligible", False))

        # Build output row
        teacher_row = {
            "episode_key": ek,
            "parent_key": row["parent_key"],
            "suite": row["suite"],
            "task_id": task_id,
            "state_id": state_id,
            "teacher_label_valid": label["teacher_label_valid"],
            "teacher_anchor_step": label["teacher_anchor_step"],
            "teacher_window_start": label["teacher_window_start"],
            "teacher_window_end": label["teacher_window_end"],
            "teacher_confidence": label["teacher_confidence"],
            "teacher_invalid_reason": label["teacher_invalid_reason"],
            "teacher_artifact_sha256": _compute_teacher_artifact_sha(ep_dir),
            "n_privileged_records": len(records),
        }
        teacher_index.append(teacher_row)

        # Stats
        if label["teacher_label_valid"]:
            stats["teacher_valid"] += 1
        else:
            stats["teacher_invalid"] += 1
            reason = label["teacher_invalid_reason"]
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1

        # Cross-validate Object500 against existing labels
        if source_fmt == "object500_v1" and existing_labels:
            issues = cross_validate(ek, task_id, state_id, label, existing_labels)
            for iss in issues:
                crossval_issues.append(iss)
                if iss["severity"] == "ERROR":
                    stats["crossval_errors"] += 1
                else:
                    stats["crossval_warnings"] += 1

        if (i + 1) % 500 == 0:
            print("  {} / {} ...".format(i + 1, len(rows)))

    # Write outputs
    print("Writing teacher label index...")
    idx_path = os.path.join(args.output_dir, "CLEAN2000_TEACHER_LABEL_INDEX.jsonl")
    with open(idx_path, "w") as f:
        for row in teacher_index:
            f.write(json.dumps(row) + "\n")
    print("  {}".format(idx_path))

    if crossval_issues:
        cv_path = os.path.join(args.output_dir, "CLEAN2000_TEACHER_CROSS_VALIDATION.json")
        with open(cv_path, "w") as f:
            json.dump({
                "gate": "CLEAN2000_TEACHER_CROSS_VALIDATION_V1",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_issues": len(crossval_issues),
                "errors": stats["crossval_errors"],
                "warnings": stats["crossval_warnings"],
                "issues": crossval_issues[:100],  # first 100
            }, f, indent=2)
        print("  {} ({} issues)".format(cv_path, len(crossval_issues)))

    # Stats report
    stats_path = os.path.join(args.output_dir, "CLEAN2000_TEACHER_STATS.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("Teacher label extraction complete.")
    print("  Total episodes:       {}".format(stats["total"]))
    print("  Teacher valid:        {}".format(stats["teacher_valid"]))
    print("  Teacher invalid:      {}".format(stats["teacher_invalid"]))
    print("  Invalid reasons:")
    for reason, count in sorted(stats["by_reason"].items(), key=lambda x: -x[1]):
        print("    {}: {}".format(reason, count))
    if stats["crossval_errors"] > 0:
        print("  Cross-val ERRORS:     {}".format(stats["crossval_errors"]))
    if stats["crossval_warnings"] > 0:
        print("  Cross-val WARNINGS:   {}".format(stats["crossval_warnings"]))


def _compute_teacher_artifact_sha(ep_dir):
    """Hash the privileged_step_records as the teacher artifact."""
    p = os.path.join(ep_dir, "privileged_step_records.jsonl")
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    main()
