#!/usr/bin/env python3
"""Phase D2+D3: Build candidate window dataset and Teacher V2 labels from FIT 800.

Reads S1 Teacher records, extracts close candidate segments, classifies each
window, and produces a sealed candidate-window dataset.

All output goes to a NEW directory; nothing in the original S1 or CLEAN roots
is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── constants ──────────────────────────────────────────────────────────
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))  # states 0-19
MIN_WINDOW_DURATION = 3
SUSTAIN_GAP = 5
MIN_CRITICAL_DURATION = 10

WINDOW_CATEGORIES = [
    "VALID_RETENTION",
    "CLOSE_WITHOUT_SUPPORT",
    "SUPPORT_WITHOUT_RETENTION",
    "PREMATURE_RELEASE",
    "UNSTABLE_MULTI_CLOSE",
    "POST_RETENTION_RELEASE",
    "NO_CLOSE_SIMPLE_NEGATIVE",
    "UNKNOWN_OR_AMBIGUOUS",
]

SCHEMA = "DETECTOR_V4_CANDIDATE_WINDOW_V1"


# ── helpers ────────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── candidate window extraction ────────────────────────────────────────
def extract_close_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract close event segments from teacher records."""
    events = []
    i = 0
    while i < len(records):
        r = records[i]
        if r.get("event_close_onset") and r.get("event_end_step", -1) >= 0:
            onset = r["step"]
            end = r["event_end_step"]
            if end >= onset + MIN_WINDOW_DURATION:
                events.append({
                    "onset_step": onset,
                    "end_step": min(end, len(records) - 1),
                    "duration": end - onset + 1,
                })
            i = end + 1 if end > i else i + 1
        else:
            i += 1
    return events


def classify_window(records: list[dict[str, Any]], onset: int, end: int,
                    close_events_in_episode: int) -> str:
    """Classify a close candidate window into one of the hard-negative categories.

    Uses privileged Teacher evidence fields (event_support, retention_active,
    grasp_support, release_imminent) AND cross-references with the original
    Teacher's retention_continuation_t10 labels to avoid labeling non-T10
    windows as VALID_RETENTION.
    """
    window_records = records[onset:end + 1]

    has_support = any(r.get("event_support") for r in window_records)
    has_retention = any(r.get("retention_active") and not r.get("retention_unknown_mask")
                       for r in window_records)
    has_grasp = any(r.get("grasp_support") and not r.get("retention_unknown_mask")
                   for r in window_records)
    has_release_imminent = any(r.get("release_imminent") and not r.get("retention_unknown_mask")
                               for r in window_records)
    has_release_event = any(r.get("event_release_onset") for r in window_records)
    has_opening = any(r.get("event_opening_stable") for r in window_records)
    has_valid_evidence = all(r.get("event_evidence_valid", True) for r in window_records)
    has_t10 = any(r.get("retention_continuation_t10") and not r.get("retention_unknown_mask")
                  for r in window_records)

    if not has_valid_evidence:
        return "UNKNOWN_OR_AMBIGUOUS"

    # Check if this is a valid retention window:
    # 1. Has support + retention + grasp
    # 2. No release_imminent during window
    # 3. Duration >= MIN_CRITICAL_DURATION
    # 4. Original Teacher labels some steps as retention_continuation_t10
    if has_support and has_retention and has_grasp and not has_release_imminent:
        if (end - onset + 1) >= MIN_CRITICAL_DURATION and has_t10:
            return "VALID_RETENTION"
        # Short window or no T10 label but all other signals present
        if not has_t10:
            return "SUPPORT_WITHOUT_RETENTION"
        post_window = records[end + 1:min(end + SUSTAIN_GAP + 1, len(records))]
        if any(r.get("event_release_onset") for r in post_window):
            return "PREMATURE_RELEASE"
        return "UNKNOWN_OR_AMBIGUOUS"

    # Multi-close instability (>=3 close events in episode)
    if close_events_in_episode >= 3 and not has_t10:
        return "UNSTABLE_MULTI_CLOSE"

    # Close without support
    if not has_support:
        if has_release_event or has_release_imminent:
            return "PREMATURE_RELEASE"
        return "CLOSE_WITHOUT_SUPPORT"

    # Support without retention
    if has_support and not has_retention:
        if has_grasp:
            return "SUPPORT_WITHOUT_RETENTION"
        return "CLOSE_WITHOUT_SUPPORT"

    # Release during window
    if has_release_event or has_release_imminent:
        return "PREMATURE_RELEASE"

    # Post-retention release (close after release)
    pre_window = records[max(0, onset - SUSTAIN_GAP):onset]
    if any(r.get("event_release_onset") for r in pre_window):
        return "POST_RETENTION_RELEASE"

    return "UNKNOWN_OR_AMBIGUOUS"


def build_teacher_v2_labels(records: list[dict[str, Any]],
                            windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-step Teacher V2 labels."""
    n = len(records)
    labels = []
    for step in range(n):
        r = records[step]
        in_close = any(w["onset_step"] <= step <= w["end_step"] for w in windows)
        in_valid_retention = any(
            w["onset_step"] <= step <= w["end_step"] and w["category"] == "VALID_RETENTION"
            for w in windows
        )
        # Find which window this step belongs to
        window_category = "NO_CLOSE_SIMPLE_NEGATIVE"
        window_id = -1
        for wi, w in enumerate(windows):
            if w["onset_step"] <= step <= w["end_step"]:
                window_category = w["category"]
                window_id = wi
                break

        is_veto = in_close and window_category != "VALID_RETENTION"
        has_retention = r.get("retention_active") and not r.get("retention_unknown_mask")
        has_grasp_support = r.get("grasp_support") and not r.get("retention_unknown_mask")
        is_release_imminent = r.get("release_imminent") and not r.get("retention_unknown_mask")
        evidence_valid = r.get("event_evidence_valid", True)
        has_t10 = r.get("retention_continuation_t10") and not r.get("retention_unknown_mask")

        # valid_retention: in close window + retention_active + grasp_support + NOT release_imminent + has T10
        step_valid_retention = (
            in_close and has_retention and has_grasp_support
            and not is_release_imminent and has_t10 and evidence_valid
        )

        labels.append({
            "step": step,
            "candidate_close": in_close,
            "valid_retention": step_valid_retention,
            "critical_retention_window": step_valid_retention,
            "false_trigger_veto": is_veto,
            "release_imminent": is_release_imminent,
            "support_without_retention": has_grasp_support and not has_retention and in_close,
            "close_without_support": in_close and window_category == "CLOSE_WITHOUT_SUPPORT",
            "unstable_close": window_category == "UNSTABLE_MULTI_CLOSE",
            "premature_release": window_category == "PREMATURE_RELEASE",
            "event_valid_mask": evidence_valid and not r.get("retention_unknown_mask", False),
            "window_id": window_id,
            "window_category": window_category,
        })
    return labels


# ── main pipeline ──────────────────────────────────────────────────────
def process_identity(s1_root: Path, suite: str, task: int, state: int
                     ) -> Optional[dict[str, Any]]:
    """Process one FIT identity: extract windows + build Teacher V2 labels."""
    ident_dir = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    teacher_path = ident_dir / "teacher_retention_records.jsonl"
    if not teacher_path.exists():
        return None

    records = jsonl(teacher_path)
    if not records:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
    close_events = extract_close_events(records)

    # Classify each window
    for w in close_events:
        w["category"] = classify_window(
            records, w["onset_step"], w["end_step"],
            len(close_events)
        )

    # Build Teacher V2 labels
    teacher_v2 = build_teacher_v2_labels(records, close_events)

    # Episode-level stats
    has_any_close = len(close_events) > 0
    n_valid = sum(1 for w in close_events if w["category"] == "VALID_RETENTION")
    n_hard_neg = sum(1 for w in close_events
                     if w["category"] not in ("VALID_RETENTION", "UNKNOWN_OR_AMBIGUOUS",
                                              "NO_CLOSE_SIMPLE_NEGATIVE"))
    n_close_steps = sum(1 for t in teacher_v2 if t["candidate_close"])

    return {
        "identity": cid,
        "suite": suite,
        "task_idx": task,
        "state_id": state,
        "n_steps": len(records),
        "n_close_events": len(close_events),
        "n_valid_retention_windows": n_valid,
        "n_hard_negative_windows": n_hard_neg,
        "has_any_close": has_any_close,
        "n_close_steps": n_close_steps,
        "windows": close_events,
        "teacher_v2_labels": teacher_v2,
        "close_event_hashes": [sha256_text(json.dumps(w, sort_keys=True))
                              for w in close_events],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-root", type=Path, required=True,
                       help="S1 root (OFFICIAL_V3_S1_FIT_V1_5e27d7c)")
    parser.add_argument("--output-root", type=Path, required=True,
                       help="New output directory (will be created)")
    parser.add_argument("--suite", choices=SUITES, default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--state", type=int, default=None)
    args = parser.parse_args()

    s1 = args.s1_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    # Determine scope
    if args.suite is not None and args.task is not None and args.state is not None:
        scope = [(args.suite, args.task, args.state)]
    else:
        scope = [(s, t, st) for s in SUITES for t in range(N_TASKS)
                for st in FIT_STATES]

    # Process
    episode_summaries = []
    all_windows = []
    window_counts: dict[str, int] = defaultdict(int)
    per_fold: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for suite, task, state in scope:
        result = process_identity(s1, suite, task, state)
        if result is None:
            print(f"SKIP: {suite}/task_{task:02d}/state_{state:02d} (no teacher data)")
            continue

        fold_id = state // 5  # 0-3 for states 0-19
        cid = result["identity"]

        episode_summaries.append({
            k: v for k, v in result.items()
            if k not in ("windows", "teacher_v2_labels")
        })

        for w in result["windows"]:
            cat = w["category"]
            window_counts[cat] += 1
            per_fold[fold_id][cat] += 1
            all_windows.append({
                "identity": cid,
                "suite": suite,
                "task_idx": task,
                "state_id": state,
                "fold_id": fold_id,
                "onset_step": w["onset_step"],
                "end_step": w["end_step"],
                "duration": w["duration"],
                "category": cat,
                "close_events_in_episode": result["n_close_events"],
            })

        # Write per-identity teacher V2 labels
        ident_out = out / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "teacher_v2_labels.jsonl", "w", encoding="utf-8") as fh:
            for label in result["teacher_v2_labels"]:
                fh.write(json.dumps(label, ensure_ascii=False) + "\n")

        # Write windows manifest per identity
        with open(ident_out / "candidate_windows.json", "w", encoding="utf-8") as fh:
            json.dump({
                "schema": SCHEMA,
                "identity": cid,
                "windows": result["windows"],
            }, fh, indent=2, ensure_ascii=False)

        print(f"OK: {cid}  close_events={result['n_close_events']}  "
              f"valid={result['n_valid_retention_windows']}  "
              f"hard_neg={result['n_hard_negative_windows']}")

    # ── Summary outputs ──
    # Episode-level summary
    with open(out / "episode_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "schema": f"{SCHEMA}_EPISODE_SUMMARY",
            "n_episodes_processed": len(episode_summaries),
            "episodes": episode_summaries,
        }, fh, indent=2, ensure_ascii=False)

    # Window-level CSV
    with open(out / "candidate_window_census.csv", "w", encoding="utf-8") as fh:
        headers = ["identity", "suite", "task_idx", "state_id", "fold_id",
                   "onset_step", "end_step", "duration", "category",
                   "close_events_in_episode"]
        fh.write(",".join(headers) + "\n")
        for w in all_windows:
            fh.write(",".join(str(w[h]) for h in headers) + "\n")

    # Window counts
    print("\n=== WINDOW COUNTS ===")
    for cat in WINDOW_CATEGORIES:
        print(f"  {cat}: {window_counts[cat]}")

    print("\n=== PER-FOLD COUNTS ===")
    for fid in sorted(per_fold):
        print(f"  Fold {fid}:")
        for cat in WINDOW_CATEGORIES:
            if per_fold[fid].get(cat, 0) > 0:
                print(f"    {cat}: {per_fold[fid][cat]}")

    # Build SHA256SUMS
    print("\n=== BUILDING SHA256SUMS ===")
    files = sorted(out.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = fp.relative_to(out)
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel}\n")
    sha = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    # Census manifest
    with open(out / "candidate_window_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({
            "schema": f"{SCHEMA}_MANIFEST",
            "s1_root_sha256": "15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf",
            "n_episodes": len(episode_summaries),
            "n_windows": len(all_windows),
            "window_counts": dict(window_counts),
            "per_fold_counts": {str(k): dict(v) for k, v in per_fold.items()},
            "sha256sums_sha256": sha,
            "categories": WINDOW_CATEGORIES,
        }, fh, indent=2, ensure_ascii=False)

    print(f"\nDONE: {len(episode_summaries)} episodes, {len(all_windows)} windows")
    print(f"SHA256SUMS: {sha}")


if __name__ == "__main__":
    main()
