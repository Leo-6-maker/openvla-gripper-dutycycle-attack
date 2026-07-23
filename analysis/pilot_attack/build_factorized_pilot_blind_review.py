#!/usr/bin/env python3
"""B4: Blind video review — two physically separate sealed roots.

BLIND_PACKAGE_ROOT: reviewer-accessible, no condition info.
UNBLINDING_ROOT: separately sealed, maps blind_id -> condition.
The two roots are independent sealed directories with different seals.
"""
from __future__ import annotations

import argparse, csv, json, os, random, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, load_strict_json, seal_dir_in_place

SELF_SHA = None
REVIEW_LABELS = ("premature_opening", "slip", "drop", "transport_failure",
                 "placement_failure", "recovery", "uncertain", "not_reviewable")

FORBIDDEN_IN_BLIND = frozenset({
    "condition", "attack_type", "TRUE", "RAND", "ORACLE", "CLEAN",
    "attack_timing", "epsilon", "pgd", "gradient", "perturbation", "RANDOM_TIME",
})


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-video-index", type=Path, required=True)
    ap.add_argument("--blind-package-root", type=Path, required=True)
    ap.add_argument("--unblinding-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blind_root = args.blind_package_root.resolve()
    unblind_root = args.unblinding_root.resolve()
    if blind_root.exists(): raise SystemExit(f"BLIND_ROOT_EXISTS: {blind_root}")
    if unblind_root.exists(): raise SystemExit(f"UNBLIND_ROOT_EXISTS: {unblind_root}")
    if blind_root == unblind_root:
        raise SystemExit("BLIND_AND_UNBLIND_SAME_ROOT")

    run_ledger = load_strict_json(args.pilot_run_ledger, "LEDGER")
    video = load_strict_json(args.pilot_video_index, "VIDEO")
    rng = random.Random(args.seed)

    runs = run_ledger.get("runs", [])
    video_entries = video.get("entries", [])

    blind_entries: list[dict[str, Any]] = []
    unblind_entries: list[dict[str, Any]] = []
    blind_ids: set[str] = set()

    for i, run in enumerate(runs):
        pid = run.get("parent_id", f"unknown_{i}")
        cond = run.get("condition", "UNKNOWN")
        vp = run.get("video_path", "")

        while True:
            blind_id = f"B{rng.randint(10000, 99999)}"
            if blind_id not in blind_ids: break
        blind_ids.add(blind_id)

        blind_entries.append({
            "blind_id": blind_id, "video_reference": vp,
            "reviewer_a_labels": list(REVIEW_LABELS),
            "reviewer_b_labels": list(REVIEW_LABELS),
        })

        unblind_entries.append({
            "blind_id": blind_id, "parent_id": pid, "condition": cond,
            "original_video_path": vp,
        })

    # Fix 11: Verify blind package does NOT expose condition
    blind_json_str = json.dumps(blind_entries, sort_keys=True).lower()
    for fb in FORBIDDEN_IN_BLIND:
        if fb.lower() in blind_json_str:
            raise SystemExit(f"BLIND_LEAK_DETECTED: '{fb}' found in blind package content")

    # Write BLIND_PACKAGE_ROOT (reviewer gets this only)
    blind_root.mkdir(parents=True)
    with open(blind_root / "PILOT_BLIND_REVIEW_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "video_reference"])
        w.writeheader()
        for e in blind_entries: w.writerow({"blind_id": e["blind_id"], "video_reference": e["video_reference"]})

    (blind_root / "PILOT_BLIND_REVIEW_V0.json").write_text(
        json.dumps({"entries": blind_entries, "n_videos": len(blind_entries),
                     "n_reviewers": 2, "labels": list(REVIEW_LABELS),
                     "instructions": "Review each video for the listed labels. Do NOT attempt to identify conditions."},
                   indent=2, sort_keys=True) + "\n")
    seal_dir_in_place(blind_root)

    # Write UNBLINDING_ROOT (separately sealed, NOT given to reviewer)
    unblind_root.mkdir(parents=True)
    with open(unblind_root / "PILOT_UNBLINDING_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "parent_id", "condition", "original_video_path"])
        w.writeheader()
        for e in unblind_entries: w.writerow(e)

    (unblind_root / "PILOT_UNBLINDING_V0.json").write_text(
        json.dumps({"entries": unblind_entries, "n_entries": len(unblind_entries)}, indent=2) + "\n")
    seal_dir_in_place(unblind_root)

    print(f"Blind Review: blind={blind_root} ({len(blind_entries)} videos)")
    print(f"  Unblinding: {unblind_root} (SEPARATE — do not share with reviewer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
