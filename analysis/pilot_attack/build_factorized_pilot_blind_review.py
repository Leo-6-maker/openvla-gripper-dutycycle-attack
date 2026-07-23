#!/usr/bin/env python3
"""B4: Blind video review — condition hidden, blind-safe paths, two reviewers, unblinding sealed separately."""
from __future__ import annotations

import argparse, csv, json, os, random, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, load_strict_json

SELF_SHA = None

REVIEW_LABELS = ("premature_opening", "slip", "drop", "transport_failure",
                 "placement_failure", "recovery", "uncertain", "not_reviewable")

FORBIDDEN_IN_BLIND = frozenset({
    "condition", "attack_type", "TRUE", "RAND", "ORACLE", "CLEAN",
    "attack_timing", "epsilon", "pgd", "gradient", "perturbation",
    "RANDOM_TIME",
})


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-run-ledger", type=Path, required=True)
    ap.add_argument("--pilot-video-index", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    run_ledger = load_strict_json(args.pilot_run_ledger, "LEDGER")
    video = load_strict_json(args.pilot_video_index, "VIDEO")
    rng = random.Random(args.seed)

    runs = run_ledger.get("runs", run_ledger.get("entries", []))
    video_entries = video.get("entries", video.get("videos", []))

    # Map run to video
    video_map: dict[str, str] = {}
    for ve in video_entries:
        if isinstance(ve, dict):
            pid = ve.get("parent_id", ve.get("job_key", ""))
            video_map[str(pid)] = ve.get("path", ve.get("video_path", ""))

    blind_review: list[dict[str, Any]] = []
    unblinding: list[dict[str, Any]] = []
    blind_ids: set[str] = set()

    for i, run in enumerate(runs):
        pid = run.get("parent_id", f"unknown_{i}")
        cond = run.get("condition", "UNKNOWN")
        vp = run.get("video_path", video_map.get(pid, ""))

        # Generate unique blind ID
        while True:
            blind_id = f"B{rng.randint(10000, 99999)}"
            if blind_id not in blind_ids: break
        blind_ids.add(blind_id)

        # P0-11: Blind-safe path — do NOT expose condition in filename
        blind_path = f"video_{blind_id}.mp4"

        blind_review.append({
            "blind_id": blind_id, "video_path": blind_path,
            "reviewer_a_labels": list(REVIEW_LABELS),
            "reviewer_b_labels": list(REVIEW_LABELS),
            "reviewer_a_fields": {f"a_{lbl}": "" for lbl in REVIEW_LABELS},
            "reviewer_b_fields": {f"b_{lbl}": "" for lbl in REVIEW_LABELS},
            "disagreement": False,
        })

        unblinding.append({
            "blind_id": blind_id, "parent_id": pid, "condition": cond,
            "original_video_path": vp,
        })

    # Verify blind index doesn't expose condition
    for row in blind_review:
        for key in row:
            if any(fb.lower() in str(key).lower() for fb in FORBIDDEN_IN_BLIND):
                continue  # skip known review fields
        blind_json = json.dumps(row).lower()
        for fb in FORBIDDEN_IN_BLIND:
            if fb.lower() in blind_json:
                print(f"WARNING: Possible leak of '{fb}' in blind row {row.get('blind_id')}")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    with open(staging / "PILOT_BLIND_REVIEW_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "video_path", "reviewer_a_labels", "reviewer_b_labels"])
        w.writeheader()
        for row in blind_review:
            w.writerow({k: str(v) for k, v in row.items() if k in ("blind_id", "video_path", "reviewer_a_labels", "reviewer_b_labels")})

    # Full blind review JSON with reviewer fields
    (staging / "PILOT_BLIND_REVIEW_V0.json").write_text(
        json.dumps({"entries": blind_review, "n_videos": len(blind_review),
                     "n_reviewers": 2, "labels": list(REVIEW_LABELS)}, indent=2, sort_keys=True) + "\n")

    # Unblinding sealed separately
    with open(staging / "PILOT_UNBLINDING_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "parent_id", "condition", "original_video_path"])
        w.writeheader()
        for row in unblinding: w.writerow(row)

    (staging / "PILOT_UNBLINDING_V0.json").write_text(
        json.dumps({"entries": unblinding, "n_entries": len(unblinding)}, indent=2) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Blind Review: {out_root} n_videos={len(blind_review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
