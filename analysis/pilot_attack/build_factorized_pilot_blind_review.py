#!/usr/bin/env python3
"""B4: Build blind video review package — condition hidden, separate unblinding table."""
from __future__ import annotations

import argparse, csv, json, os, random, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from pilot_integrity import sha256_file, load_strict_json, seal_output_dir

SELF_SHA = None
BLIND_FIELDS = ("opening", "slip", "drop", "premature_release", "transport_failure",
                "placement_failure", "recovery", "uncertain")


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

    runs = run_ledger.get("runs", run_ledger.get("entries", []))
    rng = random.Random(args.seed)

    # Generate blind IDs
    blind_map: dict[str, str] = {}
    blind_review: list[dict[str, Any]] = []
    unblinding: list[dict[str, Any]] = []

    for i, run in enumerate(runs):
        pid = run.get("parent_id", f"unknown_{i}")
        cond = run.get("condition", "UNKNOWN")
        vp = run.get("video_path", "")
        blind_id = f"B{rng.randint(1000, 9999)}"

        blind_review.append({
            "blind_id": blind_id, "video_path": vp, "parent_id": pid,
            "review_fields": list(BLIND_FIELDS),
        })
        unblinding.append({
            "blind_id": blind_id, "parent_id": pid, "condition": cond,
        })

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    with open(staging / "PILOT_BLIND_REVIEW_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "video_path", "parent_id", "review_fields"])
        w.writeheader()
        for row in blind_review: w.writerow(row)

    with open(staging / "PILOT_UNBLINDING_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "parent_id", "condition"])
        w.writeheader()
        for row in unblinding: w.writerow(row)

    (staging / "PILOT_UNBLINDING_V0.json").write_text(json.dumps(unblinding, indent=2) + "\n")

    # Seal staging then move to out_root
    seal = sha256_file(staging / "PILOT_BLIND_REVIEW_V0.csv")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)
    print(f"Blind Review Package: {out_root} n_videos={len(blind_review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
