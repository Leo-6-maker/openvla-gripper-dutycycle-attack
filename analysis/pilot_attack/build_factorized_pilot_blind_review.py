#!/usr/bin/env python3
"""B4 v2.3.1: Blind video review — cross-receipt seal binding, mandatory video SHA, symlink-safe, no leaks."""
from __future__ import annotations

import argparse, csv, json, os, random, shutil, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, is_64char_hex, consume_sealed_root, guard_path_safe, seal_dir_in_place

SELF_SHA = None
REVIEW_LABELS = ("premature_opening", "slip", "drop", "transport_failure",
                 "placement_failure", "recovery", "uncertain", "not_reviewable")

FORBIDDEN_IN_BLIND = frozenset({
    "condition", "attack_type", "TRUE", "RAND", "ORACLE", "CLEAN",
    "attack_timing", "epsilon", "pgd", "gradient", "perturbation", "RANDOM_TIME",
})

MANDATORY_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-execution-validation-root", type=Path, required=True)
    ap.add_argument("--pilot-run-ledger-root", type=Path, required=True)
    ap.add_argument("--pilot-video-index-root", type=Path, required=True)
    ap.add_argument("--evidence-root", type=Path, required=True)
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
    # ── Reject nested roots ───────────────────────────────────────────────
    try:
        blind_root.relative_to(unblind_root)
        raise SystemExit(f"BLIND_NESTED_INSIDE_UNBLIND: blind={blind_root} unblind={unblind_root}")
    except ValueError: pass
    try:
        unblind_root.relative_to(blind_root)
        raise SystemExit(f"UNBLIND_NESTED_INSIDE_BLIND: blind={blind_root} unblind={unblind_root}")
    except ValueError: pass

    evidence_root = args.evidence_root.resolve()

    # ── Consume all sealed roots ──────────────────────────────────────────
    exec_val, exec_val_seal = consume_sealed_root(
        args.pilot_execution_validation_root, "PILOT_EXECUTION_VALIDATION_V0", "EXEC_VAL")
    if exec_val.get("status") != "PASS":
        raise SystemExit("EXEC_VALIDATION_NOT_PASS: cannot build blind package on HOLD execution")

    run_ledger, run_ledger_seal = consume_sealed_root(
        args.pilot_run_ledger_root, "PILOT_RUN_LEDGER_V0", "RUN_LEDGER")
    video_index, video_index_seal = consume_sealed_root(
        args.pilot_video_index_root, "PILOT_VIDEO_INDEX_V0", "VIDEO_INDEX")

    # ── Cross-receipt seal binding ────────────────────────────────────────
    declared_seals = exec_val.get("input_seals", {})
    declared_run_seal = declared_seals.get("run_ledger", "")
    declared_video_seal = declared_seals.get("video_index", "")
    if declared_run_seal and declared_run_seal != run_ledger_seal:
        raise SystemExit(f"BLIND_RUN_SEAL_BINDING: declared={declared_run_seal[:16]} actual={run_ledger_seal[:16]}")
    if declared_video_seal and declared_video_seal != video_index_seal:
        raise SystemExit(f"BLIND_VIDEO_SEAL_BINDING: declared={declared_video_seal[:16]} actual={video_index_seal[:16]}")

    runs = run_ledger.get("runs", [])
    video_entries = video_index.get("entries", [])

    # Build video index by job_id
    video_by_id: dict[str, dict[str, Any]] = {}
    for ve in video_entries:
        jid = ve.get("job_id", "")
        if jid:
            video_by_id[jid] = ve

    rng = random.Random(args.seed)

    blind_entries: list[dict[str, Any]] = []
    unblind_entries: list[dict[str, Any]] = []
    blind_ids: set[str] = set()

    blind_videos_dir = blind_root / "videos"
    blind_videos_dir.mkdir(parents=True)

    for run in runs:
        jid = run.get("job_id", "")
        pid = run.get("parent_id", "UNKNOWN")
        cond = run.get("condition", "UNKNOWN")
        vp = run.get("video_path", "")

        ve = video_by_id.get(jid, {})
        index_path = ve.get("path", "")
        declared_sha = ve.get("sha256", "")

        # ── Video SHA mandatory ───────────────────────────────────────
        if not is_64char_hex(declared_sha):
            raise SystemExit(f"VIDEO_SHA_MISSING_OR_INVALID: jid={jid} sha={declared_sha[:40]!r}")

        # ── Index path must match run path exactly ─────────────────────
        if not vp:
            raise SystemExit(f"MISSING_VIDEO_PATH: jid={jid}")
        if index_path and index_path != vp:
            raise SystemExit(f"VIDEO_PATH_INDEX_RUN_MISMATCH: jid={jid} run={vp} index={index_path}")

        # ── Symlink-safe evidence verification ─────────────────────────
        source_video = guard_path_safe(vp, evidence_root, f"VIDEO_{jid}")
        if not source_video.is_file():
            raise SystemExit(f"VIDEO_NOT_FOUND: {source_video}")
        actual_sha = sha256_file(source_video)
        if actual_sha != declared_sha:
            raise SystemExit(f"VIDEO_SHA_MISMATCH: {vp} declared={declared_sha[:16]} actual={actual_sha[:16]}")

        # ── Generate blind-safe name ───────────────────────────────────
        ext = Path(vp).suffix
        if ext.lower() not in MANDATORY_EXTENSIONS:
            ext = ".mp4"

        while True:
            blind_id = f"B{rng.randint(10000, 99999)}"
            if blind_id not in blind_ids: break
        blind_ids.add(blind_id)

        blind_name = f"{blind_id}{ext}"
        blind_target = blind_videos_dir / blind_name

        try:
            os.link(source_video, blind_target)
        except OSError:
            shutil.copy2(source_video, blind_target)

        blind_video_sha = sha256_file(blind_target)
        if blind_video_sha != declared_sha:
            raise SystemExit(f"BLIND_COPY_SHA_MISMATCH: {blind_name} expected={declared_sha[:16]} got={blind_video_sha[:16]}")

        blind_entries.append({
            "blind_id": blind_id,
            "video_file": blind_name,
            "video_sha256": blind_video_sha,
            "reviewer_a_labels": list(REVIEW_LABELS),
            "reviewer_b_labels": list(REVIEW_LABELS),
        })

        unblind_entries.append({
            "blind_id": blind_id, "job_id": jid,
            "parent_id": pid, "condition": cond,
        })

    # ── Verify blind package does NOT leak condition info ─────────────────
    blind_json_str = json.dumps(blind_entries, sort_keys=True).lower()
    for fb in FORBIDDEN_IN_BLIND:
        if fb.lower() in blind_json_str:
            raise SystemExit(f"BLIND_LEAK_DETECTED: '{fb}' found in blind package content")

    # ── Write blind package (reviewer-facing, NO condition info) ──────────
    with open(blind_root / "PILOT_BLIND_REVIEW_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "video_file"])
        w.writeheader()
        for e in blind_entries:
            w.writerow({"blind_id": e["blind_id"], "video_file": e["video_file"]})

    (blind_root / "PILOT_BLIND_REVIEW_V0.json").write_text(
        json.dumps({"entries": blind_entries, "n_videos": len(blind_entries),
                     "n_reviewers": 2, "labels": list(REVIEW_LABELS),
                     "instructions": "Review each video for the listed labels. Do not attempt to identify groups."},
                   indent=2, sort_keys=True) + "\n")
    seal_dir_in_place(blind_root)

    # ── Write unblinding root (separate, NOT shared with reviewer) ────────
    unblind_root.mkdir(parents=True)
    with open(unblind_root / "PILOT_UNBLINDING_V0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "job_id", "parent_id", "condition"])
        w.writeheader()
        for e in unblind_entries:
            w.writerow(e)

    (unblind_root / "PILOT_UNBLINDING_V0.json").write_text(
        json.dumps({"entries": unblind_entries, "n_entries": len(unblind_entries),
                     "execution_validation_seal": exec_val_seal},
                   indent=2) + "\n")
    seal_dir_in_place(unblind_root)

    print(f"Blind Review: blind={blind_root} ({len(blind_entries)} videos, {len(blind_entries)} linked/copied)")
    print(f"  Unblinding: {unblind_root} (SEPARATE — do not share with reviewer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
