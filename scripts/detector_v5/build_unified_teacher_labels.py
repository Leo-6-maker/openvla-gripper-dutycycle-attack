#!/usr/bin/env python3
"""Stage 1: Build unified Factorized Teacher labels for FIT-TRAIN/DEV/CAL/CHECK/H.

Uses the exact same Teacher builder for states 0-34. Writes each split as a
sealed subdirectory under output-root, then seals the output root in place.
No data is deleted after generation.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None
LABEL_SPLITS = ("FIT-TRAIN", "FIT-DEV", "CAL", "CHECK", "H")
STATE_RANGES: dict[str, tuple[int, int]] = {
    "FIT-TRAIN": (0, 19),
    "FIT-DEV":   (20, 23),
    "CAL":       (24, 26),
    "CHECK":     (27, 29),
    "H":         (30, 34),
}
EXPECTED_SUITES = ("Spatial", "Object", "Goal", "LIBERO-10")


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean2000-root", type=Path, required=True)
    ap.add_argument("--clean2000-audit-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    clean2000 = args.clean2000_root.resolve()

    # Verify audit passed
    audit_json = args.clean2000_audit_root / "CLEAN2000_PROVENANCE_RECEIPT_V1.json"
    if not audit_json.is_file():
        raise SystemExit(f"AUDIT_RECEIPT_MISSING: {audit_json}")
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise SystemExit(f"AUDIT_NOT_PASS: status={audit.get('status')}")

    # Verify clean2000 root is a real directory with suites
    if not clean2000.is_dir():
        raise SystemExit(f"CLEAN2000_NOT_DIR: {clean2000}")
    for suite in EXPECTED_SUITES:
        if not (clean2000 / suite).is_dir():
            raise SystemExit(f"CLEAN2000_SUITE_MISSING: {suite}")

    # ── Build teacher labels for each split ──────────────────────────
    scripts = ROOT / "scripts/detector_v5"
    errors: list[str] = []
    split_seals: dict[str, str] = {}

    out_root.mkdir(parents=True)

    for split in LABEL_SPLITS:
        split_dir = out_root / split.lower().replace("-", "_")
        lo, hi = STATE_RANGES[split]
        print(f"\nBuilding Teacher labels for {split} (states {lo}-{hi})...")
        result = subprocess.run(
            [sys.executable, str(scripts / "build_v5_factorized_teacher.py"),
             "--clean2000-root", str(clean2000),
             "--split", split,
             "--state-range", f"{lo}-{hi}",
             "--output-root", str(split_dir)],
            capture_output=False)
        if result.returncode != 0:
            errors.append(f"TEACHER_BUILD_FAILED: {split}")
            continue
        # Read the seal of the split's output
        sums_file = split_dir / "SHA256SUMS"
        if sums_file.is_file():
            split_seals[split] = sha256_file(sums_file)
        else:
            errors.append(f"SPLIT_NOT_SEALED: {split}")

    if errors:
        raise SystemExit("STAGE_1_FAILED: " + "; ".join(errors))

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "UNIFIED_TEACHER_LABELS_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "PASS",
        "labeled_splits": list(LABEL_SPLITS),
        "state_ranges": STATE_RANGES,
        "split_seals": split_seals,
        "a_fec_labeled": False,
        "uses_attack_outcome": False,
        "clean2000_root": str(clean2000),
    }
    (out_root / "UNIFIED_TEACHER_LABELS_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # ── Seal in place (does NOT delete split subdirectories) ─────────
    names = sorted(p.relative_to(out_root).as_posix() for p in out_root.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (out_root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(out_root / name)}  {name}\n" for name in names))
    seal = sha256_file(out_root / "SHA256SUMS")
    (out_root / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")

    print(f"Stage 1 complete: {len(LABEL_SPLITS)} splits, seal={seal[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
