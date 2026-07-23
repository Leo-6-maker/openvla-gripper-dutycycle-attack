#!/usr/bin/env python3
"""Stage 1: Build unified Factorized Teacher labels for FIT-TRAIN/DEV/CAL/CHECK/H.

Uses the exact same Teacher builder version for states 0-34.
A/FEC states 35-49 are NOT labeled. Output is sealed per split.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

SELF_SHA = None
LABEL_SPLITS = ("FIT-TRAIN", "FIT-DEV", "CAL", "CHECK", "H")

# State ranges by split (inclusive)
STATE_RANGES = {
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
    ap.add_argument("--clean2000-audit-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # Verify audit passed
    audit_receipt = json.loads((args.clean2000_audit_root / "CLEAN2000_PROVENANCE_RECEIPT_V1.json").read_text())
    if audit_receipt.get("status") != "PASS":
        raise SystemExit("AUDIT_NOT_PASS: cannot build teacher labels on failed audit")

    # For each split, invoke the factorized teacher builder
    # This delegates to the existing build_v5_factorized_teacher.py infrastructure
    import subprocess
    scripts = ROOT / "scripts/detector_v5"
    errors: list[str] = []
    split_roots: dict[str, Path] = {}

    for split in LABEL_SPLITS:
        split_root = out_root / split.lower()
        print(f"\nBuilding Teacher labels for {split} (states {STATE_RANGES[split]})...")
        result = subprocess.run(
            [sys.executable, str(scripts / "build_v5_factorized_teacher.py"),
             "--clean2000-root", str(args.clean2000_audit_root.parent.parent),
             "--split", split,
             "--state-range", f"{STATE_RANGES[split][0]}-{STATE_RANGES[split][1]}",
             "--output-root", str(split_root)],
            capture_output=False)
        if result.returncode != 0:
            errors.append(f"TEACHER_BUILD_FAILED: {split}")
            continue
        split_roots[split] = split_root

    if errors:
        raise SystemExit("STAGE_1_FAILED: " + "; ".join(errors))

    # ── Build Stage 1 receipt ────────────────────────────────────────
    receipt = {
        "schema": "UNIFIED_TEACHER_LABELS_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "PASS",
        "labeled_splits": LABEL_SPLITS,
        "split_roots": {s: str(split_roots[s]) for s in LABEL_SPLITS if s in split_roots},
        "a_fec_labeled": False,
        "teacher_builder_version": "build_v5_factorized_teacher.py",
        "uses_attack_outcome": False,
        "uses_A_pool": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "UNIFIED_TEACHER_LABELS_RECEIPT_V1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
