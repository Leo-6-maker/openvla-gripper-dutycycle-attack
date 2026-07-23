#!/usr/bin/env python3
"""Stage 1: Build unified Factorized Teacher labels for all 5 detector splits.

Reuses the core derive_factorized_rows from gripper_attack.v5_factorized_teacher.
Accepts --target-split to build labels for FIT_TRAIN, FIT_DEV, CAL, CHECK, or H.
Calls the existing builder infrastructure with a split-filtered registry CSV.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None

EXPECTED_COUNTS = {
    "FIT_TRAIN": 800,
    "FIT_DEV": 160,
    "CAL": 120,
    "CHECK": 120,
    "H": 200,
}

SOURCE_SCRIPTS = ROOT / "scripts/detector_v5"
TEACHER_BUILDER = SOURCE_SCRIPTS / "build_v5_factorized_teacher.py"


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def _filter_registry(input_csv: Path, output_csv: Path, target_split: str) -> int:
    """Read registry CSV, keep only rows matching target_split, write filtered copy."""
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("split") == target_split]

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean2000-audit-root", type=Path, required=True,
                    help="Stage 0 output root (contains CLEAN2000_REGISTRY_V1.csv)")
    ap.add_argument("--target-split", type=str, default=None,
                    choices=list(EXPECTED_COUNTS.keys()),
                    help="Build labels for a single split (omit to build all 5)")
    ap.add_argument("--output-root", type=Path, required=True)
    # Pre-existing frozen artifacts (required by teacher builder)
    ap.add_argument("--registry-root", type=Path, required=True,
                    help="Root for resolving registry-relative paths")
    ap.add_argument("--decoder-root", type=Path, required=True)
    ap.add_argument("--physics-audit-root", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True,
                    help="OFFICIAL_PROTOCOL_CONFIG_V1.json path")
    ap.add_argument("--k10-root", type=Path, required=True)
    ap.add_argument("--expected-k10-schema", type=str, required=True)
    ap.add_argument("--expected-source-commit", type=str, default=None)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # Verify Stage 0 audit passed
    audit_json = args.clean2000_audit_root / "CLEAN2000_PROVENANCE_RECEIPT_V1.json"
    if not audit_json.is_file():
        raise SystemExit(f"AUDIT_RECEIPT_MISSING: {audit_json}")
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise SystemExit(f"AUDIT_NOT_PASS: status={audit.get('status')}")

    registry_csv = args.clean2000_audit_root / "CLEAN2000_REGISTRY_V1.csv"
    if not registry_csv.is_file():
        raise SystemExit(f"REGISTRY_CSV_MISSING: {registry_csv}")

    # Verify pre-existing artifacts
    for label, path in [("registry-root", args.registry_root),
                         ("decoder-root", args.decoder_root),
                         ("physics-audit-root", args.physics_audit_root),
                         ("protocol", args.protocol),
                         ("k10-root", args.k10_root)]:
        if not path.is_dir() if Path(path).suffix == "" else not Path(path).is_file():
            if not Path(path).exists():
                raise SystemExit(f"ARTIFACT_NOT_FOUND: {label}={path}")

    # ── Determine splits to build ────────────────────────────────────
    splits_to_build = [args.target_split] if args.target_split else list(EXPECTED_COUNTS.keys())

    errors: list[str] = []
    split_seals: dict[str, str] = {}
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    for split_name in splits_to_build:
        expected_n = EXPECTED_COUNTS[split_name]
        split_dir = staging / split_name.lower().replace("-", "_")
        split_csv = staging / f"registry_{split_name.lower()}.csv"

        # Filter registry for this split
        actual_n = _filter_registry(registry_csv, split_csv, split_name)
        if actual_n != expected_n:
            errors.append(f"SPLIT_COUNT_MISMATCH: {split_name} expected={expected_n} actual={actual_n}")
            continue

        # Call teacher builder
        cmd = [sys.executable, str(TEACHER_BUILDER),
               "--registry-csv", str(split_csv),
               "--registry-root", str(args.registry_root),
               "--decoder-root", str(args.decoder_root),
               "--physics-audit-root", str(args.physics_audit_root),
               "--protocol", str(args.protocol),
               "--k10-root", str(args.k10_root),
               "--expected-k10-schema", args.expected_k10_schema,
               "--target-split", split_name,
               "--output-root", str(split_dir)]
        if args.expected_source_commit:
            cmd.extend(["--expected-source-commit", args.expected_source_commit])

        print(f"Building Teacher for {split_name} ({actual_n} rows)...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            errors.append(f"TEACHER_BUILD_FAILED: {split_name}")
            continue

        # Capture seal
        sums = split_dir / "SHA256SUMS"
        if sums.is_file():
            split_seals[split_name] = sha256_file(sums)

    if errors:
        raise SystemExit("STAGE_1_FAILED: " + "; ".join(errors))

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "UNIFIED_TEACHER_LABELS_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "PASS",
        "labeled_splits": splits_to_build,
        "split_seals": split_seals,
        "a_fec_labeled": False,
        "uses_attack_outcome": False,
        "teacher_builder": str(TEACHER_BUILDER),
    }
    (staging / "UNIFIED_TEACHER_LABELS_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # ── Seal atomically ──────────────────────────────────────────────
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Stage 1 complete: {len(splits_to_build)} splits sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
