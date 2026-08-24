#!/usr/bin/env python3
"""Stage 0: CLEAN2000 provenance audit.

Reads the official manifest CSV from the CLEAN2000 root. Verifies every
episode's artifact directory, required files, and recursive SHA. Produces
a registry CSV consumable by the teacher builder.

Real server structure (as verified 2026-07-24):
  clean/libero_spatial/task_00/state_00/  (libero_ prefix, zero-padded state)
  manifests/OFFICIAL_CLEAN_2000_MANIFEST_V3.csv
  provenance/OFFICIAL_PROTOCOL_CONFIG_V1.json
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

SELF_SHA = None

REQUIRED_FILES = [
    "episode_metadata.json",
    "runtime_audit.json",
    "artifact_sha256.json",
    "step_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
]


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean2000-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    root = args.clean2000_root.resolve()
    if not root.is_dir(): raise SystemExit(f"CLEAN2000_ROOT_NOT_DIR: {root}")

    errors: list[str] = []
    registry_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    # ── Read official manifest CSV ────────────────────────────────────
    manifest_csv = root / "manifests" / "OFFICIAL_CLEAN_2000_MANIFEST_V3.csv"
    if not manifest_csv.is_file():
        raise SystemExit(f"MANIFEST_MISSING: {manifest_csv}")

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    if len(manifest_rows) != 2000:
        errors.append(f"MANIFEST_COUNT: expected=2000 actual={len(manifest_rows)}")

    # ── Verify protocol ──────────────────────────────────────────────
    protocol_path = root / "provenance" / "OFFICIAL_PROTOCOL_CONFIG_V1.json"
    protocol = {}
    protocol_sha = None
    if not protocol_path.is_file():
        errors.append(f"PROTOCOL_MISSING: {protocol_path}")
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_sha = sha256_file(protocol_path)

    # ── Verify official manifest SHA file ────────────────────────────
    manifest_sha_path = root / "manifests" / "OFFICIAL_CLEAN_2000_MANIFEST_V3.csv.sha256"
    if manifest_sha_path.is_file():
        declared = manifest_sha_path.read_text(encoding="utf-8").strip().split()[0]
        actual = sha256_file(manifest_csv)
        if declared != actual:
            errors.append(f"MANIFEST_SHA_MISMATCH: declared={declared[:16]} actual={actual[:16]}")

    # ── Verify each episode ──────────────────────────────────────────
    clean_dir = root / "clean"
    if not clean_dir.is_dir():
        raise SystemExit(f"CLEAN_DIR_MISSING: {clean_dir}")

    for row in manifest_rows:
        suite = row["suite"]
        task_idx = int(row["task_idx"])
        state_id = int(row["state_id"])
        split = row["split"]
        eid = row["canonical_parent_key"]

        # Build path: clean/libero_spatial/task_00/state_00/
        state_dir = clean_dir / suite / f"task_{task_idx:02d}" / f"state_{state_id:02d}"

        if not state_dir.is_dir():
            errors.append(f"STATE_MISSING: {eid} expected={state_dir}")
            continue

        # Verify required files
        missing = [f for f in REQUIRED_FILES if not (state_dir / f).is_file()]
        if missing:
            errors.append(f"FILES_MISSING: {eid} missing={missing}")
            continue

        # Read recursive SHA
        artifact_json = state_dir / "artifact_sha256.json"
        artifact = json.loads(artifact_json.read_text(encoding="utf-8"))
        recursive_sha = artifact.get("recursive_sha256", "")
        if not isinstance(recursive_sha, str) or len(recursive_sha) != 64:
            errors.append(f"RECURSIVE_SHA_INVALID: {eid} sha={recursive_sha[:40]!r}")
            continue

        # Validate episode metadata
        metadata = json.loads((state_dir / "episode_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("condition") != "CLEAN":
            errors.append(f"NOT_CLEAN: {eid} condition={metadata.get('condition')}")
            continue

        split_counts[split] = split_counts.get(split, 0) + 1

        registry_rows.append({
            "canonical_parent_key": eid,
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "split": split,
            "formal_selected": True,
            "selected_artifact_root": str(state_dir),
            "selected_artifact_recursive_sha256": recursive_sha,
        })

    # ── Validate split counts ────────────────────────────────────────
    # Real split scheme: FIT=960, CAL=120, CHECK=120, FINAL_EVAL_CANDIDATE=800
    for split_name, expected in [("FIT", 960), ("CAL", 120), ("CHECK", 120),
                                   ("FINAL_EVAL_CANDIDATE", 800)]:
        actual = split_counts.get(split_name, 0)
        if actual != expected:
            errors.append(f"SPLIT_COUNT: {split_name} expected={expected} actual={actual}")

    n_found = len(registry_rows)
    if n_found != 2000:
        errors.append(f"EPISODE_COUNT: expected=2000 actual={n_found}")

    # ── Write registry CSV ───────────────────────────────────────────
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    fieldnames = ["canonical_parent_key", "suite", "task_idx", "state_id", "split",
                  "formal_selected", "selected_artifact_root",
                  "selected_artifact_recursive_sha256"]

    with open(staging / "CLEAN2000_REGISTRY_V1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(registry_rows, key=lambda r: r["canonical_parent_key"]):
            w.writerow(row)

    # Split-specific registries
    for split_name in split_counts:
        split_rows = [r for r in registry_rows if r["split"] == split_name]
        csv_path = staging / f"CLEAN2000_REGISTRY_{split_name}_V1.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in sorted(split_rows, key=lambda r: r["canonical_parent_key"]):
                w.writerow(row)

    # ── Receipt ──────────────────────────────────────────────────────
    receipt = {
        "schema": "CLEAN2000_PROVENANCE_RECEIPT_V1",
        "auditor_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_episodes": n_found,
        "n_expected": 2000,
        "split_counts": split_counts,
        "manifest_csv_sha256": sha256_file(manifest_csv),
        "protocol_sha256": protocol_sha,
        "protocol_id": protocol.get("protocol_id", "UNKNOWN"),
        "n_errors": len(errors),
        "errors": errors[:200],
    }
    (staging / "CLEAN2000_PROVENANCE_RECEIPT_V1.json").write_text(
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

    print(f"CLEAN2000 Audit: {receipt['status']} episodes={n_found} errors={len(errors)}")
    for e in errors[:10]: print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
