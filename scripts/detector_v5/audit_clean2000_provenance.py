#!/usr/bin/env python3
"""Stage 0: CLEAN2000 provenance audit + registry CSV builder.

Verifies every episode's artifact closure, runtime validity, and CLEAN condition.
Produces a registry CSV with recursive SHA binding consumable by the teacher builder.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

SELF_SHA = None
EXPECTED_SUITES = ("Spatial", "Object", "Goal", "LIBERO-10")
TASKS_PER_SUITE = 10
STATES_PER_TASK = 50
EXPECTED_TOTAL = 2000

IDENTITY_SPLITS: dict[str, tuple[int, int]] = {
    "FIT_TRAIN": (0, 19),
    "FIT_DEV":   (20, 23),
    "CAL":       (24, 26),
    "CHECK":     (27, 29),
    "H":         (30, 34),
    "A":         (35, 44),
    "FEC":       (45, 49),
}

# Files required by the teacher builder's _verify_source_artifact()
REQUIRED_FILES = [
    "episode_metadata.json",
    "runtime_audit.json",
    "artifact_sha256.json",
    "step_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
]

REQUIRED_PROTOCOL_FIELDS = [
    "protocol_id", "episodes", "suites", "tasks_per_suite", "states_per_task",
    "official_seed", "num_steps_wait", "condition", "generation_policy",
]


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def _assign_split(state: int) -> str:
    for split_name, (lo, hi) in IDENTITY_SPLITS.items():
        if lo <= state <= hi:
            return split_name
    return "UNKNOWN"


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
    identity_counts: dict[str, int] = {k: 0 for k in IDENTITY_SPLITS}
    runtime_invalid: list[str] = []
    not_clean: list[str] = []

    # ── Verify official clean protocol ────────────────────────────────
    protocol_path = root / "provenance" / "OFFICIAL_PROTOCOL_CONFIG_V1.json"
    if not protocol_path.is_file():
        errors.append(f"PROTOCOL_MISSING: {protocol_path}")
        protocol_sha = None
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_sha = sha256_file(protocol_path)
        for fld in REQUIRED_PROTOCOL_FIELDS:
            if fld not in protocol:
                errors.append(f"PROTOCOL_MISSING_FIELD: {fld}")
        if protocol.get("episodes") != EXPECTED_TOTAL:
            errors.append(f"PROTOCOL_EPISODE_COUNT: expected={EXPECTED_TOTAL} got={protocol.get('episodes')}")
        if protocol.get("condition") != "CLEAN":
            errors.append(f"PROTOCOL_NOT_CLEAN: {protocol.get('condition')}")
        if protocol.get("official_seed") != 7:
            errors.append(f"PROTOCOL_SEED: expected=7 got={protocol.get('official_seed')}")

    # ── Walk episode directories ─────────────────────────────────────
    for suite in EXPECTED_SUITES:
        suite_dir = root / suite
        if not suite_dir.is_dir():
            errors.append(f"SUITE_MISSING: {suite}")
            continue
        for task_num in range(TASKS_PER_SUITE):
            task_name = f"task_{task_num:02d}"
            task_dir = suite_dir / task_name
            if not task_dir.is_dir():
                errors.append(f"TASK_MISSING: {suite}/{task_name}")
                continue
            for state in range(STATES_PER_TASK):
                state_dir = task_dir / f"state_{state}"
                eid = f"{suite}/{task_name}/state_{state}"
                split = _assign_split(state)

                if not state_dir.is_dir():
                    errors.append(f"STATE_MISSING: {eid}")
                    continue

                # Verify all required files exist
                missing = [f for f in REQUIRED_FILES if not (state_dir / f).is_file()]
                if missing:
                    errors.append(f"FILES_MISSING: {eid} missing={missing}")
                    continue

                # Read metadata for condition/runtime checks
                metadata = json.loads((state_dir / "episode_metadata.json").read_text(encoding="utf-8"))
                if metadata.get("condition") != "CLEAN":
                    not_clean.append(eid)
                    continue
                if not metadata.get("runtime_valid", False):
                    runtime_invalid.append(eid)
                    continue

                # Read recursive SHA from artifact manifest
                artifact_manifest = json.loads((state_dir / "artifact_sha256.json").read_text(encoding="utf-8"))
                recursive_sha = artifact_manifest.get("recursive_sha256", "")
                if not isinstance(recursive_sha, str) or len(recursive_sha) != 64:
                    errors.append(f"RECURSIVE_SHA_MISSING: {eid}")
                    continue

                identity_counts[split] = identity_counts.get(split, 0) + 1

                registry_rows.append({
                    "canonical_parent_key": eid,
                    "suite": suite,
                    "task_idx": task_num,
                    "state_id": state,
                    "split": split,
                    "formal_selected": True,
                    "selected_artifact_root": str(state_dir),
                    "selected_artifact_recursive_sha256": recursive_sha,
                })

    # ── Accumulate condition errors ───────────────────────────────────
    if not_clean:
        errors.append(f"NOT_CLEAN: {len(not_clean)} episodes — {not_clean[:5]}")
    if runtime_invalid:
        errors.append(f"RUNTIME_INVALID: {len(runtime_invalid)} episodes — {runtime_invalid[:5]}")

    # ── Validate counts ──────────────────────────────────────────────
    n_found = len(registry_rows)
    if n_found != EXPECTED_TOTAL:
        errors.append(f"EPISODE_COUNT: expected={EXPECTED_TOTAL} actual={n_found}")

    for split_name, (lo, hi) in IDENTITY_SPLITS.items():
        expected = (hi - lo + 1) * TASKS_PER_SUITE * len(EXPECTED_SUITES)
        actual = identity_counts.get(split_name, 0)
        if actual != expected:
            errors.append(f"SPLIT_COUNT: {split_name} expected={expected} actual={actual}")

    # A+FEC = 600
    a_fec = identity_counts.get("A", 0) + identity_counts.get("FEC", 0)
    if a_fec != 600:
        errors.append(f"A_FEC_COUNT: expected=600 actual={a_fec}")

    # ── Write registry CSVs ──────────────────────────────────────────
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
    for split_name in IDENTITY_SPLITS:
        split_rows = [r for r in registry_rows if r["split"] == split_name]
        csv_path = staging / f"CLEAN2000_REGISTRY_{split_name}_V1.csv"  # noqa
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
        "n_expected": EXPECTED_TOTAL,
        "identity_counts": identity_counts,
        "a_fec_count": a_fec,
        "protocol_sha256": protocol_sha,
        "n_runtime_invalid": len(runtime_invalid),
        "n_not_clean": len(not_clean),
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
