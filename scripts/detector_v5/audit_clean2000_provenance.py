#!/usr/bin/env python3
"""Stage 0: CLEAN2000 provenance audit — real episode traversal, protocol verification, identity closure."""
from __future__ import annotations

import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

SELF_SHA = None
EXPECTED_SUITES = ("Spatial", "Object", "Goal", "LIBERO-10")
TASKS_PER_SUITE = 10
STATES_PER_TASK = 50
EXPECTED_TOTAL = 2000

IDENTITY_SPLITS: dict[str, tuple[int, int]] = {
    "FIT-TRAIN": (0, 19),
    "FIT-DEV":   (20, 23),
    "CAL":       (24, 26),
    "CHECK":     (27, 29),
    "H":         (30, 34),
    "A":         (35, 44),
    "FEC":       (45, 49),
}


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
    episodes: list[dict[str, Any]] = []
    identity_episodes: dict[str, list[str]] = {k: [] for k in IDENTITY_SPLITS}
    suite_task_counts: dict[str, dict[str, int]] = {}
    file_missing: list[str] = []

    # ── Verify protocol file ──────────────────────────────────────────
    protocol_path = root / "provenance" / "OFFICIAL_PROTOCOL_CONFIG_V1.json"
    if not protocol_path.is_file():
        errors.append(f"PROTOCOL_MISSING: {protocol_path}")
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("episodes") != EXPECTED_TOTAL:
            errors.append(f"PROTOCOL_EPISODE_COUNT: expected={EXPECTED_TOTAL} got={protocol.get('episodes')}")
        protocol_sha = sha256_file(protocol_path)

    # ── Walk episode directories ─────────────────────────────────────
    for suite in EXPECTED_SUITES:
        suite_dir = root / suite
        if not suite_dir.is_dir():
            errors.append(f"SUITE_MISSING: {suite}")
            continue
        suite_task_counts.setdefault(suite, {})
        for task_num in range(TASKS_PER_SUITE):
            task_name = f"task_{task_num:02d}"
            task_dir = suite_dir / task_name
            if not task_dir.is_dir():
                errors.append(f"TASK_MISSING: {suite}/{task_name}")
                continue
            suite_task_counts[suite][task_name] = 0
            for state in range(STATES_PER_TASK):
                state_dir = task_dir / f"state_{state}"
                eid = f"{suite}/{task_name}/state_{state}"

                if not state_dir.is_dir():
                    errors.append(f"STATE_MISSING: {eid}")
                    continue

                suite_task_counts[suite][task_name] += 1

                # Check required files exist
                required_files = ["metadata.json", "step_records.jsonl"]
                missing_files = [f for f in required_files if not (state_dir / f).is_file()]
                if missing_files:
                    file_missing.append(f"{eid}: {missing_files}")
                    continue

                ep = {
                    "episode_id": eid,
                    "suite": suite,
                    "task": task_name,
                    "state": state,
                    "metadata_sha256": sha256_file(state_dir / "metadata.json"),
                    "steps_sha256": sha256_file(state_dir / "step_records.jsonl"),
                }
                episodes.append(ep)

                for split_name, (lo, hi) in IDENTITY_SPLITS.items():
                    if lo <= state <= hi:
                        identity_episodes[split_name].append(eid)

    # ── Validate counts ──────────────────────────────────────────────
    n_found = len(episodes)
    if n_found != EXPECTED_TOTAL:
        errors.append(f"EPISODE_COUNT: expected={EXPECTED_TOTAL} actual={n_found}")

    for suite in EXPECTED_SUITES:
        for tn in [f"task_{i:02d}" for i in range(TASKS_PER_SUITE)]:
            n = suite_task_counts.get(suite, {}).get(tn, 0)
            if n != STATES_PER_TASK:
                errors.append(f"STATE_COUNT: {suite}/{tn} expected={STATES_PER_TASK} actual={n}")

    # ── Identity split validation ────────────────────────────────────
    all_eids: set[str] = set()
    for split_name, eids in identity_episodes.items():
        lo, hi = IDENTITY_SPLITS[split_name]
        expected_n = (hi - lo + 1) * TASKS_PER_SUITE * len(EXPECTED_SUITES)
        if len(eids) != expected_n:
            errors.append(f"SPLIT_COUNT: {split_name} expected={expected_n} actual={len(eids)}")
        overlap = all_eids & set(eids)
        if overlap:
            errors.append(f"SPLIT_OVERLAP: {split_name} overlaps on {sorted(overlap)[:5]}")
        all_eids.update(eids)

    # ── A+FEC = 600 (10+5)*40 ────────────────────────────────────────
    a_fec_eids = set(identity_episodes["A"]) | set(identity_episodes["FEC"])
    expected_afec = (10 + 5) * 40  # (44-35+1 + 49-45+1) * 4*10 = 15*40 = 600
    if len(a_fec_eids) != expected_afec:
        errors.append(f"A_FEC_COUNT: expected={expected_afec} actual={len(a_fec_eids)}")

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "CLEAN2000_PROVENANCE_RECEIPT_V1",
        "auditor_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_episodes": n_found,
        "n_expected": EXPECTED_TOTAL,
        "n_suites": len(suite_task_counts),
        "identity_split_counts": {k: len(v) for k, v in identity_episodes.items()},
        "a_fec_count": len(a_fec_eids),
        "protocol_sha256": protocol_sha if 'protocol_sha' in dir() else None,
        "n_file_missing": len(file_missing),
        "n_errors": len(errors),
        "errors": errors[:200],
        "file_missing_sample": file_missing[:20],
    }

    # ── Atomic sealed output ─────────────────────────────────────────
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "CLEAN2000_PROVENANCE_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (staging / "CLEAN2000_EPISODE_MANIFEST_V1.json").write_text(
        json.dumps({"schema": "CLEAN2000_EPISODE_MANIFEST_V1", "n_episodes": n_found,
                    "episodes": episodes, "identity_splits": IDENTITY_SPLITS},
                   indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
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
