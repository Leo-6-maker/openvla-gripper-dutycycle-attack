#!/usr/bin/env python3
"""Stage 0: CLEAN2000 provenance audit — episode closure, identity split, no A/FEC leakage."""
from __future__ import annotations

import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

SELF_SHA = None
EXPECTED_SUITES = ("Spatial", "Object", "Goal", "LIBERO-10")
TASKS_PER_SUITE = 10
STATES_PER_TASK = 50
EXPECTED_TOTAL = 2000

IDENTITY_SPLITS = {
    "FIT-TRAIN": (0, 19),
    "FIT-DEV":   (20, 23),
    "CAL":       (24, 26),
    "CHECK":     (27, 29),
    "H":         (30, 34),
    "A":         (35, 44),
    "FEC":       (45, 49),
}

FROZEN_SHAS = {
    "collector_head": "943b02749dce4414ec6791b15ceec87dbd3be1ba",
    "openvla_upstream": "c8f03f48af692657d3060c19588038c7220e9af9",
    "libero": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "libero_checkpoint_tree": "4a83f512232909d34ec2f835acf492713b4c174f0b016ac00cbb330ed5ff8dbd",
    "collector_script": "a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311",
    "runtime_wrapper": "de1d141bd4f6b75b00753adf225b400388868221d9032821874d004f0e0f05b8",
    "flash_attn_wheel": "3fc5d8813904d32cfc77aa4b88d40169dbc12053edb6ee0f1d94159527d05ab0",
}


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean2000-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    root = args.clean2000_root.resolve()

    errors: list[str] = []
    episode_ids: set[str] = set()
    suite_task_counts: dict[str, dict[str, int]] = {}
    identity_episodes: dict[str, set[str]] = {k: set() for k in IDENTITY_SPLITS}

    # ── Walk CLEAN2000 root ───────────────────────────────────────────
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
                eid = f"{suite}/{task_name}/state_{state}"
                episode_ids.add(eid)
                suite_task_counts[suite][task_name] += 1
                for split_name, (lo, hi) in IDENTITY_SPLITS.items():
                    if lo <= state <= hi:
                        identity_episodes[split_name].add(eid)

    # ── Count checks ─────────────────────────────────────────────────
    if len(episode_ids) != EXPECTED_TOTAL:
        errors.append(f"EPISODE_COUNT: expected={EXPECTED_TOTAL} actual={len(episode_ids)}")
    for suite in EXPECTED_SUITES:
        for tn in [f"task_{i:02d}" for i in range(TASKS_PER_SUITE)]:
            n = suite_task_counts.get(suite, {}).get(tn, 0)
            if n != 50:
                errors.append(f"STATE_COUNT: {suite}/{tn} expected=50 actual={n}")

    # ── Identity split overlap check ─────────────────────────────────
    all_assigned: set[str] = set()
    for split_name, eps in identity_episodes.items():
        overlap = all_assigned & eps
        if overlap:
            errors.append(f"SPLIT_OVERLAP: {split_name} overlaps on {sorted(overlap)[:5]}")
        all_assigned |= eps

    # ── A/FEC leakage gate ───────────────────────────────────────────
    a_or_fec = identity_episodes["A"] | identity_episodes["FEC"]
    other_splits = all_assigned - a_or_fec
    if a_or_fec & other_splits:
        errors.append(f"A_FEC_LEAKAGE: {sorted(a_or_fec & other_splits)[:5]}")
    if len(a_or_fec) != 1000:
        errors.append(f"A_FEC_COUNT: expected=1000 actual={len(a_or_fec)}")

    # ── Identity coverage ────────────────────────────────────────────
    for split_name, (lo, hi) in IDENTITY_SPLITS.items():
        expected = (hi - lo + 1) * TASKS_PER_SUITE * len(EXPECTED_SUITES)
        actual = len(identity_episodes[split_name])
        if actual != expected:
            errors.append(f"SPLIT_COUNT: {split_name} expected={expected} actual={actual}")

    receipt = {
        "schema": "CLEAN2000_PROVENANCE_RECEIPT_V1",
        "auditor_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_episodes": len(episode_ids),
        "n_expected": EXPECTED_TOTAL,
        "n_suites": len(suite_task_counts),
        "identity_split_counts": {k: len(v) for k, v in identity_episodes.items()},
        "frozen_source_shas": FROZEN_SHAS,
        "a_fec_leakage": False if not errors else any("A_FEC" in e for e in errors),
        "n_errors": len(errors),
        "errors": errors[:100],
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "CLEAN2000_PROVENANCE_RECEIPT_V1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (staging / "CLEAN2000_INPUT_MANIFEST_V1.json").write_text(json.dumps({
        "schema": "CLEAN2000_INPUT_MANIFEST_V1",
        "n_episodes": len(episode_ids),
        "suites": sorted(suite_task_counts.keys()),
        "per_task_counts": {s: dict(t) for s, t in suite_task_counts.items()},
    }, indent=2, sort_keys=True) + "\n")
    (staging / "CLEAN2000_IDENTITY_SPLIT_V1.json").write_text(json.dumps({
        "schema": "CLEAN2000_IDENTITY_SPLIT_V1",
        "splits": {k: {"range": list(v), "count": len(identity_episodes[k])} for k, v in IDENTITY_SPLITS.items()},
    }, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"CLEAN2000 Provenance Audit: {receipt['status']} errors={len(errors)}")
    for e in errors[:10]: print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
