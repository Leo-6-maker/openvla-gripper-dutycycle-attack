#!/usr/bin/env python3
"""Build release matched-load jobs with CLEAN seed bound to clean execution.

The base builder already freezes attack timing, paired objective seeds, load, and
coverage. This release wrapper additionally sets each CLEAN row's objective_seed to
its preregistered eval_seed, which is the seed used by the detector-only CLEAN
worker. The attack rows remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from src.gripper_attack.c2g_matched_load_manifest import validate_core_2x2_manifest

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "scripts" / "stageb" / "build_c2g_matched_load_jobs.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("base job builder did not produce JSON objects")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--detector-timing", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=6.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=(6.0 / 255.0) * 0.075)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--control-objective", default="SHUFFLED_GRIPPER_GRADIENT")
    parser.add_argument("--require-all-parents-attackable", action="store_true")
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(BASE),
        "--parents", str(args.parents.resolve()),
        "--detector-timing", str(args.detector_timing.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--detector-config", str(args.detector_config.resolve()),
        "--output", str(args.output.resolve()),
        "--master-seed", str(args.master_seed),
        "--burst-length", str(args.burst_length),
        "--epsilon", str(args.epsilon),
        "--step-size", str(args.step_size),
        "--pgd-steps", str(args.pgd_steps),
        "--control-objective", args.control_objective,
    ]
    if args.require_all_parents_attackable:
        command.append("--require-all-parents-attackable")
    completed = subprocess.run(command, cwd=REPO)
    if completed.returncode != 0:
        return completed.returncode

    output = args.output.resolve()
    rows = read_jsonl(output)
    clean_count = 0
    for row in rows:
        if row["condition"] == "CLEAN":
            row["objective_seed"] = int(row["eval_seed"])
            clean_count += 1
    validation = validate_core_2x2_manifest(
        rows,
        strict_objective_seed_pairing=True,
    )
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = output.with_suffix(output.suffix + ".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "status": "PASS_C2G_RELEASE_MATCHED_LOAD_JOBS_BUILT",
            "jobs_sha256": sha256_file(output),
            "clean_seed_binding": "objective_seed_equals_eval_seed",
            "clean_row_count": clean_count,
            "validation": validation,
        }
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
