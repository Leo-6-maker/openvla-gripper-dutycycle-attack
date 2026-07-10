#!/usr/bin/env python3
"""Execute strict matched-load jobs with exact per-suite OpenVLA models."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_matched_load_manifest import validate_core_2x2_manifest

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "scripts" / "stageb" / "run_c2g_matched_load_jobs_release.py"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("jobs file must contain JSON objects")
    return rows


def read_model_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("suite model map must be a JSON object")
    output = {suite: str(value.get(suite, "")).strip() for suite in SUITES}
    missing = [suite for suite, model_path in output.items() if not model_path]
    if missing:
        raise ValueError("suite model map missing: " + ", ".join(missing))
    for suite, model_path in output.items():
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"{suite} model directory missing: {model_path}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs.resolve())
    validate_core_2x2_manifest(jobs, strict_objective_seed_pairing=True)
    model_map = read_model_map(args.suite_model_map.resolve())
    results: list[dict[str, Any]] = []
    launched = 0
    with tempfile.TemporaryDirectory(prefix="c2g_release_map_") as td:
        temporary = Path(td)
        for suite in SUITES:
            suite_jobs = [row for row in jobs if str(row.get("suite")) == suite]
            if not suite_jobs:
                continue
            validate_core_2x2_manifest(
                suite_jobs,
                strict_objective_seed_pairing=True,
            )
            remaining = 0 if args.max_jobs <= 0 else max(0, args.max_jobs - launched)
            if args.max_jobs > 0 and remaining == 0:
                break
            manifest = temporary / f"{suite}.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in suite_jobs),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(RELEASE),
                "--jobs", str(manifest),
                "--output-root", str(args.output_root),
                "--checkpoint", args.checkpoint,
                "--expected-git-commit", args.expected_git_commit,
                "--device", args.device,
                "--model-path", model_map[suite],
                "--max-steps", str(args.max_steps),
                "--resume" if args.resume else "--no-resume",
            ]
            if suite == "libero_goal":
                command.extend(
                    ["--policy-model-manifest", str(args.goal_model_manifest.resolve())]
                )
            if remaining > 0:
                command.extend(["--max-jobs", str(remaining)])
            if args.dry_run:
                command.append("--dry-run")
            completed = subprocess.run(command, cwd=REPO)
            results.append(
                {
                    "suite": suite,
                    "job_count": len(suite_jobs),
                    "returncode": completed.returncode,
                    "status": "PASS" if completed.returncode == 0 else "HOLD",
                }
            )
            if completed.returncode != 0:
                break
            launched += min(len(suite_jobs), remaining) if remaining > 0 else len(suite_jobs)

    status = (
        "PASS_C2G_RELEASE_MODEL_MAP"
        if results and all(row["status"] == "PASS" for row in results)
        else "HOLD_C2G_RELEASE_MODEL_MAP"
    )
    report = {
        "gate": "C2G_RELEASE_MODEL_MAP",
        "status": status,
        "suite_model_map": str(args.suite_model_map.resolve()),
        "goal_model_manifest": str(args.goal_model_manifest.resolve()),
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "c2g_release_model_map_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
