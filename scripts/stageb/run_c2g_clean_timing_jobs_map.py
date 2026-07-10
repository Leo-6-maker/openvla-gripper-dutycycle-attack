#!/usr/bin/env python3
"""Run detector-only CLEAN timing jobs with exact per-suite OpenVLA models."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "scripts" / "stageb" / "run_c2g_clean_timing_jobs.py"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("parents", value.get("episodes", value)) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("parent manifest must contain a list of objects")
    return [dict(row) for row in rows]


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
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    rows = read_rows(args.parents.resolve())
    model_map = read_model_map(args.suite_model_map.resolve())
    results: list[dict[str, Any]] = []
    total_launched = 0
    with tempfile.TemporaryDirectory(prefix="c2g_timing_map_") as td:
        temporary = Path(td)
        for suite in SUITES:
            suite_rows = [row for row in rows if str(row.get("suite")) == suite]
            if not suite_rows:
                continue
            remaining = 0 if args.max_jobs <= 0 else max(0, args.max_jobs - total_launched)
            if args.max_jobs > 0 and remaining == 0:
                break
            manifest = temporary / f"{suite}.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in suite_rows),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(BASE),
                "--parents", str(manifest),
                "--checkpoint", args.checkpoint,
                "--output-root", str(args.output_root),
                "--expected-git-commit", args.expected_git_commit,
                "--device", args.device,
                "--model-path", model_map[suite],
                "--burst-length", str(args.burst_length),
                "--resume" if args.resume else "--no-resume",
            ]
            if remaining > 0:
                command.extend(["--max-jobs", str(remaining)])
            if args.dry_run:
                command.append("--dry-run")
            completed = subprocess.run(command, cwd=REPO)
            results.append(
                {
                    "suite": suite,
                    "parent_count": len(suite_rows),
                    "returncode": completed.returncode,
                    "status": "PASS" if completed.returncode == 0 else "HOLD",
                }
            )
            if completed.returncode != 0:
                break
            total_launched += min(len(suite_rows), remaining) if remaining > 0 else len(suite_rows)

    status = (
        "PASS_C2G_CLEAN_TIMING_MODEL_MAP"
        if results and all(row["status"] == "PASS" for row in results)
        else "HOLD_C2G_CLEAN_TIMING_MODEL_MAP"
    )
    report = {
        "gate": "C2G_CLEAN_TIMING_MODEL_MAP",
        "status": status,
        "suite_model_map": str(args.suite_model_map.resolve()),
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "c2g_clean_timing_model_map_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
