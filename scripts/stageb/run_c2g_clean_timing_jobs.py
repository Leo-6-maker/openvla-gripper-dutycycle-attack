#!/usr/bin/env python3
"""Run detector-only CLEAN passes for a preregistered evaluation parent manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "stageb" / "run_c2g_clean_window_vis_pgd.py"
PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("parents", value.get("episodes", value)) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("parent manifest must contain a list of objects")
    return [dict(row) for row in rows]


def normalized_parent(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("suite", "task_index", "state_id"):
        if field not in row:
            raise ValueError(f"parent row missing {field}")
    suite = str(row["suite"])
    task_index = int(row["task_index"])
    state_id = int(row["state_id"])
    parent_key = str(row.get("parent_key") or f"{suite}/task_{task_index}/state_{state_id}")
    return {
        "parent_key": parent_key,
        "suite": suite,
        "task_index": task_index,
        "state_id": state_id,
        "eval_seed": int(row.get("eval_seed", row.get("seed", 42))),
        "max_steps": int(row.get("max_steps", 300)),
    }


def complete(path: Path, parent: dict[str, Any], expected_commit: str) -> bool:
    step_path = path.with_name("step_records.jsonl")
    if not path.is_file() or not step_path.is_file() or step_path.stat().st_size == 0:
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return False
    return bool(
        rows
        and metadata.get("runtime_valid") is True
        and metadata.get("parent_key") == parent["parent_key"]
        and metadata.get("condition") == "CLEAN"
        and metadata.get("suite") == parent["suite"]
        and int(metadata.get("task_index", -1)) == parent["task_index"]
        and int(metadata.get("state_id", -1)) == parent["state_id"]
        and metadata.get("protocol_name") == PROTOCOL_NAME
        and metadata.get("protocol_version") == PROTOCOL_VERSION
        and metadata.get("git_commit") == expected_commit
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default="", help="optional format string containing {suite}")
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    parents = [normalized_parent(row) for row in read_rows(args.parents.resolve())]
    if len({row["parent_key"] for row in parents}) != len(parents):
        raise ValueError("duplicate parent_key")
    if args.max_jobs > 0:
        parents = parents[: args.max_jobs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, 1):
        metadata_path = args.output_root / parent["parent_key"] / "CLEAN" / "episode_metadata.json"
        if args.resume and complete(metadata_path, parent, args.expected_git_commit):
            results.append({"parent_key": parent["parent_key"], "status": "SKIP_COMPLETE"})
            continue
        command = [
            sys.executable, str(WORKER),
            "--parent-key", parent["parent_key"],
            "--condition", "CLEAN",
            "--checkpoint", args.checkpoint,
            "--output-dir", str(args.output_root),
            "--expected-git-commit", args.expected_git_commit,
            "--device", args.device,
            "--max-steps", str(parent["max_steps"]),
            "--burst-length", str(args.burst_length),
            "--objective-seed", str(parent["eval_seed"]),
        ]
        if args.model_path:
            command.extend(["--model-path", args.model_path.format(suite=parent["suite"])])
        print(f"[{index}/{len(parents)}] " + " ".join(command), flush=True)
        if args.dry_run:
            results.append({"parent_key": parent["parent_key"], "status": "DRY_RUN", "command": command})
            continue
        completed = subprocess.run(command, cwd=REPO)
        status = "PASS" if completed.returncode == 0 and complete(metadata_path, parent, args.expected_git_commit) else "HOLD"
        results.append({"parent_key": parent["parent_key"], "status": status, "returncode": completed.returncode})
        if status != "PASS":
            break
    status = "PASS_C2G_CLEAN_TIMING_RUNS" if all(row["status"] in {"PASS", "SKIP_COMPLETE", "DRY_RUN"} for row in results) else "HOLD_C2G_CLEAN_TIMING_RUNS"
    report = {"status": status, "parent_count": len(parents), "results": results}
    (args.output_root / "c2g_clean_timing_launcher_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
