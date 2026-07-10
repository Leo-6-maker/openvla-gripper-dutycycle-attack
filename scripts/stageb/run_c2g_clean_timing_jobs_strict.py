#!/usr/bin/env python3
"""Run detector-only CLEAN timing with exact suite models and Goal provenance."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "stageb" / "run_c2g_clean_window_vis_pgd.py"
PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"
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


def validate_goal_manifest(path: Path, goal_model_path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED":
        raise ValueError("Goal model manifest status is not PASS")
    if Path(str(value.get("model_path", ""))).resolve() != goal_model_path.resolve():
        raise ValueError("Goal model manifest path does not match suite model map")
    if value.get("missing_referenced_shards"):
        raise ValueError("Goal model manifest reports missing shards")


def normalized_parent(row: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("suite", "task_index", "state_id"):
        if field not in row:
            raise ValueError(f"parent row missing {field}")
    suite = str(row["suite"])
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
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


def complete(path: Path, parent: Mapping[str, Any], expected_commit: str, checkpoint_sha: str) -> bool:
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
        and metadata.get("condition") == "CLEAN"
        and metadata.get("parent_key") == parent["parent_key"]
        and metadata.get("suite") == parent["suite"]
        and int(metadata.get("task_index", -1)) == int(parent["task_index"])
        and int(metadata.get("state_id", -1)) == int(parent["state_id"])
        and metadata.get("protocol_name") == PROTOCOL_NAME
        and metadata.get("protocol_version") == PROTOCOL_VERSION
        and metadata.get("git_commit") == expected_commit
        and metadata.get("detector_checkpoint_sha256") == checkpoint_sha
        and int(metadata.get("attack_delivery_count", -1)) == 0
        and not any(bool(row.get("attack_delivered")) for row in rows)
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
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
    model_map = read_model_map(args.suite_model_map.resolve())
    validate_goal_manifest(
        args.goal_model_manifest.resolve(),
        Path(model_map["libero_goal"]),
    )
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, parent in enumerate(parents, 1):
        metadata_path = output_root / parent["parent_key"] / "CLEAN" / "episode_metadata.json"
        if args.resume and complete(metadata_path, parent, args.expected_git_commit, checkpoint_sha):
            results.append({"parent_key": parent["parent_key"], "status": "SKIP_COMPLETE"})
            continue
        command = [
            sys.executable,
            str(WORKER),
            "--parent-key", parent["parent_key"],
            "--condition", "CLEAN",
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_root),
            "--expected-git-commit", args.expected_git_commit,
            "--device", args.device,
            "--model-path", model_map[parent["suite"]],
            "--max-steps", str(parent["max_steps"]),
            "--burst-length", str(args.burst_length),
            "--objective-seed", str(parent["eval_seed"]),
        ]
        if parent["suite"] == "libero_goal":
            command.extend(["--policy-model-manifest", str(args.goal_model_manifest.resolve())])
        print(f"[{index}/{len(parents)}] " + " ".join(command), flush=True)
        if args.dry_run:
            results.append({"parent_key": parent["parent_key"], "status": "DRY_RUN", "command": command})
            continue
        completed = subprocess.run(command, cwd=REPO)
        status = (
            "PASS"
            if completed.returncode == 0
            and complete(metadata_path, parent, args.expected_git_commit, checkpoint_sha)
            else "HOLD"
        )
        results.append(
            {
                "parent_key": parent["parent_key"],
                "status": status,
                "returncode": completed.returncode,
            }
        )
        if status != "PASS":
            break

    status = (
        "PASS_C2G_CLEAN_TIMING_STRICT"
        if results and all(row["status"] in {"PASS", "SKIP_COMPLETE", "DRY_RUN"} for row in results)
        else "HOLD_C2G_CLEAN_TIMING_STRICT"
    )
    report = {
        "gate": "C2G_CLEAN_TIMING_STRICT",
        "status": status,
        "parent_count": len(parents),
        "suite_model_map": str(args.suite_model_map.resolve()),
        "goal_model_manifest": str(args.goal_model_manifest.resolve()),
        "detector_checkpoint_sha256": checkpoint_sha,
        "results": results,
    }
    (output_root / "c2g_clean_timing_strict_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
