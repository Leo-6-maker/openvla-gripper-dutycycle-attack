"""Register the SHA-bound, fail-closed Stage O command plan."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .audit_stage_v_closure import atomic_write_json
except ImportError:  # direct server execution
    from audit_stage_v_closure import atomic_write_json


FORBIDDEN = ("eval160", "protected_eval", "vis", "pgd", "attack", "student", "scheduler", "final_detector", "guard")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    stage_v_root = args.stage_v_root.resolve()
    goal_root = args.goal_root.resolve()
    start = json.loads((stage_v_root / "SUPERVISOR_START.json").read_text(encoding="utf-8"))
    parent_manifest = Path(str(start.get("parent_manifest", ""))).resolve()
    for path in (parent_manifest, args.runner):
        if not path.is_file():
            raise SystemExit(f"missing command binding file: {path}")
    if any(token in args.runner_command.lower() for token in FORBIDDEN):
        raise SystemExit("forbidden Stage O runner command")
    return {
        "schema": "STAGE_O_COMMAND_PLAN_V1",
        "stage": "O_OBSERVABILITY",
        "read_only": True,
        "stage_v_root": str(stage_v_root),
        "stage_v_source_commit": args.stage_v_source_commit,
        "stage_v_source_tree": args.stage_v_source_tree,
        "parent_manifest": str(parent_manifest),
        "parent_manifest_sha256": sha256_file(parent_manifest),
        "stage_o_runner_path": str(args.runner.resolve()),
        "stage_o_runner_sha256": sha256_file(args.runner),
        "runner_command": args.runner_command,
        "gpus": args.gpus,
        "salt": args.salt,
        "output_root_template": str(goal_root / "STAGE_O_OBSERVABILITY_{commit8}_{utc}"),
        "lock_path": str(goal_root / ".stage_o_observability.lock"),
        "cwd": str(args.cwd.resolve()),
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        "command_template": [
            args.python, str(args.runner.resolve()), "--parent-manifest", str(parent_manifest),
            "--output-root", "{output_root}", "--runner-command", args.runner_command,
            "--source-commit", "{source_commit}", "--source-tree", "{source_tree}",
            "--salt", args.salt, "--gpus", args.gpus,
        ],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", required=True, type=Path)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--stage-v-source-commit", required=True)
    parser.add_argument("--stage-v-source-tree", required=True)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--salt", default="STAGE_O_OBSERVABILITY_V1_20260807")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = (args.output or args.goal_root.resolve() / "MONITOR" / "STAGE_O_COMMAND_PLAN.json").resolve()
    atomic_write_json(output, build_plan(args))
    print(json.dumps({"schema": "STAGE_O_COMMAND_PLAN_V1", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
