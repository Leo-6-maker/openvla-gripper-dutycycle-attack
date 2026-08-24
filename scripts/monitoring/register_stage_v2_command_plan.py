"""Register the SHA-bound, fail-closed Stage V2 command plan.

The plan is intentionally not the launch command: the monitor creates the
final command only after it can hash the real Stage V closure receipt.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .audit_stage_v_closure import atomic_write_json, sha256_file
except ImportError:  # pragma: no cover - direct server execution.
    from audit_stage_v_closure import atomic_write_json, sha256_file


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    goal_root = args.goal_root.resolve()
    stage_v_root = args.stage_v_root.resolve()
    start = load(stage_v_root / "SUPERVISOR_START.json")
    if not isinstance(start, Mapping) or not start.get("parent_manifest"):
        raise SystemExit("parent manifest binding missing")
    parent_manifest = Path(str(start["parent_manifest"])).resolve()
    run_manifest = stage_v_root / "RUN_MANIFEST.json"
    for path in (parent_manifest, run_manifest, args.runner, args.auditor, args.config):
        if not path.is_file():
            raise SystemExit(f"missing command binding file: {path}")
    plan = {
        "schema": "STAGE_V2_COMMAND_PLAN_V1",
        "stage": "V2_TEACHER_ENRICHMENT",
        "read_only": True,
        "stage_v_root": str(stage_v_root),
        "stage_v_source_commit": args.stage_v_source_commit,
        "stage_v_source_tree": args.stage_v_source_tree,
        "stage_v2_source_commit": args.stage_v2_source_commit,
        "stage_v2_source_tree": args.stage_v2_source_tree,
        "expected_parent_manifest_sha256": sha256_file(parent_manifest),
        "expected_run_manifest_sha256": sha256_file(run_manifest),
        "stage_v2_runner_path": str(args.runner.resolve()),
        "stage_v2_runner_sha256": sha256_file(args.runner),
        "stage_v2_auditor_path": str(args.auditor.resolve()),
        "stage_v2_auditor_sha256": sha256_file(args.auditor),
        "stage_v2_config_path": str(args.config.resolve()),
        "stage_v2_config_sha256": sha256_file(args.config),
        "output_root_template": str(goal_root / "STAGE_V2_TEACHER_ENRICHMENT_{commit8}_{utc}"),
        "lock_path": str(goal_root / ".stage_v2_teacher_enrichment.lock"),
        "cwd": str(args.cwd.resolve()),
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "STAGE_V2_RUNNER_PATH": str(args.runner.resolve()),
        },
        "command_template": [
            args.python,
            str(args.runner.resolve()),
            "--stage-v-root",
            "{stage_v_root}",
            "--output-root",
            "{output_root}",
            "--config",
            str(args.config.resolve()),
            "--expected-source-commit",
            "{source_commit}",
            "--expected-source-tree",
            "{source_tree}",
            "--expected-parent-manifest-sha256",
            "{expected_parent_manifest_sha256}",
            "--expected-run-manifest-sha256",
            "{expected_run_manifest_sha256}",
            "--run-independent-audit",
        ],
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-root", required=True, type=Path)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--stage-v-source-commit", required=True)
    parser.add_argument("--stage-v-source-tree", required=True)
    parser.add_argument("--stage-v2-source-commit", required=True)
    parser.add_argument("--stage-v2-source-tree", required=True)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--auditor", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    output = (args.output or args.goal_root.resolve() / "MONITOR" / "STAGE_V2_COMMAND_PLAN.json").resolve()
    atomic_write_json(output, build_plan(args))
    print(json.dumps({"schema": "STAGE_V2_COMMAND_PLAN_V1", "output": str(output)}, sort_keys=True))
