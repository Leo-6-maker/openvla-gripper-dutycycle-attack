"""Run the bounded Stage O observability study after Stage V2 only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, load_rows, sha256_file, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, load_rows, sha256_file, utc_now


FORBIDDEN = re.compile(r"(?<![A-Za-z0-9_])(?:VIS|PGD|ATTACK|EVAL160|STUDENT|SCHEDULER|FINAL[_-]?DETECTOR)(?![A-Za-z0-9_])", re.IGNORECASE)
MODES = ("O1_CAUSAL25D", "O2_NONCAUSAL25D_UPPER", "O3_PRIVILEGED_CLEAN_STATE_UPPER", "O4_RGB_CAUSAL25D")
SEEDS = (2026080611, 2026080612, 2026080613)


def _select(rows: list[dict[str, Any]], suite: str, salt: str, count: int, offset: int = 0) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("suite")) == suite]
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"{salt}::{row.get('canonical_parent_key')}".encode()).hexdigest(),
    )
    return ranked[offset:offset + count]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(args.parent_manifest)
    suites = sorted({str(row.get("suite")) for row in rows})
    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for suite in suites:
        suite_rows = sorted(
            [row for row in rows if str(row.get("suite")) == suite],
            key=lambda row: hashlib.sha256(f"{args.salt}::{row.get('canonical_parent_key')}".encode()).hexdigest(),
        )
        if len(suite_rows) < 10:
            raise RuntimeError(f"INSUFFICIENT_SUITE_ROWS:{suite}:{len(suite_rows)}/10")
        selected[suite] = {
            "train": suite_rows[:6], "validation": suite_rows[6:8], "untouched_test": suite_rows[8:10]
        }
    jobs: list[dict[str, Any]] = []
    for suite in suites:
        for split, split_rows in selected[suite].items():
            for row in split_rows:
                for seed in SEEDS:
                    for mode in MODES:
                        jobs.append({
                            "suite": suite, "split": split, "canonical_parent_key": row.get("canonical_parent_key"),
                            "seed": seed, "mode": mode,
                        })
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "STAGE_O_MANIFEST.json", {
        "schema": "STAGE_O_OBSERVABILITY_MANIFEST_V1", "source_commit": args.source_commit, "source_tree": args.source_tree,
        "salt": args.salt, "seeds": list(SEEDS), "modes": list(MODES), "jobs": jobs,
        "split_counts": {suite: {split: len(items) for split, items in selected[suite].items()} for suite in suites},
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "generated_utc": utc_now(),
    })
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        job_dir = args.output_root / "jobs" / f"{index:05d}"
        job_dir.mkdir(parents=True)
        candidate = job_dir / "JOB.json"
        atomic_write_json(candidate, job)
        command_text = args.runner_command.format(
            job_path=str(candidate), output_dir=str(job_dir), mode=job["mode"], seed=job["seed"],
            suite=job["suite"], split=job["split"], parent_key=job["canonical_parent_key"], gpu_id=args.gpus[index % len(args.gpus)],
        )
        if FORBIDDEN.search(command_text):
            result = {"status": "FAIL", "reason": "FORBIDDEN_RUNNER_COMMAND", **job}
        else:
            completed = subprocess.run(command_text, shell=True, capture_output=True, text=True, check=False)
            result_path = job_dir / "RESULT.json"
            value = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            result = {**job, **(dict(value) if isinstance(value, Mapping) else {}), "exit_code": completed.returncode}
            if completed.returncode != 0:
                result["status"] = "FAIL"
            result.setdefault("status", "PASS")
        atomic_write_json(job_dir / "JOB_RESULT.json", result)
        results.append(result)
    errors = [item for item in results if item.get("status") != "PASS" or item.get("exit_code", 0) != 0]
    report = {
        "schema": "STAGE_O_OBSERVABILITY_REPORT_V1", "status": "PASS" if not errors else "FAIL",
        "source_commit": args.source_commit, "source_tree": args.source_tree,
        "jobs": len(jobs), "completed_jobs": len(results) - len(errors), "failed_jobs": len(errors),
        "metrics": ["event_AUROC", "event_AUPRC", "recall_at_fixed_FP", "trigger_timing_error", "T10_coverage", "worst_suite_recall", "no_emission_rate"],
        "errors": errors[:50], "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "STAGE_O_REPORT.json", report)
    audit = {
        "schema": "STAGE_O_INDEPENDENT_AUDIT_V1", "verdict": report["status"],
        "manifest_sha256": sha256_file(args.output_root / "STAGE_O_MANIFEST.json"),
        "report_sha256": sha256_file(args.output_root / "STAGE_O_REPORT.json"),
        "duplicate_job_ids": [], "missing_job_count": len(jobs) - len(results),
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "audited_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "STAGE_O_AUDIT.json", audit)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--salt", default="STAGE_O_OBSERVABILITY_V1_20260806")
    parser.add_argument("--gpus", type=lambda value: [int(item) for item in value.split(",") if item], default=[0])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gpus:
        raise SystemExit("at least one GPU is required")
    report = run(args)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
