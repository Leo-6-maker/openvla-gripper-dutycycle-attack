#!/usr/bin/env python3
"""Adapter for C6 primary-three-suite source smoke execution."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.validate_c6_source_condition_outcomes_v1 import COLUMNS, PRIMARY, read_rows, validate_rows, sha256_file  # noqa: E402

CONDITION_ORDER = ["CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"]


class C6SmokeAdapterError(ValueError):
    pass


def fail(message: str) -> None:
    raise C6SmokeAdapterError(message)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_cmd(template: list[str], mapping: dict[str, str]) -> list[str]:
    try:
        return [part.format(**mapping) for part in template]
    except KeyError as exc:
        fail(f"runner command_template missing placeholder: {exc}")


def load_config(path: str | Path) -> dict[str, Any]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        fail("runner_config_json must be object")
    template = obj.get("command_template")
    if not isinstance(template, list) or not all(isinstance(x, str) for x in template) or not template:
        fail("runner_config_json.command_template must be nonempty string list")
    return obj


def load_parents(path: str | Path) -> list[dict[str, str]]:
    obj = read_json(path)
    parents = obj.get("parents", obj) if isinstance(obj, dict) else obj
    if not isinstance(parents, list) or not parents:
        fail("selected parents must be nonempty list or object with parents")
    out = []
    required = {"parent_id", "episode_key", "suite", "task_id"}
    for item in parents:
        if not isinstance(item, dict) or not required <= set(item):
            fail("selected parent missing required fields")
        if item["suite"] not in PRIMARY:
            fail("selected parent must be primary suite only")
        out.append({k: str(item.get(k, "")) for k in sorted(set(item) | required)})
    seen_suites = {p["suite"] for p in out}
    if seen_suites != PRIMARY:
        fail("selected parents must include all primary suites")
    return out


def write_sums(root: Path, files: list[str]) -> tuple[str, str]:
    lines = [f"{sha256_file(root / name)}  {name}" for name in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(root / "SHA256SUMS"), sha256_file(root / "SHA256SUMS.sha256")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not args.runner_config_json or not Path(args.runner_config_json).is_file():
        report = {"status": "HOLD_NO_EXECUTION_RUNNER", "reason": "runner_config_json missing", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "source_condition_outcomes": "NOT_CREATED"}
        write_json(root / "execution_capability_report.json", report)
        return report
    config = load_config(args.runner_config_json)
    parents = load_parents(args.selected_parents_json)
    rows = []
    raw_runs = []
    run_dir = root / "raw_condition_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for parent in parents:
        for condition in CONDITION_ORDER:
            safe_parent = parent["parent_id"].replace("/", "_")
            work_dir = run_dir / f"{safe_parent}_{condition}"
            work_dir.mkdir(parents=True, exist_ok=True)
            result_path = work_dir / "result.json"
            mapping = {**parent, "condition": condition, "result_json": str(result_path), "output_json": str(result_path), "work_dir": str(work_dir), "raw_output_dir": str(work_dir), "legacy_result_json": str(work_dir / "legacy_result.json")}
            cmd = format_cmd(config["command_template"], mapping)
            completed = subprocess.run(cmd, text=True, capture_output=True, timeout=int(config.get("timeout_seconds", args.timeout_seconds)))
            (work_dir / "stdout.txt").write_text(completed.stdout[-20000:], encoding="utf-8")
            (work_dir / "stderr.txt").write_text(completed.stderr[-20000:], encoding="utf-8")
            raw_runs.append({"parent_id": parent["parent_id"], "suite": parent["suite"], "condition": condition, "returncode": completed.returncode, "result_json": str(result_path), "work_dir": str(work_dir)})
            if completed.returncode != 0:
                write_json(root / "raw_condition_runs_manifest.json", {"status": "HOLD_RUNNER_FAILED", "runs": raw_runs})
                fail("external runner failed")
            if not result_path.is_file():
                fail("external runner did not write result_json")
            result = read_json(result_path)
            if not isinstance(result, dict):
                fail("runner result_json must be object")
            row = {col: result.get(col, parent.get(col, "")) for col in COLUMNS}
            row["condition"] = condition
            row["suite"] = parent["suite"]
            row["task_id"] = parent["task_id"]
            row["episode_key"] = parent["episode_key"]
            row["parent_id"] = parent["parent_id"]
            rows.append(row)
    source = root / "source_condition_outcomes.csv"
    write_csv(source, rows)
    validation = validate_rows(read_rows(source))
    validation["source_csv_sha256"] = sha256_file(source)
    write_json(root / "source_schema_validation_report.json", validation)
    write_json(root / "selected_smoke_parents.json", {"parents": parents})
    write_json(root / "raw_condition_runs_manifest.json", {"status": "PASS", "runs": raw_runs})
    manifest = {"status": "PASS_SOURCE_SMOKE_READY", "source_condition_outcomes_sha256": sha256_file(source), "row_count": len(rows), "parent_count": len(parents), "conditions": CONDITION_ORDER, "primary_suites": sorted(PRIMARY), "libero_10_positive_denominator": "EXCLUDED", "OpenVLA": "PERFORMED_SMOKE", "LIBERO": "PERFORMED_SMOKE", "rollout": "PERFORMED_SMOKE", "intervention": "PERFORMED_SMOKE"}
    write_json(root / "source_smoke_manifest.json", manifest)
    sums, side = write_sums(root, ["source_condition_outcomes.csv", "source_smoke_manifest.json", "source_schema_validation_report.json", "raw_condition_runs_manifest.json", "selected_smoke_parents.json"])
    manifest["SHA256SUMS"] = sums
    manifest["SHA256SUMS.sha256"] = side
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-config-json")
    parser.add_argument("--selected-parents-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True))
        return 0
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, csv.Error, C6SmokeAdapterError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
