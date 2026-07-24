#!/usr/bin/env python3
"""Validate C6 condition-matrix artifact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = [
    "matrix_manifest.json",
    "detector_freeze_identity.json",
    "replay_identity.json",
    "run_config.json",
    "metrics_summary.json",
    "outcomes_overall.csv",
    "outcomes_by_suite.csv",
    "outcomes_by_task.csv",
    "gripper_bridge_report.json",
    "command_duty_report.json",
    "control_integrity_report.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
]
REQUIRED_CONDITIONS = {"CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"}
REQUIRED_BOUNDARIES = {"label_mutation": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"}


class C6MatrixValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise C6MatrixValidationError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"{path.name}: expected JSON object")
    return obj


def flat(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flat(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            out.update(flat(value, f"{prefix}[{idx}]"))
    else:
        out[prefix] = obj
    return out


def find(obj: dict[str, Any], keys: list[str]) -> Any:
    f = flat(obj)
    for key in keys:
        if key in obj:
            return obj[key]
        if key in f:
            return f[key]
    tails = {k.lower().split(".")[-1]: v for k, v in f.items()}
    for key in keys:
        tail = key.lower().split(".")[-1]
        if tail in tails:
            return tails[tail]
    return None


def require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail(f"{field}: expected sha256")


def validate_sums(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    side = root / "SHA256SUMS.sha256"
    entries = 0
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            fail(f"SHA256SUMS:{line_no}: malformed")
        digest, rel = parts
        require_digest(digest, f"SHA256SUMS:{line_no}")
        target = root / rel
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not target.is_file() or sha256_file(target) != digest:
            fail(f"SHA256SUMS:{line_no}: digest mismatch")
        entries += 1
    side_parts = side.read_text(encoding="utf-8").strip().split()
    if len(side_parts) != 2 or side_parts[1] != "SHA256SUMS":
        fail("SHA256SUMS.sha256 malformed")
    require_digest(side_parts[0], "SHA256SUMS.sha256")
    if sha256_file(sums) != side_parts[0]:
        fail("SHA256SUMS.sha256 mismatch")
    return {"entry_count": entries, "sha256sums_sha256": side_parts[0]}


def read_conditions(path: Path) -> tuple[int, set[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not reader.fieldnames or not rows:
        fail(f"{path.name}: expected nonempty csv")
    candidates = [c for c in ["condition", "mode", "treatment"] if c in reader.fieldnames]
    if not candidates:
        fail(f"{path.name}: missing condition/mode/treatment column")
    col = candidates[0]
    return len(rows), {row[col] for row in rows}


def require_pass(obj: dict[str, Any], name: str) -> None:
    if find(obj, ["status", "validation_status", "matrix_status"]) not in ("PASS", "OK", True):
        fail(f"{name}: status is not PASS")


def validate_boundaries(objs: list[dict[str, Any]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for obj in objs:
        for key, value in flat(obj).items():
            tail = key.lower().split(".")[-1]
            for need, expected in REQUIRED_BOUNDARIES.items():
                if tail == need and value == expected:
                    observed[need] = expected
    missing = sorted(set(REQUIRED_BOUNDARIES) - set(observed))
    if missing:
        fail(f"missing boundary markers: {missing}")
    return observed


def validate(
    root: str | Path,
    expected_freeze_sha256: str,
    expected_replay_sha256: str,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(root)
    if not path.is_dir():
        fail("root does not exist")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))
    sums = validate_sums(path)
    objs = {name: read_json(path / name) for name in REQUIRED_FILES if name.endswith(".json")}
    for name in ["matrix_manifest.json", "metrics_summary.json", "gripper_bridge_report.json", "command_duty_report.json", "control_integrity_report.json"]:
        require_pass(objs[name], name)
    if find(objs["detector_freeze_identity.json"], ["freeze_manifest_sha256", "detector_freeze_sha256", "sha256"]) != expected_freeze_sha256:
        fail("freeze sha256 mismatch")
    if find(objs["replay_identity.json"], ["replay_manifest_sha256", "detector_replay_sha256", "sha256"]) != expected_replay_sha256:
        fail("replay sha256 mismatch")
    if find(objs["run_config.json"], ["exact_prefix_shared", "shared_prefix"] ) not in (True, "true", "TRUE"):
        fail("shared exact-prefix marker missing")
    if find(objs["run_config.json"], ["clean_success_parent_denominator", "parent_denominator"] ) not in (True, "true", "TRUE"):
        fail("clean-success parent denominator marker missing")
    overall_rows, conditions = read_conditions(path / "outcomes_overall.csv")
    suite_rows, suite_conditions = read_conditions(path / "outcomes_by_suite.csv")
    task_rows, task_conditions = read_conditions(path / "outcomes_by_task.csv")
    manifest_conditions = find(objs["matrix_manifest.json"], ["conditions", "condition_set"])
    if isinstance(manifest_conditions, list):
        conditions |= {str(x) for x in manifest_conditions}
    missing_conditions = sorted(REQUIRED_CONDITIONS - conditions)
    if missing_conditions:
        fail(f"missing required conditions: {missing_conditions}")
    if not REQUIRED_CONDITIONS <= suite_conditions | conditions:
        fail("suite outcomes missing required conditions")
    if not REQUIRED_CONDITIONS <= task_conditions | conditions:
        fail("task outcomes missing required conditions")
    report = {
        "status": "PASS",
        "schema_version": "c6_matrix_artifact_validation_v1",
        "root": str(path),
        "freeze_sha256": expected_freeze_sha256,
        "replay_sha256": expected_replay_sha256,
        "conditions": sorted(conditions),
        "outcomes_overall_rows": overall_rows,
        "outcomes_by_suite_rows": suite_rows,
        "outcomes_by_task_rows": task_rows,
        "sha256sums": sums,
        "boundaries": validate_boundaries(list(objs.values())),
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-replay-sha256", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        report = validate(args.root, args.expected_freeze_sha256, args.expected_replay_sha256, args.output_json)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
