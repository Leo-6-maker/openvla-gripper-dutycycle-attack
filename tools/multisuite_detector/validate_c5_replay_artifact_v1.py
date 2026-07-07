#!/usr/bin/env python3
"""Validate C5 detector-only replay artifact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = [
    "replay_manifest.json",
    "detector_freeze_identity.json",
    "dataset_identity.json",
    "threshold_identity.json",
    "replay_config.json",
    "metrics_overall.json",
    "metrics_by_suite.csv",
    "metrics_by_task.csv",
    "timing_error_report.json",
    "emission_rate_report.json",
    "safety_false_trigger_report.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
]
NON_ACTIONS = {"simulator": "NOT_PERFORMED", "policy_run": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED"}


class C5ReplayValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise C5ReplayValidationError(message)


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
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
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
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            fail(f"SHA256SUMS:{line_no}: unsafe path")
        target = root / rel_path
        if not target.is_file() or sha256_file(target) != digest:
            fail(f"SHA256SUMS:{line_no}: digest mismatch")
        entries += 1
    side_parts = side.read_text(encoding="utf-8").strip().split()
    if len(side_parts) != 2 or side_parts[1] != "SHA256SUMS":
        fail("SHA256SUMS.sha256 malformed")
    require_digest(side_parts[0], "SHA256SUMS.sha256")
    if sha256_file(sums) != side_parts[0]:
        fail("SHA256SUMS.sha256 mismatch")
    return {"entry_count": entries, "sha256sums_sha256": side_parts[0]}


def require_csv(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not reader.fieldnames or not rows:
        fail(f"{path.name}: expected nonempty csv")
    return len(rows)


def require_pass(obj: dict[str, Any], name: str) -> None:
    if find(obj, ["status", "validation_status", "replay_status"]) not in ("PASS", "OK", True):
        fail(f"{name}: status is not PASS")


def require_non_actions(objs: list[dict[str, Any]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for obj in objs:
        for key, value in flat(obj).items():
            tail = key.lower().split(".")[-1]
            for need, expected in NON_ACTIONS.items():
                if tail == need and value == expected:
                    observed[need] = expected
    missing = sorted(set(NON_ACTIONS) - set(observed))
    if missing:
        fail(f"missing non-action markers: {missing}")
    return observed


def validate(
    root: str | Path,
    expected_freeze_sha256: str,
    expected_checkpoint_sha256: str,
    expected_dataset_csv_sha256: str,
    expected_split_csv_sha256: str,
    expected_state_index_sha256: str,
    expected_threshold: float,
) -> dict[str, Any]:
    path = Path(root)
    if not path.is_dir():
        fail("root does not exist")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))
    sums = validate_sums(path)
    objs = {name: read_json(path / name) for name in REQUIRED_FILES if name.endswith(".json")}
    for name in ["replay_manifest.json", "metrics_overall.json", "timing_error_report.json", "emission_rate_report.json", "safety_false_trigger_report.json"]:
        require_pass(objs[name], name)
    freeze = objs["detector_freeze_identity.json"]
    dataset = objs["dataset_identity.json"]
    threshold = objs["threshold_identity.json"]
    config = objs["replay_config.json"]
    if find(freeze, ["freeze_manifest_sha256", "detector_freeze_sha256", "sha256"]) != expected_freeze_sha256:
        fail("freeze sha256 mismatch")
    if find(freeze, ["checkpoint_sha256", "best_checkpoint_sha256"]) != expected_checkpoint_sha256:
        fail("checkpoint sha256 mismatch")
    if find(dataset, ["dataset_csv_sha256"]) != expected_dataset_csv_sha256:
        fail("dataset_csv_sha256 mismatch")
    if find(dataset, ["split_csv_sha256", "split_manifest_sha256"]) != expected_split_csv_sha256:
        fail("split_csv_sha256 mismatch")
    if find(dataset, ["state_index_sha256"]) != expected_state_index_sha256:
        fail("state_index_sha256 mismatch")
    try:
        threshold_value = float(find(threshold, ["threshold", "selected_threshold", "validation_selected_threshold"]))
    except (TypeError, ValueError):
        fail("threshold missing")
    if abs(threshold_value - expected_threshold) > 1e-12:
        fail("threshold mismatch")
    if find(threshold, ["threshold_source", "selection_source", "selected_on"]) not in ("validation", "val", "validation_selected", "validation_set"):
        fail("threshold must be validation-selected")
    if find(config, ["exact_prefix", "exact_prefix_replay"]) not in (True, "true", "TRUE", "DETECTOR_ONLY"):
        fail("exact-prefix marker missing")
    if find(config, ["detector_only", "detector_replay_only"]) not in (True, "true", "TRUE", "DETECTOR_ONLY"):
        fail("detector-only marker missing")
    suite_rows = require_csv(path / "metrics_by_suite.csv")
    task_rows = require_csv(path / "metrics_by_task.csv")
    return {
        "status": "PASS",
        "schema_version": "c5_replay_artifact_validation_v1",
        "root": str(path),
        "freeze_sha256": expected_freeze_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "dataset_csv_sha256": expected_dataset_csv_sha256,
        "split_csv_sha256": expected_split_csv_sha256,
        "state_index_sha256": expected_state_index_sha256,
        "threshold": threshold_value,
        "metrics_by_suite_rows": suite_rows,
        "metrics_by_task_rows": task_rows,
        "sha256sums": sums,
        "non_actions": require_non_actions(list(objs.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-dataset-csv-sha256", required=True)
    parser.add_argument("--expected-split-csv-sha256", required=True)
    parser.add_argument("--expected-state-index-sha256", required=True)
    parser.add_argument("--expected-threshold", type=float, required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        report = validate(args.root, args.expected_freeze_sha256, args.expected_checkpoint_sha256, args.expected_dataset_csv_sha256, args.expected_split_csv_sha256, args.expected_state_index_sha256, args.expected_threshold)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
