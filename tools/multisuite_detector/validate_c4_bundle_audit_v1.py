#!/usr/bin/env python3
"""Validate C4-2 detector bundle audit evidence.

This is a CPU-only evidence validator. It does not train detectors, run
OpenVLA/LIBERO, run rollouts, run attacks, or use GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "bundle_identity.json",
    "checkpoint_identity.json",
    "dataset_identity.json",
    "threshold_identity.json",
    "metrics_overall.json",
    "metrics_by_suite.csv",
    "metrics_by_task.csv",
    "metrics_by_population.csv",
    "safety_false_trigger_report.json",
    "emission_rate_report.json",
    "bundle_load_report.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
]

REQUIRED_NON_ACTIONS = {
    "OpenVLA": "NOT_PERFORMED",
    "LIBERO": "NOT_PERFORMED",
    "rollout": "NOT_PERFORMED",
    "attack": "NOT_PERFORMED",
}


class C4BundleAuditError(ValueError):
    pass


def fail(message: str) -> None:
    raise C4BundleAuditError(message)


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


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            out.update(flatten_json(value, next_prefix))
    else:
        out[prefix] = obj
    return out


def find_value(obj: dict[str, Any], candidate_keys: list[str]) -> Any:
    flat = flatten_json(obj)
    for key in candidate_keys:
        if key in obj:
            return obj[key]
        if key in flat:
            return flat[key]
    lowered = {k.lower().split(".")[-1]: v for k, v in flat.items()}
    for key in candidate_keys:
        tail = key.lower().split(".")[-1]
        if tail in lowered:
            return lowered[tail]
    return None


def require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail(f"{field}: expected lowercase sha256 digest")


def validate_sha256sums(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file():
        fail("missing SHA256SUMS")
    if not sidecar.is_file():
        fail("missing SHA256SUMS.sha256")
    entries: dict[str, str] = {}
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            fail(f"SHA256SUMS:{line_no}: malformed line")
        digest, rel = parts
        require_digest(digest, f"SHA256SUMS:{line_no}")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            fail(f"SHA256SUMS:{line_no}: unsafe relative path")
        if rel in entries:
            fail(f"SHA256SUMS:{line_no}: duplicate entry {rel}")
        target = root / rel_path
        if not target.is_file():
            fail(f"SHA256SUMS:{line_no}: missing file {rel}")
        observed = sha256_file(target)
        if observed != digest:
            fail(f"SHA256SUMS:{line_no}: digest mismatch for {rel}")
        entries[rel] = digest
    sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(sidecar_parts) != 2:
        fail("SHA256SUMS.sha256 malformed")
    side_digest, side_name = sidecar_parts
    require_digest(side_digest, "SHA256SUMS.sha256")
    if side_name != "SHA256SUMS":
        fail("SHA256SUMS.sha256 must name SHA256SUMS")
    if sha256_file(sums) != side_digest:
        fail("SHA256SUMS.sha256 digest mismatch")
    return {"entry_count": len(entries), "entries": entries, "sha256sums_sha256": side_digest}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            fail(f"{path.name}: empty CSV header")
        if not rows:
            fail(f"{path.name}: no rows")
        return rows


def require_status_pass(obj: dict[str, Any], path: Path) -> None:
    status = find_value(obj, ["status", "bundle_load_status", "load_status", "validation_status"])
    if status not in (None, "PASS", "OK", True):
        fail(f"{path.name}: non-pass status {status!r}")


def validate_audit(
    *,
    audit_root: str | Path,
    expected_checkpoint_sha256: str,
    expected_dataset_csv_sha256: str,
    expected_split_csv_sha256: str,
    expected_state_index_sha256: str,
    expected_threshold: float,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(audit_root)
    if not root.is_dir():
        fail("audit root does not exist")
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            fail(f"missing required audit file: {rel}")

    sha_report = validate_sha256sums(root)
    checkpoint = read_json(root / "checkpoint_identity.json")
    dataset = read_json(root / "dataset_identity.json")
    threshold = read_json(root / "threshold_identity.json")
    metrics = read_json(root / "metrics_overall.json")
    safety = read_json(root / "safety_false_trigger_report.json")
    emission = read_json(root / "emission_rate_report.json")
    bundle = read_json(root / "bundle_load_report.json")

    observed_ckpt = find_value(checkpoint, ["checkpoint_sha256", "best_checkpoint_sha256", "sha256"])
    if observed_ckpt != expected_checkpoint_sha256:
        fail("checkpoint sha256 mismatch")
    observed_dataset = find_value(dataset, ["dataset_csv_sha256"])
    observed_split = find_value(dataset, ["split_csv_sha256"])
    observed_state = find_value(dataset, ["state_index_sha256"])
    if observed_dataset != expected_dataset_csv_sha256:
        fail("dataset_csv_sha256 mismatch")
    if observed_split != expected_split_csv_sha256:
        fail("split_csv_sha256 mismatch")
    if observed_state != expected_state_index_sha256:
        fail("state_index_sha256 mismatch")

    observed_threshold = find_value(threshold, ["threshold", "validation_selected_threshold", "selected_threshold"])
    try:
        observed_threshold_float = float(observed_threshold)
    except (TypeError, ValueError):
        fail("threshold_identity: missing numeric threshold")
    if abs(observed_threshold_float - expected_threshold) > 1e-12:
        fail("threshold mismatch")

    for name, obj in [
        ("metrics_overall.json", metrics),
        ("safety_false_trigger_report.json", safety),
        ("emission_rate_report.json", emission),
        ("bundle_load_report.json", bundle),
    ]:
        require_status_pass(obj, root / name)

    suite_rows = read_csv_rows(root / "metrics_by_suite.csv")
    task_rows = read_csv_rows(root / "metrics_by_task.csv")
    population_rows = read_csv_rows(root / "metrics_by_population.csv")

    non_actions = {}
    for obj in [bundle, metrics, safety, emission]:
        flat = flatten_json(obj)
        for key, expected in REQUIRED_NON_ACTIONS.items():
            for flat_key, value in flat.items():
                if flat_key.lower().endswith(key.lower()) and value == expected:
                    non_actions[key] = expected
    missing_non_actions = sorted(set(REQUIRED_NON_ACTIONS) - set(non_actions))
    if missing_non_actions:
        fail(f"missing non-action markers: {missing_non_actions}")

    report = {
        "status": "PASS",
        "audit_root": str(root),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "dataset_csv_sha256": expected_dataset_csv_sha256,
        "split_csv_sha256": expected_split_csv_sha256,
        "state_index_sha256": expected_state_index_sha256,
        "threshold": expected_threshold,
        "sha256sums": sha_report,
        "metrics_by_suite_rows": len(suite_rows),
        "metrics_by_task_rows": len(task_rows),
        "metrics_by_population_rows": len(population_rows),
        "non_actions": non_actions,
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-dataset-csv-sha256", required=True)
    parser.add_argument("--expected-split-csv-sha256", required=True)
    parser.add_argument("--expected-state-index-sha256", required=True)
    parser.add_argument("--expected-threshold", required=True, type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = validate_audit(
            audit_root=args.audit_root,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_dataset_csv_sha256=args.expected_dataset_csv_sha256,
            expected_split_csv_sha256=args.expected_split_csv_sha256,
            expected_state_index_sha256=args.expected_state_index_sha256,
            expected_threshold=args.expected_threshold,
            output_json=args.output_json,
        )
    except (C4BundleAuditError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
