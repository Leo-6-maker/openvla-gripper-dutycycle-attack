#!/usr/bin/env python3
"""Validate a C4 detector freeze evidence bundle.

This validator is CPU-only and evidence-only. It does not train detectors, run
OpenVLA/LIBERO, perform rollouts, run attacks, run exact-prefix replay, or use
GPU. It is intended to gate the transition from detector training to C5
replay/attack preparation.
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
    "freeze_manifest.json",
    "bundle_identity.json",
    "checkpoint_identity.json",
    "dataset_identity.json",
    "split_identity.json",
    "normalization_identity.json",
    "threshold_identity.json",
    "metrics_summary.json",
    "bundle_load_report.json",
    "SHA256SUMS",
    "SHA256SUMS.sha256",
]

FINAL_SPLIT_TYPES = {
    "object_task_heldout_with_val_v1",
    "suite_loso_with_val_v1",
}

CANDIDATE_SPLIT_TYPES = {
    "parent_random_split_v1",
    "parent_random",
}

VALIDATION_THRESHOLD_SOURCES = {
    "validation",
    "val",
    "validation_set",
    "validation_selected",
    "val_selected",
}

REQUIRED_NON_ACTIONS = {
    "OpenVLA": "NOT_PERFORMED",
    "LIBERO": "NOT_PERFORMED",
    "rollout": "NOT_PERFORMED",
    "attack": "NOT_PERFORMED",
    "exact_prefix_replay": "NOT_PERFORMED",
    "victim_inference": "NOT_PERFORMED",
}


class C4DetectorFreezeError(ValueError):
    pass


def fail(message: str) -> None:
    raise C4DetectorFreezeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        fail(f"{field}: expected lowercase sha256 digest")


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"{path.name}: expected JSON object")
    return obj


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            out.update(flatten(value, next_prefix))
    else:
        out[prefix] = obj
    return out


def find_value(obj: dict[str, Any], candidate_keys: list[str]) -> Any:
    flat = flatten(obj)
    for key in candidate_keys:
        if key in obj:
            return obj[key]
        if key in flat:
            return flat[key]
    tails: dict[str, Any] = {}
    for key, value in flat.items():
        tail = key.lower().split(".")[-1]
        tails.setdefault(tail, value)
    for key in candidate_keys:
        tail = key.lower().split(".")[-1]
        if tail in tails:
            return tails[tail]
    return None


def find_split_types(split_identity: dict[str, Any], freeze_manifest: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for obj in [split_identity, freeze_manifest]:
        for key in ["split_type", "split_types", "schema_version"]:
            value = find_value(obj, [key])
            if value is not None:
                values.append(value)
    out: set[str] = set()
    for value in values:
        if isinstance(value, str):
            out.add(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    out.add(item)
    return out


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
            fail(f"SHA256SUMS:{line_no}: unsafe path {rel}")
        if rel in entries:
            fail(f"SHA256SUMS:{line_no}: duplicate entry {rel}")
        target = root / rel_path
        if not target.is_file():
            fail(f"SHA256SUMS:{line_no}: missing file {rel}")
        observed = sha256_file(target)
        if observed != digest:
            fail(f"SHA256SUMS:{line_no}: digest mismatch for {rel}")
        entries[rel] = digest
    side_parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(side_parts) != 2:
        fail("SHA256SUMS.sha256 malformed")
    digest, rel = side_parts
    require_digest(digest, "SHA256SUMS.sha256")
    if rel != "SHA256SUMS":
        fail("SHA256SUMS.sha256 must name SHA256SUMS")
    if sha256_file(sums) != digest:
        fail("SHA256SUMS.sha256 digest mismatch")
    return {"entry_count": len(entries), "entries": entries, "sha256sums_sha256": digest}


def require_pass_status(obj: dict[str, Any], name: str) -> None:
    status = find_value(obj, ["status", "freeze_status", "bundle_load_status", "validation_status"])
    if status not in ("PASS", "FROZEN", "OK", True):
        fail(f"{name}: status is not PASS/FROZEN")


def require_non_actions(objs: list[dict[str, Any]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for obj in objs:
        flat = flatten(obj)
        for required_key, expected in REQUIRED_NON_ACTIONS.items():
            for key, value in flat.items():
                if key.lower().endswith(required_key.lower()) and value == expected:
                    observed[required_key] = expected
    missing = sorted(set(REQUIRED_NON_ACTIONS) - set(observed))
    if missing:
        fail(f"missing non-action markers: {missing}")
    return observed


def read_optional_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"{path.name}: empty CSV header")
        return sum(1 for _ in reader)


def validate_freeze(
    *,
    freeze_root: str | Path,
    expected_checkpoint_sha256: str,
    expected_dataset_csv_sha256: str,
    expected_split_csv_sha256: str,
    expected_state_index_sha256: str,
    expected_threshold: float,
    allow_parent_random_candidate: bool = False,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(freeze_root)
    if not root.is_dir():
        fail("freeze root does not exist")
    for rel in REQUIRED_FILES:
        if not (root / rel).is_file():
            fail(f"missing required freeze file: {rel}")
    sha_report = validate_sha256sums(root)

    freeze_manifest = read_json(root / "freeze_manifest.json")
    bundle_identity = read_json(root / "bundle_identity.json")
    checkpoint_identity = read_json(root / "checkpoint_identity.json")
    dataset_identity = read_json(root / "dataset_identity.json")
    split_identity = read_json(root / "split_identity.json")
    normalization_identity = read_json(root / "normalization_identity.json")
    threshold_identity = read_json(root / "threshold_identity.json")
    metrics_summary = read_json(root / "metrics_summary.json")
    bundle_load = read_json(root / "bundle_load_report.json")

    require_pass_status(freeze_manifest, "freeze_manifest.json")
    require_pass_status(bundle_load, "bundle_load_report.json")

    checkpoint_sha = find_value(checkpoint_identity, ["checkpoint_sha256", "best_checkpoint_sha256", "sha256"])
    if checkpoint_sha != expected_checkpoint_sha256:
        fail("checkpoint sha256 mismatch")
    dataset_sha = find_value(dataset_identity, ["dataset_csv_sha256"])
    split_sha = find_value(dataset_identity, ["split_csv_sha256"])
    state_sha = find_value(dataset_identity, ["state_index_sha256"])
    if split_sha is None:
        split_sha = find_value(split_identity, ["split_csv_sha256", "split_manifest_sha256"])
    if dataset_sha != expected_dataset_csv_sha256:
        fail("dataset_csv_sha256 mismatch")
    if split_sha != expected_split_csv_sha256:
        fail("split_csv_sha256 mismatch")
    if state_sha != expected_state_index_sha256:
        fail("state_index_sha256 mismatch")

    threshold = find_value(threshold_identity, ["threshold", "selected_threshold", "validation_selected_threshold"])
    try:
        threshold_float = float(threshold)
    except (TypeError, ValueError):
        fail("threshold_identity: missing numeric threshold")
    if abs(threshold_float - expected_threshold) > 1e-12:
        fail("threshold mismatch")
    threshold_source = find_value(threshold_identity, ["threshold_source", "selection_source", "selected_on"])
    if not isinstance(threshold_source, str) or threshold_source not in VALIDATION_THRESHOLD_SOURCES:
        fail("threshold must be validation-selected")

    norm_source = find_value(normalization_identity, ["normalization_source"])
    if norm_source != "train_only":
        fail("normalization_source must be train_only")

    split_types = find_split_types(split_identity, freeze_manifest)
    if not split_types:
        fail("split type missing")
    if split_types & CANDIDATE_SPLIT_TYPES and not allow_parent_random_candidate:
        fail("parent-random detector cannot be final-frozen without explicit candidate allowance")
    if not (split_types & FINAL_SPLIT_TYPES or split_types & CANDIDATE_SPLIT_TYPES):
        fail(f"unsupported split type(s): {sorted(split_types)}")

    non_actions = require_non_actions([
        freeze_manifest,
        bundle_identity,
        checkpoint_identity,
        dataset_identity,
        split_identity,
        normalization_identity,
        threshold_identity,
        metrics_summary,
        bundle_load,
    ])

    per_suite_rows = read_optional_csv_rows(root / "metrics_by_suite.csv")
    per_task_rows = read_optional_csv_rows(root / "metrics_by_task.csv")

    report = {
        "status": "PASS",
        "schema_version": "c4_detector_freeze_validation_v1",
        "freeze_root": str(root),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "dataset_csv_sha256": expected_dataset_csv_sha256,
        "split_csv_sha256": expected_split_csv_sha256,
        "state_index_sha256": expected_state_index_sha256,
        "threshold": threshold_float,
        "threshold_source": threshold_source,
        "split_types": sorted(split_types),
        "allow_parent_random_candidate": allow_parent_random_candidate,
        "normalization_source": norm_source,
        "per_suite_metric_rows": per_suite_rows,
        "per_task_metric_rows": per_task_rows,
        "sha256sums": sha_report,
        "non_actions": non_actions,
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-dataset-csv-sha256", required=True)
    parser.add_argument("--expected-split-csv-sha256", required=True)
    parser.add_argument("--expected-state-index-sha256", required=True)
    parser.add_argument("--expected-threshold", required=True, type=float)
    parser.add_argument("--allow-parent-random-candidate", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = validate_freeze(
            freeze_root=args.freeze_root,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_dataset_csv_sha256=args.expected_dataset_csv_sha256,
            expected_split_csv_sha256=args.expected_split_csv_sha256,
            expected_state_index_sha256=args.expected_state_index_sha256,
            expected_threshold=args.expected_threshold,
            allow_parent_random_candidate=args.allow_parent_random_candidate,
            output_json=args.output_json,
        )
    except (C4DetectorFreezeError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
