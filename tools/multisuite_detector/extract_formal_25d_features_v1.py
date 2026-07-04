#!/usr/bin/env python3
"""Build and validate the formal 25D clean-feature CSV from clean telemetry.

This is an implementation path only. It does not run simulation, inference,
training, rollout, attack, or GPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.detector_dataset_closure_v1 import (  # noqa: E402
    FEATURE_SCHEMA_SHA256,
    META_COLUMNS,
    SC5_FEATURES,
    load_feature_artifact,
)
from tools.multisuite_detector.load_label_v2_artifact import load_label_v2_artifact  # noqa: E402


SCHEMA_VERSION = "formal_25d_feature_artifact_v1"
OUTPUT_CSV = "formal_25d_features_v1.csv"
MANIFEST = "extraction_manifest.json"
SHA_FILE = "SHA256SUMS"
SOURCE_META = ["source_record_path", "source_condition", "initial_state_hash_provenance"]
SOURCE_COLUMNS = META_COLUMNS + SOURCE_META + SC5_FEATURES
OUTPUT_COLUMNS = META_COLUMNS + SC5_FEATURES
ALLOWED_STATE_PROVENANCE = {
    "SIMULATOR_RESET_STATE_SERIALIZED",
    "CLEAN_METADATA_RESET_STATE_HASH",
    "CANONICAL_INITIAL_SIM_STATE_FIELDS_HASH",
}
ALLOWED_CLEAN_CONDITIONS = {"CLEAN", "FORMAL_CLEAN", "CLEAN_ROLLOUT"}
FORBIDDEN_COLUMN_MARKERS = ("attack", "adversarial", "future", "teacher", "label", "split")
HEX = set("0123456789abcdef")


class FormalFeatureError(ValueError):
    pass


def fail(message: str) -> None:
    raise FormalFeatureError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_csv_strict(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            fail(f"{path.name}: expected exact header")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None or v == "" for v in row.values()):
                fail(f"{path.name}:{line_no}: empty field")
            rows.append(row)
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str, field: str, episode: str) -> int:
    try:
        out = int(value)
    except ValueError:
        fail(f"{episode}: {field} must be integer")
    if str(out) != value:
        fail(f"{episode}: {field} must be canonical integer")
    return out


def parse_float(value: str, field: str, episode: str) -> float:
    try:
        out = float(value)
    except ValueError:
        fail(f"{episode}: {field} must be finite float")
    if not math.isfinite(out):
        fail(f"{episode}: {field} must be finite float")
    return out


def validate_state_hash(row: dict[str, str]) -> None:
    episode = row["episode_key"]
    value = row["initial_state_hash"]
    if len(value) != 64 or any(ch not in HEX for ch in value):
        fail(f"{episode}: initial_state_hash must be 64 lowercase hex")
    for field in ("episode_key", "parent_key", "source_record_path"):
        if hashlib.sha256(row[field].encode()).hexdigest() == value:
            fail(f"{episode}: initial_state_hash is derived from forbidden {field}")
    if row["initial_state_hash_provenance"] not in ALLOWED_STATE_PROVENANCE:
        fail(f"{episode}: initial_state_hash provenance is not bound")


def resolve_source_path(raw: str, approved_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = approved_root / path
    resolved = path.resolve()
    root = approved_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        fail(f"source path outside approved root: {raw}")
    if not resolved.is_file():
        fail(f"source record does not exist: {raw}")
    return resolved


def reject_forbidden_columns(columns: list[str]) -> None:
    allowed = set(SOURCE_COLUMNS)
    extras = [c for c in columns if c not in allowed]
    for col in extras:
        lower = col.lower()
        if any(marker in lower for marker in FORBIDDEN_COLUMN_MARKERS):
            fail(f"forbidden source column: {col}")
    if extras:
        fail(f"unknown source columns: {','.join(extras)}")


def load_source_rows(source_csv: Path, approved_root: Path) -> list[dict[str, str]]:
    with source_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
    reject_forbidden_columns(columns)
    rows = read_csv_strict(source_csv, SOURCE_COLUMNS)
    for row in rows:
        episode = row["episode_key"]
        if row["source_condition"] not in ALLOWED_CLEAN_CONDITIONS:
            fail(f"{episode}: source condition is not clean")
        resolve_source_path(row["source_record_path"], approved_root)
        validate_state_hash(row)
        trace_length = parse_int(row["trace_length"], "trace_length", episode)
        step = parse_int(row["step"], "step", episode)
        if trace_length <= 0 or step < 0 or step >= trace_length:
            fail(f"{episode}: invalid step coverage")
        for name in SC5_FEATURES:
            parse_float(row[name], name, episode)
    return rows


def validate_against_label(rows: list[dict[str, str]], label_root: Path, expected_label_mode: str) -> dict[str, object]:
    label = load_label_v2_artifact(label_root, expected_mode=expected_label_mode)
    label_rows = {row["episode_key"]: row for row in label["label_rows"]}
    episodes = {row["episode_key"] for row in rows}
    if episodes != set(label_rows):
        fail("feature source episode set does not match Label V2")
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_key"], []).append(row)
    for episode, step_rows in by_episode.items():
        label_row = label_rows[episode]
        first = step_rows[0]
        for field in ("parent_key", "suite", "task_id", "trace_length"):
            if first[field] != label_row[field]:
                fail(f"{episode}: {field} mismatch against Label V2")
        trace_length = int(first["trace_length"])
        steps = [int(row["step"]) for row in step_rows]
        if sorted(steps) != list(range(trace_length)):
            fail(f"{episode}: steps must cover 0..trace_length-1")
        if len(steps) != len(set(steps)):
            fail(f"{episode}: duplicate step")
    return label


def build_feature_artifact(source_csv: str | Path, label_root: str | Path, output_root: str | Path, approved_source_root: str | Path, *, expected_label_mode: str = "synthetic-dry-run") -> dict[str, object]:
    source_csv = Path(source_csv)
    label_root = Path(label_root)
    output_root = Path(output_root)
    approved_root = Path(approved_source_root)
    if output_root.exists() and any(output_root.iterdir()):
        fail("output root must be empty")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_source_rows(source_csv, approved_root)
    label = validate_against_label(rows, label_root, expected_label_mode)
    out_rows = [{col: row[col] for col in OUTPUT_COLUMNS} for row in rows]
    out_csv = output_root / OUTPUT_CSV
    write_csv(out_csv, OUTPUT_COLUMNS, out_rows)
    loaded = load_feature_artifact(out_csv)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_csv_path": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "approved_source_root": str(approved_root),
        "label_v2_artifact_root": str(label_root),
        "label_v2_artifact_sha256": label_artifact_sha(label_root),
        "feature_csv": OUTPUT_CSV,
        "feature_csv_sha256": sha256_file(out_csv),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_names": list(SC5_FEATURES),
        "feature_count": len(SC5_FEATURES),
        "episode_count": len(loaded["episodes"]),
        "row_count": len(out_rows),
        "initial_state_hash_provenance": "BOUND",
        "exact_set_join": "PASS",
        "finite_feature_values": "PASS",
        "formal_feature_artifact_build": "PASS",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }
    (output_root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_sha256sums(output_root)
    return manifest


def write_sha256sums(root: Path) -> None:
    names = [OUTPUT_CSV, MANIFEST]
    (root / SHA_FILE).write_text("".join(f"{sha256_file(root / name)}  {name}\n" for name in names), encoding="utf-8")


def label_artifact_sha(label_root: Path) -> str:
    return sha256_file(label_root / "SHA256SUMS")


def validate_sha256sums(root: Path) -> None:
    seen = set()
    with (root / SHA_FILE).open(encoding="utf-8") as f:
        for line in f:
            digest, name = line.strip().split(maxsplit=1)
            seen.add(name)
            if sha256_file(root / name) != digest:
                fail(f"SHA256SUMS mismatch: {name}")
    if seen != {OUTPUT_CSV, MANIFEST}:
        fail("SHA256SUMS file set mismatch")


def validate_feature_artifact(artifact_root: str | Path, label_root: str | Path, *, expected_label_mode: str = "synthetic-dry-run") -> dict[str, object]:
    artifact_root = Path(artifact_root)
    label_root = Path(label_root)
    validate_sha256sums(artifact_root)
    manifest = json.loads((artifact_root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("manifest schema_version mismatch")
    if manifest.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        fail("feature schema SHA mismatch")
    if manifest.get("label_v2_artifact_sha256") != label_artifact_sha(label_root):
        fail("label artifact SHA mismatch")
    if manifest.get("initial_state_hash_provenance") != "BOUND":
        fail("initial_state_hash provenance is not BOUND")
    if manifest.get("formal_detector_dataset_build") != "NOT_PERFORMED" or manifest.get("training") != "NOT_PERFORMED" or manifest.get("gpu") != "NOT_PERFORMED":
        fail("manifest execution boundary mismatch")
    feature_csv = artifact_root / manifest["feature_csv"]
    if manifest.get("feature_csv_sha256") != sha256_file(feature_csv):
        fail("feature CSV SHA mismatch")
    source_csv = Path(manifest["source_csv_path"])
    if not source_csv.is_file() or manifest.get("source_csv_sha256") != sha256_file(source_csv):
        fail("source CSV SHA mismatch")
    features = load_feature_artifact(feature_csv)
    label = load_label_v2_artifact(label_root, expected_mode=expected_label_mode)
    label_rows = {row["episode_key"]: row for row in label["label_rows"]}
    if set(features["episodes"]) != set(label_rows):
        fail("feature artifact episode set does not match Label V2")
    for episode, feature in features["episodes"].items():
        label_row = label_rows[episode]
        for field in ("parent_key", "suite", "task_id", "trace_length"):
            if str(feature[field]) != str(label_row[field]):
                fail(f"{episode}: {field} mismatch against Label V2")
    report = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "label_v2_artifact_sha256": label_artifact_sha(label_root),
        "feature_csv_sha256": sha256_file(feature_csv),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "episode_count": len(features["episodes"]),
        "row_count": sum(len(record["steps"]) for record in features["episodes"].values()),
        "initial_state_hash_provenance": "BOUND",
        "exact_set_join": "PASS",
        "finite_feature_values": "PASS",
        "formal_feature_artifact_build": "PASS",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }
    return report


def audit_source_schema(candidate: str | Path, output_json: str | Path | None = None) -> dict[str, object]:
    path = Path(candidate)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        sample = [row for _, row in zip(range(3), reader)]
    report = {
        "status": "PASS" if columns == SOURCE_COLUMNS else "HOLD",
        "candidate_path": str(path),
        "candidate_sha256": sha256_file(path),
        "columns": columns,
        "row_sample_count": len(sample),
        "has_exact_source_schema": columns == SOURCE_COLUMNS,
        "has_exact_sc5_order": columns[-25:] == SC5_FEATURES,
        "formal_extraction": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }
    if output_json:
        Path(output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def print_json(obj: object) -> int:
    print(json.dumps(obj, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--source-csv", required=True)
    p.add_argument("--label-artifact-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--approved-source-root", required=True)
    p.add_argument("--expected-label-mode", default="synthetic-dry-run")
    p = sub.add_parser("validate")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--label-artifact-root", required=True)
    p.add_argument("--expected-label-mode", default="synthetic-dry-run")
    p = sub.add_parser("audit-source")
    p.add_argument("--candidate", required=True)
    p.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "extract":
            return print_json(build_feature_artifact(args.source_csv, args.label_artifact_root, args.output_root, args.approved_source_root, expected_label_mode=args.expected_label_mode))
        if args.cmd == "validate":
            return print_json(validate_feature_artifact(args.artifact_root, args.label_artifact_root, expected_label_mode=args.expected_label_mode))
        if args.cmd == "audit-source":
            return print_json(audit_source_schema(args.candidate, args.output_json))
    except (FormalFeatureError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
