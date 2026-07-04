#!/usr/bin/env python3
"""Build the exact C2-FX source CSV from clean per-step features.

This tool only transforms already-existing clean telemetry tables into the
C2-FX exact-source schema. It does not run simulation, model inference,
detector training, detector dataset construction, rollout, or GPU work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.extract_formal_25d_features_v1 import (  # noqa: E402
    ALLOWED_CLEAN_CONDITIONS,
    ALLOWED_STATE_PROVENANCE,
    FormalFeatureError,
    SC5_FEATURES,
    SOURCE_COLUMNS,
    load_source_rows,
    sha256_file,
    validate_against_label,
)
from tools.multisuite_detector.load_label_v2_artifact import (  # noqa: E402
    LabelV2ArtifactError,
    load_label_v2_artifact,
)


SCHEMA_VERSION = "formal_25d_source_csv_builder_v1"
FEATURE_INPUT_COLUMNS = ["episode_key", "step", "source_record_path", "source_condition"] + SC5_FEATURES
STATE_INPUT_COLUMNS = ["episode_key", "initial_state_hash", "initial_state_hash_provenance"]


class SourceCsvBuildError(ValueError):
    pass


def fail(message: str) -> None:
    raise SourceCsvBuildError(message)


def read_csv_exact(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            fail(f"{path.name}: header mismatch")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            empty = [key for key, value in row.items() if value in (None, "")]
            if empty:
                fail(f"{path.name}:{line_no}: empty fields: {empty}")
            rows.append(row)
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict[str, object], report_json: Path | None, sha256_output: Path | None) -> None:
    if report_json is None:
        return
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sha256_output is not None:
        sha256_output.parent.mkdir(parents=True, exist_ok=True)
        sha256_output.write_text(f"{sha256_file(report_json)}  {report_json.name}\n", encoding="utf-8")


def parse_step(value: str, episode: str) -> int:
    try:
        step = int(value)
    except ValueError:
        fail(f"{episode}: step must be integer")
    if str(step) != value or step < 0:
        fail(f"{episode}: step must be canonical nonnegative integer")
    return step


def build_source_csv(
    *,
    label_artifact_root: str | Path,
    per_step_features_csv: str | Path,
    state_metadata_csv: str | Path,
    approved_source_root: str | Path,
    output_csv: str | Path,
    report_json: str | Path | None = None,
    sha256_output: str | Path | None = None,
    expected_label_mode: str = "formal-ledger-build",
) -> dict[str, object]:
    label_root = Path(label_artifact_root)
    feature_csv = Path(per_step_features_csv)
    state_csv = Path(state_metadata_csv)
    source_root = Path(approved_source_root)
    out_csv = Path(output_csv)
    report_path = Path(report_json) if report_json else None
    sha_path = Path(sha256_output) if sha256_output else None

    if out_csv.exists():
        fail("output CSV already exists")

    label = load_label_v2_artifact(label_root, expected_mode=expected_label_mode)
    label_rows = {row["episode_key"]: row for row in label["label_rows"]}
    state_rows = read_csv_exact(state_csv, STATE_INPUT_COLUMNS)
    state_by_episode = {}
    for row in state_rows:
        episode = row["episode_key"]
        if episode in state_by_episode:
            fail(f"duplicate state metadata episode: {episode}")
        if row["initial_state_hash_provenance"] not in ALLOWED_STATE_PROVENANCE:
            fail(f"{episode}: unsupported initial_state_hash_provenance")
        state_by_episode[episode] = row

    if set(state_by_episode) != set(label_rows):
        fail("state metadata episode set does not match Label V2")

    feature_rows = read_csv_exact(feature_csv, FEATURE_INPUT_COLUMNS)
    by_episode: dict[str, list[dict[str, str]]] = {}
    seen_step: set[tuple[str, int]] = set()
    for row in feature_rows:
        episode = row["episode_key"]
        if episode not in label_rows:
            fail(f"feature row episode not in Label V2: {episode}")
        if row["source_condition"] not in ALLOWED_CLEAN_CONDITIONS:
            fail(f"{episode}: source condition is not clean")
        step = parse_step(row["step"], episode)
        key = (episode, step)
        if key in seen_step:
            fail(f"{episode}: duplicate step {step}")
        seen_step.add(key)
        by_episode.setdefault(episode, []).append(row)

    if set(by_episode) != set(label_rows):
        fail("feature episode set does not match Label V2")

    out_rows: list[dict[str, str]] = []
    for episode, label_row in label_rows.items():
        trace_length = int(label_row["trace_length"])
        rows = by_episode[episode]
        steps = sorted(parse_step(row["step"], episode) for row in rows)
        if steps != list(range(trace_length)):
            fail(f"{episode}: steps must cover 0..trace_length-1")
        state = state_by_episode[episode]
        for row in sorted(rows, key=lambda item: parse_step(item["step"], episode)):
            built = {
                "episode_key": episode,
                "parent_key": label_row["parent_key"],
                "suite": label_row["suite"],
                "task_id": label_row["task_id"],
                "initial_state_hash": state["initial_state_hash"],
                "trace_length": label_row["trace_length"],
                "step": row["step"],
                "source_record_path": row["source_record_path"],
                "source_condition": row["source_condition"],
                "initial_state_hash_provenance": state["initial_state_hash_provenance"],
            }
            for feature_name in SC5_FEATURES:
                built[feature_name] = row[feature_name]
            out_rows.append(built)

    write_csv(out_csv, SOURCE_COLUMNS, out_rows)
    validated_rows = load_source_rows(out_csv, source_root)
    validate_against_label(validated_rows, label_root, expected_label_mode)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "label_v2_artifact_root": str(label_root),
        "label_v2_sha256sums_sha256": sha256_file(label_root / "SHA256SUMS"),
        "per_step_features_csv": str(feature_csv),
        "per_step_features_csv_sha256": sha256_file(feature_csv),
        "state_metadata_csv": str(state_csv),
        "state_metadata_csv_sha256": sha256_file(state_csv),
        "approved_source_root": str(source_root),
        "output_csv": str(out_csv),
        "output_csv_sha256": sha256_file(out_csv),
        "episode_count": len(label_rows),
        "row_count": len(out_rows),
        "feature_count": len(SC5_FEATURES),
        "source_columns": list(SOURCE_COLUMNS),
        "source_columns_count": len(SOURCE_COLUMNS),
        "exact_source_csv_build": "PASS",
        "formal_feature_extraction": "NOT_PERFORMED",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }
    write_report(report, report_path, sha_path)
    return report


def print_json(obj: object) -> int:
    print(json.dumps(obj, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-artifact-root", required=True)
    parser.add_argument("--per-step-features-csv", required=True)
    parser.add_argument("--state-metadata-csv", required=True)
    parser.add_argument("--approved-source-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--sha256-output")
    parser.add_argument("--expected-label-mode", default="formal-ledger-build")
    args = parser.parse_args(argv)
    try:
        return print_json(
            build_source_csv(
                label_artifact_root=args.label_artifact_root,
                per_step_features_csv=args.per_step_features_csv,
                state_metadata_csv=args.state_metadata_csv,
                approved_source_root=args.approved_source_root,
                output_csv=args.output_csv,
                report_json=args.report_json,
                sha256_output=args.sha256_output,
                expected_label_mode=args.expected_label_mode,
            )
        )
    except (SourceCsvBuildError, FormalFeatureError, LabelV2ArtifactError, OSError, json.JSONDecodeError, csv.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
