#!/usr/bin/env python3
"""Prepare a read-only readiness report for formal 25D feature extraction.

This tool does not run formal extraction, detector dataset construction, model
training, rollout, simulator work, or GPU work. It only binds identities and, if
provided, audits an exact-source CSV that can later feed the C2-FX extractor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.extract_formal_25d_features_v1 import (  # noqa: E402
    FormalFeatureError,
    SC5_FEATURES,
    SOURCE_COLUMNS,
    audit_source_schema,
    load_source_rows,
    sha256_file,
    validate_against_label,
)
from tools.multisuite_detector.load_label_v2_artifact import (  # noqa: E402
    LabelV2ArtifactError,
    load_label_v2_artifact,
)


READY_SCHEMA_VERSION = "formal_25d_feature_extraction_readiness_v1"
DEFAULT_PROPOSED_OUTPUT = "/mnt/sdc/dty_user/openvla_attack_outputs/formal_25d_features_af8217c"


class ReadinessError(ValueError):
    pass


def fail(message: str) -> None:
    raise ReadinessError(message)


def artifact_sums_status(label_root: Path, expected_label_mode: str) -> dict[str, object]:
    artifact = load_label_v2_artifact(label_root, expected_mode=expected_label_mode)
    return {
        "status": "PASS",
        "label_rows": len(artifact["label_rows"]),
        "manual_rows": len(artifact["manual_audit_rows"]),
        "artifact_sha256sums_sha256": sha256_file(label_root / "SHA256SUMS"),
    }


def classify_source_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "initial_state_hash" in message or "provenance" in message:
        return "HOLD_MISSING_INITIAL_STATE_PROVENANCE"
    return "HOLD_MISSING_FEATURE_INPUTS"


def write_report(report: dict[str, object], output_json: Path, sha256_output: Path | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sha256_output is not None:
        sha256_output.parent.mkdir(parents=True, exist_ok=True)
        sha256_output.write_text(f"{sha256_file(output_json)}  {output_json.name}\n", encoding="utf-8")


def build_readiness_report(
    *,
    label_artifact_root: str | Path,
    repo_head: str,
    extractor_path: str | Path,
    validator_path: str | Path,
    output_json: str | Path,
    sha256_output: str | Path | None = None,
    exact_source_csv: str | Path | None = None,
    approved_source_root: str | Path | None = None,
    proposed_output_root: str = DEFAULT_PROPOSED_OUTPUT,
    expected_label_mode: str = "formal-ledger-build",
) -> dict[str, object]:
    label_root = Path(label_artifact_root)
    extractor = Path(extractor_path)
    validator = Path(validator_path)
    output_path = Path(output_json)
    sha_path = Path(sha256_output) if sha256_output else None

    if not extractor.is_file():
        fail(f"extractor path missing: {extractor}")
    if not validator.is_file():
        fail(f"validator path missing: {validator}")

    label_status = artifact_sums_status(label_root, expected_label_mode)
    report: dict[str, object] = {
        "schema_version": READY_SCHEMA_VERSION,
        "status": "HOLD_NEEDS_SOURCE_CSV_CONSTRUCTION",
        "gate_a1": "PASS_FROZEN_FOR_DETECTOR_DATASET_BUILD",
        "repo_head": repo_head,
        "extractor_path": str(extractor),
        "extractor_sha256": sha256_file(extractor),
        "validator_path": str(validator),
        "validator_sha256": sha256_file(validator),
        "label_v2_artifact": str(label_root),
        "label_v2_sha256sums_status": label_status["status"],
        "label_v2_artifact_sha256sums_sha256": label_status["artifact_sha256sums_sha256"],
        "exact_source_csv": None,
        "approved_source_root": str(approved_source_root) if approved_source_root else None,
        "output_root_proposed": proposed_output_root,
        "formal_feature_extraction": "NOT_PERFORMED",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
        "recommendation": "AUTHORIZE_SOURCE_CSV_CONSTRUCTION",
    }

    if exact_source_csv is None:
        write_report(report, output_path, sha_path)
        return report

    if approved_source_root is None:
        fail("approved_source_root is required when exact_source_csv is provided")

    source_csv = Path(exact_source_csv)
    source_root = Path(approved_source_root)
    if not source_csv.is_file():
        report["status"] = "HOLD_NEEDS_SOURCE_CSV_CONSTRUCTION"
        report["recommendation"] = "AUTHORIZE_SOURCE_CSV_CONSTRUCTION"
        report["source_error"] = f"exact source CSV missing: {source_csv}"
        write_report(report, output_path, sha_path)
        return report

    audit = audit_source_schema(source_csv)
    report["source_schema_audit"] = audit
    if not audit.get("has_exact_source_schema"):
        report["status"] = "HOLD_MISSING_FEATURE_INPUTS"
        report["recommendation"] = "AUTHORIZE_SOURCE_CSV_CONSTRUCTION"
        write_report(report, output_path, sha_path)
        return report

    try:
        source_rows = load_source_rows(source_csv, source_root)
        validate_against_label(source_rows, label_root, expected_label_mode)
    except (FormalFeatureError, LabelV2ArtifactError, OSError, json.JSONDecodeError) as exc:
        report["status"] = classify_source_error(exc)
        report["recommendation"] = "BLOCKED_NEEDS_SCIENTIFIC_DECISION"
        report["source_error"] = str(exc)
        write_report(report, output_path, sha_path)
        return report

    episodes = {row["episode_key"] for row in source_rows}
    provenance_values = sorted({row["initial_state_hash_provenance"] for row in source_rows})
    report.update(
        {
            "status": "READY_EXACT_SOURCE_CSV_BOUND",
            "exact_source_csv": {
                "path": str(source_csv),
                "sha256": sha256_file(source_csv),
                "row_count": len(source_rows),
                "episode_count": len(episodes),
                "has_exact_schema": True,
                "has_exact_sc5_order": True,
                "initial_state_hash_provenance": "BOUND",
                "initial_state_hash_provenance_values": provenance_values,
                "feature_count": len(SC5_FEATURES),
                "source_columns_sha256": sha256_file(source_csv),
            },
            "approved_source_root": str(source_root),
            "recommendation": "AUTHORIZE_EXTRACTION",
        }
    )
    write_report(report, output_path, sha_path)
    return report


def print_json(obj: object) -> int:
    print(json.dumps(obj, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-artifact-root", required=True)
    parser.add_argument("--repo-head", required=True)
    parser.add_argument("--extractor-path", default="tools/multisuite_detector/extract_formal_25d_features_v1.py")
    parser.add_argument("--validator-path", default="tools/multisuite_detector/validate_formal_25d_features_v1.py")
    parser.add_argument("--exact-source-csv")
    parser.add_argument("--approved-source-root")
    parser.add_argument("--proposed-output-root", default=DEFAULT_PROPOSED_OUTPUT)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sha256-output")
    parser.add_argument("--expected-label-mode", default="formal-ledger-build")
    args = parser.parse_args(argv)
    try:
        return print_json(
            build_readiness_report(
                label_artifact_root=args.label_artifact_root,
                repo_head=args.repo_head,
                extractor_path=args.extractor_path,
                validator_path=args.validator_path,
                exact_source_csv=args.exact_source_csv,
                approved_source_root=args.approved_source_root,
                proposed_output_root=args.proposed_output_root,
                output_json=args.output_json,
                sha256_output=args.sha256_output,
                expected_label_mode=args.expected_label_mode,
            )
        )
    except (ReadinessError, FormalFeatureError, LabelV2ArtifactError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
