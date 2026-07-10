#!/usr/bin/env python3
"""Verify clean collection metadata remains bound to frozen policy bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.bind_c2g_collection_model_provenance import BINDING_SCHEMA
from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file
from scripts.stageb.verify_c2g_suite_model_map_strict import verify as verify_models


def verify_collection(
    *,
    collection_root: Path,
    binding_report_path: Path,
    model_map: Path,
    model_report: Path,
    goal_manifest: Path,
    model_verification_report: Path,
) -> dict[str, Any]:
    model_verification = verify_models(model_map, model_report, goal_manifest)
    recorded_verification = json.loads(
        model_verification_report.read_text(encoding="utf-8")
    )
    if recorded_verification.get("status") != "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION":
        raise ValueError("recorded model verification is not PASS")
    if recorded_verification.get("frozen_report_sha256") != sha256_file(model_report):
        raise ValueError("recorded model verification binds another model report")
    if recorded_verification.get("model_map_sha256") != sha256_file(model_map):
        raise ValueError("recorded model verification binds another model map")

    binding_report = json.loads(binding_report_path.read_text(encoding="utf-8"))
    if binding_report.get("status") != "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING":
        raise ValueError("collection binding report is not PASS")
    recorded_entries = binding_report.get("episodes")
    if not isinstance(recorded_entries, list):
        raise ValueError("collection binding report lacks episode entries")
    recorded = {str(row["path"]): row for row in recorded_entries}
    if len(recorded) != len(recorded_entries):
        raise ValueError("collection binding report contains duplicate paths")

    strict_report = json.loads(model_report.read_text(encoding="utf-8"))
    suite_models = strict_report.get("suite_models")
    if not isinstance(suite_models, Mapping):
        raise ValueError("strict model report lacks suite_models")
    expected_common = {
        "schema_version": BINDING_SCHEMA,
        "suite_model_map_path": str(model_map.resolve()),
        "suite_model_map_sha256": sha256_file(model_map),
        "suite_model_report_path": str(model_report.resolve()),
        "suite_model_report_sha256": sha256_file(model_report),
        "goal_model_manifest_path": str(goal_manifest.resolve()),
        "goal_model_manifest_sha256": sha256_file(goal_manifest),
        "model_verification_report_path": str(model_verification_report.resolve()),
        "model_verification_report_sha256": sha256_file(model_verification_report),
    }

    metadata_paths = sorted(collection_root.rglob("episode_metadata.json"))
    actual_paths = {
        path.relative_to(collection_root).as_posix(): path for path in metadata_paths
    }
    missing = sorted(set(recorded) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(recorded))
    if missing or unexpected:
        raise ValueError(
            f"collection metadata closure failed missing={missing} unexpected={unexpected}"
        )

    verified_rows: list[dict[str, Any]] = []
    for relative, path in sorted(actual_paths.items()):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} must contain a JSON object")
        suite = str(metadata.get("suite", ""))
        if suite not in SUITES:
            raise ValueError(f"{path} has invalid suite {suite!r}")
        frozen = suite_models.get(suite)
        if not isinstance(frozen, Mapping):
            raise ValueError(f"strict model report missing {suite}")
        expected_binding = {
            **expected_common,
            "suite": suite,
            "suite_model_path": str(frozen.get("model_path", "")),
            "suite_full_model_manifest_sha256": str(
                frozen.get("full_model_manifest_sha256", "")
            ),
        }
        if metadata.get("c2g_model_binding") != expected_binding:
            raise ValueError(f"{path} model binding differs from frozen contract")
        metadata_sha = sha256_file(path)
        entry = recorded[relative]
        if entry.get("metadata_sha256") != metadata_sha:
            raise ValueError(f"{path} SHA256 differs from binding report")
        if entry.get("suite_full_model_manifest_sha256") != expected_binding[
            "suite_full_model_manifest_sha256"
        ]:
            raise ValueError(f"{path} full model digest differs from binding report")
        verified_rows.append(
            {
                "path": relative,
                "suite": suite,
                "metadata_sha256": metadata_sha,
                "suite_full_model_manifest_sha256": expected_binding[
                    "suite_full_model_manifest_sha256"
                ],
            }
        )

    aggregate = hashlib.sha256(
        "".join(
            f"{row['path']}|{row['metadata_sha256']}|{row['suite_full_model_manifest_sha256']}\n"
            for row in verified_rows
        ).encode("utf-8")
    ).hexdigest()
    if aggregate != binding_report.get("episode_metadata_manifest_sha256"):
        raise ValueError("collection metadata aggregate differs from binding report")
    if int(binding_report.get("episode_count", -1)) != len(verified_rows):
        raise ValueError("collection episode count differs from binding report")
    return {
        "gate": "C2G_CLEAN_COLLECTION_MODEL_BINDING_VERIFICATION",
        "status": "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING_VERIFICATION",
        "collection_root": str(collection_root.resolve()),
        "binding_report": str(binding_report_path.resolve()),
        "binding_report_sha256": sha256_file(binding_report_path),
        "episode_count": len(verified_rows),
        "episode_metadata_manifest_sha256": aggregate,
        "model_verification": model_verification,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_collection(
        collection_root=args.collection_root.resolve(),
        binding_report_path=args.binding_report.resolve(),
        model_map=args.model_map.resolve(),
        model_report=args.model_report.resolve(),
        goal_manifest=args.goal_model_manifest.resolve(),
        model_verification_report=args.model_verification_report.resolve(),
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
