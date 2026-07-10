#!/usr/bin/env python3
"""Bind every clean collection episode to the frozen full suite model manifest.

This command runs after clean collection and before Teacher-v2 audit/materialization.
It atomically adds the suite model map/report, Goal manifest, verification report,
and per-suite full model digest to each episode metadata file.  The materializer's
input manifest therefore cryptographically binds the training dataset to the exact
policy bytes used to produce clean actions and policy-intent features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file
from scripts.stageb.verify_c2g_suite_model_map_strict import verify

BINDING_SCHEMA = "c2g.clean_collection_model_binding.2026-07-10.v1"


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bind(
    collection_root: Path,
    model_map: Path,
    model_report: Path,
    goal_manifest: Path,
    verification_report: Path,
) -> dict[str, Any]:
    verification = verify(model_map, model_report, goal_manifest)
    if verification_report.is_file():
        recorded = json.loads(verification_report.read_text(encoding="utf-8"))
        if recorded.get("status") != "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION":
            raise ValueError("existing model verification report is not PASS")
        if recorded.get("frozen_report_sha256") != sha256_file(model_report):
            raise ValueError("existing model verification report binds another frozen report")
    else:
        verification_report.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(verification_report, verification)

    report_value = json.loads(model_report.read_text(encoding="utf-8"))
    suite_models = report_value.get("suite_models")
    if not isinstance(suite_models, Mapping):
        raise ValueError("strict suite model report lacks suite_models")
    binding_common = {
        "schema_version": BINDING_SCHEMA,
        "suite_model_map_path": str(model_map.resolve()),
        "suite_model_map_sha256": sha256_file(model_map),
        "suite_model_report_path": str(model_report.resolve()),
        "suite_model_report_sha256": sha256_file(model_report),
        "goal_model_manifest_path": str(goal_manifest.resolve()),
        "goal_model_manifest_sha256": sha256_file(goal_manifest),
        "model_verification_report_path": str(verification_report.resolve()),
        "model_verification_report_sha256": sha256_file(verification_report),
    }
    metadata_paths = sorted(collection_root.rglob("episode_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata found under {collection_root}")
    updated: list[dict[str, Any]] = []
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} must contain a JSON object")
        suite = str(metadata.get("suite", ""))
        if suite not in SUITES:
            raise ValueError(f"{path} has invalid suite {suite!r}")
        frozen = suite_models.get(suite)
        if not isinstance(frozen, Mapping):
            raise ValueError(f"strict model report missing {suite}")
        full_digest = str(frozen.get("full_model_manifest_sha256", ""))
        if len(full_digest) != 64:
            raise ValueError(f"invalid full model digest for {suite}")
        binding = {
            **binding_common,
            "suite": suite,
            "suite_model_path": str(frozen.get("model_path", "")),
            "suite_full_model_manifest_sha256": full_digest,
        }
        existing = metadata.get("c2g_model_binding")
        if existing is not None and existing != binding:
            raise ValueError(f"{path} already contains a conflicting model binding")
        metadata["c2g_model_binding"] = binding
        atomic_json_write(path, metadata)
        updated.append(
            {
                "path": path.relative_to(collection_root).as_posix(),
                "suite": suite,
                "metadata_sha256": sha256_file(path),
                "suite_full_model_manifest_sha256": full_digest,
            }
        )
    aggregate = hashlib.sha256(
        "".join(
            f"{row['path']}|{row['metadata_sha256']}|{row['suite_full_model_manifest_sha256']}\n"
            for row in updated
        ).encode("utf-8")
    ).hexdigest()
    return {
        "gate": "C2G_CLEAN_COLLECTION_MODEL_BINDING",
        "status": "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING",
        "collection_root": str(collection_root.resolve()),
        "episode_count": len(updated),
        "binding": binding_common,
        "episode_metadata_manifest_sha256": aggregate,
        "episodes": updated,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = bind(
        args.collection_root.resolve(),
        args.model_map.resolve(),
        args.model_report.resolve(),
        args.goal_model_manifest.resolve(),
        args.model_verification_report.resolve(),
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.output_report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
