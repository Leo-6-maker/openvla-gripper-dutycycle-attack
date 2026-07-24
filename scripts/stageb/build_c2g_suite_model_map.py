#!/usr/bin/env python3
"""Freeze the exact per-suite OpenVLA model map used by C2g.

The command is CPU-only and performs no model loading. It resolves the existing
repository suite map, verifies required model metadata files, hashes selected model
and processor configuration artifacts, and optionally requires the audited Goal
model-integrity manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
HASHED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processing_prismatic.py",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
)
LEGACY_GOAL_STATUS = "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED"
V2_GOAL_STATUS = "PASS_C2G_GOAL_MODEL_INTEGRITY_AUDITED_V2"
V2_GOAL_SCHEMA = "c2g.goal_model_integrity.2026-07-10.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_model_manifest(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    files: list[dict[str, Any]] = []
    for name in HASHED_FILES:
        path = model_path / name
        if path.is_file():
            files.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not any(row["name"] == "config.json" for row in files):
        raise FileNotFoundError(f"config.json missing from {model_path}")
    if not any(row["name"] in {"tokenizer.json", "tokenizer.model"} for row in files):
        raise FileNotFoundError(f"tokenizer artifact missing from {model_path}")
    aggregate = hashlib.sha256(
        "".join(
            f"{row['name']}|{row['bytes']}|{row['sha256']}\n"
            for row in sorted(files, key=lambda value: value["name"])
        ).encode("utf-8")
    ).hexdigest()
    return {
        "model_path": str(model_path),
        "selected_file_count": len(files),
        "selected_files": files,
        "selected_manifest_sha256": aggregate,
    }


def _verify_goal_manifest_files(
    value: Mapping[str, Any], model_path: Path
) -> tuple[int, str]:
    rows = value.get("files")
    if rows is None:
        # Backward-compatible synthetic/legacy fixtures may not include a file ledger.
        # Real audited manifests do, and are verified below whenever present.
        return 0, ""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Goal model manifest files ledger must be a nonempty list")
    model_path = model_path.resolve()
    aggregate = hashlib.sha256()
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Goal model manifest file row must be an object")
        relative = str(
            row.get("relative_path") or Path(str(row.get("path", ""))).name
        ).strip()
        if not relative or relative in seen:
            raise ValueError(f"invalid or duplicate Goal manifest relative path: {relative!r}")
        seen.add(relative)
        path = (model_path / relative).resolve()
        try:
            path.relative_to(model_path)
        except ValueError as exc:
            raise ValueError(f"Goal manifest path escapes model directory: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Goal manifest file missing: {path}")
        expected_size = int(row.get("size_bytes", -1))
        expected_sha = str(row.get("sha256", ""))
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise ValueError(
                "GOAL_MANIFEST_FILE_HASH_MISMATCH: "
                f"{relative}: expected size/hash {expected_size}/{expected_sha}, "
                f"got {actual_size}/{actual_sha}"
            )
        aggregate.update(
            f"{relative}|{actual_size}|{actual_sha}\n".encode("utf-8")
        )
    referenced = value.get("referenced_shards", [])
    if referenced:
        if not isinstance(referenced, list):
            raise ValueError("Goal referenced_shards must be a list")
        missing_ledger = sorted(set(str(item) for item in referenced) - seen)
        if missing_ledger:
            raise ValueError(
                "Goal manifest file ledger omits referenced shards: "
                + ", ".join(missing_ledger)
            )
    expected_aggregate = str(value.get("files_aggregate_sha256", ""))
    actual_aggregate = aggregate.hexdigest()
    if expected_aggregate and expected_aggregate != actual_aggregate:
        raise ValueError(
            "Goal manifest aggregate SHA256 mismatch: "
            f"{expected_aggregate} != {actual_aggregate}"
        )
    return len(seen), actual_aggregate


def validate_goal_manifest(path: Path, model_path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Goal model manifest must be a JSON object")
    status = str(value.get("status", ""))
    if status not in {LEGACY_GOAL_STATUS, V2_GOAL_STATUS}:
        raise ValueError("Goal model manifest status is not an accepted PASS status")
    recorded = Path(str(value.get("model_path", ""))).resolve()
    model_path = model_path.resolve()
    if recorded != model_path:
        raise ValueError(f"Goal model path mismatch: {recorded} != {model_path}")
    if value.get("missing_referenced_shards"):
        raise ValueError("Goal model manifest reports missing shards")
    verified_file_count, files_aggregate = _verify_goal_manifest_files(value, model_path)

    provenance_mode = str(value.get("provenance_mode", ""))
    if status == V2_GOAL_STATUS:
        if value.get("schema_version") != V2_GOAL_SCHEMA:
            raise ValueError("Goal v2 manifest schema mismatch")
        if provenance_mode not in {
            "RESTORED_FROZEN_BYTES",
            "EXPLICIT_REBASE_CURRENT_BYTES",
        }:
            raise ValueError("Goal v2 manifest provenance_mode is invalid")
        load_audit = value.get("load_only_validation")
        if not isinstance(load_audit, Mapping) or load_audit.get("status") != "PASS_C2G_GOAL_MODEL_LOAD_ONLY":
            raise ValueError("Goal v2 manifest lacks a PASS load-only validation")
        if int(load_audit.get("parameter_count", 0)) <= 0:
            raise ValueError("Goal v2 load-only validation has no parameters")
        if not str(load_audit.get("token_semantics_sha256", "")):
            raise ValueError("Goal v2 load-only validation lacks token semantics hash")
        boundaries = value.get("boundaries")
        if not isinstance(boundaries, Mapping):
            raise ValueError("Goal v2 manifest lacks execution boundaries")
        if int(boundaries.get("libero_rollouts_launched", -1)) != 0:
            raise ValueError("Goal v2 manifest reports a LIBERO rollout during load audit")
        if int(boundaries.get("attacks_launched", -1)) != 0:
            raise ValueError("Goal v2 manifest reports an attack during load audit")
        if bool(boundaries.get("attack_outcomes_read", True)):
            raise ValueError("Goal v2 manifest reports attack-outcome access")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": status,
        "schema_version": value.get("schema_version", "legacy"),
        "unnorm_key": value.get("unnorm_key"),
        "provenance_mode": provenance_mode or "LEGACY_FROZEN_BYTES",
        "verified_file_count": verified_file_count,
        "files_aggregate_sha256": files_aggregate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="override one suite path as suite=/absolute/model/path",
    )
    parser.add_argument(
        "--require-goal-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS

    model_map = {suite: str(Path(SUITE_MODELS[suite]).resolve()) for suite in SUITES}
    for item in args.override:
        if "=" not in item:
            raise ValueError("--override must be suite=/absolute/model/path")
        suite, raw_path = item.split("=", 1)
        if suite not in SUITES:
            raise ValueError(f"unknown suite override: {suite}")
        model_map[suite] = str(Path(raw_path).resolve())

    manifests: dict[str, Any] = {}
    for suite in SUITES:
        manifests[suite] = selected_model_manifest(Path(model_map[suite]))
    goal_manifest = None
    if args.goal_model_manifest is not None:
        goal_manifest = validate_goal_manifest(
            args.goal_model_manifest.resolve(),
            Path(model_map["libero_goal"]),
        )
    elif args.require_goal_manifest:
        raise ValueError("--goal-model-manifest is required for the primary four-suite map")

    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(
        json.dumps(model_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "gate": "C2G_SUITE_MODEL_MAP",
        "status": "PASS_C2G_SUITE_MODEL_MAP",
        "model_map": str(args.output_map.resolve()),
        "model_map_sha256": sha256_file(args.output_map.resolve()),
        "suite_models": manifests,
        "goal_model_manifest": goal_manifest,
        "openvla_models_loaded": 0,
        "gpu_jobs_launched": 0,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
