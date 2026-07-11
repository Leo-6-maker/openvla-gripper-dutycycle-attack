#!/usr/bin/env python3
"""Fail-closed static integrity audit for the current LIBERO-Goal OpenVLA bytes.

This command is read-only and does not load OpenVLA. It verifies the safetensors
index/header structure, tensor-to-shard closure, lightweight model metadata, and full
file hashes. When a prior manifest is supplied, any byte drift is reported explicitly.
A later load-only smoke must finalize a re-based manifest before the C2g pipeline may
use changed Goal bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "c2g.goal_model_static_integrity.2026-07-10.v2"
PASS_STATUS = "PASS_C2G_GOAL_MODEL_STATIC_INTEGRITY_V2"
HOLD_STATUS = "HOLD_C2G_GOAL_MODEL_STATIC_INTEGRITY_V2"
LIGHTWEIGHT_REQUIRED = (
    "config.json",
    "dataset_statistics.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(payload, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def _safe_relative_files(model_path: Path) -> list[Path]:
    files = [path for path in model_path.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: path.name)


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 10:
        raise ValueError(f"safetensors file too small: {path}")
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"cannot read safetensors header length: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 1 or header_bytes > min(size - 8, 128 << 20):
            raise ValueError(f"invalid safetensors header length {header_bytes}: {path}")
        header_raw = handle.read(header_bytes)
    header = json.loads(header_raw.decode("utf-8"))
    if not isinstance(header, Mapping):
        raise ValueError(f"safetensors header is not an object: {path}")
    payload_bytes = size - 8 - header_bytes
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"invalid tensor header for {name} in {path}")
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"incomplete tensor header for {name} in {path}")
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"invalid tensor shape for {name} in {path}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end <= payload_bytes):
            raise ValueError(f"invalid tensor offsets for {name} in {path}")
        tensors[str(name)] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }
        ranges.append((start, end, str(name)))
    if not tensors:
        raise ValueError(f"no tensors found in {path}")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"overlapping tensor ranges in {path}: {previous[2]} and {current[2]}"
            )
    return {
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "tensor_count": len(tensors),
        "tensor_names": sorted(tensors),
        "max_data_offset": max(end for _, end, _ in ranges),
    }


def _prior_manifest_comparison(
    previous_manifest: Path | None,
    model_path: Path,
    current_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if previous_manifest is None:
        return {
            "provided": False,
            "manifest_path": "",
            "manifest_sha256": "",
            "mismatches": [],
            "matches_previous_bytes": False,
        }
    previous = _load_json(previous_manifest)
    previous_rows = previous.get("files")
    if not isinstance(previous_rows, list):
        raise ValueError("previous Goal manifest lacks files list")
    mismatches: list[dict[str, Any]] = []
    for row in previous_rows:
        if not isinstance(row, Mapping):
            continue
        relative = str(row.get("relative_path") or Path(str(row.get("path", ""))).name)
        if not relative:
            continue
        current = current_files.get(relative)
        expected_sha = str(row.get("sha256", ""))
        expected_size = int(row.get("size_bytes", -1))
        if current is None:
            mismatches.append({
                "relative_path": relative,
                "reason": "MISSING_CURRENT_FILE",
                "expected_sha256": expected_sha,
                "actual_sha256": "",
                "expected_size_bytes": expected_size,
                "actual_size_bytes": -1,
            })
            continue
        if current["sha256"] != expected_sha or int(current["size_bytes"]) != expected_size:
            mismatches.append({
                "relative_path": relative,
                "reason": "FILE_HASH_OR_SIZE_MISMATCH",
                "expected_sha256": expected_sha,
                "actual_sha256": current["sha256"],
                "expected_size_bytes": expected_size,
                "actual_size_bytes": int(current["size_bytes"]),
                "actual_mtime_ns": int(current["mtime_ns"]),
            })
    return {
        "provided": True,
        "manifest_path": str(previous_manifest.resolve()),
        "manifest_sha256": sha256_file(previous_manifest),
        "recorded_model_path": str(previous.get("model_path", "")),
        "current_model_path": str(model_path),
        "mismatches": mismatches,
        "matches_previous_bytes": not mismatches,
    }


def audit(model_path: Path, previous_manifest: Path | None = None) -> dict[str, Any]:
    model_path = model_path.resolve()
    problems: list[str] = []
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    for name in LIGHTWEIGHT_REQUIRED:
        if not (model_path / name).is_file():
            problems.append(f"MISSING_REQUIRED_FILE:{name}")

    file_rows: list[dict[str, Any]] = []
    current_by_name: dict[str, dict[str, Any]] = {}
    for path in _safe_relative_files(model_path):
        row = {
            "path": str(path.resolve()),
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        file_rows.append(row)
        current_by_name[path.name] = row

    index_path = model_path / "model.safetensors.index.json"
    referenced_shards: list[str] = []
    header_audit: dict[str, Any] = {}
    tensor_index_count = 0
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            problems.append("INVALID_WEIGHT_MAP")
        else:
            tensor_index_count = len(weight_map)
            referenced_shards = sorted({str(value) for value in weight_map.values()})
            index_names_by_shard = {
                shard: {str(name) for name, value in weight_map.items() if str(value) == shard}
                for shard in referenced_shards
            }
            header_union: set[str] = set()
            for shard in referenced_shards:
                shard_path = model_path / shard
                if not shard_path.is_file():
                    problems.append(f"MISSING_REFERENCED_SHARD:{shard}")
                    continue
                try:
                    header = _read_safetensors_header(shard_path)
                    header_names = set(header["tensor_names"])
                    expected_names = index_names_by_shard[shard]
                    missing = sorted(expected_names - header_names)
                    unexpected = sorted(header_names - expected_names)
                    if missing:
                        problems.append(f"INDEX_TENSORS_MISSING_FROM_SHARD:{shard}:{len(missing)}")
                    if unexpected:
                        problems.append(f"UNINDEXED_TENSORS_IN_SHARD:{shard}:{len(unexpected)}")
                    header["index_tensor_count"] = len(expected_names)
                    header["missing_index_tensors"] = missing[:50]
                    header["unindexed_tensors"] = unexpected[:50]
                    header_audit[shard] = header
                    header_union.update(header_names)
                except Exception as exc:
                    problems.append(f"SAFETENSORS_HEADER_ERROR:{shard}:{type(exc).__name__}:{exc}")
            index_names = {str(name) for name in weight_map}
            if header_union and header_union != index_names:
                problems.append("GLOBAL_INDEX_HEADER_TENSOR_SET_MISMATCH")

    config_summary: dict[str, Any] = {}
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            config = _load_json(config_path)
            architectures = config.get("architectures", [])
            config_summary = {
                "architectures": architectures,
                "model_type": config.get("model_type"),
                "vocab_size": config.get("text_config", {}).get("vocab_size")
                if isinstance(config.get("text_config"), Mapping)
                else config.get("vocab_size"),
            }
            if architectures and "OpenVLAForActionPrediction" not in architectures:
                problems.append("UNEXPECTED_MODEL_ARCHITECTURE")
        except Exception as exc:
            problems.append(f"CONFIG_PARSE_ERROR:{type(exc).__name__}:{exc}")

    stats_summary: dict[str, Any] = {}
    stats_path = model_path / "dataset_statistics.json"
    if stats_path.is_file():
        try:
            stats = _load_json(stats_path)
            keys = sorted(str(key) for key in stats)
            stats_summary = {"keys": keys}
            if not any(key in {"libero_goal", "libero_goal_no_noops"} for key in keys):
                problems.append("LIBERO_GOAL_NORM_STATS_MISSING")
        except Exception as exc:
            problems.append(f"DATASET_STATS_PARSE_ERROR:{type(exc).__name__}:{exc}")

    aggregate = hashlib.sha256()
    for row in sorted(file_rows, key=lambda item: item["relative_path"]):
        aggregate.update(
            f"{row['relative_path']}|{row['size_bytes']}|{row['sha256']}\n".encode("utf-8")
        )

    prior = _prior_manifest_comparison(previous_manifest, model_path, current_by_name)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "C2G_GOAL_MODEL_STATIC_INTEGRITY_V2",
        "status": PASS_STATUS if not problems else HOLD_STATUS,
        "model_path": str(model_path),
        "unnorm_key": "libero_goal",
        "files": file_rows,
        "file_count": len(file_rows),
        "files_aggregate_sha256": aggregate.hexdigest(),
        "referenced_shards": referenced_shards,
        "missing_referenced_shards": [
            shard for shard in referenced_shards if not (model_path / shard).is_file()
        ],
        "safetensors_header_audit": header_audit,
        "tensor_index_count": tensor_index_count,
        "config_summary": config_summary,
        "dataset_statistics_summary": stats_summary,
        "previous_manifest_comparison": prior,
        "problems": problems,
        "boundaries": {
            "read_only": True,
            "openvla_models_loaded": 0,
            "libero_rollouts_launched": 0,
            "gpu_jobs_launched": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path)
    args = parser.parse_args(argv)
    report = audit(
        args.model_path,
        args.previous_manifest.resolve() if args.previous_manifest else None,
    )
    write_json(args.output_report.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "model_path": report["model_path"],
        "file_count": report["file_count"],
        "referenced_shards": report["referenced_shards"],
        "previous_manifest_mismatch_count": len(
            report["previous_manifest_comparison"]["mismatches"]
        ),
        "problems": report["problems"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
