#!/usr/bin/env python3
"""Freeze a full per-suite OpenVLA model map including weight-shard hashes.

Unlike the lightweight inventory helper, this strict release builder hashes every
weight shard referenced by the model index (or every monolithic supported weight
file). The resulting report is suitable for binding materialization and online
execution to the same policy bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.build_c2g_suite_model_map import (
    SUITES,
    selected_model_manifest,
    sha256_file,
    validate_goal_manifest,
)


def weight_files(model_path: Path) -> list[Path]:
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        value = json.loads(index.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"invalid weight_map in {index}")
        names = sorted({str(name) for name in weight_map.values()})
        files = [model_path / name for name in names]
    else:
        candidates = sorted(model_path.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(model_path.glob("pytorch_model*.bin"))
        files = candidates
    if not files:
        raise FileNotFoundError(f"no model weight files found in {model_path}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("referenced model shards missing: " + ", ".join(missing))
    return files


def full_model_manifest(model_path: Path) -> dict[str, Any]:
    light = selected_model_manifest(model_path)
    weights = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weight_files(model_path)
    ]
    full_digest = hashlib.sha256(
        (
            light["selected_manifest_sha256"]
            + "\n"
            + "".join(
                f"{row['name']}|{row['bytes']}|{row['sha256']}\n"
                for row in weights
            )
        ).encode("utf-8")
    ).hexdigest()
    return {
        **light,
        "weight_file_count": len(weights),
        "weight_total_bytes": sum(int(row["bytes"]) for row in weights),
        "weight_files": weights,
        "full_model_manifest_sha256": full_digest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
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

    manifests = {
        suite: full_model_manifest(Path(model_map[suite]))
        for suite in SUITES
    }
    goal_manifest = validate_goal_manifest(
        args.goal_model_manifest.resolve(),
        Path(model_map["libero_goal"]),
    )
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(
        json.dumps(model_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "gate": "C2G_STRICT_SUITE_MODEL_MAP",
        "status": "PASS_C2G_STRICT_SUITE_MODEL_MAP",
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
