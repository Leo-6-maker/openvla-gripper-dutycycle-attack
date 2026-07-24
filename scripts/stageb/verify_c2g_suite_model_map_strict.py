#!/usr/bin/env python3
"""Recompute and verify the strict four-suite OpenVLA model manifest.

This command is CPU-only. It ensures the suite model map, every referenced model
weight shard, lightweight model/processor files, and the audited Goal manifest are
unchanged since `build_c2g_suite_model_map_strict.py` produced the frozen report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file, validate_goal_manifest
from scripts.stageb.build_c2g_suite_model_map_strict import full_model_manifest


def verify(
    model_map_path: Path,
    report_path: Path,
    goal_manifest_path: Path,
) -> dict[str, Any]:
    model_map = json.loads(model_map_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(model_map, Mapping) or not isinstance(report, Mapping):
        raise ValueError("model map and report must be JSON objects")
    if report.get("status") != "PASS_C2G_STRICT_SUITE_MODEL_MAP":
        raise ValueError("strict suite model report status is not PASS")
    if report.get("model_map_sha256") != sha256_file(model_map_path):
        raise ValueError("suite model map SHA256 differs from frozen report")
    frozen_models = report.get("suite_models")
    if not isinstance(frozen_models, Mapping):
        raise ValueError("strict suite model report lacks suite_models")

    checked: dict[str, Any] = {}
    for suite in SUITES:
        raw_path = str(model_map.get(suite, "")).strip()
        if not raw_path:
            raise ValueError(f"suite model map missing {suite}")
        actual = full_model_manifest(Path(raw_path).resolve())
        frozen = frozen_models.get(suite)
        if not isinstance(frozen, Mapping):
            raise ValueError(f"frozen report missing {suite}")
        if actual["full_model_manifest_sha256"] != frozen.get("full_model_manifest_sha256"):
            raise ValueError(
                f"{suite} full model manifest changed: "
                f"{actual['full_model_manifest_sha256']} != "
                f"{frozen.get('full_model_manifest_sha256')}"
            )
        checked[suite] = {
            "model_path": raw_path,
            "full_model_manifest_sha256": actual["full_model_manifest_sha256"],
            "weight_file_count": actual["weight_file_count"],
            "weight_total_bytes": actual["weight_total_bytes"],
        }

    goal = validate_goal_manifest(
        goal_manifest_path.resolve(),
        Path(str(model_map["libero_goal"])).resolve(),
    )
    frozen_goal = report.get("goal_model_manifest")
    if not isinstance(frozen_goal, Mapping) or frozen_goal.get("sha256") != goal["sha256"]:
        raise ValueError("Goal model manifest SHA256 differs from frozen report")
    return {
        "gate": "C2G_STRICT_SUITE_MODEL_VERIFICATION",
        "status": "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION",
        "model_map": str(model_map_path.resolve()),
        "model_map_sha256": sha256_file(model_map_path),
        "frozen_report": str(report_path.resolve()),
        "frozen_report_sha256": sha256_file(report_path),
        "goal_model_manifest": goal,
        "suite_models": checked,
        "openvla_models_loaded": 0,
        "gpu_jobs_launched": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-map", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(
        args.model_map.resolve(),
        args.model_report.resolve(),
        args.goal_model_manifest.resolve(),
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
