#!/usr/bin/env python3
"""Aggregate exactly 24 sealed held-out FIT validation bundles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.b3_training_protocol import load_fit_fold_bundle


def _prediction_module():
    path = Path(__file__).with_name("build_b3_v3_prediction_bundle.py")
    spec = importlib.util.spec_from_file_location("b3_v3_prediction_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prediction bundle module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluator():
    path = Path(__file__).with_name("evaluate_b3_v3_viability.py")
    spec = importlib.util.spec_from_file_location("b3_v3_viability", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load viability evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        if mode == "close_only":
            emit = float(row.get("recent_close_streak", 0.0)) >= 3.0
        elif mode == "time_since_close":
            emit = float(row.get("time_since_close", -1.0)) >= 5.0
        else:
            raise ValueError(mode)
        rows.append(dict(row, pred_emit=emit, release_imminent=False))
    return rows


def aggregate_viability(run_roots: list[Path], output_root: Path, *, fold_root: Path | None = None) -> dict[str, Any]:
    if len(run_roots) != 24:
        raise ValueError(f"viability matrix requires exactly 24 run roots, got {len(run_roots)}")
    prediction = _prediction_module()
    evaluator = _evaluator()
    folds = load_fit_fold_bundle(fold_root) if fold_root is not None else None
    seen: set[tuple[int, str, int]] = set()
    run_reports: list[dict[str, Any]] = []
    shared_bindings: dict[str, Any] | None = None
    for root in run_roots:
        manifest, records = prediction.load_prediction_bundle(root)
        coordinate = (int(manifest["fold_id"]), str(manifest["variant"]), int(manifest["seed"]))
        if coordinate in seen or len({row["canonical_parent_key"] for row in records}) != 200:
            raise ValueError("duplicate or incomplete viability run")
        if folds is not None:
            expected_ids = folds["folds"][coordinate[0]]["validation_identities"]
            if manifest.get("validation_identities") != expected_ids or manifest.get("validation_identity_sha256") != folds["folds"][coordinate[0]]["validation_identity_sha256"]:
                raise ValueError(f"validation identity closure failed for {coordinate}")
            bindings = manifest.get("source_bindings")
            if not isinstance(bindings, dict) or bindings.get("fold_bundle_sha256") != sha256_file(fold_root / "SHA256SUMS"):
                raise ValueError(f"prediction source binding closure failed for {coordinate}")
            if shared_bindings is None:
                shared_bindings = bindings
            else:
                for name in ("registry_root_sha256", "s1_root_sha256", "authorization_payload_sha256", "runner_binding_sha256"):
                    if bindings.get(name) != shared_bindings.get(name):
                        raise ValueError(f"prediction shared source binding drift: {name}")
        seen.add(coordinate)
        metrics = evaluator.event_level_metrics(records)
        baselines = {
            "close_only": evaluator.event_level_metrics(_baseline(records, "close_only")),
            "time_since_close": evaluator.event_level_metrics(_baseline(records, "time_since_close")),
        }
        metrics["baseline_comparison"] = {
            name: {
                "full_t10_event_hit_rate": value.get("full_t10_event_hit_rate"),
                "negative_episode_any_emit_rate": value.get("negative_episode_any_emit_rate"),
            }
            for name, value in baselines.items()
        }
        run_reports.append({
            "schema": "B3_OFFICIAL_V3_FIT_VIABILITY_RUN_REPORT_V1",
            **{key: manifest[key] for key in ("fold_id", "seed", "variant", "checkpoint_sha256", "validation_identity_count", "validation_identity_sha256")},
            "metrics": metrics,
            "source_bindings": manifest.get("source_bindings", {}),
            "attack_enabled": False,
            "teacher_inputs_consumed": False,
        })
    expected = {(fold, variant, seed) for fold in range(4) for variant in ("B3_25D", "B3_25D9D") for seed in (20260717, 20260718, 20260719)}
    if seen != expected:
        raise ValueError(f"viability coordinate closure failed: missing={sorted(expected - seen)} extra={sorted(seen - expected)}")
    by_variant: dict[str, list[float]] = defaultdict(list)
    for report in run_reports:
        value = report["metrics"].get("full_t10_event_hit_rate")
        if value is not None:
            by_variant[report["variant"]].append(float(value))
    aggregate = {
        "schema": "B3_OFFICIAL_V3_FIT_VIABILITY_AGGREGATE_V1",
        "status": "PASS_PREPARATION_ONLY",
        "run_count": len(run_reports),
        "runs": sorted(run_reports, key=lambda item: (item["fold_id"], item["variant"], item["seed"])),
        "variant_summary": {
            variant: {"run_count": len(values), "mean_full_t10_event_hit_rate": sum(values) / len(values) if values else None}
            for variant, values in sorted(by_variant.items())
        },
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "fold_bundle_sha256": sha256_file(fold_root / "SHA256SUMS") if fold_root is not None else None,
    }
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.staging")
    try:
        staging.mkdir(parents=True)
        path = staging / "viability_aggregate.json"
        path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (path.with_name(path.name + ".sha256")).write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
        seal_directory(staging)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_viability(args.run_root, args.output_root, fold_root=args.fold_root)
    print(json.dumps({"status": report["status"], "run_count": report["run_count"], "formal_training_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
