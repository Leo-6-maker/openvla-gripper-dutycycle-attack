"""Machine-built PASS/HOLD decision for the FIT-only viability matrix.

The aggregate is intentionally descriptive (``PASS_PREPARATION_ONLY``).  This
module is the separate, sealed decision layer consumed by the full-FIT refit.
It never authorizes an attack.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from .b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory


DECISION_SCHEMA = "B3_OFFICIAL_V3_FIT_VIABILITY_DECISION_V1"


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_FIT_VIABILITY_DECISION_CONFIG_V1" or value.get("status") != "PRE_REGISTERED":
        raise ValueError("viability decision config is not the frozen pre-registered schema")
    return value, sha256_file(path)


def _load_aggregate(root: Path) -> tuple[dict[str, Any], str]:
    verify_sealed_directory(root)
    path = root / "viability_aggregate.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_FIT_VIABILITY_AGGREGATE_V1" or value.get("status") != "PASS_PREPARATION_ONLY":
        raise ValueError("viability aggregate is not a preparation-only aggregate")
    if value.get("run_count") != 24 or value.get("formal_training_authorized") is not False or value.get("formal_attack_authorized") is not False:
        raise ValueError("viability aggregate closure is incomplete")
    return value, sha256_file(path)


def build_viability_decision(aggregate_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    aggregate, aggregate_sha = _load_aggregate(aggregate_root)
    config, config_sha = _load_config(config_path)
    runs = list(aggregate.get("runs", []))
    expected = {(fold, variant, seed) for fold in range(4) for variant in ("B3_25D", "B3_25D9D") for seed in (20260717, 20260718, 20260719)}
    actual = {(int(run.get("fold_id", -1)), str(run.get("variant")), int(run.get("seed", -1))) for run in runs}
    if actual != expected:
        raise ValueError(f"viability decision coordinate closure failed: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    criteria = config.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("viability decision criteria are missing")
    variants: dict[str, dict[str, Any]] = {}
    overall_pass = True
    for variant in ("B3_25D", "B3_25D9D"):
        variant_runs = [run for run in runs if run.get("variant") == variant]
        hit_values = [float(run["metrics"]["full_t10_event_hit_rate"]) for run in variant_runs if run["metrics"].get("full_t10_event_hit_rate") is not None]
        negative_values = [float(run["metrics"].get("negative_episode_any_emit_rate", 1.0)) for run in variant_runs]
        release_values = [float(run["metrics"].get("release_overlap_count", 1.0)) for run in variant_runs]
        later_values = [run["metrics"].get("later_event_hit_rate") for run in variant_runs if run["metrics"].get("later_event_hit_rate") is not None]
        rule = criteria.get(variant)
        if not isinstance(rule, dict):
            raise ValueError(f"missing pre-registered criteria for {variant}")
        checks = {
            "run_count": len(variant_runs) == 12,
            "full_t10_event_hit_rate": bool(hit_values) and mean(hit_values) >= float(rule["min_mean_full_t10_event_hit_rate"]),
            "negative_episode_any_emit_rate": bool(negative_values) and max(negative_values) <= float(rule["max_negative_episode_any_emit_rate"]),
            "release_overlap_count": bool(release_values) and max(release_values) <= float(rule["max_release_overlap_count"]),
            "later_event_coverage": bool(later_values) and max(later_values) >= float(rule["min_later_event_hit_rate"]),
        }
        baseline_ok = True
        if rule.get("require_close_baseline_not_worse", True):
            for run in variant_runs:
                own = run["metrics"].get("full_t10_event_hit_rate")
                base = run["metrics"].get("baseline_comparison", {}).get("close_only", {}).get("full_t10_event_hit_rate")
                if own is None or base is None or float(own) < float(base) + float(rule.get("baseline_margin", 0.0)):
                    baseline_ok = False
                    break
        checks["close_baseline_comparison"] = baseline_ok
        variant_pass = all(checks.values())
        overall_pass = overall_pass and variant_pass
        variants[variant] = {
            "run_count": len(variant_runs),
            "mean_full_t10_event_hit_rate": mean(hit_values) if hit_values else None,
            "max_negative_episode_any_emit_rate": max(negative_values) if negative_values else None,
            "max_release_overlap_count": max(release_values) if release_values else None,
            "checks": checks,
            "status": "PASS" if variant_pass else "HOLD",
        }
    decision = {
        "schema": DECISION_SCHEMA,
        "status": "PASS" if overall_pass else "HOLD",
        "run_count": len(runs),
        "aggregate_sha256": aggregate_sha,
        "aggregate_root_sha256": sha256_file(aggregate_root / "SHA256SUMS"),
        "decision_config_sha256": config_sha,
        "decision_config": config,
        "variants": variants,
        "selected_variants": [name for name, value in variants.items() if value["status"] == "PASS"],
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "full_fit_refit_authorized": overall_pass,
    }
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        path = staging / "viability_decision.json"
        path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (path.with_name(path.name + ".sha256")).write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return decision


def load_viability_decision(root: Path) -> dict[str, Any]:
    verify_sealed_directory(root)
    path = root / "viability_decision.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != DECISION_SCHEMA or value.get("formal_training_authorized") is not False or value.get("formal_attack_authorized") is not False:
        raise ValueError("viability decision authorization boundary failed")
    if value.get("status") not in ("PASS", "HOLD") or value.get("full_fit_refit_authorized") is not (value.get("status") == "PASS"):
        raise ValueError("viability decision status is inconsistent")
    return value


__all__ = ["DECISION_SCHEMA", "build_viability_decision", "load_viability_decision"]
