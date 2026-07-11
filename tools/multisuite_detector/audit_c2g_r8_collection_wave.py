#!/usr/bin/env python3
"""Audit one R8 clean-collection wave against the frozen R7/R8 contracts.

The audit distinguishes exact collection completion from scientific label quality.
Missing or structurally ineligible expected episodes produce a successful HOLD
report.  Duplicate identities, out-of-wave collection, provenance drift, or a
changed baseline source corpus fail closed.

No model, environment, rollout, embedding, training, calibration, or attack is
launched by this program.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.audit_c2g_clean_source_inventory import (  # noqa: E402
    _support,
    audit_episode,
    discover_sources,
    write_json,
    write_jsonl,
)
from tools.multisuite_detector.build_c2g_r8_collection_waves import (  # noqa: E402
    DETECTOR_CANARY,
    PASS_STATUS as WAVE_PLAN_PASS,
    SCHEMA as WAVE_PLAN_SCHEMA,
    WAVES,
    identity,
    load_bound_inputs,
    read_json,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (  # noqa: E402
    CleanTeacherThresholds,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import SUITES  # noqa: E402

SCHEMA = "c2g.r8.collection_wave_audit.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R8_COLLECTION_WAVE_AUDIT"
OPERATIONAL_PASS = "PASS_C2G_R8_COLLECTION_WAVE_COMPLETE"
OPERATIONAL_HOLD = "HOLD_C2G_R8_COLLECTION_WAVE_INCOMPLETE"
QUALITY_PASS = "PASS_C2G_R8_CANARY_MINIMUM_QUALITY_OBSERVATION"
QUALITY_HOLD = "HOLD_C2G_R8_CANARY_QUALITY_REVIEW"
QUALITY_NA = "NOT_APPLICABLE_NON_CANARY_WAVE"


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _assert_hash(path: Path, expected: str, *, name: str) -> str:
    expected = _require_sha256(expected, name=name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _assert_new_external_outputs(report: Path, reusable_manifest: Path) -> None:
    report = report.resolve()
    reusable_manifest = reusable_manifest.resolve()
    if report == reusable_manifest:
        raise ValueError("report and reusable manifest paths must differ")
    for path in (report, reusable_manifest):
        if path.exists():
            raise FileExistsError(path)
        if _is_within(path, REPO):
            raise ValueError("R8 runtime audit outputs must be outside the repository")


def load_wave_plan(
    wave_plan_report: Path,
    expected_wave_plan_report_sha256: str,
    wave: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    wave_plan_report = wave_plan_report.resolve()
    if not wave_plan_report.is_file():
        raise FileNotFoundError(wave_plan_report)
    _assert_hash(
        wave_plan_report,
        expected_wave_plan_report_sha256,
        name="R8 collection wave plan report",
    )
    report = read_json(wave_plan_report)
    if report.get("schema") != WAVE_PLAN_SCHEMA or report.get("status") != WAVE_PLAN_PASS:
        raise ValueError("R8 wave plan report is not accepted")
    if wave not in WAVES or wave not in report.get("waves", {}):
        raise ValueError(f"unknown or absent R8 wave: {wave}")
    if not str(report.get("training_authorization", "")).startswith("HOLD_"):
        raise ValueError("R8 wave plan unexpectedly authorizes training")

    registry, baseline_reusable, _, _ = load_bound_inputs(
        registry_path=Path(report["r7_registry"]),
        plan_report_path=Path(report["r7_plan_report"]),
        expected_plan_report_sha256=report["r7_plan_report_sha256"],
        source_audit_report_path=Path(report["r7_source_audit_report"]),
        expected_source_audit_report_sha256=report["r7_source_audit_report_sha256"],
        reusable_manifest_path=Path(report["r7_reusable_manifest"]),
        expected_reusable_manifest_sha256=report["r7_reusable_manifest_sha256"],
    )
    wave_info = dict(report["waves"][wave])
    manifest_path = Path(str(wave_info.get("manifest", ""))).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if str(wave_info.get("manifest_sha256", "")) != sha256_file(manifest_path):
        raise ValueError("R8 wave manifest hash differs from wave plan")
    wave_rows = read_jsonl(manifest_path)
    if len(wave_rows) != int(wave_info.get("episode_count", -1)):
        raise ValueError("R8 wave manifest count differs from wave plan")
    expected_ids = [identity(row) for row in wave_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("R8 wave manifest contains duplicate identity")
    return report, registry, baseline_reusable, {**wave_info, "rows": wave_rows}


def _audit_roots(
    roots: Sequence[Path],
    *,
    registry_lookup: Mapping[tuple[str, int, int], Mapping[str, Any]],
    persistence_window: int,
    persistence_required: int,
    burst_length: int,
    hash_rgb: bool,
) -> list[dict[str, Any]]:
    thresholds = CleanTeacherThresholds(burst_length=burst_length)
    rows: list[dict[str, Any]] = []
    for root, metadata, steps in discover_sources(roots):
        rows.append(
            audit_episode(
                root,
                metadata,
                steps,
                registry_lookup=registry_lookup,
                thresholds=thresholds,
                persistence_window=persistence_window,
                persistence_required=persistence_required,
                hash_rgb=hash_rgb,
            )
        )
    return rows


def _mark_reusable(rows: Sequence[dict[str, Any]]) -> None:
    counts = Counter(identity(row) for row in rows)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate suite/task/state identities across source roots: {duplicates[:20]}")
    for row in rows:
        row["duplicate_identity"] = False
        row["reusable"] = bool(row.get("registered") and row.get("structurally_eligible"))


def audit_collection_wave(
    *,
    wave_plan_report: Path,
    expected_wave_plan_report_sha256: str,
    wave: str,
    baseline_source_roots: Sequence[Path],
    new_source_roots: Sequence[Path],
    output_report: Path,
    output_reusable_manifest: Path,
    audit_head: str = "",
    persistence_window: int = 3,
    persistence_required: int = 2,
    burst_length: int = 10,
    hash_rgb: bool = True,
) -> dict[str, Any]:
    _assert_new_external_outputs(output_report, output_reusable_manifest)
    plan, registry, baseline_reusable, wave_info = load_wave_plan(
        wave_plan_report,
        expected_wave_plan_report_sha256,
        wave,
    )
    registry_lookup = {identity(row): row for row in registry}
    expected_rows = list(wave_info["rows"])
    expected_ids = {identity(row) for row in expected_rows}
    frozen_baseline_ids = {identity(row) for row in baseline_reusable}

    baseline_rows = _audit_roots(
        baseline_source_roots,
        registry_lookup=registry_lookup,
        persistence_window=persistence_window,
        persistence_required=persistence_required,
        burst_length=burst_length,
        hash_rgb=hash_rgb,
    )
    new_rows = _audit_roots(
        new_source_roots,
        registry_lookup=registry_lookup,
        persistence_window=persistence_window,
        persistence_required=persistence_required,
        burst_length=burst_length,
        hash_rgb=hash_rgb,
    )
    all_rows = baseline_rows + new_rows
    _mark_reusable(all_rows)

    observed_baseline_ids = {identity(row) for row in baseline_rows if row["reusable"]}
    if observed_baseline_ids != frozen_baseline_ids:
        missing = sorted(frozen_baseline_ids - observed_baseline_ids)
        added = sorted(observed_baseline_ids - frozen_baseline_ids)
        raise ValueError(f"baseline reusable source drift missing={missing[:20]} added={added[:20]}")
    if any(not row["reusable"] for row in baseline_rows):
        failures = [row.get("failure_reason") for row in baseline_rows if not row["reusable"]]
        raise ValueError(f"baseline source became ineligible: {failures[:20]}")

    observed_new_ids = {identity(row) for row in new_rows}
    outside = sorted(observed_new_ids - expected_ids)
    if outside:
        raise ValueError(f"new source root contains out-of-wave identities: {outside[:20]}")

    new_reusable_by_id = {identity(row): row for row in new_rows if row["reusable"]}
    completed_ids = expected_ids & set(new_reusable_by_id)
    missing_ids = expected_ids - observed_new_ids
    ineligible_ids = {
        identity(row) for row in new_rows if identity(row) in expected_ids and not row["reusable"]
    }
    operational_complete = not missing_ids and not ineligible_ids and completed_ids == expected_ids
    operational_status = OPERATIONAL_PASS if operational_complete else OPERATIONAL_HOLD

    completed_rows = [new_reusable_by_id[key] for key in sorted(completed_ids)]
    per_suite: dict[str, Any] = {}
    for suite in SUITES:
        expected_local = [row for row in expected_rows if row["suite"] == suite]
        completed_local = [row for row in completed_rows if row["suite"] == suite]
        per_suite[suite] = {
            **_support(completed_local),
            "clean_success_observed_count": sum(
                bool(row.get("clean_success_observed")) for row in completed_local
            ),
            "required_episode_count": len(expected_local),
            "missing_or_ineligible_episode_count": len(expected_local) - len(completed_local),
            "cohort_counts": dict(sorted(Counter(row["cohort"] for row in completed_local).items())),
        }
    per_cohort: dict[str, Any] = {}
    for cohort in sorted({str(row["cohort"]) for row in expected_rows}):
        expected_local = [row for row in expected_rows if row["cohort"] == cohort]
        completed_local = [row for row in completed_rows if row["cohort"] == cohort]
        per_cohort[cohort] = {
            **_support(completed_local),
            "clean_success_observed_count": sum(
                bool(row.get("clean_success_observed")) for row in completed_local
            ),
            "required_episode_count": len(expected_local),
            "missing_or_ineligible_episode_count": len(expected_local) - len(completed_local),
        }

    quality_violations: list[str] = []
    if wave == DETECTOR_CANARY:
        if not operational_complete:
            quality_violations.append("canary collection is incomplete")
        for suite in SUITES:
            support = per_suite[suite]
            if support["clean_success_observed_count"] < 1:
                quality_violations.append(f"{suite}: no clean success observed")
            if support["triggerable_positive_episode_count"] < 1:
                quality_violations.append(f"{suite}: no triggerable positive episode")
            if support["known_negative_steps"] < 1:
                quality_violations.append(f"{suite}: no known-negative steps")
        quality_status = QUALITY_PASS if not quality_violations else QUALITY_HOLD
    else:
        quality_status = QUALITY_NA

    combined_reusable = sorted(
        [row for row in all_rows if row["reusable"]],
        key=lambda row: (SUITES.index(row["suite"]), row["task_index"], row["state_id"]),
    )
    output_reusable_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_reusable_manifest, combined_reusable)
    report = {
        "schema": SCHEMA,
        "gate": "C2G_R8_COLLECTION_WAVE_AUDIT",
        "status": PASS_STATUS,
        "audit_head": audit_head,
        "wave": wave,
        "wave_plan_report": str(wave_plan_report.resolve()),
        "wave_plan_report_sha256": sha256_file(wave_plan_report.resolve()),
        "wave_manifest": wave_info["manifest"],
        "wave_manifest_sha256": wave_info["manifest_sha256"],
        "baseline_source_roots": [str(path.resolve()) for path in baseline_source_roots],
        "new_source_roots": [str(path.resolve()) for path in new_source_roots],
        "expected_episode_count": len(expected_rows),
        "observed_new_episode_count": len(new_rows),
        "completed_reusable_episode_count": len(completed_rows),
        "missing_episode_count": len(missing_ids),
        "ineligible_episode_count": len(ineligible_ids),
        "missing_identities": [list(value) for value in sorted(missing_ids)],
        "ineligible_identities": [list(value) for value in sorted(ineligible_ids)],
        "operational_status": operational_status,
        "canary_quality_status": quality_status,
        "canary_quality_violation_count": len(quality_violations),
        "canary_quality_violations": quality_violations,
        "per_suite": per_suite,
        "per_cohort": per_cohort,
        "episode_audits": new_rows,
        "combined_reusable_manifest": str(output_reusable_manifest.resolve()),
        "combined_reusable_manifest_sha256": sha256_file(output_reusable_manifest.resolve()),
        "detector_full_collection_authorization": "HOLD_PENDING_EXPLICIT_POST_CANARY_REVIEW",
        "attack_eval_collection_authorization": "HOLD_UNTIL_DETECTOR_FROZEN",
        "training_authorization": "HOLD_PENDING_FULL_CORPUS_MATERIALIZATION_AND_AUDIT",
        "next_stage": (
            "STOP_FOR_R8_CANARY_REVIEW"
            if wave == DETECTOR_CANARY
            else "STOP_FOR_R8_COLLECTION_WAVE_REVIEW"
        ),
        "boundaries": {
            "clean_only": True,
            "attack_outcomes_read": False,
            "openvla_models_loaded": 0,
            "libero_environments_created": 0,
            "clean_rollouts_launched_by_audit": 0,
            "attacks_launched": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
            "source_assets_modified": False,
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_report, report)
    return {**report, "output_report_sha256": sha256_file(output_report)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-plan-report", type=Path, required=True)
    parser.add_argument("--expected-wave-plan-report-sha256", required=True)
    parser.add_argument("--wave", choices=WAVES, required=True)
    parser.add_argument("--baseline-source-root", action="append", type=Path, default=[])
    parser.add_argument("--new-source-root", action="append", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-reusable-manifest", type=Path, required=True)
    parser.add_argument("--audit-head", default="")
    parser.add_argument("--persistence-window", type=int, default=3)
    parser.add_argument("--persistence-required", type=int, default=2)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--hash-rgb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_collection_wave(
        wave_plan_report=args.wave_plan_report,
        expected_wave_plan_report_sha256=args.expected_wave_plan_report_sha256,
        wave=args.wave,
        baseline_source_roots=args.baseline_source_root,
        new_source_roots=args.new_source_root,
        output_report=args.output_report,
        output_reusable_manifest=args.output_reusable_manifest,
        audit_head=args.audit_head,
        persistence_window=args.persistence_window,
        persistence_required=args.persistence_required,
        burst_length=args.burst_length,
        hash_rgb=args.hash_rgb,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
