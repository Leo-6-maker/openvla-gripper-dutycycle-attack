#!/usr/bin/env python3
"""Audit the 24-parent R8T Teacher-v2 collection canary.

The audit is CPU/read-only.  It verifies exact parent closure, complete student and
Teacher-v2 source fields, full raw/applied 7D actions, reproducibility provenance,
and minimum per-suite label support.  It never authorizes training automatically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.build_c2g_r8t_teacher_v2_canary import (
    PASS_STATUS as PLAN_PASS,
    SCHEMA as PLAN_SCHEMA,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)
from gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES

SCHEMA = "c2g.r8t.teacher_v2_canary_audit.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R8T_TEACHER_V2_CANARY_AUDIT"
GO_STATUS = "GO_R8T_DETECTOR_TRAIN_COLLECTION_DESIGN"
HOLD_INTEGRITY = "HOLD_R8T_CANARY_INTEGRITY"
HOLD_QUALITY = "HOLD_R8T_CANARY_QUALITY"
TEACHER_SCHEMA = "c2g.teacher_v2.raw_privileged_evidence.2026-07-11.v1"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
STEP_REQUIRED_KEYS = (
    "teacher_schema_version",
    "rgb_path",
    "task_language",
    "features_25d",
    "clean_policy_intent_9d",
    "clean_action_raw_7d",
    "applied_action_7d",
    "mujoco_contact_pairs",
    "active_target_known",
    "active_target_entity",
    "active_subgoal_index",
    "object_relative_lift",
    "target_distance_decrease",
    "constrained_manipulation_active",
    "manipulation_progress_active",
    "near_target",
    "supported_at_target",
    "release_safe",
    "target_object_position",
    "target_destination_position",
    "fixture_joint_motion",
    "active_target_contact",
    "active_target_bilateral_contact",
)
META_REQUIRED_KEYS = (
    "teacher_schema_version",
    "event_tracking_schema",
    "structured_goal_metadata",
    "goal_event_bindings",
    "resolution",
    "model_path",
    "model_selected_hashes",
    "suite_model_map_sha256",
    "suite_model_report_sha256",
    "goal_model_manifest_sha256",
    "model_verification_report_sha256",
    "unnorm_key",
    "token_semantics_sha256",
    "raw_action_order",
    "applied_action_order",
    "action_semantics",
    "controller_config",
    "runtime_versions",
    "bddl_file",
    "bddl_sha256",
    "official_init_state_sha256",
    "official_init_state_shape",
    "official_init_state_dtype",
    "replay_seed",
    "max_steps",
    "dummy_wait",
    "thresholds",
)
EPISODE_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "metadata_path", "step_records_path", "metadata_sha256", "step_records_sha256",
    "runtime_valid", "n_steps", "rgb_count", "all_rgb_present", "step_contiguous",
    "features_25d_complete", "policy_intent_9d_complete", "raw_action_7d_complete",
    "applied_action_7d_complete", "teacher_raw_field_schema_complete",
    "metadata_provenance_complete", "teacher_v2_rebuild_success",
    "known_positive_steps", "known_negative_steps", "unknown_steps",
    "positive_episode", "fully_known_negative_episode", "triggerable_positive_episode",
    "clean_success_observed", "structurally_complete", "failure_reason",
)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_vector(value: Any, length: int) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return False
    return vector.shape == (length,) and bool(np.isfinite(vector).all())


def triggerable(values: Sequence[bool], window: int = 3, required: int = 2) -> bool:
    for end in range(window - 1, len(values)):
        if sum(bool(value) for value in values[end - window + 1 : end + 1]) >= required:
            return True
    return False


def metadata_complete(metadata: Mapping[str, Any]) -> tuple[bool, list[str]]:
    missing = [key for key in META_REQUIRED_KEYS if key not in metadata or metadata[key] in (None, "", [], {})]
    runtime = metadata.get("runtime_versions")
    if isinstance(runtime, Mapping):
        for key in ("python", "torch", "transformers", "libero", "robosuite"):
            if runtime.get(key) in (None, "", "NOT_INSTALLED", "UNRESOLVED"):
                missing.append(f"runtime_versions.{key}")
    else:
        missing.append("runtime_versions.mapping")
    controller = metadata.get("controller_config")
    if not isinstance(controller, Mapping) or controller.get("controller_class") in (None, "", "UNRESOLVED"):
        missing.append("controller_config.controller_class")
    return not missing, sorted(set(missing))


def audit_episode(metadata_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    steps_path = metadata_path.with_name("step_records.jsonl")
    result = {
        "suite": str(expected["suite"]),
        "task_index": int(expected["task_index"]),
        "state_id": int(expected["state_id"]),
        "parent_key": str(expected["parent_key"]),
        "cohort": str(expected["cohort"]),
        "split": str(expected["split"]),
        "metadata_path": str(metadata_path),
        "step_records_path": str(steps_path),
        "runtime_valid": False,
        "structurally_complete": False,
        "failure_reason": "",
    }
    try:
        metadata = read_json(metadata_path)
        steps = read_jsonl(steps_path)
        observed = (
            str(metadata.get("suite", "")),
            int(metadata.get("task_index", -1)),
            int(metadata.get("state_id", -1)),
        )
        if observed != identity(expected):
            raise ValueError(f"identity mismatch {observed} != {identity(expected)}")
        if str(metadata.get("parent_key", "")) != str(expected["parent_key"]):
            raise ValueError("parent_key mismatch")
        if metadata.get("runtime_valid") is not True or metadata.get("condition") != "CLEAN":
            raise ValueError("runtime/condition invalid")
        if metadata.get("teacher_schema_version") != TEACHER_SCHEMA:
            raise ValueError("metadata Teacher-v2 schema mismatch")
        indices = [int(row.get("step", index)) for index, row in enumerate(steps)]
        contiguous = indices == list(range(len(steps)))
        if len(steps) < 16 or not contiguous:
            raise ValueError("trajectory is short or discontinuous")
        features_ok = all(finite_vector(row.get("features_25d"), 25) for row in steps)
        policy_ok = all(finite_vector(row.get("clean_policy_intent_9d"), len(CLEAN_POLICY_FEATURE_NAMES)) for row in steps)
        raw_action_ok = all(finite_vector(row.get("clean_action_raw_7d"), 7) for row in steps)
        applied_action_ok = all(finite_vector(row.get("applied_action_7d"), 7) for row in steps)
        teacher_fields_ok = all(
            all(key in row for key in STEP_REQUIRED_KEYS)
            and row.get("teacher_schema_version") == TEACHER_SCHEMA
            for row in steps
        )
        if not all((features_ok, policy_ok, raw_action_ok, applied_action_ok, teacher_fields_ok)):
            raise ValueError("step-level feature/action/Teacher-v2 schema incomplete")
        rgb_paths = [(metadata_path.parent / str(row["rgb_path"])).resolve() for row in steps]
        all_rgb = all(path.is_file() and path.stat().st_size > 0 for path in rgb_paths)
        if not all_rgb:
            raise FileNotFoundError("one or more RGB frames missing")
        meta_ok, meta_missing = metadata_complete(metadata)
        if not meta_ok:
            raise ValueError("metadata provenance missing: " + ",".join(meta_missing))
        labels = build_clean_teacher_episode(steps, metadata, thresholds=CleanTeacherThresholds(burst_length=10))
        if len(labels) != len(steps):
            raise ValueError("Teacher-v2 label count mismatch")
        known = [bool(row["label_known_mask"]) for row in labels]
        positive = [bool(row["y_gripper_critical_window"]) if known[index] else False for index, row in enumerate(labels)]
        known_positive = sum(k and p for k, p in zip(known, positive))
        known_negative = sum(k and not p for k, p in zip(known, positive))
        unknown = len(labels) - sum(known)
        result.update(
            metadata_sha256=sha256_file(metadata_path),
            step_records_sha256=sha256_file(steps_path),
            runtime_valid=True,
            n_steps=len(steps),
            rgb_count=len(rgb_paths),
            all_rgb_present=True,
            step_contiguous=True,
            features_25d_complete=True,
            policy_intent_9d_complete=True,
            raw_action_7d_complete=True,
            applied_action_7d_complete=True,
            teacher_raw_field_schema_complete=True,
            metadata_provenance_complete=True,
            teacher_v2_rebuild_success=True,
            known_positive_steps=known_positive,
            known_negative_steps=known_negative,
            unknown_steps=unknown,
            positive_episode=known_positive > 0,
            fully_known_negative_episode=all(known) and known_positive == 0,
            triggerable_positive_episode=triggerable([k and p for k, p in zip(known, positive)]),
            clean_success_observed=bool(metadata.get("clean_success_observed", False)),
            structurally_complete=True,
        )
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def run_audit(
    *,
    plan_report: Path,
    expected_plan_report_sha256: str,
    scheduler_root: Path,
    expected_scheduler_report_sha256: str,
    output_dir: Path,
    audit_head: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plan_report = plan_report.resolve()
    if sha256_file(plan_report) != expected_plan_report_sha256:
        raise ValueError("R8T plan report hash mismatch")
    plan = read_json(plan_report)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_PASS:
        raise ValueError("R8T plan is not accepted")
    scheduler_report_path = scheduler_root.resolve() / "c2g_r8t_dynamic_gpu_scheduler_report.json"
    if sha256_file(scheduler_report_path) != expected_scheduler_report_sha256:
        raise ValueError("scheduler report hash mismatch")
    scheduler = read_json(scheduler_report_path)
    if scheduler.get("status") != "PASS_C2G_R8T_DYNAMIC_GPU_CANARY":
        raise ValueError("scheduler did not complete all shards")

    expected_rows = read_jsonl(Path(str(plan["manifest"])))
    expected_lookup = {identity(row): row for row in expected_rows}
    if len(expected_lookup) != 24:
        raise ValueError("R8T canary must contain exactly 24 identities")
    metadata_paths = sorted(scheduler_root.rglob("episode_metadata.json"))
    observed_lookup: dict[tuple[str, int, int], Path] = {}
    for path in metadata_paths:
        metadata = read_json(path)
        key = (
            str(metadata.get("suite", "")),
            int(metadata.get("task_index", -1)),
            int(metadata.get("state_id", -1)),
        )
        if key in observed_lookup:
            raise ValueError(f"duplicate collected identity: {key}")
        observed_lookup[key] = path
    outside = sorted(set(observed_lookup) - set(expected_lookup))
    missing = sorted(set(expected_lookup) - set(observed_lookup))
    if outside:
        raise ValueError(f"out-of-plan identities: {outside}")

    rows: list[dict[str, Any]] = []
    for key, expected in sorted(
        expected_lookup.items(),
        key=lambda item: (SUITES.index(item[0][0]), item[0][1], item[0][2]),
    ):
        path = observed_lookup.get(key)
        if path is None:
            rows.append({
                "suite": key[0], "task_index": key[1], "state_id": key[2],
                "parent_key": expected["parent_key"], "cohort": expected["cohort"],
                "split": expected["split"], "runtime_valid": False,
                "structurally_complete": False, "failure_reason": "MISSING_EPISODE",
            })
        else:
            rows.append(audit_episode(path, expected))

    complete = [row for row in rows if row.get("structurally_complete")]
    integrity_pass = not missing and len(complete) == len(expected_rows)
    per_suite: dict[str, Any] = {}
    quality_violations: list[str] = []
    for suite in SUITES:
        local = [row for row in complete if row["suite"] == suite]
        support = {
            "episode_count": len(local),
            "clean_success_episode_count": sum(bool(row.get("clean_success_observed")) for row in local),
            "positive_episode_count": sum(bool(row.get("positive_episode")) for row in local),
            "triggerable_positive_episode_count": sum(bool(row.get("triggerable_positive_episode")) for row in local),
            "fully_known_negative_episode_count": sum(bool(row.get("fully_known_negative_episode")) for row in local),
            "known_positive_steps": sum(int(row.get("known_positive_steps", 0)) for row in local),
            "known_negative_steps": sum(int(row.get("known_negative_steps", 0)) for row in local),
            "unknown_steps": sum(int(row.get("unknown_steps", 0)) for row in local),
        }
        per_suite[suite] = support
        if support["episode_count"] != 6:
            quality_violations.append(f"{suite}: incomplete six-episode shard")
        if support["clean_success_episode_count"] < 1:
            quality_violations.append(f"{suite}: no clean success")
        if support["triggerable_positive_episode_count"] < 1:
            quality_violations.append(f"{suite}: no triggerable positive episode")
        if support["known_negative_steps"] < 1:
            quality_violations.append(f"{suite}: no known-negative step")
    quality_pass = integrity_pass and not quality_violations
    final_decision = GO_STATUS if quality_pass else HOLD_QUALITY if integrity_pass else HOLD_INTEGRITY

    output_dir.mkdir(parents=True)
    ledger_path = output_dir / "c2g_r8t_canary_episode_audit.csv"
    write_csv(ledger_path, rows, EPISODE_FIELDS)
    reusable_path = output_dir / "c2g_r8t_canary_structurally_complete.jsonl"
    write_jsonl(reusable_path, complete)
    report = {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "audit_head": audit_head,
        "plan_report": str(plan_report),
        "plan_report_sha256": sha256_file(plan_report),
        "scheduler_report": str(scheduler_report_path),
        "scheduler_report_sha256": sha256_file(scheduler_report_path),
        "expected_episode_count": len(expected_rows),
        "observed_episode_count": len(observed_lookup),
        "structurally_complete_episode_count": len(complete),
        "missing_episode_count": len(missing),
        "outside_identity_count": len(outside),
        "per_suite": per_suite,
        "quality_violations": quality_violations,
        "integrity_pass": integrity_pass,
        "quality_pass": quality_pass,
        "final_decision": final_decision,
        "post_canary_collection_authorization": (
            "ELIGIBLE_FOR_SEPARATE_USER_REVIEW" if quality_pass else "HOLD"
        ),
        "training_authorization": "HOLD",
        "episode_ledger": str(ledger_path),
        "episode_ledger_sha256": sha256_file(ledger_path),
        "structurally_complete_manifest": str(reusable_path),
        "structurally_complete_manifest_sha256": sha256_file(reusable_path),
        "invariants": {
            "train_only": all(row.get("cohort") == "DETECTOR_TRAIN" and row.get("split") == "train" for row in rows),
            "identity_partition_closed": len(expected_lookup) == len(observed_lookup) + len(missing),
            "no_outside_identity": not outside,
            "no_attack_eval": all(row.get("cohort") != "ATTACK_EVAL_PREREGISTERED" for row in rows),
        },
        "boundaries": {
            "attack_outcomes_read": False,
            "models_loaded": 0,
            "environments_created": 0,
            "rollouts_launched": 0,
            "training_epochs": 0,
            "attacks": 0,
            "storage_deletions": 0,
        },
    }
    report_path = output_dir / "c2g_r8t_teacher_v2_canary_audit.json"
    write_json(report_path, report)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    hashes = {path.name: sha256_file(path) for path in files}
    sums = output_dir / "SHA256SUMS"
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    self_binding = output_dir / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {
        **report,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "artifact_sha256": hashes,
        "sha256s_sha256": sums_sha,
        "sha256s_self_binding_sha256": sha256_file(self_binding),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--scheduler-root", type=Path, required=True)
    parser.add_argument("--expected-scheduler-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        scheduler_root=args.scheduler_root,
        expected_scheduler_report_sha256=args.expected_scheduler_report_sha256,
        output_dir=args.output_dir,
        audit_head=args.audit_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
