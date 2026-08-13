#!/usr/bin/env python3
"""Independent audit for the exact 40x24 plan-only gate."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from gripper_attack.stage_v_causal_observation_snapshot import (  # noqa: E402
    CausalSnapshotError,
    assert_primary_observation_exact,
    load_snapshot,
)


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
CORRIDOR_COMMIT = "3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2"
CORRIDOR_TREE = "2492a075e782a112d1e857248956b2647e751039"
CORRIDOR_RUNNER_SHA = "26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279"
REQUIRED_PAYLOAD = {"probe", "full_simulator_state", "controller_and_wrapper_runtime_state", "required_rng_state", "raw_observation", "raw_observation_sha256", "canonical_policy_rgb_224", "processed_image", "processed_image_sha256", "input_ids", "pixel_values", "attention_mask", "attention_mask_present", "prompt", "decode_config", "clean_reference_action_window"}
FORBIDDEN_BRANCH_NAMES = {"M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", "M4_TREATMENT_OBSERVATIONS_V1.jsonl", "M4_V_PHYS_LABELS_V1.jsonl", "COUNTERFACTUAL_BRANCHES.jsonl", "TREATMENT_REPETITION_OBSERVATIONS.jsonl", "COLLAPSED_PROBE_DOSE_LABELS.jsonl"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(value: Any) -> str:
    return canonical_sha256(canonical_value(value))


def _inside(root: Path, relative: Any) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def _audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest_path = root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
        manifest = _load(manifest_path)
        inputs = {
            "final": _load(root / "inputs/FINAL_PARENT_MANIFEST.json"),
            "split": _load(root / "inputs/FINAL_PARENT_SPLIT.json"),
            "attempt": _load(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json"),
        }
        run_registry = _load(root / "PARENT_RUN_REGISTRY.json")
    except (OSError, ValueError, KeyError) as exc:
        result = {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT_V1", "status": "FAIL", "auditor_role": "independent_plan_snapshot_auditor", "errors": [f"INPUT_LOAD:{type(exc).__name__}:{exc}"], "protected_counters": dict(COUNTERS)}
        (root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    if manifest.get("schema") != "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1":
        errors.append("MANIFEST_SCHEMA_INVALID")
    if manifest.get("status") not in {"FROZEN_PLAN_ONLY_PENDING_INDEPENDENT_AUDIT", "PASS_EXACT_40X24_PLAN_ONLY", "HOLD_SEALED_EXACT_40X24_PLAN_ONLY"}:
        errors.append("MANIFEST_STATUS_INVALID")
    if manifest.get("parent_count") != 40 or manifest.get("probe_count_per_parent") != 24 or manifest.get("probe_count_total") != 960 or manifest.get("planned_branch_authority_count") != 3840:
        errors.append("GLOBAL_COUNT_INVALID")
    if manifest.get("selection_outcomes_read") is not False or manifest.get("intervention_executed") is not False or manifest.get("v_phys_generated") is not False or manifest.get("teacher_predictions_read") is not False or manifest.get("student_predictions_read") is not False or manifest.get("protected_counters") != COUNTERS:
        errors.append("GLOBAL_OUTCOME_BOUNDARY_INVALID")
    if manifest.get("corridor_source") != {"commit": CORRIDOR_COMMIT, "tree": CORRIDOR_TREE, "runner_sha256": CORRIDOR_RUNNER_SHA}:
        errors.append("GLOBAL_CORRIDOR_SOURCE_INVALID")
    final = inputs["final"]
    split = inputs["split"]
    attempt = inputs["attempt"]
    final_sha = _sha_file(root / "inputs/FINAL_PARENT_MANIFEST.json")
    split_sha = _sha_file(root / "inputs/FINAL_PARENT_SPLIT.json")
    attempt_sha = _sha_file(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json")
    if manifest.get("final40_manifest_sha256") != final_sha or manifest.get("final_split_sha256") != split_sha or manifest.get("exact55_attempt_registry_sha256") != attempt_sha:
        errors.append("GLOBAL_INPUT_HASH_BINDING_INVALID")
    if final.get("schema") != "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" or final.get("status") != "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" or final.get("parent_count") != 40 or final.get("formal_m4_authorized") is not False or final.get("outcomes_read") is not False:
        errors.append("FINAL40_INPUT_INVALID")
    if split.get("schema") != "STAGE_V_M4_FINAL_PARENT_SPLIT_V2" or split.get("status") != "FROZEN" or split.get("final_manifest_sha256") != final_sha or split.get("counts") != {"TRAIN": 24, "VAL": 8, "TEST": 8}:
        errors.append("FINAL_SPLIT_INPUT_INVALID")
    if attempt.get("schema") != "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1" or attempt.get("status") != "FROZEN_EXACT55_CORRIDOR_ATTEMPT_FIREWALL" or attempt.get("attempted_identity_count") != 55 or attempt.get("unique_identity_count") != 55 or attempt.get("duplicate_count") != 0 or attempt.get("outcomes_read") is not False or attempt.get("protected_counters") != COUNTERS:
        errors.append("EXACT55_INPUT_INVALID")
    final_rows = [row for row in final.get("parents", []) if isinstance(row, Mapping)]
    final_keys = {str(row.get("canonical_parent_key")) for row in final_rows}
    split_keys = {str(row.get("canonical_parent_key")) for row in split.get("parents", []) if isinstance(row, Mapping)}
    attempted_keys = {str(row.get("canonical_parent_key")) for row in attempt.get("attempted_identities", []) if isinstance(row, Mapping)}
    if len(final_rows) != 40 or len(final_keys) != 40 or final_keys != split_keys or len(attempted_keys) != 55 or not final_keys.issubset(attempted_keys):
        errors.append("IDENTITY_SET_INVALID")
    if Counter(str(row.get("suite")) for row in final_rows) != Counter({"libero_10": 10, "libero_goal": 10, "libero_object": 10, "libero_spatial": 10}):
        errors.append("FINAL40_SUITE_COUNTS_INVALID")
    run_rows = [row for row in run_registry.get("results", []) if isinstance(row, Mapping)]
    if run_registry.get("parent_count") != 40 or len(run_rows) != 40 or any(row.get("return_code") != 0 for row in run_rows):
        errors.append("PARENT_RUN_REGISTRY_INVALID")
    by_gpu: dict[str, list[tuple[datetime, datetime, str]]] = {}
    for row in run_rows:
        try:
            start = datetime.fromisoformat(str(row["started_utc"]))
            finish = datetime.fromisoformat(str(row["finished_utc"]))
            if finish <= start:
                raise ValueError("NONPOSITIVE_INTERVAL")
            by_gpu.setdefault(str(row["gpu"]), []).append((start, finish, str(row["canonical_parent_key"])))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"RESOURCE_INTERVAL_INVALID:{row.get('canonical_parent_key')}:{exc}")
    for gpu, intervals in by_gpu.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                errors.append(f"RESOURCE_GPU_OVERLAP:{gpu}:{previous[2]}:{current[2]}")

    global_probes = {(str(row.get("canonical_parent_key")), str(row.get("probe_id"))): row for row in manifest.get("probe_authorities", []) if isinstance(row, Mapping)}
    global_branches = {(str(row.get("canonical_parent_key")), str(row.get("probe_id")), str(row.get("arm"))): row for row in manifest.get("branch_authorities", []) if isinstance(row, Mapping)}
    if len(global_probes) != 960 or len(global_branches) != 3840:
        errors.append("GLOBAL_AUTHORITY_UNIQUENESS_INVALID")
    expected_parent_keys = {str(row.get("canonical_parent_key")) for row in manifest.get("parents", []) if isinstance(row, Mapping)}
    if expected_parent_keys != final_keys:
        errors.append("GLOBAL_PARENT_SET_INVALID")
    for parent in final_rows:
        key = str(parent.get("canonical_parent_key"))
        parent_root = root / "parents" / key.replace("/", "__")
        try:
            receipt = _load(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
            audit = _load(parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json")
            clean = _load(parent_root / "CLEAN_TRAJECTORY_V1_4.json")
            plan = _load(parent_root / "PROBE_PLAN_V1_4.json")
            taxonomy_path = parent_root / "TAXONOMY_BINDING.json"
            taxonomy = _load(taxonomy_path)
            if receipt.get("status") != "PASS" or audit.get("status") != "PASS":
                errors.append(f"PARENT_GATE_A_STATUS:{key}")
            if receipt.get("canonical_parent_key") != key or receipt.get("snapshot_count") != 24 or receipt.get("source_commit") != manifest["downstream_source"]["commit"] or receipt.get("source_tree") != manifest["downstream_source"]["tree"] or receipt.get("intervention_executed") is not False or receipt.get("outcomes_read") is not False or receipt.get("protected_counters") != COUNTERS:
                errors.append(f"PARENT_RECEIPT_BINDING:{key}")
            if taxonomy.get("status") != "PASS" or receipt.get("taxonomy_binding_sha256") != _sha_file(taxonomy_path):
                errors.append(f"PARENT_TAXONOMY_BINDING:{key}")
            rows = clean.get("rows")
            probes = plan.get("probe_steps")
            if not isinstance(rows, list) or not isinstance(probes, list) or len(probes) != 24 or len({str(row.get("probe_id")) for row in probes}) != 24 or plan.get("selection_version") != "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1" or plan.get("outcomes_read") is not False:
                errors.append(f"PARENT_PLAN_BINDING:{key}")
                continue
            actions = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in rows]
            action_sha = _sha_json(actions)
            parent_entry = next((row for row in manifest.get("parents", []) if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == key), None)
            if parent_entry is None or parent_entry.get("status") != "PASS" or parent_entry.get("clean_reference_action_sequence_sha256") != action_sha:
                errors.append(f"PARENT_GLOBAL_ENTRY:{key}")
            snapshot_rows = receipt.get("snapshots", [])
            canary_rows = receipt.get("canary_receipts", [])
            for index, probe in enumerate(probes):
                probe_id = str(probe["probe_id"])
                snapshot_row = snapshot_rows[index] if index < len(snapshot_rows) else {}
                snapshot_root = parent_root / str(snapshot_row.get("path"))
                loaded = load_snapshot(snapshot_root, materialize_torch=False)
                snapshot_manifest = loaded["manifest"]
                payload = loaded["payload"]
                if not REQUIRED_PAYLOAD.issubset(payload):
                    errors.append(f"SNAPSHOT_FIELDS:{key}:{probe_id}")
                hashes = assert_primary_observation_exact(payload)
                if snapshot_row.get("manifest_sha256") != _sha_file(snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json") or snapshot_row.get("manifest_sha256") != global_probes.get((key, probe_id), {}).get("snapshot_manifest_sha256") or snapshot_manifest.get("binding", {}).get("parent_key") != key or snapshot_manifest.get("binding", {}).get("source_commit") != manifest["downstream_source"]["commit"] or snapshot_manifest.get("binding", {}).get("source_tree") != manifest["downstream_source"]["tree"] or snapshot_manifest.get("fresh_render_equality_gate_used") is not False or snapshot_manifest.get("primary_input_authority") != "loaded_frozen_canonical_bytes":
                    errors.append(f"SNAPSHOT_BINDING:{key}:{probe_id}")
                if hashes.get("raw_observation_sha256") != global_probes.get((key, probe_id), {}).get("raw_observation_sha256") or hashes.get("policy_rgb_224_sha256") != probe.get("policy_rgb_224_sha256") or hashes.get("policy_input_sha256") != probe.get("policy_input_sha256"):
                    errors.append(f"SNAPSHOT_PRIMARY_HASH:{key}:{probe_id}")
                state = payload.get("full_simulator_state", {}).get("registered_flat_state")
                if state is None or hashlib.sha256(np.asarray(state).tobytes(order="C")).hexdigest() != str(probe.get("state_sha256")):
                    errors.append(f"SNAPSHOT_STATE_HASH:{key}:{probe_id}")
                window = payload.get("clean_reference_action_window")
                if _sha_json(window) != global_probes.get((key, probe_id), {}).get("clean_reference_action_window_sha256"):
                    errors.append(f"SNAPSHOT_ACTION_WINDOW:{key}:{probe_id}")
                if index >= len(canary_rows) or canary_rows[index].get("status") != "PASS" or canary_rows[index].get("intervention_executed") is not False or canary_rows[index].get("fresh_render_equality_gate_used") is not False:
                    errors.append(f"SNAPSHOT_CANARY:{key}:{probe_id}")
                for arm in ("CONTROL", "T3", "T5", "T10"):
                    branch = global_branches.get((key, probe_id, arm))
                    expected_id = "m4-v2-plan-" + hashlib.sha256(f"M4_V2_PLAN::{key}::{probe_id}::R0::{arm}".encode()).hexdigest()
                    if not isinstance(branch, Mapping) or branch.get("branch_id") != expected_id or branch.get("execution_status") != "PLANNED_NOT_EXECUTED" or branch.get("outcomes_read") is not False or branch.get("protected_counters") != COUNTERS:
                        errors.append(f"BRANCH_AUTHORITY:{key}:{probe_id}:{arm}")
            for path in parent_root.rglob("*"):
                if path.is_file() and path.name in FORBIDDEN_BRANCH_NAMES:
                    errors.append(f"INTERVENTION_ARTIFACT_PRESENT:{key}:{path.name}")
        except (OSError, ValueError, KeyError, TypeError, CausalSnapshotError, AssertionError) as exc:
            errors.append(f"PARENT_AUDIT_ERROR:{key}:{type(exc).__name__}:{exc}")
    result = {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT_V1", "status": "PASS" if not errors else "FAIL", "auditor_role": "independent_plan_snapshot_auditor", "root": str(root), "parent_count": 40, "probe_count": 960, "planned_branch_authority_count": 3840, "selection_outcomes_read": False, "intervention_executed": False, "v_phys_generated": False, "errors": sorted(set(errors)), "protected_counters": dict(COUNTERS)}
    (root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = _audit(args.root.resolve())
    print(json.dumps({"status": result["status"], "errors": len(result.get("errors", []))}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
