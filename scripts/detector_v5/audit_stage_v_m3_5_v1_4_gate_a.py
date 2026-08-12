#!/usr/bin/env python3
"""Independent Gate-A audit; this does not import producer decision helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_canonical_execution_core import canonical_value
from gripper_attack.stage_v_causal_observation_snapshot import (
    CausalSnapshotError,
    assert_primary_observation_exact,
    load_snapshot,
)


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
REQUIRED_PAYLOAD = {
    "probe",
    "full_simulator_state",
    "controller_and_wrapper_runtime_state",
    "required_rng_state",
    "raw_observation",
    "raw_observation_sha256",
    "canonical_policy_rgb_224",
    "processed_image",
    "processed_image_sha256",
    "input_ids",
    "pixel_values",
    "attention_mask",
    "attention_mask_present",
    "prompt",
    "decode_config",
    "clean_reference_action_window",
}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _inside(root: Path, relative: Any) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def _gripper_runtime_complete(runtime: Any) -> bool:
    if not isinstance(runtime, Mapping) or runtime.get("schema") != "STAGE_V_CONTROLLER_WRAPPER_RUNTIME_STATE_V2":
        return False
    robots = runtime.get("robots")
    if not isinstance(robots, list) or not robots:
        return False
    required = {"current_action", "speed", "dof"}
    return all(isinstance(row, Mapping) and isinstance(row.get("gripper"), Mapping) and required.issubset(row["gripper"]) for row in robots)


def _audit(root: Path, *, parent_key: str, source_commit: str, source_tree: str) -> dict[str, Any]:
    errors: list[str] = []
    receipt_path = root / "M3_5_V1_4_GATE_A_RECEIPT.json"
    if not receipt_path.is_file():
        errors.append("GATE_A_RECEIPT_MISSING")
        receipt = {}
    else:
        receipt = _load(receipt_path)
    if receipt.get("schema") != "STAGE_V_M3_5_V1_4_GATE_A_RECEIPT_V1":
        errors.append("RECEIPT_SCHEMA_INVALID")
    if receipt.get("status") != "PASS":
        errors.append("PRODUCER_GATE_A_NOT_PASS")
    if receipt.get("canonical_parent_key") != parent_key:
        errors.append("PARENT_IDENTITY_MISMATCH")
    if receipt.get("source_commit") != source_commit or receipt.get("source_tree") != source_tree:
        errors.append("SOURCE_BINDING_MISMATCH")
    if receipt.get("intervention_executed") is not False or receipt.get("outcomes_read") is not False:
        errors.append("ZERO_TREATMENT_OR_OUTCOME_BOUNDARY_INVALID")
    if receipt.get("fresh_render_equality_gate_used") is not False or receipt.get("primary_input_authority") != "loaded_frozen_canonical_bytes":
        errors.append("PRIMARY_RENDER_BOUNDARY_INVALID")
    if receipt.get("protected_counters") != COUNTERS:
        errors.append("PROTECTED_COUNTERS_NONZERO")
    snapshot_rows = receipt.get("snapshots") if isinstance(receipt.get("snapshots"), list) else []
    if len(snapshot_rows) != 24:
        errors.append(f"SNAPSHOT_COUNT_INVALID:{len(snapshot_rows)}/24")
    canary_rows = receipt.get("canary_receipts") if isinstance(receipt.get("canary_receipts"), list) else []
    if len(canary_rows) != len(snapshot_rows):
        errors.append("CANARY_COUNT_MISMATCH")
    for index, row in enumerate(snapshot_rows):
        if not isinstance(row, Mapping):
            errors.append(f"SNAPSHOT_ROW_INVALID:{index}")
            continue
        observation_hashes = None
        try:
            snapshot_root = _inside(root, row.get("path"))
            loaded = load_snapshot(snapshot_root, materialize_torch=True)
            manifest = loaded["manifest"]
            payload = loaded["payload"]
            if _sha_file(snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json") != str(row.get("manifest_sha256")):
                errors.append(f"SNAPSHOT_MANIFEST_BINDING_MISMATCH:{index}")
            if manifest.get("binding", {}).get("parent_key") != parent_key:
                errors.append(f"SNAPSHOT_PARENT_BINDING_MISMATCH:{index}")
            if manifest.get("fresh_render_equality_gate_used") is not False or manifest.get("primary_input_authority") != "loaded_frozen_canonical_bytes":
                errors.append(f"SNAPSHOT_PRIMARY_AUTHORITY_INVALID:{index}")
            if not REQUIRED_PAYLOAD.issubset(payload):
                errors.append(f"SNAPSHOT_PAYLOAD_FIELDS_MISSING:{index}")
            observation_hashes = assert_primary_observation_exact(payload)
            runtime = payload.get("controller_and_wrapper_runtime_state", {})
            if not _gripper_runtime_complete(runtime):
                errors.append(f"GRIPPER_RUNTIME_STATE_BINDING_INVALID:{index}")
            if not isinstance(runtime, Mapping) or not isinstance(runtime.get("rng"), Mapping) or _sha_json(canonical_value(runtime.get("rng"))) != _sha_json(canonical_value(payload.get("required_rng_state"))):
                errors.append(f"RNG_BINDING_MISMATCH:{index}")
            window = payload.get("clean_reference_action_window")
            if not isinstance(window, list) or len(window) < 20:
                errors.append(f"REFERENCE_WINDOW_SHORT:{index}")
            else:
                steps = [item.get("step") for item in window if isinstance(item, Mapping)]
                if steps != list(range(int(payload["probe"]["step"]), int(payload["probe"]["step"]) + len(window))):
                    errors.append(f"REFERENCE_WINDOW_NONCONTIGUOUS:{index}")
                for action_index, action in enumerate(window):
                    if not isinstance(action, Mapping) or not isinstance(action.get("raw_policy_action"), list) or not isinstance(action.get("env_action"), list) or len(action["raw_policy_action"]) != 7 or len(action["env_action"]) != 7:
                        errors.append(f"REFERENCE_ACTION_INVALID:{index}:{action_index}")
                        continue
                    expected_sha = _sha_json({"raw": action["raw_policy_action"], "env": action["env_action"]})
                    if action.get("action_sha256") != expected_sha:
                        errors.append(f"REFERENCE_ACTION_SHA_MISMATCH:{index}:{action_index}")
        except (OSError, ValueError, KeyError, CausalSnapshotError, TypeError) as exc:
            errors.append(f"SNAPSHOT_AUDIT_ERROR:{index}:{type(exc).__name__}:{exc}")
        if index < len(canary_rows):
            canary = canary_rows[index]
            if not isinstance(canary, Mapping) or canary.get("status") != "PASS" or canary.get("intervention_executed") is not False or canary.get("fresh_render_equality_gate_used") is not False:
                errors.append(f"CANARY_HARD_GATE_INVALID:{index}")
            checks = canary.get("primary_exact_checks", {}) if isinstance(canary, Mapping) else {}
            if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
                errors.append(f"CANARY_EXACT_CHECKS_INVALID:{index}")
            if not isinstance(canary, Mapping) or canary.get("primary_observation_hashes") != observation_hashes:
                errors.append(f"CANARY_OBSERVATION_BINDING_INVALID:{index}")
    result = {
        "schema": "STAGE_V_M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT_V1",
        "status": "PASS" if not errors else "FAIL",
        "auditor_role": "independent_gate_a_snapshot_auditor_no_producer_decision_helper",
        "canonical_parent_key": parent_key,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "snapshot_count": len(snapshot_rows),
        "errors": sorted(set(errors)),
        "protected_counters": dict(COUNTERS),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = _audit(args.root.resolve(), parent_key=args.parent_key, source_commit=args.source_commit, source_tree=args.source_tree)
    output = args.output.resolve() if args.output else args.root.resolve() / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
    output.write_bytes((json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
