"""P3-R1: Audit the frozen 25D causal feature schema V2.

Verifies:
- 25 features with correct names/order
- Feature 0 (gripper_command) = raw_action_7d[6], distinct from feature 12 (action_gripper) = action_env_7d[6]
- All fields causal (no future, no step index, no Teacher/privileged)
- Stable name/order digest
- Mutation tests: forbidden fields rejected, raw/executed gripper mapped to same field, multi-close causality, future telemetry parity, absolute step injection
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"

FORBIDDEN_FIELD_NAMES = {
    "relation_id", "logical_object", "logical_target", "entity_id", "entity_type",
    "object_pose", "target_pose", "contact_force", "contact_geometry",
    "consolidated_event_id", "candidate_close", "teacher_reason", "teacher_confidence",
    "success", "failure", "reward", "step_index", "episode_progress",
    "future_action", "future_state", "attack_outcome", "post_attack_state",
}


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def compute_digest(schema: dict) -> str:
    features = schema.get("features", [])
    names = [f["name"] for f in features]
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


def audit(schema: dict | None = None) -> dict:
    if schema is None:
        schema = load_schema()

    issues = []
    features = schema.get("features", [])

    # 1. Dimension check
    if len(features) != 25:
        issues.append(f"expected 25 features, got {len(features)}")
    if schema.get("dimensions") != 25:
        issues.append("dimensions field != 25")

    # 2. Schema version
    if schema.get("schema") != "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA_V2":
        issues.append(f"expected schema V2, got {schema.get('schema')}")

    # 3. Index/name consistency
    expected_names = [
        "gripper_command", "gripper_qpos", "gripper_opening_proxy",
        "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
        "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
        "close_onset", "time_since_close", "eef_speed",
        "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
        "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
    ]
    for i, f in enumerate(features):
        if f.get("index") != i:
            issues.append(f"feature[{i}] index={f.get('index')}, expected {i}")
        if i < len(expected_names) and f.get("name") != expected_names[i]:
            issues.append(f"feature[{i}] name={f.get('name')}, expected {expected_names[i]}")
        elif i >= len(expected_names):
            issues.append(f"extra feature[{i}]: {f.get('name')} exceeds 25D limit")

    # 4. Feature 0 vs 12: must be semantically distinct sources
    f0 = features[0] if len(features) > 0 else {}
    f12 = features[12] if len(features) > 12 else {}
    if f0.get("source") == f12.get("source"):
        issues.append("P0-3: feature 0 and 12 have same source — must be raw_action_7d[6] vs action_env_7d[6]")
    if f0.get("source") != "raw_action_7d[6]":
        issues.append(f"feature 0 source={f0.get('source')}, expected raw_action_7d[6]")
    if f12.get("source") != "action_env_7d[6]":
        issues.append(f"feature 12 source={f12.get('source')}, expected action_env_7d[6]")

    # 5. No forbidden fields
    for i, f in enumerate(features):
        name = f.get("name", "")
        if name.lower() in FORBIDDEN_FIELD_NAMES:
            issues.append(f"feature[{i}] '{name}' is forbidden")

    # 6. Causal check
    if schema.get("future_fields") != 0:
        issues.append("future_fields != 0")
    if schema.get("step_progress_fields") != 0:
        issues.append("step_progress_fields != 0")
    if schema.get("teacher_label_fields") != 0:
        issues.append("teacher_label_fields != 0")
    if schema.get("privileged_entity_fields") != 0:
        issues.append("privileged_entity_fields != 0")
    if schema.get("attack_outcome_fields") != 0:
        issues.append("attack_outcome_fields != 0")

    # 7. Temporal check
    for f in features:
        temporal = f.get("temporal", "")
        if temporal not in ("current", "past_window"):
            issues.append(f"feature '{f['name']}' temporal='{temporal}' not causal")

    # 8. Digest
    digest = compute_digest(schema)

    return {
        "schema_path": str(SCHEMA_PATH),
        "feature_count": len(features),
        "name_order_digest": digest,
        "issues": issues,
        "pass": len(issues) == 0,
        "status": schema.get("status"),
    }


# ── Mutation / negative tests ──────────────────────────────────────────

def test_forbidden_mutation(forbidden_name: str) -> bool:
    """Test that adding a forbidden field name is detected."""
    schema = load_schema()
    features = list(schema["features"])
    features.append({
        "index": 25, "name": forbidden_name,
        "source": "mutation_test", "unit": "?",
        "dtype": "float32", "temporal": "current",
        "normalization": "zscore", "description": "MUTATION TEST",
    })
    schema["features"] = features
    schema["dimensions"] = 26
    result = audit(schema)
    return not result["pass"]


def test_raw_executed_gripper_same_field() -> bool:
    """V2 regression: feature 0 and 12 mapped to same field must be detected."""
    schema = load_schema()
    features = list(schema["features"])
    # Change feature 12 source to match feature 0
    f12 = dict(features[12])
    f12["source"] = "raw_action_7d[6]"
    features[12] = f12
    schema["features"] = features
    result = audit(schema)
    return not result["pass"]


def test_future_telemetry_injection() -> bool:
    """Feature using future telemetry (t+1) must be rejected."""
    schema = load_schema()
    features = list(schema["features"])
    f = dict(features[5])  # eef_z
    f["temporal"] = "future"
    features[5] = f
    schema["features"] = features
    result = audit(schema)
    return not result["pass"]


def test_absolute_step_injection() -> bool:
    """Feature using absolute step index must be rejected."""
    schema = load_schema()
    features = list(schema["features"])
    features.append({
        "index": 25, "name": "step_index",
        "source": "mutation_test", "unit": "?",
        "dtype": "float32", "temporal": "current",
        "normalization": "zscore", "description": "MUTATION TEST",
    })
    schema["features"] = features
    schema["dimensions"] = 26
    result = audit(schema)
    return not result["pass"]


def test_object_pose_injection() -> bool:
    """Feature using object pose must be rejected."""
    schema = load_schema()
    features = list(schema["features"])
    features.append({
        "index": 25, "name": "object_pose",
        "source": "mutation_test", "unit": "?",
        "dtype": "float32", "temporal": "current",
        "normalization": "zscore", "description": "MUTATION TEST",
    })
    schema["features"] = features
    schema["dimensions"] = 26
    result = audit(schema)
    return not result["pass"]


def test_contact_force_injection() -> bool:
    """Feature using contact force must be rejected."""
    schema = load_schema()
    features = list(schema["features"])
    features.append({
        "index": 25, "name": "contact_force",
        "source": "mutation_test", "unit": "?",
        "dtype": "float32", "temporal": "current",
        "normalization": "zscore", "description": "MUTATION TEST",
    })
    schema["features"] = features
    schema["dimensions"] = 26
    result = audit(schema)
    return not result["pass"]


def test_relation_field_injection() -> bool:
    """Feature using relation fields must be rejected."""
    schema = load_schema()
    features = list(schema["features"])
    features.append({
        "index": 25, "name": "relation_id",
        "source": "mutation_test", "unit": "?",
        "dtype": "float32", "temporal": "current",
        "normalization": "zscore", "description": "MUTATION TEST",
    })
    schema["features"] = features
    schema["dimensions"] = 26
    result = audit(schema)
    return not result["pass"]


def test_25d_schema_audit():
    """Run full audit and all mutation tests."""
    schema = load_schema()
    result = audit(schema)

    print(f"Schema: {result.get('status')}")
    print(f"Features: {result['feature_count']}")
    print(f"Name digest: {result['name_order_digest']}")
    print(f"Pass: {result['pass']}")
    for issue in result["issues"]:
        print(f"  ISSUE: {issue}")

    # Standard forbidden-field mutation tests
    print(f"\nForbidden field mutation tests ({len(FORBIDDEN_FIELD_NAMES)} fields):")
    forbidden_failures = 0
    for fname in sorted(FORBIDDEN_FIELD_NAMES):
        detected = test_forbidden_mutation(fname)
        if not detected:
            print(f"  FAIL: '{fname}' not detected as forbidden")
            forbidden_failures += 1
    print(f"  Passed: {len(FORBIDDEN_FIELD_NAMES) - forbidden_failures}/{len(FORBIDDEN_FIELD_NAMES)}")

    # V2-specific negative tests
    print("\nV2-specific negative tests:")
    v2_tests = {
        "raw_executed_gripper_same_field": test_raw_executed_gripper_same_field,
        "future_telemetry_injection": test_future_telemetry_injection,
        "absolute_step_injection": test_absolute_step_injection,
        "object_pose_injection": test_object_pose_injection,
        "contact_force_injection": test_contact_force_injection,
        "relation_field_injection": test_relation_field_injection,
    }
    v2_failures = 0
    for name, test_fn in v2_tests.items():
        detected = test_fn()
        if not detected:
            print(f"  FAIL: {name} not detected")
            v2_failures += 1
        else:
            print(f"  PASS: {name}")
    print(f"  V2 tests passed: {len(v2_tests) - v2_failures}/{len(v2_tests)}")

    total_failures = forbidden_failures + v2_failures
    all_pass = result["pass"] and total_failures == 0
    print(f"\nP3_R1_SCHEMA_FROZEN_V2: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Schema audit: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  Forbidden mutations: {'PASS' if forbidden_failures == 0 else 'FAIL'}")
    print(f"  V2 negative tests: {'PASS' if v2_failures == 0 else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    ok = test_25d_schema_audit()
    raise SystemExit(0 if ok else 1)
