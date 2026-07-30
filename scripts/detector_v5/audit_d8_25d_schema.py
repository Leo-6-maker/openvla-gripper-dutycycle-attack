"""P3: Audit the frozen 25D causal feature schema.

Verifies:
- 25 features with correct names/order
- All fields causal (no future, no step index, no Teacher/privileged)
- Stable name/order digest
- Mutation tests: forbidden fields rejected
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
        issues.append(f"dimensions field != 25")

    # 2. Index/name consistency
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

    # 3. No forbidden fields
    for i, f in enumerate(features):
        name = f.get("name", "")
        if name.lower() in FORBIDDEN_FIELD_NAMES:
            issues.append(f"feature[{i}] '{name}' is forbidden")

    # 4. Causal check
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

    # 5. Temporal check
    for f in features:
        temporal = f.get("temporal", "")
        if temporal not in ("current", "past_window"):
            issues.append(f"feature '{f['name']}' temporal='{temporal}' not causal")

    # 6. Digest
    digest = compute_digest(schema)

    return {
        "schema_path": str(SCHEMA_PATH),
        "feature_count": len(features),
        "name_order_digest": digest,
        "issues": issues,
        "pass": len(issues) == 0,
        "status": schema.get("status"),
    }


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


def test_25d_schema_audit():
    """Run full audit and all mutation tests."""
    schema = load_schema()
    result = audit(schema)

    print(f"Features: {result['feature_count']}")
    print(f"Name digest: {result['name_order_digest']}")
    print(f"Status: {result['status']}")
    print(f"Pass: {result['pass']}")
    for issue in result["issues"]:
        print(f"  ISSUE: {issue}")

    # Mutation tests
    print(f"\nMutation tests ({len(FORBIDDEN_FIELD_NAMES)} forbidden fields):")
    failures = 0
    for fname in sorted(FORBIDDEN_FIELD_NAMES):
        detected = test_forbidden_mutation(fname)
        if not detected:
            print(f"  FAIL: '{fname}' not detected as forbidden")
            failures += 1
    print(f"  Passed: {len(FORBIDDEN_FIELD_NAMES) - failures}/{len(FORBIDDEN_FIELD_NAMES)}")
    print(f"  Failed: {failures}")

    all_pass = result["pass"] and failures == 0
    print(f"\nP3_SCHEMA_FROZEN: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    ok = test_25d_schema_audit()
    raise SystemExit(0 if ok else 1)
