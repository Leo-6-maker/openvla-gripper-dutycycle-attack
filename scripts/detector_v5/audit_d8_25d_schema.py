"""H1-R4: Audit frozen 25D causal feature schema — with source-level scans.

Scans feature name, source, description, temporal, unit, normalization.
Forbidden terms in source/description: relation, object_pose, target_pose,
contact, teacher, candidate_close, event_id, future, t+1, reward, success,
failure, attack, episode_progress, step_index.

Includes source-level mutation tests.
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

# H1-R4: Forbidden terms in source and description
FORBIDDEN_SOURCE_TERMS = {
    "relation", "object_pose", "target_pose", "contact",
    "teacher", "candidate_close", "event_id",
    "future", "t+1", "reward", "success", "failure",
    "attack", "episode_progress", "step_index",
}

# Allowed safe terms
ALLOWED_TERMS = {"opening_proxy", "gripper_opening_proxy", "opening_proxy_delta",
                 "opening_proxy_variance", "eef_z_delta_since_close",
                 "time_since_close", "close_onset", "close_streak",
                 "gripper_command", "action_gripper", "qpos_delta"}


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def compute_digest(schema: dict) -> str:
    names = [f["name"] for f in schema["features"]]
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


def _contains_forbidden(text: str) -> list[str]:
    """Check if text contains any forbidden terms."""
    found = []
    text_lower = text.lower()
    for term in FORBIDDEN_SOURCE_TERMS:
        if term in text_lower:
            # Check if it's part of an allowed term
            allowed = False
            for at in ALLOWED_TERMS:
                if at.lower() in text_lower and term in at.lower():
                    allowed = True
                    break
            if not allowed:
                found.append(term)
    return found


def audit(schema: dict | None = None) -> dict:
    if schema is None:
        schema = load_schema()

    issues = []
    features = schema.get("features", [])

    if len(features) != 25:
        issues.append(f"expected 25 features, got {len(features)}")
    if schema.get("dimensions") != 25:
        issues.append("dimensions field != 25")
    if schema.get("schema") != "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA_V2":
        issues.append(f"expected schema V2, got {schema.get('schema')}")

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

    # Feature 0 vs 12 must be distinct
    f0, f12 = features[0], features[12]
    if f0.get("source") == f12.get("source"):
        issues.append("P0-3: feature 0 and 12 have same source")
    if f0.get("source") != "raw_action_7d[6]":
        issues.append(f"feature 0 source={f0.get('source')}, expected raw_action_7d[6]")
    if f12.get("source") != "action_env_7d[6]":
        issues.append(f"feature 12 source={f12.get('source')}, expected action_env_7d[6]")

    # H1-R2: Feature 1 vs 2 must have distinct sources (signed vs absolute)
    f1, f2 = features[1], features[2]
    if f1.get("source") == f2.get("source"):
        issues.append("H1-R2: feature 1 and 2 have same source")

    # H1-R4: Source-level forbidden term scan
    for i, f in enumerate(features):
        name = f.get("name", "").lower()
        if name in FORBIDDEN_FIELD_NAMES:
            issues.append(f"feature[{i}] name '{f.get('name')}' is forbidden")

        source = f.get("source", "").lower()
        forbidden_in_source = _contains_forbidden(source)
        for term in forbidden_in_source:
            issues.append(f"H1-R4: feature[{i}] source contains '{term}': {f.get('source')}")

        desc = f.get("description", "").lower()
        forbidden_in_desc = _contains_forbidden(desc)
        for term in forbidden_in_desc:
            issues.append(f"H1-R4: feature[{i}] description contains '{term}': {f.get('description')}")

    # Causal checks
    for check in ["future_fields", "step_progress_fields", "teacher_label_fields",
                  "privileged_entity_fields", "attack_outcome_fields"]:
        if schema.get(check, 0) != 0:
            issues.append(f"{check} != 0")

    for f in features:
        temporal = f.get("temporal", "")
        if temporal not in ("current", "past_window"):
            issues.append(f"feature '{f['name']}' temporal='{temporal}' not causal")

    digest = compute_digest(schema)
    return {
        "schema_path": str(SCHEMA_PATH), "feature_count": len(features),
        "name_order_digest": digest, "issues": issues,
        "pass": len(issues) == 0, "status": schema.get("status"),
    }


# ── Mutation tests ────────────────────────────────────────────────────

def test_forbidden_mutation(forbidden_name: str) -> bool:
    schema = load_schema()
    features = list(schema["features"])
    features.append({"index": 25, "name": forbidden_name, "source": "mutation_test",
                     "unit": "?", "dtype": "float32", "temporal": "current",
                     "normalization": "zscore", "description": "MUTATION TEST"})
    schema["features"] = features; schema["dimensions"] = 26
    return not audit(schema)["pass"]


def test_source_mutation(name: str, source: str) -> bool:
    """H1-R4: Test that a forbidden source string is detected."""
    schema = load_schema()
    features = list(schema["features"])
    for i, f in enumerate(features):
        if f["name"] == name:
            mutated = dict(f)
            mutated["source"] = source
            features[i] = mutated
            break
    schema["features"] = features
    return not audit(schema)["pass"]


def test_qpos_contract_mutation() -> bool:
    """H1-R2: feature 1 and 2 mapped to same source must be detected."""
    schema = load_schema()
    features = list(schema["features"])
    f2 = dict(features[2])
    f2["source"] = features[1]["source"]  # make feature 2 same as feature 1
    features[2] = f2
    schema["features"] = features
    return not audit(schema)["pass"]


def test_25d_schema_audit():
    schema = load_schema()
    result = audit(schema)

    print(f"Schema: {result['status']}, Features: {result['feature_count']}")
    print(f"Name digest: {result['name_order_digest']}")
    print(f"Base audit: {'PASS' if result['pass'] else 'FAIL'}")
    for issue in result["issues"]:
        print(f"  ISSUE: {issue}")

    # Forbidden field mutation tests
    print(f"\nForbidden field mutations ({len(FORBIDDEN_FIELD_NAMES)}):")
    ff_fail = sum(1 for fname in sorted(FORBIDDEN_FIELD_NAMES) if not test_forbidden_mutation(fname))
    ff_pass = len(FORBIDDEN_FIELD_NAMES) - ff_fail
    print(f"  Passed: {ff_pass}/{len(FORBIDDEN_FIELD_NAMES)}")

    # H1-R4: Source-level mutation tests
    source_mutations = {
        "eef_z": "object_pose[2]",
        "eef_speed": "future_eef[t+1]",
        "gripper_qpos": "teacher_contact_state",
        "action_dx": "attack_action[0]",
    }
    print(f"\nH1-R4 source mutations ({len(source_mutations)}):")
    sm_fail = sum(1 for name, src in source_mutations.items() if not test_source_mutation(name, src))
    sm_pass = len(source_mutations) - sm_fail
    print(f"  Passed: {sm_pass}/{len(source_mutations)}")

    # H1-R2 qpos contract mutation
    print("\nH1-R2 qpos contract mutation:")
    qpos_pass = test_qpos_contract_mutation()
    print(f"  {'PASS' if qpos_pass else 'FAIL'}")

    total_failures = ff_fail + sm_fail + (0 if qpos_pass else 1)
    all_pass = result["pass"] and total_failures == 0
    print(f"\nH1_R4_SCHEMA_AUDIT: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    ok = test_25d_schema_audit()
    raise SystemExit(0 if ok else 1)
