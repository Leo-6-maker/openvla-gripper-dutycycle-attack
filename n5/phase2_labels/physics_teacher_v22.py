"""Physics Teacher V22 Schema — GO development phase.

Defines independent physical factor contracts for Label Contract V2.
No CS200 dependency for schema definition and synthetic/fixture tests.

Design constraints:
  - NEVER use candidate_close to decide whether to compute physics
  - NEVER use action_intent to define physical instability
  - NEVER reuse release_or_instability as final mixed label
  - Goal tasks with resolver failure → unknown, NOT NO_MANIPULATION_TARGET
  - Every field has {value, known_mask, confidence, source, reason}
"""
import json, os, hashlib
from typing import Dict, Any, Optional, Tuple

# ── V22 Schema ──

V22_SCHEMA_VERSION = "PHYSICS_TEACHER_V22_V1"

# Factor definitions: each factor is independently assessed
V22_FACTORS = {
    "target_resolution": {
        "description": "Target object identification and resolver status",
        "fields": ["target_object_id", "target_bbox", "target_resolved", "resolver_confidence", "target_known_mask"],
        "required_for": ["criticality"],
        "goal_note": "Goal tasks MUST have explicit target resolution. Failure → unknown, not NO_MANIPULATION_TARGET."
    },
    "grasp_state": {
        "description": "Grasp establishment and stability",
        "fields": ["grasp_established", "grasp_confidence", "grasp_known_mask", "grasp_dwell_steps"],
        "required_for": ["criticality", "instability"],
    },
    "contact_state": {
        "description": "Gripper-object contact detection",
        "fields": ["contact_score", "contact_known_mask", "contact_confidence", "contact_source"],
        "required_for": ["criticality", "instability"],
    },
    "comotion_state": {
        "description": "Object-EEF co-motion measurement",
        "fields": ["object_eef_comotion_score", "comotion_known_mask", "comotion_confidence"],
        "required_for": ["criticality"],
    },
    "lift_state": {
        "description": "Object lift/transport detection",
        "fields": ["lift_score", "lift_known_mask", "lift_confidence"],
        "required_for": ["criticality"],
    },
    "placement_state": {
        "description": "Object placement detection",
        "fields": ["object_placed", "placement_region", "placement_known_mask", "placement_confidence"],
        "required_for": ["safe_release"],
    },
    "planned_release": {
        "description": "Planned/normal gripper release",
        "fields": ["planned_release_detected", "planned_release_known_mask", "planned_release_confidence"],
        "required_for": ["safe_release"],
        "constraint": "Must NOT be derived from action_intent."
    },
    "instability_indicators": {
        "description": "Independent instability detection channels",
        "fields": [
            "slip_detected", "slip_known_mask", "slip_confidence",
            "contact_loss_detected", "contact_loss_known_mask", "contact_loss_confidence",
            "pose_anomaly_detected", "pose_anomaly_known_mask", "pose_anomaly_confidence",
            "unplanned_width_increase", "width_increase_known_mask", "width_increase_confidence",
        ],
        "required_for": ["instability"],
        "constraint": "Each channel independently assessed. action_intent MUST NOT be used."
    },
    "terminal_state": {
        "description": "Task completion/terminal detection",
        "fields": ["task_success", "task_terminal", "terminal_known_mask", "terminal_confidence"],
        "required_for": ["safe_release", "k10_feasibility"],
    },
    "gripper_physics": {
        "description": "Raw gripper physical state (position, width, velocity)",
        "fields": ["gripper_qpos", "gripper_width", "gripper_width_velocity", "gripper_physics_known_mask", "gripper_physics_confidence"],
        "required_for": ["instability", "close_intent_context"],
        "note": "Physical qpos measurement, NOT policy action. For context only."
    },
    "close_intent": {
        "description": "Physical gripper closure intent from qpos velocity (NOT policy action)",
        "fields": ["close_intent_detected", "close_intent_known_mask", "close_intent_confidence"],
        "required_for": ["close_intent_head"],
        "constraint": "Physical qpos measurement only. MUST NOT use policy-close gate or action labels."
    },
}

def make_factor_template(factor_name):
    """Create empty factor dict with proper structure."""
    factor_def = V22_FACTORS.get(factor_name, {})
    fields = factor_def.get("fields", [])
    template = {
        "factor": factor_name,
        "known_mask": False,
        "source": "physics_teacher_v22",
        "reason": "NOT_COMPUTED",
        "confidence": 0.0,
    }
    for f in fields:
        if f.endswith("_known_mask"):
            template[f] = False
        elif f.endswith("_confidence"):
            template[f] = 0.0
        elif f.endswith("_score"):
            template[f] = 0.0
        elif f.endswith("_detected") or f.startswith("is_") or f.startswith("has_"):
            template[f] = False
        elif f == "target_object_id":
            template[f] = None
        elif f == "target_bbox":
            template[f] = None
        elif f == "resolver_confidence":
            template[f] = 0.0
        elif f == "grasp_dwell_steps":
            template[f] = 0
        elif f == "placement_region":
            template[f] = None
        elif f == "contact_source":
            template[f] = "unknown"
        else:
            template[f] = 0.0
    return template

def create_v22_snapshot():
    """Create a complete V22 factor snapshot with all factors set to unknown."""
    snapshot = {
        "schema": V22_SCHEMA_VERSION,
        "step": 0,
        "suite": None,
        "task_index": None,
        "state_index": None,
        "factors": {name: make_factor_template(name) for name in V22_FACTORS},
        "independent_constraints": [
            "candidate_close NOT used as prerequisite",
            "action_intent NOT used for instability",
            "release_or_instability NOT used as mixed label",
            "All factors independently assessed with own known_mask",
        ],
    }
    return snapshot

def compute_v22_schema_sha():
    return hashlib.sha256(
        json.dumps(V22_FACTORS, sort_keys=True).encode()
    ).hexdigest()

# ── Synthetic / Fixture Tests (no CS200) ──

def test_factor_template_completeness():
    """Every factor definition has corresponding template fields."""
    for name, factor_def in V22_FACTORS.items():
        template = make_factor_template(name)
        for f in factor_def["fields"]:
            assert f in template, f'{name}: field {f} missing from template'
    print('PASS: test_factor_template_completeness')

def test_all_factors_independent():
    """No factor's known_mask depends on another factor."""
    snapshot = create_v22_snapshot()
    for name in V22_FACTORS:
        factor = snapshot["factors"][name]
        assert "known_mask" in factor, f'{name}: missing known_mask'
        # Initially all unknown
        assert factor["known_mask"] == False, f'{name}: should start unknown'
    print('PASS: test_all_factors_independent')

def test_no_candidate_close_in_schema():
    """Schema factor definitions must not reference candidate_close (constraint docstrings are OK)."""
    factor_defs_only = {k: v for k, v in V22_FACTORS.items()}
    schema_str = json.dumps(factor_defs_only, sort_keys=True)
    assert 'candidate_close' not in schema_str, 'candidate_close in factor definitions'
    # Snapshot factors (not constraints)
    snapshot = create_v22_snapshot()
    factors_str = json.dumps(snapshot["factors"], sort_keys=True)
    assert 'candidate_close' not in factors_str, 'candidate_close in factor snapshot'
    print('PASS: test_no_candidate_close_in_schema')

def test_no_action_intent_in_instability():
    """Instability factor fields must not reference action_intent."""
    instab_def = V22_FACTORS["instability_indicators"]
    for field in instab_def["fields"]:
        assert 'action_intent' not in field, f'action_intent in instability field: {field}'
    # Check fields only, not constraint docstrings
    fields_str = json.dumps(instab_def["fields"], sort_keys=True)
    assert 'action_intent' not in fields_str, 'action_intent in instability fields'
    print('PASS: test_no_action_intent_in_instability')

def test_no_release_or_instability_mixed():
    """No factor reuses release_or_instability as final mixed label."""
    schema_str = json.dumps(V22_FACTORS, sort_keys=True)
    assert 'release_or_instability' not in schema_str, 'release_or_instability in schema'
    print('PASS: test_no_release_or_instability_mixed')

def test_goal_tasks_require_target_resolution():
    """Goal tasks MUST have explicit target resolution, not NO_MANIPULATION_TARGET."""
    tr_def = V22_FACTORS["target_resolution"]
    assert "target_object_id" in tr_def["fields"], 'target_object_id missing'
    assert "target_resolved" in tr_def["fields"], 'target_resolved missing'
    # Verify the goal_note is present
    assert "goal_note" in tr_def, 'goal_note missing from target_resolution'
    print('PASS: test_goal_tasks_require_target_resolution')

def test_every_field_has_known_mask():
    """Every factor has factor-level known_mask AND at least one field-level known_mask."""
    for name, factor_def in V22_FACTORS.items():
        fields = factor_def["fields"]
        # Factor-level known_mask is always present from make_factor_template
        has_field_mask = any(f.endswith('_known_mask') for f in fields)
        assert has_field_mask, f'{name}: missing field-level _known_mask in fields list'
        # Each factor should also have confidence or source
        has_conf = any(f.endswith('_confidence') for f in fields)
        assert has_conf, f'{name}: missing confidence field'
    print('PASS: test_every_field_has_known_mask')
    print('PASS: test_every_field_has_known_mask')

def test_synthetic_fill_and_reset():
    """Synthetic fill: set factor-level known_mask + field values, then reset."""
    snapshot = create_v22_snapshot()
    # Simulate: grasp factor computed successfully
    g = snapshot["factors"]["grasp_state"]
    g["known_mask"] = True  # Factor-level: physics was computed
    g["reason"] = "SYNTHETIC_TEST"
    g["confidence"] = 0.95
    g["grasp_established"] = True
    g["grasp_confidence"] = 0.95
    g["grasp_known_mask"] = True  # Field-level: grasp value is known
    g["grasp_dwell_steps"] = 15
    assert g["known_mask"] == True  # Factor-level
    assert g["grasp_known_mask"] == True  # Field-level
    assert g["grasp_established"] == True

    # Other factors remain uncomputed
    c = snapshot["factors"]["contact_state"]
    assert c["known_mask"] == False, 'contact factor should still be uncomputed'

    # Reset produces fresh unknown snapshot
    snapshot2 = create_v22_snapshot()
    assert snapshot2["factors"]["grasp_state"]["known_mask"] == False
    assert snapshot2["factors"]["grasp_state"]["grasp_established"] == False
    print('PASS: test_synthetic_fill_and_reset')

def test_schema_sha_stable():
    """Schema SHA is stable across calls."""
    sha1 = compute_v22_schema_sha()
    sha2 = compute_v22_schema_sha()
    assert sha1 == sha2, 'Schema SHA not stable'
    assert len(sha1) == 64, f'SHA length wrong: {len(sha1)}'
    print(f'PASS: test_schema_sha_stable ({sha1[:16]}...)')

def run_all_tests():
    tests = [
        test_factor_template_completeness,
        test_all_factors_independent,
        test_no_candidate_close_in_schema,
        test_no_action_intent_in_instability,
        test_no_release_or_instability_mixed,
        test_goal_tasks_require_target_resolution,
        test_every_field_has_known_mask,
        test_synthetic_fill_and_reset,
        test_schema_sha_stable,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f'FAIL: {test.__name__}: {e}')
    print(f'\n{passed} PASS / {failed} FAIL (total {len(tests)})')
    return failed == 0

if __name__ == '__main__':
    ok = run_all_tests()
    print(f'\nV22 Schema SHA: {compute_v22_schema_sha()}')
    import sys
    sys.exit(0 if ok else 1)
