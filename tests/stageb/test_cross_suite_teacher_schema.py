from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "docs" / "schemas" / "cross_suite_teacher_label_schema_v1.md"
PREREG = REPO / "reports" / "CROSS_SUITE_LAYER1_RESOLVER_PREREG.md"

EPISODE_FIELDS = [
    "teacher_executed",
    "teacher_run_id",
    "teacher_version",
    "ontology_version",
    "resolver_version",
    "source_episode_sha",
    "mechanism_type",
    "mechanism_eligible",
    "object_binding_status",
    "target_binding_status",
    "teacher_status",
    "teacher_semantic_abstain",
    "abstain_reason",
    "event_count",
    "manual_review_required",
]
EVENT_FIELDS = [
    "event_id",
    "object_body_name",
    "object_joint_name",
    "target_body_or_site_name",
    "binding_source",
    "binding_confidence_class",
    "close_onset_step",
    "grasp_established_step",
    "lift_onset_step",
    "stable_carry_start",
    "teacher_window_start",
    "teacher_anchor_step",
    "teacher_window_end",
    "release_onset_step",
    "event_valid",
    "event_invalid_reason",
]
STATUSES = [
    "ELIGIBLE_EVENT",
    "CORRECT_SEMANTIC_ABSTAIN",
    "NO_RELEVANT_GRASP_EVENT",
    "OBJECT_BINDING_AMBIGUOUS",
    "TARGET_BINDING_AMBIGUOUS",
    "MULTI_EVENT_AUDIT_ONLY",
    "RESOLVER_FAILED",
    "SCHEMA_INVALID",
]
FORBIDDEN = ["mlp_emit_step", "mlp_triggered", "corridor_p", "release_p", "pred_phase"]


def test_teacher_schema_declares_required_fields_and_statuses():
    text = SCHEMA.read_text(encoding="utf-8")
    for field in EPISODE_FIELDS + EVENT_FIELDS:
        assert f"`{field}`" in text
    for status in STATUSES:
        assert status in text
    assert "privileged_sidecar_resolved_v1.json" in text
    assert "teacher_event_labels_v1.csv" in text
    assert "must not overwrite" in text


def test_prereg_is_gate_only_and_forbids_detector_leakage():
    text = PREREG.read_text(encoding="utf-8")
    assert "LAYER1_STAGE = PREREG_ONLY" in text
    assert "FULL_RESOLVER = NOT_RUN" in text
    assert "VIS_RAND_ATTACK = NO_GO" in text
    for field in FORBIDDEN:
        assert field in text
    assert "No Teacher labels are generated yet" in text
    assert "No Layer 2 zero-shot timing transfer is evaluated" in text



def test_teacher_schema_join_fields_binding_enums_and_invariants():
    text = SCHEMA.read_text(encoding="utf-8")
    for field in ["episode_key", "suite", "task_idx", "state_id", "source_episode_relpath"]:
        assert f"`{field}`" in text
    for status in ["BOUND_EXACT", "BOUND_BDDL_ONTOLOGY", "BOUND_STRUCTURED_FALLBACK", "AMBIGUOUS", "NOT_APPLICABLE", "FAILED"]:
        assert status in text
    for phrase in [
        "teacher_executed=false implies",
        "teacher_status=ELIGIBLE_EVENT implies",
        "teacher_status=CORRECT_SEMANTIC_ABSTAIN implies",
        "teacher_status=MULTI_EVENT_AUDIT_ONLY implies",
    ]:
        assert phrase in text
    assert "teacher_run`" not in text
