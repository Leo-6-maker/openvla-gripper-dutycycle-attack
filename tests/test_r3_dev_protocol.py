import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs" / "R3_DEV_PROTOCOL.json").read_text(encoding="utf-8"))


def test_r3_protocol_is_frozen_before_consumption():
    assert PROTOCOL["status"] == "FROZEN_BEFORE_R3_1_CONSUMPTION"
    assert PROTOCOL["lineage"]["base_commit"] == "6504e94567d9f6bc6394185daf26a60eccf0bb19"
    assert PROTOCOL["lineage"]["valid_r2_execution_commit"] == "f6f619b4fc6b1706aff1cf1967c73e8cc10b8c28"


def test_r3_authorization_is_fit_only_and_fail_closed():
    auth = PROTOCOL["authorization"]
    assert auth["fit_development_only"] is True
    assert all(auth[key] is False for key in (
        "protected_reads", "cal_reads", "check_reads", "g10_reads", "t2rd_reads",
        "openvla_inference", "rollout", "training_authorized", "full_fit_authorized", "attack_authorized",
    ))
    assert PROTOCOL["input_consumption"]["old_fresh40_proxy_substitution"] == "FORBIDDEN"


def test_r3_teacher_contract_has_five_heads_and_tri_state():
    teacher = PROTOCOL["teacher"]
    assert len(teacher["heads"]) == 5
    assert teacher["truth_values"] == ["TRUE", "FALSE", "UNKNOWN"]
    assert teacher["unknown_mask"] is False
    assert teacher["unknown_reason_required"] is True
    assert teacher["future_frames_allowed"] is False
    assert len(teacher["forbidden_inputs"]) >= 5
    assert set(teacher["head_input_allowlist"]) == set(teacher["heads"])


def test_r3_contact_schema_is_complete():
    contact = PROTOCOL["teacher"]["contact_schema"]
    for key in (
        "contact_pairs", "contact_ncon_total", "contact_truncated", "contact_position",
        "contact_normal", "normal_constraint_force_scalar", "object_gripper_contact_binding",
        "forward_before_capture",
    ):
        assert key in contact
    assert contact["contact_truncated"] is False
    assert contact["forward_before_capture"] is True


def test_r3_split_and_threshold_are_frozen():
    split = PROTOCOL["split"]
    assert split["identity_overlap"] is False
    assert split["task_group_overlap"] is False
    assert split["seed"] == 20260717
    assert split["tranches"] == [8, 40, 80, 160, 320, 670]
    threshold = PROTOCOL["threshold"]
    assert threshold["engineering_threshold"] == 0.5
    assert threshold["selection_scope"] == "FIT_DEV only"
    assert threshold["selection_count"] == 1
    assert threshold["cal_check_g10_t2rd_selection"] is False


def test_r3_student_contract_is_causal_and_25d():
    student = PROTOCOL["student"]
    assert student["feature_schema"] == "SC5StreamingFeatureAdapterV2_25D"
    assert student["input_dim"] == 25
    assert student["future_frames"] == 0
    assert student["teacher_fields_in_input"] is False
    assert student["candidate_close_in_input"] is False
    assert student["dtype"] == "float32"


def test_r3_source_sha_fields_are_valid():
    for key in ("base_commit", "valid_r2_execution_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", PROTOCOL["lineage"][key])


def test_r3_forbidden_substitutions_are_explicit():
    forbidden = set(PROTOCOL["forbidden_substitutions"])
    assert "UNKNOWN mapped to FALSE" in forbidden
    assert "Fresh40 proxy telemetry for missing contact pairs" in forbidden
    assert "future frames or future labels" in forbidden


def test_fast_closure_amendment_freezes_current_coverage_gate():
    assert PROTOCOL["protocol_revision"] == "R3-FAST-CLOSURE-R1"
    assert PROTOCOL["minimum_coverage_for_student"]["per_head_positive_events"] == 20
    assert PROTOCOL["minimum_coverage_for_student"]["per_head_negative_events"] == 20
    assert PROTOCOL["input_consumption"]["canonical_source"].endswith("fixed_sealed_development_tranche")
    assert PROTOCOL["teacher"]["semantic_rules"]["k10"].endswith("safe_release_computed is FALSE")
