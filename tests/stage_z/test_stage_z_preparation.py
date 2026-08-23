from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage_z_preparation.adapters import M0Adapter, OFTAdapter, OFTQueueState, Pi05Adapter, Pi05ReplanState
from stage_z_preparation.analysis import parent_bootstrap_mean, summarize_synthetic_rows
from stage_z_preparation.anchors import AnchorCandidate, select_anchor
from stage_z_preparation.contract import (
    ExecutionAuthorization,
    ProtectedCounters,
    StageZHold,
    Z0R2_PASS,
    intervene_gripper_open,
    require_execution_authorized,
)
from stage_z_preparation.matrix import StageZArm, action_for_arm, prepare_five_arm_matrix
from stage_z_preparation.panel import FrozenPanel, STRUCTURAL_MISSING, SUITE_COUNTS
from stage_z_preparation.runner import run_authorized_callback
from stage_z_preparation.runner import require_engineering_canary
from stage_z_preparation.telemetry import make_telemetry_record, validate_synthetic_row


ACTION = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7)


def _authorization(**overrides) -> ExecutionAuthorization:
    values = dict(
        execution_enabled=True,
        z0r2_status=Z0R2_PASS,
        root_seal_sha256="root",
        expected_root_seal_sha256="root",
        model_authority_sha256="model",
        expected_model_authority_sha256="model",
        common_libero_sha256="libero",
        expected_common_libero_sha256="libero",
        panel_sha256="panel",
        expected_panel_sha256="panel",
        frozen_parent_keys=frozenset({"libero_10/task_00/state_33"}),
        phase="Z1",
        authorized_phases=frozenset({"Z1"}),
        counters=ProtectedCounters(),
    )
    values.update(overrides)
    return ExecutionAuthorization(**values)


def _anchors(model_id: str = "Z-M0_OPENVLA"):
    return [
        AnchorCandidate("libero_10/task_00/state_33", model_id, 4, "CRITICAL"),
        AnchorCandidate("libero_10/task_00/state_33", model_id, 8, "CRITICAL"),
        AnchorCandidate("libero_10/task_00/state_33", model_id, 2, "NONCRITICAL"),
    ]


def test_final_action_requires_exactly_seven_dimensions() -> None:
    with pytest.raises(StageZHold, match="DIMENSION"):
        intervene_gripper_open((0.0,) * 6, duration=3)


def test_open_intervention_changes_only_final_coordinate() -> None:
    result = intervene_gripper_open(ACTION, duration=5)
    assert result[:6] == ACTION[:6]
    assert result[6] == -1.0


def test_open_value_is_frozen_and_not_inferred_from_model_family() -> None:
    result = intervene_gripper_open(ACTION, duration=3)
    assert result[6] == -1.0
    assert M0Adapter("libero_10", "m0-10").expose_final_action(ACTION).values == ACTION
    assert OFTAdapter("libero_10", "m1-10").expose_final_action(ACTION, OFTQueueState()).values == ACTION


def test_m0_is_suite_matched_and_m1_residual_queue_is_rejected() -> None:
    assert M0Adapter("libero_object", "m0-object").suite == "libero_object"
    with pytest.raises(StageZHold, match="RESIDUAL_QUEUE"):
        OFTAdapter("libero_10", "m1-10").expose_final_action(ACTION, OFTQueueState(residual_actions=1))


def test_m2_requires_fresh_replan_without_residual_chunk() -> None:
    adapter = Pi05Adapter("m2-pi05")
    with pytest.raises(StageZHold, match="FRESH_REPLAN"):
        adapter.expose_final_action(ACTION, Pi05ReplanState(steps_since_replan=1))
    with pytest.raises(StageZHold, match="FRESH_REPLAN"):
        adapter.expose_final_action(ACTION, Pi05ReplanState(residual_actions=1))


def test_frozen_panel_has_36_parents_and_preserves_structural_missingness() -> None:
    report_path = Path(__file__).parents[2] / "reports" / "STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
    panel = FrozenPanel.from_record(json.loads(report_path.read_text(encoding="utf-8")))
    assert len(panel.selected_parent_keys) == 36
    assert panel.structural_missing_cells == STRUCTURAL_MISSING
    assert {suite: sum(key.startswith(f"{suite}/") for key in panel.selected_parent_keys) for suite in SUITE_COUNTS} == SUITE_COUNTS
    with pytest.raises(StageZHold, match="STRUCTURAL_MISSING"):
        panel.require_scientific_parent("libero_goal/task_01")


def test_anchor_selection_is_deterministic_and_abstains_without_replacement() -> None:
    selected_a = select_anchor(_anchors(), salt="critical-v1", model_id="Z-M0_OPENVLA", parent_key="libero_10/task_00/state_33", anchor_class="CRITICAL")
    selected_b = select_anchor(_anchors(), salt="critical-v1", model_id="Z-M0_OPENVLA", parent_key="libero_10/task_00/state_33", anchor_class="CRITICAL")
    assert selected_a.selected == selected_b.selected
    empty = select_anchor(_anchors(), salt="critical-v1", model_id="Z-M1_OPENVLA_OFT", parent_key="libero_10/task_00/state_33", anchor_class="CRITICAL")
    assert empty.status == "NO_CRITICAL_ANCHOR"


def test_anchor_rejects_student_or_outcome_leakage() -> None:
    leaked = AnchorCandidate(
        "libero_10/task_00/state_33",
        "Z-M0_OPENVLA",
        4,
        "CRITICAL",
        metadata={"detector_score": 0.9},
    )
    with pytest.raises(StageZHold, match="LEAKAGE"):
        select_anchor([leaked], salt="x", model_id="Z-M0_OPENVLA", parent_key=leaked.parent_key, anchor_class="CRITICAL")


def test_five_arm_matrix_is_exact_and_action_has_gripper_only_change() -> None:
    critical = select_anchor(_anchors(), salt="critical", model_id="Z-M0_OPENVLA", parent_key="libero_10/task_00/state_33", anchor_class="CRITICAL")
    noncritical = select_anchor(_anchors(), salt="noncritical", model_id="Z-M0_OPENVLA", parent_key="libero_10/task_00/state_33", anchor_class="NONCRITICAL")
    matrix = prepare_five_arm_matrix(critical_anchor=critical, noncritical_anchor=noncritical)
    assert matrix.status == "READY_FIVE_ARMS"
    assert len(matrix.arms) == 5
    assert [arm.arm for arm in matrix.arms] == list(StageZArm)
    result = action_for_arm(ACTION, StageZArm.COMMAND_OPEN_T10_CRITICAL)
    assert result[:6] == ACTION[:6]
    assert result[6] == -1.0


def test_default_execution_is_disabled_and_callback_is_not_called() -> None:
    called = []
    with pytest.raises(StageZHold, match="EXECUTION_DISABLED"):
        run_authorized_callback(
            authorization=ExecutionAuthorization(),
            parent_key="libero_10/task_00/state_33",
            phase="Z1",
            callback=lambda: called.append(True),
        )
    assert called == []


def test_engineering_canary_is_separate_from_scientific_panel() -> None:
    require_engineering_canary("libero_10/task_00/state_49", {"libero_10/task_00/state_33"})
    with pytest.raises(StageZHold, match="OVERLAPS"):
        require_engineering_canary("libero_10/task_00/state_33", {"libero_10/task_00/state_33"})


def test_execution_requires_z0r2_root_and_all_authority_digests() -> None:
    with pytest.raises(StageZHold, match="Z0R2"):
        require_execution_authorized(_authorization(z0r2_status="HOLD"), parent_key="libero_10/task_00/state_33", phase="Z1")
    with pytest.raises(StageZHold, match="ROOT_SEAL"):
        require_execution_authorized(_authorization(root_seal_sha256=None), parent_key="libero_10/task_00/state_33", phase="Z1")


def test_protected_and_eval160_counters_fail_closed() -> None:
    with pytest.raises(StageZHold, match="COUNTERS"):
        require_execution_authorized(
            _authorization(counters=ProtectedCounters(eval160_reads=1)),
            parent_key="libero_10/task_00/state_33",
            phase="Z1",
        )


def test_telemetry_schema_is_predeclared_and_synthetic_rows_are_labeled() -> None:
    record = make_telemetry_record(
        model_id="Z-M0_OPENVLA",
        parent_key="libero_10/task_00/state_33",
        arm=StageZArm.COMMAND_OPEN_T5_CRITICAL,
        requested_open_duration=5,
    )
    with pytest.raises(StageZHold, match="SYNTHETIC"):
        validate_synthetic_row(record)
    record["evidence_status"] = "TEST_ONLY_NON_SCIENTIFIC"
    record["model_inference"] = 0
    record["env_step"] = 0
    record["protected_reads"] = 0
    validate_synthetic_row(record)
    summary = summarize_synthetic_rows([dict(record, suite="libero_10")])
    assert summary["status"] == "TEST_ONLY_NON_SCIENTIFIC"
    assert summary["complete_case_substitution"] is False


def test_parent_bootstrap_is_deterministic_and_parent_unit() -> None:
    first = parent_bootstrap_mean([0.0, 1.0, 1.0], seed=17, replicates=100)
    second = parent_bootstrap_mean([0.0, 1.0, 1.0], seed=17, replicates=100)
    assert first == second
    assert first[0] == pytest.approx(2 / 3)
