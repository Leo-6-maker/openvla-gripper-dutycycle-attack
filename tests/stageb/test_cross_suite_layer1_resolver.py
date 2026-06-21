import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.cross_suite_layer1_resolver import (  # noqa: E402
    BindingResult,
    RESOLVER_NOT_IMPLEMENTED,
    SUPPLEMENTARY_EVENT_ELIGIBLE,
    bind_unique,
    bind_target,
    build_blind_review_manifest,
    build_dev_canary_manifest,
    build_review_package,
    detect_physical_event,
    load_ontology,
    load_step_rows,
    resolve_episode,
    run_resolver,
    teacher_timeline_rows,
    validate_episode_rows,
)


ONTOLOGY = REPO / "configs" / "cross_suite_task_ontology_v1.yaml"


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _trajectory(kind: str, *, object_count: int = 1, ambiguous_target: bool = False, missing_gripper_site: bool = False):
    body_names = ["world"]
    if object_count == 1:
        body_names.append("black_bowl_1_main")
    else:
        body_names.extend(["alphabet_soup_1_main", "tomato_sauce_1_main"])
    body_names.append("plate_1_main")
    site_names = ["plate_1_default_site", "gripper0_grip_site"]
    if missing_gripper_site:
        site_names = ["plate_1_default_site", "not_the_gripper_site"]
    if ambiguous_target:
        site_names.insert(1, "plate_2_default_site")

    n = 8
    body_xpos = np.zeros((n, len(body_names), 3), dtype=np.float32)
    site_xpos = np.zeros((n, len(site_names), 3), dtype=np.float32)
    body_xquat = np.zeros((n, len(body_names), 4), dtype=np.float32)
    qpos = np.zeros((n, 2), dtype=np.float32)
    qvel = np.zeros((n, 2), dtype=np.float32)
    ctrl = np.zeros((n, 2), dtype=np.float32)

    target_site_idx = 0
    grip_site_idx = 1
    site_xpos[:, target_site_idx, :] = np.array([1.0, 0.0, 0.05], dtype=np.float32)
    if ambiguous_target:
        site_xpos[:, 1, :] = np.array([1.1, 0.0, 0.05], dtype=np.float32)
    body_xpos[:, -1, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    valid_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.01],
            [0.30, 0.0, 0.04],
            [0.60, 0.0, 0.05],
            [0.95, 0.0, 0.05],
            [0.97, 0.0, 0.05],
            [0.98, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    if kind == "no_grasp":
        obj = valid_obj.copy()
        grip = obj + np.array([0.5, 0.0, 0.0], dtype=np.float32)
    elif kind == "no_lift":
        obj = valid_obj.copy()
        obj[:, 2] = 0.0
        grip = obj.copy()
    elif kind == "release_far":
        obj = valid_obj.copy()
        obj[:, 0] = np.linspace(0.0, 0.3, n, dtype=np.float32)
        grip = obj.copy()
    elif kind == "late_event":
        obj = valid_obj.copy()
        obj[:3] = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        obj[3:, 2] = np.array([0.0, 0.0, 0.01, 0.04, 0.05], dtype=np.float32)
        grip = obj.copy()
        grip[:3] = obj[:3] + np.array([0.5, 0.0, 0.0], dtype=np.float32)
    else:
        obj = valid_obj
        grip = valid_obj.copy()

    if object_count == 1:
        body_xpos[:, 1, :] = obj
    else:
        body_xpos[:, 1, :] = obj
        body_xpos[:, 2, :] = obj + np.array([0.0, 0.1, 0.0], dtype=np.float32)
    site_xpos[:, grip_site_idx, :] = grip
    return body_names, site_names, body_xpos, body_xquat, site_xpos, qpos, qvel, ctrl


def _episode(
    tmp_path: Path,
    name: str,
    *,
    suite="libero_spatial",
    task_idx=0,
    state_id=0,
    trajectory="valid",
    object_count=1,
    ambiguous_object=False,
    ambiguous_target=False,
    missing_gripper_site=False,
    false_close_first=False,
):
    ep = tmp_path / name
    body_names, site_names, body_xpos, body_xquat, site_xpos, qpos, qvel, ctrl = _trajectory(
        trajectory,
        object_count=object_count,
        ambiguous_target=ambiguous_target,
        missing_gripper_site=missing_gripper_site,
    )
    if ambiguous_object:
        body_names = ["world", "black_bowl_1_main", "black_bowl_2_main", "plate_1_main"]
        body_xpos = np.concatenate([body_xpos[:, :2, :], body_xpos[:, 1:2, :] + 0.05, body_xpos[:, -1:, :]], axis=1)
        body_xquat = np.zeros((len(body_xpos), len(body_names), 4), dtype=np.float32)

    _write_json(
        ep / "episode_manifest.json",
        {
            "condition": "CLEAN",
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "eval_seed": 0,
        },
    )
    _write_json(
        ep / "episode_summary.json",
        {
            "condition": "CLEAN",
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "eval_seed": 0,
            "n_steps": len(qpos),
            "task_success": True,
        },
    )
    _write_json(
        ep / "sim_state_manifest.json",
        {
            "arrays": {
                "qpos": list(qpos.shape),
                "qvel": list(qvel.shape),
                "body_xpos": list(body_xpos.shape),
                "body_xquat": list(body_xquat.shape),
                "site_xpos": list(site_xpos.shape),
                "ctrl": list(ctrl.shape),
            },
            "metadata": {
                "body_names": body_names,
                "site_names": site_names,
                "joint_names": ["black_bowl_1_joint0"],
            },
        },
    )
    rows = []
    for step in range(len(qpos)):
        close = step in {2, 3, 4, 5, 6}
        if false_close_first:
            close = step in {1, 3, 4, 5, 6, 7}
        rows.append(
            {
                "step": step,
                "raw_gripper": 0.0 if close else 1.0,
                "env_gripper": 1.0 if close else -1.0,
                "gripper_qpos": 0.01 if close else 0.03,
                "gripper_opening_proxy": 0.01 if close else 0.03,
                "eef_x": 0,
                "eef_y": 0,
                "eef_z": 0,
                "mlp_emit": 99,
                "mlp_triggered": True,
                "corridor_p": 0.9,
                "release_p": 0.1,
                "pred_phase": "release",
            }
        )
    _write_csv(ep / "step_telemetry.csv", rows)
    (ep / "rollout_raw.mp4").write_bytes(b"video")
    np.savez(
        ep / "sim_state_stream.npz",
        qpos=qpos,
        qvel=qvel,
        body_xpos=body_xpos,
        body_xquat=body_xquat,
        site_xpos=site_xpos,
        ctrl=ctrl,
    )
    return ep


def _ledger_row(ep: Path, *, suite="libero_spatial", task_idx=0, state_id=0, success="True", sha="abc"):
    return {
        "canonical_key": f"{suite}|{task_idx}|{state_id}|0|CLEAN",
        "episode_path": str(ep),
        "suite": suite,
        "task_idx": str(task_idx),
        "state_id": str(state_id),
        "eval_seed": "0",
        "condition": "CLEAN",
        "status": "COMPLETE_VALID",
        "task_success": success,
        "n_steps": "8",
        "artifact_recursive_sha256": sha,
    }


def test_load_step_rows_drops_forbidden_detector_fields(tmp_path):
    ep = _episode(tmp_path, "ep")
    rows = load_step_rows(ep / "step_telemetry.csv")
    assert rows
    assert "raw_gripper" in rows[0]
    for forbidden in ["mlp_emit", "mlp_triggered", "corridor_p", "release_p", "pred_phase"]:
        assert forbidden not in rows[0]


def test_structured_prefix_suffix_binding_without_arbitrary_substring():
    bound = bind_unique(["akita_black_bowl_1_main"], ("black_bowl",))
    assert bound.status == "BOUND_STRUCTURED_FALLBACK"
    assert bound.name == "akita_black_bowl_1_main"
    ambiguous = bind_unique(["akita_black_bowl_1_main", "akita_black_bowl_2_main"], ("black_bowl",))
    assert ambiguous.status == "AMBIGUOUS"
    failed = bind_unique(["black_marker_1_main"], ("mug",))
    assert failed.status == "FAILED"


def test_region_specific_target_does_not_fallback_to_generic_default_site():
    generic_cabinet_site = ["wooden_cabinet_1_default_site"]
    generic_cabinet_body = ["wooden_cabinet_1_main"]
    binding, kind = bind_target(generic_cabinet_site, generic_cabinet_body, ("cabinet_top", "cabinet"))
    assert binding.status == "FAILED"
    assert binding.source == "region_specific_target_missing:cabinet_top"
    assert kind == "site"

    generic_caddy_site = ["caddy_1_default_site"]
    generic_caddy_body = ["caddy_1_main"]
    caddy, caddy_kind = bind_target(generic_caddy_site, generic_caddy_body, ("back_compartment_of_caddy", "caddy"))
    assert caddy.status == "FAILED"
    assert caddy.source == "region_specific_target_missing:back_compartment_of_caddy"
    assert caddy_kind == "site"


def test_single_object_episode_requires_physical_grasp_lift_carry_target(tmp_path):
    ep = _episode(tmp_path, "ep", trajectory="valid")
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    episode, events = resolve_episode(_ledger_row(ep), task, teacher_run_id="dev")
    assert episode["teacher_status"] == "ELIGIBLE_EVENT"
    assert episode["manual_review_required"] is True
    assert episode["object_binding_status"] in {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY", "BOUND_STRUCTURED_FALLBACK"}
    assert episode["target_binding_status"] in {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY", "BOUND_STRUCTURED_FALLBACK"}
    assert len(events) == 1
    assert events[0]["close_onset_step"] == 2
    assert events[0]["grasp_established_step"] == 3
    assert events[0]["lift_onset_step"] == 4
    assert events[0]["stable_carry_start"] == 4
    assert events[0]["target_proximity_step"] == 5
    assert events[0]["event_valid"] is True


def test_close_without_grasp_or_lift_is_not_eligible(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    no_grasp_ep = _episode(tmp_path, "no_grasp", trajectory="no_grasp")
    no_lift_ep = _episode(tmp_path, "no_lift", trajectory="no_lift", state_id=1)
    no_grasp, no_grasp_events = resolve_episode(_ledger_row(no_grasp_ep), task, teacher_run_id="dev")
    no_lift, no_lift_events = resolve_episode(_ledger_row(no_lift_ep, state_id=1), task, teacher_run_id="dev")
    assert no_grasp["teacher_status"] == "NO_RELEVANT_GRASP_EVENT"
    assert "no_grasp_proximity" in no_grasp["abstain_reason"]
    assert no_grasp_events == []
    assert no_lift["teacher_status"] == "NO_RELEVANT_GRASP_EVENT"
    assert "no_object_lift" in no_lift["abstain_reason"]
    assert no_lift_events == []


def test_far_from_target_carry_is_valid_event_but_not_placement_complete(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    ep = _episode(tmp_path, "far_release", trajectory="release_far")
    episode, events = resolve_episode(_ledger_row(ep), task, teacher_run_id="dev")
    assert episode["teacher_status"] == "ELIGIBLE_EVENT"
    assert len(events) == 1
    assert events[0]["target_proximity_step"] == ""
    assert events[0]["placement_complete"] is False


def test_physics_object_gripper_near_threshold_changes_event_acceptance(tmp_path):
    body_names, site_names, body_xpos, body_xquat, site_xpos, qpos, qvel, ctrl = _trajectory("valid")
    site_xpos[:, 1, :] = body_xpos[:, 1, :] + np.array([0.05, 0.0, 0.0], dtype=np.float32)
    step_rows = []
    for step in range(len(body_xpos)):
        close = step in {2, 3, 4, 5, 6}
        step_rows.append(
            {
                "step": step,
                "raw_gripper": 0.0 if close else 1.0,
                "env_gripper": 1.0 if close else -1.0,
            }
        )
    sim_arrays = {"body_xpos": body_xpos, "site_xpos": site_xpos}
    object_binding = BindingResult("black_bowl_1_main", 1, "BOUND_EXACT", "test", ("black_bowl_1_main",))
    target_binding = BindingResult("plate_1_default_site", 0, "BOUND_EXACT", "test", ("plate_1_default_site",))
    default_physics = {
        "thresholds": {
            "object_gripper_near_m": 0.12,
            "object_gripper_separated_m": 0.18,
            "object_lift_delta_m": 0.025,
            "stable_carry_min_frames": 3,
            "grasp_min_frames": 2,
            "object_target_near_m": 0.14,
            "max_close_to_grasp_delay_frames": 1,
            "motion_coupling_max_delta_m": 0.06,
            "orientation_jump_max": 0.25,
        }
    }
    strict_physics = json.loads(json.dumps(default_physics))
    strict_physics["thresholds"]["object_gripper_near_m"] = 0.001

    accepted = detect_physical_event(
        step_rows=step_rows,
        sim_arrays=sim_arrays,
        site_names=site_names,
        object_binding=object_binding,
        target_binding=target_binding,
        target_kind="site",
        physics=default_physics,
    )
    rejected = detect_physical_event(
        step_rows=step_rows,
        sim_arrays=sim_arrays,
        site_names=site_names,
        object_binding=object_binding,
        target_binding=target_binding,
        target_kind="site",
        physics=strict_physics,
    )

    assert accepted.status == "PHYSICAL_EVENT_VALID"
    assert rejected.status == "PHYSICAL_EVENT_INCOMPLETE"
    assert "no_grasp_proximity_after_close" in rejected.event_invalid_reason


def test_late_valid_close_candidate_is_selected(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    ep = _episode(tmp_path, "late", trajectory="late_event", false_close_first=True)
    episode, events = resolve_episode(_ledger_row(ep), task, teacher_run_id="dev")
    assert episode["teacher_status"] == "ELIGIBLE_EVENT"
    assert events[0]["close_onset_step"] == 3


def test_close_attempt_reopen_prevents_merging_failed_grasp_with_later_collision(tmp_path):
    body_names, site_names, body_xpos, body_xquat, site_xpos, qpos, qvel, ctrl = _trajectory("valid")
    n = len(body_xpos)
    # First close has proximity but no lift; gripper reopens. Later close only
    # collides with/pushes the object while the gripper stays far, so the two
    # phases must not be merged into one false carry event.
    body_xpos[:, 1, :] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.03],
            [0.5, 0.0, 0.06],
            [0.6, 0.0, 0.02],
            [0.6, 0.0, 0.00],
        ],
        dtype=np.float32,
    )
    site_xpos[:, 1, :] = body_xpos[:, 1, :] + np.array([0.5, 0.0, 0.0], dtype=np.float32)
    site_xpos[2:4, 1, :] = body_xpos[2:4, 1, :]
    step_rows = []
    for step in range(n):
        close = step in {2, 4, 5, 6}
        step_rows.append({"step": step, "raw_gripper": 0.0 if close else 1.0, "env_gripper": 1.0 if close else -1.0})
    event = detect_physical_event(
        step_rows=step_rows,
        sim_arrays={"body_xpos": body_xpos, "body_xquat": body_xquat, "site_xpos": site_xpos},
        site_names=site_names,
        object_binding=BindingResult("black_bowl_1_main", 1, "BOUND_EXACT", "test", ("black_bowl_1_main",)),
        target_binding=BindingResult("plate_1_default_site", 0, "BOUND_EXACT", "test", ("plate_1_default_site",)),
        target_kind="site",
    )
    assert event.status == "PHYSICAL_EVENT_INCOMPLETE"
    assert event.event_valid is False
    assert "no_stable_carry" in event.event_invalid_reason or "no_grasp_proximity_after_close" in event.event_invalid_reason


def test_owner_phase_order_regression_step86_grasp_not_lift():
    n = 96
    body_xpos = np.zeros((n, 2, 3), dtype=np.float32)
    site_xpos = np.zeros((n, 2, 3), dtype=np.float32)
    body_xquat = np.zeros((n, 2, 4), dtype=np.float32)
    body_xquat[:, :, 0] = 1.0
    site_names = ["plate_1_default_site", "gripper0_grip_site"]
    body_xpos[:, 1, :] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    site_xpos[:, 1, :] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    site_xpos[86:, 1, :] = body_xpos[86:, 1, :]
    # Lift starts after the reviewed stable-grasp frame.
    body_xpos[90:, 1, 2] = 0.04
    site_xpos[90:, 1, 2] = 0.04
    step_rows = [{"step": step, "raw_gripper": 0.0 if step >= 48 else 1.0, "env_gripper": 1.0 if step >= 48 else -1.0} for step in range(n)]
    event = detect_physical_event(
        step_rows=step_rows,
        sim_arrays={"body_xpos": body_xpos, "body_xquat": body_xquat, "site_xpos": site_xpos},
        site_names=site_names,
        object_binding=BindingResult("black_bowl_1_main", 1, "BOUND_EXACT", "test", ("black_bowl_1_main",)),
        target_binding=BindingResult("plate_1_default_site", 0, "BOUND_EXACT", "test", ("plate_1_default_site",)),
        target_kind="site",
    )
    assert event.grasp_established_step == 86
    assert event.lift_onset_step != 86
    assert int(event.lift_onset_step) > 86


def test_window_end_preserves_minimum_stable_carry_duration():
    body_names, site_names, body_xpos, body_xquat, site_xpos, qpos, qvel, ctrl = _trajectory("valid")
    n = len(body_xpos)
    # Target proximity begins at the same frame as carry evidence. The Teacher
    # window must not be truncated to the carry start before minimum carry duration.
    body_xpos[:, 1, :] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.01],
            [0.06, 0.0, 0.05],
            [0.06, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    site_xpos[:, 0, :] = body_xpos[:, 1, :]
    site_xpos[:, 1, :] = body_xpos[:, 1, :]
    rows = []
    for step in range(n):
        rows.append(
            {
                "step": step,
                "raw_gripper": 0.0 if step >= 2 else 1.0,
                "env_gripper": 1.0 if step >= 2 else -1.0,
            }
        )
    event = detect_physical_event(
        step_rows=rows,
        sim_arrays={"body_xpos": body_xpos, "body_xquat": body_xquat, "site_xpos": site_xpos},
        site_names=site_names,
        object_binding=BindingResult("black_bowl_1_main", 1, "BOUND_EXACT", "test", ("black_bowl_1_main",)),
        target_binding=BindingResult("plate_1_default_site", 0, "BOUND_EXACT", "test", ("plate_1_default_site",)),
        target_kind="site",
    )
    assert event.event_valid is True
    assert int(event.teacher_window_end) > int(event.stable_carry_start)


def test_missing_gripper_site_fails_closed(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    ep = _episode(tmp_path, "missing_grip", missing_gripper_site=True)
    episode, events = resolve_episode(_ledger_row(ep), task, teacher_run_id="dev")
    assert episode["teacher_status"] == "NO_RELEVANT_GRASP_EVENT"
    assert "missing_gripper0_grip_site" in episode["abstain_reason"]
    assert events == []


def test_ambiguous_object_or_target_fails_closed(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    amb_obj_ep = _episode(tmp_path, "amb_obj", ambiguous_object=True)
    amb_target_ep = _episode(tmp_path, "amb_target", ambiguous_target=True, state_id=1)
    amb_obj, _ = resolve_episode(_ledger_row(amb_obj_ep), task, teacher_run_id="dev")
    amb_target, _ = resolve_episode(_ledger_row(amb_target_ep, state_id=1), task, teacher_run_id="dev")
    assert amb_obj["teacher_status"] == "OBJECT_BINDING_AMBIGUOUS"
    assert amb_obj["object_binding_status"] == "AMBIGUOUS"
    assert amb_target["teacher_status"] == "TARGET_BINDING_AMBIGUOUS"
    assert amb_target["target_binding_status"] == "AMBIGUOUS"


def test_negative_and_supplementary_status_invariants(tmp_path):
    ep_goal = _episode(tmp_path, "goal", suite="libero_goal", task_idx=0)
    ep_l10 = _episode(tmp_path, "l10", suite="libero_10", task_idx=0, object_count=2)
    ontology = load_ontology(ONTOLOGY)
    neg, _ = resolve_episode(_ledger_row(ep_goal, suite="libero_goal", task_idx=0), ontology[("libero_goal", 0)], teacher_run_id="dev")
    multi, events = resolve_episode(_ledger_row(ep_l10, suite="libero_10", task_idx=0), ontology[("libero_10", 0)], teacher_run_id="dev")
    assert neg["teacher_status"] == "CORRECT_SEMANTIC_ABSTAIN"
    assert multi["teacher_status"] == SUPPLEMENTARY_EVENT_ELIGIBLE
    assert multi["mechanism_eligible"] is False
    assert multi["teacher_semantic_abstain"] is True
    assert multi["label_role"] == "supplementary_multievent_grasp_carry_bridge"
    assert multi["primary_or_supplementary"] == "supplementary"
    assert multi["primary_supplementary_event_id"]
    assert all(event["supplementary_event"] for event in events)
    assert validate_episode_rows([neg, multi]) == []


def test_mixed_and_multi_event_regression_classes(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    expected = {
        ("libero_goal", 3): "mixed_articulated_pick_place",
        ("libero_10", 2): "mixed_articulated_pick_place",
        ("libero_10", 3): "mixed_articulated_pick_place",
        ("libero_10", 9): "mixed_articulated_pick_place",
        ("libero_10", 4): "multi_object_transfer",
        ("libero_10", 6): "multi_object_transfer",
        ("libero_10", 8): "multi_object_transfer",
    }
    for key, mechanism in expected.items():
        assert ontology[key].mechanism_type == mechanism
        ep = _episode(tmp_path, f"{key[0]}_{key[1]}", suite=key[0], task_idx=key[1])
        row, events = resolve_episode(_ledger_row(ep, suite=key[0], task_idx=key[1]), ontology[key], teacher_run_id="dev")
        assert row["mechanism_eligible"] is False
        if row["teacher_status"] == SUPPLEMENTARY_EVENT_ELIGIBLE:
            assert row["manual_review_required"] is True
            assert row["label_role"] == "supplementary_multievent_grasp_carry_bridge"
            assert row["primary_or_supplementary"] == "supplementary"
            assert row["primary_supplementary_event_id"]
            assert all(event["supplementary_event"] for event in events)
        else:
            assert row["teacher_status"] == "NO_RELEVANT_GRASP_EVENT"
            assert row["manual_review_required"] is False


def test_supplementary_event_does_not_require_target_binding(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    ep = _episode(tmp_path, "supp_target_missing", suite="libero_10", task_idx=0, object_count=2)
    row, events = resolve_episode(_ledger_row(ep, suite="libero_10", task_idx=0), ontology[("libero_10", 0)], teacher_run_id="dev")
    assert row["teacher_status"] == SUPPLEMENTARY_EVENT_ELIGIBLE
    assert row["target_binding_status"] == "NOT_REQUIRED_FOR_SUPPLEMENTARY"
    assert row["mechanism_eligible"] is False
    assert row["label_role"] == "supplementary_multievent_grasp_carry_bridge"
    assert row["primary_supplementary_event_id"]
    assert len(events) >= 1
    assert validate_episode_rows([row]) == []


def test_supplementary_primary_event_selection_is_deterministic(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    ep = _episode(tmp_path, "supp_multi", suite="libero_10", task_idx=0, object_count=2)
    row, events = resolve_episode(_ledger_row(ep, suite="libero_10", task_idx=0), ontology[("libero_10", 0)], teacher_run_id="dev")
    assert row["teacher_status"] == SUPPLEMENTARY_EVENT_ELIGIBLE
    assert len(events) == 2
    selected = sorted(
        events,
        key=lambda e: (
            int(e["stable_carry_start"]),
            int(e["close_onset_step"]),
            e["object_body_name"],
            e["event_id"],
        ),
    )[0]
    assert row["primary_supplementary_event_id"] == selected["event_id"]


def test_supplementary_no_grasp_carry_is_negative_only(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    ep = _episode(tmp_path, "supp_no_grasp", suite="libero_10", task_idx=0, object_count=2, trajectory="no_grasp")
    row, events = resolve_episode(_ledger_row(ep, suite="libero_10", task_idx=0), ontology[("libero_10", 0)], teacher_run_id="dev")
    assert row["teacher_status"] == "NO_RELEVANT_GRASP_EVENT"
    assert row["label_role"] == "negative_only"
    assert row["primary_or_supplementary"] == "negative"
    assert row["event_count"] == 0
    assert events == []


def test_dev_and_blind_manifests_are_deterministic_and_disjoint(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    rows = []
    for suite, task_idx in [
        ("libero_spatial", 0),
        ("libero_spatial", 1),
        ("libero_spatial", 2),
        ("libero_spatial", 3),
        ("libero_goal", 1),
        ("libero_goal", 2),
        ("libero_goal", 0),
        ("libero_goal", 5),
        ("libero_10", 5),
        ("libero_10", 5),
        ("libero_10", 4),
        ("libero_10", 6),
    ]:
        idx = len(rows)
        ep = _episode(tmp_path, f"ep{idx}", suite=suite, task_idx=task_idx, state_id=idx)
        rows.append(_ledger_row(ep, suite=suite, task_idx=task_idx, state_id=idx, sha=f"sha{idx}"))
    dev = build_dev_canary_manifest(rows, ontology)
    blind = build_blind_review_manifest(rows, ontology, exclude_keys={r["canonical_key"] for r in dev["selected"]}, count=4)
    assert dev["selected_count"] == 12
    assert {r["canonical_key"] for r in dev["selected"]}.isdisjoint({r["canonical_key"] for r in blind["selected"]})


def test_run_resolver_and_blind_package_outputs_are_event_level_and_blind(tmp_path):
    ep = _episode(tmp_path, "ep")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "selected": [_ledger_row(ep)],
        },
    )
    out = tmp_path / "resolver"
    result = run_resolver(manifest, ONTOLOGY, out, teacher_run_id="dev")
    assert result["episode_count"] == 1
    assert result["event_count"] == 1
    package = build_review_package(manifest, out, tmp_path / "review")
    assert package["review_count"] == 1
    review_csv = (tmp_path / "review" / "blind_review_queue.csv").read_text(encoding="utf-8")
    assert "reviewer_id" in review_csv
    assert "proposed_grasp_established" in review_csv
    assert "teacher_only_timeline_path" in review_csv
    assert "teacher_only_overlay_status" in review_csv
    assert "task_success" not in review_csv
    assert "mlp_emit" not in review_csv
    timeline_paths = sorted((tmp_path / "review" / "teacher_timelines").glob("*.csv"))
    assert len(timeline_paths) == 1
    timeline_csv = timeline_paths[0].read_text(encoding="utf-8")
    assert "close_onset" in timeline_csv
    assert "mlp_emit" not in timeline_csv
    hidden = (tmp_path / "review" / "blind_review_hidden_audit_manifest.csv").read_text(encoding="utf-8")
    assert "task_success" in hidden


def test_teacher_timeline_rows_are_teacher_only():
    label = {"episode_key": "ep", "teacher_status": "ELIGIBLE_EVENT"}
    event = {
        "event_id": "ep|event0",
        "close_onset_step": "7",
        "teacher_window_start": "5",
        "teacher_window_end": "12",
        "object_body_name": "object_main",
        "target_body_or_site_name": "target_site",
    }
    rows = teacher_timeline_rows(label, event)
    assert {r["marker"] for r in rows} >= {"window_start", "close_onset", "window_end"}
    assert all("mlp_emit" not in r for r in rows)
