from __future__ import annotations

import json
from pathlib import Path

import pytest

from gripper_attack.v5_dataset import load_fit_registry, load_v5_episode
from gripper_attack.v5_physics import PHYSICS_TEACHER_FIELDS
from gripper_attack.v5_protocol import canonical_variant


def test_v5_registry_filters_complete_global_2000_to_fit_800(tmp_path: Path):
    path = tmp_path / "registry.csv"
    fields = ["canonical_parent_key", "suite", "task_idx", "state_id", "split"]
    rows = []
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        for task in range(10):
            for state in range(50):
                rows.append(f"{suite}/task_{task:02d}/state_{state:02d},{suite},{task},{state},FIT_TRAIN\n")
    path.write_text(",".join(fields) + "\n" + "".join(rows), encoding="utf-8")
    fit = load_fit_registry(path)
    assert len(fit) == 800
    assert max(row["state_id"] for row in fit) == 19


def test_v5_registry_rejects_incomplete_global_universe(tmp_path: Path):
    path = tmp_path / "registry.csv"
    path.write_text("canonical_parent_key,suite,task_idx,state_id\nlibero_object/task_00/state_00,libero_object,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="800"):
        load_fit_registry(path)


def test_v5_loader_adapts_sealed_physics_teacher_rows(tmp_path: Path):
    identity = "libero_object/task_00/state_00"
    row = {"canonical_parent_key": identity, "suite": "libero_object", "task_idx": 0, "state_id": 0}
    s1 = tmp_path / "s1" / "libero_object" / "task_00" / "state_00"
    teacher = tmp_path / "teacher" / "labels" / "libero_object" / "task_00" / "state_00"
    s1.mkdir(parents=True)
    teacher.mkdir(parents=True)
    student_rows = []
    physics_rows = []
    for step in range(12):
        student_rows.append({
            "schema": "STUDENT",
            "source_schema": "S1",
            "feature_order_sha256": "a" * 64,
            "suite": "libero_object",
            "task_idx": 0,
            "state_id": 0,
            "canonical_parent_key": identity,
            "step": step,
            "features_25d": [float(step)] * 25,
            "valid": True,
        })
        known = True
        tier = 2 if step < 10 else 1
        physics_rows.append({
            "step": step,
            "candidate_close": True,
            "student_valid": True,
            "gripper_contact_score": 1.0,
            "object_contact": True,
            "support_contact": False,
            "relative_pose_stability": 0.9,
            "object_eef_comotion_score": 0.9,
            "lift_score": 0.8,
            "target_progress": 0.5,
            "target_progress_known": True,
            "task_grasp_necessity": 1.0,
            "stable_grasp_score": 0.9,
            "stable_grasp_dwell": 10,
            "release_risk": 0.7 if tier == 1 else 0.1,
            "regrasp_or_instability_risk": 0.1,
            "support_removed": 1.0,
            "utility_score": 0.8 if tier == 2 else 0.3,
            "known_mask": known,
            "utility_tier": tier,
            "phase_name": "VALID_RETENTION" if tier == 2 else "RELEASE_IMMINENT_TAIL",
            "teacher_confidence": 1.0,
            "window_id": "candidate:0",
            "window_start": 0,
            "window_end": 11,
            "suite": "libero_object",
            "task_idx": 0,
            "manipulated_objects": ["obj"],
            "target_names": ["target"],
            "support_names": ["table"],
            "task_role_status": "PASS",
            "task_role_reason": "synthetic",
            "physics_teacher_proxy": True,
            "counterfactual_attack_label": False,
            "canonical_parent_key": identity,
            "state_id": 0,
            "source_artifact_recursive_sha256": "b" * 64,
            "physics_protocol_schema": "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V1",
        })
    (s1 / "student_input_records.jsonl").write_text("".join(json.dumps(item) + "\n" for item in student_rows), encoding="utf-8")
    (teacher / "physics_teacher_v2.jsonl").write_text("".join(json.dumps(item) + "\n" for item in physics_rows), encoding="utf-8")
    (tmp_path / "teacher" / "protocol.json").write_text(json.dumps({
        "schema": "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V1",
        "fixed_constants": {"tier2_max_release_risk": 0.6, "tier2_max_regrasp_risk": 0.6},
    }), encoding="utf-8")
    episode = load_v5_episode(tmp_path / "s1", tmp_path / "teacher", row)
    assert episode.canonical_parent_key == identity
    assert len(episode.windows) == 2
    assert episode.windows[0].utility_tier == 2
    assert episode.windows[1].utility_tier == 1
    assert bool(episode.release_imminent[-1])
    for item in physics_rows:
        item["causal_trigger_eligible"] = True
        item["component_valid_mask"] = {"target_progress": True}
        item["tier_onset_step"] = 0
    (teacher / "physics_teacher_v21.jsonl").write_text("".join(json.dumps(item) + "\n" for item in physics_rows), encoding="utf-8")
    (tmp_path / "teacher" / "protocol.json").write_text(json.dumps({
        "schema": "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21",
        "fixed_constants": {"tier2_max_release_risk": 0.6, "tier2_max_regrasp_risk": 0.6},
        "window_policy": {"loader_preserve_candidate_segment": True},
    }), encoding="utf-8")
    v21_episode = load_v5_episode(tmp_path / "s1", tmp_path / "teacher", row)
    assert len(v21_episode.windows) == 1
    assert v21_episode.windows[0].utility_tier == 2
    assert canonical_variant("V5_A_PHYSICS") == "V5_A_PROPRIO"
