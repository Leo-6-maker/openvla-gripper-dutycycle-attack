from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ONTOLOGY = REPO / "configs" / "cross_suite_task_ontology_v1.yaml"
ALLOWED = {
    "single_object_pick_place",
    "multi_object_transfer",
    "mixed_articulated_pick_place",
    "articulated_only",
    "push_or_planar",
    "rearrangement_non_grasp",
    "unknown_or_low_signal",
}
REQUIRED = {
    "suite",
    "task_idx",
    "task_name",
    "instruction",
    "bddl_file_pattern",
    "mechanism_type",
    "teacher_applicable",
    "expected_event_count_class",
    "manipulated_object_aliases",
    "target_aliases",
    "articulated_body_aliases",
    "binding_source_priority",
    "positive_eligibility_rule",
    "expected_abstain_reason",
    "multi_event_policy",
    "notes",
}
FORBIDDEN = {"mlp_emit_step", "mlp_triggered", "corridor_p", "release_p", "pred_phase"}


def load_ontology():
    return yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))


def test_ontology_covers_exactly_30_suite_tasks():
    data = load_ontology()
    tasks = data["tasks"]
    assert len(tasks) == 30
    keys = {(t["suite"], int(t["task_idx"])) for t in tasks}
    assert len(keys) == 30
    assert {s for s, _ in keys} == {"libero_spatial", "libero_goal", "libero_10"}
    for suite in {"libero_spatial", "libero_goal", "libero_10"}:
        assert {idx for s, idx in keys if s == suite} == set(range(10))


def test_ontology_required_fields_and_mechanism_sets():
    data = load_ontology()
    for task in data["tasks"]:
        assert REQUIRED <= set(task)
        assert task["mechanism_type"] in ALLOWED
        assert isinstance(task["manipulated_object_aliases"], list)
        assert isinstance(task["target_aliases"], list)
        assert task["binding_source_priority"][-1] == "fail_closed_abstention"
    primary = [t for t in data["tasks"] if t["mechanism_type"] == "single_object_pick_place"]
    assert len(primary) == 17
    assert all(t["teacher_applicable"] == "positive_primary" for t in primary)


def test_ontology_keeps_detector_fields_forbidden_not_inputs():
    text = ONTOLOGY.read_text(encoding="utf-8")
    data = load_ontology()
    assert data["detector_independence"]["detector_fields_read"] is False
    forbidden_list = set(data["detector_independence"]["forbidden_fields"])
    assert FORBIDDEN <= forbidden_list
    for task in data["tasks"]:
        joined = "\n".join(str(v) for v in task.values())
        assert not any(field in joined for field in FORBIDDEN)
    assert "attack outcome" not in text.lower()
