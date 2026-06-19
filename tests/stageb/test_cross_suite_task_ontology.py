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



def test_positive_aliases_are_role_clean_and_disjoint():
    data = load_ontology()
    for task in data["tasks"]:
        objs = set(task["manipulated_object_aliases"] or [])
        targets = set(task["target_aliases"] or [])
        if task["mechanism_type"] == "single_object_pick_place":
            assert objs, task
            assert targets, task
            assert objs.isdisjoint(targets), task
        if task["suite"] == "libero_spatial":
            assert "plate" not in objs
            assert "bowl" not in targets


def test_requested_mixed_and_multi_event_reclassifications():
    data = load_ontology()
    tasks = {(t["suite"], int(t["task_idx"])): t for t in data["tasks"]}
    for key in [("libero_goal", 3), ("libero_10", 2), ("libero_10", 3), ("libero_10", 9)]:
        assert tasks[key]["mechanism_type"] == "mixed_articulated_pick_place"
        assert tasks[key]["teacher_applicable"] == "supplementary_event_level_audit"
        assert tasks[key]["articulated_body_aliases"]
    for key in [("libero_10", 4), ("libero_10", 6), ("libero_10", 8)]:
        assert tasks[key]["mechanism_type"] == "multi_object_transfer"
        assert tasks[key]["teacher_applicable"] == "supplementary_event_level_audit"
        assert tasks[key]["expected_event_count_class"] == "multi"


def test_mechanism_applicability_event_count_consistency():
    data = load_ontology()
    for task in data["tasks"]:
        mech = task["mechanism_type"]
        app = task["teacher_applicable"]
        count = task["expected_event_count_class"]
        if mech == "single_object_pick_place":
            assert app == "positive_primary"
            assert count == "single"
        elif mech in {"multi_object_transfer", "mixed_articulated_pick_place"}:
            assert app == "supplementary_event_level_audit"
            assert count in {"single", "multi"}
        else:
            assert app == "semantic_abstain_negative"
            assert count == "zero"


def test_prereg_summary_counts_match_ontology():
    data = load_ontology()
    by_suite = {}
    for task in data["tasks"]:
        suite_counts = by_suite.setdefault(task["suite"], {})
        mech = task["mechanism_type"]
        suite_counts[mech] = suite_counts.get(mech, 0) + 1

    prereg = (REPO / "reports" / "CROSS_SUITE_LAYER1_RESOLVER_PREREG.md").read_text(encoding="utf-8")
    assert "libero_spatial: 10 single_object_pick_place" in prereg
    assert "libero_goal: 6 single_object_pick_place, 2 articulated_only, 1 mixed_articulated_pick_place, 1 push_or_planar" in prereg
    assert "libero_10: 1 single_object_pick_place, 6 multi_object_transfer, 3 mixed_articulated_pick_place" in prereg
    assert by_suite == {
        "libero_spatial": {"single_object_pick_place": 10},
        "libero_goal": {
            "single_object_pick_place": 6,
            "articulated_only": 2,
            "mixed_articulated_pick_place": 1,
            "push_or_planar": 1,
        },
        "libero_10": {
            "single_object_pick_place": 1,
            "multi_object_transfer": 6,
            "mixed_articulated_pick_place": 3,
        },
    }
