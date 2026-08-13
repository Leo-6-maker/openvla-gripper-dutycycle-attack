import json

from scripts.detector_v5.audit_stage_v_m4_prebranch_abort_closure import audit


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _root(tmp_path):
    root = tmp_path / "parent"
    _write(root / "gate" / "PARENT_STATUS.json", {
        "schema": "STAGE_V_M4_FORMAL_PARENT_STATUS_V2",
        "canonical_parent_key": "libero_10/task_06/state_30",
        "status": "HOLD_FORMAL_M4_STRUCTURAL_FAILURE",
        "intervention_started": False,
        "intervention_executed": False,
        "m4_outcomes_materialized": False,
        "v_phys_generated": False,
        "outcomes_read": False,
        "outcomes_read_uncertain": True,
        "protected_counters": COUNTERS,
    })
    _write(root / "gate" / "RESOURCE_RELEASE.json", {
        "schema": "STAGE_V_M4_FORMAL_RESOURCE_RELEASE_V1",
        "status": "PASS",
        "release_ok": True,
        "outcomes_read": False,
        "protected_counters": COUNTERS,
    })
    (root / "science").mkdir(parents=True)
    return root


def test_prebranch_closure_passes_without_science_artifacts(tmp_path):
    report = audit(_root(tmp_path), tmp_path / "audit", "libero_10/task_06/state_30")
    assert report["status"] == "PASS_PREBRANCH_ABORT_CLOSURE"
    assert report["branch_records"] == report["primary_window_steps"] == 0
    assert report["outcomes_read_uncertain"] is True


def test_prebranch_closure_rejects_branch_artifact(tmp_path):
    root = _root(tmp_path)
    (root / "science" / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl").write_text("{}\n", encoding="utf-8")
    report = audit(root, tmp_path / "audit", "libero_10/task_06/state_30")
    assert report["status"] == "HOLD_PREBRANCH_CLOSURE_INVALID"


def test_prebranch_closure_rejects_nested_science_artifact(tmp_path):
    root = _root(tmp_path)
    nested = root / "science" / "nested"
    nested.mkdir()
    (nested / "M4_V_PHYS_LABELS_V1.jsonl").write_text("{}\n", encoding="utf-8")
    report = audit(root, tmp_path / "audit", "libero_10/task_06/state_30")
    assert report["status"] == "HOLD_PREBRANCH_CLOSURE_INVALID"
