import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.run_provisional_layer1_batch import (  # noqa: E402
    _is_complete_clean,
    _episode_path,
    build_full_manifest,
    component_manifest,
)
from scripts.stageb.cross_suite_layer1_resolver import load_ontology  # noqa: E402


ONTOLOGY = REPO / "configs" / "cross_suite_task_ontology_v1.yaml"


def _clean_row(*, state=0, status="COMPLETE_VALID", key=None, invalid="0"):
    return {
        "canonical_key": key or f"libero_spatial|0|{state}|0|CLEAN",
        "episode_path": f"/tmp/clean_s{state}",
        "suite": "libero_spatial",
        "task_idx": "0",
        "state_id": str(state),
        "eval_seed": "0",
        "condition": "CLEAN",
        "status": status,
        "clean_only_contract": "True",
        "invalid_feature_steps": invalid,
        "task_success": "False",
        "n_steps": "123",
        "artifact_recursive_sha256": f"sha{state}",
    }


def _train_row(*, state=10, primary_status="COMPLETE", condition="CLEAN"):
    return {
        "canonical_key": f"libero_spatial|0|{state}|0|CLEAN",
        "primary_output_dir": f"/tmp/train_s{state}",
        "suite": "libero_spatial",
        "task_idx": "0",
        "state_id": str(state),
        "eval_seed": "0",
        "condition": condition,
        "primary_status": primary_status,
        "invalid_feature_steps": "0",
        "task_success": "True",
        "n_steps": "88",
    }


def test_component_manifest_matches_frozen_h2_snapshot():
    manifest = component_manifest()
    assert manifest["status"] == "PASS"
    assert set(manifest["components"]) == {"ontology", "physics_config", "teacher_schema", "timing_contract"}


def test_train300_rows_use_primary_output_dir_and_complete_primary_status():
    row = _train_row()
    ok, reason = _is_complete_clean(row, ledger_kind="train300")
    assert ok, reason
    assert _episode_path(row) == "/tmp/train_s10"

    bad = dict(row)
    bad["primary_status"] = "INFRA_FAILED"
    ok, reason = _is_complete_clean(bad, ledger_kind="train300")
    assert not ok
    assert reason == "primary_status_not_complete"


def test_clean300_rows_fail_closed_on_status_contract_and_invalid_features():
    ok, reason = _is_complete_clean(_clean_row(), ledger_kind="clean300")
    assert ok, reason

    for row, expected in [
        (_clean_row(status="RUNNING"), "status_not_complete_valid"),
        (_clean_row(invalid="1"), "invalid_feature_steps_nonzero"),
        ({**_clean_row(), "clean_only_contract": "False"}, "clean_only_contract_false"),
    ]:
        ok, reason = _is_complete_clean(row, ledger_kind="clean300")
        assert not ok
        assert reason == expected


def test_full_manifest_selects_exact_state_range_and_rejects_duplicates():
    ontology = load_ontology(ONTOLOGY)
    rows = [_clean_row(state=0), _clean_row(state=1), _clean_row(state=10), _clean_row(state=1, key="libero_spatial|0|1|0|CLEAN")]
    manifest = build_full_manifest(
        ledger_rows=rows,
        ontology=ontology,
        ledger_kind="clean300",
        split_name="clean300_test_s0_9",
        state_min=0,
        state_max=9,
        expected_count=2,
    )
    assert manifest["status"] == "FAIL"
    assert manifest["selected_count"] == 2
    assert manifest["duplicate_keys"] == ["libero_spatial|0|1|0|CLEAN"]
    assert {r["state_id"] for r in manifest["selected"]} == {0, 1}
    assert all(r["split_name"] == "clean300_test_s0_9" for r in manifest["selected"])


def test_full_manifest_passes_when_expected_count_and_unique_keys_match():
    ontology = load_ontology(ONTOLOGY)
    rows = [_train_row(state=10), _train_row(state=11), _train_row(state=18)]
    manifest = build_full_manifest(
        ledger_rows=rows,
        ontology=ontology,
        ledger_kind="train300",
        split_name="train300_train_s10_17",
        state_min=10,
        state_max=17,
        expected_count=2,
    )
    assert manifest["status"] == "PASS"
    assert [r["state_id"] for r in manifest["selected"]] == [10, 11]
    assert all(r["episode_path"].startswith("/tmp/train_s") for r in manifest["selected"])
