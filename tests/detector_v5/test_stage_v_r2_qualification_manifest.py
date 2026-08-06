from __future__ import annotations

import json
from pathlib import Path

from scripts.detector_v5.prepare_stage_v_r2_qualification_manifest import prepare, sha256_file


def test_hash_ranked_manifest_is_20_per_suite_and_clean_only(tmp_path: Path) -> None:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    rows = [
        {"canonical_parent_key": f"{suite}/task_00/state_{index:02d}", "suite": suite,
         "task_index": 0, "state_index": index, "legacy_g10_test_only": True}
        for suite in suites for index in range(20)
    ]
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1",
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0,
                  "attack_informed_tuning": False, "new_cohort_clean_only_until_freeze": True},
        "selection_frozen_before_new_rollouts": True,
        "final_attack_test_parents_are_separate": True,
        "candidates": rows,
    }), encoding="utf-8")
    out = tmp_path / "out"
    manifest = prepare(pool, out, expected_sha256=sha256_file(pool), per_suite=20,
                       salt="STAGE_V_R2_CONTROL_QUALIFICATION_20260807")
    assert manifest["selected_count"] == 80
    assert manifest["selected_per_suite"] == {suite: 20 for suite in suites}
    assert json.loads((out / "STAGE_V_R2_QUALIFICATION_CANDIDATE_AUDIT.json").read_text())["verdict"] == "PASS"
