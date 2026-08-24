from __future__ import annotations

import json
from pathlib import Path

from scripts.detector_v5.freeze_stage_vi_b2_fresh_spatial_population import freeze


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _spec(path: Path, role: str) -> dict[str, str]:
    return {"path": str(path), "role": role}


def test_freeze_is_static_and_outcome_blind(tmp_path: Path) -> None:
    keys = [
        "libero_spatial/task_00/state_20",
        "libero_spatial/task_00/state_21",
        "libero_spatial/task_00/state_22",
        "libero_spatial/task_01/state_20",
        "libero_spatial/task_01/state_21",
        "libero_spatial/task_01/state_22",
    ]
    g10 = tmp_path / "G10.json"
    exposure = tmp_path / "exposure.json"
    clean = tmp_path / "clean.json"
    formal = tmp_path / "formal.json"
    matrix = tmp_path / "matrix.json"
    b2 = tmp_path / "b2.json"
    v2 = tmp_path / "v2.json"
    _write(g10, {"identities": keys})
    _write(exposure, {"schema": "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4", "status": "PASS", "excluded_parent_keys": [keys[0]], "branch_results_read": False})
    _write(clean, {"schema": "STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2", "status": "PASS", "excluded_parent_keys": [keys[1]]})
    for path, rows in ((formal, [keys[2]]), (matrix, [keys[3]]), (b2, [keys[0]]), (v2, [keys[1]])):
        _write(path, {"status": "FROZEN_OUTCOME_BLIND", "parents": [{"canonical_parent_key": key} for key in rows]})
    protocol = tmp_path / "protocol.json"
    _write(protocol, {
        "schema": "STAGE_VI_B2_FRESH_SPATIAL_POPULATION_PROTOCOL_V1",
        "status": "FROZEN_OUTCOME_BLIND_PRE_CLEAN_ROLLOUT",
        "selection": {"salt": "test", "target_new_spatial_parents": 2},
        "inputs": {
            "g10_manifest": _spec(g10, "g10"),
            "exposure_union": _spec(exposure, "exposure"),
            "clean_attempt_union": _spec(clean, "clean"),
            "stage_v_formal_manifests": [_spec(formal, "formal")],
            "stage_v_physical_matrix_manifests": [_spec(matrix, "matrix")],
            "stage_vi_b2_manifests": [_spec(b2, "b2")],
            "stage_vi_b2_development_manifests": [_spec(formal, "development")],
            "rejected_v2_manifest": _spec(v2, "rejected"),
            "prior_physical_named_roots": [],
        },
    })
    result = freeze(protocol, tmp_path / "frozen")
    assert result["status"] == "PASS_FROZEN_FRESH_SPATIAL_UNIVERSE"
    assert result["fresh_candidate_count"] == 2
    assert result["selection"]["candidate_outcomes_read"] is False
    assert result["protected_counters"]["protected_reads"] == 0
    assert (tmp_path / "frozen" / "SHA256SUMS.sha256").is_file()
