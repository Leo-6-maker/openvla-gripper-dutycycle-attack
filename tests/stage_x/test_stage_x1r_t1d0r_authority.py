import json
from pathlib import Path

from scripts.stage_x.audit_stage_x1r_t1d0r_authority import (
    derive_population,
    load_source,
)


ROOT = Path(__file__).resolve().parents[2]


def test_source_fixture_drives_identity_join_and_selection(tmp_path):
    g10 = [
        "libero_10/task_00/state_20",
        "libero_10/task_00/state_21",
        "libero_10/task_01/state_20",
        "libero_goal/task_00/state_20",
    ]
    g10_path = tmp_path / "g10.json"
    clean_path = tmp_path / "clean.json"
    g10_path.write_text(json.dumps({"identities": g10}), encoding="utf-8")
    clean_path.write_text(json.dumps({"excluded_parent_keys": [g10[0]]}), encoding="utf-8")
    g10_set, _ = load_source({"name": "g10", "kind": "g10_identities", "path": str(g10_path), "json_path": ["identities"]})
    clean_set, _ = load_source({"name": "clean", "kind": "json_identity_list", "path": str(clean_path), "json_path": ["excluded_parent_keys"]}, g10_set)
    result = derive_population(
        g10_set,
        {"clean": clean_set},
        "fixture-salt",
        suites=("libero_10", "libero_goal"),
        tasks=range(2),
    )
    assert len(result["g10_rows"]) == len(g10)
    assert len(result["exclusion_union"]) == 1
    assert len(result["parent_rows"]) == 3
    assert len([row for row in result["design_rows"] if not row["selected"]]) == 1
    assert result["design_rows"][0]["missing_reason"] is not None or result["design_rows"][1]["missing_reason"] is not None


def test_protocol_is_frozen_and_fail_closed():
    protocol = json.loads((ROOT / "configs/STAGE_X_X1R_T1D0R_TIMING_PARENT_AUTHORITY_V1.json").read_text(encoding="utf-8"))
    timing = protocol["timing_freeze"]
    assert timing["timing_semantic_origin"] == "NEW_PROSPECTIVE_PI_FREEZE_20260818"
    assert timing["attack_start_step"] == "t_emit"
    assert timing["attack_window"]["length"] == 5
    assert timing["physical_followup"]["length"] == 10
    assert timing["prev_delta_contract"]["entry"] == "reset_to_zero_at_attack_window_entry"
    assert protocol["source_recompute"]["universe_contract"]["replacement"] is False
    assert all(value is False for value in protocol["authorization"].values() if isinstance(value, bool))
    assert protocol["protected_boundary"]["counters"] == {key: 0 for key in protocol["protected_boundary"]["counters"]}
    assert len(protocol["source_recompute"]["sources"]) == 10
    script = (ROOT / "scripts/stage_x/audit_stage_x1r_t1d0r_authority.py").read_text(encoding="utf-8").lower()
    assert "import torch" not in script
    assert "transformers" not in script
    assert "env.step(" not in script
    assert "selected_parent_keys" not in script
