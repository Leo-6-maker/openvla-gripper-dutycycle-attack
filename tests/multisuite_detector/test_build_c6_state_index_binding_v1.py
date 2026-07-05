from tools.multisuite_detector import build_c6_state_index_binding_v1 as m


def test_gate_name():
    assert m.GATE == "C6_1H_STATE_INDEX_BINDING_AUDIT_BUILD"


def test_duplicate_same_path_decision_not_ambiguous():
    rows = [
        {"is_concrete_binding": True, "resolved_path": "/tmp/a.json", "resolved_path_exists": True, "index_fields": "{}"},
        {"is_concrete_binding": True, "resolved_path": "/tmp/a.json", "resolved_path_exists": True, "index_fields": "{}"},
    ]
    assert m.decide_status(rows, []) == m.PASS_STATE_PATH


def test_distinct_indices_are_ambiguous():
    rows = [
        {"is_concrete_binding": True, "resolved_path": "", "resolved_path_exists": False, "index_fields": "{\"episode_idx\": \"7\"}"},
        {"is_concrete_binding": True, "resolved_path": "", "resolved_path_exists": False, "index_fields": "{\"episode_idx\": \"8\"}"},
    ]
    assert m.decide_status(rows, []) == m.HOLD_AMBIGUOUS
