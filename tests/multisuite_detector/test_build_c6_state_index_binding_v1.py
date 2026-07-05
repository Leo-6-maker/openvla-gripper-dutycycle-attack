from tools.multisuite_detector import build_c6_state_index_binding_v1 as m


def test_gate_name():
    assert m.GATE == "C6_1H_STATE_INDEX_BINDING_AUDIT_BUILD"


def test_task_and_state_tokens():
    assert m.task_token(1) == "task_01"
    assert m.state_token("libero_goal/task_01/state_000") == "state_000"


def test_collect_fields_path_and_index():
    fields = m.collect_fields({"outer": {"state_path": "a.json", "episode_idx": 7}})
    assert fields["state_path"] == "a.json"
    assert fields["episode_idx"] == "7"


def test_identity_match_reason_parent_fields():
    parent = {"parent_id": "libero_goal/task_01/state_000", "episode_key": "episode_A", "suite": "libero_goal", "task_id": 1}
    reasons = m.identity_match_reason({"parent_id": "libero_goal/task_01/state_000", "state_path": "x.json"}, parent)
    assert "parent_id" in reasons


def test_duplicate_same_path_passes():
    rows = [
        {"is_concrete_binding": True, "resolved_path": "/tmp/a.json", "resolved_path_exists": True, "index_fields": "{}"},
        {"is_concrete_binding": True, "resolved_path": "/tmp/a.json", "resolved_path_exists": True, "index_fields": "{}"},
    ]
    assert m.decide_status(rows, []) == m.PASS_STATE_PATH
    assert m.unique_handles(rows) == {"path:/tmp/a.json"}
    assert "state_path" in m.recommendation(m.PASS_STATE_PATH)


def test_single_index_passes():
    rows = [{"is_concrete_binding": True, "resolved_path": "", "resolved_path_exists": False, "index_fields": "{\"episode_idx\": \"7\"}"}]
    assert m.decide_status(rows, []) == m.PASS_STATE_INDEX
    assert "state index" in m.recommendation(m.PASS_STATE_INDEX)


def test_distinct_indices_are_ambiguous():
    rows = [
        {"is_concrete_binding": True, "resolved_path": "", "resolved_path_exists": False, "index_fields": "{\"episode_idx\": \"7\"}"},
        {"is_concrete_binding": True, "resolved_path": "", "resolved_path_exists": False, "index_fields": "{\"episode_idx\": \"8\"}"},
    ]
    assert m.decide_status(rows, []) == m.HOLD_AMBIGUOUS


def test_exact_file_match_passes():
    files = [{"sha256_matches_initial_state_hash": True}]
    assert m.decide_status([], files) == m.PASS_FILE_HASH
    assert "SHA" in m.recommendation(m.PASS_FILE_HASH)


def test_multiple_exact_file_matches_ambiguous():
    files = [{"sha256_matches_initial_state_hash": True}, {"sha256_matches_initial_state_hash": True}]
    assert m.decide_status([], files) == m.HOLD_AMBIGUOUS


def test_missing_path_holds():
    rows = [{"is_concrete_binding": True, "resolved_path": "/tmp/missing.json", "resolved_path_exists": False, "index_fields": "{}"}]
    assert m.decide_status(rows, []) == m.HOLD_PATH_MISSING
    assert "missing" in m.recommendation(m.HOLD_PATH_MISSING)


def test_no_binding_holds():
    assert m.decide_status([], []) == m.HOLD_NO_BINDING
    assert "No concrete" in m.recommendation(m.HOLD_NO_BINDING)
