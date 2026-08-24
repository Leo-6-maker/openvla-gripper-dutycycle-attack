from scripts.detector_v5.stage_v_runtime_diff import diff


def test_runtime_diff_is_structured_and_deterministic():
    expected = {"state": {"values": [1, 2], "kind": "expected"}, "missing": 3}
    actual = {"state": {"values": [1, "2"], "kind": "actual"}, "extra": 4}
    context = {
        "parent_key": "libero_10/task_01/state_42", "probe_id": "Q00", "branch_id": "ZERO",
        "snapshot_source_commit": "a" * 40, "snapshot_source_tree": "b" * 40,
        "current_runtime_commit": "c" * 40, "current_runtime_tree": "d" * 40,
        "runtime_worktree": "/runtime", "closure_report_sha256": "e" * 64,
    }
    first = diff(expected, actual, context=context)
    second = diff(expected, actual, context=context)
    assert first == second
    assert [row["canonical_path"] for row in first] == ["$.extra", "$.missing", "$.state.kind", "$.state.values[1]"]
    assert all({"expected_type", "actual_type", "expected_shape", "actual_shape", "expected_sha256", "actual_sha256"}.issubset(row) for row in first)
    assert all(row["parent_key"] == context["parent_key"] and row["probe_id"] == "Q00" for row in first)
