from __future__ import annotations

from scripts.detector_v5.run_stage_v_clean_replay_frozen import worker_identity_row


def test_worker_identity_row_bridges_qualification_field_names() -> None:
    row = worker_identity_row(
        {"canonical_parent_key": "libero_10/task_00/state_47", "task_index": 0, "state_index": 47},
        "state-sha",
    )
    assert row["task_idx"] == 0
    assert row["state_id"] == 47
    assert row["initial_state_sha256"] == "state-sha"
    assert row["split"] == "STAGE_V_R2_QUALIFICATION"
