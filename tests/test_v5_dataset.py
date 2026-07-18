from __future__ import annotations

from pathlib import Path

import pytest

from gripper_attack.v5_dataset import load_fit_registry


def test_v5_registry_filters_complete_global_2000_to_fit_800(tmp_path: Path):
    path = tmp_path / "registry.csv"
    fields = ["canonical_parent_key", "suite", "task_idx", "state_id", "split"]
    rows = []
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        for task in range(10):
            for state in range(50):
                rows.append(f"{suite}/task_{task:02d}/state_{state:02d},{suite},{task},{state},FIT_TRAIN\n")
    path.write_text(",".join(fields) + "\n" + "".join(rows), encoding="utf-8")
    fit = load_fit_registry(path)
    assert len(fit) == 800
    assert max(row["state_id"] for row in fit) == 19


def test_v5_registry_rejects_incomplete_global_universe(tmp_path: Path):
    path = tmp_path / "registry.csv"
    path.write_text("canonical_parent_key,suite,task_idx,state_id\nlibero_object/task_00/state_00,libero_object,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="800"):
        load_fit_registry(path)
