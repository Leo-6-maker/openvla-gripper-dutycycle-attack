#!/usr/bin/env python3
"""Tests for P4 Object Teacher labeler — determinism, parse errors, boundary."""
import json, tempfile, csv, os
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))

from scripts.migration.label_m1c_object_teacher import (
    build_records_from_telemetry, label_one, load_teacher_config,
    TEACHER_CONFIG_PATH,
)


def _make_cell(tmpdir, tel_content=None):
    d = Path(tmpdir) / "cell"
    d.mkdir()
    if tel_content:
        (d / "step_telemetry.csv").write_text(tel_content)
    (d / "episode_summary.json").write_text('{"task_success":true,"n_steps":100}')
    return d


def test_build_records():
    tel = "step,obj_x,obj_y,obj_z,eef_x,eef_y,eef_z,qpos_sum,raw_gripper\n0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n1,0.11,0.21,0.31,0.41,0.51,0.61,0.71,0.81\n"
    recs = build_records_from_telemetry(tel)
    assert len(recs) == 2
    assert recs[0]["policy_step_idx"] == 0


def test_label_one_ok():
    with tempfile.TemporaryDirectory() as tmp:
        tel = "step,obj_x,obj_y,obj_z,eef_x,eef_y,eef_z,qpos_sum,raw_gripper\n"
        tel += "0,0.0,0.0,0.05,0.0,0.0,0.1,0.0,0.0\n"
        # Add enough rows for a full grasp-carry-release cycle
        for i in range(1, 50):
            z = 0.05 + i * 0.01
            tel += f"{i},0.0,0.0,{z},0.0,0.0,{z+0.05},0.0,{0.1+i*0.001}\n"
        cell = _make_cell(tmp, tel)
        tc = load_teacher_config()
        result, err = label_one(cell, tc, "test_sha")
        assert err is None, f"Error: {err}"
        assert isinstance(result, dict)
        assert "teacher_valid" in result
        assert "hard_negative_category" in result


def test_label_determinism():
    """Same telemetry → same labels every time."""
    tel = "step,obj_x,obj_y,obj_z,eef_x,eef_y,eef_z,qpos_sum,raw_gripper\n"
    for i in range(50):
        z = 0.05 + i * 0.01
        tel += f"{i},0.0,0.0,{z},0.0,0.0,{z+0.05},0.0,{0.1+i*0.001}\n"
    with tempfile.TemporaryDirectory() as tmp:
        cell = _make_cell(tmp, tel)
        tc = load_teacher_config()
        r1, _ = label_one(cell, tc, "sha1")
        r2, _ = label_one(cell, tc, "sha1")
        assert r1["teacher_valid"] == r2["teacher_valid"]
        assert r1["teacher_anchor"] == r2["teacher_anchor"]
        assert r1["hard_negative_category"] == r2["hard_negative_category"]


def test_missing_telemetry():
    with tempfile.TemporaryDirectory() as tmp:
        cell = _make_cell(tmp)  # no telemetry
        tc = load_teacher_config()
        result, err = label_one(cell, tc, "sha")
        assert result is None
        assert "missing_telemetry" in err


def test_config_loads():
    tc = load_teacher_config()
    assert tc.guard == 5
    assert tc.K == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
