"""Tests for V4 trace remapper (RC1a-corrected semantics)."""

import csv
import os
import sys
import tempfile

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/stageb")

from remap_v4_trace_for_l12 import remap_v4_to_l12


def _make_v4_csv(rows: list[dict]) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="l12_test_")
    with os.fdopen(fd, "w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _base_v4_row(t: int, env=-1.0, decoded=1, qpos=0.0) -> dict:
    return {
        "step": str(t), "state_id": "0", "task": "tomato_sauce", "condition": "clean",
        "in_window": "1", "attack_this_step": "0",
        "clean_gripper_env": str(env), "executed_gripper_env": str(env),
        "decoded_open_bool": str(decoded),
        "gripper_qpos_before": str(qpos), "gripper_qpos_after": str(qpos),
        "physical_gripper_opening_delta": "0",
        "eef_x": "0.0", "eef_y": "0.0", "eef_z": "0.2", "eef_z_after": "0.2",
        "obj_x": "0.3", "obj_y": "0.0", "obj_z": "0.05", "obj_z_after": "0.05",
        "target_object_name": "tomato_sauce_1_main",
        "pgd_applied": "0", "random_seed_str": "", "random_seed_mode": "n/a",
        "perturbation_space": "none",
        "success_done": "0", "success_check": "0", "success_primary": "0",
        "reward": "0.0", "attack_seed": "0", "job_id": "0",
        "infra_status": "ok", "window_start": "0", "window_end": "10",
    }


def _s(val) -> str:
    """Convert to string for comparison with CSV-written values."""
    return str(val)


# ── E0.1: Gripper semantics ──

def test_env_negative_is_open():
    rows = [_base_v4_row(0, env=-1.0, decoded=1)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        assert len(issues) == 0
        assert float(result[0]["clean_gripper_raw_proxy"]) == 1.0
        assert int(result[0]["clean_gripper_raw_is_proxy"]) == 1
        assert int(result[0]["decoded_open_bool"]) == 1
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_env_positive_is_close():
    rows = [_base_v4_row(0, env=1.0, decoded=0)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        assert len(issues) == 0
        assert float(result[0]["clean_gripper_raw_proxy"]) == 0.0
        assert int(result[0]["decoded_open_bool"]) == 0
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_env_decoded_invariant_open():
    rows = [_base_v4_row(0, env=-1.0, decoded=0)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out", raise_on_invariant=False)
        assert len(issues) == 1
        assert "decoded_open=0" in issues[0]
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_env_decoded_invariant_close():
    rows = [_base_v4_row(0, env=1.0, decoded=1)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out", raise_on_invariant=False)
        assert len(issues) == 1
        assert "decoded_open=1" in issues[0]
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_env_neutral_is_not_forced():
    rows = [_base_v4_row(0, env=0.0, decoded=0)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        assert int(result[0]["gripper_semantics_valid"]) == 0
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_env_negative_is_open_multi_step():
    rows = [
        _base_v4_row(0, env=-1.0, decoded=1),
        _base_v4_row(1, env=-1.0, decoded=1),
        _base_v4_row(2, env=1.0, decoded=0),
        _base_v4_row(3, env=1.0, decoded=0),
    ]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        assert len(issues) == 0
        assert float(result[0]["clean_gripper_raw_proxy"]) == 1.0
        assert int(result[0]["clean_close"]) == 0
        assert float(result[2]["clean_gripper_raw_proxy"]) == 0.0
        assert int(result[2]["clean_close"]) == 1
        assert int(result[2]["close_onset"]) == 1
        assert int(result[2]["close_streak"]) == 1
        assert int(result[3]["close_onset"]) == 0
        assert int(result[3]["close_streak"]) == 2
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


# ── E0.2: Field validity ──

def test_missing_object_pose_stays_na():
    rows = [_base_v4_row(0)]
    rows[0]["obj_x"] = ""; rows[0]["obj_y"] = ""; rows[0]["obj_z"] = ""
    path = _make_v4_csv(rows)
    try:
        result, issues, field_issues = remap_v4_to_l12(path, path + ".out")
        assert int(result[0]["object_pose_valid"]) == 0
        assert result[0]["eef_to_obj_distance"] == ""
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_missing_eef_pose_stays_na():
    rows = [_base_v4_row(0)]
    rows[0]["eef_x"] = ""; rows[0]["eef_y"] = ""; rows[0]["eef_z"] = ""
    path = _make_v4_csv(rows)
    try:
        result, issues, field_issues = remap_v4_to_l12(path, path + ".out")
        assert int(result[0]["eef_pose_valid"]) == 0
        assert result[0]["eef_to_obj_distance"] == ""
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_does_not_compute_distance_from_defaults():
    rows = [_base_v4_row(0)]
    rows[0]["obj_x"] = "nan"; rows[0]["obj_y"] = ""
    path = _make_v4_csv(rows)
    try:
        result, issues, field_issues = remap_v4_to_l12(path, path + ".out")
        assert result[0]["eef_to_obj_distance"] == ""
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_raw_proxy_is_always_marked():
    rows = [_base_v4_row(t, env=(-1.0 if t % 2 == 0 else 1.0),
                          decoded=(1 if t % 2 == 0 else 0)) for t in range(10)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        for row in result:
            assert int(row["clean_gripper_raw_is_proxy"]) == 1
            assert row["clean_gripper_raw_source"] == "reconstructed_from_env_rc1a"
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")


def test_remapper_version_field():
    rows = [_base_v4_row(0)]
    path = _make_v4_csv(rows)
    try:
        result, issues, _ = remap_v4_to_l12(path, path + ".out")
        assert "rc1a_corrected" in str(result[0]["remapper_version"])
    finally:
        os.unlink(path)
        if os.path.exists(path + ".out"): os.unlink(path + ".out")
