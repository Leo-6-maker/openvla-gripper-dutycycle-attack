"""Test privileged state extraction utilities."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils.libero_privileged_state import (
    _normalize_name,
    _parse_task_objects,
    _find_obs_key,
    extract_teacher_privileged_state,
    build_sim_debug_metadata,
    dump_sim_names,
)


class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("MILK") == "milk"

    def test_underscore(self):
        assert _normalize_name("cream_cheese") == "cream cheese"

    def test_strip_the(self):
        assert _normalize_name("the milk") == "milk"


class TestParseTaskObjects:
    def test_pick_place_in(self):
        obj, rec = _parse_task_objects("pick up the milk and place it in the basket")
        assert obj == "milk"
        assert rec == "basket"

    def test_pick_place_on(self):
        obj, rec = _parse_task_objects("pick up the chocolate pudding and place it on the plate")
        assert obj == "chocolate pudding"
        assert rec == "plate"

    def test_pick_only(self):
        obj, rec = _parse_task_objects("pick up the alphabet soup")
        assert obj == "alphabet soup"
        assert rec is None


class TestFindObsKey:
    def test_exact_match(self):
        obs = {"milk_1_pos": [1, 2, 3], "basket_1_pos": [4, 5, 6]}
        key = _find_obs_key(obs, "milk", "_pos")
        assert key == "milk_1_pos"

    def test_underscore_normalized(self):
        obs = {"cream_cheese_1_pos": [1, 2, 3]}
        key = _find_obs_key(obs, "cream cheese", "_pos")
        assert key == "cream_cheese_1_pos"

    def test_missing(self):
        obs = {"other_pos": [1, 2, 3]}
        key = _find_obs_key(obs, "milk", "_pos")
        assert key is None

    def test_quat_suffix(self):
        obs = {"milk_1_quat": [0, 0, 0, 1]}
        key = _find_obs_key(obs, "milk", "_quat")
        assert key == "milk_1_quat"


class TestExtractPrivilegedState:
    def test_obs_based_extraction(self):
        """With proper obs keys, extraction should find object and target."""
        obs = {
            "milk_1_pos": np.array([0.1, 0.2, 0.3]),
            "milk_1_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "basket_1_pos": np.array([0.4, 0.5, 0.6]),
            "robot0_eef_pos": np.array([0.15, 0.25, 0.35]),
        }
        # Mock env with a sim attribute
        env = MagicMock()
        sim = MagicMock()
        sim.model.body_names = ["world", "milk_1_main", "basket_1_main"]
        sim.model.site_names = ["basket_1_contain_region"]
        sim.data.body_xpos = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        sim.data.body_xquat = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]
        sim.data.site_xpos = [np.array([0.4, 0.5, 0.7])]
        env.env.sim = sim

        result = extract_teacher_privileged_state(
            env, obs, "pick up the milk and place it in the basket"
        )
        assert result["teacher_privileged_state_available"] is True
        assert result["object_pose_json"] != ""
        assert result["target_pose_json"] != ""
        assert result["object_to_target_distance"] != ""
        assert result["object_eef_distance"] != ""

    def test_missing_object_returns_false(self):
        """Missing object and target should return teacher_privileged_state_available=False."""
        obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.0])}
        env = MagicMock()
        sim = MagicMock()
        sim.model.body_names = ["world", "robot0_base"]
        sim.model.site_names = []
        sim.data.body_xpos = [np.array([0.0, 0.0, 0.0])] * 2
        sim.data.site_xpos = []
        env.env.sim = sim

        result = extract_teacher_privileged_state(
            env, obs, "pick up the milk and place it in the basket"
        )
        assert result["teacher_privileged_state_available"] is False
        assert result["privileged_state_error"] != ""

    def test_sim_body_fallback(self):
        """When obs keys are missing, sim body lookup should provide fallback."""
        obs = {"robot0_eef_pos": np.array([0.0, 0.0, 0.0])}
        env = MagicMock()
        sim = MagicMock()
        sim.model.body_names = ["world", "milk_1_main", "basket_1_main"]
        sim.model.site_names = []
        sim.data.body_xpos = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
        ]
        sim.data.body_xquat = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]
        sim.data.body_xpos = sim.data.body_xpos
        sim.data.body_xquat = sim.data.body_xquat
        env.env.sim = sim

        result = extract_teacher_privileged_state(
            env, obs, "pick up the milk and place it in the basket"
        )
        # Should find milk via body match but basket might not be found via site
        assert result["object_pose_json"] != ""
        # object found via sim body fallback


class TestBuildSimDebug:
    def test_debug_metadata_schema(self):
        """Debug metadata should contain all required fields."""
        env = MagicMock()
        sim = MagicMock()
        sim.model.body_names = ["world", "milk_1_main", "basket_1_main"]
        sim.model.site_names = ["basket_1_contain_region"]
        sim.model.geom_names = ["floor", "milk_geom"]
        env.env.sim = sim

        meta = build_sim_debug_metadata(env, "pick up the milk and place it in the basket")
        assert "body_names" in meta
        assert "site_names" in meta
        assert "geom_names" in meta
        assert "env_unwrap_chain" in meta
        assert meta["parsed_target_object"] == "milk"
        assert meta["parsed_target_receptacle"] == "basket"
        assert meta["selected_object_body"] == "milk_1_main"
        assert meta["selected_target_body_or_site"] == "basket_1_main"


class TestDumpSimNames:
    def test_dump_schema(self):
        """dump_sim_names should return body/site/geom/env info."""
        env = MagicMock()
        sim = MagicMock()
        sim.model.body_names = ["world"]
        sim.model.site_names = ["site1"]
        sim.model.geom_names = ["geom1"]
        env.env.sim = sim

        result = dump_sim_names(env)
        assert result["body_names"] == ["world"]
        assert result["site_names"] == ["site1"]
        assert result["geom_names"] == ["geom1"]
        assert "env_type" in result
        assert "inner_env_type" in result
