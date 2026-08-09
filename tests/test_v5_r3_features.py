from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gripper_attack.v5_r3_features import ACTION_GRIPPER_SOURCE, FEATURE_ORDER, load_feature_binding, materialize_fit670_features
from scripts.detector_v5.run_r3_micro_overfit import _is_false_flag, _known_label

ROOT = Path(__file__).resolve().parents[1]


def _episode(n: int = 5) -> dict:
    steps = []
    telemetry = []
    for step in range(n):
        raw = [0.1, 0.0, 0.01, 0.0, 0.0, 0.0, 0.2 if step < 3 else 0.9]
        steps.append({"step": step, "raw_action_7d": raw, "action_raw_7d": list(raw), "action_env_7d": [*raw[:6], 1.0 if raw[6] <= 0.5 else -1.0]})
        telemetry.append({"step": step, "robot0_gripper_qpos": [0.03, -0.03], "robot0_eef_pos": [0.0, 0.0, 0.5 + 0.01 * step]})
    return {"steps": steps, "telemetry": telemetry}


def test_official_25d_shape_order_and_finite():
    rows = materialize_fit670_features(_episode())
    assert len(FEATURE_ORDER) == 25
    assert all(len(row["features_25d"]) == 25 for row in rows)
    assert np.isfinite(np.asarray([row["features_25d"] for row in rows])).all()
    assert all(row["feature_order"] == list(FEATURE_ORDER) for row in rows)
    assert all(row["action_gripper_source"] == ACTION_GRIPPER_SOURCE for row in rows)
    assert rows[0]["features_25d"][12] == pytest.approx(0.2)
    assert rows[0]["features_25d"][1] == pytest.approx(0.06)


def test_gripper_qpos_uses_absolute_finger_sum():
    episode = _episode(1)
    episode["telemetry"][0]["robot0_gripper_qpos"] = [0.03, -0.04]
    row = materialize_fit670_features(episode)[0]
    assert row["features_25d"][1] == pytest.approx(0.07)


def test_missing_env_action_fails_closed():
    episode = _episode(1)
    del episode["steps"][0]["action_env_7d"]
    try:
        materialize_fit670_features(episode)
    except ValueError as exc:
        assert "action_env_7d" in str(exc)
    else:
        raise AssertionError("missing env action was accepted")


def test_boundary_gripper_fails_closed():
    episode = _episode(1)
    episode["steps"][0]["raw_action_7d"][6] = 0.5
    episode["steps"][0]["action_raw_7d"][6] = 0.5
    episode["steps"][0]["action_env_7d"][6] = 1.0
    try:
        materialize_fit670_features(episode)
    except ValueError as exc:
        assert "boundary" in str(exc)
    else:
        raise AssertionError("boundary gripper was accepted")


def test_gripper_ranges_fail_closed():
    episode = _episode(1)
    episode["steps"][0]["raw_action_7d"][6] = 2.0
    episode["steps"][0]["action_raw_7d"][6] = 2.0
    try:
        materialize_fit670_features(episode)
    except ValueError as exc:
        assert "outside [0,1]" in str(exc)
    else:
        raise AssertionError("out-of-range raw gripper was accepted")


def test_prefix_is_invariant_to_future_changes():
    base = _episode()
    changed = copy.deepcopy(base)
    changed["steps"][-1]["raw_action_7d"][0] = 0.77
    changed["steps"][-1]["action_raw_7d"][0] = 0.77
    changed["telemetry"][-1]["robot0_eef_pos"][2] = 9.0
    left = materialize_fit670_features(base)
    right = materialize_fit670_features(changed)
    assert [row["features_25d"] for row in left[:-1]] == [row["features_25d"] for row in right[:-1]]


def test_raw_action_alias_mismatch_fails_closed():
    episode = _episode()
    episode["steps"][1]["action_raw_7d"][0] += 1e-3
    try:
        materialize_fit670_features(episode)
    except ValueError as exc:
        assert "aliases disagree" in str(exc)
    else:
        raise AssertionError("mismatched runtime action aliases were accepted")


def test_feature_binding_matches_source_and_explicitly_supersedes_legacy_schema():
    path = ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json"
    binding = json.loads(path.read_text(encoding="utf-8"))
    order_sha = hashlib.sha256(json.dumps(list(FEATURE_ORDER), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert binding["feature_order_sha256"] == order_sha
    source_bytes = (ROOT / "src" / "gripper_attack" / "sc5_streaming_features_v2.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source_bytes).hexdigest() == binding["adapter_source_sha256"]
    assert binding["adapter_source_hash_algorithm"] == "SHA256"
    assert binding["adapter_source_hash_normalization"] == "UTF-8 text with CRLF normalized to LF"
    assert binding["action_gripper"]["range"] == [0.0, 1.0]
    assert binding["legacy_schema_conflict"]["status"] == "NOT_CONSUMABLE_FOR_R3"
    assert load_feature_binding(path, ROOT)["schema"] == binding["schema"]


def test_feature_binding_rejects_mutated_order(tmp_path):
    path = tmp_path / "binding.json"
    binding = json.loads((ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json").read_text(encoding="utf-8"))
    binding["feature_order"][0] = "mutated"
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="feature-order"):
        load_feature_binding(path, ROOT)


def test_feature_binding_uses_canonical_source_sha():
    binding = load_feature_binding(ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json", ROOT)
    source = (ROOT / binding["adapter_source"]).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    assert hashlib.sha256(source).hexdigest() == binding["adapter_source_sha256"]


def test_feature_binding_status_and_label_schema_fail_closed(tmp_path):
    path = tmp_path / "binding.json"
    binding = json.loads((ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json").read_text(encoding="utf-8"))
    binding["status"] = "MUTATED"
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen"):
        load_feature_binding(path, ROOT)
    valid = {"value": "TRUE", "valid_mask": True, "mask": True, "right_censored": False}
    assert _known_label(valid, "k10_feasibility")
    assert not _known_label({**valid, "value": "UNKNOWN"}, "k10_feasibility")
    with pytest.raises(ValueError, match="right_censored"):
        _known_label({"value": "FALSE", "valid_mask": True, "mask": True}, "k10_feasibility")
    with pytest.raises(ValueError, match="valid_mask"):
        _known_label({**valid, "valid_mask": 1}, "k10_feasibility")


def test_micro_false_flag_accepts_zero_but_rejects_true_and_one():
    assert _is_false_flag(False)
    assert _is_false_flag(0)
    assert not _is_false_flag(True)
    assert not _is_false_flag(1)
    assert not _is_false_flag(None)
