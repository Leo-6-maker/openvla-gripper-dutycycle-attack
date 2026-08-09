from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gripper_attack.stage_v_canonical_execution_core import EpisodeTrace, write_raw_capture_artifacts
from scripts.detector_v5.analyze_stage_v_m1_visual_divergence import (
    _classify_pair,
    classify_repeatability,
    make_capture_plan,
    numeric_pair,
)
from scripts.detector_v5.run_stage_v_canonical_clean import _load_raw_capture_plan


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json"


def _trace(equal: bool = True, *, full_sim_equal: bool = True) -> dict:
    names = ("raw_observation", "physical_state", "full_sim_state", "policy_rgb", "model_input", "token", "postprocessed_action")
    traces = {name: {"equal": equal if name != "full_sim_state" else full_sim_equal, "first_mismatch_step": None if equal else 0, "first_mismatches": []} for name in names}
    return {
        "initial_state_exact": equal,
        "terminal_step_exact": equal,
        "terminal_outcome_exact": equal,
        "traces": traces,
        "first_mismatch_by_component": {"input_ids": None, "attention_mask": None},
    }


def test_m1_v1_protocol_blob_is_frozen() -> None:
    digest = hashlib.sha256(V1.read_bytes()).hexdigest()
    assert digest == "18d2421b172cef881b8de70c4bccf0c65174dd66a8402b75f0696d1878f96a69"
    value = json.loads(V1.read_text(encoding="utf-8"))
    assert value["tolerance_allowed"] is False
    assert "full_sim_state_trace_sha256" not in value["common_trace_hash_fields"]


def test_same_exact_pair_is_unclassified_not_pass_claim() -> None:
    assert _classify_pair(_trace(True)["traces"]) == "UNCLASSIFIED"


def test_full_sim_mismatch_blocks_renderer_classification() -> None:
    pair = _trace(False, full_sim_equal=False)
    pair["traces"]["raw_observation"]["equal"] = False
    assert _classify_pair(pair["traces"]) == "SIMULATOR_RUNTIME_NONDETERMINISM"


def test_raw_observation_only_difference() -> None:
    pair = _trace(True)
    pair["traces"]["raw_observation"]["equal"] = False
    assert _classify_pair(pair["traces"]) == "RAW_OBSERVATION_NON_POLICY_DIFFERENCE"


def test_policy_rgb_difference_with_stable_actions() -> None:
    pair = _trace(True)
    pair["traces"]["policy_rgb"]["equal"] = False
    pair["traces"]["model_input"]["equal"] = False
    assert _classify_pair(pair["traces"]) == "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE"


def test_repeatability_same_mode_difference_is_not_cross_mode_claim() -> None:
    pairs = {name: _trace(True) for name in ("SAME_MODE_Q", "SAME_MODE_C", "CROSS_MODE_R1", "CROSS_MODE_R2")}
    pairs["SAME_MODE_Q"]["traces"]["policy_rgb"]["equal"] = False
    pairs["SAME_MODE_Q"]["traces"]["model_input"]["equal"] = False
    assert classify_repeatability(pairs) == "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE"


def test_repeatability_cross_mode_only() -> None:
    pairs = {name: _trace(True) for name in ("SAME_MODE_Q", "SAME_MODE_C", "CROSS_MODE_R1", "CROSS_MODE_R2")}
    pairs["CROSS_MODE_R1"]["traces"]["raw_observation"]["equal"] = False
    assert classify_repeatability(pairs) == "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE"


def test_capture_plan_is_mechanical_and_includes_step_zero(tmp_path: Path) -> None:
    report = tmp_path / "M1_REPEATABILITY_REPORT.json"
    matrix = tmp_path / "M1_REPEATABILITY_PAIR_MATRIX.json"
    report.write_text(json.dumps({"schema": "test"}), encoding="utf-8")
    matrix.write_text(json.dumps({"pairs": {"SAME_MODE_Q": {"first_mismatch_by_component": {"policy_rgb": 223, "pixel_values": 223}}}}), encoding="utf-8")
    output = tmp_path / "M1_RAW_CAPTURE_PLAN.json"
    plan = make_capture_plan(report, "libero_10/task_08/state_47", output)
    assert plan["t_star"] == 223
    assert plan["capture_steps"] == [0, 221, 222, 223, 224, 225]


def test_capture_plan_rejects_unregistered_identity(tmp_path: Path) -> None:
    plan = {"schema": "STAGE_V_M1_RAW_CAPTURE_PLAN_V1", "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN", "identity": "libero_10/task_08/state_47", "capture_steps": [0]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Exception, match="IDENTITY_MISMATCH"):
        _load_raw_capture_plan(path, {"canonical_parent_key": "libero_10/task_03/state_49"}, 520)


def test_raw_bfloat16_sidecar_preserves_bytes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import numpy as np

    trace = EpisodeTrace(
        identity={"canonical_parent_key": "libero_10/task_08/state_47"},
        initial_state_sha256="a" * 64,
        steps=[{
            "step": 223,
            "raw_capture": {
                "raw_observation": {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)},
                "policy_rgb_224": np.ones((2, 2, 3), dtype=np.uint8),
                "model_inputs": {"pixel_values": torch.tensor([1.0], dtype=torch.bfloat16)},
            },
        }],
        actions=[], terminal_outcome="TASK_FAILURE", termination_step=223, termination_reason="test",
    )
    manifest = write_raw_capture_artifacts(tmp_path / "raw_capture", trace)
    pixel = next(item for item in manifest["entries"] if item["field"] == "pixel_values")
    assert pixel["dtype"] == "torch.bfloat16"
    assert pixel["byte_length"] == 2
    assert len(pixel["raw_sha256"]) == 64


def test_numeric_pair_reports_exact_bytes_and_descriptive_metrics(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    for side, value in (("left", np.zeros((2, 2, 3), dtype=np.uint8)), ("right", np.ones((2, 2, 3), dtype=np.uint8))):
        run = tmp_path / side
        raw = run / "trace/raw_capture/step_000000"
        raw.mkdir(parents=True)
        data = value.tobytes(order="C")
        binary = raw / "policy_rgb_224__policy_rgb_224.bin"
        binary.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        descriptor = {"dtype": "uint8", "shape": [2, 2, 3], "raw_sha256": digest, "byte_length": len(data), "binary_path": "step_000000/policy_rgb_224__policy_rgb_224.bin", "descriptor_path": "step_000000/policy_rgb_224__policy_rgb_224.json", "group": "policy_rgb_224", "field": "policy_rgb_224"}
        (raw / "policy_rgb_224__policy_rgb_224.json").write_text(json.dumps(descriptor), encoding="utf-8")
        (run / "M1_RAW_CAPTURE_MANIFEST.json").write_text(json.dumps({"entries": [descriptor]}), encoding="utf-8")
    result = numeric_pair(tmp_path / "left", tmp_path / "right")
    assert result["different_fields"][0]["metrics"]["num_different_elements"] == 12
    assert result["different_fields"][0]["metrics"]["max_abs_diff"] == 1.0
