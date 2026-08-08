from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.detector_v5 import run_stage_v_m1_v2_8gpu as supervisor
from scripts.detector_v5.analyze_stage_v_m1_v2_multigpu import (
    GPU_IDS,
    LABELS,
    classify_v2,
    classify_v2_with_profile,
    cross_gpu_pairs,
    evidence_profile,
    local_pairs,
    make_r2_plan,
)
from scripts.detector_v5 import audit_stage_v_m1_v2_8gpu as auditor
from scripts.detector_v5.run_stage_v_canonical_clean import _load_raw_capture_plan


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json"
V2_1 = ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_1_8gpu.json"
V1 = ROOT / "configs/stage_v_m1_visual_determinism_protocol_v1.json"


def _pair(*, exact: bool = True, visual: bool = False, full_sim: bool = True) -> dict:
    names = ("raw_observation", "physical_state", "full_sim_state", "policy_rgb", "model_input", "token", "postprocessed_action")
    traces = {name: {"equal": exact and not (visual and name in {"policy_rgb", "model_input"}), "first_mismatch_step": None if exact else 3, "first_mismatches": []} for name in names}
    traces["full_sim_state"]["equal"] = full_sim
    return {
        "initial_state_exact": exact,
        "terminal_step_exact": exact,
        "terminal_outcome_exact": exact,
        "traces": traces,
        "first_mismatch_by_component": {"policy_rgb": None if not visual else 3, "pixel_values": None if not visual else 3},
    }


def _matrices(*, same_mismatch: set[int] = set(), mode_mismatch: set[int] = set(), gpu_diff: bool = False, full_sim: bool = True):
    gpus = {}
    for gpu in GPU_IDS:
        pairs = {
            f"SAME_MODE_Q_GPU{gpu}": _pair(exact=gpu not in same_mismatch, visual=gpu in same_mismatch, full_sim=full_sim),
            f"SAME_MODE_C_GPU{gpu}": _pair(exact=gpu not in same_mismatch, visual=gpu in same_mismatch, full_sim=full_sim),
            f"CROSS_MODE_R1_GPU{gpu}": _pair(exact=gpu not in mode_mismatch, visual=gpu in mode_mismatch, full_sim=full_sim),
            f"CROSS_MODE_R2_GPU{gpu}": _pair(exact=gpu not in mode_mismatch, visual=gpu in mode_mismatch, full_sim=full_sim),
        }
        gpus[f"gpu_{gpu:02d}"] = {"gpu_id": gpu, "pairs": pairs}
    cross = {label: {f"CROSS_GPU_{label}_GPU0_GPU1": _pair(exact=not gpu_diff, visual=gpu_diff, full_sim=full_sim)} for label in LABELS}
    return {"gpus": gpus}, {"labels": cross}


def _write_run(root: Path, gpu: int, label: str, *, run_set: str = "runs") -> None:
    run = root / run_set / f"gpu_{gpu:02d}" / label
    trace = run / "trace"
    trace.mkdir(parents=True)
    row = {"step": 0, "observation": {"agentview_image": {"dtype": "uint8", "shape": [1, 1, 3], "raw_sha256": "a" * 64}}, "physical_state": {"qpos": [0]}, "model_inputs": {"pixel_values": [0], "input_ids": [1], "attention_mask": [1]}, "token_ids": [1], "raw_action": [0], "postprocessed_action": [0]}
    for name, value in {
        "observation_trace.jsonl": {"step": 0, "observation": row["observation"]},
        "physical_state_trace.jsonl": {"step": 0, "physical_state": row["physical_state"]},
        "full_sim_state_trace.jsonl": {"step": 0, "full_sim_state": {"qpos": [0]}},
        "policy_rgb_224_trace.jsonl": {"step": 0, "policy_rgb_224": [0]},
        "model_input_trace.jsonl": {"step": 0, "model_inputs": row["model_inputs"]},
        "policy_token_trace.jsonl": {"step": 0, "token_ids": [1], "raw_action": [0]},
        "postprocessed_action_trace.jsonl": {"step": 0, "postprocessed_action": [0]},
    }.items():
        (trace / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (run / "RB1_INDEPENDENT_RECEIPT.json").write_text(json.dumps({"canonical_parent_key": "libero_10/task_08/state_47", "mode": "CLEAN_QUALIFICATION", "initial_state_sha256": "a" * 64, "termination_step": 0, "terminal_outcome": "TASK_FAILURE", **{key: 0 for key in supervisor.BOUNDARIES}}), encoding="utf-8")


def test_v2_schema_and_status() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["schema"] == "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_8GPU"
    assert value["status"] == "FROZEN_DIAGNOSTIC_ONLY_NO_SCIENCE_AUTHORIZATION"


def test_v1_visual_protocol_is_unchanged() -> None:
    digest = hashlib.sha256(subprocess.run(["git", "show", "HEAD:configs/stage_v_m1_visual_determinism_protocol_v1.json"], check=True, capture_output=True).stdout).hexdigest()
    assert digest == "863ec1b043f7b40b4f59593262e4d5f3ac709576bd24d2a4e9cb73d1a819fdb0"


def test_v2_protocol_is_unchanged_and_v2_1_is_prospective() -> None:
    digest = hashlib.sha256(subprocess.run(["git", "show", "HEAD:configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json"], check=True, capture_output=True).stdout).hexdigest()
    assert digest == "063b3c5cf321a2eab93b41c6b03e55f5e0c5600ec165391099420716b93aeae6"
    assert json.loads(V2_1.read_text(encoding="utf-8"))["schema"] == "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_1_8GPU"
    assert json.loads(V2_1.read_text(encoding="utf-8"))["base_v2_protocol_sha256"] == digest


def test_v2_1_protocol_validates_graphics_and_fresh_gate_contract() -> None:
    value = supervisor.validate_protocol(V2_1)
    assert value["fresh_preflight_per_gate"] is True
    assert value["system_graphics_baseline"]["process_name"] == "Xorg"
    assert value["system_graphics_baseline"]["owner"] == "gdm"


def test_exact_gpu_inventory() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["gpu_ids"] == list(range(8))


def test_fixed_worker_mapping() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["worker_gpu_mapping"] == "FIXED_WORKER_I_TO_GPU_I"


def test_eight_workers_and_parallelism() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["workers"] == value["parallelism"] == 8


def test_phase_order_is_lockstep() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["phase_order"] == ["Q1", "C1", "Q2", "C2"]
    assert value["lockstep_barriers"] is True


def test_every_run_is_fresh() -> None:
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["fresh_subprocess_per_run"] is True


def test_gpu5_is_authorized_only_in_v2() -> None:
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["gpu5_authorized"] is True
    assert json.loads(V1.read_text(encoding="utf-8"))["gpu5_authorized"] is False


def test_science_boundaries_are_false() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(value[key] is False for key in ("new_science_rollouts_authorized", "formal_parent_promotion_authorized", "student_training_authorized", "protected_evaluation_authorized", "eval160_authorized", "vis_pgd_authorized"))


def test_classification_enum_includes_gpu_context() -> None:
    assert "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE" in json.loads(CONFIG.read_text(encoding="utf-8"))["classification_enum"]


def test_pair_counts_are_registered() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["gpu_local_pair_count"] == 32
    assert value["cross_gpu_pair_count"] == 112


def test_r2_plan_has_global_and_local_steps(tmp_path: Path) -> None:
    local, cross = _matrices(same_mismatch={0})
    plan = make_r2_plan(tmp_path, supervisor.IDENTITY, local, cross, "a" * 64, "b" * 64)
    assert plan["global_t_star"] == 3
    assert plan["local_t_star"]["gpu_00"] == 3


def test_r2_plan_includes_step_zero() -> None:
    local, cross = _matrices(same_mismatch={0})
    plan = make_r2_plan(Path("."), supervisor.IDENTITY, local, cross, "a" * 64, "b" * 64)
    assert all(0 in steps for steps in plan["capture_steps_by_gpu"].values())


def test_r2_plan_keys_are_physical_gpu_ids() -> None:
    local, cross = _matrices()
    plan = make_r2_plan(Path("."), supervisor.IDENTITY, local, cross, "a" * 64, "b" * 64)
    assert set(plan["capture_steps_by_gpu"]) == {str(gpu) for gpu in GPU_IDS}


def test_same_mode_visual_classification() -> None:
    local, cross = _matrices(same_mismatch={0, 1})
    assert classify_v2(local, cross) == "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM"


def test_gpu_context_classification() -> None:
    local, cross = _matrices(gpu_diff=True)
    assert classify_v2(local, cross) == "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE"


def test_mode_path_classification() -> None:
    local, cross = _matrices(mode_mismatch=set(range(8)))
    assert classify_v2(local, cross) == "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE"


def test_heterogeneous_classification() -> None:
    local, cross = _matrices(same_mismatch={0}, mode_mismatch={1})
    assert classify_v2(local, cross) == "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"


def test_simulator_divergence_has_priority() -> None:
    local, cross = _matrices(same_mismatch={0}, full_sim=False)
    assert classify_v2(local, cross) == "SIMULATOR_RUNTIME_NONDETERMINISM"


def test_evidence_profile_does_not_hide_mixed_raw_and_visual_mechanisms() -> None:
    local, cross = _matrices()
    raw = _pair()
    raw["initial_state_exact"] = False
    raw["traces"]["raw_observation"]["equal"] = False
    local["gpus"]["gpu_00"]["pairs"]["SAME_MODE_Q_GPU0"] = raw
    local["gpus"]["gpu_01"]["pairs"]["SAME_MODE_Q_GPU1"] = _pair(exact=True, visual=True)
    classification, profile = classify_v2_with_profile(local, cross)
    assert classification == "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"
    assert profile["mixed_mechanisms"] is True
    assert profile["raw_only_pairs"] == ["SAME_MODE_Q_GPU0"]
    assert profile["action_stable"] is True


def test_evidence_profile_keeps_action_divergence_separate() -> None:
    local, cross = _matrices(same_mismatch={0})
    local["gpus"]["gpu_00"]["pairs"]["SAME_MODE_Q_GPU0"]["traces"]["token"]["equal"] = False
    classification, profile = classify_v2_with_profile(local, cross)
    assert classification == "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM"
    assert profile["action_stable"] is False
    assert set(profile["action_divergent_pairs"]) == {"SAME_MODE_Q_GPU0", "SAME_MODE_C_GPU0"}


def test_independent_auditor_profile_matches_producer_profile_without_shared_classifier() -> None:
    local, cross = _matrices(same_mismatch={0})
    produced = evidence_profile(local, cross)
    audited = auditor._independent_profile(local, cross)
    assert audited == produced


def test_local_matrix_contains_32_pairs(tmp_path: Path) -> None:
    for gpu in GPU_IDS:
        for label in LABELS:
            _write_run(tmp_path, gpu, label)
    matrix = local_pairs(tmp_path, supervisor.IDENTITY)
    assert sum(len(item["pairs"]) for item in matrix["gpus"].values()) == 32


def test_cross_matrix_contains_112_pairs(tmp_path: Path) -> None:
    for gpu in GPU_IDS:
        for label in LABELS:
            _write_run(tmp_path, gpu, label)
    matrix = cross_gpu_pairs(tmp_path, supervisor.IDENTITY)
    assert sum(len(item) for item in matrix["labels"].values()) == 112


def test_v2_raw_plan_is_accepted_for_selected_gpu(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": "STAGE_V_M1_V2_RAW_CAPTURE_PLAN_V1", "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN", "identity": supervisor.IDENTITY, "capture_steps_by_gpu": {"3": [0, 221, 222, 223, 224, 225]}}), encoding="utf-8")
    plan, steps = _load_raw_capture_plan(path, {"canonical_parent_key": supervisor.IDENTITY}, 520, 3)
    assert plan["schema"].endswith("V2_RAW_CAPTURE_PLAN_V1")
    assert steps == frozenset({0, 221, 222, 223, 224, 225})


def test_v2_raw_plan_rejects_wrong_gpu(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": "STAGE_V_M1_V2_RAW_CAPTURE_PLAN_V1", "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN", "identity": supervisor.IDENTITY, "capture_steps_by_gpu": {"3": [0]}}), encoding="utf-8")
    with pytest.raises(Exception, match="GPU_MISSING"):
        _load_raw_capture_plan(path, {"canonical_parent_key": supervisor.IDENTITY}, 520, 4)


def test_v2_1_raw_plan_is_distinct_and_accepted(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": "STAGE_V_M1_V2_1_RAW_CAPTURE_PLAN_V1", "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN", "identity": supervisor.IDENTITY, "capture_steps_by_gpu": {"3": [0, 221]}}), encoding="utf-8")
    plan, steps = _load_raw_capture_plan(path, {"canonical_parent_key": supervisor.IDENTITY}, 520, 3)
    assert plan["schema"] == "STAGE_V_M1_V2_1_RAW_CAPTURE_PLAN_V1"
    assert steps == frozenset({0, 221})


def test_protected_boundary_names_are_frozen() -> None:
    assert set(json.loads(CONFIG.read_text(encoding="utf-8"))["protected_boundaries"]) == set(supervisor.BOUNDARIES)


def test_identity_is_single_and_exposed() -> None:
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["diagnostic_identity"] == supervisor.IDENTITY


def _mock_smi(monkeypatch, *, process: bool = False, used: int = 10) -> None:
    def query(*args: str) -> str:
        if args and args[0] == "pmon":
            return "# gpu pid type sm mem enc dec command\n"
        if "--query-gpu=" in " ".join(args):
            return "\n".join(f"{gpu}, GPU-{gpu}, NVIDIA A800, 81920, {used}, {81920 - used}, 0" for gpu in GPU_IDS)
        if process:
            return "GPU-3, 1234, foreign, 6696"
        return ""
    monkeypatch.setattr(supervisor, "_query_nvidia_smi", query)
    monkeypatch.setattr(supervisor, "_pid_detail", lambda pid: {"pid": pid, "owner": "foreign", "command": "foreign"})


def test_preflight_requires_all_eight_idle(monkeypatch) -> None:
    _mock_smi(monkeypatch)
    result = supervisor.gpu_preflight()
    assert result["status"] == "PASS"
    assert result["all_8_safe"] is True


def test_foreign_compute_process_makes_only_its_gpu_unsafe(monkeypatch) -> None:
    _mock_smi(monkeypatch, process=True)
    result = supervisor.gpu_preflight()
    assert result["status"] == "HOLD_WAIT_FOR_8GPU_SAFE"
    assert result["gpu_rows"][3]["safe"] is False


def test_materially_used_memory_is_not_idle(monkeypatch) -> None:
    _mock_smi(monkeypatch, used=2048)
    result = supervisor.gpu_preflight()
    assert result["status"] == "HOLD_WAIT_FOR_8GPU_SAFE"
    assert all("MEMORY_NOT_IDLE" in row["reasons"] for row in result["gpu_rows"])


def test_unmapped_process_telemetry_fails_closed(monkeypatch) -> None:
    _mock_smi(monkeypatch)
    original = supervisor._query_nvidia_smi

    def query(*args: str) -> str:
        if "--query-compute-apps=" in " ".join(args):
            return "GPU-UNKNOWN, 1234, foreign, 1"
        return original(*args)

    monkeypatch.setattr(supervisor, "_query_nvidia_smi", query)
    result = supervisor.gpu_preflight()
    assert result["status"] == "HOLD_WAIT_FOR_8GPU_SAFE"
    assert result["unmapped_processes"][0]["gpu_uuid"] == "GPU-UNKNOWN"
    assert all("UNMAPPED_PROCESS_TELEMETRY" in row["reasons"] for row in result["gpu_rows"])


def _mock_xorg(monkeypatch, *, owner: str = "gdm", command: str = "/usr/lib/xorg/Xorg vt1", used: int = 10) -> None:
    def query(*args: str) -> str:
        if args and args[0] == "pmon":
            return "# gpu pid type sm mem enc dec command\n" + "\n".join(f"{gpu} 7846 G - - - Xorg" for gpu in GPU_IDS)
        if "--query-gpu=" in " ".join(args):
            return "\n".join(f"{gpu}, GPU-{gpu}, NVIDIA A800, 81920, {used}, {81920 - used}, 0" for gpu in GPU_IDS)
        return ""
    monkeypatch.setattr(supervisor, "_query_nvidia_smi", query)
    monkeypatch.setattr(supervisor, "_pid_detail", lambda pid: {"pid": pid, "owner": owner, "command": command})


def test_system_graphics_baseline_is_whitelisted_only_when_consistent(monkeypatch) -> None:
    _mock_xorg(monkeypatch)
    contract = json.loads(V2_1.read_text(encoding="utf-8"))["system_graphics_baseline"]
    result = supervisor.gpu_preflight(system_graphics_baseline=contract)
    assert result["status"] == "PASS"
    assert len(result["baseline_system_graphics"]) == 8
    assert result["foreign_user_workloads"] == []
    assert all(row["system_graphics_processes"] for row in result["gpu_rows"])


def test_system_graphics_wrong_owner_is_foreign(monkeypatch) -> None:
    _mock_xorg(monkeypatch, owner="foreign")
    contract = json.loads(V2_1.read_text(encoding="utf-8"))["system_graphics_baseline"]
    result = supervisor.gpu_preflight(system_graphics_baseline=contract)
    assert result["status"] == "HOLD_WAIT_FOR_8GPU_SAFE"
    assert not result["baseline_system_graphics"]
    assert all(row["foreign_processes"] for row in result["gpu_rows"])


def test_system_graphics_baseline_memory_threshold_is_frozen(monkeypatch) -> None:
    _mock_xorg(monkeypatch, used=256)
    contract = json.loads(V2_1.read_text(encoding="utf-8"))["system_graphics_baseline"]
    result = supervisor.gpu_preflight(system_graphics_baseline=contract)
    assert result["status"] == "HOLD_WAIT_FOR_8GPU_SAFE"
    assert all("MEMORY_NOT_IDLE" in row["reasons"] for row in result["gpu_rows"])


def test_prepare_root_manifest_matches_auditor_authorization_schema(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(supervisor, "_git", lambda *args: "" if args == ("status", "--porcelain") else ("commit" if args == ("rev-parse", "HEAD") else "tree"))
    protocol = json.loads(V2_1.read_text(encoding="utf-8"))
    root = tmp_path / "m1-v2-1"
    supervisor.prepare_root(root, protocol, source_commit="commit", source_tree="tree", model_path="model", protocol_path=V2_1)
    manifest = json.loads((root / "M1_V2_MANIFEST.json").read_text(encoding="utf-8"))
    auditor.validate_manifest_authorization(manifest)
    assert all(manifest[key] is False for key in supervisor.AUTHORIZATION_FLAGS)
    assert manifest["protocol_schema"] == protocol["schema"]


def test_preflight_requires_prepared_manifest(tmp_path: Path) -> None:
    protocol = json.loads(V2_1.read_text(encoding="utf-8"))
    root = tmp_path / "unprepared"
    with pytest.raises(FileNotFoundError):
        supervisor._fresh_preflight(root, protocol, "PRE_CANARY", run_set="manual", protocol_path=V2_1)
    assert not root.exists()


def test_runtime_binding_receipt_is_actual_child_contract() -> None:
    receipt = {
        "schema": "STAGE_V_M1_V2_1_RUNTIME_BINDING_RECEIPT_V1", "status": "PASS",
        "logical_worker_id": "worker_3", "requested_physical_gpu": 3, "physical_gpu_index": 3,
        "cuda_visible_devices": "3", "torch_current_device": 0, "torch_device_uuid": "GPU-3",
        "mujoco_gl": "egl", "mujoco_egl_device_id": "3", "env_render_gpu_device_id": 3,
        "render_context_observed_device_id": 3, "run_set": "r1", "run_label": "Q1",
        "source_commit": "commit", "source_tree": "tree", "episode_started": False,
        "receipt_written_before_step_0": True, "pid": 12345, "renderer_device_information": {"observed_device_id": 3},
    }
    supervisor.validate_runtime_binding_receipt(receipt, 3, run_set="r1", phase="Q1", source_commit="commit", source_tree="tree")


def test_binding_receipt_contract_is_fail_closed() -> None:
    valid = {
        "logical_worker_id": "worker_3", "requested_physical_gpu": 3,
        "gpu_uuid": "GPU-3", "cuda_visible_devices": "3", "torch_current_device": 0,
        "mujoco_gl": "egl", "egl_device_identifier": 3,
        "renderer_device_information": {"observed_device_id": 3},
    }
    supervisor.validate_binding_receipt(valid, 3)
    with pytest.raises(supervisor.V2Error, match="GPU_BINDING_RECEIPT_GPU_UUID_MISSING"):
        supervisor.validate_binding_receipt({**valid, "gpu_uuid": ""}, 3)


def test_v2_modes_reject_v1_root_names(tmp_path: Path) -> None:
    with pytest.raises(supervisor.V2Error, match="V2_MUST_NOT_TOUCH_V1_ROOT"):
        supervisor._reject_v1_root(tmp_path / "M1_VISUAL_DETERMINISM_DIAGNOSTIC_8bd74ff6_20260808T051639Z")
