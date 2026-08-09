from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5 import run_stage_v_m1_v2_8gpu as supervisor
from scripts.detector_v5 import analyze_stage_v_m1_v2_multigpu as producer


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_2_dynamic_cohort_8gpu.json"


def _smi(monkeypatch: pytest.MonkeyPatch, gpu_ids: tuple[int, ...], foreign_gpu: int | None = None) -> None:
    def query(*args: str) -> str:
        if args and args[0] == "pmon":
            return "# gpu pid type sm mem enc dec command\n"
        joined = " ".join(args)
        if "--query-gpu=" in joined:
            return "\n".join(f"{gpu}, GPU-{gpu}, NVIDIA A800, 81920, 10, 81910, 0" for gpu in gpu_ids)
        if "--query-compute-apps=" in joined and foreign_gpu is not None:
            return f"GPU-{foreign_gpu}, 964381, foreign, 6715"
        return ""

    monkeypatch.setattr(supervisor, "_query_nvidia_smi", query)
    monkeypatch.setattr(supervisor, "_pid_detail", lambda pid: {"pid": pid, "owner": "huanzze", "command": "foreign"})


def test_v2_2_protocol_is_prospective_and_formula_driven() -> None:
    value = supervisor.validate_protocol(PROTOCOL)
    assert value["schema"] == supervisor.DYNAMIC_PROTOCOL_SCHEMA
    assert value["primary_clean_gpu_minimum"] == 4
    assert value["total_r1_runs"] == "4N"
    assert value["gpu_local_pair_count"] == "4N"
    assert value["cross_gpu_pair_count"] == "4*C(N,2)"
    assert value["complete_ownership"] == "AUDITOR_ONLY"


def test_dynamic_counts_for_seven_clean_gpus() -> None:
    assert supervisor.dynamic_counts((0, 1, 2, 4, 5, 6, 7)) == {
        "primary_clean_gpu_count": 7,
        "r1_run_count": 28,
        "gpu_local_pair_count": 28,
        "cross_gpu_pair_count": 84,
    }


def test_pair_matrix_metadata_is_derived_from_dynamic_cohort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pair = {"initial_state_exact": True, "terminal_step_exact": True, "terminal_outcome_exact": True, "traces": {}}
    monkeypatch.setattr(producer, "analyze_pair", lambda *args: pair)
    cohort = (0, 1, 2, 4, 5, 6, 7)
    local = producer.local_pairs(tmp_path, supervisor.IDENTITY, gpu_ids=cohort)
    cross = producer.cross_gpu_pairs(tmp_path, supervisor.IDENTITY, gpu_ids=cohort)
    assert local["gpu_count"] == 7 and local["pair_count"] == 28
    assert sum(len(item["pairs"]) for item in local["gpus"].values()) == 28
    assert cross["gpu_count"] == 7 and cross["pair_count"] == 84
    assert sum(len(item) for item in cross["labels"].values()) == 84


def test_contended_gpu_is_excluded_without_project_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, tuple(range(8)), foreign_gpu=3)
    result = supervisor.gpu_preflight(
        dynamic_cohort=True,
        system_graphics_baseline=json.loads(PROTOCOL.read_text(encoding="utf-8"))["system_graphics_baseline"],
    )
    assert result["status"] == "PASS"
    assert result["primary_clean_gpu_set"] == [0, 1, 2, 4, 5, 6, 7]
    assert result["primary_clean_gpu_count"] == 7
    assert result["gpu_rows"][3]["primary_clean"] is False
    assert "FOREIGN_PROCESS_PRESENT" in result["gpu_rows"][3]["reasons"]
    assert result["foreign_processes_touched"] is False


@pytest.mark.parametrize("gpu_ids, expected_status", [
    ((0, 1, 2, 3), "PASS"),
    ((0, 1, 2), "HOLD_PRIMARY_CLEAN_COHORT_BELOW_MINIMUM"),
])
def test_primary_clean_threshold_is_four(monkeypatch: pytest.MonkeyPatch, gpu_ids: tuple[int, ...], expected_status: str) -> None:
    _smi(monkeypatch, gpu_ids)
    result = supervisor.gpu_preflight(
        gpu_ids=gpu_ids,
        dynamic_cohort=True,
        system_graphics_baseline=json.loads(PROTOCOL.read_text(encoding="utf-8"))["system_graphics_baseline"],
    )
    assert result["status"] == expected_status
    assert result["primary_clean_gpu_count"] == len(gpu_ids)


def test_dynamic_prepare_freezes_cohort_and_rejects_gate_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(supervisor, "_git", lambda *args: "" if args == ("status", "--porcelain") else ("commit" if args == ("rev-parse", "HEAD") else "tree"))
    protocol = supervisor.validate_protocol(PROTOCOL)
    root = tmp_path / "m1-v2-2"
    supervisor.prepare_root(root, protocol, source_commit="commit", source_tree="tree", model_path="model", protocol_path=PROTOCOL)
    clean = {
        "schema": "STAGE_V_M1_V2_2_DYNAMIC_GPU_PREFLIGHT_V1", "status": "PASS",
        "gpu_ids": list(range(8)), "all_8_safe": False,
        "primary_clean_gpu_set": [0, 1, 2, 4, 5, 6, 7], "primary_clean_gpu_count": 7,
        "minimum_primary_clean_gpus": 4, "dynamic_primary_cohort": True,
        "uuid_by_gpu": {}, "gpu_rows": [], "foreign_user_workloads": [],
        "unmapped_processes": [], "baseline_system_graphics": [], "graphics_contract": {},
    }
    monkeypatch.setattr(supervisor, "gpu_preflight", lambda **kwargs: dict(clean))
    supervisor._fresh_preflight(root, protocol, "PRE_CANARY", run_set="canary", protocol_path=PROTOCOL)
    manifest = json.loads((root / "M1_V2_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["primary_clean_gpu_set"] == [0, 1, 2, 4, 5, 6, 7]
    assert manifest["total_r1_runs"] == 28
    drift = dict(clean, primary_clean_gpu_set=[0, 1, 2, 3, 4, 5, 6, 7], primary_clean_gpu_count=8)
    monkeypatch.setattr(supervisor, "gpu_preflight", lambda **kwargs: dict(drift))
    with pytest.raises(supervisor.V2Error, match="HOLD_PRIMARY_CLEAN_COHORT"):
        supervisor._fresh_preflight(root, protocol, "PRE_R1_Q1", run_set="r1", protocol_path=PROTOCOL)
