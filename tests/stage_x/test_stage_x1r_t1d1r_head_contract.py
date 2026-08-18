from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"
CONTRACT = ROOT / "configs/STAGE_X_X1R_T1D1R_STUDENT_HEAD_CONTRACT_V1.json"
BASE_RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_t1d1_screening_clean.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("stage_x_t1d1_base_test", BASE_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _student_class():
    torch = pytest.importorskip("torch")
    del torch
    sys.path.insert(0, str(ROOT / "n5/phase3_student"))
    from n5_student_model import N5MultiHeadStudent

    return N5MultiHeadStudent


def test_runtime_contract_matches_tracked_student_source():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    Student = _student_class()
    assert list(Student.HEAD_NAMES) == contract["runtime_output_keys"]
    assert contract["historical_semantic_aliases"] == {"k10_feasibility": "k10_feasible"}


def test_frozen_25d_forward_returns_exact_runtime_keys():
    torch = pytest.importorskip("torch")
    Student = _student_class()
    model = Student(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 6, 25), dtype=torch.float32), timestep_mask=torch.ones((1, 6), dtype=torch.bool))
    assert set(output) == set(Student.HEAD_NAMES)
    assert all(value.shape == (1, 6) for value in output.values())


def test_d1r_student_trace_executes_without_missing_key_access():
    torch = pytest.importorskip("torch")
    Student = _student_class()
    base = _load_base()
    model = Student(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.eval()
    features = [[0.01 * (row + column) for column in range(25)] for row in range(5)]
    result = base.student_trace(model, features, mean=__import__("numpy").zeros(25, dtype="float32"), std=__import__("numpy").ones(25, dtype="float32"))
    assert len(result) == 5
    assert all(set(row) == set(Student.HEAD_NAMES) for row in result)
    del torch


def test_scheduler_scope_excludes_non_gate_heads():
    base = _load_base()
    source = inspect.getsource(base.schedule)
    assert all(name not in source for name in ("k10_feasible", "k10_feasibility", "safe_release", "instability"))
    assert all(name in source for name in ("candidate_close", "legal", "physical_criticality", "gripper_closing_state", "emitted"))


def test_frozen_checkpoint_strict_load_when_available():
    torch = pytest.importorskip("torch")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checkpoint = Path(protocol["student"]["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip("official frozen checkpoint is server-bound; D1R CPU audit verifies it")
    Student = _student_class()
    payload = torch.load(checkpoint, map_location="cpu")
    model = Student(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.load_state_dict(payload["model"], strict=True)


def test_d1r_preserves_consumed_canaries_and_exact_continuation_set():
    hold = json.loads((ROOT / "reports/STAGE_X_X1R_T1D1_CANARY_RUNTIME_HOLD_V1.json").read_text(encoding="utf-8"))
    ledger = json.loads((ROOT / "reports/STAGE_X_X1R_T1D1R_CONTINUATION_LEDGER_V1.json").read_text(encoding="utf-8"))
    assert hold["status"] == "HOLD_RUNTIME_INVALID_AFTER_FIRST_POLICY_DECISION"
    assert sorted(row["ordinal"] for row in hold["canaries"]) == [1, 11, 20, 30]
    assert all(row["retry_eligible"] is False for row in hold["canaries"])
    assert len(ledger["rows"]) == 35
    assert not {1, 11, 20, 30}.intersection(row["ordinal"] for row in ledger["rows"])
    assert ledger["repair_canary_ordinal"] == 2
    assert ledger["replacement"] is False
    assert ledger["rerank"] is False
