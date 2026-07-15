import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("torch")


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "detector" / "run_b3_stateful_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("b3_stateful_gpu_smoke", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engineering_smoke_uses_frozen_json_contract():
    root = Path(__file__).parents[1]
    config = _load_runner().load_smoke_config(
        root / "configs" / "B3_STATEFUL_ENGINEERING_SMOKE_V1.json"
    )
    assert config["padding_total_steps"] in config["sequence_lengths"]
    assert config["checkpoint_test_length"] in config["sequence_lengths"]
    assert config["uses_official_teacher_labels"] is False
    assert config["formal_training_ready"] is False
