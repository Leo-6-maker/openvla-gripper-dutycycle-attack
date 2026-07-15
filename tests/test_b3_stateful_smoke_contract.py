from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from scripts.detector.run_b3_stateful_gpu_smoke import load_smoke_config  # noqa: E402


def test_engineering_smoke_uses_frozen_json_contract():
    config = load_smoke_config(
        Path("configs/B3_STATEFUL_ENGINEERING_SMOKE_V1.json")
    )
    assert config["padding_total_steps"] in config["sequence_lengths"]
    assert config["checkpoint_test_length"] in config["sequence_lengths"]
    assert config["uses_official_teacher_labels"] is False
    assert config["formal_training_ready"] is False
