import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5MLP, SC5_FEATURES, SC5_PHASES  # noqa: E402


def write_checkpoint(path: Path, split_mode: str):
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": list(SC5_FEATURES),
            "phase_classes": list(SC5_PHASES),
            "dataset_sha256": "abc",
            "mean": np.zeros(len(SC5_FEATURES), dtype=np.float32),
            "std": np.ones(len(SC5_FEATURES), dtype=np.float32),
            "split_mode": split_mode,
        },
        path,
    )


def test_detector_runtime_rejects_provisional_split_by_default(tmp_path):
    ckpt = tmp_path / "model.pt"
    write_checkpoint(ckpt, "provisional_cross_suite_frozen")
    try:
        SC5DetectorRuntime(str(ckpt))
        assert False, "expected split_mode rejection"
    except ValueError as exc:
        assert "provisional_cross_suite_frozen" in str(exc)


def test_detector_runtime_accepts_provisional_split_when_explicit(tmp_path):
    ckpt = tmp_path / "model.pt"
    write_checkpoint(ckpt, "provisional_cross_suite_frozen")
    runtime = SC5DetectorRuntime(str(ckpt), allowed_split_modes=("frozen", "provisional_cross_suite_frozen"))
    assert runtime.dataset_sha256 == "abc"

