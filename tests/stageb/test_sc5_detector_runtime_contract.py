import numpy as np
import pytest
import torch

from gripper_attack.sc5_detector_runtime import (
    SC5_FEATURES,
    SC5_PHASES,
    SC5DetectorRuntime,
    SC5MLP,
)


def _write_checkpoint(path, *, split_mode: str):
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    torch.save(
        {
            "feature_names": list(SC5_FEATURES),
            "phase_classes": list(SC5_PHASES),
            "dataset_sha256": "d" * 64,
            "split_mode": split_mode,
            "mean": np.zeros(len(SC5_FEATURES), dtype=np.float32),
            "std": np.ones(len(SC5_FEATURES), dtype=np.float32),
            "model_state": model.state_dict(),
        },
        path,
    )


@pytest.mark.parametrize("split_mode", ["frozen", "provisional_cross_suite_frozen"])
def test_runtime_accepts_frozen_split_variants(tmp_path, split_mode):
    ckpt = tmp_path / f"{split_mode}.pt"
    _write_checkpoint(ckpt, split_mode=split_mode)

    runtime = SC5DetectorRuntime(str(ckpt))
    assert runtime.split_mode == split_mode


def test_runtime_rejects_non_frozen_split(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    _write_checkpoint(ckpt, split_mode="legacy_random")

    with pytest.raises(ValueError, match="split_mode=legacy_random"):
        SC5DetectorRuntime(str(ckpt))
