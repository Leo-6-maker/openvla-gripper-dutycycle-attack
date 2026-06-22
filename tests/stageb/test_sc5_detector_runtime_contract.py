import numpy as np
import pytest
import torch

from gripper_attack.sc5_detector_runtime import (
    SC5_FEATURES,
    SC5_PHASES,
    SC5DetectorRuntime,
    SC5MLP,
)


def _write_checkpoint(path, *, split_mode: str, tau_corridor: float = 0.9, tau_release: float = 0.1):
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    torch.save(
        {
            "feature_names": list(SC5_FEATURES),
            "phase_classes": list(SC5_PHASES),
            "dataset_sha256": "d" * 64,
            "split_mode": split_mode,
            "selected_tau_corridor": tau_corridor,
            "selected_tau_release": tau_release,
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
    assert runtime.threshold_source == "checkpoint"
    assert runtime.tau_c == pytest.approx(0.9)
    assert runtime.tau_r == pytest.approx(0.1)


def test_runtime_rejects_non_frozen_split(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    _write_checkpoint(ckpt, split_mode="legacy_random")

    with pytest.raises(ValueError, match="split_mode=legacy_random"):
        SC5DetectorRuntime(str(ckpt))


def test_runtime_rejects_missing_checkpoint_thresholds(tmp_path):
    ckpt = tmp_path / "missing_tau.pt"
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    torch.save(
        {
            "feature_names": list(SC5_FEATURES),
            "phase_classes": list(SC5_PHASES),
            "dataset_sha256": "d" * 64,
            "split_mode": "provisional_cross_suite_frozen",
            "mean": np.zeros(len(SC5_FEATURES), dtype=np.float32),
            "std": np.ones(len(SC5_FEATURES), dtype=np.float32),
            "model_state": model.state_dict(),
        },
        ckpt,
    )

    with pytest.raises(ValueError, match="Missing checkpoint-selected detector thresholds"):
        SC5DetectorRuntime(str(ckpt))


def test_runtime_rejects_silent_threshold_override(tmp_path):
    ckpt = tmp_path / "override.pt"
    _write_checkpoint(ckpt, split_mode="provisional_cross_suite_frozen")

    with pytest.raises(ValueError, match="threshold override requires allow_threshold_override"):
        SC5DetectorRuntime(str(ckpt), tau_corridor=0.3, tau_release=0.3)


def test_runtime_allows_explicit_diagnostic_threshold_override(tmp_path):
    ckpt = tmp_path / "diagnostic.pt"
    _write_checkpoint(ckpt, split_mode="provisional_cross_suite_frozen")

    runtime = SC5DetectorRuntime(
        str(ckpt),
        tau_corridor=0.3,
        tau_release=0.3,
        allow_threshold_override=True,
        override_reason="diagnostic_zero_emit_reproduction",
    )
    assert runtime.threshold_source == "override"
    assert runtime.override_reason == "diagnostic_zero_emit_reproduction"
    assert runtime.tau_c == pytest.approx(0.3)
    assert runtime.tau_r == pytest.approx(0.3)


def test_runtime_uses_checkpoint_std_without_extra_epsilon(tmp_path):
    ckpt = tmp_path / "std_contract.pt"
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.shared[0].weight[0, 0] = 1.0
        model.shared[2].weight[0, 0] = 1.0
        model.corridor_head.weight[0, 0] = 1.0
        model.phase_head.bias[SC5_PHASES.index("stable_carry")] = 1.0
        model.release_head.bias[0] = -1.0
    mean = np.zeros(len(SC5_FEATURES), dtype=np.float32)
    std = np.ones(len(SC5_FEATURES), dtype=np.float32)
    std[0] = 1e-7
    torch.save(
        {
            "feature_names": list(SC5_FEATURES),
            "phase_classes": list(SC5_PHASES),
            "dataset_sha256": "d" * 64,
            "split_mode": "provisional_cross_suite_frozen",
            "selected_tau_corridor": 0.9,
            "selected_tau_release": 0.1,
            "mean": mean,
            "std": std,
            "model_state": model.state_dict(),
        },
        ckpt,
    )

    runtime = SC5DetectorRuntime(str(ckpt))
    features = {name: 0.0 for name in SC5_FEATURES}
    features[SC5_FEATURES[0]] = 1e-7
    out = runtime.update(features, 0)

    assert out["corridor_p"] == pytest.approx(float(torch.sigmoid(torch.tensor(1.0))), abs=1e-6)
