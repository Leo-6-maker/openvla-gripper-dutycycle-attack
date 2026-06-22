#!/usr/bin/env python3
"""F0: Freeze recovered Object checkpoint contract — verify and document."""
import hashlib, json, os, sys, numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_PATH = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
sys.path.insert(0, os.path.join(REPO, "src"))
from gripper_attack.sc5_detector_runtime import SC5_FEATURES, SC5_PHASES


def main():
    # Compute file SHA
    sha = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()
    size = os.path.getsize(CKPT_PATH)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"SHA256: {sha}")
    print(f"Size: {size} bytes")

    # Strict load
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    # Dataset SHA
    ds_sha = ckpt.get("dataset_sha256", "?")
    print(f"Dataset SHA256: {ds_sha}")

    # Model state dict
    state = ckpt["model_state"]
    print("\nState dict:")
    for k, v in state.items():
        print(f"  {k}: {tuple(v.shape)}")
    n_params = sum(v.numel() for v in state.values())
    print(f"Total params: {n_params}")
    assert n_params == 6604, f"Expected 6604 params, got {n_params}"

    # Verify confidence_head exists
    assert "confidence_head.weight" in state, "Missing confidence_head.weight"
    assert "confidence_head.bias" in state, "Missing confidence_head.bias"
    print("confidence_head: PRESENT (preserved, not stripped)")

    # Feature order
    feats = ckpt.get("feature_names", [])
    assert len(feats) == 25, f"Expected 25 features, got {len(feats)}"
    assert feats == SC5_FEATURES, "Feature order mismatch"
    print(f"Feature order: MATCH ({len(feats)}D)")

    # Phase classes
    phases = ckpt.get("phase_classes", [])
    assert len(phases) == 9, f"Expected 9 phases, got {len(phases)}"
    assert phases == SC5_PHASES, "Phase order mismatch"
    print(f"Phase classes: MATCH ({len(phases)})")

    # Normalization
    mean_np = np.asarray(ckpt["mean"])
    std_np = np.asarray(ckpt["std"])
    assert mean_np.shape == (25,), f"mean shape {mean_np.shape}"
    assert std_np.shape == (25,), f"std shape {std_np.shape}"
    assert np.isfinite(mean_np).all(), "mean has NaN/Inf"
    assert np.isfinite(std_np).all(), "std has NaN/Inf"
    assert (std_np > 0).all(), "std has non-positive values"
    print("Normalization: PASS (mean/std 25D, finite, positive)")

    # Split
    print(f"Split mode: {ckpt.get('split_mode','?')}")
    print(f"Train rows: {ckpt.get('n_train','?')}")
    print(f"Val rows: {ckpt.get('n_val','?')}")

    # Trigger params
    print(f"Default thresholds: tau_c=0.3 tau_r=0.3 guard=5 K=10")

    # Strict load verification
    import copy
    sys.path.insert(0, os.path.join(REPO, "src"))
    from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime
    try:
        detector = SC5DetectorRuntime(CKPT_PATH, tau_corridor=0.3, tau_release=0.3, guard=5)
        print(f"SC5DetectorRuntime strict load: PASS")
        print(f"  checkpoint_sha256={detector.checkpoint_sha256[:16]}...")
        print(f"  dataset_sha256={detector.dataset_sha256[:16]}...")
    except Exception as e:
        print(f"Detector load: FAIL — {e}")
        return 1

    print("\nF0_OBJECT_ARTIFACT_FREEZE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
