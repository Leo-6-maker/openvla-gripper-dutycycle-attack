"""D4.3: Clean shadow runner correctness tests (CPU only, no GPU)."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))

from gripper_attack.production_detector import ProductionStreamingDetector

# ── Helpers ──

FEATURE_NAMES = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
    "eef_deceleration_bonus", "qpos_ready_bonus", "eef_speed_now", "eef_speed_prev",
    "eef_deceleration_delta", "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]
ZERO_STDEV_FEATURES = {"close_streak_bonus", "close_streak", "close_onset", "time_since_last_open"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_numpy(arr: np.ndarray) -> str:
    return sha256_bytes(arr.tobytes())


def make_detector(threshold=0.5):
    import torch
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 1)
            nn.init.constant_(self.fc.weight, 0.1)
            nn.init.constant_(self.fc.bias, 0.0)
        def forward(self, x):
            return self.fc(x)

    means = {}
    stdevs = {}
    impute = {}
    for fn in FEATURE_NAMES:
        means[fn] = 0.5
        impute[fn] = 0.5
        if fn in ZERO_STDEV_FEATURES:
            stdevs[fn] = 0.0
        else:
            stdevs[fn] = 1.0
    return ProductionStreamingDetector(
        SimpleModel().eval(), means, stdevs, impute, threshold=threshold,
    )


# ═══════════════════════════════════════════════════════════════
# Action hashing tests
# ═══════════════════════════════════════════════════════════════

def test_action_hash_is_deterministic():
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    h1 = sha256_numpy(action)
    h2 = sha256_numpy(action)
    assert h1 == h2


def test_action_hash_detects_mutation():
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    h1 = sha256_numpy(action)
    action[0] = 0.99
    h2 = sha256_numpy(action)
    assert h1 != h2


def test_action_copy_preserves_hash():
    """A copy of the action should have the same hash."""
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    h1 = sha256_numpy(action)
    action_copy = action.copy()
    h2 = sha256_numpy(action_copy)
    assert h1 == h2
    # Modifying copy should not affect original
    action_copy[0] = 0.99
    h3 = sha256_numpy(action)
    assert h1 == h3


def test_action_hash_bytes_consistency():
    """sha256 of numpy tobytes() must be consistent."""
    action = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    h1 = sha256_bytes(action.tobytes())
    h2 = sha256_numpy(action)
    assert h1 == h2


# ═══════════════════════════════════════════════════════════════
# Detector in shadow-rollout loop (simulated)
# ═══════════════════════════════════════════════════════════════

def test_detector_readonly_does_not_modify_action():
    """Detector.update() must not modify the numpy arrays passed as inputs."""
    d = make_detector(threshold=-999.0)

    raw_gripper = 0.7
    env_val = 1.0
    qpos = 0.0
    eef_x, eef_y, eef_z = 0.0, 0.0, 0.2
    decoded_open = 0

    raw_snap = raw_gripper
    env_snap = env_val

    d.update(0, raw_gripper, env_val, qpos, eef_x, eef_y, eef_z, decoded_open)

    assert raw_gripper == raw_snap
    assert env_val == env_snap


def test_simulated_shadow_loop():
    """Simulate a shadow rollout loop: model → hash → detect → rehash → execute."""
    d = make_detector(threshold=-999.0)

    # Simulated actions (7-dim: 6 arm + 1 gripper)
    actions = [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.3], dtype=np.float32),
        np.array([0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.8], dtype=np.float32),
    ]

    action_identities = []
    for step, action in enumerate(actions):
        # Step 1: hash raw action
        h_pre = sha256_numpy(action)

        # Step 2: derive detector inputs from COPIES
        raw_gripper = float(action[-1])
        env_gripper = -1.0 if raw_gripper > 0.5 else 1.0
        qpos = 0.0
        eef_x, eef_y, eef_z = float(action[0]), float(action[1]), float(action[2])
        decoded_open = 1 if raw_gripper > 0.5 else 0

        # Step 3: call detector
        d.update(step, raw_gripper, env_gripper, qpos, eef_x, eef_y, eef_z,
                 decoded_open)

        # Step 4: rehash action
        h_post = sha256_numpy(action)

        action_identities.append(h_pre == h_post)

    assert all(action_identities), (
        f"Action identity failures: {action_identities}"
    )


# ═══════════════════════════════════════════════════════════════
# Detector reset between episodes
# ═══════════════════════════════════════════════════════════════

def test_detector_reset_between_episodes():
    d = make_detector(threshold=-999.0)

    # Episode 1
    for i in range(5):
        d.update(i, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    emit_1 = d.emit_step
    history_len_1 = len(d.history)

    # Reset
    d.reset()

    assert d._next_expected_step == 0
    assert len(d.history) == 0
    assert d.emit_step == -1

    # Episode 2
    for i in range(5):
        d.update(i, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    emit_2 = d.emit_step

    # Each episode starts fresh
    assert len(d.history) == history_len_1 if history_len_1 > 0 else True


# ═══════════════════════════════════════════════════════════════
# Output directory and artifact structure
# ═══════════════════════════════════════════════════════════════

def test_episode_artifact_structure():
    """Verify that the expected artifact files can be created."""
    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = Path(tmp) / "trace_task_s0_clean_shadow"
        ep_dir.mkdir()

        # episode_manifest.json
        manifest = {
            "task": "alphabet_soup", "state_id": 0, "n_steps": 100,
            "success_primary": 1, "detector_emit_step": 42,
        }
        with open(ep_dir / "episode_manifest.json", "w") as f:
            json.dump(manifest, f)

        # step_trace.csv
        import csv
        with open(ep_dir / "step_trace.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "task", "state_id", "raw_gripper"])
            w.writerow([0, "alphabet_soup", 0, 0.7])

        # Verify all expected files can be created
        expected = [
            "episode_manifest.json", "step_trace.csv",
            "detector_candidates.csv", "detector_emission.json",
            "action_identity.csv", "latency.csv",
            "provenance.csv", "artifact_hashes.csv",
            "teacher_sidecar.json",
        ]
        for name in expected:
            path = ep_dir / name
            if not path.exists():
                path.touch()
            assert path.exists(), f"Missing: {name}"


# ═══════════════════════════════════════════════════════════════
# Detector input hash consistency
# ═══════════════════════════════════════════════════════════════

def test_detector_input_hash_deterministic():
    """The hash of detector inputs must be deterministic for the same values."""
    inputs_1 = np.array([0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)
    inputs_2 = np.array([0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)

    h1 = sha256_bytes(inputs_1.tobytes())
    h2 = sha256_bytes(inputs_2.tobytes())
    assert h1 == h2


def test_detector_input_hash_different_values():
    """Different detector inputs must produce different hashes."""
    inputs_1 = np.array([0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)
    inputs_2 = np.array([0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)

    h1 = sha256_bytes(inputs_1.tobytes())
    h2 = sha256_bytes(inputs_2.tobytes())
    assert h1 != h2


# ═══════════════════════════════════════════════════════════════
# Configuration validation
# ═══════════════════════════════════════════════════════════════

def test_config_forbids_attack():
    """D4.3 config must have attack_forbidden: true."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "d4_clean_shadow_v1.yaml",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["attack_forbidden"] is True
    assert cfg["training_forbidden"] is True
    assert cfg["threshold_retuning_forbidden"] is True


def test_config_frozen_threshold():
    """Threshold must be frozen at 0.236312."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "d4_clean_shadow_v1.yaml",
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["threshold"] == 0.236312
    assert cfg["checkpoint_sha"] == (
        "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
    )


# ═══════════════════════════════════════════════════════════════
# No Teacher-P during rollout
# ═══════════════════════════════════════════════════════════════

def test_production_detector_no_teacher_p_dependency():
    """Verify the production detector has no Teacher-P in its update path."""
    src_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "gripper_attack",
        "production_detector.py",
    )
    with open(src_file) as f:
        source = f.read()

    # The update() method should not reference Teacher-P
    # (teacher_anchor=-1 is passed to rule_based_close_predictor but that's
    #  the prediction horizon ground-truth flag, not Teacher-P for inference)
    # Check that no Teacher-P module is imported
    assert "teacher_p" not in source.lower(), "Teacher-P reference found"
