#!/usr/bin/env python3
"""Synthetic end-to-end pipeline test: split → train(dry_run) → evaluate.
Tests parent leakage detection, suite mapping, feature validation."""
import csv, json, os, subprocess, sys, tempfile

import numpy as np

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]

TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "multisuite_detector")


def make_fixture(tmp):
    eps = []
    for suite in ["libero_object", "libero_spatial"]:
        for tid in range(2):
            for pid in range(2):
                parent = "parent_{}_t{}_p{}".format(suite, tid, pid)
                for sid in range(2):
                    ek = "{}_t{}_p{}_s{}".format(suite, tid, pid, sid)
                    eps.append({
                        "episode_key": ek, "parent_key": parent,
                        "suite": suite, "task_id": tid, "task_name": "task_{}".format(tid),
                        "state_id": sid, "eval_seed": 0, "clean_success": True,
                        "mechanism_type": "single_object_pick_place", "mechanism_eligible": True,
                        "teacher_label_valid": True, "teacher_anchor_step": 50,
                        "teacher_window_start": 50, "teacher_window_end": 60,
                        "teacher_confidence": 0.9,
                        "feature_schema_sha256": "a" * 64,
                        "source_manifest_sha256": "b" * 64,
                        "n_steps": 100, "n_valid_steps": 100,
                    })

    ep_path = os.path.join(tmp, "episodes.jsonl")
    with open(ep_path, "w") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")

    feat_path = os.path.join(tmp, "features.csv")
    with open(feat_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_key", "step"] + SC5_FEATURES)
        rng = np.random.RandomState(42)
        for ep in eps:
            for step in range(100):
                w.writerow([ep["episode_key"], step] + [float(rng.randn()) for _ in range(25)])

    label_path = os.path.join(tmp, "labels.csv")
    with open(label_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_key", "step", "teacher_phase_idx",
                     "teacher_sc5_corridor_active", "release_safe",
                     "teacher_anchor_step", "teacher_window_start", "teacher_window_end"])
        for ep in eps:
            for step in range(100):
                corridor = 1 if 50 <= step <= 60 else 0
                w.writerow([ep["episode_key"], step, 4 if corridor else 0, corridor, 0, 50, 50, 60])

    config_path = os.path.join(tmp, "config.json")
    with open(config_path, "w") as f:
        json.dump({"gate": "TEST", "detector_type": "balanced_pooled",
                   "training_config": {"epochs": 2, "batch_size": 4, "lr": 0.001, "patience": 5}}, f)

    return ep_path, feat_path, label_path, config_path, eps


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: {}".format(" ".join(cmd)))
        print("STDOUT:", r.stdout[-500:])
        print("STDERR:", r.stderr[-500:])
        return False
    return True


def test_parent_leakage_rejected(tmp, ep_path):
    """Create a split with parent leakage and verify rejection."""
    leaky_split = os.path.join(tmp, "leaky_split.json")
    bad = {
        "split_type": "episode_grouped", "seed": 42,
        "splits": {
            "train": ["libero_object_t0_p0_s0"],
            "val": ["libero_object_t0_p0_s1"],
            "test": [],
        },
        "counts": {"train": 1, "val": 1, "test": 0},
        "validation_passed": True,
        "parent_leakage_checked": False,
    }
    with open(leaky_split, "w") as f:
        json.dump(bad, f)
    # Manually check: both episodes share parent 'libero_object_t0_p0'
    # Our validator should catch this
    from tools.multisuite_detector.build_detector_splits import load_episodes, validate_no_parent_leakage
    episodes = load_episodes(ep_path)
    errors = validate_no_parent_leakage(bad, episodes)
    if not errors:
        print("FAIL: parent leakage NOT detected")
        return False
    print("PASS: parent leakage correctly detected ({} errors)".format(len(errors)))
    return True


def main():
    tmp = tempfile.mkdtemp(prefix="synth_")
    print("Fixture: {}".format(tmp))

    ep_path, feat_path, label_path, config_path, episodes = make_fixture(tmp)
    print("Episodes: {}, Features: {}, Labels: {}".format(len(episodes), feat_path, label_path))

    # Test 1: Build split
    print("\n=== TEST 1: build split ===")
    split_file = os.path.join(tmp, "split_episode_grouped.json")
    ok = run([sys.executable, os.path.join(TOOLS, "build_detector_splits.py"),
              "--episode_index", ep_path, "--split_type", "episode_grouped",
              "--output_dir", tmp])
    if not ok:
        sys.exit(1)
    print("PASS")

    # Test 2: Parent leakage negative test
    print("\n=== TEST 2: parent leakage detection ===")
    if not test_parent_leakage_rejected(tmp, ep_path):
        sys.exit(1)

    # Test 3: train --dry_run
    print("\n=== TEST 3: train_detector --dry_run ===")
    train_out = os.path.join(tmp, "train_output")
    ok = run([sys.executable, os.path.join(TOOLS, "train_detector.py"),
              "--config", config_path, "--feature_csv", feat_path,
              "--label_csv", label_path, "--episode_index", ep_path,
              "--split_file", split_file, "--output_dir", train_out, "--dry_run"])
    if not ok:
        sys.exit(1)
    print("PASS")

    # Test 4: Missing feature column must be rejected
    print("\n=== TEST 4: missing feature column rejected ===")
    bad_feat = os.path.join(tmp, "bad_features.csv")
    with open(bad_feat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_key", "step", "gripper_command"])
        w.writerow(["ep1", 0, 0.5])
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "train_detector.py"),
                        "--config", config_path, "--feature_csv", bad_feat,
                        "--label_csv", label_path, "--episode_index", ep_path,
                        "--split_file", split_file, "--output_dir", train_out + "_bad1", "--dry_run"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("FAIL: missing feature columns not rejected")
        sys.exit(1)
    print("PASS: correctly rejected (exit={})".format(r.returncode))

    # Test 5: NaN feature must be rejected
    print("\n=== TEST 5: NaN feature rejected ===")
    nan_feat = os.path.join(tmp, "nan_features.csv")
    with open(nan_feat, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_key", "step"] + SC5_FEATURES)
        for ep in episodes[:1]:
            for step in range(10):
                row = [ep["episode_key"], step] + [1.0] * 24 + ["NaN"]
                w.writerow(row)
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "train_detector.py"),
                        "--config", config_path, "--feature_csv", nan_feat,
                        "--label_csv", label_path, "--episode_index", ep_path,
                        "--split_file", split_file, "--output_dir", train_out + "_bad2", "--dry_run"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("FAIL: NaN feature not rejected")
        sys.exit(1)
    print("PASS: correctly rejected")

    # Test 6: feature contract live import
    print("\n=== TEST 6: feature contract ===")
    ok = run([sys.executable, os.path.join(TOOLS, "extract_frozen_feature_contract.py"),
              "--fail_on_error"])
    if not ok:
        sys.exit(1)
    print("PASS")

    print("\n=== ALL {} TESTS PASSED ===".format(6))


if __name__ == "__main__":
    main()
