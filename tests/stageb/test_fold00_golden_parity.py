#!/usr/bin/env python3
"""
Fold 00 Golden Parity Test — verifies V3 builder reproduces frozen Fold 00.

Tests: argv-callable builder, split membership exact, per-row phase parity,
       corridor/release support exact, teacher config exact, normalization
       mean+std parity, heldout labels absent, test task not in train+val CSV.

Requires: pytest, server access to frozen artifacts.
"""
import json, sys, os, pytest, tempfile, csv, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "stageb"))
from build_sc5_loto_fold_v3 import main as build_fold
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.v2_privileged_teacher import (
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
)

BASE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/strict_fold_00_combined500"
PROTOCOL_V2 = str(Path(__file__).resolve().parents[2] / "docs" / "gpu" / "LOTO_10FOLD_PROTOCOL_FREEZE_V2.json")
FEATURE_CSV = BASE + "/FOLD00_FEATURE_DATASET.csv"
FROZEN_LABELS_PATH = BASE + "/teacher_labels_fold00_train_val_only.jsonl"
FROZEN_NORM = BASE + "/FOLD00_TRAIN_NORMALIZATION.json"
FROZEN_TC = BASE + "/TEACHER_CONFIG_FOLD00_COMBINED500.json"

SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]
K_SC5 = 10; GUARD_SC5 = 5


def load_frozen_labels():
    labels = {}
    with open(FROZEN_LABELS_PATH) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line)
            labels[(lab["task_idx"], lab["state_id"], lab["step_idx"])] = lab
    return labels


def load_rebuilt_labels(rebuilt_dir):
    labels = {}
    path = rebuilt_dir + "/FOLD00_teacher_labels_train_val.jsonl"
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line)
            labels[(lab["task_idx"], lab["state_id"], lab["step_idx"])] = lab
    return labels


def compute_corridor_release_support(labels_dict):
    """Compute corridor-positive and release-positive counts using trainer logic."""
    # Group by episode
    from collections import defaultdict
    eps = defaultdict(list)
    for (t, s, step), lab in sorted(labels_dict.items()):
        eps[(t, s)].append(lab)
    for k in eps:
        eps[k] = sorted(eps[k], key=lambda x: x["step_idx"])

    corr_pos = 0; corr_neg = 0; rel_pos = 0; rel_neg = 0
    for ep_key, ep_labels in eps.items():
        for lab in ep_labels:
            if lab["phase"] == "release_safe": rel_pos += 1
            else: rel_neg += 1

        sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD_SC5)
        if sc5["valid"]:
            corridor_info = compute_sc5_valid_start_corridor(ep_labels, sc5["anchor"], K=K_SC5)
            corridor_active = corridor_info["corridor_active_at_t"]
        else:
            corridor_active = set()

        for lab in ep_labels:
            if lab["step_idx"] in corridor_active: corr_pos += 1
            else: corr_neg += 1

    return {"corridor_pos": corr_pos, "corridor_neg": corr_neg,
            "release_pos": rel_pos, "release_neg": rel_neg}


class TestFold00GoldenParity:

    @pytest.fixture(scope="class")
    def rebuilt(self):
        tmp = tempfile.mkdtemp(prefix="fold00_parity_")
        build_fold(["--fold", "00", "--output_dir", tmp,
                     "--feature_dataset", FEATURE_CSV,
                     "--protocol", PROTOCOL_V2])
        yield tmp
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

    @pytest.fixture(scope="class")
    def frozen_labels(self):
        return load_frozen_labels()

    @pytest.fixture(scope="class")
    def rebuilt_labels(self, rebuilt):
        return load_rebuilt_labels(rebuilt)

    # ── 1. Split membership ──
    def test_split_counts(self, rebuilt):
        with open(rebuilt + "/FOLD00_MANIFEST.json") as f:
            m = json.load(f)
        assert m["train_episodes"] == 400
        assert m["val_episodes"] == 50
        assert m["test_episodes"] == 50
        assert m["test_task"] == 8
        assert m["val_task"] == 6
        assert set(m["train_tasks"]) == {0,1,2,3,4,5,7,9}

    def test_train_episodes_exact_set(self, rebuilt):
        """Train must have all 400 episodes: 8 tasks × 50 states. Val = task 6 × 50."""
        train_eps = set(); val_eps = set()
        with open(rebuilt + "/FOLD00_TRAIN_VAL_FEATURE_DATASET.csv") as f:
            for r in csv.DictReader(f):
                key = (int(r["task_idx"]), int(r["state_id"]))
                if r["split"] == "train": train_eps.add(key)
                elif r["split"] == "val": val_eps.add(key)
        expected_train = set()
        for t in [0,1,2,3,4,5,7,9]:
            for s in range(50):
                expected_train.add((t, s))
        expected_val = set((6, s) for s in range(50))
        assert train_eps == expected_train, "Train episode set mismatch"
        assert val_eps == expected_val, "Val episode set mismatch"

    # ── 2. Per-row phase parity ──
    def test_label_key_sets_equal(self, frozen_labels, rebuilt_labels):
        f_keys = set(frozen_labels.keys())
        r_keys = set(rebuilt_labels.keys())
        assert f_keys == r_keys, "extra=%d missing=%d" % (
            len(r_keys - f_keys), len(f_keys - r_keys))

    def test_per_step_phase_identical(self, frozen_labels, rebuilt_labels):
        mismatches = 0
        for key in frozen_labels:
            fp = frozen_labels[key]["phase"]
            rp = rebuilt_labels[key]["phase"]
            if fp != rp:
                mismatches += 1
                if mismatches <= 5:
                    print("  MISMATCH %s: frozen=%s rebuilt=%s" % (str(key), fp, rp))
        assert mismatches == 0, "%d phase mismatches" % mismatches

    # ── 3. Corridor + release support exact ──
    def test_corridor_support_exact(self, frozen_labels, rebuilt_labels):
        f_sup = compute_corridor_release_support(frozen_labels)
        r_sup = compute_corridor_release_support(rebuilt_labels)
        assert r_sup["corridor_pos"] == f_sup["corridor_pos"], \
            "corridor_pos: rebuilt=%d frozen=%d" % (r_sup["corridor_pos"], f_sup["corridor_pos"])
        assert r_sup["corridor_neg"] == f_sup["corridor_neg"]
        print("  corridor_pos=%d (PASS)" % r_sup["corridor_pos"])

    def test_release_support_exact(self, frozen_labels, rebuilt_labels):
        f_sup = compute_corridor_release_support(frozen_labels)
        r_sup = compute_corridor_release_support(rebuilt_labels)
        assert r_sup["release_pos"] == f_sup["release_pos"], \
            "release_pos: rebuilt=%d frozen=%d" % (r_sup["release_pos"], f_sup["release_pos"])
        assert r_sup["release_neg"] == f_sup["release_neg"]
        print("  release_pos=%d (PASS)" % r_sup["release_pos"])

    # ── 4. Teacher config exact ──
    def test_teacher_config_full_parity(self, rebuilt):
        with open(rebuilt + "/FOLD00_teacher_config.json") as f:
            new_tc = json.load(f)
        # Compare all thresholds to known Fold 00 frozen values
        frozen_th = {
            "grasp_close_sustain": 3, "grasp_open_proxy_max": 0.0639,
            "eef_obj_dist_max": 0.1515, "eef_obj_dist_stable_var": 0.005,
            "lift_z_threshold": 0.0934, "lift_sustain_steps": 2,
            "carry_obj_z_var_max": 0.01, "carry_window": 8,
            "preplace_target_dist_min": 0.05, "preplace_target_dist_max": 0.3718,
            "release_target_dist_max": 0.1611, "regrasp_eef_obj_dist_max": 0.1,
            "stability_window": 5,
        }
        for k, expected in frozen_th.items():
            actual = new_tc["thresholds"][k]
            abs_err = abs(actual - expected)
            assert abs_err < 0.01, "%s: rebuilt=%.4f frozen=%.4f (abs_err=%.4f)" % (
                k, actual, expected, abs_err)

    # ── 5. Normalization mean + std parity ──
    def test_normalization_mean_and_std(self, rebuilt):
        with open(rebuilt + "/FOLD00_TRAIN_NORMALIZATION.json") as f:
            new_norm = json.load(f)
        with open(FROZEN_NORM) as f:
            frozen = json.load(f)
        for key in frozen["mean"]:
            assert abs(new_norm["mean"][key] - frozen["mean"][key]) < 1e-4, \
                "Mean mismatch: %s" % key
            assert abs(new_norm["std"][key] - frozen["std"][key]) < 1e-4, \
                "Std mismatch: %s" % key

    # ── 6. Heldout labels NOT generated ──
    def test_no_heldout_labels(self, rebuilt):
        assert not os.path.exists(rebuilt + "/FOLD00_teacher_labels_heldout.jsonl")

    # ── 7. No test task in train+val ──
    def test_no_test_task_in_train_val(self, rebuilt):
        with open(rebuilt + "/FOLD00_TRAIN_VAL_FEATURE_DATASET.csv") as f:
            for r in csv.DictReader(f):
                assert int(r["task_idx"]) != 8
                assert r["split"] in ("train", "val")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
