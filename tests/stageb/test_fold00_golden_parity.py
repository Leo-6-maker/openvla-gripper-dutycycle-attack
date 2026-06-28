#!/usr/bin/env python3
"""
Fold 00 Golden Parity Test — verifies V3 builder reproduces the historical Fold 00.

Gate: builder V3 must produce identical split membership, teacher thresholds,
corridor/release support counts, and normalization as the frozen Fold 00 artifacts.

Run: pytest tests/stageb/test_fold00_golden_parity.py -v
"""
import json, sys, os, pytest, tempfile, csv, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "stageb"))
from build_sc5_loto_fold_v3 import main as build_fold

BASE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/strict_fold_00_combined500"
PROTOCOL_V2 = str(Path(__file__).resolve().parents[2] / "docs" / "gpu" / "LOTO_10FOLD_PROTOCOL_FREEZE_V2.json")
FEATURE_CSV = BASE + "/FOLD00_FEATURE_DATASET.csv"
FROZEN_LABELS = BASE + "/teacher_labels_fold00_train_val_only.jsonl"
FROZEN_NORM = BASE + "/FOLD00_TRAIN_NORMALIZATION.json"


class TestFold00GoldenParity:

    @pytest.fixture(scope="class")
    def rebuilt(self):
        """Build Fold 00 using V3 builder in a temp directory."""
        tmp = tempfile.mkdtemp(prefix="fold00_parity_")
        build_fold(["--fold", "00", "--output_dir", tmp,
                     "--feature_dataset", FEATURE_CSV,
                     "--protocol", PROTOCOL_V2])
        yield tmp
        import shutil; shutil.rmtree(tmp, ignore_errors=True)

    def test_split_membership_exact(self, rebuilt):
        """Train/val/test episode counts must match frozen Fold 00."""
        with open(rebuilt + "/FOLD00_MANIFEST.json") as f:
            m = json.load(f)
        assert m["train_episodes"] == 400
        assert m["val_episodes"] == 50
        assert m["test_episodes"] == 50
        assert m["test_task"] == 8
        assert m["val_task"] == 6
        assert set(m["train_tasks"]) == {0,1,2,3,4,5,7,9}

    def test_teacher_thresholds_match(self, rebuilt):
        """Thresholds from V3 calibration must approximately match frozen Fold 00."""
        with open(rebuilt + "/FOLD00_teacher_config.json") as f:
            new_tc = json.load(f)
        new_th = new_tc["thresholds"]
        # Frozen Fold 00 thresholds (from Step B audit)
        frozen = {"grasp_open_proxy_max": 0.0639, "eef_obj_dist_max": 0.1515,
                  "lift_z_threshold": 0.0934, "release_target_dist_max": 0.1611,
                  "preplace_target_dist_max": 0.3718}
        for k, expected in frozen.items():
            actual = new_th[k]
            rel_err = abs(actual - expected) / max(abs(expected), 1e-6)
            assert rel_err < 0.01, "%s: new=%.4f frozen=%.4f (rel_err=%.4f)" % (
                k, actual, expected, rel_err)

    def test_label_support_matches(self, rebuilt):
        """Corridor/release support counts must approximately match frozen."""
        with open(rebuilt + "/FOLD00_MANIFEST.json") as f:
            m = json.load(f)
        # Frozen: train corr=14,264, val corr=2,158 (within 1% tolerance)
        assert abs(m["train_stable_carry_rows"] - 14264) / 14264 < 0.01
        assert abs(m["val_stable_carry_rows"] - 2158) / 2158 < 0.01

    def test_normalization_matches(self, rebuilt):
        """Normalization mean/std must approximately match frozen."""
        with open(rebuilt + "/FOLD00_TRAIN_NORMALIZATION.json") as f:
            new_norm = json.load(f)
        with open(FROZEN_NORM) as f:
            frozen = json.load(f)
        for key in frozen["mean"]:
            assert abs(new_norm["mean"][key] - frozen["mean"][key]) < 1e-4, \
                "Mean mismatch for %s: new=%.6f frozen=%.6f" % (
                    key, new_norm["mean"][key], frozen["mean"][key])

    def test_heldout_labels_not_generated(self, rebuilt):
        """Phase A must NOT generate heldout labels."""
        assert not os.path.exists(rebuilt + "/FOLD00_teacher_labels_heldout.jsonl")

    def test_feature_split_has_no_test_rows(self, rebuilt):
        """Train+val CSV must contain 0 test task rows."""
        with open(rebuilt + "/FOLD00_TRAIN_VAL_FEATURE_DATASET.csv") as f:
            reader = csv.DictReader(f)
            for r in reader:
                assert r["split"] in ("train", "val"), \
                    "Found split=%s in train+val CSV" % r["split"]
                assert int(r["task_idx"]) != 8, "Test task 8 in train+val CSV"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
