import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.stageb.build_c2g_matched_load_jobs import deterministic_random_start
from scripts.stageb.run_c2g_matched_load_jobs import (
    array_sha256,
    combined_file_sha256,
    random_start_flag,
)
from src.gripper_attack.c2g_matched_load_manifest import (
    AttackLoadSpec,
    CORE_CONDITIONS,
    deterministic_objective_seed,
    validate_core_2x2_manifest,
)


def jobs():
    parent = "libero_object/task_0/state_0/eval_000"
    load = AttackLoadSpec(
        burst_length=10,
        epsilon=6.0 / 255.0,
        step_size=(6.0 / 255.0) * 0.075,
        pgd_steps=20,
        projection="processor_space_linf_fp32_then_model_cast",
        cast_policy="budget_safe_bf16_or_fp16",
        preprocessing="official_pil_lanczos_center_crop_224",
        image_height=224,
        image_width=224,
        random_start_policy="uniform_linf_seeded",
        temporal_init_policy="prev_delta",
        num_loss_forwards_per_frame=21,
        num_backwards_per_frame=20,
        num_adv_decodes_per_frame=1,
    )
    output = []
    for condition in CORE_CONDITIONS:
        clean = condition == "CLEAN"
        detector = condition.startswith("DET_")
        gripper = "GRIPPER" in condition and "RANDOM" not in condition
        objective = "NONE" if clean else (
            "GRIPPER_TARGETED_VIS_PGD" if gripper else "SHUFFLED_GRIPPER_GRADIENT"
        )
        seed_family = "CLEAN" if clean else objective
        output.append(
            {
                "condition": condition,
                "parent_key": parent,
                "suite": "libero_object",
                "task_index": 0,
                "state_id": 0,
                "eval_seed": 42,
                "clean_parent_sha256": "1" * 64,
                "initial_state_sha256": "2" * 64,
                "detector_checkpoint_sha256": "3" * 64,
                "detector_config_sha256": "4" * 64,
                "timing_source": "NONE" if clean else (
                    "DETECTOR" if detector else "RANDOM_TIME_MATCHED"
                ),
                "objective_family": objective,
                "objective_seed": deterministic_objective_seed(parent, seed_family, 42),
                "attack_enabled": not clean,
                "expected_attacked_frames": 0 if clean else 10,
                "planned_start_step": None if clean else (30 if detector else 60),
                "load_spec": load.__dict__,
            }
        )
    return output


class StrictManifestTests(unittest.TestCase):
    def test_paired_objective_seeds_pass(self):
        summary = validate_core_2x2_manifest(
            jobs(),
            strict_objective_seed_pairing=True,
        )
        self.assertTrue(summary["strict_objective_seed_pairing"])
        parent = summary["parents"][0]
        self.assertEqual(
            len(parent["objective_seed_sets"]["GRIPPER_TARGETED_VIS_PGD"]),
            1,
        )

    def test_unpaired_objective_seed_fails_strict(self):
        rows = jobs()
        for row in rows:
            if row["condition"] == "RANDTIME_GRIPPER_VIS_PGD":
                row["objective_seed"] += 1
        with self.assertRaisesRegex(ValueError, "not paired"):
            validate_core_2x2_manifest(
                rows,
                strict_objective_seed_pairing=True,
            )

    def test_legacy_non_strict_reader_remains_available(self):
        rows = jobs()
        for row in rows:
            if row["condition"] == "RANDTIME_GRIPPER_VIS_PGD":
                row["objective_seed"] += 1
        validate_core_2x2_manifest(rows)

    def test_route_reported_counts_accept_k_plus_one(self):
        load = jobs()[1]["load_spec"]
        self.assertEqual(load["pgd_steps"], 20)
        self.assertEqual(load["num_loss_forwards_per_frame"], 21)
        AttackLoadSpec(**load).validate()


class BindingHelperTests(unittest.TestCase):
    def test_random_start_policy_is_fail_closed(self):
        self.assertEqual(random_start_flag("uniform_linf_seeded"), "--random-start")
        self.assertEqual(random_start_flag("none"), "--no-random-start")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            random_start_flag("sometimes")

    def test_hash_helpers_match_content_and_array_contract(self):
        with tempfile.TemporaryDirectory() as td:
            metadata = Path(td) / "episode_metadata.json"
            steps = Path(td) / "step_records.jsonl"
            metadata.write_text(json.dumps({"a": 1}), encoding="utf-8")
            steps.write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
            digest = combined_file_sha256((metadata, steps))
            self.assertEqual(len(digest), 64)
            self.assertNotEqual(digest, combined_file_sha256((steps, metadata)))
        value = np.asarray([1.0, 2.0], dtype=np.float32)
        self.assertEqual(array_sha256(value), array_sha256(value.copy()))
        self.assertNotEqual(array_sha256(value), array_sha256(value.astype(np.float64)))

    def test_deterministic_random_time_is_not_detector_time(self):
        first = deterministic_random_start(
            "parent",
            minimum=0,
            maximum_inclusive=100,
            detector_start=50,
            master_seed=7,
        )
        second = deterministic_random_start(
            "parent",
            minimum=0,
            maximum_inclusive=100,
            detector_start=50,
            master_seed=7,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, 50)


if __name__ == "__main__":
    unittest.main()
