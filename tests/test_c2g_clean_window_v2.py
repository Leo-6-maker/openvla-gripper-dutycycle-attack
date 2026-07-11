import copy
import unittest
from dataclasses import asdict

try:
    import torch as _torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _torch = None
    TORCH_AVAILABLE = False

from src.gripper_attack.c2g_clean_policy_signals import (
    CLEAN_POLICY_FEATURE_NAMES,
    clean_policy_feature_tensor,
    summarize_clean_gripper_logits,
)
from src.gripper_attack.c2g_clean_window_schema import (
    CLEAN_TEACHER_SCHEMA_VERSION,
    assert_clean_student_feature_names,
    validate_clean_teacher_row,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    FixedBurstTriggerScheduler,
    HEAD_NAMES,
    SchedulerState,
    clean_window_loss,
)
from src.gripper_attack.c2g_matched_load_manifest import (
    AttackLoadSpec,
    CORE_CONDITIONS,
    deterministic_objective_seed,
    validate_core_2x2_manifest,
)
from tools.multisuite_detector.c2g_clean_dataset_adapter import (
    assert_student_feature_payload,
    clean_window_split_coverage,
    derive_episode_fully_known_negative,
    teacher_row_to_model_targets,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)


def valid_teacher_row():
    return {
        "teacher_schema_version": CLEAN_TEACHER_SCHEMA_VERSION,
        "episode_key": "libero_object/task_0/ep_0",
        "step": 10,
        "suite": "libero_object",
        "task_index": 0,
        "mechanism_type": "pick_place_transfer",
        "mechanism_eligible": True,
        "teacher_phase": "TRANSPORT",
        "teacher_reason_code": "TARGET_CRITICAL_WINDOW",
        "teacher_confidence": 0.9,
        "grounding_confidence": 1.0,
        "teacher_known": True,
        "label_known_mask": True,
        "resolved_target_objects": ["milk"],
        "resolved_target_manipulable_entities": [],
        "contacted_entities": ["milk"],
        "uses_privileged_sim_state": True,
        "uses_attack_outcome": False,
        "uses_future_student_input": False,
        "y_target_relevant": True,
        "y_contact_or_grasp_stable": True,
        "y_gripper_dependency": True,
        "y_clean_close_intent": True,
        "y_lift_transport_or_constraint": True,
        "y_release_safe": False,
        "y_gripper_critical_window": True,
        "y_burst_feasible": False,
        "y_attack_start_b": False,
    }


def clean_metadata(mechanism="pick_place_transfer"):
    return {
        "episode_key": "libero_object/task_0/ep_0",
        "suite": "libero_object",
        "task_index": 0,
        "mechanism_type": mechanism,
        "object_declarations": ["milk", "ketchup"],
        "receptacle_declarations": ["basket"],
        "structured_goal_metadata": {
            "target_objects": ["milk"],
            "target_receptacles": ["basket"],
        },
        "gripper_command_semantics": "positive_is_close",
    }


def target_contact(entity="milk"):
    return [
        ["robot0_left_finger_collision", f"{entity}_collision"],
        ["robot0_right_finger_collision", f"{entity}_collision"],
    ]


def positive_rows(count=5):
    return [
        {
            "step": step,
            "contact_pairs": target_contact("milk"),
            "gripper_command": 1.0,
            "object_relative_lift": 0.03,
            "near_target": False,
        }
        for step in range(count)
    ]


class CleanWindowSchemaTests(unittest.TestCase):
    def test_valid_known_row(self):
        validate_clean_teacher_row(valid_teacher_row())

    def test_critical_label_must_equal_clean_conjunction(self):
        row = valid_teacher_row()
        row["y_clean_close_intent"] = False
        with self.assertRaisesRegex(ValueError, "frozen clean-only conjunction"):
            validate_clean_teacher_row(row)

    def test_unknown_rows_require_null_labels(self):
        row = valid_teacher_row()
        row.update(
            teacher_phase="TARGET_UNRESOLVED",
            teacher_reason_code="TARGET_UNRESOLVED",
            teacher_known=False,
            label_known_mask=False,
            resolved_target_objects=[],
            contacted_entities=[],
        )
        for key in [key for key in row if key.startswith("y_")]:
            row[key] = None
        validate_clean_teacher_row(row)
        row["y_target_relevant"] = False
        with self.assertRaisesRegex(ValueError, "null labels"):
            validate_clean_teacher_row(row)

    def test_release_safe_veto(self):
        row = valid_teacher_row()
        row.update(
            teacher_phase="RELEASE_SAFE",
            teacher_reason_code="TARGET_RELEASE_SAFE",
            y_release_safe=True,
            y_gripper_critical_window=False,
        )
        validate_clean_teacher_row(row)
        row["y_gripper_critical_window"] = True
        with self.assertRaises(ValueError):
            validate_clean_teacher_row(row)

    def test_attack_outcome_field_rejected(self):
        row = valid_teacher_row()
        row["vis_success"] = True
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_clean_teacher_row(row)

    def test_student_feature_guard(self):
        assert_clean_student_feature_names(
            ["gripper_qpos", "clean_open_probability_mass", "siglip_feature_0"]
        )
        for bad in (
            ["task_index"],
            ["normalized_step"],
            ["teacher_phase"],
            ["y_gripper_critical_window"],
            ["qpos_delta_after_attack"],
        ):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                assert_clean_student_feature_names(bad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class CleanPolicySignalTests(unittest.TestCase):
    def test_feature_order_and_open_close_mass(self):
        logits = _torch.tensor([[0.0, 2.0, -1.0, 1.0]])
        summary = summarize_clean_gripper_logits(
            logits,
            open_token_ids=[1],
            close_token_ids=[3],
        )
        self.assertEqual(set(summary), set(CLEAN_POLICY_FEATURE_NAMES))
        self.assertGreater(
            float(summary["clean_open_probability_mass"]),
            float(summary["clean_close_probability_mass"]),
        )
        self.assertGreater(float(summary["clean_open_minus_close_log_mass"]), 0.0)
        self.assertEqual(float(summary["clean_top1_is_open"]), 1.0)
        stacked = clean_policy_feature_tensor(
            logits,
            open_token_ids=[1],
            close_token_ids=[3],
        )
        self.assertEqual(stacked.shape, (1, len(CLEAN_POLICY_FEATURE_NAMES)))

    def test_sequence_shape(self):
        logits = _torch.zeros(2, 3, 8)
        features = clean_policy_feature_tensor(
            logits, open_token_ids=[6, 7], close_token_ids=[0, 1]
        )
        self.assertEqual(features.shape, (2, 3, len(CLEAN_POLICY_FEATURE_NAMES)))
        self.assertTrue(_torch.isfinite(features).all())

    def test_token_semantics_fail_closed(self):
        logits = _torch.zeros(1, 4)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            summarize_clean_gripper_logits(logits, open_token_ids=[1], close_token_ids=[1])
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_clean_gripper_logits(logits, open_token_ids=[], close_token_ids=[1])
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            summarize_clean_gripper_logits(logits, open_token_ids=[4], close_token_ids=[1])


@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class CriticalWindowDetectorTests(unittest.TestCase):
    def test_full_model_is_causal_and_has_clean_heads(self):
        config = C2gDetectorConfig(
            visual_dim=8,
            language_dim=6,
            policy_intent_dim=9,
            hidden=12,
            dropout=0.0,
        )
        model = C2gGripperCriticalWindowDetector(config).eval()
        x1 = _torch.zeros(2, 5, 25)
        x2 = x1.clone()
        x2[:, -1] = 10.0
        policy = _torch.zeros(2, 5, 9)
        visual = _torch.zeros(2, 5, 8)
        language = _torch.zeros(2, 6)
        with _torch.no_grad():
            y1 = model(
                x1,
                language,
                policy_intent=policy,
                siglip_visual=visual,
                return_sequence=True,
            )
            y2 = model(
                x2,
                language,
                policy_intent=policy,
                siglip_visual=visual,
                return_sequence=True,
            )
        self.assertEqual(set(y1), set(HEAD_NAMES))
        self.assertEqual(y1["critical_window"].shape, (2, 5))
        _torch.testing.assert_close(
            y1["critical_window"][:, :-1], y2["critical_window"][:, :-1]
        )

    def test_ablation_modes_and_patch_attention(self):
        temporal_only = C2gGripperCriticalWindowDetector(
            C2gDetectorConfig(
                visual_dim=8,
                language_dim=6,
                hidden=10,
                dropout=0.0,
                use_policy_intent=False,
                use_visual=False,
                use_language_conditioning=False,
            )
        )
        output = temporal_only(_torch.zeros(1, 4, 25), _torch.zeros(1, 6))
        self.assertEqual(output["critical_window"].shape, (1,))
        with self.assertRaisesRegex(ValueError, "use_visual=false"):
            temporal_only(
                _torch.zeros(1, 4, 25),
                _torch.zeros(1, 6),
                siglip_visual=_torch.zeros(1, 8),
            )

        patch_model = C2gGripperCriticalWindowDetector(
            C2gDetectorConfig(
                visual_dim=8,
                language_dim=6,
                policy_intent_dim=9,
                hidden=10,
                dropout=0.0,
                patch_dim=7,
            )
        )
        output = patch_model(
            _torch.zeros(1, 4, 25),
            _torch.zeros(1, 6),
            policy_intent=_torch.zeros(1, 4, 9),
            patch_tokens=_torch.zeros(1, 4, 3, 7),
            patch_token_mask=_torch.ones(1, 4, 3, dtype=_torch.bool),
            return_sequence=True,
        )
        self.assertEqual(output["grounding_confidence"].shape, (1, 4))

    def test_clean_window_loss_rejects_outcome_targets_and_is_finite(self):
        outputs = {name: _torch.zeros(2, 4) for name in HEAD_NAMES}
        targets = {name: _torch.zeros(2, 4) for name in HEAD_NAMES}
        masks = {name: _torch.ones(2, 4, dtype=_torch.bool) for name in HEAD_NAMES}
        targets["critical_window"][0, 2:] = 1
        targets["window_active"][0, 2:] = 1
        targets["window_start"][0, 2] = 1
        masks["episode_fully_known_negative"] = _torch.tensor([False, True])
        loss = clean_window_loss(outputs, targets, masks)
        self.assertTrue(_torch.isfinite(loss["total"]))
        leaked = dict(targets)
        leaked["y_cmdopen_vulnerable"] = _torch.zeros(2, 4)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            clean_window_loss(outputs, leaked, masks)

    def test_scheduler_starts_on_two_of_three_and_emits_exact_fixed_burst(self):
        scheduler = FixedBurstTriggerScheduler(
            burst_length=4,
            tau_critical=0.7,
            tau_release=0.5,
            tau_ground=0.6,
        )
        decisions = []
        for critical in (0.8, 0.2, 0.9, 0.0, 0.0, 0.0):
            decisions.append(
                scheduler.update(
                    critical_probability=critical,
                    release_safe_probability=0.1,
                    grounding_confidence_probability=0.9,
                )
            )
        self.assertTrue(decisions[2].trigger_started)
        self.assertEqual([d.attack_active for d in decisions].count(True), 4)
        self.assertEqual(
            [d.attack_index for d in decisions if d.attack_active], [0, 1, 2, 3]
        )
        self.assertEqual(decisions[-1].state, SchedulerState.DONE)

    def test_scheduler_veto_before_start_does_not_shorten_burst(self):
        scheduler = FixedBurstTriggerScheduler(
            burst_length=2,
            tau_critical=0.5,
            tau_release=0.5,
            tau_ground=0.5,
        )
        for release, ground in ((0.9, 0.9), (0.1, 0.1), (0.1, 0.9)):
            decision = scheduler.update(
                critical_probability=0.9,
                release_safe_probability=release,
                grounding_confidence_probability=ground,
            )
        self.assertFalse(decision.attack_active)
        decision = scheduler.update(
            critical_probability=0.9,
            release_safe_probability=0.1,
            grounding_confidence_probability=0.9,
        )
        self.assertTrue(decision.trigger_started)
        decision2 = scheduler.update(
            critical_probability=0.0,
            release_safe_probability=1.0,
            grounding_confidence_probability=0.0,
        )
        self.assertTrue(decision2.attack_active)
        self.assertEqual(decision2.state, SchedulerState.DONE)


class CleanWindowLabelBuilderTests(unittest.TestCase):
    def test_target_transport_builds_fixed_burst_start(self):
        rows = build_clean_teacher_episode(
            positive_rows(5),
            clean_metadata(),
            thresholds=CleanTeacherThresholds(burst_length=3),
        )
        self.assertTrue(all(row["label_known_mask"] for row in rows))
        self.assertTrue(all(row["y_gripper_critical_window"] for row in rows))
        self.assertEqual(sum(row["y_burst_feasible"] for row in rows), 3)
        self.assertEqual(sum(row["y_attack_start_b"] for row in rows), 1)

    def test_release_safe_vetoes_critical_window(self):
        rows = positive_rows(3)
        for row in rows:
            row["near_target"] = True
            row["supported_at_target"] = True
        labels = build_clean_teacher_episode(
            rows,
            clean_metadata(),
            thresholds=CleanTeacherThresholds(burst_length=2),
        )
        self.assertTrue(all(row["y_release_safe"] for row in labels))
        self.assertTrue(all(not row["y_gripper_critical_window"] for row in labels))

    def test_distractor_contact_is_known_negative_not_unknown(self):
        rows = positive_rows(3)
        for row in rows:
            row["contact_pairs"] = target_contact("ketchup")
        labels = build_clean_teacher_episode(
            rows,
            clean_metadata(),
            thresholds=CleanTeacherThresholds(burst_length=2),
        )
        self.assertTrue(all(row["label_known_mask"] for row in labels))
        self.assertTrue(
            all(row["teacher_reason_code"] == "DISTRACTOR_CONTACT" for row in labels)
        )
        self.assertTrue(all(not row["y_gripper_critical_window"] for row in labels))

    def test_absolute_eef_z_is_not_progress_evidence(self):
        rows = [
            {
                "step": step,
                "contact_pairs": target_contact("milk"),
                "gripper_command": 1.0,
                "eef_z": 1.2,
                "near_target": False,
            }
            for step in range(3)
        ]
        labels = build_clean_teacher_episode(rows, clean_metadata())
        self.assertTrue(all(not row["label_known_mask"] for row in labels))
        self.assertTrue(
            all(
                row["teacher_reason_code"] == "PROGRESS_SEMANTICS_UNRESOLVED"
                for row in labels
            )
        )

    def test_unknown_command_polarity_stays_unknown(self):
        meta = clean_metadata()
        meta.pop("gripper_command_semantics")
        labels = build_clean_teacher_episode(positive_rows(3), meta)
        self.assertTrue(all(not row["label_known_mask"] for row in labels))
        self.assertTrue(
            all(
                row["teacher_reason_code"] == "CLOSE_SEMANTICS_UNRESOLVED"
                for row in labels
            )
        )

    def test_articulated_target_uses_constrained_manipulation(self):
        meta = {
            "episode_key": "libero_goal/task_1/ep_0",
            "suite": "libero_goal",
            "task_index": 1,
            "mechanism_type": "articulated_object",
            "object_declarations": [],
            "fixture_declarations": ["drawer"],
            "structured_goal_metadata": {"target_fixtures": ["drawer"]},
            "gripper_command_semantics": "positive_is_close",
        }
        rows = [
            {
                "step": step,
                "contact_pairs": target_contact("drawer"),
                "gripper_command": 1.0,
                "constrained_manipulation_active": True,
                "release_safe": False,
            }
            for step in range(3)
        ]
        labels = build_clean_teacher_episode(
            rows,
            meta,
            thresholds=CleanTeacherThresholds(burst_length=2),
        )
        self.assertTrue(all(row["y_gripper_critical_window"] for row in labels))
        self.assertTrue(
            all(row["teacher_phase"] == "CONSTRAINED_MANIPULATION" for row in labels)
        )

    def test_attacked_fields_are_rejected(self):
        rows = positive_rows(3)
        rows[0]["vis_success"] = True
        with self.assertRaisesRegex(ValueError, "attacked"):
            build_clean_teacher_episode(rows, clean_metadata())


class CleanDatasetAdapterTests(unittest.TestCase):
    @staticmethod
    def row(episode, split, step, positive, known=True):
        labels = {
            "y_target_relevant": bool(positive),
            "y_contact_or_grasp_stable": bool(positive),
            "y_gripper_dependency": bool(positive),
            "y_clean_close_intent": bool(positive),
            "y_lift_transport_or_constraint": bool(positive),
            "y_release_safe": False,
            "y_gripper_critical_window": bool(positive),
            "y_burst_feasible": False,
            "y_attack_start_b": False,
        }
        if not known:
            labels = {key: None for key in labels}
        return {
            "teacher_schema_version": CLEAN_TEACHER_SCHEMA_VERSION,
            "episode_key": episode,
            "step": step,
            "suite": "s",
            "task_index": 0,
            "split": split,
            "mechanism_type": "pick_place_transfer",
            "mechanism_eligible": True,
            "teacher_phase": (
                "TRANSPORT"
                if known and positive
                else "APPROACH"
                if known
                else "CONTACT_UNRESOLVED"
            ),
            "teacher_reason_code": (
                "TARGET_CRITICAL_WINDOW"
                if known and positive
                else "APPROACH_NO_CONTACT"
                if known
                else "CONTACT_UNRESOLVED"
            ),
            "teacher_confidence": 1.0 if known else 0.0,
            "grounding_confidence": 1.0,
            "teacher_known": known,
            "label_known_mask": known,
            "resolved_target_objects": ["obj"],
            "resolved_target_manipulable_entities": [],
            "contacted_entities": ["obj"] if known and positive else [],
            "uses_privileged_sim_state": True,
            "uses_attack_outcome": False,
            "uses_future_student_input": False,
            **labels,
        }

    def test_target_mapping_and_unknown_mask(self):
        mapped = teacher_row_to_model_targets(self.row("p", "train", 0, True))
        self.assertEqual(mapped["targets"]["critical_window"], 1.0)
        unknown = teacher_row_to_model_targets(
            self.row("u", "test", 0, False, known=False)
        )
        self.assertFalse(unknown["masks"]["critical_window"])

    def test_fully_known_negative_never_uses_partial_unknown(self):
        rows = [
            self.row("negative", "train", 0, False),
            self.row("negative", "train", 1, False),
            self.row("partial", "train", 0, False),
            self.row("partial", "train", 1, False, known=False),
            self.row("positive", "train", 0, True),
        ]
        flags = derive_episode_fully_known_negative(rows)
        self.assertTrue(flags["negative"])
        self.assertFalse(flags["partial"])
        self.assertFalse(flags["positive"])

    def test_clean_split_coverage_uses_correct_label(self):
        rows = []
        for split in ("train", "val", "test"):
            rows.extend(
                [
                    self.row(f"{split}-p", split, 0, True),
                    self.row(f"{split}-p", split, 1, True),
                    self.row(f"{split}-n", split, 0, False),
                ]
            )
        coverage = clean_window_split_coverage(rows)
        self.assertEqual(coverage["test"]["attackable_episodes"], 1)
        self.assertEqual(coverage["test"]["fully_known_negative_episodes"], 1)

    def test_student_payload_is_exact_and_leak_free(self):
        assert_student_feature_payload(
            ["gripper_qpos", "eef_vx"],
            [{"gripper_qpos": 0.0, "eef_vx": 0.1}],
        )
        with self.assertRaises(ValueError):
            assert_student_feature_payload(
                ["teacher_phase"], [{"teacher_phase": 0.0}]
            )


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def load_spec():
    return AttackLoadSpec(
        burst_length=10,
        epsilon=0.1,
        step_size=0.02,
        pgd_steps=20,
        projection="processor_linf",
        cast_policy="project_and_cast_processor_values",
        preprocessing="upstream_tf_jpeg",
        image_height=224,
        image_width=224,
        random_start_policy="zero",
        temporal_init_policy="prev_delta",
        num_loss_forwards_per_frame=20,
        num_backwards_per_frame=20,
        num_adv_decodes_per_frame=1,
    )


def manifest_rows():
    rows = []
    for condition in CORE_CONDITIONS:
        attack = condition != "CLEAN"
        timing = (
            "NONE"
            if condition == "CLEAN"
            else "DETECTOR"
            if condition.startswith("DET_")
            else "RANDOM_TIME_MATCHED"
        )
        objective = (
            "NONE"
            if condition == "CLEAN"
            else "GRIPPER_TARGETED_VIS_PGD"
            if "GRIPPER" in condition
            else "RANDOM_DIRECTION_PGD_LOOP"
        )
        rows.append(
            {
                "condition": condition,
                "parent_key": "libero_object|0|1|0",
                "suite": "libero_object",
                "task_index": 0,
                "state_id": 1,
                "eval_seed": 0,
                "clean_parent_sha256": SHA_A,
                "initial_state_sha256": SHA_B,
                "detector_checkpoint_sha256": SHA_C,
                "detector_config_sha256": SHA_D,
                "timing_source": timing,
                "objective_family": objective,
                "objective_seed": deterministic_objective_seed(
                    "libero_object|0|1|0", condition, 7
                ),
                "attack_enabled": attack,
                "expected_attacked_frames": 10 if attack else 0,
                "planned_start_step": (
                    50
                    if timing == "DETECTOR"
                    else 30
                    if timing == "RANDOM_TIME_MATCHED"
                    else None
                ),
                "load_spec": asdict(load_spec()),
            }
        )
    return rows


class MatchedLoadManifestTests(unittest.TestCase):
    def test_core_matrix_passes_and_is_deterministic(self):
        result = validate_core_2x2_manifest(manifest_rows())
        self.assertEqual(result["status"], "PASS_CORE_2X2_MATCHED_LOAD")
        self.assertEqual(result["job_count"], 5)
        self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_missing_condition_and_duplicate_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "condition closure"):
            validate_core_2x2_manifest(manifest_rows()[:-1])
        rows = manifest_rows()
        rows.append(copy.deepcopy(rows[-1]))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_core_2x2_manifest(rows)

    def test_load_mismatch_is_rejected(self):
        rows = manifest_rows()
        rows[2]["load_spec"] = dict(rows[2]["load_spec"])
        rows[2]["load_spec"]["epsilon"] = 0.2
        with self.assertRaisesRegex(ValueError, "not exactly matched"):
            validate_core_2x2_manifest(rows)

    def test_timing_pair_and_parent_provenance_must_match(self):
        rows = manifest_rows()
        rows[3]["planned_start_step"] = 31
        with self.assertRaisesRegex(ValueError, "one identical planned_start_step"):
            validate_core_2x2_manifest(rows)
        rows = manifest_rows()
        rows[1]["initial_state_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            validate_core_2x2_manifest(rows)

    def test_uniform_noise_cannot_be_primary_same_load_control(self):
        rows = manifest_rows()
        rows[2]["objective_family"] = "UNIFORM_NOISE"
        with self.assertRaisesRegex(ValueError, "compute-matched"):
            validate_core_2x2_manifest(rows)


if __name__ == "__main__":
    unittest.main()
