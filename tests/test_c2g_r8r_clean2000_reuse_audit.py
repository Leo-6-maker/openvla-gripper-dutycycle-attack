import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8r_clean2000_reuse import (
    A_DIRECT, B_AUGMENT, C_LEGACY, D_RECOLLECT, HOLD_IDENTITY, HOLD_TEACHER,
    SOURCE_SPEC_SCHEMA, run_audit, sha256_file,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    CohortCounts, materialize_plan,
)

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]


def inventory(state_count=5):
    return [{"suite": suite, "task_index": 0, "state_ids": list(range(state_count))}
            for suite in SUITES]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_episode(root: Path, row: dict, *, kind="legacy", missing_rgb=False):
    episode = root / row["suite"] / f"task_{row['task_index']}" / f"state_{row['state_id']}"
    rgb = episode / "rgb"
    rgb.mkdir(parents=True)
    metadata = {
        "episode_key": row["parent_key"], "parent_key": row["parent_key"],
        "suite": row["suite"], "task_index": row["task_index"],
        "state_id": row["state_id"], "runtime_valid": True, "condition": "CLEAN",
        "task_language": "pick up the milk and place it in the basket",
        "feature_names_25d": FEATURES, "clean_success_observed": True,
        "model_path": "/models/openvla", "processor_path": "/models/openvla",
    }
    if kind in {"a", "b"}:
        metadata.update({
            "mechanism_type": "pick_place_transfer", "object_declarations": ["milk"],
            "receptacle_declarations": ["basket"],
            "structured_goal_metadata": {
                "target_objects": ["milk"], "target_receptacles": ["basket"],
            },
            "gripper_command_semantics": "positive_is_close",
        })
    (episode / "episode_metadata.json").write_text(json.dumps(metadata))
    steps = []
    for step in range(16):
        frame = rgb / f"frame_{step:06d}.png"
        if not (missing_rgb and step == 0):
            frame.write_bytes(b"r8r-rgb")
        item = {
            "step": step, "rgb_path": f"rgb/frame_{step:06d}.png",
            "task_language": metadata["task_language"], "features_25d": [float(step)] * 25,
        }
        if kind == "legacy":
            item.update(teacher_phase="stable_carry", corridor_label=1, release_label=0)
        if kind in {"a", "b"}:
            item.update(
                clean_close_intent=True,
                contact_pairs=[
                    ["robot0_left_finger_collision", "milk_collision"],
                    ["robot0_right_finger_collision", "milk_collision"],
                ],
                object_relative_lift=0.03, near_target=False,
            )
        if kind == "a":
            item["clean_policy_intent_9d"] = [0.1] * 9
        steps.append(item)
    (episode / "step_records.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in steps)
    )


def make_fixture(root: Path, *, mixed=False, conflict=False):
    counts = CohortCounts(train=2, val=1, test=1, attack_eval=1)
    plan = materialize_plan(
        inventory(counts.total), output_dir=root / "plan", counts=counts, seed=42,
        max_steps=300, expected_git_commit="a" * 40, inventory_source="unit_test",
    )
    registry = read_jsonl(Path(plan["registry"]))
    raw, merged, replacement = root / "raw", root / "merged", root / "replacement"
    for index, row in enumerate(registry):
        kind, missing = "legacy", False
        if mixed:
            if index == 0: kind = "a"
            elif index == 1: kind = "b"
            elif index == 3: missing = True
        write_episode(raw, row, kind=kind, missing_rgb=missing)
        write_episode(merged, row, kind="legacy")
        if row["suite"] == "libero_object":
            write_episode(replacement, row, kind=kind, missing_rgb=missing)
    evidence = root / "source_evidence.json"
    evidence.write_text(json.dumps({"status": "PASS"}))
    def view(name, source_root, source_class, suites, priority):
        return {
            "name": name, "root": str(source_root), "source_class": source_class,
            "canonical_suites": list(suites), "priority": priority, "clean_only": True,
            "runtime_valid_by_manifest": True, "model_provenance_bound": True,
            "processor_provenance_bound": True, "feature_25d_order_bound": True,
            "evidence_paths": [str(evidence)],
        }
    views = [
        view("raw", raw, "RAW_COLLECTION_SOURCE", SUITES if conflict else SUITES[1:], 10),
        view("merged", merged, "MERGED_VIEW", SUITES if conflict else (), 5),
        view("object_v1_1", replacement, "REPLACEMENT_SOURCE", ("libero_object",), 20),
    ]
    predecessor = root / "predecessor"
    predecessor.mkdir()
    (predecessor / "partial.log").write_text("partial")
    spec = root / "source_spec.json"
    spec.write_text(json.dumps({
        "schema": SOURCE_SPEC_SCHEMA, "views": views,
        "predecessor_roots": [str(predecessor)],
    }))
    source_audit = root / "r7_source_audit.json"
    source_audit.write_text(json.dumps({"status": "PASS"}))
    reusable = root / "r7_reusable.jsonl"
    reusable.write_text("{}\n")
    return plan, registry, spec, source_audit, reusable


def execute(root, *, mixed=False, conflict=False, output_name="audit"):
    plan, registry, spec, source_audit, reusable = make_fixture(
        root, mixed=mixed, conflict=conflict,
    )
    result = run_audit(
        registry_path=Path(plan["registry"]), plan_report_path=Path(plan["report"]),
        expected_plan_report_sha256=plan["report_sha256"],
        source_audit_report_path=source_audit,
        expected_source_audit_report_sha256=sha256_file(source_audit),
        reusable_manifest_path=reusable,
        expected_reusable_manifest_sha256=sha256_file(reusable),
        source_spec_path=spec, output_dir=root / output_name,
        expected_git_commit="b" * 40, verify_git_state=False,
    )
    return result, registry


class R8RClean2000AuditTests(unittest.TestCase):
    def test_legacy_corpus_closes_counts_without_1800_bug(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = execute(Path(temporary))
            self.assertEqual(result["r7_registry_identities"], 20)
            self.assertEqual(result["physical_episode_views"], 45)
            self.assertEqual(result["canonical_registered_identities"], 20)
            self.assertEqual(result["duplicate_source_views"], 25)
            self.assertEqual(result["identities_with_multiple_views"], 20)
            self.assertEqual(result["duplicate_conflicts"], 0)
            self.assertEqual(result["replacement_lineages"], 1)
            self.assertEqual(result["replaced_identity_count"], 5)
            self.assertEqual(result["classification_counts"][C_LEGACY], 20)
            self.assertEqual(result["detector_required_parent_count"], 16)
            self.assertEqual(result["attack_eval_required_parent_count"], 4)
            self.assertEqual(result["residual_detector_collection_required"], 16)
            self.assertEqual(result["residual_attack_eval_collection_required"], 4)
            self.assertEqual(result["total_current_contract_deficit"], 20)
            self.assertEqual(result["final_decision"], HOLD_TEACHER)
            self.assertTrue(all(result["invariants"].values()))

    def test_required_artifacts_and_sha_ledgers_are_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = execute(Path(temporary))
            output = Path(result["output_root"])
            required = {
                "clean2000_r7_reuse_audit_report.json", "clean2000_r7_episode_ledger.csv",
                "clean2000_r7_source_view_ledger.csv", "clean2000_r7_field_coverage.csv",
                "clean2000_r7_identity_reconciliation.csv",
                "clean2000_r7_teacher_v2_support.csv", "clean2000_r7_direct_reuse.jsonl",
                "clean2000_r7_offline_augmentation.jsonl", "clean2000_r7_legacy_only.jsonl",
                "clean2000_r7_recollect_required.jsonl",
                "clean2000_r7_legacy_semantic_salvage_candidates.jsonl",
                "clean2000_candidate_roots.jsonl", "bound_source_spec.json",
                "SHA256SUMS", "SHA256SUMS.sha256",
            }
            self.assertEqual(required, {path.name for path in output.iterdir() if path.is_file()})
            for line in (output / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(sha256_file(output / name), digest)
            declared, name = (output / "SHA256SUMS.sha256").read_text().split()
            self.assertEqual((name, declared), ("SHA256SUMS", sha256_file(output / "SHA256SUMS")))
            self.assertTrue(result["sha256sums_self_binding_pass"])
            self.assertEqual(len(result["report_sha256"]), 64)
            self.assertTrue(result["predecessor_root_status"])

    def test_a_b_c_d_classification_is_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, registry = execute(Path(temporary), mixed=True)
            import csv
            with (Path(result["output_root"]) / "clean2000_r7_episode_ledger.csv").open() as handle:
                ledger = {
                    (row["suite"], int(row["task_index"]), int(row["state_id"])): row
                    for row in csv.DictReader(handle)
                }
            keys = [(row["suite"], row["task_index"], row["state_id"]) for row in registry[:4]]
            self.assertEqual(ledger[keys[0]]["classification"], A_DIRECT)
            self.assertEqual(ledger[keys[1]]["classification"], B_AUGMENT)
            self.assertEqual(ledger[keys[2]]["classification"], C_LEGACY)
            self.assertEqual(ledger[keys[3]]["classification"], D_RECOLLECT)

    def test_multiple_canonical_views_fail_identity_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = execute(Path(temporary), conflict=True)
            self.assertGreater(result["duplicate_conflicts"], 0)
            self.assertEqual(result["final_decision"], HOLD_IDENTITY)

    def test_existing_output_root_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _, spec, source_audit, reusable = make_fixture(root)
            output = root / "audit"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                run_audit(
                    registry_path=Path(plan["registry"]),
                    plan_report_path=Path(plan["report"]),
                    expected_plan_report_sha256=plan["report_sha256"],
                    source_audit_report_path=source_audit,
                    expected_source_audit_report_sha256=sha256_file(source_audit),
                    reusable_manifest_path=reusable,
                    expected_reusable_manifest_sha256=sha256_file(reusable),
                    source_spec_path=spec, output_dir=output,
                    expected_git_commit="b" * 40, verify_git_state=False,
                )


if __name__ == "__main__":
    unittest.main()
