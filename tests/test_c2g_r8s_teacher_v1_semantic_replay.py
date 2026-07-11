import csv
import json
import tempfile
import unittest
from pathlib import Path

import tools.multisuite_detector.audit_c2g_r8s_teacher_v1_semantic_replay as r8s

SUITES = r8s.SUITES


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_fixture(root: Path, *, full_action=False, strict_prov=False, n_tasks=2, states_per_task=5):
    episodes = []
    for suite in SUITES:
        for task in range(n_tasks):
            for state in range(states_per_task):
                cohort = "ATTACK_EVAL_PREREGISTERED" if state == states_per_task - 1 else "DETECTOR_TRAIN"
                split = "attack_eval" if cohort.startswith("ATTACK") else "train"
                episode = root / "source" / suite / f"task_{task}" / f"state_{state}"
                episode.mkdir(parents=True)
                metadata = {
                    "suite": suite,
                    "task_index": task,
                    "state_id": state,
                    "task_language": "pick up object",
                    "max_steps": 300,
                }
                if strict_prov:
                    metadata.update(
                        libero_commit="abc",
                        robosuite_version="1.4",
                        mujoco_version="2.3",
                        controller_config={"type": "OSC_POSE"},
                        action_semantics="unnormalized_7d_delta_pose",
                        bddl_file="task.bddl",
                        seed=42,
                    )
                write_json(episode / "episode_metadata.json", metadata)
                rows = []
                for step in range(16):
                    row = {
                        "step": step,
                        "teacher_phase": "stable_carry",
                        "teacher_event_role": "primary_attackable",
                        "teacher_primary_attackable": True,
                        "teacher_release_safe": False,
                        "features_25d": [0.0] * 25,
                    }
                    if full_action:
                        row["clean_action"] = [0.1] * 7
                    rows.append(row)
                (episode / "step_records.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                episodes.append(
                    {
                        "suite": suite,
                        "task_index": task,
                        "state_id": state,
                        "parent_key": f"{suite}/task_{task}/state_{state}/p",
                        "cohort": cohort,
                        "split": split,
                        "metadata_path": str(episode / "episode_metadata.json"),
                        "step_records_path": str(episode / "step_records.jsonl"),
                    }
                )
    r8r = root / "r8r"
    r8r.mkdir()
    report = {
        "canonical_registered_identities": len(episodes),
        "final_decision": "HOLD_TEACHER_V2_RAW_EVIDENCE",
        "classification_counts": {"C_LEGACY_ONLY": len(episodes)},
    }
    write_json(r8r / "clean2000_r7_reuse_audit_report.json", report)
    with (r8r / "clean2000_r7_episode_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    files = [
        r8r / "clean2000_r7_reuse_audit_report.json",
        r8r / "clean2000_r7_episode_ledger.csv",
    ]
    hashes = {path.name: r8s.sha256_file(path) for path in files}
    (r8r / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    (r8r / "SHA256SUMS.sha256").write_text(
        f"{r8s.sha256_file(r8r / 'SHA256SUMS')}  SHA256SUMS\n",
        encoding="utf-8",
    )
    return r8r, episodes


class R8STeacherV1SemanticReplayTests(unittest.TestCase):
    def test_legacy_only_goes_auxiliary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r8r_root, episodes = build_fixture(root)
            result = r8s.run_audit(
                repo=root,
                expected_git_commit="a" * 40,
                r8r_root=r8r_root,
                expected_r8r_report_sha256=r8s.sha256_file(
                    r8r_root / "clean2000_r7_reuse_audit_report.json"
                ),
                output_dir=root / "out",
                verify_git_state=False,
            )
            self.assertEqual(result["final_decision"], r8s.GO_AUX)
            self.assertEqual(result["strict_replay_ready_count"], 0)
            self.assertEqual(result["legacy_auxiliary_eligible_count"], len(episodes))
            self.assertEqual(result["exact_equivalent_mapping_count"], 0)
            self.assertTrue(all(result["invariants"].values()))

    def test_strict_actions_generate_24_canary_without_attack_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r8r_root, episodes = build_fixture(
                root,
                full_action=True,
                strict_prov=True,
                n_tasks=2,
                states_per_task=5,
            )
            result = r8s.run_audit(
                repo=root,
                expected_git_commit="a" * 40,
                r8r_root=r8r_root,
                expected_r8r_report_sha256=r8s.sha256_file(
                    r8r_root / "clean2000_r7_reuse_audit_report.json"
                ),
                output_dir=root / "out",
                verify_git_state=False,
            )
            self.assertEqual(result["final_decision"], r8s.GO_REPLAY)
            self.assertEqual(result["strict_replay_ready_count"], len(episodes))
            self.assertEqual(result["replay_canary_parent_count"], 24)
            rows = [
                json.loads(line)
                for line in (root / "out" / "r8s_replay_canary_manifest.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 24)
            self.assertTrue(all(row["cohort"] != r8s.ATTACK_EVAL for row in rows))

    def test_partial_25d_is_not_full_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r8r_root, _ = build_fixture(root, full_action=False, strict_prov=True)
            result = r8s.run_audit(
                repo=root,
                expected_git_commit="a" * 40,
                r8r_root=r8r_root,
                expected_r8r_report_sha256=r8s.sha256_file(
                    r8r_root / "clean2000_r7_reuse_audit_report.json"
                ),
                output_dir=root / "out",
                verify_git_state=False,
            )
            self.assertEqual(
                result["coverage"]["partial_action_4d_complete"]["fraction"], 1.0
            )
            self.assertEqual(
                result["coverage"]["full_action_7d_complete"]["fraction"], 0.0
            )
            self.assertEqual(result["strict_replay_candidate_count"], 0)

    def test_r8r_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r8r_root, _ = build_fixture(root)
            (r8r_root / "clean2000_r7_episode_ledger.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "SHA256SUMS mismatch"):
                r8s.run_audit(
                    repo=root,
                    expected_git_commit="a" * 40,
                    r8r_root=r8r_root,
                    expected_r8r_report_sha256=r8s.sha256_file(
                        r8r_root / "clean2000_r7_reuse_audit_report.json"
                    ),
                    output_dir=root / "out",
                    verify_git_state=False,
                )

    def test_artifacts_and_hash_ledgers_are_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            r8r_root, _ = build_fixture(root)
            r8s.run_audit(
                repo=root,
                expected_git_commit="a" * 40,
                r8r_root=r8r_root,
                expected_r8r_report_sha256=r8s.sha256_file(
                    r8r_root / "clean2000_r7_reuse_audit_report.json"
                ),
                output_dir=root / "out",
                verify_git_state=False,
            )
            expected = {
                "r8s_episode_ledger.csv",
                "r8s_legacy_field_coverage.csv",
                "r8s_semantic_mapping_matrix.csv",
                "r8s_replay_feasibility.csv",
                "r8s_replay_canary_manifest.jsonl",
                "r8s_current_contract_uncovered.jsonl",
                "r8s_semantic_replay_audit_report.json",
                "SHA256SUMS",
                "SHA256SUMS.sha256",
            }
            self.assertEqual(expected, {path.name for path in (root / "out").iterdir()})
            for line in (root / "out" / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, r8s.sha256_file(root / "out" / name))
            self.assertNotEqual(
                r8s.sha256_file(root / "out" / "r8s_episode_ledger.csv"),
                r8s.sha256_file(root / "out" / "r8s_legacy_field_coverage.csv"),
            )


if __name__ == "__main__":
    unittest.main()
