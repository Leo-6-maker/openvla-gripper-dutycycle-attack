import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.collect_c2g_clean_window_rollouts_release import (
    forbidden_clean_keys,
    partition_manifest_rows,
    rebuild_combined_collection_report,
    suite_command,
)
from scripts.stageb.run_c2g_clean_timing_jobs_strict import validate_goal_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_goal_v2_manifest(model: Path, output: Path) -> None:
    rows = []
    aggregate = hashlib.sha256()
    for path in sorted(model.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        sha = sha256_file(path)
        row = {
            "path": str(path),
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha,
        }
        rows.append(row)
        aggregate.update(f"{path.name}|{path.stat().st_size}|{sha}\n".encode("utf-8"))
    output.write_text(
        json.dumps(
            {
                "schema_version": "c2g.goal_model_integrity.2026-07-10.v2",
                "status": "PASS_C2G_GOAL_MODEL_INTEGRITY_AUDITED_V2",
                "model_path": str(model.resolve()),
                "unnorm_key": "libero_goal",
                "files": rows,
                "files_aggregate_sha256": aggregate.hexdigest(),
                "referenced_shards": ["model.safetensors"],
                "missing_referenced_shards": [],
                "provenance_mode": "EXPLICIT_REBASE_CURRENT_BYTES",
                "load_only_validation": {
                    "status": "PASS_C2G_GOAL_MODEL_LOAD_ONLY",
                    "parameter_count": 7,
                    "token_semantics_sha256": "a" * 64,
                },
                "boundaries": {
                    "libero_rollouts_launched": 0,
                    "attacks_launched": 0,
                    "attack_outcomes_read": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class ServerResumeHardeningTests(unittest.TestCase):
    def test_strict_pipeline_exports_repo_pythonpath(self):
        repo = Path(__file__).resolve().parents[1]
        wrapper = (repo / "scripts/stageb/run_c2g_clean_window_pipeline_strict.sh").read_text()
        self.assertIn(
            'export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
            wrapper,
        )

    def test_partition_manifest_rows_isolates_suites(self):
        rows = [
            {"suite": "libero_object", "parent_key": "o"},
            {"suite": "libero_goal", "parent_key": "g"},
            {"suite": "libero_object", "parent_key": "o2"},
        ]
        partitions = partition_manifest_rows(rows)
        self.assertEqual([row["parent_key"] for row in partitions["libero_object"]], ["o", "o2"])
        self.assertEqual([row["parent_key"] for row in partitions["libero_goal"]], ["g"])
        self.assertNotIn("libero_10", partitions)

    def test_suite_command_replaces_global_manifest_and_cap(self):
        command = suite_command(
            [
                "--manifest", "/old.jsonl",
                "--output-root", "/out",
                "--suite", "libero_goal",
                "--max-episodes", "4",
                "--expected-git-commit", "abc",
            ],
            Path("/tmp/new.jsonl"),
            "libero_object",
        )
        self.assertEqual(command.count("--manifest"), 1)
        self.assertEqual(command.count("--suite"), 1)
        self.assertEqual(command.count("--max-episodes"), 1)
        self.assertIn("collect_c2g_clean_window_rollouts_strict.py", command[1])
        self.assertEqual(command[command.index("--suite") + 1], "libero_object")
        self.assertEqual(command[command.index("--max-episodes") + 1], "0")

    def test_forbidden_key_scan_ignores_documentation_values(self):
        metadata = {
            "student_forbidden_modalities": ["attack_outcome", "post_intervention"],
            "nested": {"safe": True},
        }
        self.assertEqual(forbidden_clean_keys(metadata), [])
        metadata["nested"]["attack_outcome"] = False
        self.assertEqual(forbidden_clean_keys(metadata), ["nested.attack_outcome"])

    def test_combined_report_accepts_forbidden_names_as_documentation_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episode = root / "episodes" / "libero_object" / "parent"
            episode.mkdir(parents=True)
            metadata = {
                "git_commit": "abc",
                "condition": "CLEAN",
                "parent_key": "libero_object/task_0/state_0/seed_1/rep_0",
                "suite": "libero_object",
                "task_index": 0,
                "state_id": 0,
                "n_steps": 1,
                "student_forbidden_modalities": ["attack_outcome", "post_intervention"],
            }
            (episode / "episode_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (episode / "step_records.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
            report = rebuild_combined_collection_report(
                root,
                expected_git_commit="abc",
                suite_runs=[{"suite": "libero_object", "status": "PASS"}],
            )
            self.assertEqual(report["status"], "PASS_CLEAN_COLLECTION")
            self.assertEqual(report["episode_count"], 1)

    def test_clean_timing_accepts_byte_verified_goal_v2_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "goal"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.model").write_bytes(b"tokenizer")
            (model / "model.safetensors").write_bytes(b"weights")
            manifest = root / "goal_manifest_v2.json"
            write_goal_v2_manifest(model, manifest)
            result = validate_goal_manifest(manifest, model)
            self.assertEqual(result["status"], "PASS_C2G_GOAL_MODEL_INTEGRITY_AUDITED_V2")
            self.assertEqual(result["verified_file_count"], 3)
            (model / "model.safetensors").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "GOAL_MANIFEST_FILE_HASH_MISMATCH"):
                validate_goal_manifest(manifest, model)


if __name__ == "__main__":
    unittest.main()
