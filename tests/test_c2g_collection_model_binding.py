import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.bind_c2g_collection_model_provenance import BINDING_SCHEMA, bind
from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file
from scripts.stageb.build_c2g_suite_model_map_strict import full_model_manifest


class CollectionModelBindingTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        model_map = {}
        manifests = {}
        for suite in SUITES:
            model = root / "models" / suite
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.model").write_bytes(b"tokenizer")
            (model / "model.safetensors").write_bytes(suite.encode("utf-8"))
            model_map[suite] = str(model)
            manifests[suite] = full_model_manifest(model)
        map_path = root / "model_map.json"
        map_path.write_text(json.dumps(model_map, sort_keys=True), encoding="utf-8")
        goal_path = root / "goal.json"
        goal_path.write_text(
            json.dumps(
                {
                    "status": "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED",
                    "model_path": model_map["libero_goal"],
                    "missing_referenced_shards": [],
                    "unnorm_key": "libero_goal",
                }
            ),
            encoding="utf-8",
        )
        goal_summary = {
            "path": str(goal_path),
            "sha256": sha256_file(goal_path),
            "status": "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED",
            "unnorm_key": "libero_goal",
        }
        report_path = root / "model_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "status": "PASS_C2G_STRICT_SUITE_MODEL_MAP",
                    "model_map_sha256": sha256_file(map_path),
                    "suite_models": manifests,
                    "goal_model_manifest": goal_summary,
                }
            ),
            encoding="utf-8",
        )
        verification_path = root / "verification.json"
        collection = root / "collection"
        for index, suite in enumerate(SUITES):
            episode = collection / suite / f"ep{index}"
            episode.mkdir(parents=True)
            (episode / "episode_metadata.json").write_text(
                json.dumps({"suite": suite, "runtime_valid": True}),
                encoding="utf-8",
            )
        return collection, map_path, report_path, goal_path, verification_path, manifests

    def test_binding_updates_every_episode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            collection, model_map, model_report, goal, verification, manifests = self.build_fixture(root)
            result = bind(collection, model_map, model_report, goal, verification)
            self.assertEqual(result["status"], "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING")
            self.assertEqual(result["episode_count"], 4)
            for suite in SUITES:
                path = next((collection / suite).rglob("episode_metadata.json"))
                metadata = json.loads(path.read_text())
                binding = metadata["c2g_model_binding"]
                self.assertEqual(binding["schema_version"], BINDING_SCHEMA)
                self.assertEqual(
                    binding["suite_full_model_manifest_sha256"],
                    manifests[suite]["full_model_manifest_sha256"],
                )

    def test_conflicting_existing_binding_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            collection, model_map, model_report, goal, verification, _ = self.build_fixture(root)
            first = next(collection.rglob("episode_metadata.json"))
            metadata = json.loads(first.read_text())
            metadata["c2g_model_binding"] = {"schema_version": "wrong"}
            first.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                bind(collection, model_map, model_report, goal, verification)


if __name__ == "__main__":
    unittest.main()
