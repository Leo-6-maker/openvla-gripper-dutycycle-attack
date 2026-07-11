import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.bind_c2g_collection_model_provenance import bind
from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file
from scripts.stageb.build_c2g_suite_model_map_strict import full_model_manifest
from scripts.stageb.verify_c2g_collection_model_provenance import verify_collection


class CollectionModelVerificationTests(unittest.TestCase):
    def fixture(self, root: Path):
        model_map = {}
        manifests = {}
        for suite in SUITES:
            model = root / "models" / suite
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.model").write_bytes(b"tokenizer")
            (model / "model.safetensors").write_bytes((suite + "-weights").encode())
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
        model_report = root / "model_report.json"
        model_report.write_text(
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
        collection = root / "collection"
        for index, suite in enumerate(SUITES):
            episode = collection / suite / f"ep{index}"
            episode.mkdir(parents=True)
            (episode / "episode_metadata.json").write_text(
                json.dumps({"suite": suite, "runtime_valid": True}),
                encoding="utf-8",
            )
            (episode / "step_records.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
        artifact_paths = sorted(
            path
            for name in ("episode_metadata.json", "step_records.jsonl")
            for path in collection.rglob(name)
        )
        artifact_manifest = collection / "c2g_clean_collection_input_manifest.jsonl"
        artifact_manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "path": path.relative_to(collection).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    },
                    sort_keys=True,
                )
                + "\n"
                for path in artifact_paths
            ),
            encoding="utf-8",
        )
        (collection / "c2g_clean_collection_report.json").write_text(
            json.dumps(
                {
                    "artifact_manifest": str(artifact_manifest.resolve()),
                    "artifact_manifest_sha256": sha256_file(artifact_manifest),
                }
            ),
            encoding="utf-8",
        )
        model_verification = root / "verification.json"
        binding_report = root / "binding.json"
        binding = bind(
            collection,
            map_path,
            model_report,
            goal_path,
            model_verification,
        )
        binding_report.write_text(json.dumps(binding), encoding="utf-8")
        return (
            collection,
            binding_report,
            map_path,
            model_report,
            goal_path,
            model_verification,
        )

    def test_verification_passes_unchanged_collection(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            result = verify_collection(
                collection_root=args[0],
                binding_report_path=args[1],
                model_map=args[2],
                model_report=args[3],
                goal_manifest=args[4],
                model_verification_report=args[5],
            )
            self.assertEqual(
                result["status"],
                "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING_VERIFICATION",
            )
            self.assertEqual(result["episode_count"], 4)

    def test_metadata_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            metadata_path = next(args[0].rglob("episode_metadata.json"))
            metadata = json.loads(metadata_path.read_text())
            metadata["new_field"] = "tampered"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_collection(
                    collection_root=args[0],
                    binding_report_path=args[1],
                    model_map=args[2],
                    model_report=args[3],
                    goal_manifest=args[4],
                    model_verification_report=args[5],
                )

    def test_unexpected_episode_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.fixture(Path(td))
            extra = args[0] / "libero_object" / "extra"
            extra.mkdir(parents=True)
            (extra / "episode_metadata.json").write_text(
                json.dumps({"suite": "libero_object"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "closure"):
                verify_collection(
                    collection_root=args[0],
                    binding_report_path=args[1],
                    model_map=args[2],
                    model_report=args[3],
                    goal_manifest=args[4],
                    model_verification_report=args[5],
                )


if __name__ == "__main__":
    unittest.main()
