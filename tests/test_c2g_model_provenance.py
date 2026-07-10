import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.build_c2g_suite_model_map import SUITES, sha256_file
from scripts.stageb.build_c2g_suite_model_map_strict import full_model_manifest
from scripts.stageb.verify_c2g_suite_model_map_strict import verify


class ModelProvenanceTests(unittest.TestCase):
    def build_model(self, root: Path, name: str) -> Path:
        model = root / name
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.model").write_bytes(b"tokenizer")
        (model / "processor_config.json").write_text("{}", encoding="utf-8")
        (model / "model.safetensors").write_bytes((name + "-weights").encode("utf-8"))
        return model

    def frozen_fixture(self, root: Path):
        model_map = {suite: str(self.build_model(root, suite)) for suite in SUITES}
        map_path = root / "model_map.json"
        map_path.write_text(json.dumps(model_map, sort_keys=True), encoding="utf-8")
        manifests = {
            suite: full_model_manifest(Path(path))
            for suite, path in model_map.items()
        }
        goal_manifest_path = root / "goal_manifest.json"
        goal_manifest_path.write_text(
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
            "path": str(goal_manifest_path),
            "sha256": sha256_file(goal_manifest_path),
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
        return map_path, report_path, goal_manifest_path, model_map

    def test_full_weight_bound_verification_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            map_path, report_path, goal_manifest, _ = self.frozen_fixture(root)
            result = verify(map_path, report_path, goal_manifest)
            self.assertEqual(
                result["status"],
                "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION",
            )
            self.assertEqual(set(result["suite_models"]), set(SUITES))

    def test_weight_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            map_path, report_path, goal_manifest, model_map = self.frozen_fixture(root)
            Path(model_map["libero_object"]).joinpath("model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify(map_path, report_path, goal_manifest)

    def test_model_map_mutation_is_detected_before_weight_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            map_path, report_path, goal_manifest, model_map = self.frozen_fixture(root)
            model_map["libero_object"] = model_map["libero_spatial"]
            map_path.write_text(json.dumps(model_map), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "map SHA256"):
                verify(map_path, report_path, goal_manifest)


if __name__ == "__main__":
    unittest.main()
