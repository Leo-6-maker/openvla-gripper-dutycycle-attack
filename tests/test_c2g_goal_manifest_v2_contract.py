import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.build_c2g_suite_model_map import (
    V2_GOAL_SCHEMA,
    V2_GOAL_STATUS,
    sha256_file,
    validate_goal_manifest,
)


class GoalManifestV2ContractTests(unittest.TestCase):
    def test_finalizer_cli_runs_outside_repo(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                [sys.executable, str(repo / "scripts/stageb/finalize_c2g_goal_model_manifest_v2.py"), "--help"],
                cwd=td,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def build_model(self, root: Path) -> Path:
        model = root / "libero-goal"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        (model / "config.json").write_text("{}", encoding="utf-8")
        return model

    def manifest_for(self, model: Path, *, status: str = V2_GOAL_STATUS) -> dict:
        rows = []
        for name in ("config.json", "model.safetensors"):
            path = model / name
            rows.append({
                "relative_path": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        return {
            "schema_version": V2_GOAL_SCHEMA,
            "status": status,
            "model_path": str(model),
            "unnorm_key": "libero_goal",
            "files": rows,
            "referenced_shards": ["model.safetensors"],
            "missing_referenced_shards": [],
            "provenance_mode": "EXPLICIT_REBASE_CURRENT_BYTES",
            "load_only_validation": {
                "status": "PASS_C2G_GOAL_MODEL_LOAD_ONLY",
                "parameter_count": 10,
                "token_semantics_sha256": "1" * 64,
            },
            "boundaries": {
                "libero_rollouts_launched": 0,
                "attacks_launched": 0,
                "attack_outcomes_read": False,
            },
        }

    def test_v2_manifest_verifies_current_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.build_model(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(self.manifest_for(model)), encoding="utf-8")
            result = validate_goal_manifest(manifest, model)
            self.assertEqual(result["status"], V2_GOAL_STATUS)
            self.assertEqual(result["verified_file_count"], 2)
            self.assertEqual(result["provenance_mode"], "EXPLICIT_REBASE_CURRENT_BYTES")

    def test_manifest_detects_later_shard_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.build_model(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(self.manifest_for(model)), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "GOAL_MANIFEST_FILE_HASH_MISMATCH"):
                validate_goal_manifest(manifest, model)

    def test_v2_manifest_requires_zero_rollout_load_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.build_model(root)
            value = self.manifest_for(model)
            value["boundaries"]["libero_rollouts_launched"] = 1
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LIBERO rollout"):
                validate_goal_manifest(manifest, model)


if __name__ == "__main__":
    unittest.main()
