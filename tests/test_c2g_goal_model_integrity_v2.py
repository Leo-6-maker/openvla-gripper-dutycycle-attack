import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_goal_model_integrity_v2 import (
    HOLD_STATUS,
    PASS_STATUS,
    audit,
)


def write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, raw in tensors.items():
        start = offset
        payload.extend(raw)
        offset += len(raw)
        header[name] = {
            "dtype": "U8",
            "shape": [len(raw)],
            "data_offsets": [start, offset],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


class GoalModelIntegrityV2Tests(unittest.TestCase):
    def build_model(self, root: Path) -> Path:
        model = root / "libero-goal"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({
                "architectures": ["OpenVLAForActionPrediction"],
                "model_type": "openvla",
                "text_config": {"vocab_size": 32000},
            }),
            encoding="utf-8",
        )
        (model / "dataset_statistics.json").write_text(
            json.dumps({"libero_goal": {"action": {}}}),
            encoding="utf-8",
        )
        (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        write_safetensors(model / "model-00001-of-00001.safetensors", {"layer.weight": b"abcd"})
        (model / "model.safetensors.index.json").write_text(
            json.dumps({
                "metadata": {"total_size": 4},
                "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
            }),
            encoding="utf-8",
        )
        return model

    def test_valid_static_model_passes_and_records_prior_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.build_model(root)
            previous = root / "previous.json"
            previous.write_text(
                json.dumps({
                    "model_path": str(model),
                    "files": [{
                        "relative_path": "model-00001-of-00001.safetensors",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }],
                }),
                encoding="utf-8",
            )
            report = audit(model, previous)
            self.assertEqual(report["status"], PASS_STATUS)
            self.assertEqual(report["tensor_index_count"], 1)
            self.assertEqual(len(report["previous_manifest_comparison"]["mismatches"]), 1)
            self.assertEqual(report["problems"], [])

    def test_index_header_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = self.build_model(root)
            index = json.loads((model / "model.safetensors.index.json").read_text())
            index["weight_map"]["missing.weight"] = "model-00001-of-00001.safetensors"
            (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            report = audit(model)
            self.assertEqual(report["status"], HOLD_STATUS)
            self.assertTrue(any(problem.startswith("INDEX_TENSORS_MISSING_FROM_SHARD") for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()
