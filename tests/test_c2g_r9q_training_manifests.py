import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.multisuite_detector.build_c2g_r9q_training_manifests import build_views


class R9QManifestTests(unittest.TestCase):
    def _combined(self, root: Path) -> Path:
        combined = root / "combined"
        combined.mkdir()
        rows = []
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            for split in ("FIT", "CAL", "CHECK"):
                path = combined / f"{suite}_{split}.npz"
                np.savez(path, x=np.zeros(1, dtype=np.float32))
                rows.append({
                    "suite": suite,
                    "task_index": 0,
                    "state_id": len(rows),
                    "parent_key": f"{suite}/task_0/state_{len(rows)}",
                    "cohort": "DETECTOR_TRAIN",
                    "preview_split": split,
                    "task_language": f"do {suite}",
                    "npz_path": str(path.resolve()),
                    "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        (combined / "dataset_index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (combined / "normalization.json").write_text("{}\n", encoding="utf-8")
        return combined

    def test_a2_excludes_l10_and_b2_includes_partial_l10(self):
        with tempfile.TemporaryDirectory() as tmp:
            combined = self._combined(Path(tmp))
            a2 = build_views(combined_root=combined, output_root=Path(tmp) / "a2", mode="a2", expected_fit=3, expected_cal=3)
            b2 = build_views(combined_root=combined, output_root=Path(tmp) / "b2", mode="b2", expected_fit=4, expected_cal=4)
            self.assertEqual(a2["fit_count"], 3)
            self.assertEqual(b2["fit_count"], 4)
            a2_rows = [json.loads(line) for line in (Path(tmp) / "a2" / "fit_manifest.jsonl").read_text().splitlines()]
            b2_rows = [json.loads(line) for line in (Path(tmp) / "b2" / "fit_manifest.jsonl").read_text().splitlines()]
            self.assertNotIn("libero_10", {row["suite"] for row in a2_rows})
            self.assertIn("libero_10", {row["suite"] for row in b2_rows})
            for view in (Path(tmp) / "a2", Path(tmp) / "b2"):
                for line in (view / "SHA256SUMS").read_text().splitlines():
                    digest, name = line.split()
                    self.assertEqual(digest, hashlib.sha256((view / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
