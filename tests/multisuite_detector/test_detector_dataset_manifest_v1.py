import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_feature_binding_manifest_v1 import make_binding


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_detector_dataset_manifest_metadata_only(tmp_path):
    _, _, _, binding_manifest, _ = make_binding(tmp_path)
    out = tmp_path / "metadata"
    subprocess.run([
        sys.executable,
        str(ROOT / "tools/multisuite_detector/build_detector_dataset_manifest_v1.py"),
        "--binding-manifest",
        str(binding_manifest),
        "--output-root",
        str(out),
    ], check=True)
    names = {p.name for p in out.iterdir()}
    assert names == {
        "dataset_manifest.json",
        "dataset_statistics.json",
        "population_summary.csv",
        "feature_summary.csv",
    }
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert manifest["training"] == "NOT_PERFORMED"
    assert manifest["gpu"] == "NOT_PERFORMED"
    assert len(read_csv(out / "feature_summary.csv")) == 25
