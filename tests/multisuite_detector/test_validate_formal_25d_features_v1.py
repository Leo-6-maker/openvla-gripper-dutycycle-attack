import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact
from tests.multisuite_detector.test_extract_formal_25d_features_v1 import make_sources
from tools.multisuite_detector.extract_formal_25d_features_v1 import build_feature_artifact


def test_validator_cli_reports_pass(tmp_path):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, _ = make_sources(tmp_path, label_rows)
    artifact_root = tmp_path / "features"
    build_feature_artifact(source_csv, label_root, artifact_root, approved_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/multisuite_detector/validate_formal_25d_features_v1.py"),
            "--artifact-root",
            str(artifact_root),
            "--label-artifact-root",
            str(label_root),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["formal_detector_dataset_build"] == "NOT_PERFORMED"
    assert report["gpu"] == "NOT_PERFORMED"
