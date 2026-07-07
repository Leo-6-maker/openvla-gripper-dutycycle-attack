import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_feature_binding_manifest_v1 import make_binding


def test_validate_feature_binding_cli_passes_and_fails_tamper(tmp_path):
    _, _, _, manifest, _ = make_binding(tmp_path)
    cmd = [
        sys.executable,
        str(ROOT / "tools/multisuite_detector/validate_feature_binding_v1.py"),
        "--binding-manifest",
        str(manifest),
        "--expected-label-mode",
        "synthetic-dry-run",
    ]
    assert subprocess.run(cmd, check=True, capture_output=True, text=True).returncode == 0
    obj = json.loads(manifest.read_text())
    obj["feature_count"] = 24
    manifest.write_text(json.dumps(obj), encoding="utf-8")
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode == 1
    assert "feature_count mismatch" in failed.stderr
