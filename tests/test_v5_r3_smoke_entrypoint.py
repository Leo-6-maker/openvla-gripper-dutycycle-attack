import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "detector_v5" / "run_r3_learnability_smoke.py"


def test_synthetic_smoke_is_explicitly_nonconsumable():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--synthetic", "--epochs", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "ENGINEERING_NONCONSUMABLE"
    assert report["formal_training_authorized"] is False
    assert report["protected_reads"] == 0
    assert report["attack_authorized"] is False
