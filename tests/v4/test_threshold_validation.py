from pathlib import Path
import pytest

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from v4_run_eval_openvla import validate_thresholds_for_trigger


def test_validate_thresholds_ignores_non_threshold_trigger(tmp_path):
    validate_thresholds_for_trigger("clean", "", {}, "task", 0.1)


def test_validate_thresholds_fails_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        validate_thresholds_for_trigger("entropy_threshold", str(tmp_path / "missing.json"), {}, "task", 0.1)


def test_validate_thresholds_requires_rollout_and_min_steps(tmp_path):
    p = tmp_path / "thresholds.json"
    p.write_text("{}", encoding="utf-8")
    thresholds = {"calibration_source": "rollout_passive_clean_observer", "tasks": {"task": {"rho_0.10": {"entropy": 1.0, "margin": 2.0, "num_steps": 1001}}}}
    validate_thresholds_for_trigger("entropy_threshold", str(p), thresholds, "task", 0.1, require_rollout_source=True, min_steps=1000)
    with pytest.raises(SystemExit):
        validate_thresholds_for_trigger("margin_threshold", str(p), {**thresholds, "calibration_source": "offline_demo"}, "task", 0.1, require_rollout_source=True)
    with pytest.raises(SystemExit):
        validate_thresholds_for_trigger("entropy_threshold", str(p), thresholds, "task", 0.1, min_steps=2000)


def test_validate_thresholds_rejects_dry_run_when_rollout_required(tmp_path):
    p = tmp_path / "thresholds.json"
    p.write_text("{}", encoding="utf-8")
    thresholds = {
        "dry_run": True,
        "calibration_source": "rollout_passive_clean_observer",
        "tasks": {"task": {"rho_0.10": {"entropy": 1.0, "margin": 2.0, "num_steps": 1001}}},
    }
    with pytest.raises(SystemExit):
        validate_thresholds_for_trigger("entropy_threshold", str(p), thresholds, "task", 0.1, require_rollout_source=True, min_steps=1000)
