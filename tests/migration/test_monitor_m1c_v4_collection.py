#!/usr/bin/env python3
"""Tests for M1C V4 collection monitor — safe output, alert triggers, forbidden fields."""
import json, tempfile, os, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.migration.monitor_m1c_v4_collection import (
    monitor, ALLOWED_METRICS, FORBIDDEN_METRICS, ALERT_CONDITIONS,
)


def _make_manifest(tmpdir):
    """Create a minimal valid manifest written to disk."""
    out_root = Path(tmpdir) / "corpus"
    out_root.mkdir()
    for pool in ["train", "validation"]:
        (out_root / pool).mkdir()
    data = {
        "gate": "test", "a800_host": "test", "total_planned_cells": 350,
        "output_root": str(out_root),
        "gpu_assignments": {
            "gpu2": {"pool": "train", "state_range": "3-9", "planned_cells": 70},
            "gpu3": {"pool": "train", "state_range": "16-27", "planned_cells": 120},
            "gpu4": {"pool": "validation", "state_range": "28-37", "planned_cells": 100},
        },
        "asset_sha_consistency": {
            "detector_checkpoint": "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628",
        },
    }
    mp = Path(tmpdir) / "manifest.json"
    mp.write_text(json.dumps(data))
    return str(mp)


def test_monitor_output_has_allowed_only():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _make_manifest(tmp)
        report = monitor(mp, None)
        for key in report.keys():
            assert key not in FORBIDDEN_METRICS, f"forbidden key: {key}"
        for s in report.get("shards", {}).values():
            for key in s.keys():
                assert key in ALLOWED_METRICS, f"unknown key: {key}"


def test_monitor_detects_no_new_done():
    import platform
    if platform.system() == "Windows":
        pytest.skip("mtime manipulation unreliable on Windows tmp filesystem")
    with tempfile.TemporaryDirectory() as tmp:
        mp = _make_manifest(tmp)
        done_file = Path(tmp) / "corpus" / "train" / "task0_state3" / ".done"
        done_file.parent.mkdir(parents=True)
        done_file.write_text('{"exit_code":0}')
        old = os.path.getmtime(str(done_file)) - 1800
        os.utime(str(done_file), (old, old))
        report = monitor(mp, None)
        alerts = report.get("alerts", [])
        assert any("NO_NEW_DONE_20MIN" in a for a in alerts)


def test_monitor_attack_detection():
    with tempfile.TemporaryDirectory() as tmp:
        mp = _make_manifest(tmp)
        cell = Path(tmp) / "corpus" / "train" / "task0_state3"
        cell.mkdir(parents=True)
        (cell / ".done").write_text('{"exit_code":0}')
        (cell / "step_telemetry.csv").write_text("step\n0\n")
        ep = {"mlp_triggered": True, "attack_frames": 5, "condition": "CLEAN"}
        (cell / "episode_summary.json").write_text(json.dumps(ep))
        report = monitor(mp, None)
        alerts = report.get("alerts", [])
        assert any("ATTACK_FRAMES_NONZERO" in a for a in alerts)


def test_forbidden_metrics_not_in_output():
    assert "success_rate" in FORBIDDEN_METRICS
    assert "emit_count" in FORBIDDEN_METRICS
    assert "teacher_valid" in FORBIDDEN_METRICS
    assert "no_corridor" in FORBIDDEN_METRICS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
