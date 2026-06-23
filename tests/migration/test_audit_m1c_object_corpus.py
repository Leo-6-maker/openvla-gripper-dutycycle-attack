#!/usr/bin/env python3
"""Tests for M1C corpus integrity auditor v2."""
import json, tempfile, os, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.migration.audit_m1c_object_corpus import (
    scan_filesystem, build_planned_set, cell_key,
)


def _make_fs(tmpdir, cells):
    """Create cells on filesystem. cells = [(pool, task, state), ...]."""
    for pool, task, state in cells:
        d = Path(tmpdir) / pool / f"task{task}_state{state}"
        d.mkdir(parents=True)
        (d / ".done").write_text('{"exit_code":0,"telemetry_sha":"aa"}')
        (d / "step_telemetry.csv").write_text("step,col\n0,a\n")
        ep = {"condition": "CLEAN", "attack_frames": 0, "checkpoint_sha256": "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628"}
        (d / "episode_summary.json").write_text(json.dumps(ep))
    return str(Path(tmpdir))


def _make_manifest(tmpdir, train_range="3-9", val_range="28-37"):
    return {
        "output_root": str(Path(tmpdir) / "corpus"),
        "total_planned_cells": 350,
        "train_planned": 250,
        "validation_planned": 100,
        "gpu_assignments": {
            "gpu2": {"pool": "train", "state_range": train_range, "planned_cells": 70},
            "gpu3": {"pool": "train", "state_range": "16-27", "planned_cells": 120},
            "gpu4": {"pool": "validation", "state_range": val_range, "planned_cells": 100},
        },
    }


def test_scan_finds_cells():
    with tempfile.TemporaryDirectory() as tmp:
        fs = _make_fs(tmp, [("train", 0, 3), ("validation", 0, 28)])
        actual = scan_filesystem(fs)
        assert cell_key(0, 3) in actual
        assert cell_key(0, 28) in actual


def test_scan_excludes_non_cell_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp).mkdir(exist_ok=True)
        (Path(tmp) / "not_a_cell").mkdir()
        actual = scan_filesystem(str(tmp))
        assert len(actual) == 0


def test_planned_matches_manifest():
    manifest = _make_manifest("/tmp/test")
    planned = build_planned_set(manifest)
    assert ("train", 3, 3) in planned
    assert ("validation", 5, 30) in planned
    assert ("blind", 0, 40) not in planned


def test_missing_cell_detected():
    with tempfile.TemporaryDirectory() as tmp:
        fs = _make_fs(tmp, [("train", 0, 3)])  # only one cell
        manifest = _make_manifest(str(tmp))
        manifest["output_root"] = fs
        actual = scan_filesystem(fs)
        planned = build_planned_set(manifest)
        missing_keys = []
        for pool, task, state in planned:
            if cell_key(task, state) not in actual:
                missing_keys.append(f"{pool}/{cell_key(task,state)}")
        assert len(missing_keys) > 0  # most cells are missing


def test_unexpected_cell_detected():
    with tempfile.TemporaryDirectory() as tmp:
        fs = _make_fs(tmp, [("train", 0, 40)])  # state 40 is compromised blind!
        manifest = _make_manifest(str(tmp))
        manifest["output_root"] = fs
        actual = scan_filesystem(fs)
        planned = build_planned_set(manifest)
        unexpected = []
        for key, cell in actual.items():
            if (cell["pool"], cell["task"], cell["state"]) not in planned:
                unexpected.append(key)
        assert len(unexpected) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
