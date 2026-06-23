#!/usr/bin/env python3
"""Tests for M1C corpus integrity auditor v2."""
import json, tempfile, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.migration.audit_m1c_object_corpus import (
    scan_filesystem, build_planned_set, storage_key, semantic_key,
)


def _make_fs(tmpdir, cells):
    for pool, task, state in cells:
        d = Path(tmpdir) / pool / f"task{task}_state{state}"
        d.mkdir(parents=True)
        (d / ".done").write_text('{"exit_code":0,"telemetry_sha":"aa"}')
        (d / "step_telemetry.csv").write_text("step,col\n0,a\n")
        ep = {"condition": "CLEAN", "attack_frames": 0, "checkpoint_sha256": "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628"}
        (d / "episode_summary.json").write_text(json.dumps(ep))
    return str(Path(tmpdir))


def _make_manifest(train_range="3-9", val_range="28-37"):
    return {
        "output_root": "/tmp/test",
        "total_planned_cells": 350,
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
        assert storage_key("train", 0, 3) in actual
        assert storage_key("validation", 0, 28) in actual


def test_scan_excludes_non_cell_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp).mkdir(exist_ok=True)
        (Path(tmp) / "not_a_cell").mkdir()
        actual = scan_filesystem(str(tmp))
        assert len(actual) == 0


def test_planned_matches_manifest():
    manifest = _make_manifest()
    planned = build_planned_set(manifest)
    assert ("train", 3, 3) in planned
    assert ("validation", 5, 30) in planned
    assert ("blind", 0, 40) not in planned


def test_storage_key_unique():
    k1 = storage_key("train", 0, 3)
    k2 = storage_key("validation", 0, 3)
    assert k1 != k2  # same task/state, different pool = different key


def test_semantic_key_match():
    s1 = semantic_key(0, 3)
    s2 = semantic_key(0, 3)
    assert s1 == s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
