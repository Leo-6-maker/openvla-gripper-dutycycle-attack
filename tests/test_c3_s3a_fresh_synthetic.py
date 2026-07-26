import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))
import run_c3_s3a_fresh_synthetic as fresh  # noqa: E402


def test_gate_split_separates_c3_s3a_and_d0():
    config = json.loads((ROOT / "configs" / "C3_S3A_D0_GATE_SPLIT_V1.json").read_text(encoding="utf-8"))
    assert config["gate_split"]["C3-S3A"]["clean2000_payload_read"] is False
    assert config["gate_split"]["D0"]["status"] == "HOLD"


def test_synthetic_relation_plan_is_exactly_11_31_2():
    plan = fresh.build_relation_plan()
    assert len(plan) == 44
    counts = {category: sum(row["category"] == category for row in plan) for category in fresh.EXPECTED}
    assert counts == {"STATIC": 11, "DYNAMIC": 31, "ARTICULATED": 2}
    assert min(row["step_count"] for row in plan if row["category"] == "STATIC") >= 10
    assert min(row["step_count"] for row in plan if row["category"] != "STATIC") >= 100
    for row in plan:
        if row["category"] == "ARTICULATED":
            chain = row["joint_chain"]
            assert chain["kind"] == "ARTICULATED_JOINT_CHAIN"
            assert isinstance(chain["qpos_index"], int)
            assert len(chain["axis"]) == 3
            assert len(chain["limits"]) == 2
            assert chain["ancestor_chain"] == ["world", "parent", "entity"]


def test_canonical_payload_excludes_run_id():
    plan = fresh.build_relation_plan()
    assert len({row["relation_id"] for row in plan}) == 44


def test_source_faults_mutate_source_only():
    source = [{
        "episode_id": "e",
        "step": 0,
        "entities": [{
            "entity_id": "x",
            "reconstruction": {
                "kind": "ARTICULATED",
                "parent_world_pose": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                "local_pose": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
                "articulated_joint_chain": {
                    "kind": "ARTICULATED_JOINT_CHAIN", "joint_name": "j", "qpos_index": 0,
                    "axis": [0.0, 0.0, 1.0], "limits": [-1.0, 1.0], "ancestor_chain": ["world", "parent", "entity"], "qpos": 0.0,
                },
            },
        }],
    }]
    original = json.loads(json.dumps(source))
    for fault in ("translation", "rotation", "local-transform", "qpos", "joint-axis"):
        mutated = fresh.apply_source_fault(source, fault)
        assert mutated != original
        assert source == original


def test_atomic_noreplace_rejects_existing(tmp_path):
    if os.name != "posix":
        pytest.skip("renameat2 is a Linux server contract")
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (target / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fresh._rename_noreplace(source, target)
    assert (target / "sentinel").read_text(encoding="utf-8") == "keep"


def test_runner_has_no_legacy_shared_pose_or_replace():
    text = (ROOT / "n5" / "phase3_student" / "run_c3_s3a_fresh_synthetic.py").read_text(encoding="utf-8")
    assert "def _step_poses" not in text
    assert "os.replace" not in text
