import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from audit_r3_contact_input import R3ContractError, audit, sha256_file


def _entity(name, role, position):
    return {
        "logical_name": name,
        "alias_to": None,
        "role": role,
        "entity_id": f"body:{name}",
        "body_origin": list(position),
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _row(identity, step):
    pair = {
        "entity_a": {"logical_name": "cube_1", "role": "MANIPULATED_OBJECT", "entity_id": "body:cube_1"},
        "entity_b": {"logical_name": "gripper", "role": "GRIPPER", "entity_id": "body:gripper"},
        "position": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "normal_constraint_force_scalar": 1.0,
    }
    return {
        "episode_id": identity,
        "step": step,
        "valid": True,
        "candidate_close": True,
        "entities": [_entity("cube_1", "MANIPULATED_OBJECT", [0.0, 0.0, 0.0]), _entity("gripper", "GRIPPER", [0.0, 0.0, 0.0])],
        "contact_pairs": [pair],
        "contact_ncon_total": 1,
        "contact_truncated": False,
        "forward_before_capture": True,
        "eef_pos": [0.0, 0.0, 0.0],
        "eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.0, 0.0],
        "protocol_steps_remaining": 520 - step,
    }


def _seal(root):
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def _root(tmp_path):
    root = tmp_path / "canary"
    episodes = []
    (root / "episodes").mkdir(parents=True)
    for index in range(8):
        identity = f"libero_10/task_{index:02d}/state_00"
        path = root / "episodes" / f"ep_{index}.jsonl"
        path.write_text("".join(json.dumps(_row(identity, step), sort_keys=True) + "\n" for step in range(2)), encoding="utf-8")
        episodes.append({
            "episode_id": identity,
            "suite": "libero_10",
            "task_id": index,
            "state_id": 0,
            "seed": 0,
            "relative_path": path.relative_to(root).as_posix(),
            "step_count": 2,
            "source_sha256": sha256_file(path),
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_command": "synthetic",
            "environment": "synthetic",
        })
    (root / "MANIFEST.json").write_text(json.dumps({
        "schema": "FIT670_V2_CANARY_CONTACT_TELEMETRY_V1",
        "status": "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE",
        "protected_reads": False,
        "attack_enabled": False,
        "episodes": episodes,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _seal(root)
    return root


def test_valid_eight_episode_canary_audits(tmp_path):
    report = audit(_root(tmp_path), expected_count=8)
    assert report["status"] == "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE"
    assert report["identity_count"] == 8
    assert report["step_count"] == 16
    assert report["protected_reads"] == 0


def test_wrong_gate_status_is_rejected(tmp_path):
    root = _root(tmp_path)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["status"] = "WRITING"
    (root / "MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _seal(root)
    with pytest.raises(ValueError, match="not consumable"):
        audit(root, expected_count=8)
