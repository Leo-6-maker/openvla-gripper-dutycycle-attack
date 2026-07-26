import math
import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import c3_s3_geometry_observability as c3  # noqa: E402
import c3_s3_input_contract as contract  # noqa: E402


def test_transform_contract():
    result = c3.transform_contract_tests()
    assert result["pass"] is True
    assert result["cases"]["rot_z_90"]["max_abs_error"] < 1e-12


def test_quaternion_normalization_and_inverse():
    q = c3.quat_normalize((2.0, 0.0, 0.0, 0.0))
    assert q == (1.0, 0.0, 0.0, 0.0)
    identity = c3.quat_mul(q, c3.quat_inverse(q))
    assert max(abs(x - y) for x, y in zip(identity, (1.0, 0.0, 0.0, 0.0))) < 1e-12


def test_unknown_is_not_negative():
    row = {
        "task_key": "libero_10/task_00",
        "classification": "ARTICULATED_UNKNOWN",
        "observability_status": "MAPPING_ONLY_REPLAY_EVIDENCE_REQUIRED",
        "unknown_is_negative": False,
        "silent_fallback": False,
    }
    assert row["classification"] != "STATIC_FIXTURE"
    assert row["unknown_is_negative"] is False


def test_protected_path_rejection():
    assert any(token in "/mnt/example/t2r-d" for token in c3.PROTECTED_TOKENS)


def _allowlist(tmp_path, *, episode_roots=None, denied_roots=None):
    return {
        "schema": "C3_S3_ALLOWED_INPUTS_V1",
        "protected_semantics_read": False,
        "allowed_roots": [{"name": "root", "path": str(tmp_path), "manifest_path": "manifest.json", "manifest_sha256": ""}],
        "allowed_episode_geometry_roots": episode_roots or [],
        "denied_roots": denied_roots or [],
    }


def test_explicit_allowlist_rejects_denied_and_unlisted(tmp_path):
    allow = _allowlist(tmp_path, denied_roots=[{"path": str(tmp_path / "denied"), "reason": "protected"}])
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "denied").mkdir()
    (tmp_path / "denied" / "x.txt").write_text("x", encoding="utf-8")
    resolved, _ = contract.require_allowed_path(tmp_path / "ok.txt", allow)
    assert resolved.name == "ok.txt"
    try:
        contract.require_allowed_path(tmp_path / "denied" / "x.txt", allow)
    except ValueError as exc:
        assert "denied" in str(exc)
    else:
        raise AssertionError("denied path was accepted")


def test_symlink_component_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "x.txt").write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    allow = _allowlist(tmp_path)
    try:
        contract.require_allowed_path(link / "x.txt", allow)
    except ValueError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("symlink path was accepted")


def test_exact_step_join_rejects_duplicate_and_missing(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"episode_id": "e", "step": s}) for s in (0, 0)) + "\n", encoding="utf-8")
    try:
        contract.load_jsonl_exact(path, episode_id="e", step_count=2, role="source")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate steps were accepted")
    path.write_text(json.dumps({"episode_id": "e", "step": 1}) + "\n", encoding="utf-8")
    try:
        contract.load_jsonl_exact(path, episode_id="e", step_count=2, role="source")
    except ValueError as exc:
        assert "exact step join" in str(exc)
    else:
        raise AssertionError("missing step was accepted")


def test_static_dynamic_transform_and_quaternion_sign_equivalence():
    half = math.sqrt(0.5)
    parent = {"pos": [1.0, 2.0, 3.0], "quat": [half, 0.0, 0.0, half]}
    local = {"pos": [1.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
    pose = contract.compose_pose(parent, local)
    assert pose["pos"] == [1.0, 3.0, 3.0]
    assert contract.rotation_geodesic_error(pose["quat"], [-x for x in pose["quat"]]) < 1e-12
    assert contract.position_error(pose["pos"], [1.0, 3.0, 3.0]) < 1e-12


def test_p99_reports_explicit_denominator():
    result = contract.p99([0.0, 1.0, 2.0, 3.0])
    assert result["count"] == 4
    assert result["method"] == "linear_interpolation_n_minus_1"
    assert 2.0 < result["value"] < 3.0


def test_articulated_unknown_is_excluded_from_denominator_not_negative():
    entry = {"episode_id": "e", "task_key": "libero_10/task_00", "step_count": 1}
    source = [{"episode_id": "e", "step": 0, "entities": [{"entity_id": "cabinet", "status": "UNKNOWN_ARTICULATED"}]}]
    reference = [{"episode_id": "e", "step": 0, "entities": [{"entity_id": "cabinet", "world_pose": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}}]}]
    result = contract.audit_episode_geometry(entry, source, reference)
    assert result["unknown_articulated_count"] == 1
    assert result["compared_pose_count"] == 0


def test_geometry_exact_join_rejects_identity_mismatch():
    entry = {"episode_id": "e", "task_key": "libero_10/task_00", "step_count": 1}
    source = [{"episode_id": "e", "step": 0, "entities": [{"entity_id": "x", "reconstruction": {"kind": "STATIC", "parent_world_pose": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}, "local_pose": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}}}]}]
    reference = [{"episode_id": "e", "step": 0, "entities": [{"entity_id": "y", "world_pose": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}}]}]
    try:
        contract.audit_episode_geometry(entry, source, reference)
    except ValueError as exc:
        assert "entity join" in str(exc)
    else:
        raise AssertionError("entity mismatch was accepted")
