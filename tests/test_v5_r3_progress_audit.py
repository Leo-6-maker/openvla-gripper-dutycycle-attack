import json
import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))
from audit_r3_progress import audit, sha256_file
from gripper_attack.seal_utils import rename_noreplace


def _seal(root):
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def _root(tmp_path, duplicate=False):
    root = tmp_path / "formal"
    published = []
    for index in range(2):
        identity = f"libero_10/task_00/state_{index:02d}"
        episode = root / "episodes" / "libero_10" / "task_00" / f"state_{index:02d}"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text(json.dumps({"episode_id": identity}), encoding="utf-8")
        _seal(episode)
        published.append(identity)
    worker = root / "gpu_0"
    worker.mkdir(parents=True)
    rows = [{"episode_id": identity} for identity in published]
    if duplicate:
        rows.append({"episode_id": published[0]})
    (worker / "WORKER_MANIFEST.json").write_text(json.dumps({"results": rows, "n_success": len(rows)}), encoding="utf-8")
    _seal(worker)
    allowlist = tmp_path / "allowlist.json"
    allowlist_data = {
        "schema": "FIT670_IDENTITY_ALLOWLIST_V1",
        "protected_overlap": 0,
        "identities": [{"episode_id": identity, "suite": "libero_10", "task_id": 0, "state_id": index, "collection_seed": 0, "initial_state_sha256": "0" * 64} for index, identity in enumerate(published)],
    }
    allowlist_data["identity_set_digest"] = hashlib.sha256(json.dumps(allowlist_data["identities"], sort_keys=True).encode()).hexdigest()
    allowlist.write_text(json.dumps(allowlist_data), encoding="utf-8")
    return root, allowlist


def test_progress_audit_closes_published_and_worker_sets(tmp_path):
    root, allowlist = _root(tmp_path)
    report = audit(root, allowlist, expected_worker_ids={"gpu_0"})
    assert report["status"] == "PASS"
    assert report["valid_sealed_episodes"] == 2
    assert report["per_shard_unique_sum"] == 2
    assert report["protected_reads"] == []
    assert report["protected_read_audit"]["status"] == "PASS"


def test_progress_audit_rejects_worker_duplicate(tmp_path):
    root, allowlist = _root(tmp_path, duplicate=True)
    report = audit(root, allowlist)
    assert report["status"].startswith("HOLD")
    assert report["duplicate_worker_id_count"] == 1


def test_progress_audit_does_not_consume_staging(tmp_path):
    root, allowlist = _root(tmp_path)
    (root / ".episode.staging.1").mkdir(parents=True)
    report = audit(root, allowlist)
    assert report["status"].startswith("HOLD")
    assert report["staging_residue_count"] == 1


def test_progress_audit_rejects_protected_looking_root(tmp_path):
    root = tmp_path / "protected" / "formal"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="forbidden"):
        audit(root)


def test_no_clobber_publish(tmp_path):
    source = tmp_path / "staging"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    if os.name == "posix":
        with pytest.raises(FileExistsError):
            rename_noreplace(source, target)
    else:
        with pytest.raises(RuntimeError, match="unsupported"):
            rename_noreplace(source, target)
