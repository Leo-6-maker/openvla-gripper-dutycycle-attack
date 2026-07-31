from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DETECTOR_SCRIPTS = ROOT / "scripts" / "detector_v5"
if str(DETECTOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SCRIPTS))

import hashlib
import json
import pytest

from d8_source_contract import (
    SourceContractError,
    load_and_validate_source_snapshot,
    verify_sha256_manifest,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_snapshot_validates_exact_bytes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n")
    (repo / "b.json").write_text("{}\n")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({
        "schema": "SOURCE_SNAPSHOT_V2",
        "executable_source_commit": "1" * 40,
        "executable_source_tree": "2" * 40,
        "file_sha256_map": {
            "a.py": sha(repo / "a.py"),
            "b.json": sha(repo / "b.json"),
        },
    }))
    result = load_and_validate_source_snapshot(snapshot, repo, ("a.py",))
    assert result["executable_source_commit"] == "1" * 40
    (repo / "a.py").write_text("tampered\n")
    with pytest.raises(SourceContractError):
        load_and_validate_source_snapshot(snapshot, repo, ("a.py",))


def test_manifest_verifies_consumed_json_but_allows_unsealed_png(tmp_path: Path):
    root = tmp_path / "telemetry"
    root.mkdir()
    episode = root / "episode.json"
    episode.write_text("{}\n")
    (root / "frame.png").write_bytes(b"png")
    (root / "SHA256SUMS").write_text(f"{sha(episode)}  episode.json\n")
    manifest_sha = sha(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{manifest_sha}  SHA256SUMS\n")
    receipt = verify_sha256_manifest(
        root, required_files=(episode,), require_all_files_listed=False
    )
    assert receipt["listed_file_count"] == 1
    with pytest.raises(SourceContractError):
        verify_sha256_manifest(root, require_all_files_listed=True)
