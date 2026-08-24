from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from scripts.detector_v5.prepare_stage_v2_upstream_provenance import git_blob_sha1, prepare, sha256_file


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def test_clean_path_binding_preserves_dirty_source_lineage(tmp_path: Path) -> None:
    clean_root = tmp_path / "openvla-clean"
    clean_root.mkdir()
    _git(clean_root, "init", "-q")
    (clean_root / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(clean_root, "add", "tracked.txt")
    subprocess.run(
        ["git", "-C", str(clean_root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "baseline"],
        check=True,
    )
    commit = _git(clean_root, "rev-parse", "HEAD")
    tree = _git(clean_root, "rev-parse", "HEAD^{tree}")

    original_root = tmp_path / "openvla-original"
    shutil.copytree(clean_root, original_root)
    (original_root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    source = tmp_path / "UPSTREAM_PROVENANCE.json"
    source.write_text(json.dumps({"openvla_upstream": {"checkout": str(original_root), "commit": commit}}), encoding="utf-8")
    out = tmp_path / "binding"

    audit = prepare(
        source, clean_root, out,
        expected_source_sha256=sha256_file(source),
        expected_source_blob_sha1=git_blob_sha1(source),
        expected_commit=commit,
        expected_tree=tree,
    )

    assert audit["verdict"] == "PASS"
    assert audit["source_snapshot"]["openvla_status"]
    bound = json.loads((out / "STAGE_V2_UPSTREAM_PROVENANCE_CLEAN.json").read_text(encoding="utf-8"))
    assert bound["openvla_upstream"]["checkout"] == str(clean_root.resolve())
    assert bound["openvla_upstream"]["tree"] == tree
    assert bound["stage_v2_upstream_provenance_binding"]["science_definitions_modified"] is False
