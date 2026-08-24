#!/usr/bin/env python3
"""CPU-only integrity check for the paper evidence authority map."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "paper" / "PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(spec: str) -> str:
    return subprocess.check_output(["git", "rev-parse", spec], cwd=ROOT, text=True).strip()


def main() -> int:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    assert data["status"] == "PAPER_V1_EVIDENCE_AUTHORITY_MAP_PASS"
    assert data["unit_policy"]["identity_join"] == "NONE across X0, Black Bowl, VI-B2, VII, VIII, IX, E2, E3, and E4"
    assert data["canonicalization"]["protected_boundary"]["eval160"] == "UNREAD"

    checked = 0
    for source in data["sources"]:
        for path_text, expected in zip(source.get("source_paths", []), source.get("local_artifact_sha256", [])):
            if path_text.startswith("git-history:"):
                continue
            path = ROOT / path_text
            assert path.is_file(), path
            assert sha256(path) == expected, (path_text, sha256(path), expected)
            checked += 1

    stage_viii = next(item for item in data["sources"] if item["id"] == "VIII")
    assert git_blob("34b8d435264737734f7d4e4ecc9a3343e57d7c1:docs/handoffs/STAGE_VIII_R1_RELATIVE_SELECTOR_NEGATIVE_HANDOFF_20260817.md") == stage_viii["immutable_git_blob_sha256"][0]
    assert git_blob("b918104e2e6279364891590fd37a17720dbb6628:docs/handoffs/PR117_STAGE_VIII_DOCUMENTATION_CLOSURE_20260817.md") == stage_viii["immutable_git_blob_sha256"][1]

    e4 = next(item for item in data["sources"] if item["id"] == "E3_E4")
    assert e4["sealed_artifact_sha256"][-1] == sha256(ROOT / "reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/E4_ROOT_SEAL_V1.json")
    assert data["terminal_gate"] == "PAPER_V1_EVIDENCE_AUTHORITY_MAP_PASS"
    print(f"PAPER_V1_EVIDENCE_AUTHORITY_MAP_PASS checked_local_artifacts={checked} checked_immutable_git_sources=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
