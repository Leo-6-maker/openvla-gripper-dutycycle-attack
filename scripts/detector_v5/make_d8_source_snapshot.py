"""Generate an external SOURCE_SNAPSHOT_V2 from an exact Git checkout."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from d8_source_contract import REVIEW_REQUIRED_SOURCE_FILES, sha256_file


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--remote", default="")
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    output_abs = args.output.absolute()
    try:
        output_abs.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("source snapshot output must be outside the Git worktree")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("source snapshot requires a clean Git worktree")
    commit = git(root, "rev-parse", "HEAD")
    tree = git(root, "show", "-s", "--format=%T", "HEAD")

    file_map = {}
    for rel in REVIEW_REQUIRED_SOURCE_FILES:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"required source file missing: {rel}")
        file_map[rel] = sha256_file(path)

    payload = {
        "schema": "SOURCE_SNAPSHOT_V2",
        "executable_source_commit": commit,
        "executable_source_tree": tree,
        "github_branch": args.branch,
        "github_remote": args.remote,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_sha256_map": file_map,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(f"commit={commit}")
    print(f"tree={tree}")
    print(f"files={len(file_map)}")
    print(f"snapshot_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
