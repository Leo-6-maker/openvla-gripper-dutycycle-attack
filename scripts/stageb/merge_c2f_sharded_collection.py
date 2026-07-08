#!/usr/bin/env python3
"""Merge isolated C2f collection worker roots into one canonical collection root.

Input should be the parent root created by launch_c2f_clean2000_sharded.py, or
its `shards/` subdirectory.  The script copies or hardlinks episode directories
from:

  <launch-root>/shards/<suite>/worker_<NN>/episodes/<suite>/<parent_key>/

into:

  <merged-root>/episodes/<suite>/<parent_key>/

It then writes a canonical collection manifest and SHA256SUMS.  It does not run
LIBERO/OpenVLA and does not read D7 outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    raise ValueError(f"Unknown mode={mode}")


def copy_tree(src_dir: Path, dst_dir: Path, mode: str) -> int:
    n = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        copy_file(src, dst, mode)
        n += 1
    return n


def write_sha256s(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append((sha256_file(p), p.relative_to(root).as_posix()))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{h}  {rel}\n" for h, rel in rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge C2f sharded collection roots")
    ap.add_argument("--sharded-root", required=True, help="Launch root or launch-root/shards")
    ap.add_argument("--output-root", required=True, help="Canonical merged C2f collection root")
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    input_root = Path(args.sharded_root)
    shard_root = input_root / "shards" if (input_root / "shards").exists() else input_root
    out = Path(args.output_root)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(shard_root.glob("*/*/episodes/*/*/episode_metadata.json"))
    if not meta_paths:
        raise RuntimeError(f"No shard episode metadata found under {shard_root}")

    episodes: List[Dict[str, Any]] = []
    collisions: List[str] = []
    copied_files = 0
    for meta_path in meta_paths:
        src_ep = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        suite = str(meta.get("suite"))
        parent_key = str(meta.get("parent_key"))
        dst_ep = out / "episodes" / suite / parent_key
        if dst_ep.exists():
            collisions.append(f"{suite}/{parent_key}")
            continue
        copied_files += copy_tree(src_ep, dst_ep, args.mode)
        episodes.append({
            "suite": suite,
            "parent_key": parent_key,
            "task_index": int(meta.get("task_index", -1)),
            "task_name": str(meta.get("task_name", "")),
            "task_language": str(meta.get("task_language", "")),
            "n_steps": int(meta.get("n_steps", 0)),
            "clean_success": bool(meta.get("clean_success", False)),
            "episode_dir": str((out / "episodes" / suite / parent_key)),
            "source_shard_episode_dir": str(src_ep),
        })

    status = "PASS_MERGED" if not collisions else "FAIL_COLLISIONS"
    manifest = {
        "schema": "C2F_OBS_LANG_CLEAN_COLLECTION_V1",
        "created_at_unix": time.time(),
        "git_commit": args.git_commit,
        "source_commit": args.git_commit,
        "n_episodes": len(episodes),
        "episodes": episodes,
        "merge": {
            "input_root": str(input_root),
            "shard_root": str(shard_root),
            "mode": args.mode,
            "copied_files": copied_files,
            "collisions": collisions,
            "runtime_seconds": time.time() - t0,
        },
        "boundaries": {
            "condition": "CLEAN_ONLY",
            "attack": "NOT_PERFORMED",
            "d7b2_outcome_read": False,
            "merged_from_isolated_worker_roots": True,
        },
    }
    write_json(out / "manifest.json", manifest)
    write_sha256s(out)
    print(json.dumps({"status": status, "n_episodes": len(episodes), "collisions": len(collisions), "output_root": str(out)}, indent=2))
    return 0 if not collisions else 2


if __name__ == "__main__":
    raise SystemExit(main())
