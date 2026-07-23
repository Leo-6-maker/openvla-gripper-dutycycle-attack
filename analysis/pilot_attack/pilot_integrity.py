"""Minimal fail-closed integrity primitives for pilot analysis (standalone, no PR #98 deps)."""
from __future__ import annotations

import hashlib, json, os, uuid
from pathlib import Path
from typing import Any


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    dups: list[str] = []
    def hook(pairs):
        seen = set(); result = {}
        for k, v in pairs:
            if k in seen: dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE: {path} {e}")
    if dups: raise SystemExit(f"{label}_DUP_KEYS: {path}")
    if not isinstance(value, dict): raise SystemExit(f"{label}_NOT_OBJECT: {path}")
    return value


def seal_output_dir(root: Path) -> str:
    import shutil
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    if root.exists(): shutil.rmtree(root, ignore_errors=True)
    try: os.replace(staging, root)
    except OSError:
        if root.exists(): shutil.rmtree(root)
        os.replace(staging, root)
    return seal
