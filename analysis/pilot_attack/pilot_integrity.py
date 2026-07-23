"""Fail-closed integrity primitives for pilot analysis (standalone, no PR #98 deps)."""
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


def is_strict_int(v: Any) -> bool:
    """True if v is int but NOT bool."""
    return not isinstance(v, bool) and isinstance(v, int)


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


def require_schema(manifest: dict[str, Any], expected: str, label: str) -> None:
    """Hard-reject wrong schema."""
    actual = manifest.get("schema", "")
    if actual != expected:
        raise SystemExit(f"{label}_SCHEMA: expected={expected!r} got={actual!r}")


def guard_path_safe(rel: str, root: Path, label: str) -> Path:
    """Reject path escapes, absolutes, symlinks."""
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts:
        raise SystemExit(f"{label}_PATH_ESCAPE: {rel}")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{label}_PATH_OUTSIDE: {rel}")
    if resolved.is_symlink():
        raise SystemExit(f"{label}_SYMLINK: {rel}")
    return resolved


def verify_evidence_file(root: Path, rel: str, declared_sha: str | None, label: str) -> None:
    """Verify file exists under root, is safe, non-empty, SHA matches if declared."""
    if not rel or not isinstance(rel, str):
        raise SystemExit(f"{label}_PATH_EMPTY: {rel!r}")
    file_path = guard_path_safe(rel, root, label)
    if not file_path.is_file():
        raise SystemExit(f"{label}_NOT_FOUND: {rel}")
    if file_path.stat().st_size == 0:
        raise SystemExit(f"{label}_EMPTY: {rel}")
    if declared_sha and is_64char_hex(declared_sha):
        actual = sha256_file(file_path)
        if actual != declared_sha:
            raise SystemExit(f"{label}_SHA_MISMATCH: {rel} declared={declared_sha[:16]} actual={actual[:16]}")


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


def seal_dir_in_place(root: Path) -> str:
    """Add SHA256SUMS + .sha256 to existing dir without moving."""
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names))
    seal = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    return seal
