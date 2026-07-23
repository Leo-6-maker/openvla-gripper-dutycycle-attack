"""Fail-closed integrity primitives for pilot analysis v2.2 (standalone, no PR #98 deps)."""
from __future__ import annotations

import hashlib, json, math, os, uuid
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
    return not isinstance(v, bool) and isinstance(v, int)


def is_finite_number(v: Any) -> bool:
    """Reject bool, NaN, Inf; accept finite int or float."""
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


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
    actual = manifest.get("schema", "")
    if actual != expected:
        raise SystemExit(f"{label}_SCHEMA: expected={expected!r} got={actual!r}")


def require_nonempty_list(value: Any, label: str) -> list:
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label}_EMPTY_OR_NOT_LIST")
    return value


# ── Fix 1: Sealed root verification ────────────────────────────────────────

def verify_sealed_root(root: Path, label: str) -> str:
    """Verify SHA256SUMS + .sha256, return seal SHA. Raises SystemExit on any violation."""
    bp = root.resolve()
    if bp.is_symlink():
        raise SystemExit(f"{label}_ROOT_SYMLINK")
    if not bp.is_dir():
        raise SystemExit(f"{label}_NOT_DIR")
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if sums.is_symlink() or sidecar.is_symlink():
        raise SystemExit(f"{label}_SEAL_SYMLINK")
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED")
    expected_seal = sha256_file(sums)
    sidecar_line = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_line or sidecar_line[0] != expected_seal:
        raise SystemExit(f"{label}_SIDECAR_BROKEN")
    listed: set[str] = set()
    with open(sums, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: raise SystemExit(f"{label}_SEAL_PARSE: {line}")
            file_sha, rel = parts[0], " ".join(parts[1:])
            if not is_64char_hex(file_sha): raise SystemExit(f"{label}_SEAL_SHA_INVALID: {rel}")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise SystemExit(f"{label}_SEAL_ESCAPE: {rel}")
            target = bp / rel_path
            if target.is_symlink(): raise SystemExit(f"{label}_SEAL_SYMLINK: {rel}")
            try: target.resolve().relative_to(bp)
            except ValueError: raise SystemExit(f"{label}_SEAL_ESCAPE: {rel}")
            if rel in listed: raise SystemExit(f"{label}_SEAL_DUP: {rel}")
            listed.add(rel)
            if not target.is_file() or sha256_file(target) != file_sha:
                raise SystemExit(f"{label}_SEAL_MISMATCH: {rel}")
    for p in bp.rglob("*"):
        if p.is_symlink(): raise SystemExit(f"{label}_SEAL_SYMLINK: {p.relative_to(bp).as_posix()}")
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            if p.relative_to(bp).as_posix() not in listed:
                raise SystemExit(f"{label}_SEAL_EXTRA: {p.relative_to(bp).as_posix()}")
    return expected_seal


def consume_sealed_root(root: Path, expected_schema: str, label: str) -> tuple[dict[str, Any], str]:
    """Verify sealed root, find JSON, check schema. Returns (data, seal_sha)."""
    seal = verify_sealed_root(root, label)
    json_files = sorted(p for p in root.iterdir()
                        if p.is_file() and p.suffix == ".json"
                        and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    if len(json_files) != 1:
        raise SystemExit(f"{label}_AMBIGUOUS_JSON: found {len(json_files)}")
    data = load_strict_json(json_files[0], label)
    require_schema(data, expected_schema, label)
    return data, seal


# ── Fix 9: Path safety with component-level symlink check ───────────────────

def guard_path_safe(rel: str, root: Path, label: str) -> Path:
    """Reject path escapes, absolutes, symlinks at every component."""
    if not isinstance(rel, str) or not rel:
        raise SystemExit(f"{label}_PATH_EMPTY: {rel!r}")
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts:
        raise SystemExit(f"{label}_PATH_ESCAPE: {rel}")
    # Check each component for symlink
    current = root.resolve()
    for part in parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise SystemExit(f"{label}_COMPONENT_SYMLINK: {rel} at {part}")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{label}_PATH_OUTSIDE: {rel}")
    return resolved


# ── Fix 7: Evidence file verification (SHA mandatory) ───────────────────────

def verify_evidence_file(root: Path, rel: str, declared_sha: str, label: str) -> None:
    """Verify file exists, safe, non-empty, SHA matches. SHA is MANDATORY."""
    if not rel or not isinstance(rel, str):
        raise SystemExit(f"{label}_PATH_EMPTY_OR_TYPE: {rel!r}")
    file_path = guard_path_safe(rel, root, label)
    if not file_path.is_file():
        raise SystemExit(f"{label}_NOT_FOUND: {rel}")
    if file_path.stat().st_size == 0:
        raise SystemExit(f"{label}_EMPTY: {rel}")
    if not is_64char_hex(declared_sha):
        raise SystemExit(f"{label}_SHA_MISSING_OR_INVALID: {declared_sha[:40]!r}")
    actual = sha256_file(file_path)
    if actual != declared_sha:
        raise SystemExit(f"{label}_SHA_MISMATCH: {rel} declared={declared_sha[:16]} actual={actual[:16]}")


# ── Output sealing ─────────────────────────────────────────────────────────

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
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(root / name)}  {name}\n" for name in names))
    seal = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    return seal
