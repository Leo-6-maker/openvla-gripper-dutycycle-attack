#!/usr/bin/env python3
"""Strict duplicate-key-aware V3.1 handoff loader. Single entry point."""
from __future__ import annotations

import hashlib, json, os, sys
from pathlib import Path


def load_handoff_file(path: Path, repo_root: Path) -> dict:
    """Load and validate a V3.1 handoff JSON with full path security."""
    if not path.is_file():
        raise SystemExit(f"HANDOFF_NOT_FOUND: {path}")

    dup_errors = []

    def make_hook(context=""):
        def hook(pairs):
            seen = set()
            for k, v in pairs:
                if k in seen:
                    dup_errors.append(f"DUPLICATE_KEY: '{k}' at {context or 'root'}")
                seen.add(k)
            return dict(pairs)
        return hook

    with open(path) as f:
        raw = f.read()
    try:
        handoff = json.loads(raw, object_pairs_hook=make_hook("root"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"HANDOFF_JSON_PARSE_ERROR: {e}")

    if dup_errors:
        for e in dup_errors:
            print(f"  {e}")
        raise SystemExit(
            f"CODEX_V3_1_HANDOFF = BLOCKED_DUPLICATE_JSON_KEY ({len(dup_errors)} duplicates)\n"
            "AUTHORITATIVE_L3 = HOLD"
        )

    schema = handoff.get("schema", "")
    if schema != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1":
        if "HANDOFF_V2" in schema:
            raise SystemExit("CODEX_V2_HANDOFF = STATIC_REJECTED")
        raise SystemExit(f"UNKNOWN_HANDOFF_SCHEMA: {schema}")

    _validate_all_refs(handoff, repo_root)
    return handoff


def validate_ref_path(rel, repo_root):
    """Validate one repo-relative path for security. Raises SystemExit."""
    if not isinstance(rel, str):
        raise SystemExit(f"PATH_NOT_STRING: {rel}")
    if Path(rel).is_absolute():
        raise SystemExit(f"PATH_ABSOLUTE: {rel}")
    parts = Path(rel).parts
    if ".." in parts:
        raise SystemExit(f"PATH_ESCAPE: {rel}")
    candidate = repo_root / rel
    # Check each component for symlink BEFORE resolve
    current = repo_root
    for part in parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise SystemExit(f"PATH_SYMLINK: {rel} at component {part}")
    resolved = candidate.resolve(strict=True)
    repo_resolved = repo_root.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        raise SystemExit(f"PATH_OUTSIDE_REPO: {rel} -> {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"REF_NOT_FOUND: {rel}")
    return resolved


def verify_ref_sha(resolved, expected_sha, label):
    """Verify SHA256 of resolved file matches expected."""
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise SystemExit(f"REF_SHA_INVALID: {label} not 64-char hex")
    d = hashlib.sha256()
    with open(resolved, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    actual = d.hexdigest()
    if actual != expected_sha:
        raise SystemExit(f"REF_SHA_MISMATCH: {label} exp={expected_sha[:16]} act={actual[:16]}")


def _validate_all_refs(obj, repo_root, prefix=""):
    """Recursively validate all {path, sha256} references."""
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj and prefix:
            resolved = validate_ref_path(obj["path"], repo_root)
            verify_ref_sha(resolved, obj["sha256"], prefix)
        for k, v in obj.items():
            _validate_all_refs(v, repo_root, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _validate_all_refs(v, repo_root, f"{prefix}[{i}]")
