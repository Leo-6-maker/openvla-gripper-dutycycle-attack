#!/usr/bin/env python3
"""Fail closed if registered scientific authority paths or bytes change."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json"


def git(*args: str, binary: bool = False) -> bytes | str:
    value = subprocess.check_output(["git", *args], cwd=ROOT)
    return value if binary else value.decode().strip()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def crlf_materialization(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def local_path(path_text: str) -> Path:
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path_text:
        raise ValueError(f"unsafe registry path: {path_text}")
    path = ROOT.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(path_text)
    return path


def canonical_local(entry: dict[str, object]) -> bytes:
    path_text = str(entry["path"])
    path = local_path(path_text)
    committed_blob = str(git("rev-parse", f"HEAD:{path_text}"))
    if committed_blob != entry["git_blob"]:
        raise ValueError(f"committed Git blob changed: {path_text}")
    working_blob = str(git("hash-object", f"--path={path_text}", str(path)))
    if working_blob != committed_blob:
        raise ValueError(f"working-tree bytes changed: {path_text}")
    return bytes(git("cat-file", "blob", committed_blob, binary=True))


def audit_entry(entry: dict[str, object]) -> None:
    path_text = str(entry["path"])
    if not entry.get("path_sensitive") or not entry.get("byte_sensitive") or not entry.get("immutable_by_governance"):
        raise ValueError(f"registry weakens immutable flags: {path_text}")
    if not entry.get("authority_types") or not entry.get("referencing_artifacts"):
        raise ValueError(f"registry entry lacks authority provenance: {path_text}")

    storage = entry.get("storage")
    if storage == "working_tree":
        payload = canonical_local(entry)
    elif storage == "git_object":
        spec = str(entry["git_spec"])
        blob = str(git("rev-parse", spec))
        if blob != entry["git_blob"]:
            raise ValueError(f"historical Git blob changed: {path_text}")
        payload = bytes(git("cat-file", "blob", blob, binary=True))
    else:
        raise ValueError(f"unknown storage class for {path_text}: {storage}")

    if sha256(payload) != entry["current_sha256"]:
        raise ValueError(f"canonical SHA-256 changed: {path_text}")
    for declared in entry.get("declared_digests", []):
        basis = declared["basis"]
        if basis == "git_blob":
            actual = sha256(payload)
        elif basis == "crlf_materialization":
            actual = sha256(crlf_materialization(payload))
        else:
            raise ValueError(f"unknown digest basis for {path_text}: {basis}")
        if actual != declared["sha256"]:
            raise ValueError(f"declared authority digest changed: {path_text} ({basis})")


def audit() -> tuple[dict[str, object], list[str]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    errors: list[str] = []
    if registry.get("schema") != "IMMUTABLE_AUTHORITY_PATHS_V1":
        errors.append("registry schema mismatch")
    if registry.get("status") != "CODE_R1_AUTHORITY_FIREWALL_PASS":
        errors.append("registry status mismatch")

    try:
        source_head = str(registry["source_head"])
        subprocess.check_call(["git", "merge-base", "--is-ancestor", source_head, "HEAD"], cwd=ROOT)
        if git("rev-parse", f"{source_head}^{{tree}}") != registry["source_tree"]:
            errors.append("registry source tree mismatch")
    except (KeyError, subprocess.CalledProcessError) as exc:
        errors.append(f"registry source ancestry invalid: {exc}")

    entries = registry.get("entries", [])
    paths = [str(entry.get("path")) for entry in entries]
    if len(paths) != len(set(paths)):
        errors.append("duplicate registry path")
    if len(entries) != registry.get("entry_count"):
        errors.append("registry entry count mismatch")

    for entry in entries:
        try:
            audit_entry(entry)
        except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(str(exc))

    by_path = {str(entry.get("path")): entry for entry in entries}
    for authority_input in registry.get("authority_inputs", []):
        if authority_input not in by_path:
            errors.append(f"authority input is not registered: {authority_input}")
    if registry.get("compatibility_mappings"):
        errors.append("compatibility mappings require an explicit auditor update")

    for pair in registry.get("sidecar_pairs", []):
        artifact = str(pair["artifact"])
        sidecar = str(pair["sidecar"])
        if artifact not in by_path or sidecar not in by_path:
            errors.append(f"incomplete root/sidecar pair: {artifact} -> {sidecar}")
            continue
        try:
            payload = canonical_local(by_path[artifact])
            basis = pair["digest_basis"]
            expected = sha256(payload if basis == "git_blob" else crlf_materialization(payload))
            token = local_path(sidecar).read_text(encoding="utf-8").split()[0].lower()
            if token != expected:
                errors.append(f"root/sidecar digest mismatch: {artifact} -> {sidecar}")
        except (IndexError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(str(exc))
    return registry, errors


def main() -> int:
    registry, errors = audit()
    if errors:
        print("CODE_R1_AUTHORITY_FIREWALL_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "CODE_R1_AUTHORITY_FIREWALL_PASS "
        f"entries={registry['entry_count']} sidecar_pairs={len(registry['sidecar_pairs'])} "
        f"git_objects={registry['git_object_entry_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
