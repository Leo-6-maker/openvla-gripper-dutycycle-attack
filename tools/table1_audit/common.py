from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


REQUIRED_BUNDLE_FILES = [
    "MANIFEST.sha256",
    "accepted_job_keys.txt",
    "RESULT_INVENTORY.json",
    "PROVENANCE_AUDIT.json",
    "PAIRING_AUDIT.json",
    "ARTIFACT_SHA256SUMS.txt",
    "CONDITION_RESULTS.json",
    "CONDITION_FREEZE.json",
    "BUNDLE_INVENTORY.json",
    "BUNDLE_SHA256SUMS.txt",
    "README_RESTORE.txt",
]

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def is_valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(HEX64_RE.fullmatch(value))


def first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{i}: malformed jsonl: {exc}") from exc
    return rows


def canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def canonical_digest(data: object) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(canonical_json(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def job_key(row: dict) -> str:
    return str(first_non_none(row.get("job_key"), row.get("job_id"), ""))


def parent_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("fold")), str(first_non_none(row.get("task_id"), row.get("task"), "")), str(row.get("state_id")), str(row.get("detector_seed")))


def replicate_key(row: dict) -> str:
    return str(row.get("perturbation_seed"))


def safe_relative_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or not value or value in {".", os.curdir}:
        raise ValueError(f"unsafe relative path: {value}")
    return p


def ensure_within_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved == root_resolved:
        raise ValueError(f"path equals root: {path}")
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {path}") from exc
    return resolved


def reject_symlink(path: Path, root: Path | None = None) -> None:
    root_resolved = root.resolve() if root else None
    probe = path.resolve() if path.exists() else path.absolute()
    parts = probe.parts
    start = 1
    if root_resolved:
        try:
            rel = probe.relative_to(root_resolved)
            parts = root_resolved.parts + rel.parts
            start = len(root_resolved.parts)
        except ValueError:
            pass
    cur = Path(*parts[:start]) if start else Path(parts[0])
    for part in parts[start:]:
        cur = cur / part
        if cur.exists() and cur.is_symlink():
            raise ValueError(f"symlink path rejected: {cur}")


def output_dir(row: dict, manifest_dir: Path, condition_root: Path | None = None) -> Path:
    raw = row.get("output_dir")
    if not raw:
        raise ValueError(f"manifest row lacks output_dir: {job_key(row)}")
    p = Path(str(raw))
    out = p if p.is_absolute() else manifest_dir / p
    if condition_root:
        if ".." in p.parts:
            raise ValueError(f"output_dir contains traversal: {raw}")
        reject_symlink(out, condition_root)
        return ensure_within_root(out, condition_root)
    return out


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(text)
    tmp.replace(path)


def parse_manifest(path: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    problems: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append({"class": "malformed_manifest_row", "line": line_no, "detail": str(exc)})
                continue
            if not isinstance(row, dict):
                problems.append({"class": "malformed_manifest_row", "line": line_no, "detail": "row is not an object"})
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows, problems


def read_sha256sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not is_valid_sha256(parts[0]):
            raise ValueError(f"{path}:{line_no}: malformed sha256sum line")
        rel = parts[1].strip()
        if rel.startswith("*"):
            rel = rel[1:]
        safe_relative_path(rel)
        out[rel] = parts[0]
    return out


def classify_artifact_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".csv"):
        return "csv"
    return path.suffix.lower().lstrip(".") or "file"


def validate_artifact_syntax(path: Path) -> str | None:
    if path.stat().st_size == 0:
        return "zero-byte artifact"
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            load_json(path)
        elif suffix == ".jsonl":
            load_jsonl(path)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            if not header or not any(str(c).strip() for c in header):
                return "empty csv header"
    except Exception as exc:
        return str(exc)
    return None


def recursive_inventory(root: Path, *, reject_links: bool = True) -> list[dict]:
    root = root.resolve()
    rows: list[dict] = []
    for p in sorted(root.rglob("*")):
        if reject_links and p.is_symlink():
            raise ValueError(f"symlink artifact rejected: {p}")
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        safe_relative_path(rel)
        before = p.stat()
        digest = sha256_file(p)
        after = p.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
            raise ValueError(f"source artifact changed during hashing: {p}")
        rows.append({
            "relative_path": rel,
            "sha256": digest,
            "size_bytes": after.st_size,
            "artifact_type": classify_artifact_type(p),
            "job_key": "",
        })
    return rows


def add_path_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    parser.add_argument(name, type=Path, **kwargs)
