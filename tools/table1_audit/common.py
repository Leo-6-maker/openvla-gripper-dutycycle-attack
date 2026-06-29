from __future__ import annotations

import argparse
import hashlib
import json
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
    "README_RESTORE.txt",
]


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


def write_json(path: Path, data: object) -> None:
    path.write_text(canonical_json(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def job_key(row: dict) -> str:
    return str(row.get("job_key") or row.get("job_id") or "")


def parent_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("fold")), str(row.get("state_id")), str(row.get("detector_seed")))


def replicate_key(row: dict) -> str:
    return str(row.get("perturbation_seed"))


def output_dir(row: dict, manifest_dir: Path) -> Path:
    raw = row.get("output_dir")
    if not raw:
        raise ValueError(f"manifest row lacks output_dir: {job_key(row)}")
    p = Path(str(raw))
    return p if p.is_absolute() else manifest_dir / p


def add_path_arg(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    parser.add_argument(name, type=Path, **kwargs)
