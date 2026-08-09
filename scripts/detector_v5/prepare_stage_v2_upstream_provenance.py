"""Create an explicit clean-path binding for the frozen OpenVLA provenance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False, capture_output=True, text=True,
    ).stdout.strip()


def git_blob_sha1(path: Path) -> str:
    return subprocess.run(
        ["git", "hash-object", "--no-filters", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(
    source: Path,
    clean_root: Path,
    output_dir: Path,
    *,
    expected_source_sha256: str,
    expected_source_blob_sha1: str,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    source = source.resolve()
    clean_root = clean_root.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file() or not clean_root.is_dir():
        raise ValueError("provenance source or clean upstream root is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"provenance output must be new/empty: {output_dir}")
    source_sha256 = sha256_file(source)
    source_blob_sha1 = git_blob_sha1(source)
    if source_sha256 != expected_source_sha256 or source_blob_sha1 != expected_source_blob_sha1:
        raise ValueError("frozen provenance source hash mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen provenance is not a JSON object")
    openvla = payload.get("openvla_upstream")
    if not isinstance(openvla, Mapping) or openvla.get("commit") != expected_commit:
        raise ValueError("frozen provenance OpenVLA commit mismatch")
    actual_commit = git_value(clean_root, "rev-parse", "HEAD")
    actual_tree = git_value(clean_root, "rev-parse", "HEAD^{tree}")
    clean_status = git_value(clean_root, "status", "--porcelain", "--untracked-files=all")
    if actual_commit != expected_commit or actual_tree != expected_tree or clean_status:
        raise ValueError("clean OpenVLA checkout binding mismatch")

    bound = copy.deepcopy(dict(payload))
    bound_openvla = dict(bound["openvla_upstream"])
    original_checkout = str(bound_openvla.get("checkout", ""))
    bound_openvla.update({"checkout": str(clean_root), "tree": actual_tree})
    bound["openvla_upstream"] = bound_openvla
    binding = {
        "schema": "STAGE_V_R2_UPSTREAM_PROVENANCE_BINDING_V1",
        "source_snapshot": {
            "path": str(source),
            "sha256": source_sha256,
            "git_blob_sha1": source_blob_sha1,
            "openvla_checkout": original_checkout,
            "openvla_commit": expected_commit,
            "openvla_tree": git_value(Path(original_checkout), "rev-parse", "HEAD^{tree}") if Path(original_checkout).is_dir() else None,
            "openvla_status": git_value(Path(original_checkout), "status", "--porcelain", "--untracked-files=all") if Path(original_checkout).is_dir() else None,
        },
        "bound_clean_checkout": {
            "path": str(clean_root),
            "commit": actual_commit,
            "tree": actual_tree,
            "status": clean_status,
        },
        "science_definitions_modified": False,
        "generated_by": "prepare_stage_v2_upstream_provenance.py",
    }
    bound["stage_v2_upstream_provenance_binding"] = binding
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "STAGE_V2_UPSTREAM_PROVENANCE_CLEAN.json"
    atomic_write_json(snapshot, bound)
    snapshot_sha256 = sha256_file(snapshot)
    snapshot_blob_sha1 = git_blob_sha1(snapshot)
    (output_dir / "STAGE_V2_UPSTREAM_PROVENANCE_CLEAN.sha256").write_text(
        f"{snapshot_sha256}  {snapshot.name}\n", encoding="utf-8",
    )
    (output_dir / "STAGE_V2_UPSTREAM_PROVENANCE_CLEAN.gitblob").write_text(
        f"{snapshot_blob_sha1}  {snapshot.name}\n", encoding="utf-8",
    )
    audit = {
        **binding,
        "verdict": "PASS",
        "snapshot_path": str(snapshot),
        "snapshot_sha256": snapshot_sha256,
        "snapshot_git_blob_sha1": snapshot_blob_sha1,
    }
    atomic_write_json(output_dir / "STAGE_V2_UPSTREAM_PROVENANCE_BINDING_AUDIT.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-blob-sha1", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    args = parser.parse_args(argv)
    audit = prepare(
        args.source, args.clean_root, args.output_dir,
        expected_source_sha256=args.expected_source_sha256,
        expected_source_blob_sha1=args.expected_source_blob_sha1,
        expected_commit=args.expected_commit,
        expected_tree=args.expected_tree,
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
