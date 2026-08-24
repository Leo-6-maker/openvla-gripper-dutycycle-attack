"""Read-only field-level comparison for two sealed G-REC materializations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_file_map(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing seal: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if side != [sha256_file(sums), "SHA256SUMS"]:
        raise RuntimeError(f"sidecar mismatch: {root}")
    expected: dict[str, str] = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(None, 1)
        name = name.lstrip("*").strip()
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise RuntimeError(f"unsafe sealed path: {name}")
        target = root / path
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise RuntimeError(f"sealed payload mismatch: {target}")
        expected[path.as_posix()] = digest
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if expected != actual:
        raise RuntimeError(f"sealed closure mismatch: {root}")
    return expected


def structured(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_bytes()


def diff_values(left: Any, right: Any, path: str, out: list[dict[str, Any]]) -> None:
    if type(left) is not type(right):
        out.append({"field": path, "left": left, "right": right})
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                out.append({"field": child, "left": left.get(key), "right": right.get(key)})
            else:
                diff_values(left[key], right[key], child, out)
        return
    if isinstance(left, list):
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                out.append({"field": child, "left": left[index] if index < len(left) else None, "right": right[index] if index < len(right) else None})
            else:
                diff_values(left[index], right[index], child, out)
        return
    if left != right:
        out.append({"field": path, "left": left, "right": right})


def compare(left_root: Path, right_root: Path) -> dict[str, Any]:
    left_files = sealed_file_map(left_root)
    right_files = sealed_file_map(right_root)
    changes: list[dict[str, Any]] = []
    for name in sorted(set(left_files) | set(right_files)):
        if name not in left_files or name not in right_files:
            changes.append({"file": name, "field": "<file>", "left": left_files.get(name), "right": right_files.get(name)})
            continue
        left_path, right_path = left_root / name, right_root / name
        if left_files[name] == right_files[name]:
            continue
        local: list[dict[str, Any]] = []
        try:
            diff_values(structured(left_path), structured(right_path), "", local)
        except (UnicodeDecodeError, json.JSONDecodeError):
            local = [{"field": "<bytes>", "left": left_files[name], "right": right_files[name]}]
        for item in local:
            changes.append({"file": name, **item})
    file_counts: dict[str, int] = {}
    for change in changes:
        file_counts[change["file"]] = file_counts.get(change["file"], 0) + 1
    return {
        "schema": "V23_G_REC_FIELD_COMPARISON_V1",
        "status": "PASS" if not changes else "HOLD_A_B_FIELD_DIFFERENCE",
        "run_A": {"root": str(left_root), "sha256s_sha256": sha256_file(left_root / "SHA256SUMS"), "file_count": len(left_files)},
        "run_B": {"root": str(right_root), "sha256s_sha256": sha256_file(right_root / "SHA256SUMS"), "file_count": len(right_files)},
        "changed_file_count": len(file_counts),
        "changed_field_count": len(changes),
        "changed_fields_by_file": dict(sorted(file_counts.items())),
        "top_100_changed_fields": changes[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.run_a, args.run_b)
    except Exception as exc:
        result = {"schema": "V23_G_REC_FIELD_COMPARISON_V1", "status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
