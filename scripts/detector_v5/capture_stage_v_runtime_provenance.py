"""Capture external runtime provenance without importing the science runner."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "commit": _git(path, "rev-parse", "HEAD") if path.is_dir() else None,
        "tree": _git(path, "rev-parse", "HEAD^{tree}") if path.is_dir() else None,
        "status_porcelain": _git(path, "status", "--porcelain") if path.is_dir() else None,
    }


def artifact_binding(path: Path, *, role: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_file():
        return {"role": role, "path": str(path), "kind": "file", "sha256": sha256_file(path)}
    if not path.is_dir():
        return {"role": role, "path": str(path), "kind": "missing"}
    rows = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append({"path": child.relative_to(path).as_posix(), "size": child.stat().st_size, "sha256": sha256_file(child)})
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "role": role, "path": str(path), "kind": "directory", "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "tree_sha256": hashlib.sha256(raw).hexdigest(),
    }


def module_binding(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"name": name, "status": "HOLD", "error": f"FIND_SPEC:{type(exc).__name__}:{exc}"}
    origin = str(spec.origin) if spec and spec.origin and spec.origin != "built-in" else None
    return {
        "name": name,
        "status": "PASS" if spec is not None else "HOLD",
        "origin": origin,
        "origin_sha256": sha256_file(Path(origin)) if origin and Path(origin).is_file() else None,
    }


def _python_binding(python_path: Path) -> dict[str, Any]:
    requested_path = Path(python_path)
    python_path = requested_path.resolve()
    try:
        version = subprocess.check_output(
            [str(python_path), "-c", "import platform; print(platform.python_version())"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        version = None
        error = f"PYTHON_VERSION:{type(exc).__name__}:{exc}"
    else:
        error = None
    return {
        "path": str(requested_path),
        "resolved_path": str(python_path),
        "exists": python_path.is_file(),
        "sha256": sha256_file(python_path) if python_path.is_file() else None,
        "version": version,
        "error": error,
    }


def build(
    *,
    python_path: Path,
    source_worktree: Path,
    snapshot_root: Path | None = None,
    upstream_root: Path | None = None,
    modules: Sequence[str] = (),
    artifacts: Sequence[tuple[str, Path]] = (),
    files: Sequence[Path] = (),
    diagnostic_gpu_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    python = _python_binding(Path(python_path))
    source = git_binding(Path(source_worktree))
    snapshot = git_binding(Path(snapshot_root)) if snapshot_root else None
    upstream = git_binding(Path(upstream_root)) if upstream_root else None
    imported = [module_binding(name) for name in sorted(set(str(item) for item in modules))]
    bound_artifacts = [artifact_binding(path, role=role) for role, path in artifacts]
    bound_files = [{"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve()) if Path(path).is_file() else None} for path in files]
    errors = []
    if not python["exists"] or not python["version"]:
        errors.append("OFFICIAL_PYTHON_UNAVAILABLE")
    if not source["exists"] or not source["commit"] or not source["tree"]:
        errors.append("SOURCE_WORKTREE_BINDING_INCOMPLETE")
    errors.extend(f"MODULE_UNAVAILABLE:{item['name']}" for item in imported if item["status"] != "PASS")
    errors.extend(f"ARTIFACT_MISSING:{item['role']}" for item in bound_artifacts if item["kind"] == "missing")
    errors.extend(f"FILE_MISSING:{item['path']}" for item in bound_files if item["sha256"] is None)
    return {
        "schema": "STAGE_V_EXTERNAL_RUNTIME_PROVENANCE_V1",
        "status": "PASS_RUNTIME_PROVENANCE_CAPTURED" if not errors else "HOLD_RUNTIME_PROVENANCE_INCOMPLETE",
        "official_python": python,
        "source_worktree": source,
        "snapshot_root": snapshot,
        "upstream_root": upstream,
        "imported_modules": imported,
        "artifacts": bound_artifacts,
        "files": bound_files,
        "diagnostic_gpu_identity": dict(diagnostic_gpu_identity) if diagnostic_gpu_identity else None,
        "protected_counters": dict(COUNTERS),
        "errors": sorted(set(errors)),
        "runtime_authorized": False,
        "outcomes_read": False,
        "intervention_executed": False,
        "python_observer_version": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", dest="python_path", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--file", action="append", type=Path, default=[])
    parser.add_argument("--gpu-identity-json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifacts = []
    for item in args.artifact:
        role, separator, path = item.partition("=")
        if not separator or not role or not path:
            raise SystemExit(f"ARTIFACT_FORMAT:{item}")
        artifacts.append((role, Path(path)))
    gpu_identity = json.loads(args.gpu_identity_json) if args.gpu_identity_json else None
    receipt = build(
        python_path=args.python_path, source_worktree=args.source_worktree,
        snapshot_root=args.snapshot_root, upstream_root=args.upstream_root,
        modules=args.module, artifacts=artifacts, files=args.file,
        diagnostic_gpu_identity=gpu_identity,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    output.with_name(output.name + ".sha256").write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output), "errors": receipt["errors"]}, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
