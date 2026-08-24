#!/usr/bin/env python3
"""Static, provenance-only audit for the current X1R2 runtime.

This intentionally imports no model and constructs no simulator.  It binds the
Git source, suite model trees, frozen Student artifacts, and interpreter
package surface before any Q3R2 fixture is exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def tree_digest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"file_count": len(rows), "bytes": sum(row["size"] for row in rows), "tree_sha256": hashlib.sha256(raw).hexdigest()}


def package_surface() -> dict[str, Any]:
    names = ("torch", "transformers", "numpy", "PIL", "cv2", "mujoco", "robosuite", "libero", "tokenizers", "safetensors", "accelerate", "torchvision")
    packages = {}
    for name in names:
        module = importlib.import_module(name)
        packages[name] = {"version": getattr(module, "__version__", None), "file": str(getattr(module, "__file__", ""))}
    freeze = subprocess.check_output([sys.executable, "-s", "-m", "pip", "freeze"], text=True)
    pyvenv = Path(sys.prefix) / "pyvenv.cfg"
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "include_system_site_packages": "include-system-site-packages = true" in pyvenv.read_text(encoding="utf-8") if pyvenv.is_file() else None,
        "sys_path": sys.path,
        "packages": packages,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def audit(config: dict[str, Any], expected_commit: str | None) -> dict[str, Any]:
    errors: list[str] = []
    binding = config["source_binding"]
    runtime_commit = expected_commit or str(binding["runtime_code_commit"])
    runtime_tree = str(binding["runtime_code_tree"])
    observed_head = git("rev-parse", "HEAD")
    observed_tree = git("rev-parse", "HEAD^{tree}")
    if git("status", "--porcelain"):
        errors.append("RUNTIME_WORKTREE_DIRTY")
    if git("rev-parse", f"{runtime_commit}^{{tree}}") != runtime_tree:
        errors.append("RUNTIME_CODE_TREE_BINDING_INVALID")

    source_rows = []
    for row in config["source_binding"]["runtime_files"]:
        path = REPO / str(row["path"])
        if not path.is_file():
            errors.append(f"RUNTIME_FILE_MISSING:{row['path']}")
            continue
        actual_blob = git("hash-object", "--no-filters", str(path))
        expected_blob = git("rev-parse", f"{runtime_commit}:{row['path']}")
        if actual_blob != expected_blob or actual_blob != row["git_blob_sha"]:
            errors.append(f"RUNTIME_FILE_BLOB_MISMATCH:{row['path']}")
        source_rows.append({"path": row["path"], "git_blob_sha": actual_blob, "raw_sha256": sha256_file(path)})

    contract_ref = config["victim_contract"]
    contract_path = REPO / str(contract_ref["repo_path"])
    contract = json_load(contract_path)
    if contract.get("status") != "FROZEN_FOR_CLEAN_PARITY_ONLY" or contract.get("scientific_authority") != "X1R_NOT_AUTHORIZED":
        errors.append("VICTIM_CONTRACT_SCOPE_INVALID")
    if git("hash-object", "--no-filters", str(contract_path)) != contract_ref["git_blob_sha"]:
        errors.append("VICTIM_CONTRACT_BLOB_MISMATCH")

    model_rows = {}
    for suite, suite_cfg in contract["suites"].items():
        model_path = Path(str(suite_cfg["model_path"]))
        if not model_path.is_dir():
            errors.append(f"MODEL_DIR_MISSING:{suite}")
            continue
        observed = tree_digest(model_path)
        expected = suite_cfg["model_identity"]
        if any(observed[key] != expected[key] for key in ("file_count", "bytes", "tree_sha256")):
            errors.append(f"MODEL_TREE_MISMATCH:{suite}")
        key_files = {}
        for relative, expected_sha in expected.get("key_files", {}).items():
            path = model_path / relative
            actual = sha256_file(path) if path.is_file() else "MISSING"
            key_files[relative] = actual
            if actual != expected_sha:
                errors.append(f"MODEL_KEY_MISMATCH:{suite}:{relative}")
        model_rows[suite] = {"path": str(model_path), "observed": observed, "key_files": key_files, "unnorm_key": suite_cfg["unnorm_key"]}

    student_rows = []
    for row in config["student_binding"]["artifacts"]:
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = REPO / path
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != row["sha256"]:
            errors.append(f"STUDENT_ARTIFACT_MISMATCH:{row['name']}")
        student_rows.append({"name": row["name"], "path": str(path), "sha256": actual})

    environment = package_surface()
    expected_environment = config["environment"]
    if environment["executable"] != expected_environment["python_realpath"] or environment["prefix"] != expected_environment["prefix"] or environment["base_prefix"] != expected_environment["base_prefix"] or not environment["python"].startswith(expected_environment["python_version"]):
        errors.append("PYTHON_RUNTIME_BINDING_MISMATCH")
    if environment["include_system_site_packages"] != expected_environment["include_system_site_packages"]:
        errors.append("PYVENV_SITE_PACKAGE_POLICY_MISMATCH")
    expected_packages = expected_environment["packages"]
    for name, expected in expected_packages.items():
        actual = environment["packages"].get(name, {})
        if actual.get("version") != expected["version"] or actual.get("file") != expected["file"]:
            errors.append(f"PACKAGE_BINDING_MISMATCH:{name}")
    if environment["pip_freeze_sha256"] != expected_environment["pip_freeze_sha256"]:
        errors.append("PIP_FREEZE_MISMATCH")

    return {
        "schema": "STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_AUDIT_V1",
        "status": "STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_PASS" if not errors else "HOLD_Q3R2_RUNTIME_AUTHORITY",
        "scope": "static provenance only; no model inference, simulator construction, env.step, attack, V_phys, Eval160, or protected read",
        "source": {"observed_head": observed_head, "observed_tree": observed_tree, "runtime_code_commit": runtime_commit, "runtime_code_tree": runtime_tree, "runtime_files": source_rows},
        "victim_contract": {"path": str(contract_path), "git_blob_sha": git("hash-object", "--no-filters", str(contract_path)), "status": contract.get("status"), "scientific_authority": contract.get("scientific_authority")},
        "models": model_rows,
        "student_artifacts": student_rows,
        "environment": environment,
        "errors": errors,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json")
    parser.add_argument("--output", type=Path, default=REPO / "reports/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_AUDIT_V1.json")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    report = audit(json_load(args.config), args.expected_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "errors": report["errors"]}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_PASS" else 2)


if __name__ == "__main__":
    main()
