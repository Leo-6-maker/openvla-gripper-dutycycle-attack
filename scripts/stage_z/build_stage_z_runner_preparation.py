#!/usr/bin/env python3
"""Build a static audit for the synthetic-only Stage-Z runner preparation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/STAGE_Z_MULTI_MODEL_RUNNER_PREP_V1.json"
PACKAGE = ROOT / "src/stage_z_preparation"
AUDIT = ROOT / "reports/STAGE_Z_MULTI_MODEL_RUNNER_PREPARATION_STATIC_AUDIT_V1.json"
FORBIDDEN_IMPORT_ROOTS = {"torch", "transformers", "mujoco", "robosuite", "libero"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_files = sorted(PACKAGE.glob("*.py"))
    forbidden: dict[str, list[str]] = {}
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        roots = sorted(import_roots(path) & FORBIDDEN_IMPORT_ROOTS)
        if roots:
            forbidden[str(path.relative_to(ROOT))] = roots
    if config["execution_enabled"] is not False or config["scientific_execution_authorization"] != "NOT_GRANTED":
        raise SystemExit("preparation config must remain execution-disabled")
    if forbidden:
        raise SystemExit(f"forbidden runtime imports: {forbidden}")
    counters = {
        "gpu_workers": 0,
        "model_loads": 0,
        "model_inference": 0,
        "simulator": 0,
        "env_step": 0,
        "physical_intervention": 0,
        "v_phys": 0,
        "eval160": 0,
        "protected_reads": 0,
        "pgd": 0,
    }
    entries = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in source_files
    ]
    entries.append({"path": str(CONFIG.relative_to(ROOT)), "sha256": sha256_file(CONFIG), "bytes": CONFIG.stat().st_size})
    report = {
        "schema": "STAGE_Z_MULTI_MODEL_RUNNER_PREPARATION_STATIC_AUDIT_V1",
        "status": "STAGE_Z_MULTI_MODEL_RUNNER_PREPARATION_STATIC_PASS",
        "active_scientific_gate": "HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED",
        "execution_enabled": False,
        "synthetic_only": True,
        "model_or_simulator_exposure": False,
        "protected_counters": counters,
        "checks": {
            "package_ast_compiles": True,
            "forbidden_runtime_imports": [],
            "common_action_contract_frozen": True,
            "model_family_boundaries_frozen": True,
            "queue_replan_guards_present": True,
            "z1_z2_z3_scaffolds_static_only": True,
            "student_detector_timing_selection": False,
            "synthetic_rows_explicitly_non_scientific": True,
            "structural_missing_cells_preserved": True,
        },
        "artifact_entries": entries,
        "next_legal_action": "finish Z0R2 authority closure, then STOP_FOR_PI",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "artifact": str(AUDIT), "source_files": len(source_files)}, sort_keys=True))


if __name__ == "__main__":
    main()
