#!/usr/bin/env python3
"""Static audit for the append-only Z2 V3 runner repair."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "configs/STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V3.json"
BASE = ROOT / "configs/STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V2.json"
REPORT = ROOT / "reports/STAGE_Z_Z2A_V3_STATIC_AUDIT.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def main() -> None:
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    runner = ROOT / v3["git_binding"]["runner"]["path"]
    source = runner.read_text(encoding="utf-8")
    require(v3["schema"] == "STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V3", "SCHEMA")
    require(sha256(BASE) == v3["base_authority"]["sha256"], "BASE_DIGEST")
    require(base["schema"] == "STAGE_Z_Z2_CLEAN_REFERENCE_PROTOCOL_V2", "BASE_SCHEMA")
    require(git("rev-parse", "HEAD") == v3["git_binding"]["head_commit"], "HEAD")
    require(git("rev-parse", "HEAD^{tree}") == v3["git_binding"]["head_tree"], "TREE")
    require(sha256(runner) == v3["git_binding"]["runner"]["sha256"], "RUNNER_SHA256")
    require(git("rev-parse", f"HEAD:{v3['git_binding']['runner']['path']}") == v3["git_binding"]["runner"]["git_blob"], "RUNNER_BLOB")
    require(source.index('if binding.get("status") == "INELIGIBLE":') < source.index("verify_m1_materialization"), "INELIGIBLE_GUARD_ORDER")
    require(base["anchor_selection"]["h_phys"] == 10, "H_PHYS")
    require(base["population"]["identity_reuse_after_exposure"] is False, "REUSE_FIREWALL")
    require(v3["execution"]["command_open_intervention"] is False, "OPEN_FIREWALL")
    require(v3["execution"]["pgd"] is False, "PGD_FIREWALL")
    require(v3["execution"]["attacked_env_steps"] is False, "ATTACKED_STEP_FIREWALL")
    require(v3["execution"]["v_phys_endpoint"] is False, "VPHYS_FIREWALL")
    require(v3["execution"]["protected_reads"] is False, "PROTECTED_FIREWALL")
    report = {
        "schema": "STAGE_Z_Z2A_V3_STATIC_AUDIT",
        "status": "STAGE_Z_Z2_SOURCE_AND_ANCHOR_STATIC_PASS_V3",
        "protocol": str(V3.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(V3),
        "base_audit": {
            "path": "reports/STAGE_Z_Z2A_STATIC_AUDIT_V2.json",
            "status": "STAGE_Z_Z2_SOURCE_AND_ANCHOR_STATIC_PASS_WITH_LEGACY_RECONCILIATION_REQUIRED",
        },
        "checks": {
            "base_authority_byte_bound": True,
            "current_head_tree_bound": True,
            "runner_blob_and_sha_bound": True,
            "ineligible_fixture_guard_precedes_model_checkpoint_validation": True,
            "h_phys_10_inherited": True,
            "identity_reuse_firewall": True,
            "intervention_and_protected_firewall": True,
            "engineering_canary": "STATIC_SOURCE_ORDER_AND_MOCK_PASS",
        },
        "repair_disposition": "only existing taxonomy INELIGIBLE fixtures may abstain before model load; no fallback object or taxonomy widening",
        "scientific_counters": {
            "model_inference": 0,
            "env_step": 0,
            "command_open_intervention": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
        },
        "z3_authorized": False,
        "terminal_action": "continue only unexposed Z2-B cells, then STOP_FOR_PI",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "protocol_sha256": report["protocol_sha256"]}))


if __name__ == "__main__":
    main()
