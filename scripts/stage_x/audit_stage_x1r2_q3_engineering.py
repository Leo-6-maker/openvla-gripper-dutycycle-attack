"""CPU/static authority audit for the sealed X1R2 Q3 engineering fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_X_X1R2_Q3_ENGINEERING_PROTOCOL_V1.json"
FIXTURES = ROOT / "reports/STAGE_X_X1R2_Q3_ENGINEERING_FIXTURES_V1.json"
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r2_q3_engineering.py"
PRIMARY_RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"
OUTPUT = ROOT / "reports/STAGE_X_X1R2_Q3_ENGINEERING_STATIC_AUDIT_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    blob = subprocess.check_output(
        ["git", "-C", str(ROOT), "cat-file", "blob", f"HEAD:{relative}"]
    )
    return hashlib.sha256(blob).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def main() -> int:
    errors: list[str] = []
    protocol = load(PROTOCOL)
    fixtures = load(FIXTURES)
    runner_text = RUNNER.read_text(encoding="utf-8")
    primary_text = PRIMARY_RUNNER.read_text(encoding="utf-8")

    victim_contract = ROOT / str(protocol.get("victim_contract", {}).get("path", ""))
    if not victim_contract.is_file():
        errors.append("VICTIM_CONTRACT_MISSING")
    elif git_blob_sha256(victim_contract) != protocol.get("victim_contract", {}).get("sha256"):
        errors.append("VICTIM_CONTRACT_SHA256_MISMATCH")

    if protocol.get("status") != "FROZEN_ENGINEERING_ONLY_PRE_GPU":
        errors.append("Q3_PROTOCOL_STATUS")
    if fixtures.get("status") != "STAGE_X_X1R2_Q3_ENGINEERING_FIXTURES_FROZEN":
        errors.append("FIXTURE_REPORT_STATUS")
    if sha256(FIXTURES) != protocol.get("fixture_report", {}).get("sha256"):
        errors.append("FIXTURE_REPORT_SHA")
    rows = fixtures.get("fixtures", [])
    suites = {row.get("suite") for row in rows}
    if len(rows) != 4 or suites != {"libero_10", "libero_goal", "libero_object", "libero_spatial"}:
        errors.append("FOUR_SUITE_FIXTURE_SET")
    for row in rows:
        if row.get("manual_contact_label") not in {"FAIL", "ABSTAIN"}:
            errors.append(f"FIXTURE_LABEL:{row.get('fixture_id')}")
        if row.get("clean_success") is not True or row.get("student_status") != "PASS_CAUSAL_TRACE":
            errors.append(f"FIXTURE_CLEAN_STATUS:{row.get('fixture_id')}")
        if row.get("permanent_exclusion") is not True:
            errors.append(f"FIXTURE_PERMANENCE:{row.get('fixture_id')}")
        if row.get("review_id") in set(fixtures.get("exclusion_sets", {}).get("x1r_v1_cohort_review_ids", [])):
            errors.append(f"FIXTURE_INTERSECTS_V1:{row.get('review_id')}")

    static_checks = {
        "clean_student_path": "D1.run_parent" in runner_text and "student_paths" in runner_text and "D1.load_student" in runner_text,
        "attack_path_uses_primary_runner": "primary.run_condition" in runner_text,
        "all_engineering_arms": all(name in runner_text for name in ("TRUE_PGD_T5_ENGINEERING", "RAND_UNIFORM_T5_ENGINEERING", "SHUFFLED_GRAD_T5_ENGINEERING", "TRUE_RANDOM_TIME_T5_ENGINEERING")),
        "random_time_is_deterministic_and_legal": "random_time_start" in runner_text and "no overlap" not in runner_text,
        "full_episode_route": "official_horizon_reached" in primary_text and "final_policy_steps_executed" in primary_text,
        "canonical_attack_wrapper": "OpenVLAVisualAttacker(" in primary_text and "TokenPrefixPGDAttacker(" not in primary_text,
        "strict_route_and_target_audit": "strict_route" in runner_text and "NATIVE_OPEN" in runner_text and "31745" in runner_text,
        "arm_isolation_audit": "ENGINEERING_ARM_ISOLATION_INVALID" in runner_text,
        "no_protected_path": all(name not in runner_text for name in ("read_eval160(", "read_vphys(", "physical_intervention(")),
    }
    for name, value in static_checks.items():
        if not value:
            errors.append(f"STATIC:{name}")

    report = {
        "schema": "STAGE_X_X1R2_Q3_ENGINEERING_STATIC_AUDIT_V1",
        "status": "STAGE_X_X1R2_Q3_ENGINEERING_STATIC_AUTHORITY_PASS" if not errors else "STAGE_X_X1R2_Q3_HOLD_STATIC_AUTHORITY",
        "scope": "CPU/static only; no model load, inference, simulator reset/step, PGD, physical intervention, V_phys, Eval160, or protected read",
        "source": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")},
        "fixture_report_sha256": sha256(FIXTURES),
        "protocol_sha256": sha256(PROTOCOL),
        "victim_contract_git_blob_sha256": git_blob_sha256(victim_contract) if victim_contract.is_file() else None,
        "fixture_ids": [row.get("fixture_id") for row in rows],
        "suites": sorted(suites),
        "static_checks": static_checks,
        "errors": errors,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "next_gate": "STAGE_X_X1R2_Q3_REAL_EXECUTABLE_QUALIFICATION" if not errors else "OWNER_REVIEW_X1R2_Q3_STATIC_AUTHORITY_HOLD",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
