"""Run the sealed, permanently excluded X1R2 engineering fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.stage_x import run_stage_x1r_primary_matrix as primary

Q3_PROTOCOL = ROOT / "configs/STAGE_X_X1R2_Q3_ENGINEERING_PROTOCOL_V1.json"
FIXTURE_REPORT = ROOT / "reports/STAGE_X_X1R2_Q3_ENGINEERING_FIXTURES_V1.json"
VICTIM_CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
D1_PROTOCOL = ROOT / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_receipt() -> dict[str, Any]:
    return primary.source_receipt()


def durable_preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(root)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    probe = root / ".q3_write_probe"
    probe.write_bytes(b"STAGE_X_X1R2_Q3_ENGINEERING_PROBE_V1\n")
    probe_sha = sha256_file(probe)
    probe.unlink()
    if free_bytes <= 4 * 1024**3:
        raise RuntimeError(f"HOLD_DURABLE_STORAGE:{free_bytes}")
    return {"root": str(root), "free_bytes": free_bytes, "write_probe_sha256": probe_sha}


def random_time_start(parent: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[int, str]:
    horizon = int(parent["policy_horizon"])
    emit = int(parent["first_emit_step"])
    legal = [
        start
        for start in range(0, horizon - 5 - 10 + 1)
        if start + 4 < emit or start > emit + 4
    ]
    if not legal:
        raise RuntimeError("NO_LEGAL_RANDOM_TIME_ENGINEERING_START")
    salt = str(protocol["timing"]["random_time_salt"])
    rank = hashlib.sha256(f"{salt}|{parent['canonical_parent_key']}".encode()).hexdigest()
    return legal[int(rank[:8], 16) % len(legal)], rank


def attacked_row_audit(path: Path, core_condition: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in (path / "step_telemetry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    attack_rows = [row for row in rows if bool(row.get("attack", {}).get("attack_executed"))]
    if len(attack_rows) != 5:
        raise RuntimeError(f"ENGINEERING_ATTACK_BURST_NOT_T5:{core_condition}:{len(attack_rows)}")
    for row in attack_rows:
        if core_condition in {"TRUE_PGD_T5", "SHUFFLED_GRAD_T5"}:
            route = row.get("attack", {}).get("route", {})
            if route.get("strict_route") is not True or route.get("allow_fallback") is not False or route.get("fallback_used") is not False:
                raise RuntimeError(f"ENGINEERING_STRICT_ROUTE_INVALID:{core_condition}:{row.get('step')}")
            if route.get("target_execution_class") != "NATIVE_OPEN" or int(route.get("target_token_id", -1)) != 31745:
                raise RuntimeError(f"ENGINEERING_TARGET_AUTHORITY_INVALID:{core_condition}:{row.get('step')}")
            if float(route.get("pixel_budget_adv_inputs_linf", 1.0)) > 0.03 + 1e-6:
                raise RuntimeError(f"ENGINEERING_PROCESSOR_LINF_INVALID:{core_condition}:{row.get('step')}")
        if row.get("arm_token_ids_equal") is not True:
            raise RuntimeError(f"ENGINEERING_ARM_ISOLATION_INVALID:{core_condition}:{row.get('step')}")
    return {
        "rows": len(rows),
        "attack_rows": len(attack_rows),
        "first_attack_step": int(attack_rows[0]["step"]),
        "last_attack_step": int(attack_rows[-1]["step"]),
        "telemetry_sha256": sha256_file(path / "step_telemetry.jsonl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    protocol = read_json(Q3_PROTOCOL)
    fixtures = read_json(FIXTURE_REPORT)
    if protocol.get("status") != "FROZEN_ENGINEERING_ONLY_PRE_GPU":
        raise SystemExit("Q3_PROTOCOL_NOT_FROZEN")
    if fixtures.get("status") != "STAGE_X_X1R2_Q3_ENGINEERING_FIXTURES_FROZEN":
        raise SystemExit("Q3_FIXTURE_REPORT_NOT_FROZEN")
    if sha256_file(FIXTURE_REPORT) != protocol["fixture_report"]["sha256"]:
        raise SystemExit("Q3_FIXTURE_REPORT_SHA_MISMATCH")
    fixture_rows = [row for row in fixtures.get("fixtures", []) if row.get("fixture_id") == args.fixture_id]
    if len(fixture_rows) != 1:
        raise SystemExit("Q3_FIXTURE_ID_INVALID")
    fixture = fixture_rows[0]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_REQUESTED_PHYSICAL_GPU")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    durable = durable_preflight(args.output_root)
    contract = read_json(VICTIM_CONTRACT)
    if sha256_file(VICTIM_CONTRACT) != protocol["victim_contract"]["sha256"]:
        raise SystemExit("VICTIM_CONTRACT_SHA_MISMATCH")
    suite = str(fixture["suite"])
    primary.verify_model_identity(contract, suite)

    from scripts.stage_x.run_stage_x1r_t1d1r_screening_clean import D1

    d1_protocol = read_json(D1_PROTOCOL)
    d1_contract = D1.load_suite_contract(d1_protocol)
    student_paths = D1.student_paths(d1_protocol)
    student = D1.load_student(d1_protocol, student_paths)
    suite_cfg = contract["suites"][suite]
    gpu = primary.gpu_receipt(args.physical_gpu, require_free=True)
    model, processor, device, action_dim = D1.load_openvla(Path(str(suite_cfg["model_path"])), suite)
    parent = dict(fixture)
    parent["legal_horizon"] = True
    clean_root = args.output_root / "fixtures" / args.fixture_id / "CLEAN_ENGINEERING"
    clean_root.mkdir(parents=True, exist_ok=False)
    clean_receipt = D1.run_parent(parent, d1_protocol, d1_contract, model, processor, device, action_dim, student, args.physical_gpu, gpu, clean_root, 0)
    if clean_receipt.get("status") != "PASS_SCREENING_CLEAN_EPISODE":
        raise RuntimeError("Q3_CLEAN_ENGINEERING_NOT_PASS")
    if clean_receipt.get("first_emit_step") != fixture.get("first_emit_step"):
        raise RuntimeError(f"Q3_STUDENT_EMIT_REPLAY_MISMATCH:{clean_receipt.get('first_emit_step')}!={fixture.get('first_emit_step')}")

    arms = [
        ("TRUE_PGD_T5_ENGINEERING", "TRUE_PGD_T5", False),
        ("RAND_UNIFORM_T5_ENGINEERING", "RAND_UNIFORM_T5", False),
        ("SHUFFLED_GRAD_T5_ENGINEERING", "SHUFFLED_GRAD_T5", False),
        ("TRUE_RANDOM_TIME_T5_ENGINEERING", "TRUE_PGD_T5", True),
    ]
    arm_receipts: list[dict[str, Any]] = []
    for engineering_label, core_condition, random_time in arms:
        run_parent = copy.deepcopy(parent)
        random_rank = None
        attack_start = int(parent["first_emit_step"])
        if random_time:
            attack_start, random_rank = random_time_start(parent, protocol)
            run_parent["first_emit_step"] = attack_start
        output = args.output_root / "fixtures" / args.fixture_id / engineering_label
        output.mkdir(parents=True, exist_ok=False)
        branch = primary.run_condition(run_parent, core_condition, model, processor, device, contract, {
            "seed_contract": {
                "eval_seed_namespace": "STAGE_X_X1R2_Q3_ENGINEERING_EVAL_V1_20260819",
                "perturb_seed_namespace": "STAGE_X_X1R2_Q3_ENGINEERING_PERTURB_V1_20260819",
                "arm_order_namespace": "STAGE_X_X1R2_Q3_ENGINEERING_ARM_ORDER_V1_20260819",
                "arm_order_base": ["CLEAN_EVAL", "RAND_UNIFORM_T5", "SHUFFLED_GRAD_T5", "TRUE_PGD_T5"]
            }
        }, args.physical_gpu, output, len(arm_receipts))
        if not branch.get("structural_valid"):
            raise RuntimeError(f"Q3_BRANCH_NOT_STRUCTURAL:{engineering_label}")
        telemetry = attacked_row_audit(output, core_condition)
        primary.write_json(output / "engineering_receipt.json", {
            "schema": "STAGE_X_X1R2_Q3_ENGINEERING_BRANCH_RECEIPT_V1",
            "status": "PASS_Q3_ENGINEERING_BRANCH",
            "engineering_label": engineering_label,
            "core_condition": core_condition,
            "fixture_id": args.fixture_id,
            "canonical_parent_key": fixture["canonical_parent_key"],
            "frozen_student_emit_step": int(parent["first_emit_step"]),
            "attack_start_step": attack_start,
            "random_time_rank": random_rank,
            "branch_receipt": branch,
            "telemetry_audit": telemetry,
            "scientific_use": false,
            "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        })
        arm_receipts.append({"engineering_label": engineering_label, "status": "PASS_Q3_ENGINEERING_BRANCH", "attack_start_step": attack_start, "telemetry": telemetry})

    fixture_root = args.output_root / "fixtures" / args.fixture_id
    result = {
        "schema": "STAGE_X_X1R2_Q3_ENGINEERING_FIXTURE_RECEIPT_V1",
        "status": "PASS_Q3_ENGINEERING_FIXTURE",
        "fixture": fixture,
        "source": source,
        "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
        "gpu": gpu,
        "durable_storage": durable,
        "student_clean_receipt": str(clean_root / "parents" / f"{int(parent['ordinal']):03d}_{primary.safe_name(str(parent['canonical_parent_key']))}" / "attempt_0" / "parent_receipt.json"),
        "arms": arm_receipts,
        "model_inference_calls": "engineering-only; not a scientific result",
        "scientific_use": false,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "vphys_reads": 0, "physical_interventions": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "timestamp_unix": time.time(),
    }
    primary.write_json(fixture_root / "fixture_receipt.json", result)
    print(json.dumps({"status": result["status"], "fixture_id": args.fixture_id, "arms": len(arm_receipts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
