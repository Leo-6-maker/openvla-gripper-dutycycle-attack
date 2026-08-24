#!/usr/bin/env python3
"""Run only the frozen, never-started D1R SCREENING_CLEAN identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"
BASE_PATH = REPO / "scripts/stage_x/run_stage_x1r_t1d1_screening_clean.py"
EXCLUDED = {1, 11, 20, 30}


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("stage_x_t1d1_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("D1_BASE_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D1 = load_base()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_receipt() -> dict[str, Any]:
    return D1.source_receipt()


def bind_static_contract(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hold_path = REPO / str(protocol["historical_d1"]["hold_report"])
    if sha256_file(hold_path) != protocol["historical_d1"]["hold_report_sha256"]:
        raise RuntimeError("D1_HOLD_REPORT_SHA_MISMATCH")
    hold = load_json(hold_path)
    if hold.get("status") != "HOLD_RUNTIME_INVALID_AFTER_FIRST_POLICY_DECISION":
        raise RuntimeError("D1_HOLD_REPORT_STATUS_INVALID")
    records = hold.get("canaries", [])
    if sorted(int(row["ordinal"]) for row in records) != sorted(EXCLUDED):
        raise RuntimeError("D1_CANARY_BINDING_INVALID")
    if any(row.get("retry_eligible") or row.get("first_policy_decision") is not True for row in records):
        raise RuntimeError("D1_CANARY_RETRY_DISPOSITION_INVALID")

    all_rows = D1.load_parent_rows(protocol)
    continuation = [row for row in all_rows if int(row["ordinal"]) not in EXCLUDED]
    if len(continuation) != 35:
        raise RuntimeError(f"D1R_CONTINUATION_COUNT_INVALID:{len(continuation)}")
    ledger_path = REPO / str(protocol["parent_population"]["continuation_ledger"])
    ledger = load_json(ledger_path)
    ledger_rows = ledger.get("rows", [])
    if [int(row["ordinal"]) for row in ledger_rows] != [int(row["ordinal"]) for row in continuation]:
        raise RuntimeError("D1R_CONTINUATION_ORDER_MISMATCH")
    for expected, actual in zip(continuation, ledger_rows):
        for key in ("canonical_parent_key", "expected_clean_seed", "ledger_clean_seed"):
            if expected.get(key) != actual.get(key):
                raise RuntimeError(f"D1R_CONTINUATION_LEDGER_MISMATCH:{key}")
    if int(ledger.get("repair_canary_ordinal")) != 2 or bool(ledger.get("replacement")) or bool(ledger.get("rerank")):
        raise RuntimeError("D1R_CONTINUATION_SELECTION_POLICY_INVALID")
    return all_rows, continuation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--phase", choices=("repair_canary", "continuation"), default="repair_canary")
    parser.add_argument("--ordinal", action="append", type=int)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_json(args.protocol.resolve())
    if protocol.get("schema") != "STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1" or protocol.get("status") != "FROZEN_FOR_D1R_SCREENING_CLEAN_EXECUTION":
        raise SystemExit("D1R_PROTOCOL_NOT_FROZEN")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit("WORKTREE_NOT_CLEAN")
    if source["branch"] != protocol["implementation"]["branch"]:
        raise SystemExit(f"BRANCH_BINDING_MISMATCH:{source['branch']}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().split(",")[0]
    if not visible.isdigit() or int(visible) != int(args.physical_gpu):
        raise SystemExit("CUDA_VISIBLE_DEVICES_MUST_BIND_SINGLE_REQUESTED_PHYSICAL_GPU")
    all_rows, continuation = bind_static_contract(protocol)

    contract_path = REPO / str(protocol["student"]["head_contract"])
    if sha256_file(contract_path) != protocol["student"]["head_contract_sha256"]:
        raise SystemExit("D1R_HEAD_CONTRACT_SHA_MISMATCH")
    audit_path = REPO / "reports/STAGE_X_X1R_T1D1R_HEAD_CONTRACT_AUDIT_V1.json"
    if not audit_path.is_file() or load_json(audit_path).get("status") != "STAGE_X_X1R_T1D1R_HEAD_CONTRACT_PASS":
        raise SystemExit("D1R_HEAD_CONTRACT_AUDIT_NOT_PASS")

    preflight_path = Path(str(protocol["durable_storage"]["root"])) / "preflight" / "D1_DURABLE_STORAGE_PREFLIGHT.json"
    if args.preflight_only:
        preflight = D1.durable_preflight(protocol, verify_models=True)
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 0
    if preflight_path.is_file():
        preflight = load_json(preflight_path)
        if preflight.get("status") != "PASS_DURABLE_STORAGE" or preflight.get("model_contract", {}).get("status") != "PASS":
            raise SystemExit("D1R_DURABLE_PREFLIGHT_NOT_PASS")
        if preflight.get("source", {}).get("commit") != source["commit"] or preflight.get("source", {}).get("tree") != source["tree"]:
            preflight = D1.durable_preflight(protocol, verify_models=True)
    else:
        preflight = D1.durable_preflight(protocol, verify_models=True)

    ordinals = sorted(set(args.ordinal or []))
    if args.attempt not in (0, 1):
        raise SystemExit("D1R_ATTEMPT_INVALID")
    by_ordinal = {int(row["ordinal"]): row for row in continuation}
    if args.phase == "repair_canary":
        if ordinals != [2]:
            raise SystemExit("D1R_REPAIR_CANARY_MUST_BE_ORDINAL_2")
    elif not ordinals or any(ordinal not in by_ordinal or ordinal == 2 for ordinal in ordinals):
        raise SystemExit("D1R_CONTINUATION_ORDINAL_SET_INVALID")
    selected = [by_ordinal[ordinal] for ordinal in ordinals]
    suites = {str(row["canonical_parent_key"]).split("/", 1)[0] for row in selected}
    if len(suites) != 1:
        raise SystemExit("ONE_SUITE_PER_WORKER_REQUIRED")

    gpu = D1.gpu_receipt(int(args.physical_gpu))
    suite = next(iter(suites))
    contract = D1.load_suite_contract(protocol)
    paths = D1.student_paths(protocol)
    import torch

    torch.set_num_threads(1)
    cfg = contract["suites"][suite]
    model, processor, device, action_dim = D1.load_openvla(Path(str(cfg["model_path"])), str(cfg["unnorm_key"]))
    student = D1.load_student(protocol, paths)
    root = Path(str(protocol["durable_storage"]["root"]))
    results = [D1.run_parent(parent, protocol, contract, model, processor, device, action_dim, student, int(args.physical_gpu), gpu, root, int(args.attempt)) for parent in selected]
    summary = {
        "schema": "STAGE_X_X1R_T1D1R_SCREENING_CLEAN_WORKER_RECEIPT_V1",
        "status": "PASS",
        "phase": args.phase,
        "source": source,
        "preflight": preflight,
        "gpu_before_model_load": gpu,
        "suite": suite,
        "ordinals": ordinals,
        "parent_receipts": results,
        "original_d1_excluded_ordinals": sorted(EXCLUDED),
        "forbidden_counters": {name: 0 for name in ("pgd_calls", "attack_backward_calls", "adversarial_images", "physical_interventions", "vphys_reads", "attack_outcome_reads", "eval160_reads", "protected_reads", "attacked_env_steps")},
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD"
    }
    D1.write_json(root / "workers" / f"worker_{args.phase}_{suite}_{os.getpid()}.json", summary)
    print(json.dumps({"status": summary["status"], "phase": args.phase, "suite": suite, "ordinals": ordinals, "root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
