#!/usr/bin/env python3
"""Run one AA2R2 Phase-B model-family shard without changing old receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
RECOVERY_STATUSES = {"RUNNING", "AA2_ENGINEERING_HOLD_RUNTIME_ERROR"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--start-ordinal", type=int, default=1)
    parser.add_argument("--end-ordinal", type=int, default=108)
    parser.add_argument("--root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_worktrees/stage-aa2r2-phase-b-59a48842"))
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    manifest = read_json(manifest_path)
    cells = [cell for cell in manifest["cells"] if cell["model_family"] == args.family]
    if len(cells) != 108 or not (1 <= args.start_ordinal <= args.end_ordinal <= len(cells)):
        raise RuntimeError(f"AA2R2_WORKER_SHARD_INVALID:{len(cells)}:{args.start_ordinal}:{args.end_ordinal}")
    cells = cells[args.start_ordinal - 1 : args.end_ordinal]
    runner = root / "scripts/stage_aa/run_stage_aa2r2_clean_screen.py"
    common = [
        PYTHON,
        "-u",
        str(runner),
        "--protocol", str(root / "configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"),
        "--source-authority", str(root / "reports/STAGE_AA_AA2R2_PHASE_B_RUNTIME_SOURCE_AUTHORITY_V1.json"),
        "--launch-manifest", str(manifest_path),
        "--aa0", str(root / "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"),
        "--capacity", str(root / "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"),
        "--z1-config", str(root / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
        "--m1-manifest", str(root / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"),
    ]
    old_dir = root / "reports/server_evidence/STAGE_AA_AA2/receipts"
    new_dir = root / "reports/server_evidence/STAGE_AA_AA2R2/phase_b/receipts"
    recovery_dir = root / "reports/server_evidence/STAGE_AA_AA2R2/phase_b/recovery"
    new_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"status": "AA2R2_WORKER_STARTED", "family": args.family, "gpu_id": args.gpu_id, "cell_count": len(cells), "start_ordinal": args.start_ordinal, "end_ordinal": args.end_ordinal}, sort_keys=True), flush=True)
    for number, cell in enumerate(cells, start=args.start_ordinal):
        cell_id = cell["cell_id"]
        old_path = old_dir / f"{cell_id}.json"
        old_status = read_json(old_path).get("status") if old_path.is_file() else None
        if old_status == "AA2_CLEAN_CELL_COMPLETE":
            print(json.dumps({"status": "AA2R2_CELL_PRESERVED_HISTORICAL_COMPLETE", "cell_id": cell_id, "ordinal": number}, sort_keys=True), flush=True)
            continue
        if old_status in RECOVERY_STATUSES:
            output = recovery_dir / f"{cell_id}.recovery.json"
            attempt_kind = "RECOVERY"
            recovery_of = cell_id
        elif old_status is None:
            output = new_dir / f"{cell_id}.json"
            attempt_kind = "NORMAL"
            recovery_of = None
        else:
            print(json.dumps({"status": "AA2R2_WORKER_HOLD_UNEXPECTED_HISTORICAL_STATUS", "cell_id": cell_id, "old_status": old_status}, sort_keys=True), flush=True)
            return 2
        if output.is_file():
            prior_status = read_json(output).get("status")
            if prior_status == "AA2R2_PHASE_B_CLEAN_CELL_COMPLETE":
                print(json.dumps({"status": "AA2R2_CELL_ALREADY_COMPLETE", "cell_id": cell_id, "attempt_kind": attempt_kind}, sort_keys=True), flush=True)
                continue
            print(json.dumps({"status": "AA2R2_WORKER_HOLD_PRIOR_ATTEMPT", "cell_id": cell_id, "attempt_kind": attempt_kind, "receipt_status": prior_status}, sort_keys=True), flush=True)
            return 2
        command = common + [
            "--cell-id", cell_id,
            "--gpu-id", str(args.gpu_id),
            "--output", str(output),
            "--attempt-kind", attempt_kind,
        ]
        if recovery_of is not None:
            command += ["--recovery-of", recovery_of]
        print(json.dumps({"status": "AA2R2_CELL_START", "cell_id": cell_id, "ordinal": number, "family": args.family, "attempt_kind": attempt_kind}, sort_keys=True), flush=True)
        completed = subprocess.run(command, env=env, cwd=root)
        if completed.returncode != 0:
            print(json.dumps({"status": "AA2R2_WORKER_STOPPED_ON_CELL_FAILURE", "cell_id": cell_id, "returncode": completed.returncode, "attempt_kind": attempt_kind}, sort_keys=True), flush=True)
            return completed.returncode or 1
        receipt = read_json(output)
        if receipt.get("status") != "AA2R2_PHASE_B_CLEAN_CELL_COMPLETE":
            print(json.dumps({"status": "AA2R2_WORKER_STOPPED_ON_RECEIPT_STATUS", "cell_id": cell_id, "receipt_status": receipt.get("status")}, sort_keys=True), flush=True)
            return 3
        print(json.dumps({"status": "AA2R2_CELL_COMPLETE", "cell_id": cell_id, "ordinal": number, "attempt_kind": attempt_kind}, sort_keys=True), flush=True)
    print(json.dumps({"status": "AA2R2_WORKER_COMPLETE", "family": args.family, "cell_count": len(cells), "next_legal_action": "BUILD_FULL_324_CELL_CENSUS"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
