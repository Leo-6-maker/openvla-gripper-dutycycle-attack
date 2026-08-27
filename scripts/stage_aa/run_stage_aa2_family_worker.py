#!/usr/bin/env python3
"""Run one model-family shard of the frozen AA2 cell manifest.

This is orchestration only: one child runner at a time, one GPU, no automatic
retry.  A nonzero child exit stops the shard for manual fail-closed review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--start-ordinal", type=int, default=1)
    parser.add_argument("--end-ordinal", type=int, default=108)
    parser.add_argument("--root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_worktrees/stage-aa1r1-runtime-14d14ea0"))
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    manifest = read_json(manifest_path)
    cells = [cell for cell in manifest["cells"] if cell["model_family"] == args.family]
    if len(cells) != 108 or not (1 <= args.start_ordinal <= args.end_ordinal <= len(cells)):
        raise RuntimeError(f"AA2_WORKER_SHARD_INVALID:{len(cells)}:{args.start_ordinal}:{args.end_ordinal}")
    cells = cells[args.start_ordinal - 1 : args.end_ordinal]
    runner = root / "scripts/stage_aa/run_stage_aa2_clean_screen.py"
    common = [
        PYTHON,
        "-u",
        str(runner),
        "--protocol", str(root / "configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"),
        "--source-authority", str(root / "reports/STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1.json"),
        "--launch-manifest", str(manifest_path),
        "--aa0", str(root / "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"),
        "--capacity", str(root / "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"),
        "--z1-config", str(root / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"),
        "--m1-manifest", str(root / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"),
    ]
    receipt_dir = root / "reports/server_evidence/STAGE_AA_AA2/receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    print(json.dumps({"status": "AA2_WORKER_STARTED", "family": args.family, "gpu_id": args.gpu_id, "cell_count": len(cells), "start_ordinal": args.start_ordinal, "end_ordinal": args.end_ordinal}, sort_keys=True), flush=True)
    for number, cell in enumerate(cells, start=1):
        output = receipt_dir / f"{cell['cell_id']}.json"
        if output.is_file():
            try:
                prior = read_json(output)
            except json.JSONDecodeError:
                prior = {}
            if prior.get("status") == "AA2_CLEAN_CELL_COMPLETE":
                print(json.dumps({"status": "AA2_CELL_ALREADY_COMPLETE", "cell_id": cell["cell_id"], "ordinal": number}, sort_keys=True), flush=True)
                continue
            if prior.get("status") == "AA2_ENGINEERING_HOLD_RUNTIME_ERROR":
                print(json.dumps({"status": "AA2_WORKER_HOLD_PRIOR_FAILURE", "cell_id": cell["cell_id"]}, sort_keys=True), flush=True)
                return 2
        command = common + ["--cell-id", cell["cell_id"], "--gpu-id", str(args.gpu_id), "--output", str(output)]
        print(json.dumps({"status": "AA2_CELL_START", "cell_id": cell["cell_id"], "ordinal": number, "family": args.family}, sort_keys=True), flush=True)
        completed = subprocess.run(command, env=env, cwd=root)
        if completed.returncode != 0:
            print(json.dumps({"status": "AA2_WORKER_STOPPED_ON_CELL_FAILURE", "cell_id": cell["cell_id"], "returncode": completed.returncode}, sort_keys=True), flush=True)
            return completed.returncode
        receipt = read_json(output)
        if receipt.get("status") != "AA2_CLEAN_CELL_COMPLETE":
            print(json.dumps({"status": "AA2_WORKER_STOPPED_ON_RECEIPT_STATUS", "cell_id": cell["cell_id"], "receipt_status": receipt.get("status")}, sort_keys=True), flush=True)
            return 3
        print(json.dumps({"status": "AA2_CELL_COMPLETE", "cell_id": cell["cell_id"], "ordinal": number}, sort_keys=True), flush=True)
    print(json.dumps({"status": "AA2_WORKER_COMPLETE", "family": args.family, "cell_count": len(cells), "next_legal_action": "STOP_FOR_PI_AFTER_FULL_CENSUS"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
