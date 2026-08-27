#!/usr/bin/env python3
"""Run one resumable, single-GPU shard of the AC2 clean-only census."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
DEFAULT_OUTPUT_ROOT = Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC2_CLEAN_SCREEN_V1")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def existing_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return str(value.get("status"))


def cell_args(args: argparse.Namespace, runner: ModuleType, cell: dict[str, Any], output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        protocol=args.root / "configs/STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_V1.json",
        source_authority=args.root / "reports/STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_V1.json",
        launch_manifest=args.root / "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json",
        z1_config=args.root / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json",
        m1_manifest=args.root / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json",
        cell_id=str(cell["cell_id"]),
        gpu_id=int(args.gpu_id),
        output=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--start-ordinal", type=int, default=1)
    parser.add_argument("--end-ordinal", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.start_ordinal < 1:
        raise RuntimeError("AC2_WORKER_START_ORDINAL_INVALID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["PYTHONUNBUFFERED"] = "1"
    args.root = args.root.resolve()
    args.output_root = args.output_root.resolve()

    runner = load_module(args.root / "scripts/stage_ac/run_stage_ac2_clean_screen.py", "ac2_clean_screen_runner")
    manifest = read_json(args.root / "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json")
    cells = [cell for cell in manifest.get("cells", []) if cell.get("model_family") == args.family]
    if args.suite is not None:
        cells = [cell for cell in cells if cell.get("suite") == args.suite]
    cells.sort(key=lambda cell: int(cell["cell_index"]))
    if not cells:
        raise RuntimeError("AC2_WORKER_NO_CELLS")
    if args.family in {"M0_OPENVLA", "M1_OPENVLA_OFT"} and args.suite is None:
        raise RuntimeError("AC2_WORKER_SUITE_REQUIRED_FOR_SUITE_CHECKPOINT")
    end = args.end_ordinal or len(cells)
    if not (args.start_ordinal <= end <= len(cells)):
        raise RuntimeError(f"AC2_WORKER_SHARD_INVALID:{len(cells)}:{args.start_ordinal}:{end}")
    selected = cells[args.start_ordinal - 1 : end]
    if not selected:
        raise RuntimeError("AC2_WORKER_EMPTY_SHARD")

    receipt_dir = args.output_root / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for cell in selected:
        path = receipt_dir / f"{cell['cell_id']}.json"
        status = existing_status(path)
        if status is None:
            pending.append((cell, path))
        elif status == "AC2_CLEAN_CELL_COMPLETE":
            emit({"status": "AC2_CELL_ALREADY_COMPLETE", "cell_id": cell["cell_id"]})
        else:
            raise RuntimeError(f"AC2_WORKER_PRIOR_NONFINAL_RECEIPT:{cell['cell_id']}:{status}")
    if not pending:
        emit({"status": "AC2_WORKER_COMPLETE_NO_PENDING_CELLS", "family": args.family, "gpu_id": args.gpu_id})
        return 0

    first_cell, first_output = pending[0]
    static_args = cell_args(args, runner, first_cell, first_output)
    protocol, source, _cell, _checkpoint, _checkpoint_manifest = runner.validate_static(static_args)
    config = runner.load_json(static_args.z1_config)
    gpu = runner.AA1.gpu_snapshot(int(args.gpu_id))
    emit(
        {
            "status": "AC2_WORKER_STARTED",
            "family": args.family,
            "suite": args.suite,
            "gpu_id": args.gpu_id,
            "start_ordinal": args.start_ordinal,
            "end_ordinal": end,
            "pending_cells": len(pending),
            "gpu_admission_snapshot": gpu,
        }
    )

    load_suite = str(first_cell["suite"])
    infer = model = None
    try:
        infer, model, normalization, checkpoint, checkpoint_manifest = runner.load_model(config, args.family, load_suite)
        for ordinal, (cell, output) in enumerate(pending, start=args.start_ordinal):
            emit({"status": "AC2_CELL_START", "cell_id": cell["cell_id"], "ordinal": ordinal, "family": args.family, "suite": cell["suite"], "gpu_id": args.gpu_id})
            result = runner.run_loaded_cell(
                cell_args(args, runner, cell, output),
                protocol,
                source,
                cell,
                config,
                infer,
                normalization,
                checkpoint,
                checkpoint_manifest,
                gpu,
            )
            if result.get("status") != "AC2_CLEAN_CELL_COMPLETE":
                raise RuntimeError(f"AC2_WORKER_UNEXPECTED_RECEIPT_STATUS:{cell['cell_id']}:{result.get('status')}")
            emit({"status": "AC2_CELL_COMPLETE", "cell_id": cell["cell_id"], "ordinal": ordinal, "eligibility_status": result.get("clean", {}).get("eligibility_status")})
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    emit({"status": "AC2_WORKER_COMPLETE", "family": args.family, "gpu_id": args.gpu_id, "completed_cells": len(pending), "next_legal_action": "BUILD_FULL_AC2_CENSUS_AFTER_ALL_WORKERS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
