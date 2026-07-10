#!/usr/bin/env python3
"""Finalize a v2 Goal model manifest after static and load-only validation.

The command performs one local OpenVLA model/processor load but launches no LIBERO
environment and no rollout. Changed bytes relative to a prior manifest require an
explicit rebase token so provenance discontinuity cannot be hidden accidentally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.gripper_attack.c2g_clean_window_runtime import derive_gripper_token_semantics
from tools.multisuite_detector.audit_c2g_goal_model_integrity_v2 import (
    PASS_STATUS as STATIC_PASS_STATUS,
    sha256_file,
)


SCHEMA_VERSION = "c2g.goal_model_integrity.2026-07-10.v2"
FINAL_PASS_STATUS = "PASS_C2G_GOAL_MODEL_INTEGRITY_AUDITED_V2"
REBASE_APPROVAL_TOKEN = "C2G_GOAL_MODEL_REBASE_20260710"


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(payload, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def finalize(
    static_report_path: Path,
    model_path: Path,
    *,
    device: str,
    rebase_approval: str,
) -> dict[str, Any]:
    static_report_path = static_report_path.resolve()
    model_path = model_path.resolve()
    static = _load_json(static_report_path)
    if static.get("status") != STATIC_PASS_STATUS:
        raise ValueError("Goal static integrity report is not PASS")
    if Path(str(static.get("model_path", ""))).resolve() != model_path:
        raise ValueError("Goal static report model path differs from requested model path")
    comparison = static.get("previous_manifest_comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("Goal static report lacks previous-manifest comparison")
    mismatches = comparison.get("mismatches")
    if not isinstance(mismatches, list):
        raise ValueError("Goal static report has malformed mismatch ledger")
    matches_previous = bool(comparison.get("matches_previous_bytes"))
    if mismatches and rebase_approval != REBASE_APPROVAL_TOKEN:
        raise ValueError(
            "Goal bytes differ from the prior manifest; pass the exact reviewed rebase "
            f"token {REBASE_APPROVAL_TOKEN!r} only after accepting a new C2g Goal baseline"
        )

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelClass

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelClass.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device if use_cuda else None,
    ).eval()

    norm_stats = getattr(model, "norm_stats", {})
    if not isinstance(norm_stats, Mapping):
        norm_stats = {}
    if "libero_goal" in norm_stats:
        unnorm_key = "libero_goal"
    elif "libero_goal_no_noops" in norm_stats:
        unnorm_key = "libero_goal_no_noops"
    else:
        raise ValueError("loaded Goal model exposes no libero_goal normalization statistics")
    token_semantics = derive_gripper_token_semantics(model, unnorm_key)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count <= 0:
        raise ValueError("loaded Goal model has no parameters")

    provenance_mode = (
        "RESTORED_FROZEN_BYTES" if matches_previous and not mismatches
        else "EXPLICIT_REBASE_CURRENT_BYTES"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": FINAL_PASS_STATUS,
        "model_path": str(model_path),
        "unnorm_key": unnorm_key,
        "file_count": int(static.get("file_count", 0)),
        "files": static.get("files", []),
        "files_aggregate_sha256": static.get("files_aggregate_sha256", ""),
        "referenced_shards": static.get("referenced_shards", []),
        "missing_referenced_shards": static.get("missing_referenced_shards", []),
        "static_integrity_report": str(static_report_path),
        "static_integrity_report_sha256": sha256_file(static_report_path),
        "previous_manifest_comparison": comparison,
        "provenance_mode": provenance_mode,
        "rebase_approval_token_sha256": (
            hashlib.sha256(rebase_approval.encode("utf-8")).hexdigest()
            if provenance_mode == "EXPLICIT_REBASE_CURRENT_BYTES"
            else ""
        ),
        "load_only_validation": {
            "status": "PASS_C2G_GOAL_MODEL_LOAD_ONLY",
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "parameter_count": parameter_count,
            "dtype": str(dtype),
            "device": str(device if use_cuda else "cpu"),
            "norm_stats_keys": sorted(str(key) for key in norm_stats),
            "token_semantics_sha256": token_semantics["token_semantics_sha256"],
            "open_token_count": len(token_semantics["open_token_ids"]),
            "close_token_count": len(token_semantics["close_token_ids"]),
        },
        "boundaries": {
            "openvla_model_loads": 1,
            "libero_environments_created": 0,
            "libero_rollouts_launched": 0,
            "attacks_launched": 0,
            "attack_outcomes_read": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-report", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebase-approval", default="")
    args = parser.parse_args(argv)
    result = finalize(
        args.static_report,
        args.model_path,
        device=args.device,
        rebase_approval=args.rebase_approval,
    )
    _write_json(args.output_manifest.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "model_path": result["model_path"],
        "provenance_mode": result["provenance_mode"],
        "parameter_count": result["load_only_validation"]["parameter_count"],
        "token_semantics_sha256": result["load_only_validation"]["token_semantics_sha256"],
        "boundaries": result["boundaries"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
