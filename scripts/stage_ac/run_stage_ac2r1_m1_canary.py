#!/usr/bin/env python3
"""Run one AC2R1 M1 clean-only canary after manifest reconciliation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_ac import m1_manifest_authority as M1_MANIFEST


M1 = "M1_OPENVLA_OFT"
PLAN = ROOT / "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_V1.json"
RECONCILIATION = ROOT / "reports/STAGE_AC_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_V1.json"
M1_MANIFEST_SOURCE = ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
AC0_CANARY_KEYS = {
    "libero_10/task_04/state_20",
    "libero_object/task_02/state_42",
    "libero_spatial/task_05/state_34",
}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA2R2: ModuleType | None = None


def helpers() -> ModuleType:
    global AA2R2
    if AA2R2 is None:
        AA2R2 = load_module(ROOT / "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py", "ac2r1_aa2r2_helpers")
    return AA2R2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_binding(binding: dict[str, Any], label: str) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]) or sha256_file(path) != str(binding["sha256"]):
        raise RuntimeError(f"AC2R1_SOURCE_BINDING_MISMATCH:{label}:{path}")
    return path


def validate_static(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, dict[str, Any]]:
    aa2 = helpers()
    protocol = load_json(args.protocol)
    source = load_json(args.source_authority)
    plan = load_json(args.canary_plan)
    z1 = load_json(args.z1_config)
    if protocol.get("status") != "STAGE_AC_AC2R1_PRE_GPU_REQUALIFICATION_AUTHORIZED":
        raise RuntimeError("AC2R1_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("clean_only") is not True or any(protocol.get(key) is not False for key in ("open_intervention_allowed", "attack_or_pgd_allowed", "physical_endpoint_read_allowed", "v_phys_read_allowed", "task_success_read_allowed", "protected_or_eval160_allowed")):
        raise RuntimeError("AC2R1_CLEAN_ONLY_FIREWALL_INVALID")
    if protocol.get("scientific_parent_exposure") != 0 or protocol.get("replacement_or_top_up") is not False:
        raise RuntimeError("AC2R1_SCIENTIFIC_FIREWALL_INVALID")
    if source.get("status") != "STAGE_AC_AC2R1_M1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("AC2R1_SOURCE_AUTHORITY_NOT_FROZEN")
    if plan.get("status") != "STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_FROZEN" or plan.get("cell_count") != 9:
        raise RuntimeError("AC2R1_CANARY_PLAN_NOT_FROZEN")
    for index, binding in enumerate(source.get("runtime_files", [])):
        verify_binding(binding, f"runtime_{index}")
    inputs = source.get("input_authorities", {})
    for label, binding in inputs.items():
        verify_binding(binding, label)
    if sha256_file(args.protocol) != str(inputs["protocol"]["sha256"]):
        raise RuntimeError("AC2R1_PROTOCOL_SOURCE_BINDING_INVALID")
    canary_rows = [row for row in plan.get("canaries", []) if row.get("model_family") == M1]
    expected_keys = set(AC0_CANARY_KEYS)
    if {row.get("canonical_parent_key") for row in canary_rows} != expected_keys or len(canary_rows) != 3:
        raise RuntimeError("AC2R1_M1_CANARY_SET_INVALID")
    canary = aa2.find_canary(plan, M1, args.canonical_parent_key)
    if canary.get("permanent_exclusion") is not True or canary.get("scientific_use") is not False:
        raise RuntimeError("AC2R1_CANARY_EXCLUSION_FIREWALL_INVALID")
    protocol_rows = {row["canonical_parent_key"]: row for row in protocol.get("canary_cells", [])}
    if protocol_rows.get(args.canonical_parent_key, {}).get("seed") != canary.get("seed"):
        raise RuntimeError("AC2R1_PROTOCOL_CANARY_BINDING_INVALID")
    reconciliation_path = Path(str(inputs["m1_reconciliation"]["path"]))
    if not reconciliation_path.is_absolute():
        reconciliation_path = ROOT / reconciliation_path
    M1_MANIFEST.validate_reconciliation(args.m1_manifest, reconciliation_path, args.z1_config)
    checkpoint = aa2.checkpoint_path(z1, M1, str(canary["suite"]))
    if not checkpoint.is_dir():
        raise RuntimeError(f"AC2R1_M1_CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
    runtime_manifest = M1_MANIFEST.materialize_historical_runtime_manifest(args.m1_manifest, RECONCILIATION, args.z1_config, args.output.resolve().parent.parent / "authority/STAGE_AC_AC2R1_M1_MANIFEST_HISTORICAL_CRLF_V1.json")
    manifest_result = aa2.Z1.verify_m1_materialization(Path(runtime_manifest["path"]), checkpoint, str(canary["suite"]), str(z1["model_families"][M1]["checkpoint_manifests_sha256"]))
    return protocol, source, canary, checkpoint, Path(runtime_manifest["path"]), manifest_result


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    aa2 = helpers()
    protocol = load_json(args.protocol)
    plan = load_json(args.canary_plan)
    canary = aa2.find_canary(plan, M1, args.canonical_parent_key)
    counters = {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "dummy_wait_env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "aa_v_phys_reads": 0,
        "v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "aa2_exposure": 0,
    }
    receipt: dict[str, Any] = {
        "schema": "STAGE_AC_AC2R1_M1_CANARY_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "model_family": M1,
        "canonical_parent_key": args.canonical_parent_key,
        "suite": canary["suite"],
        "task_idx": canary["task_idx"],
        "state_id": canary["state_id"],
        "seed": canary["seed"],
        "gpu_id": args.gpu_id,
        "permanent_exclusion": True,
        "scientific_use": False,
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"AC2R1_APPEND_ONLY_OUTPUT_EXISTS:{output}")
    atomic_write(output, receipt)
    context = {"receipt": receipt, "output": output, "counters": counters, "pair_audits": [], "family": M1, "current_step": 0, "failure_persisted": False}
    model = None
    try:
        protocol, source, canary, checkpoint, runtime_manifest, checkpoint_manifest = validate_static(args)
        aa2.AA1.require_single_gpu(args.gpu_id)
        receipt["gpu"] = aa2.gpu_snapshot(args.gpu_id)
        aa2.set_clean_seed(int(canary["seed"]))
        z1_config = load_json(args.z1_config)
        aa2.Z1.configure_libero(z1_config)
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_manifest"] = checkpoint_manifest
        receipt["runtime_manifest"] = {"path": str(runtime_manifest), "bytes": runtime_manifest.stat().st_size, "sha256": sha256_file(runtime_manifest)}
        receipt["runtime_environment"] = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET")}
        infer, model, normalization = aa2.Z1.load_openvla(str(checkpoint), oft=True, suite=str(canary["suite"]), return_chunk=True)
        clean = aa2.capture_engineering_clean(z1_config, M1, canary, infer, counters, context)
        if clean.get("status") != "PASS_AA2R2_ENGINEERING_CLEAN_TRAJECTORY":
            raise RuntimeError(clean.get("status", "AC2R1_CLEAN_TRAJECTORY_INVALID"))
        if len(clean["rows"]) != counters["env_step_calls"] or len(context["pair_audits"]) < len(clean["rows"]):
            raise RuntimeError("AC2R1_ACTION_TELEMETRY_AUDIT_INCOMPLETE")
        forbidden = ("open_intervention_steps", "pgd_calls", "attacked_env_steps", "aa_v_phys_reads", "v_phys_reads", "task_success_reads", "attack_outcome_reads", "eval160_reads", "protected_reads", "scientific_parent_exposure", "aa2_exposure")
        if any(counters[key] != 0 for key in forbidden):
            raise RuntimeError(f"AC2R1_SCIENTIFIC_COUNTER_NONZERO:{[(key, counters[key]) for key in forbidden if counters[key] != 0]}")
        receipt.update({
            "status": "STAGE_AC_AC2R1_M1_CANARY_PASS",
            "normalization": normalization,
            "clean_runtime": {key: value for key, value in clean.items() if key != "rows"},
            "clean_rows": clean["rows"],
            "action_pair_audit": context["pair_audits"],
            "action_pair_audit_sha256": aa2.canonical_hash(context["pair_audits"]),
            "runtime_counters": counters,
            "scientific_claim": "NONE_ENGINEERING_ONLY",
            "claim_boundary": "AC2R1 M1 manifest-byte repair and clean-only permanently-excluded canary qualification; no fresh AC2 parent or treatment exposure.",
            "next_legal_action": "STOP_FOR_PI_AFTER_AC2R1_THREE_CELL_PASS",
        })
        atomic_write(output, receipt)
        return receipt
    except Exception as exc:
        receipt.update({
            "status": "STAGE_AC_AC2R1_M1_CANARY_HOLD_RUNTIME_ERROR",
            "error": {"type": type(exc).__name__, "message": str(exc), "action_pair_audit_count": len(context["pair_audits"])},
            "action_pair_audit": context["pair_audits"],
            "action_pair_audit_sha256": aa2.canonical_hash(context["pair_audits"]),
            "runtime_counters": counters,
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
            "next_legal_action": "STOP_FOR_PI",
        })
        atomic_write(output, receipt)
        raise
    finally:
        if model is not None:
            del model
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, default=PLAN)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, default=M1_MANIFEST_SOURCE)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "model_family": M1, "canonical_parent_key": args.canonical_parent_key}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "STAGE_AC_AC2R1_M1_CANARY_HOLD_RUNTIME_ERROR", "model_family": M1, "canonical_parent_key": args.canonical_parent_key, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
