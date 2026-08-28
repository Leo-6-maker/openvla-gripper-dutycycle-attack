#!/usr/bin/env python3
"""One final seed-bound B1R1 action-only replay for the frozen target branch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TARGET = "AC3-65bcfd948a45dd0be9ac"
MODEL = "M1_OPENVLA_OFT"
SUITE = "libero_spatial"
PARENT = "libero_spatial/task_06/state_16"
SEED = 348544072
ANCHOR_STEP = 56
ANCHOR_SHA = "081a516288d260435925746772063d1375ad827dbfd704012f0818368e332157"
QUEUE_LENGTH = 8
ACTION_DIM = 7
MIN_FREE_MIB = 20_480
GATE = "STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_REPLAY_V1"
FORBIDDEN_KEYS = ("physical_class", "v_phys", "endpoint", "outcome", "telemetry")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"B1R1_MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_sanitized(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            require(not any(token in lowered for token in FORBIDDEN_KEYS), f"B1R1_POSITIVE_FIELD_FORBIDDEN:{key}")
            assert_sanitized(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_sanitized(nested)


def safe_failure(path: Path) -> dict[str, Any]:
    data = read_json(path)
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": data.get("status"),
        "branch_id": data.get("branch_id"),
        "model_family": data.get("model_family"),
        "suite": data.get("suite"),
        "condition": data.get("condition"),
        "dose": data.get("dose"),
        "error_type": error.get("type"),
        "error_message": error.get("message"),
    }


def rng_digest(ac3: ModuleType) -> dict[str, Any]:
    numpy_state = ac3.np.random.get_state()
    numpy_payload = {
        "bit_generator": str(numpy_state[0]),
        "keys": numpy_state[1].tolist(),
        "pos": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    result: dict[str, Any] = {
        "python_random_sha256": canonical_hash(random.getstate()),
        "numpy_random_sha256": canonical_hash(numpy_payload),
        "torch_cpu_random_sha256": None,
        "torch_cuda_random_sha256": None,
        "torch_cuda_device_count": 0,
    }
    try:
        import torch

        result["torch_cpu_random_sha256"] = sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes())
        if torch.cuda.is_available():
            states = torch.cuda.get_rng_state_all()
            result["torch_cuda_random_sha256"] = sha256_bytes(b"".join(state.cpu().numpy().tobytes() for state in states))
            result["torch_cuda_device_count"] = len(states)
    except ImportError:
        pass
    return result


def initial_receipt(job: dict[str, Any], target_failure: dict[str, Any], source_script: Path, gpu_id: int, b0_ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_RECEIPT_V1",
        "status": "RUNNING",
        "gate": GATE,
        "branch_id": TARGET,
        "model_family": MODEL,
        "suite": SUITE,
        "canonical_parent_key": PARENT,
        "condition": "OPEN_T10",
        "dose": 10,
        "branch_seed": {"seed": SEED, "seed_digest": job["branch_seed"]["seed_digest"]},
        "anchor": {"step": ANCHOR_STEP, "state_sha256": ANCHOR_SHA, "selection_rank_sha256": job["selected_anchor"]["selection_rank_sha256"]},
        "source_clean": {
            "receipt_path": job["source_receipt"]["path"],
            "receipt_bytes": int(job["source_receipt"]["bytes"]),
            "receipt_sha256": job["source_receipt"]["sha256"],
            "trajectory_digest": job["selected_anchor"]["source_clean_trajectory_digest"],
        },
        "target_failure_receipt": target_failure,
        "b0_authority": b0_ref,
        "runtime_source": {"path": source_script.relative_to(ROOT).as_posix(), "bytes": source_script.stat().st_size, "sha256": sha256_file(source_script)},
        "gpu_id": gpu_id,
        "scope": "single M1 action-only inference replay from the sealed anchor; no env.step, command intervention, physical read, endpoint, or outcome access",
        "seed_binding": "SET_AFTER_MODEL_LOAD_BEFORE_STATE_RESTORE_AND_INFERENCE",
        "scientific_firewall": {"new_model_inference_calls": 0, "new_env_step_calls": 0, "new_open_intervention_steps": 0, "new_protected_reads": 0},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def validate_b0(path: Path) -> dict[str, Any]:
    data = read_json(path)
    require(data.get("status") == "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_PASS_CONTINUE", "B1R1_B0_NOT_PASS")
    summary = data.get("remote_action_only_audit", {}).get("receipt_summary", {})
    require(summary.get("branches") == 372 and summary.get("v2_rejected") == 0 and summary.get("old_pass_v2_fail") == 0, "B1R1_B0_SYSTEMIC_SEMANTICS_NOT_CLOSED")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "status": data["status"], "audited_branches": summary["branches"], "v2_rejected": summary["v2_rejected"]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"B1R1_APPEND_ONLY_OUTPUT_EXISTS:{args.output_dir}")
    b0_ref = validate_b0(args.b0_report)
    base = load_module(ROOT / "scripts/stage_ac/run_stage_ac3_g2r1_b1_target_inference_only.py", "ac3_g2r1_b1_base")
    ac3 = load_module(ROOT / "scripts/stage_ac/run_stage_ac3_g2_model_suite.py", "ac3_g2_b1r1_runtime")
    ac3.load_runtime()
    aa1 = ac3.AA1
    legacy = aa1.SEMANTICS
    v2 = load_module(ROOT / "src/stage_aa/action_semantics_v2.py", "ac3_g2_b1r1_v2_semantics")
    args.model_family = MODEL
    args.suite = SUITE
    _manifest, job, source = base.static_job(ac3, args)
    _source_receipt, clean, _binding = source
    target_failure = safe_failure(args.target_failure)
    require(target_failure["branch_id"] == TARGET and target_failure["status"] == "ENGINEERING_INVALID_OR_HORIZON_CENSORED", "B1R1_TARGET_FAILURE_BINDING")
    receipt_path = args.output_dir / f"{TARGET}_INFERENCE_ONLY.json"
    root_path = args.output_dir / "STAGE_AC_AC3_G2R1_B1R1_ROOT_SEAL_V1.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    aa1.require_single_gpu(args.gpu_id)
    gpu = aa1.gpu_snapshot(args.gpu_id)
    require(int(gpu["free_memory_mib"]) > MIN_FREE_MIB, f"B1R1_GPU_FREE:{gpu['free_memory_mib']}")
    require(not gpu.get("compute_processes"), f"B1R1_GPU_FOREIGN_PROCESS:{gpu.get('compute_processes')}")
    config = read_json(args.config)
    aa1.Z1.configure_libero(config)
    receipt = initial_receipt(job, target_failure, Path(__file__), args.gpu_id, b0_ref)
    receipt["gpu_snapshot_before_model"] = gpu
    receipt["runtime_environment"] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "checkpoint_materialization": "NOT_REQUESTED"}
    write_json(receipt_path, receipt)
    model = None
    inference_started = False
    try:
        infer, model, checkpoint, checkpoint_manifest = ac3.load_model(config, MODEL, SUITE)
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_manifest"] = checkpoint_manifest
        receipt["rng_before_seed"] = rng_digest(ac3)
        ac3.set_branch_seed(SEED)
        receipt["rng_after_seed"] = rng_digest(ac3)
        receipt["seed_bound_before_inference"] = True
        env, obs = base.restore_anchor(aa1, config, job, clean)
        try:
            inference_started = True
            rows, meta = base.audit_queue(ac3, legacy, v2, infer, obs, str(clean["language"]))
        finally:
            env.close()
        v1_rejected = [row for row in rows if not row["v1"]["accepted"]]
        v2_rejected = [row for row in rows if not row["v2"]["accepted"]]
        stale_reasons = {row["v1"]["reason"] for row in v1_rejected}
        stale_validator = bool(v1_rejected) and not v2_rejected and stale_reasons == {"RAW_GRIPPER_AT_THRESHOLD"}
        if stale_validator:
            status = "STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_REPLAY_PASS_CONTINUE_TO_RECOVERY"
            next_action = "B1_TARGET_EXACT_BRANCH_RECOVERY_ONCE"
        else:
            status = "STAGE_AC_AC3_G2R1_B1R1_UNKNOWN_ACTION_SEMANTICS_CONTINUE_TO_CENSORING_ANALYSIS"
            next_action = "G2R1_C_CENSORING_AWARE_ANALYSIS"
        receipt.update({
            "status": status,
            "inference": {"model_inference_calls": 1, "fresh_boundary": meta["fresh_boundary"], "chunk_length": meta["chunk_length"], "queue_length_audited": QUEUE_LENGTH, "queue": rows},
            "reconciliation": {"v1_accepted": QUEUE_LENGTH - len(v1_rejected), "v1_rejected": len(v1_rejected), "v2_accepted": QUEUE_LENGTH - len(v2_rejected), "v2_rejected": len(v2_rejected), "v1_rejection_reasons": sorted(stale_reasons), "stale_validator_failure_explained": stale_validator, "systemic_authority_drift": False, "official_m1_rule": "raw < 0.5 -> env +1; raw == 0.5 -> env 0; raw > 0.5 -> env -1"},
            "scientific_firewall": {"new_model_inference_calls": 1, "new_env_step_calls": 0, "new_open_intervention_steps": 0, "new_protected_reads": 0},
            "next_legal_action": next_action,
        })
    except Exception as exc:
        receipt.update({"status": "STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_HOLD_STOP_FOR_PI", "error": {"type": type(exc).__name__, "message": str(exc)}, "next_legal_action": "STOP_FOR_PI"})
        receipt["scientific_firewall"]["new_model_inference_calls"] = int(inference_started)
    finally:
        if model is not None:
            del model
        gc.collect()
    assert_sanitized(receipt)
    write_json(receipt_path, receipt)
    ref = {"path": str(receipt_path), "bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)}
    payload = {"gate": GATE, "status": receipt["status"], "receipt": ref, "branch_id": TARGET, "scientific_firewall": receipt["scientific_firewall"]}
    root = {"schema": "STAGE_AC_AC3_G2R1_B1R1_ROOT_SEAL_V1", "status": receipt["status"], "gate": GATE, "root_payload": payload, "root_payload_sha256": canonical_hash(payload), "next_legal_action": receipt.get("next_legal_action")}
    assert_sanitized(root)
    write_json(root_path, root)
    return {"status": receipt["status"], "receipt": str(receipt_path), "root": str(root_path), "next_legal_action": receipt.get("next_legal_action")}


def self_test() -> None:
    assert ACTION_DIM == 7 and QUEUE_LENGTH == 8 and MIN_FREE_MIB == 20_480
    assert TARGET.startswith("AC3-") and PARENT == "libero_spatial/task_06/state_16"
    print(json.dumps({"status": "B1R1_INFERENCE_ONLY_STATIC_SELF_TEST_PASS"}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json")
    parser.add_argument("--g0-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json")
    parser.add_argument("--g1-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_G1_ROOT_SEAL_V1.json")
    parser.add_argument("--runtime-authority", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_V2.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json")
    parser.add_argument("--blind-sample", type=Path, default=ROOT / "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json")
    parser.add_argument("--b0-report", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1.json")
    parser.add_argument("--target-failure", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2_PHYSICAL_V1/receipts/AC3-65bcfd948a45dd0be9ac.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2R1_B1R1_TARGET_INFERENCE_ONLY_V1"))
    parser.add_argument("--gpu-id", type=int, default=None)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.gpu_id is None:
        parser.error("--gpu-id is required unless --self-test is used")
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"].endswith("ANALYSIS") or result["status"].endswith("RECOVERY") else 1
    except Exception as exc:
        print(json.dumps({"status": "B1R1_INFERENCE_ONLY_UNHANDLED_HOLD", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
