#!/usr/bin/env python3
"""Run the one authorized B1 M1 action-only inference replay."""

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
GATE = "STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_REPLAY_V1"
FORBIDDEN_KEYS = ("physical_class", "v_phys", "endpoint", "outcome", "telemetry")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"B1_MODULE_SPEC_UNAVAILABLE:{path}")
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
    """Keep this receipt action-only; positive/outcome fields are forbidden."""
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            require(not any(token in lowered for token in FORBIDDEN_KEYS), f"B1_POSITIVE_FIELD_FORBIDDEN:{key}")
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


def initial_receipt(job: dict[str, Any], target_failure: dict[str, Any], source_script: Path, gpu_id: int) -> dict[str, Any]:
    return {
        "schema": "STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_RECEIPT_V1",
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
        "runtime_source": {"path": source_script.relative_to(ROOT).as_posix(), "bytes": source_script.stat().st_size, "sha256": sha256_file(source_script)},
        "gpu_id": gpu_id,
        "scope": "single M1 action-only inference replay from the sealed anchor; no env.step, command intervention, physical read, endpoint, or outcome access",
        "scientific_firewall": {"new_model_inference_calls": 0, "new_env_step_calls": 0, "new_open_intervention_steps": 0, "new_protected_reads": 0},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def static_job(ac3: ModuleType, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    manifest, jobs, _blind_map = ac3.prepare_static(args)
    matches = [job for job in jobs if str(job["branch_id"]) == TARGET]
    require(len(matches) == 1, f"B1_TARGET_JOB_COUNT:{len(matches)}")
    job = matches[0]
    expected = {
        "model_family": MODEL,
        "suite": SUITE,
        "canonical_parent_key": PARENT,
        "condition": "OPEN_T10",
        "dose": 10,
        "source_task_idx": 6,
        "state_id": 16,
    }
    for key, value in expected.items():
        require(str(job.get(key)) == str(value), f"B1_TARGET_BINDING:{key}:{job.get(key)}:{value}")
    require(int(job["branch_seed"]["seed"]) == SEED, "B1_TARGET_SEED")
    anchor = job["selected_anchor"]
    require(int(anchor["step"]) == ANCHOR_STEP and str(anchor["boundary_state_sha256"]) == ANCHOR_SHA, "B1_TARGET_ANCHOR")
    require(len(anchor["actions"]) == 20, "B1_TARGET_CLEAN_WINDOW")
    unit = next(unit for unit in manifest["model_parent_units"] if unit["model_family"] == MODEL and unit["canonical_parent_key"] == PARENT)
    source_receipt, clean, binding = ac3.load_source_unit(unit)
    return manifest, job, (source_receipt, clean, binding)


def restore_anchor(aa1: ModuleType, config: dict[str, Any], job: dict[str, Any], clean: dict[str, Any]):
    env, _suite, _task = aa1.Z1.make_libero_env(config, SUITE, int(job["source_task_idx"]))
    env.reset()
    state = clean["boundary_states"][ANCHOR_STEP]
    aa1.Z1.restore_state(env, state)
    actual = aa1.Z1.snapshot_state(env)
    require(aa1.np.array_equal(actual, state), "B1_ANCHOR_RESTORE_NOT_EXACT")
    getter = getattr(getattr(env, "env", None), "_get_observations", None) or getattr(env, "_get_observations", None)
    require(callable(getter), "B1_OBSERVATION_GETTER_UNAVAILABLE")
    return env, getter()


def audit_queue(ac3: ModuleType, legacy: ModuleType, v2: ModuleType, infer: Any, obs: dict[str, Any], language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk, meta = infer(obs, language)
    raw = ac3.np.asarray(meta.get("raw_action_chunk"), dtype=ac3.np.float32)
    final = ac3.np.asarray(chunk, dtype=ac3.np.float32)
    require(meta.get("fresh_boundary") == "FRESH_OFT_ACTION_QUEUE", f"B1_MODEL_BOUNDARY:{meta.get('fresh_boundary')}")
    require(raw.ndim == 2 and final.ndim == 2 and raw.shape == final.shape and raw.shape[1] == ACTION_DIM and raw.shape[0] >= QUEUE_LENGTH, f"B1_QUEUE_SHAPE:{raw.shape}:{final.shape}")
    rows: list[dict[str, Any]] = []
    for index in range(QUEUE_LENGTH):
        raw_action = raw[index].astype(ac3.np.float32).tolist()
        final_action = final[index].astype(ac3.np.float32).tolist()
        old = legacy.validate_action_pair(MODEL, raw_action, final_action, raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
        new = v2.validate_action_pair(MODEL, raw_action, final_action, raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
        rows.append({
            "queue_index": index,
            "raw_action_7d": raw_action,
            "final_action_7d": final_action,
            "raw_gripper": float(raw_action[-1]),
            "final_gripper": float(final_action[-1]),
            "official_expected_final_gripper": float(v2.raw_gripper_to_env_gripper(float(raw_action[-1]), binarize=True)),
            "v1": {"accepted": bool(old.get("accepted")), "reason": old.get("reason"), "semantic_state": old.get("semantic_state")},
            "v2": {"accepted": bool(new.get("accepted")), "reason": new.get("reason"), "semantic_state": new.get("semantic_state"), "validator_version": new.get("validator_version")},
        })
    return rows, {"fresh_boundary": meta.get("fresh_boundary"), "chunk_length": int(meta.get("chunk_length", raw.shape[0]))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.output_dir.exists(), f"B1_APPEND_ONLY_OUTPUT_EXISTS:{args.output_dir}")
    ac3 = load_module(ROOT / "scripts/stage_ac/run_stage_ac3_g2_model_suite.py", "ac3_g2_b1_runtime")
    ac3.load_runtime()
    aa1 = ac3.AA1
    legacy = aa1.SEMANTICS
    v2 = load_module(ROOT / "src/stage_aa/action_semantics_v2.py", "ac3_g2_b1_v2_semantics")
    args.model_family = MODEL
    args.suite = SUITE
    _manifest, job, source = static_job(ac3, args)
    _source_receipt, clean, _binding = source
    target_failure = safe_failure(args.target_failure)
    require(target_failure["branch_id"] == TARGET and target_failure["status"] == "ENGINEERING_INVALID_OR_HORIZON_CENSORED", "B1_TARGET_FAILURE_BINDING")
    receipt_path = args.output_dir / f"{TARGET}_INFERENCE_ONLY.json"
    root_path = args.output_dir / "STAGE_AC_AC3_G2R1_B1_ROOT_SEAL_V1.json"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    aa1.require_single_gpu(args.gpu_id)
    gpu = aa1.gpu_snapshot(args.gpu_id)
    require(int(gpu["free_memory_mib"]) > MIN_FREE_MIB, f"B1_GPU_FREE:{gpu['free_memory_mib']}")
    require(not gpu.get("compute_processes"), f"B1_GPU_FOREIGN_PROCESS:{gpu.get('compute_processes')}")
    config = read_json(args.config)
    aa1.Z1.configure_libero(config)
    receipt = initial_receipt(job, target_failure, Path(__file__), args.gpu_id)
    receipt["gpu_snapshot_before_model"] = gpu
    receipt["runtime_environment"] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "checkpoint_materialization": "NOT_REQUESTED"}
    write_json(receipt_path, receipt)
    model = None
    try:
        infer, model, checkpoint, checkpoint_manifest = ac3.load_model(config, MODEL, SUITE)
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_manifest"] = checkpoint_manifest
        env, obs = restore_anchor(aa1, config, job, clean)
        try:
            rows, meta = audit_queue(ac3, legacy, v2, infer, obs, str(clean["language"]))
        finally:
            env.close()
        v1_rejected = [row for row in rows if not row["v1"]["accepted"]]
        v2_rejected = [row for row in rows if not row["v2"]["accepted"]]
        stale_reasons = {row["v1"]["reason"] for row in v1_rejected}
        stale_validator = bool(v1_rejected) and not v2_rejected and stale_reasons == {"RAW_GRIPPER_AT_THRESHOLD"}
        if stale_validator:
            status = "STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_REPLAY_PASS_CONTINUE_TO_RECOVERY"
            next_action = "B1_TARGET_EXACT_BRANCH_RECOVERY_ONCE"
        else:
            status = "STAGE_AC_AC3_G2R1_B1_TARGET_ACTION_CAUSE_UNRESOLVED_HOLD_STOP_FOR_PI"
            next_action = "STOP_FOR_PI"
        receipt.update({
            "status": status,
            "inference": {"model_inference_calls": 1, "fresh_boundary": meta["fresh_boundary"], "chunk_length": meta["chunk_length"], "queue_length_audited": QUEUE_LENGTH, "queue": rows},
            "reconciliation": {"v1_accepted": QUEUE_LENGTH - len(v1_rejected), "v1_rejected": len(v1_rejected), "v2_accepted": QUEUE_LENGTH - len(v2_rejected), "v2_rejected": len(v2_rejected), "v1_rejection_reasons": sorted(stale_reasons), "stale_validator_failure_explained": stale_validator, "official_m1_rule": "raw < 0.5 -> env +1; raw == 0.5 -> env 0; raw > 0.5 -> env -1"},
            "scientific_firewall": {"new_model_inference_calls": 1, "new_env_step_calls": 0, "new_open_intervention_steps": 0, "new_protected_reads": 0},
            "next_legal_action": next_action,
        })
    except Exception as exc:
        receipt.update({"status": "STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_HOLD_STOP_FOR_PI", "error": {"type": type(exc).__name__, "message": str(exc)}, "next_legal_action": "STOP_FOR_PI"})
    finally:
        if model is not None:
            del model
        gc.collect()
    assert_sanitized(receipt)
    write_json(receipt_path, receipt)
    ref = {"path": str(receipt_path), "bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)}
    payload = {"gate": GATE, "status": receipt["status"], "receipt": ref, "branch_id": TARGET, "scientific_firewall": receipt["scientific_firewall"]}
    root = {"schema": "STAGE_AC_AC3_G2R1_B1_ROOT_SEAL_V1", "status": receipt["status"], "gate": GATE, "root_payload": payload, "root_payload_sha256": canonical_hash(payload), "next_legal_action": receipt.get("next_legal_action")}
    assert_sanitized(root)
    write_json(root_path, root)
    return {"status": receipt["status"], "receipt": str(receipt_path), "root": str(root_path), "next_legal_action": receipt.get("next_legal_action")}


def self_test() -> None:
    assert ACTION_DIM == 7 and QUEUE_LENGTH == 8 and MIN_FREE_MIB == 20_480
    assert TARGET.startswith("AC3-") and PARENT == "libero_spatial/task_06/state_16"
    print(json.dumps({"status": "B1_INFERENCE_ONLY_STATIC_SELF_TEST_PASS"}, sort_keys=True))


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
    parser.add_argument("--target-failure", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2_PHYSICAL_V1/receipts/AC3-65bcfd948a45dd0be9ac.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2R1_B1_TARGET_INFERENCE_ONLY_V1"))
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
        return 0 if result["status"].endswith("CONTINUE_TO_RECOVERY") else 1
    except Exception as exc:
        print(json.dumps({"status": "B1_INFERENCE_ONLY_UNHANDLED_HOLD", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
