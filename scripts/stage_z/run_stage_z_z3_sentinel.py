#!/usr/bin/env python3
"""Run one permanently excluded Z3 engineering sentinel for one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_stage_z_z1_runtime_canary as z1
from stage_z_preparation.action_semantics import validate_action_pair
from stage_z_preparation.z3_contract import MODEL_M0, MODEL_M1, MODEL_M2, command_open_action, arm_delta_linf


MODELS = (MODEL_M0, MODEL_M1, MODEL_M2)
PARENT = "libero_10/task_04/state_03"
SUITE = "libero_10"
TASK_IDX = 4
STATE_ID = 3
ROLE = "PRIMARY"
DOSE = 3


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_env(config: dict[str, Any], counters: dict[str, int]):
    env, task_suite, task = z1.make_libero_env(config, SUITE, TASK_IDX)
    env.reset()
    initial_states = task_suite.get_task_init_states(TASK_IDX)
    obs = env.set_init_state(initial_states[STATE_ID])
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(int(config["environment"]["dummy_wait_steps"])):
        obs = env.step(dummy)[0]
        counters["env_step_calls"] += 1
    return env, task, obs


def capture_clean(model_family: str, infer: Any, env: Any, task: Any, obs: dict[str, Any], counters: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    language = str(task.language)
    if model_family == MODEL_M0:
        for index in range(DOSE):
            final, meta = infer(obs, language)
            counters["model_inference_calls"] += 1
            raw = np.asarray(meta["raw_action"], dtype=np.float32)
            final = np.asarray(final, dtype=np.float32)
            boundary = "FRESH_PER_STEP"
            obs = env.step(final.tolist())[0]
            counters["env_step_calls"] += 1
            rows.append({"index": index, "raw_action": raw.tolist(), "final_action": final.tolist(), "boundary": boundary})
        return rows, {"boundary": "FRESH_PER_STEP", "chunk_length": 1, "inference_calls": DOSE}
    chunk, meta = infer(obs, language)
    counters["model_inference_calls"] += 1
    raw_chunk = np.asarray(meta["raw_action_chunk"], dtype=np.float32)
    final_chunk = np.asarray(chunk, dtype=np.float32)
    expected = 8 if model_family == MODEL_M1 else 5
    if raw_chunk.ndim != 2 or final_chunk.ndim != 2 or raw_chunk.shape[1] != 7 or final_chunk.shape[1] != 7 or raw_chunk.shape[0] < expected or final_chunk.shape[0] < expected:
        raise RuntimeError(f"SENTINEL_CHUNK_TOO_SHORT:{model_family}")
    expected_boundary = "FRESH_OFT_ACTION_QUEUE" if model_family == MODEL_M1 else "FRESH_PI05_REPLAN"
    if meta.get("fresh_boundary") != expected_boundary:
        raise RuntimeError(f"SENTINEL_BOUNDARY_MISMATCH:{model_family}")
    for index in range(DOSE):
        raw = raw_chunk[index]
        final = final_chunk[index]
        obs = env.step(final.tolist())[0]
        counters["env_step_calls"] += 1
        rows.append({"index": index, "raw_action": raw.tolist(), "final_action": final.tolist(), "boundary": expected_boundary})
    return rows, {"boundary": expected_boundary, "chunk_length": int(raw_chunk.shape[0]), "consume_steps": DOSE, "source_contract_length": expected}


def run_sentinel(config: dict[str, Any], protocol: dict[str, Any], ledger: dict[str, Any], args: argparse.Namespace, receipt: dict[str, Any]) -> dict[str, Any]:
    z1.require_single_visible_gpu(args.gpu_id)
    gpu = z1.gpu_snapshot(args.gpu_id)
    canary = z1.static_authority(config, ledger, parent_key=PARENT, suite=SUITE, role=ROLE)
    if args.model_family not in MODELS:
        raise RuntimeError("SENTINEL_MODEL_INVALID")
    model_spec = config["model_families"][args.model_family]
    checkpoint = model_spec["paths"][SUITE] if args.model_family == MODEL_M0 else (str(Path(model_spec["checkpoint_root"]) / SUITE) if args.model_family == MODEL_M1 else model_spec["checkpoint"])
    checkpoint_manifest = None
    if args.model_family == MODEL_M1:
        checkpoint_manifest = z1.verify_m1_materialization(Path(args.m1_manifest), Path(checkpoint), SUITE, str(model_spec["checkpoint_manifests_sha256"]))
    if args.model_family == MODEL_M0:
        infer, model, normalization = z1.load_openvla(checkpoint, oft=False, suite=SUITE, return_chunk=True)
    elif args.model_family == MODEL_M1:
        infer, model, normalization = z1.load_openvla(checkpoint, oft=True, suite=SUITE, return_chunk=True)
    else:
        infer, model = z1.load_pi05(checkpoint, return_chunk=True)
        normalization = {"checkpoint_mutated": False}
    try:
        counters = receipt["runtime_counters"]
        clean_env, task, obs = prepare_env(config, counters)
        try:
            clean_pre = z1.snapshot_state(clean_env)
            clean_rows, boundary = capture_clean(args.model_family, infer, clean_env, task, obs, counters)
            clean_post = z1.snapshot_state(clean_env)
            z1.restore_state(clean_env, clean_pre)
            restore_exact = np.array_equal(clean_pre, z1.snapshot_state(clean_env))
            if not restore_exact:
                raise RuntimeError("SENTINEL_SNAPSHOT_RESTORE_NOT_EXACT")
        finally:
            clean_env.close()
        open_env, _open_task, _open_obs = prepare_env(config, counters)
        try:
            open_pre = z1.snapshot_state(open_env)
            if not np.array_equal(clean_pre, open_pre):
                raise RuntimeError("SENTINEL_FRESH_BRANCH_PRE_STATE_MISMATCH")
            open_rows: list[dict[str, Any]] = []
            for base in clean_rows:
                raw, final = command_open_action(args.model_family, base["raw_action"], base["final_action"], duration=DOSE)
                check = validate_action_pair(args.model_family, raw, final, raw_gripper=raw[-1], final_gripper=final[-1])
                if args.model_family == MODEL_M2 and not check["accepted"]:
                    raise RuntimeError(f"SENTINEL_M2_ACTION_SEMANTICS:{check['reason']}")
                if arm_delta_linf(base["final_action"], final) > 1e-7:
                    raise RuntimeError("SENTINEL_ARM_DRIFT")
                _open_obs = open_env.step(list(final))[0]
                counters["env_step_calls"] += 1
                counters["engineering_command_open_steps"] += 1
                open_rows.append({"index": base["index"], "raw_action": list(raw), "final_action": list(final), "raw_native_open": raw[-1], "final_native_open": final[-1], "arm_delta_linf": arm_delta_linf(base["final_action"], final), "action_semantics": check})
        finally:
            open_env.close()
        receipt.update({
            "status": "PASS_Z3_ENGINEERING_SENTINEL",
            "gpu": gpu,
            "canary": {"canonical_parent_key": PARENT, "suite": SUITE, "role": ROLE, "selection_rank_sha256": canary["selection_rank_sha256"], "permanent_exclusion": True},
            "checkpoint": checkpoint,
            "checkpoint_manifest": checkpoint_manifest,
            "normalization": normalization,
            "boundary": boundary,
            "branch_replay": {"snapshot_restore_exact": True, "fresh_pre_state_exact": True, "clean_pre_state_sha256": sha_bytes(clean_pre.tobytes()), "clean_post_state_sha256": sha_bytes(clean_post.tobytes()), "open_pre_state_sha256": sha_bytes(open_pre.tobytes())},
            "clean_rows": clean_rows,
            "open_rows": open_rows,
            "claim_boundary": "Engineering sentinel only; no scientific parent, physical endpoint, V_phys, task success, or protected evaluation claim.",
            "next_legal_action": "Z3_C_SCIENTIFIC_MATRIX_AFTER_PI_REVIEW",
        })
        return receipt
    finally:
        del model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODELS, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    if protocol.get("status") != "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z3_PROTOCOL_NOT_FROZEN")
    receipt: dict[str, Any] = {"schema": "STAGE_Z_Z3_ENGINEERING_SENTINEL_RECEIPT_V1", "status": "RUNNING", "model_family": args.model_family, "canonical_parent_key": PARENT, "suite": SUITE, "role": ROLE, "gpu_id": args.gpu_id, "protocol_sha256": sha_bytes(args.protocol.read_bytes()), "runtime_counters": {"model_inference_calls": 0, "env_step_calls": 0, "engineering_command_open_steps": 0, "scientific_parent_exposure": 0, "vphys_reads": 0, "task_success_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0, "attacked_env_steps": 0}, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write(args.output, receipt)
    try:
        result = run_sentinel(load(args.config), protocol, load(args.ledger), args, receipt)
        atomic_write(args.output, result)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "output": str(args.output)}, sort_keys=True))
        return 0
    except Exception as exc:
        receipt.update({"status": "ENGINEERING_INVALID_Z3_SENTINEL", "error": f"{type(exc).__name__}:{exc}", "next_legal_action": "STOP_FOR_PI"})
        atomic_write(args.output, receipt)
        print(json.dumps({"status": receipt["status"], "model_family": args.model_family, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
