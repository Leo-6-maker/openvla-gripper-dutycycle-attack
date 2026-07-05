#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import yaml

GATE = "C6_1P_RESET_ONLY_LIBERO_STATE_RESTORE_NO_MODEL_NO_ATTACK_PROOF"
INPUT_PASS = "PASS_C6_DRY_CHAIN_THROUGH_1N"
PASS = "PASS_RESET_ONLY_LIBERO_STATE_RESTORE_PROVEN"
OUT_FILES = ["reset_only_libero_restore_proof.json", "restore_attempts.jsonl", "checksum_report.json"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_name(x):
    return "".join(ch.lower() for ch in str(x) if ch.isalnum() or ch == "_")


def resolve_task_index(bench, task_name):
    names = list(bench.get_task_names())
    if task_name in names:
        return names.index(task_name)
    target = norm_name(task_name)
    for i, name in enumerate(names):
        if norm_name(name) == target:
            return i
    raise ValueError(f"task_name not found: {task_name}; available={names}")


def find_task(tasks_config, task_id):
    for row in list((tasks_config or {}).get("tasks", [])):
        if str(row.get("task_id")) == str(task_id):
            return dict(row)
    raise KeyError(f"task_id not found in tasks config: {task_id}")


def array_digest(x):
    arr = np.asarray(x)
    contig = np.ascontiguousarray(arr)
    out = {
        "shape": [int(v) for v in contig.shape],
        "dtype": str(contig.dtype),
        "sha256": sha256_bytes(contig.tobytes()),
    }
    if np.issubdtype(contig.dtype, np.number) and contig.size:
        flat = contig.astype(np.float64, copy=False).reshape(-1)
        out.update({"min": float(np.min(flat)), "max": float(np.max(flat)), "mean": float(np.mean(flat))})
    return out


def selected_init_state_digest(init_states, state_id):
    state = init_states[int(state_id)]
    try:
        return array_digest(state)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def obs_digest(obs, camera_key):
    out = {"type": type(obs).__name__, "keys": []}
    if isinstance(obs, dict):
        out["keys"] = sorted(str(k) for k in obs.keys())
        if camera_key in obs:
            try:
                out["camera"] = {"key": camera_key, **array_digest(obs[camera_key])}
            except Exception as exc:
                out["camera"] = {"key": camera_key, "error": f"{type(exc).__name__}: {exc}"}
    return out


def sim_state_digest(env):
    out = {"qpos": {}, "qvel": {}, "combined_sha256": ""}
    qpos = None
    qvel = None
    try:
        qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
        out["qpos"] = array_digest(qpos)
    except Exception as exc:
        out["qpos"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
        out["qvel"] = array_digest(qvel)
    except Exception as exc:
        out["qvel"] = {"error": f"{type(exc).__name__}: {exc}"}
    if qpos is not None or qvel is not None:
        parts = []
        if qpos is not None:
            parts.append(np.ascontiguousarray(qpos).tobytes())
        if qvel is not None:
            parts.append(np.ascontiguousarray(qvel).tobytes())
        out["combined_sha256"] = sha256_bytes(b"".join(parts))
    return out


def torch_cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available()), ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def dry_chain_status_ok(obj):
    return str((obj or {}).get("status", "")) == INPUT_PASS


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    observed = sha256_file(args.input_c6_dry_chain_json)
    status = PASS
    reason = ""
    attempts = []
    env = None
    task = {}
    task_index = None
    init_state_count = None
    cuda_ok, cuda_error = torch_cuda_available()

    if observed != args.expected_c6_dry_chain_sha256:
        status = "HOLD_C6_DRY_CHAIN_HASH_MISMATCH"
        reason = observed
    else:
        chain = read_json(args.input_c6_dry_chain_json)
        if not dry_chain_status_ok(chain):
            status = "HOLD_C6_DRY_CHAIN_STATUS_NOT_PASS"
            reason = str(chain.get("status", ""))
        else:
            pre = dict(chain.get("preview_precheck") or {})
            if str(pre.get("task_id", "")) != str(args.task_id):
                status = "HOLD_DRY_CHAIN_TASK_ID_MISMATCH"
                reason = json.dumps(pre, sort_keys=True)
            elif str(pre.get("state_ids", "")) != str(args.state_id):
                status = "HOLD_DRY_CHAIN_STATE_ID_MISMATCH"
                reason = json.dumps(pre, sort_keys=True)
            elif args.require_cuda and not cuda_ok:
                status = "HOLD_CUDA_NOT_AVAILABLE"
                reason = cuda_error or "torch.cuda.is_available() is false"

    if status == PASS:
        try:
            cfg = load_yaml(args.tasks_config)
            task = find_task(cfg, args.task_id)
        except Exception as exc:
            status = "HOLD_TASK_CONFIG_RESOLUTION_FAILED"
            reason = f"{type(exc).__name__}: {exc}"

    if status == PASS:
        try:
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import OffScreenRenderEnv
            bench = get_benchmark(task["suite"])()
            task_index = int(resolve_task_index(bench, task["task_name"]))
            init_states = bench.get_task_init_states(task_index)
            init_state_count = int(len(init_states))
            sid = int(args.state_id)
            if sid < 0 or sid >= init_state_count:
                raise IndexError(f"state_id={sid} outside init_state_count={init_state_count}")
            bddl = bench.get_task_bddl_file_path(task_index)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=int(args.image_size),
                camera_widths=int(args.image_size),
                render_gpu_device_id=int(args.render_gpu_device_id),
                horizon=int(args.horizon),
            )
            try:
                env.seed(int(args.env_seed))
            except Exception:
                pass
            init_digest = selected_init_state_digest(init_states, sid)
            for repeat_idx in range(int(args.repeat)):
                t0 = time.time()
                obs_reset = env.reset()
                obs_restore = env.set_init_state(init_states[sid])
                attempt = {
                    "repeat_idx": int(repeat_idx),
                    "state_id": int(sid),
                    "task_id": str(task["task_id"]),
                    "suite": str(task["suite"]),
                    "task_name": str(task["task_name"]),
                    "task_index": int(task_index),
                    "bddl_file_name": str(bddl),
                    "selected_init_state_digest": init_digest,
                    "obs_reset_digest": obs_digest(obs_reset, args.camera_obs_key),
                    "obs_restore_digest": obs_digest(obs_restore, args.camera_obs_key),
                    "sim_state_after_restore": sim_state_digest(env),
                    "elapsed_seconds": float(time.time() - t0),
                }
                attempts.append(attempt)
        except Exception as exc:
            status = "HOLD_LIBERO_RESET_ONLY_RESTORE_FAILED"
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    if status == PASS:
        if len(attempts) != int(args.repeat):
            status = "HOLD_RESTORE_REPEAT_COUNT_MISMATCH"
            reason = f"attempts={len(attempts)} repeat={args.repeat}"
        else:
            hashes = [str(a.get("sim_state_after_restore", {}).get("combined_sha256", "")) for a in attempts]
            if not all(hashes):
                status = "HOLD_RESTORE_SIM_STATE_HASH_MISSING"
                reason = json.dumps(hashes)
            elif len(set(hashes)) != 1:
                status = "HOLD_RESTORE_SIM_STATE_HASH_NOT_REPEATABLE"
                reason = json.dumps(hashes)

    attempts_path = out / "restore_attempts.jsonl"
    write_jsonl(attempts_path, attempts)
    write_csv(out / "restore_attempts_summary.csv", [
        {
            "repeat_idx": a.get("repeat_idx"),
            "state_id": a.get("state_id"),
            "task_id": a.get("task_id"),
            "task_index": a.get("task_index"),
            "sim_state_sha256": a.get("sim_state_after_restore", {}).get("combined_sha256", ""),
            "camera_sha256": a.get("obs_restore_digest", {}).get("camera", {}).get("sha256", ""),
            "elapsed_seconds": a.get("elapsed_seconds", ""),
        }
        for a in attempts
    ], ["repeat_idx", "state_id", "task_id", "task_index", "sim_state_sha256", "camera_sha256", "elapsed_seconds"])

    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_c6_dry_chain_json_sha256": observed,
        "expected_c6_dry_chain_json_sha256": args.expected_c6_dry_chain_sha256,
        "task_id": str(args.task_id),
        "state_id": int(args.state_id),
        "task": task,
        "task_index": task_index,
        "init_state_count": init_state_count,
        "repeat": int(args.repeat),
        "cuda_available": cuda_ok,
        "cuda_error": cuda_error,
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attempt_count": len(attempts),
        "attempts_path": str(attempts_path),
        "boundaries": {
            "legacy_runner_execution": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "RESET_ONLY",
            "env_reset": "PERFORMED",
            "env_set_init_state": "PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "reset_only_libero_restore_proof.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-dry-chain-json", required=True)
    p.add_argument("--expected-c6-dry-chain-sha256", required=True)
    p.add_argument("--tasks-config", default="configs/v4_tasks_libero.yaml")
    p.add_argument("--task-id", default="libero_goal_open_middle_drawer")
    p.add_argument("--state-id", type=int, default=0)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--camera-obs-key", default="agentview_image")
    p.add_argument("--render-gpu-device-id", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
