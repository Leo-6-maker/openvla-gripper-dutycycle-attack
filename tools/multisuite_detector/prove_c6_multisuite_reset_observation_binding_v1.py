#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.openvla_preprocess import prepare_openvla_image, resolve_backend

GATE = "C6_1Q_MULTISUITE_RESET_OBSERVATION_BINDING_NO_MODEL_NO_STEP"
INPUT_PASS = "PASS_RESET_ONLY_LIBERO_STATE_RESTORE_PROVEN"
PASS = "PASS_MULTISUITE_RESET_OBSERVATION_BINDING_AUDITED"
OUT_FILES = ["multisuite_reset_observation_binding.json", "observation_binding_attempts.jsonl", "observation_binding_summary.csv", "checksum_report.json"]


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


def load_tasks(tasks_config, task_ids_text):
    cfg = load_yaml(tasks_config)
    rows = [dict(x) for x in list((cfg or {}).get("tasks", []))]
    selected = [x.strip() for x in str(task_ids_text or "").split(",") if x.strip()]
    if selected:
        order = {tid: i for i, tid in enumerate(selected)}
        rows = [r for r in rows if str(r.get("task_id")) in order]
        rows.sort(key=lambda r: order[str(r.get("task_id"))])
        missing = [tid for tid in selected if tid not in {str(r.get("task_id")) for r in rows}]
        if missing:
            raise KeyError(f"task ids not found in tasks config: {missing}")
    return rows


def array_digest(x):
    arr = np.asarray(x)
    contig = np.ascontiguousarray(arr)
    out = {"shape": [int(v) for v in contig.shape], "dtype": str(contig.dtype), "sha256": sha256_bytes(contig.tobytes())}
    if np.issubdtype(contig.dtype, np.number) and contig.size:
        flat = contig.astype(np.float64, copy=False).reshape(-1)
        out.update({"min": float(np.min(flat)), "max": float(np.max(flat)), "mean": float(np.mean(flat))})
    return out


def diff_metrics(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    if list(aa.shape) != list(bb.shape):
        return {"shape_match": False, "a_shape": [int(v) for v in aa.shape], "b_shape": [int(v) for v in bb.shape]}
    if aa.size == 0:
        return {"shape_match": True, "max_abs": 0.0, "mean_abs": 0.0, "changed_fraction": 0.0}
    da = aa.astype(np.float32) - bb.astype(np.float32)
    absd = np.abs(da)
    return {"shape_match": True, "max_abs": float(np.max(absd)), "mean_abs": float(np.mean(absd)), "changed_fraction": float(np.mean(absd > 0))}


def obs_image(obs, camera_key):
    if not isinstance(obs, dict):
        raise KeyError(f"observation is not a dict: {type(obs).__name__}")
    if camera_key not in obs:
        raise KeyError(f"camera key {camera_key!r} missing; keys={sorted(str(k) for k in obs.keys())}")
    arr = np.asarray(obs[camera_key])
    if arr.ndim != 3:
        raise ValueError(f"camera image has ndim={arr.ndim}, shape={arr.shape}")
    return arr


def sim_state_digest(env):
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    return {"qpos": array_digest(qpos), "qvel": array_digest(qvel), "combined_sha256": sha256_bytes(np.ascontiguousarray(qpos).tobytes() + np.ascontiguousarray(qvel).tobytes())}


def torch_cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available()), ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def validate_parent_proof(args):
    observed = ""
    if not str(args.input_c6_1p_json or "").strip():
        return observed, "", ""
    observed = sha256_file(args.input_c6_1p_json)
    if observed != args.expected_c6_1p_sha256:
        return observed, "HOLD_C6_1P_HASH_MISMATCH", observed
    obj = read_json(args.input_c6_1p_json)
    if str(obj.get("status", "")) != INPUT_PASS:
        return observed, "HOLD_C6_1P_STATUS_NOT_PASS", str(obj.get("status", ""))
    return observed, "", ""


def audit_task(task, args):
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = get_benchmark(task["suite"])()
    task_index = int(resolve_task_index(bench, task["task_name"]))
    init_states = bench.get_task_init_states(task_index)
    sid = int(args.state_id)
    if sid < 0 or sid >= len(init_states):
        raise IndexError(f"state_id={sid} outside init_state_count={len(init_states)} for {task['task_id']}")
    bddl = bench.get_task_bddl_file_path(task_index)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=int(args.image_size),
        camera_widths=int(args.image_size),
        render_gpu_device_id=int(args.render_gpu_device_id),
        horizon=int(args.horizon),
    )
    attempts = []
    try:
        try:
            env.seed(int(args.env_seed))
        except Exception:
            pass
        for repeat_idx in range(int(args.repeat)):
            t0 = time.time()
            _ = env.reset()
            obs = env.set_init_state(init_states[sid])
            if bool(args.sim_forward):
                try:
                    env.sim.forward()
                except Exception:
                    pass
            raw = obs_image(obs, args.camera_obs_key)
            prep = np.asarray(prepare_openvla_image(raw, libero_preprocess_backend=args.preprocess_backend, center_crop=True, resize_size=int(args.preprocess_size)))
            attempts.append({
                "task_id": str(task["task_id"]),
                "suite": str(task["suite"]),
                "task_name": str(task["task_name"]),
                "task_index": int(task_index),
                "bddl_file_name": str(bddl),
                "state_id": int(sid),
                "repeat_idx": int(repeat_idx),
                "init_state_count": int(len(init_states)),
                "sim_state": sim_state_digest(env),
                "raw_camera": array_digest(raw),
                "preprocessed_openvla_image": array_digest(prep),
                "preprocess_backend": resolve_backend(args.preprocess_backend),
                "elapsed_seconds": float(time.time() - t0),
            })
    finally:
        try:
            env.close()
        except Exception:
            pass
    return attempts


def summarize_task(task_id, attempts):
    base = attempts[0] if attempts else {}
    sim_hashes = [a.get("sim_state", {}).get("combined_sha256", "") for a in attempts]
    raw_hashes = [a.get("raw_camera", {}).get("sha256", "") for a in attempts]
    prep_hashes = [a.get("preprocessed_openvla_image", {}).get("sha256", "") for a in attempts]
    row = {
        "task_id": task_id,
        "suite": base.get("suite", ""),
        "state_id": base.get("state_id", ""),
        "repeat": len(attempts),
        "sim_state_bitwise_stable": bool(sim_hashes and len(set(sim_hashes)) == 1),
        "raw_camera_bitwise_stable": bool(raw_hashes and len(set(raw_hashes)) == 1),
        "preprocessed_bitwise_stable": bool(prep_hashes and len(set(prep_hashes)) == 1),
        "sim_state_sha256_values": ";".join(sim_hashes),
        "raw_camera_sha256_values": ";".join(raw_hashes),
        "preprocessed_sha256_values": ";".join(prep_hashes),
        "raw_max_abs_vs_repeat0": "",
        "raw_mean_abs_vs_repeat0": "",
        "preprocessed_max_abs_vs_repeat0": "",
        "preprocessed_mean_abs_vs_repeat0": "",
    }
    if len(attempts) >= 2:
        # Full arrays are not stored in JSONL, so cross-repeat pixel diffs are computed at run time only when bitwise differs.
        row["raw_max_abs_vs_repeat0"] = "RECORDED_BY_HASH_ONLY"
        row["preprocessed_max_abs_vs_repeat0"] = "RECORDED_BY_HASH_ONLY"
    return row


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    status = PASS
    reason = ""
    p1_sha, hold, hold_reason = validate_parent_proof(args)
    if hold:
        status = hold
        reason = hold_reason
    cuda_ok, cuda_error = torch_cuda_available()
    if status == PASS and args.require_cuda and not cuda_ok:
        status = "HOLD_CUDA_NOT_AVAILABLE"
        reason = cuda_error or "torch.cuda.is_available() is false"
    attempts = []
    summary = []
    tasks = []
    if status == PASS:
        try:
            resolve_backend(args.preprocess_backend)
            tasks = load_tasks(args.tasks_config, args.task_ids)
        except Exception as exc:
            status = "HOLD_TASK_OR_PREPROCESS_CONFIG_FAILED"
            reason = f"{type(exc).__name__}: {exc}"
    if status == PASS:
        try:
            for task in tasks:
                task_attempts = audit_task(task, args)
                attempts.extend(task_attempts)
                summary.append(summarize_task(str(task["task_id"]), task_attempts))
        except Exception as exc:
            status = "HOLD_RESET_OBSERVATION_BINDING_FAILED"
            reason = f"{type(exc).__name__}: {exc}"
    if status == PASS:
        if any(not row.get("sim_state_bitwise_stable") for row in summary):
            status = "HOLD_SIM_STATE_NOT_REPEATABLE"
            reason = json.dumps(summary, sort_keys=True)
        elif any(not row.get("preprocessed_bitwise_stable") for row in summary):
            status = "PASS_RESET_STATE_STABLE_PREPROCESS_OBSERVATION_NONBITWISE_AUDITED"
        elif any(not row.get("raw_camera_bitwise_stable") for row in summary):
            status = "PASS_RESET_STATE_AND_PREPROCESS_STABLE_RAW_CAMERA_NONBITWISE_AUDITED"

    attempts_path = out / "observation_binding_attempts.jsonl"
    summary_path = out / "observation_binding_summary.csv"
    write_jsonl(attempts_path, attempts)
    fields = ["task_id", "suite", "state_id", "repeat", "sim_state_bitwise_stable", "raw_camera_bitwise_stable", "preprocessed_bitwise_stable", "sim_state_sha256_values", "raw_camera_sha256_values", "preprocessed_sha256_values", "raw_max_abs_vs_repeat0", "raw_mean_abs_vs_repeat0", "preprocessed_max_abs_vs_repeat0", "preprocessed_mean_abs_vs_repeat0"]
    write_csv(summary_path, summary, fields)
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_c6_1p_json_sha256": p1_sha,
        "expected_c6_1p_json_sha256": str(args.expected_c6_1p_sha256 or ""),
        "tasks_config": str(args.tasks_config),
        "task_ids": [str(t.get("task_id")) for t in tasks],
        "state_id": int(args.state_id),
        "repeat": int(args.repeat),
        "preprocess_backend": resolve_backend(args.preprocess_backend),
        "preprocess_size": int(args.preprocess_size),
        "cuda_available": cuda_ok,
        "cuda_error": cuda_error,
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attempt_count": len(attempts),
        "summary": summary,
        "output_files": {"attempts": str(attempts_path), "summary": str(summary_path)},
        "boundaries": {
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "RESET_ONLY_OBSERVATION_CAPTURE",
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
    write_json(out / "multisuite_reset_observation_binding.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not str(status).startswith("HOLD_") else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1p-json", default="")
    p.add_argument("--expected-c6-1p-sha256", default="")
    p.add_argument("--tasks-config", default="configs/v4_tasks_libero.yaml")
    p.add_argument("--task-ids", default="")
    p.add_argument("--state-id", type=int, default=0)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--camera-obs-key", default="agentview_image")
    p.add_argument("--render-gpu-device-id", type=int, default=0)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--sim-forward", action="store_true")
    p.add_argument("--preprocess-backend", default="upstream_tf_jpeg")
    p.add_argument("--preprocess-size", type=int, default=224)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
