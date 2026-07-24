#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gripper_attack.openvla_preprocess import resolve_backend
from v4_run_eval_openvla import (
    action_token_ids_from_gen,
    decode_with_scores,
    get_instruction,
    load_model,
    postprocess_openvla_action_for_libero,
    resolve_task_index,
    resolve_unnorm_key,
)

GATE = "C6_1R_MULTISUITE_SINGLE_CLEAN_DECODE_NO_STEP"
ACCEPTED_INPUT = {
    "PASS_MULTISUITE_RESET_OBSERVATION_BINDING_AUDITED",
    "PASS_RESET_STATE_AND_PREPROCESS_STABLE_RAW_CAMERA_NONBITWISE_AUDITED",
    "PASS_RESET_STATE_STABLE_PREPROCESS_OBSERVATION_NONBITWISE_AUDITED",
}
PASS = "PASS_MULTISUITE_SINGLE_CLEAN_DECODE_NO_STEP"
OUT_FILES = ["multisuite_single_clean_decode_no_step.json", "clean_decode_records.jsonl", "clean_decode_summary.csv", "checksum_report.json"]
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


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


def expand_path(text):
    return os.path.expandvars(os.path.expanduser(str(text or "")))


def unresolved_env_vars(text):
    return sorted({a or b for a, b in ENV_REF_RE.findall(str(text or ""))})


def apply_model_env_overrides(args):
    applied = {}
    if str(getattr(args, "openvla_model_root", "") or "").strip():
        value = expand_path(args.openvla_model_root)
        os.environ["OPENVLA_MODEL_ROOT"] = value
        applied["OPENVLA_MODEL_ROOT"] = value
    if str(getattr(args, "openvla_base_model_dir", "") or "").strip():
        value = expand_path(args.openvla_base_model_dir)
        os.environ["OPENVLA_BASE_MODEL_DIR"] = value
        applied["OPENVLA_BASE_MODEL_DIR"] = value
    return applied


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


def torch_cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available()), "", int(torch.cuda.device_count())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", 0


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def validate_parent(args):
    observed = sha256_file(args.input_c6_1q_json)
    if observed != args.expected_c6_1q_sha256:
        return observed, "HOLD_C6_1Q_HASH_MISMATCH", observed
    obj = read_json(args.input_c6_1q_json)
    status = str(obj.get("status", ""))
    if status not in ACCEPTED_INPUT:
        return observed, "HOLD_C6_1Q_STATUS_NOT_ACCEPTED", status
    return observed, "", ""


def resolve_model_path(attack_cfg, task, explicit_model_path):
    if str(explicit_model_path or "").strip():
        model_path = expand_path(explicit_model_path)
    else:
        paths = dict((attack_cfg or {}).get("model_paths", {}) or {})
        suite = str(task.get("suite", ""))
        model_path = expand_path(paths.get(suite) or paths.get("base") or paths.get("libero_goal") or "")
    unresolved = unresolved_env_vars(model_path)
    if unresolved:
        raise EnvironmentError(f"unresolved model path environment variables for task={task.get('task_id')}: {unresolved}; path={model_path}")
    return model_path


def obs_image(obs, camera_key):
    if not isinstance(obs, dict):
        raise KeyError(f"observation is not a dict: {type(obs).__name__}")
    if camera_key not in obs:
        raise KeyError(f"camera key {camera_key!r} missing; keys={sorted(str(k) for k in obs.keys())}")
    arr = np.asarray(obs[camera_key])
    if arr.ndim != 3:
        raise ValueError(f"camera image has ndim={arr.ndim}, shape={arr.shape}")
    return arr


def decode_task(task, args, attack_cfg):
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = get_benchmark(task["suite"])()
    task_index = int(resolve_task_index(bench, task["task_name"]))
    init_states = bench.get_task_init_states(task_index)
    sid = int(args.state_id)
    if sid < 0 or sid >= len(init_states):
        raise IndexError(f"state_id={sid} outside init_state_count={len(init_states)} for {task['task_id']}")
    bddl = bench.get_task_bddl_file_path(task_index)
    base_instruction = get_instruction(bench, task_index, task["task_name"])
    model_path = resolve_model_path(attack_cfg, task, args.model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"model path does not exist for task={task['task_id']}: {model_path}")

    env = None
    model = None
    processor = None
    try:
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
        _ = env.reset()
        obs = env.set_init_state(init_states[sid])
        if bool(args.sim_forward):
            try:
                env.sim.forward()
            except Exception:
                pass
        image_np = obs_image(obs, args.camera_obs_key)
        image_info = array_digest(image_np)
        model, processor, device = load_model(model_path, model_gpu_device_id=int(args.model_gpu_device_id))
        unnorm_args = SimpleNamespace(unnorm_key=str(task.get("default_unnorm_key", "") or task.get("suite", "")))
        unnorm = resolve_unnorm_key(unnorm_args, task, model)
        t0 = time.time()
        action, prefix_logits, decode_seconds, gen = decode_with_scores(
            model,
            processor,
            device,
            image_np,
            base_instruction,
            unnorm,
            int(args.k_trigger),
            libero_official_preprocess=True,
            libero_preprocess_backend=args.preprocess_backend,
            center_crop=True,
            resize_size=int(args.preprocess_size),
            drop_attention_mask=(not bool(args.keep_attention_mask)),
        )
        total_seconds = time.time() - t0
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        env_action = postprocess_openvla_action_for_libero(action, enabled=bool(args.postprocess_gripper))
        action_dim = int(model.get_action_dim(unnorm))
        token_ids = action_token_ids_from_gen(gen, action_dim)
        logits_shape = [] if prefix_logits is None else [int(v) for v in np.asarray(prefix_logits).shape]
        return {
            "task_id": str(task["task_id"]),
            "suite": str(task["suite"]),
            "task_name": str(task["task_name"]),
            "task_index": int(task_index),
            "state_id": int(sid),
            "init_state_count": int(len(init_states)),
            "bddl_file_name": str(bddl),
            "instruction": str(base_instruction),
            "model_path": str(model_path),
            "device": str(device),
            "unnorm_key": str(unnorm),
            "camera_digest": image_info,
            "preprocess_backend": resolve_backend(args.preprocess_backend),
            "preprocess_size": int(args.preprocess_size),
            "action_dim": int(action_dim),
            "clean_action": action.tolist(),
            "clean_env_action": np.asarray(env_action, dtype=np.float32).reshape(-1).tolist(),
            "clean_gripper_raw": float(action[-1]) if action.size else None,
            "clean_gripper_env": float(np.asarray(env_action).reshape(-1)[-1]) if np.asarray(env_action).size else None,
            "clean_token_ids": [int(x) for x in token_ids],
            "clean_gripper_token": int(token_ids[-1]) if token_ids else None,
            "prefix_logits_shape": logits_shape,
            "decode_seconds": float(decode_seconds),
            "total_seconds": float(total_seconds),
            "status": "PASS_SINGLE_CLEAN_DECODE",
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        del processor
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    status = PASS
    reason = ""
    records = []
    env_overrides = apply_model_env_overrides(args)
    q_sha, hold, hold_reason = validate_parent(args)
    if hold:
        status = hold
        reason = hold_reason
    cuda_ok, cuda_error, cuda_count = torch_cuda_available()
    if status == PASS and args.require_cuda and not cuda_ok:
        status = "HOLD_CUDA_NOT_AVAILABLE"
        reason = cuda_error or "torch.cuda.is_available() is false"
    tasks = []
    attack_cfg = {}
    if status == PASS:
        try:
            resolve_backend(args.preprocess_backend)
            tasks = load_tasks(args.tasks_config, args.task_ids)
            attack_cfg = load_yaml(args.attack_config)
        except Exception as exc:
            status = "HOLD_CONFIG_LOAD_FAILED"
            reason = f"{type(exc).__name__}: {exc}"
    if status == PASS:
        try:
            for task in tasks:
                records.append(decode_task(task, args, attack_cfg))
        except Exception as exc:
            status = "HOLD_SINGLE_CLEAN_DECODE_FAILED"
            reason = f"{type(exc).__name__}: {exc}"
    if status == PASS:
        if len(records) != len(tasks):
            status = "HOLD_DECODE_RECORD_COUNT_MISMATCH"
            reason = f"records={len(records)} tasks={len(tasks)}"
        elif any(str(r.get("status", "")) != "PASS_SINGLE_CLEAN_DECODE" for r in records):
            status = "HOLD_SINGLE_CLEAN_DECODE_RECORD_NOT_PASS"
            reason = json.dumps(records, sort_keys=True)

    records_path = out / "clean_decode_records.jsonl"
    summary_path = out / "clean_decode_summary.csv"
    write_jsonl(records_path, records)
    fields = ["task_id", "suite", "state_id", "task_index", "model_path", "device", "unnorm_key", "action_dim", "clean_gripper_raw", "clean_gripper_env", "clean_gripper_token", "decode_seconds", "total_seconds", "status"]
    write_csv(summary_path, records, fields)
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_c6_1q_json_sha256": q_sha,
        "expected_c6_1q_json_sha256": args.expected_c6_1q_sha256,
        "task_ids": [str(t.get("task_id")) for t in tasks],
        "state_id": int(args.state_id),
        "preprocess_backend": resolve_backend(args.preprocess_backend),
        "attack_config": str(args.attack_config),
        "tasks_config": str(args.tasks_config),
        "model_env_overrides": env_overrides,
        "model_env_snapshot": {
            "OPENVLA_MODEL_ROOT": os.environ.get("OPENVLA_MODEL_ROOT", ""),
            "OPENVLA_BASE_MODEL_DIR": os.environ.get("OPENVLA_BASE_MODEL_DIR", ""),
        },
        "record_count": len(records),
        "cuda_available": cuda_ok,
        "cuda_error": cuda_error,
        "cuda_device_count": cuda_count,
        "host": socket.gethostname(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
        "output_files": {"records": str(records_path), "summary": str(summary_path)},
        "boundaries": {
            "OpenVLA_model": "LOADED",
            "model_inference": "SINGLE_CLEAN_DECODE_ONLY",
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
    write_json(out / "multisuite_single_clean_decode_no_step.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not str(status).startswith("HOLD_") else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1q-json", required=True)
    p.add_argument("--expected-c6-1q-sha256", required=True)
    p.add_argument("--tasks-config", default="configs/v4_tasks_libero.yaml")
    p.add_argument("--attack-config", default="configs/v4_attack.yaml")
    p.add_argument("--task-ids", default="")
    p.add_argument("--state-id", type=int, default=0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--camera-obs-key", default="agentview_image")
    p.add_argument("--render-gpu-device-id", type=int, default=0)
    p.add_argument("--model-gpu-device-id", type=int, default=-1)
    p.add_argument("--model-path", default="")
    p.add_argument("--openvla-model-root", default="")
    p.add_argument("--openvla-base-model-dir", default="")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--sim-forward", action="store_true")
    p.add_argument("--preprocess-backend", default="upstream_tf_jpeg")
    p.add_argument("--preprocess-size", type=int, default=224)
    p.add_argument("--k-trigger", type=int, default=8)
    p.add_argument("--keep-attention-mask", action="store_true")
    p.add_argument("--postprocess-gripper", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
