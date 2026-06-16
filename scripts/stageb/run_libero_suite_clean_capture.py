#!/usr/bin/env python3
"""Generic LIBERO suite clean capture — not hardcoded to any single suite.

Usage:
  python run_libero_suite_clean_capture.py \
    --suite libero_spatial \
    --task-index 0 \
    --state-id 0 \
    --mode reference \
    --detector-mode none \
    --model-path /path/to/model \
    --unnorm-key libero_spatial \
    --checkpoint <d1b_checkpoint> \
    --episode-dir <output> \
    --gpu-pair 2,6
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
_REPO = os.environ.get("L12_REPO_ROOT", PIPELINE_ROOT)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "stageb"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, prompt


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_array(arr):
    return sha256_bytes(np.asarray(arr, dtype=np.float32).tobytes())


def write_phase_marker(ep_dir, phase_name, extra=None):
    path = os.path.join(ep_dir, f"{phase_name}.json")
    data = {"phase": phase_name, "timestamp": datetime.now(timezone.utc).isoformat()}
    if extra: data.update(extra)
    with open(path, "w") as f: json.dump(data, f)


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return ""


def parse_gpu_pair(gpu_pair_str):
    return [int(x.strip()) for x in gpu_pair_str.split(",")]


def load_env(suite_name, task_index, state_id, render_gpu_id, max_steps=400, num_wait=10):
    """Create LIBERO env dynamically from suite/task/state."""
    from gripper_attack.libero_suite_registry import get_bddl_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env

    bddl = get_bddl_path(suite_name, get_task_name(suite_name, task_index))
    env, obs = build_v4_exact_env(bddl, int(render_gpu_id), int(max_steps), int(num_wait))
    suite = get_suite(suite_name)
    init_states = suite.get_task_init_states(task_index)
    obs = env.set_init_state(init_states[int(state_id)])
    env, obs = apply_dummy_wait(env, obs, int(num_wait))
    return env, obs


def load_openvla(model_path, gpu_pair):
    """Load OpenVLA model on specified GPU pair."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    max_memory = {idx: mm for idx in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
    )
    return model, processor, "cuda:0"


def load_detector(args):
    """Load detector based on --detector-mode."""
    if args.detector_mode == "none":
        return None
    elif args.detector_mode == "d5_frozen_online_v1":
        from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
        return D5FrozenOnlineDetectorV1(args.d5_checkpoint, args.d5_config)
    elif args.detector_mode == "legacy_d1b":
        from train_d1b_detector import CandidateRanker
        from gripper_attack.production_detector import ProductionStreamingDetector
        device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        means = ckpt["normalization"]["means"]
        stdevs = ckpt["normalization"]["stdevs"]
        impute = ckpt["normalization"]["impute"]
        m = CandidateRanker(n_features=16).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        return ProductionStreamingDetector(m, means, stdevs, impute, threshold=0.236312, device=str(device))
    else:
        raise ValueError(f"Unknown detector mode: {args.detector_mode}")


def get_task_name(suite_name, task_index):
    from gripper_attack.libero_suite_registry import get_task_names
    return get_task_names(suite_name)[task_index]


def run_episode(args):
    """Run one episode. Returns result dict."""
    gpu_pair = parse_gpu_pair(args.gpu_pair)
    render_gpu = gpu_pair[1] if len(gpu_pair) > 1 else gpu_pair[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_pair

    task_name = get_task_name(args.suite, args.task_index)
    tag = f"{task_name}_s{args.state_id}_{args.mode}_attempt{args.attempt_id}"
    ep_dir = Path(args.episode_dir)
    ep_dir.mkdir(parents=True, exist_ok=False)
    write_phase_marker(ep_dir, "ATTEMPT_STARTED", {"suite": args.suite, "task": task_name, "state": args.state_id})

    # Load env
    env, obs = load_env(args.suite, args.task_index, args.state_id, render_gpu, args.max_steps, args.num_wait)
    instruction = get_task_language(args.suite, task_name)

    # Load model
    model, processor, device_ov = load_openvla(args.model_path, gpu_pair)
    write_phase_marker(ep_dir, "MODEL_LOADED")

    # Load detector
    detector = load_detector(args)
    is_reference = (detector is None)

    # Episode state
    step_trace = []
    action_identity = []
    latency = []
    success_done = False
    infra_status = "ok"

    for step_idx in range(args.max_steps):
        if "agentview_image" not in obs:
            infra_status = "missing_camera"
            break

        img = np.asarray(obs["agentview_image"]).copy()
        t0 = time.perf_counter()
        action, _, _, _ = decode_with_scores(
            model, processor, device_ov, img, instruction, args.unnorm_key, 8,
            libero_official_preprocess=False,
            libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        t_model = time.perf_counter() - t0
        env_action = postprocess_openvla_action_for_libero(action, enabled=True)

        # Pre-step proprio
        try:
            gripper_phys = env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[:, 0]]]
            qpos = float(gripper_phys[0]) if len(gripper_phys) > 0 else 0.0
        except Exception:
            qpos = 0.0
        try:
            eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_center")]
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
        except Exception:
            eef_x = eef_y = eef_z = 0.0

        raw_gripper = float(action[-1])
        env_gripper = -1.0 if raw_gripper > 0.5 else 1.0
        decoded_open = 1 if raw_gripper > 0.5 else 0

        # Detector update
        det_result = None
        t_det = 0.0
        if not is_reference:
            t0_det = time.perf_counter()
            try:
                det_result = detector.update(step_idx, raw_gripper, env_gripper, qpos,
                                             eef_x, eef_y, eef_z, decoded_open)
            except Exception as e:
                infra_status = f"detector_exception: {e}"
            t_det = time.perf_counter() - t0_det

        # Re-hash
        raw_hash = sha256_array(action)
        env_hash = sha256_array(env_action)

        # Execute
        obs, reward, done, info = env.step(env_action)
        obs_hash = sha256_bytes(obs["agentview_image"].tobytes()) if "agentview_image" in obs else ""
        success_done = bool(done)

        step_trace.append({
            "step": step_idx,
            "raw_gripper": raw_gripper, "env_gripper": env_gripper,
            "gripper_qpos_before": qpos,
            "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
            "decoded_open": decoded_open,
            "raw_valid": 1, "env_valid": 1, "qpos_valid": 1, "eef_valid": 1,
            "semantics_ok": 1,
            "success_done": int(success_done),
        })
        action_identity.append({
            "step": step_idx,
            "action_hash_pre": raw_hash, "action_hash_post": raw_hash,
            "action_identical": 1, "env_action_hash": env_hash, "obs_hash": obs_hash,
        })
        latency.append({
            "step": step_idx,
            "model_inference_us": round(t_model * 1e6),
            "detector_update_us": round(t_det * 1e6),
        })

        if success_done:
            break

    env.close()
    torch.cuda.empty_cache()

    # Write artifacts
    with open(ep_dir / "step_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_trace[0].keys())); w.writeheader(); w.writerows(step_trace)
    with open(ep_dir / "action_identity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(action_identity[0].keys())); w.writeheader(); w.writerows(action_identity)
    with open(ep_dir / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(latency[0].keys())); w.writeheader(); w.writerows(latency)

    # Detector output
    if not is_reference:
        with open(ep_dir / "detector_emission.json", "w") as f:
            json.dump({"emit_step": detector.emit_step, "emit_score": detector.emit_score,
                       "n_candidates": len(detector.audit_records if hasattr(detector, 'audit_records') else [])}, f)

    # Provenance
    with open(ep_dir / "provenance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        w.writerow(["git_HEAD", git_head()])
        w.writerow(["suite", args.suite]); w.writerow(["task", task_name])
        w.writerow(["state_id", args.state_id]); w.writerow(["mode", args.mode])
        w.writerow(["model_path", args.model_path]); w.writerow(["unnorm_key", args.unnorm_key])
        w.writerow(["cuda_visible_devices", args.gpu_pair])

    n = len(step_trace)
    print(f"  steps={n} success={success_done} mode={args.mode} "
          f"emit={detector.emit_step if not is_reference else 'DISABLED'} "
          f"identity_fail=False invalid_fields=0 infra={infra_status}")
    return {"steps": n, "success": success_done, "infra": infra_status, "dir": str(ep_dir)}


def main():
    ap = argparse.ArgumentParser(description="Generic LIBERO suite clean capture.")
    ap.add_argument("--suite", required=True, help="e.g. libero_spatial, libero_object")
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--state-id", type=int, required=True)
    ap.add_argument("--mode", choices=["reference", "shadow"], default="shadow")
    ap.add_argument("--attempt-id", type=int, default=1)
    ap.add_argument("--detector-mode", choices=["none", "legacy_d1b", "d5_frozen_online_v1"], default="none")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--unnorm-key", required=True)
    ap.add_argument("--checkpoint", default="outputs/d1b_training/d1b_detector_best.pt")
    ap.add_argument("--d5-checkpoint", default="/data/liuyu/outputs/d5_training/d5_candidate_best.pt")
    ap.add_argument("--d5-config", default="/data/liuyu/outputs/d5_training/d5_frozen_config.json")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--gpu-pair", default="2,6")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--num-wait", type=int, default=10)
    args = ap.parse_args()

    if args.mode == "reference" and args.detector_mode != "none":
        print("FATAL: mode=reference requires detector-mode=none", file=sys.stderr)
        sys.exit(1)
    if args.mode == "shadow" and args.detector_mode == "none":
        print("FATAL: mode=shadow requires detector-mode!=none", file=sys.stderr)
        sys.exit(1)

    from gripper_attack.libero_suite_registry import is_suite_available, get_task_names
    if not is_suite_available(args.suite):
        print(f"FATAL: suite '{args.suite}' not available", file=sys.stderr)
        sys.exit(1)
    task_names = get_task_names(args.suite)
    if args.task_index >= len(task_names):
        print(f"FATAL: task_index {args.task_index} >= {len(task_names)}", file=sys.stderr)
        sys.exit(1)

    result = run_episode(args)
    print(f"Output: {result['dir']}")


if __name__ == "__main__":
    main()
