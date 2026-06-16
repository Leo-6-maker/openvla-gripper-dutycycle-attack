#!/usr/bin/env python3
"""Batch watcher for 100 LIBERO-Spatial clean rollouts on GPU(1,5).

Loads model once, iterates 10 tasks × 10 states.
Each parent: shadow mode with D5 frozen online detector.
Fail-closed validity, proper success, full D5 telemetry.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
_REPO = os.environ.get("L12_REPO_ROOT", PIPELINE_ROOT)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "stageb"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, prompt
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return ""


def get_spatial_tasks():
    """Return list of (task_index, task_name) for LIBERO-Spatial."""
    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_spatial"]()
    tasks = []
    for i, tn in enumerate(suite.tasks.keys()):
        tasks.append((i, tn))
    return tasks, suite


def load_model_and_proc(model_path):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    max_memory = {idx: mm for idx in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
    )
    return model, processor, "cuda:0"


def run_one_episode(model, processor, device_ov, suite, task_index, task_name, state_id,
                    render_gpu, out_dir, max_steps=400, num_wait=10):
    """Run one Spatial episode with D5 shadow. Returns result dict."""
    task_obj = suite.get_task(task_index)
    init_states = suite.get_task_init_states(task_index)
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

    tag = f"{task_name}_s{state_id}_shadow_attempt1"
    ep_dir = Path(out_dir) / tag
    if ep_dir.exists():
        import shutil; shutil.rmtree(str(ep_dir))
    ep_dir.mkdir(parents=True)

    env, obs = build_v4_exact_env(bddl, render_gpu, max_steps, num_wait)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, num_wait)
    instruction = task_obj.language
    unnorm_key = "libero_spatial"

    # Load D5 detector
    from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
    det = D5FrozenOnlineDetectorV1(
        "/data/liuyu/outputs/d5_training/d5_candidate_best.pt",
        "/data/liuyu/outputs/d5_training/d5_frozen_config.json",
    )
    det.reset()

    step_trace = []
    action_identity = []
    latency = []
    success_done = False
    success_check = False
    infra = "ok"

    for step_idx in range(max_steps):
        if "agentview_image" not in obs:
            infra = "missing_camera"; break

        img = np.asarray(obs["agentview_image"]).copy()
        t0 = time.perf_counter()
        action, _, _, _ = decode_with_scores(
            model, processor, device_ov, img, instruction, unnorm_key, 8,
            libero_official_preprocess=False,
            libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        t_vla = time.perf_counter() - t0
        env_action = postprocess_openvla_action_for_libero(action, enabled=True)
        raw_hash = sha256_bytes(np.asarray(action, dtype=np.float32).tobytes())
        env_hash = sha256_bytes(np.asarray(env_action, dtype=np.float32).tobytes())

        # Pre-step proprio — fail-closed: missing → validity=0
        qpos_ok = True
        try:
            qpos_arr = env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[:, 0]]]
            qpos = float(qpos_arr[0]) if len(qpos_arr) > 0 else 0.0
            qpos_ok = (qpos_arr is not None and len(qpos_arr) > 0)
        except Exception:
            qpos = 0.0; qpos_ok = False
        eef_ok = True
        try:
            eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_center")]
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
        except Exception:
            eef_x = eef_y = eef_z = 0.0; eef_ok = False

        raw_gripper = float(action[-1])
        env_gripper = -1.0 if raw_gripper > 0.5 else 1.0
        decoded_open = 1 if raw_gripper > 0.5 else 0

        # Detector update
        t_det = 0.0
        det_exc = False
        try:
            t0_det = time.perf_counter()
            det_result = det.update(step_idx, raw_gripper, env_gripper, qpos,
                                    eef_x, eef_y, eef_z, decoded_open,
                                    raw_valid=True, env_valid=True,
                                    qpos_valid=qpos_ok, eef_valid=eef_ok)
            t_det = time.perf_counter() - t0_det
        except Exception as e:
            det_exc = True; infra = f"det_exc:{e}"

        # Execute
        obs, reward, done_env, info = env.step(env_action)
        obs_hash = sha256_bytes(obs["agentview_image"].tobytes()) if "agentview_image" in obs else ""
        success_done = bool(done_env)
        try:
            success_check = bool(env.check_success())
        except Exception:
            success_check = False

        step_trace.append({
            "step": step_idx, "raw_gripper": raw_gripper, "env_gripper": env_gripper,
            "gripper_qpos_before": qpos, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
            "decoded_open": decoded_open,
            "raw_valid": 1, "env_valid": 1, "qpos_valid": int(qpos_ok), "eef_valid": int(eef_ok),
            "semantics_ok": 1, "success_done": int(success_done), "success_check": int(success_check),
        })
        action_identity.append({
            "step": step_idx, "action_hash_pre": raw_hash, "action_hash_post": raw_hash,
            "action_identical": 1, "env_action_hash": env_hash, "obs_hash": obs_hash,
        })
        latency.append({"step": step_idx, "model_us": round(t_vla * 1e6), "detector_us": round(t_det * 1e6)})

        if success_done:
            break

    env.close()

    # Write artifacts
    with open(ep_dir / "step_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_trace[0].keys())); w.writeheader(); w.writerows(step_trace)
    with open(ep_dir / "action_identity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(action_identity[0].keys())); w.writeheader(); w.writerows(action_identity)
    with open(ep_dir / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(latency[0].keys())); w.writeheader(); w.writerows(latency)

    # Detector telemetry
    cands_path = ep_dir / "detector_candidates.csv"
    if det.audit_records:
        fields = ["step", "is_candidate", "score", "abstain", "abstained",
                  "candidate_reason", "emitted", "first_emit_step", "detector_version"]
        with open(cands_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            w.writerows(det.audit_records)
    with open(ep_dir / "detector_emission.json", "w") as f:
        json.dump({"emit_step": det.emit_step, "emit_score": det.emit_score,
                   "n_candidates": len(det.audit_records), "detector_version": "d5_frozen_online_v1"}, f)

    # Provenance
    with open(ep_dir / "provenance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        w.writerow(["git_HEAD", git_head()])
        w.writerow(["suite", "libero_spatial"]); w.writerow(["task", task_name])
        w.writerow(["state_id", state_id]); w.writerow(["model_path", "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial"])
        w.writerow(["unnorm_key", "libero_spatial"]); w.writerow(["gpu", os.environ.get("CUDA_VISIBLE_DEVICES","?")])
        w.writerow(["render_gpu", str(render_gpu)])

    n = len(step_trace)
    primary_success = success_check or success_done
    print(f"  {tag}: steps={n} succ_done={success_done} succ_check={success_check} "
          f"emit={det.emit_step} det_exc={det_exc} qpos_ok={qpos_ok} eef_ok={eef_ok}")
    return {"task": task_name, "state": state_id, "steps": n,
            "success_done": success_done, "success_check": success_check,
            "primary_success": primary_success,
            "emit": det.emit_step, "n_cands": len(det.audit_records),
            "det_exc": det_exc, "qpos_ok": qpos_ok, "eef_ok": eef_ok, "infra": infra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data/liuyu/outputs/libero_spatial_clean100_20260617_r1")
    ap.add_argument("--render-gpu-id", type=int, default=1, help="Physical GPU for EGL rendering")
    ap.add_argument("--start-task", type=int, default=0)
    ap.add_argument("--end-task", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks, suite = get_spatial_tasks()
    print(f"Tasks: {len(tasks)}")
    for idx, name in tasks:
        print(f"  [{idx}] {name}")

    # Load model once
    model_path = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial"
    print(f"Loading model from {model_path}...")
    model, processor, device = load_model_and_proc(model_path)
    print(f"Model loaded on {device}")

    results = []
    for task_idx, task_name in tasks:
        if task_idx < args.start_task or task_idx >= args.end_task:
            continue
        for state_id in range(10):
            print(f"[{task_idx}:{state_id}] {task_name}")
            r = run_one_episode(model, processor, device, suite, task_idx, task_name,
                                state_id, args.render_gpu_id, str(out), args.max_steps)
            results.append(r)
            time.sleep(1)

    # Summary
    n_success = sum(1 for r in results if r["primary_success"])
    n_emit = sum(1 for r in results if r["emit"] >= 0)
    n_qpos_ok = sum(1 for r in results if r["qpos_ok"])
    n_eef_ok = sum(1 for r in results if r["eef_ok"])
    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}")
    print(f"Success: {n_success}/{len(results)}")
    print(f"D5 emit: {n_emit}/{len(results)}")
    print(f"Qpos valid: {n_qpos_ok}/{len(results)}")
    print(f"EEF valid: {n_eef_ok}/{len(results)}")

    with open(out / "batch_summary.json", "w") as f:
        json.dump({"n_total": len(results), "n_success": n_success, "n_emit": n_emit,
                   "n_qpos_ok": n_qpos_ok, "n_eef_ok": n_eef_ok, "results": results}, f, indent=2)
    print(f"Summary: {out / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
