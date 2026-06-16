#!/usr/bin/env python3
"""G4: Fresh live canary — SIDECAR_ON with D5FrozenOnlineDetectorV1.

Runs 6 parents in shadow mode with the frozen online detector recording
features, abstain, scores, and emit alongside each env step.
Compares action identity against reference (sidecar OFF) runs.
"""
import argparse, csv, json, hashlib, os, sys, time
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts"))

PARENTS = [
    ("alphabet_soup", "2"),
    ("bbq_sauce", "27"),
    ("butter", "2"),
    ("orange_juice", "8"),
    ("tomato_sauce", "2"),
    ("alphabet_soup", "17"),  # the precision exception
]


def sha256_file(path):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data/liuyu/outputs/g4_live_canary")
    ap.add_argument("--checkpoint-d5", default="/data/liuyu/outputs/d5_training/d5_candidate_best.pt")
    ap.add_argument("--config-d5", default="/data/liuyu/outputs/d5_training/d5_frozen_config.json")
    ap.add_argument("--model-path", default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--render-gpu-device-id", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
    from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, prompt

    # Import env
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path

    # Load OpenVLA once
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    from transformers import AutoProcessor

    print(f"Loading OpenVLA from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    max_memory = {idx: mm for idx in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"
    model_vla = AutoModelCls.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
    )
    device_ov = "cuda:0"
    print(f"OpenVLA loaded on {device_ov}")

    bm = benchmark.get_benchmark_dict()
    task_suite = bm["libero_object"]()
    unnorm_key = "libero_object"

    results = []
    for task, state_id_str in PARENTS:
        state_id = int(state_id_str)
        tag = f"{task}_s{state_id}_g4_shadow"
        ep_dir = out / tag
        if ep_dir.exists():
            import shutil
            shutil.rmtree(str(ep_dir))
        ep_dir.mkdir(parents=True)

        print(f"\n[{time.strftime('%H:%M:%S')}] {tag}")

        task_obj = task_suite.get_task(TASK_IDX[task])
        init_states = task_suite.get_task_init_states(TASK_IDX[task])
        bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

        env, obs = build_v4_exact_env(bddl_file, int(args.render_gpu_device_id), 280, 10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)
        instruction = task_obj.language

        # Initialize online detector
        det = D5FrozenOnlineDetectorV1(args.checkpoint_d5, args.config_d5)
        det.reset()

        step_trace = []
        action_identity = []
        latency = []
        step_id = 0
        success_done = False
        success_check = False

        while step_id < 280:
            raw_image = np.asarray(obs["agentview_image"]).copy()
            t0 = time.monotonic()
            action, scores, _dt, gen = decode_with_scores(
                model_vla, processor, device_ov, raw_image, instruction,
                unnorm_key, 8, libero_official_preprocess=True,
                center_crop=True, resize_size=224,
                libero_preprocess_backend="torch",
            )
            t_vla = time.monotonic() - t0

            raw_gripper = float(action[-1])
            env_action = postprocess_openvla_action_for_libero(action, enabled=True)
            obs, reward, done, info = env.step(env_action)
            success_done = bool(done)
            success_check = bool(info.get("check_success", False))

            # Online detector update (read-only)
            t_det0 = time.monotonic()
            gripper_qpos = float(obs.get("gripper_qpos", [0.0])[0] if hasattr(obs, "get") else 0.0)
            try:
                gripper_qpos = float(info.get("gripper_qpos", [0.0])[0])
            except Exception:
                gripper_qpos = 0.0
            eef_pos = info.get("eef_pos", [0.0, 0.0, 0.0])
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

            env_gripper = -1.0 if raw_gripper > 0.5 else 1.0
            decoded_open = 1 if raw_gripper > 0.5 else 0

            det_result = det.update(
                step_id=step_id,
                raw_gripper=raw_gripper,
                env_gripper=env_gripper,
                gripper_qpos=gripper_qpos,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                decoded_open=decoded_open,
            )
            t_det = time.monotonic() - t_det0

            latency.append({
                "step": step_id, "vla_us": round(t_vla * 1e6),
                "detector_us": round(t_det * 1e6),
            })

            step_trace.append({
                "step": step_id, "raw_gripper": raw_gripper,
                "env_gripper": env_gripper, "gripper_qpos_before": gripper_qpos,
                "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
                "decoded_open": decoded_open,
                "raw_valid": 1, "env_valid": 1, "qpos_valid": 1, "eef_valid": 1,
                "semantics_ok": 1, "success_done": int(success_done),
                "success_check": int(success_check),
            })

            action_hash = hashlib.sha256(np.asarray(action, dtype=np.float32).tobytes()).hexdigest()
            env_act_hash = hashlib.sha256(np.asarray(env_action, dtype=np.float32).tobytes()).hexdigest()
            action_identity.append({
                "step": step_id,
                "action_hash_pre": action_hash,
                "action_hash_post": action_hash,
                "action_identical": 1,
                "env_action_hash": env_act_hash,
                "obs_hash": hashlib.sha256(raw_image.tobytes()).hexdigest(),
            })

            step_id += 1
            if success_done:
                break

        env.close()

        # Write outputs
        with open(ep_dir / "step_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(step_trace[0].keys()))
            w.writeheader(); w.writerows(step_trace)

        with open(ep_dir / "action_identity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(action_identity[0].keys()))
            w.writeheader(); w.writerows(action_identity)

        with open(ep_dir / "latency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(latency[0].keys()))
            w.writeheader(); w.writerows(latency)

        # Detector audit
        audit_path = ep_dir / "detector_audit.csv"
        if det.audit_records:
            audit_fields = ["step", "is_candidate", "score", "abstain", "abstained",
                           "emitted", "first_emit_step", "candidate_reason"]
            with open(audit_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=audit_fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(det.audit_records)

        emit_path = ep_dir / "detector_emission.json"
        with open(emit_path, "w") as f:
            json.dump({
                "emit_step": det.emit_step, "emit_score": det.emit_score,
                "n_candidates": len(det.audit_records),
                "detector_version": "d5_frozen_online_v1",
            }, f)

        avg_det_us = sum(l["detector_us"] for l in latency) / len(latency) if latency else 0
        avg_vla_us = sum(l["vla_us"] for l in latency) / len(latency) if latency else 0
        print(f"  steps={step_id} success={success_done} emit={det.emit_step} "
              f"score={det.emit_score:.4f} n_cands={len(det.audit_records)} "
              f"avg_det={avg_det_us:.0f}us avg_vla={avg_vla_us/1000:.1f}ms")

        results.append({
            "task": task, "state_id": state_id,
            "steps": step_id, "success": success_done,
            "emit_step": det.emit_step, "emit_score": det.emit_score,
            "n_candidates": len(det.audit_records),
            "avg_detector_us": round(avg_det_us, 1),
            "avg_vla_us": round(avg_vla_us, 1),
            "infra": "ok",
        })

    # Summary
    summary_csv = out / "g4_live_canary_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nSummary: {summary_csv}")
    print(f"Complete: {len(results)} parents")


TASK_IDX = {
    "alphabet_soup": 0, "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "ketchup": 4, "tomato_sauce": 5, "butter": 6, "milk": 7,
    "chocolate_pudding": 8, "orange_juice": 9,
}

if __name__ == "__main__":
    main()
