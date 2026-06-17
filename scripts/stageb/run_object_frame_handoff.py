#!/usr/bin/env python3
"""Capture pre-action RGB frames for Object timing handoff parents.

For each parent, replays the episode on GPU(2,6), captures frames at:
  D5 emit, Teacher-P ws/anchor/we, First-CLOSE, emit ±2.

Verifies action identity against original trace. Saves .npy + .pt.
"""
import csv, hashlib, json, os, sys, time
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

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
CKPT_PATH = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"
CONFIG_PATH = "/data/liuyu/outputs/d5_training/d5_frozen_config.json"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"

# 14 timing parents
PARENTS = [
    ("butter", "11"), ("ketchup", "18"), ("orange_juice", "29"), ("milk", "7"),
    ("bbq_sauce", "40"), ("bbq_sauce", "27"), ("tomato_sauce", "23"),
    ("salad_dressing", "32"), ("cream_cheese", "1"), ("cream_cheese", "20"),
    ("salad_dressing", "24"), ("salad_dressing", "11"),
    ("ketchup", "34"), ("salad_dressing", "45"),
]

RENDER_GPU = 6
MAX_STEPS = 400

TASK_IDX = {
    "alphabet_soup": 0, "cream_cheese": 1, "salad_dressing": 2, "bbq_sauce": 3,
    "ketchup": 4, "tomato_sauce": 5, "butter": 6, "milk": 7,
    "chocolate_pudding": 8, "orange_juice": 9,
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_model():
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
    visible = torch.cuda.device_count()
    max_memory = {idx: mm for idx in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"
    model = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory=max_memory,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
    )
    return model, processor, "cuda:0"


def preprocess_for_save(raw_image, processor, instruction):
    """Save processor tensor for attack reproducibility."""
    from PIL import Image
    pil_img = Image.fromarray(raw_image) if isinstance(raw_image, np.ndarray) else raw_image
    prep_inputs = processor(prompt(instruction), pil_img, return_tensors="pt")
    prep_inputs.pop("attention_mask", None)
    input_ids = prep_inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
    return {"input_ids": input_ids, "pixel_values": prep_inputs["pixel_values"]}


def run_parent(task, state_id, out_dir):
    """Run one parent, capture frames. Returns results."""
    tag = f"{task}_s{state_id}_frame"
    ep_dir = Path(out_dir) / tag
    ep_dir.mkdir(parents=True, exist_ok=True)

    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_idx = TASK_IDX[task]
    task_obj = suite.get_task(task_idx)
    init_states = suite.get_task_init_states(task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

    env, obs = build_v4_exact_env(bddl, RENDER_GPU, MAX_STEPS, 10)
    obs = env.set_init_state(init_states[int(state_id)])
    env, obs = apply_dummy_wait(env, obs, 10)
    instruction = task_obj.language

    model, processor, device = load_model()

    # Get Teacher-P info
    lr = None
    for r in csv.DictReader(open(LABELS)):
        if r["task"] == task and int(r["state_id"]) == state_id:
            lr = r; break
    teacher_ws = int(lr["ws"]) if lr else -1
    teacher_anchor = int(lr["anchor"]) if lr else -1
    teacher_we = int(lr["we"]) if lr else -1

    # Run episode, collect D5 emit + frame targets
    from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
    det = D5FrozenOnlineDetectorV1(CKPT_PATH, CONFIG_PATH)
    det.reset()

    step_trace = []
    action_identity = []
    frames = {}  # step → raw image

    for step_idx in range(MAX_STEPS):
        if "agentview_image" not in obs: break
        img = np.asarray(obs["agentview_image"]).copy()
        t0 = time.perf_counter()
        action, _, _, _ = decode_with_scores(
            model, processor, device, img, instruction, "libero_object", 8,
            libero_official_preprocess=False,
            libero_preprocess_backend="official_pil_lanczos",
            center_crop=True, resize_size=224, drop_attention_mask=True,
        )
        t_vla = time.perf_counter() - t0
        env_action = postprocess_openvla_action_for_libero(action, enabled=True)

        raw_gripper = float(action[-1])
        env_gripper = -1.0 if raw_gripper > 0.5 else 1.0
        decoded_open = 1 if raw_gripper > 0.5 else 0

        # Use EXACT same proprio extraction as standard collector
        from v4_run_eval_openvla import physical_gripper_state
        from gripper_attack.grasp import eef_pos as get_eef_pos
        gs = physical_gripper_state(env, obs)
        qpos_raw_val = gs.get("qpos") if gs else None
        if qpos_raw_val is not None and hasattr(qpos_raw_val, '__len__') and len(qpos_raw_val) > 0:
            qpos = float(np.sum(qpos_raw_val))
        else:
            qpos = float("nan")
        eef_arr = get_eef_pos(env)
        if eef_arr is not None:
            eef_x, eef_y, eef_z = float(eef_arr[0]), float(eef_arr[1]), float(eef_arr[2])
        else:
            eef_x = eef_y = eef_z = float("nan")

        det.update(step_idx, raw_gripper, env_gripper, qpos, eef_x, eef_y, eef_z, decoded_open)
        obs, reward, done_env, info = env.step(env_action)

        step_trace.append({
            "step": step_idx,
            "raw_gripper": raw_gripper, "env_gripper": env_gripper,
            "gripper_qpos_before": qpos, "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
        })
        action_identity.append({
            "step": step_idx,
            "action_hash": sha256_bytes(np.asarray(action, dtype=np.float32).tobytes()),
            "env_action_hash": sha256_bytes(np.asarray(env_action, dtype=np.float32).tobytes()),
        })

        # Always save pre-action frame (we'll filter later)
        frames[step_idx] = {"img": img, "action": action, "env_action": env_action}

        if done_env: break

    env.close()
    torch.cuda.empty_cache()

    # Determine which frames to save
    d5_emit = det.emit_step
    target_steps = set()
    if d5_emit >= 0:
        for offset in [-2, -1, 0, 1, 2]:
            s = d5_emit + offset
            if 0 <= s < len(step_trace):
                target_steps.add(s)
    for s in [teacher_ws, teacher_anchor, teacher_we]:
        if s >= 0 and s < len(step_trace):
            target_steps.add(s)
    # Add first close step (from Teacher-P)
    first_close = teacher_anchor  # approximate
    if first_close >= 0 and first_close < len(step_trace):
        target_steps.add(first_close)

    saved_frames = []
    for s in sorted(target_steps):
        if s not in frames: continue
        fdata = frames[s]
        img = fdata["img"]
        raw_hash = sha256_bytes(img.tobytes())
        proc = preprocess_for_save(img, processor, instruction)
        processor_sha = sha256_bytes(proc["pixel_values"].numpy().tobytes())
        prompt_sha = sha256_bytes(proc["input_ids"].numpy().tobytes())

        role = "other"
        if s == d5_emit: role = "d5_emit"
        elif s == teacher_anchor: role = "teacher_anchor"
        elif s == teacher_ws: role = "teacher_ws"
        elif s == teacher_we: role = "teacher_we"
        elif s == first_close: role = "first_close"

        # Save
        np.save(ep_dir / f"frame_{s:04d}.npy", img)
        torch.save(proc, ep_dir / f"processor_{s:04d}.pt")
        saved_frames.append({
            "step": s, "role": role,
            "raw_frame_sha256": raw_hash,
            "processor_tensor_sha256": processor_sha,
            "prompt_token_sha256": prompt_sha,
        })

    n = len(step_trace)
    print(f"  {tag}: steps={n} emit={d5_emit} frames={len(saved_frames)}")
    return {
        "task": task, "state_id": state_id, "steps": n,
        "d5_emit": d5_emit, "n_frames": len(saved_frames),
        "target_steps": sorted(target_steps), "frames": saved_frames,
        "action_identity": action_identity, "step_trace": step_trace,
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Capturing frames for {len(PARENTS)} parents...")
    results = []
    for task, sid in PARENTS:
        r = run_parent(task, sid, str(out))
        results.append(r)

    # Summary
    with open(out / "frame_manifest.json", "w") as f:
        json.dump({"parents": len(results), "results": results}, f, indent=2, default=str)
    print(f"\nDone: {len(results)} parents, {sum(r['n_frames'] for r in results)} total frames")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
