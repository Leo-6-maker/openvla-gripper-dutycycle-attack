#!/usr/bin/env python3
"""Determinism localization: 5 clean walks, find first divergence layer.
Uses unified V4-aligned env factory. Tests cream_cheese state 35.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

GPU_PAIR = '4,5'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_PAIR
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUT = os.environ.get('DET_OUT', '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613')
os.makedirs(os.path.join(OUT, 'determinism'), exist_ok=True)

from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait, set_init_state,
    seed_everything, TARGET_OBJECT_GUESS_V4)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = 4

print('[%s] Determinism audit: seeding everything' % time.strftime('%H:%M:%S'), flush=True)
seed_everything(0)

print('[%s] Loading model on GPU 4,5...' % time.strftime('%H:%M:%S'), flush=True)
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count(); mm = "10000MiB"
model = AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
    device_map="auto", max_memory={idx: mm for idx in range(max(visible, 1))})
device = "cuda:0"
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"): device = v; break
        if isinstance(v, int): device = "cuda:%d" % v; break
print('[%s] Model loaded on %s' % (time.strftime('%H:%M:%S'), device), flush=True)
# Record device map for reproducibility
device_map_snapshot = {str(k): str(v) for k, v in model.hf_device_map.items()} if hasattr(model, "hf_device_map") else {}
print('Device map: %s' % json.dumps(device_map_snapshot), flush=True)

from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image
from libero.libero import benchmark, get_libero_path

model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7

def decode_action_from_token_ids(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()
TASK = 'cream_cheese'; STATE_ID = 35; MAX_STEPS = 100  # enough to reach step 83
ti = 1  # cream_cheese task index
task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

# ── Run 5 walks ──
N_WALKS = 5
walk_logs = []

for walk_id in range(N_WALKS):
    print('[%s] Walk %d/%d...' % (time.strftime('%H:%M:%S'), walk_id+1, N_WALKS), flush=True)
    env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, MAX_STEPS, num_steps_wait=10)
    obs = env.set_init_state(init_states[STATE_ID])
    env, obs = apply_dummy_wait(env, obs, 10)

    steps_log = []
    for step in range(MAX_STEPS):
        obs['agentview_image']
        img_uint8 = obs['agentview_image']
        rgb_hash = hashlib.sha256(img_uint8.tobytes()).hexdigest()[:16]

        img_pil = Image.fromarray(img_uint8)
        inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
                  for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=action_dim,
                do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
        tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
        token_hash = hashlib.sha256(tids.tobytes()).hexdigest()[:16]

        clean_action = decode_action_from_token_ids(tids)
        env_action = postprocess_openvla_action_for_libero(clean_action, TARGET_OBJECT_GUESS_V4.get(TASK, TASK))

        qpos = env.sim.data.qpos.copy() if hasattr(env, 'sim') else None
        qpos_hash = hashlib.sha256(qpos.tobytes()).hexdigest()[:16] if qpos is not None else 'N/A'
        eef = env.env.robots[0]._hand_pos.copy() if hasattr(env.env.robots[0], '_hand_pos') else None

        steps_log.append({
            'step': step,
            'rgb_hash': rgb_hash,
            'token_hash': token_hash,
            'qpos_hash': qpos_hash,
            'gripper_env': round(float(env_action[-1]), 6),
            'eef_x': round(float(eef[0]), 6) if eef is not None else None,
            'eef_y': round(float(eef[1]), 6) if eef is not None else None,
            'eef_z': round(float(eef[2]), 6) if eef is not None else None,
        })

        obs, _, _, _ = env.step(env_action)

    walk_logs.append(steps_log)
    env.close()

# ── Find first divergence ──
print('\n=== Divergence Analysis ===', flush=True)
first_div_step = MAX_STEPS
div_type = 'none'

for step in range(MAX_STEPS):
    rgb_hashes = set(walk_logs[w][step]['rgb_hash'] for w in range(N_WALKS))
    token_hashes = set(walk_logs[w][step]['token_hash'] for w in range(N_WALKS))
    qpos_hashes = set(walk_logs[w][step]['qpos_hash'] for w in range(N_WALKS))
    grippers = set(walk_logs[w][step]['gripper_env'] for w in range(N_WALKS))

    if len(rgb_hashes) > 1 or len(token_hashes) > 1 or len(qpos_hashes) > 1 or len(grippers) > 1:
        first_div_step = step
        if len(rgb_hashes) > 1: div_type = 'RGB'
        elif len(qpos_hashes) > 1: div_type = 'qpos (sim state)'
        elif len(token_hashes) > 1: div_type = 'token (model output)'
        elif len(grippers) > 1: div_type = 'gripper env action'
        break

print('First divergence at step %d: %s' % (first_div_step, div_type), flush=True)
if first_div_step == MAX_STEPS:
    print('ALL 5 WALKS IDENTICAL up to step %d!' % (MAX_STEPS-1), flush=True)

# Show step 80 specifically
print('\n=== Step 80 values ===', flush=True)
for w in range(N_WALKS):
    s = walk_logs[w][80]
    print('  Walk %d: rgb=%s token=%s qpos=%s grip=%+.1f eef_z=%.4f' % (
        w+1, s['rgb_hash'], s['token_hash'], s['qpos_hash'], s['gripper_env'], s['eef_z'] or 0), flush=True)

# ── Save ──
out_json = os.path.join(OUT, 'determinism', 'cream_cheese_s35_5walks.json')
with open(out_json, 'w') as f:
    json.dump({'first_div_step': first_div_step, 'div_type': div_type, 'walks': walk_logs}, f, indent=2, default=str)
print('\nOutput: %s' % out_json, flush=True)

# Summary
print('\n=== DETERMINISM VERDICT ===', flush=True)
if first_div_step == MAX_STEPS:
    print('DETERMINISTIC: all 5 walks identical', flush=True)
elif first_div_step >= 80:
    print('DIVERGENCE AT STEP %d: trajectory stable through critical zone' % first_div_step, flush=True)
else:
    print('EARLY DIVERGENCE AT STEP %d: trajectory unstable before critical zone' % first_div_step, flush=True)
