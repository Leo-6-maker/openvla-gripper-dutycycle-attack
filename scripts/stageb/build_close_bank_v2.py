#!/usr/bin/env python3
"""Phase 1 v2: Walk full episodes with unified 256x256 env, find ALL CLOSE steps, freeze + screen.
Three-way parallel via BANK_OFFSET/BANK_MAX/GPU_PAIR env vars.
"""
import csv, hashlib, json, os, sys, time
from collections import Counter
import numpy as np
import torch

GPU_PAIR = os.environ.get('GPU_PAIR', '4,5')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_PAIR
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO + '/scripts')

OUT = os.environ.get('BANK_OUT', '/data/liuyu/outputs/stageb_v5_frozen_objective_day3_20260613/frozen_bank')
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = int(GPU_PAIR.split(',')[0])
MAX_EPISODE_STEPS = 280
QUICK_N = 10; STRICT_N = 30

# Tasks from clean scan (277 summaries, 10 tasks)
TASKS = [
    ('alphabet_soup', 0), ('cream_cheese', 1), ('bbq_sauce', 3),
    ('butter', 6), ('chocolate_pudding', 8), ('ketchup', 4),
]
# Select diverse state_ids per task (from smoke v2 and clean scan)
CANDIDATES = []
for task, tidx in TASKS:
    for sid in range(10, 44):  # state range from clean scan
        CANDIDATES.append({'task': task, 'state_id': sid, 'task_idx': tidx})
# Shuffle and limit
import random; random.seed(42); random.shuffle(CANDIDATES)
BANK_OFFSET = int(os.environ.get('BANK_OFFSET', '0'))
BANK_MAX = int(os.environ.get('BANK_MAX', '20'))
CANDIDATES = CANDIDATES[BANK_OFFSET:BANK_OFFSET + BANK_MAX]

print('[%s] Phase 1 v2: %d candidates (offset=%d) on GPUs %s' % (
    time.strftime('%H:%M:%S'), len(CANDIDATES), BANK_OFFSET, GPU_PAIR), flush=True)

# ── Load model ──
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
model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7
print('[%s] Model loaded' % time.strftime('%H:%M:%S'), flush=True)

from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image
from libero.libero import benchmark, get_libero_path
from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait, TARGET_OBJECT_GUESS_V4

bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

def decode_action(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

def repeated_decode(input_ids_t, pixel_values_t, N):
    opens = 0
    for _ in range(N):
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids_t, pixel_values=pixel_values_t,
                max_new_tokens=action_dim, do_sample=False, num_beams=1,
                return_dict_in_generate=True, output_scores=False)
        tids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
        action = decode_action(tids)
        env_g = -1.0 if action[-1] > 0.5 else (1.0 if action[-1] < -0.5 else 0.0)
        opens += int(env_g < -0.5)
    return opens / N

# Phase estimate
def estimate_phase(step, n_steps):
    frac = step / max(n_steps, 1)
    if frac < 0.10: return 'dummy_or_init'
    elif frac < 0.35: return 'grasp_transition'
    elif frac < 0.50: return 'early_transport'
    elif frac < 0.65: return 'transport'
    elif frac < 0.85: return 'preplace'
    else: return 'place_or_done'

bank = []
task_count = {}

for ci, c in enumerate(CANDIDATES):
    task = c['task']; state_id = c['state_id']; ti = c['task_idx']
    print('[%s] %d/%d: %s_s%d walk...' % (time.strftime('%H:%M:%S'), ci+1, len(CANDIDATES), task, state_id), flush=True)
    try:
        task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
        bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
        instruction = task_obj.language

        env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, MAX_EPISODE_STEPS, num_steps_wait=10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)

        close_frames = []
        for step in range(MAX_EPISODE_STEPS - 10):  # policy steps after dummy wait
            obs['agentview_image']; img_uint8 = obs['agentview_image']
            img_pil = Image.fromarray(img_uint8)
            inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
            inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
            tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
            action = decode_action(tids)
            env_action = postprocess_openvla_action_for_libero(action, TARGET_OBJECT_GUESS_V4.get(task, task))
            gripper = float(env_action[-1])
            is_close = int(gripper > 0.5)
            phase = estimate_phase(step + 10, MAX_EPISODE_STEPS)  # +10 for dummy steps

            if is_close and step >= 10 and phase in ('grasp_transition', 'early_transport', 'transport'):
                # Freeze frame immediately
                input_ids_cpu = inputs['input_ids'].detach().cpu().float().numpy().copy()
                pixel_values_cpu = inputs['pixel_values'].detach().cpu().float().numpy().copy()
                rgb_hash = hashlib.sha256(img_uint8.tobytes()).hexdigest()[:16]

                # Quick screen
                open_rate = repeated_decode(inputs['input_ids'], inputs['pixel_values'], QUICK_N)
                close_rate = 1 - open_rate

                frame = {
                    'task': task, 'state_id': state_id, 'step': step + 10,  # trace-aligned step
                    'phase': phase, 'gripper': gripper, 'rgb_hash': rgb_hash,
                    'quick_close_rate': close_rate, 'quick_open_rate': open_rate,
                }

                if close_rate >= 0.8:
                    prefix = '%s_s%d_step%d' % (task, state_id, step + 10)
                    img_uint8_saved = img_uint8.copy()
                    np.savez(os.path.join(OUT, 'frozen_%s.npz' % prefix),
                        img_uint8=img_uint8_saved, input_ids=input_ids_cpu,
                        pixel_values=pixel_values_cpu, instruction=instruction,
                        task=task, state_id=state_id, step=step+10)
                    frame['frozen_path'] = 'frozen_%s.npz' % prefix
                    frame['shortlisted'] = True
                    task_count[task] = task_count.get(task, 0) + 1
                    close_frames.append(frame)
                    print('  step=%d %s gripper=%.1f close=%.1f%% SHORTLISTED' % (
                        step+10, phase, gripper, 100*close_rate), flush=True)

                if len(close_frames) >= 3:  # max 3 CLOSE frames per episode
                    break

            obs, _, done, _ = env.step(env_action)
            if done: break

        env.close()
        bank.extend(close_frames)
        print('  Found %d CLOSE frames (total bank: %d)' % (len(close_frames), len(bank)), flush=True)

    except Exception as e:
        print('  ERROR: %s' % str(e)[:150], flush=True)

# ── Strict screen ──
print('\n[%s] Strict screening (x%d) for %d shortlisted...' % (
    time.strftime('%H:%M:%S'), STRICT_N, len(bank)), flush=True)
for frame in bank:
    fp = os.path.join(OUT, frame.get('frozen_path', ''))
    if not fp or not os.path.exists(fp): continue
    frozen = np.load(fp, allow_pickle=True)
    input_ids_t = torch.from_numpy(frozen['input_ids']).to(device)
    pixel_values_t = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)
    close_rate = 1 - repeated_decode(input_ids_t, pixel_values_t, STRICT_N)
    frame['strict_close_rate'] = close_rate
    if close_rate >= 0.90: frame['class'] = 'STABLE_CLOSE'
    elif close_rate >= 0.60: frame['class'] = 'NEAR_BOUNDARY_CLOSE'
    elif close_rate >= 0.20: frame['class'] = 'NUMERICALLY_UNSTABLE'
    else: frame['class'] = 'STABLE_OPEN'
    print('  %s: close=%.2f -> %s' % (os.path.basename(fp), close_rate, frame['class']), flush=True)

# ── Summary ──
stable = [f for f in bank if f.get('class') == 'STABLE_CLOSE']
near = [f for f in bank if f.get('class') == 'NEAR_BOUNDARY_CLOSE']
with open(os.path.join(OUT, 'frozen_bank_manifest_v2.json'), 'w') as f:
    json.dump({'bank': bank, 'stable_count': len(stable), 'near_count': len(near),
        'task_counts': task_count}, f, indent=2, default=str)

print('\n=== Bank v2 Summary ===' % (), flush=True)
print('Total candidates: %d' % len(CANDIDATES), flush=True)
print('CLOSE frames found: %d' % len(bank), flush=True)
print('STABLE_CLOSE: %d' % len(stable), flush=True)
print('NEAR_BOUNDARY: %d' % len(near), flush=True)
print('Tasks: %s' % sorted(task_count.keys()), flush=True)

if len(stable) >= 12 and len(set(f['task'] for f in stable)) >= 3:
    print('GATE G1 PASSED', flush=True)
else:
    print('GATE G1 FAILED', flush=True)
