#!/usr/bin/env python3
"""Phase 1: Build Stable-CLOSE frozen event bank.
Walks to candidate frames from smoke v2 results, freezes tensors, runs repeated decode screening.
Quick screen: ×10 → shortlist if clean_CLOSE >= 0.8
Strict screen: ×30 → classify Stable CLOSE (>=0.90), Near-boundary (0.60-0.90), Unstable (<0.60)
"""
import csv, hashlib, json, os, sys, time
from collections import Counter
import numpy as np
import torch

GPU_PAIR = '4,5'
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
RENDER_GPU = 4; MAX_CANDIDATES = 60
QUICK_N = 10; STRICT_N = 30

print('[%s] Phase 1: Building Stable-CLOSE frozen bank' % time.strftime('%H:%M:%S'), flush=True)

# ── Load model once ──
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
TASK_IDX = {'alphabet_soup':0,'cream_cheese':1,'salad_dressing':2,'bbq_sauce':3,'ketchup':4,'tomato_sauce':5,'butter':6,'milk':7,'chocolate_pudding':8,'orange_juice':9}

def decode_action(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

def decode_repeated(input_ids_t, pixel_values_t, N):
    """Repeated greedy decode, return OPEN rate."""
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

# ── Get candidate frames from smoke v2 results ──
smoke_csv = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/phase2_smoke_gpu45_output.csv'
candidates = []
if os.path.exists(smoke_csv):
    for r in csv.DictReader(open(smoke_csv)):
        cid = r['candidate_id']
        task = r['task']; state_id = int(r['state_id'])
        ws = int(r['window_start']); we = int(r['window_end'])
        # Parse center step from candidate_id: ..._c<N>_close_streak
        import re
        m = re.search(r'_c(\d+)_close', cid)
        center = int(m.group(1)) if m else ws
        candidates.append({'task': task, 'state_id': state_id, 'step': center, 'candidate_id': cid})

# Add cream_cheese_s35 step80 (the C2O candidate)
candidates.insert(0, {'task': 'cream_cheese', 'state_id': 35, 'step': 80, 'candidate_id': 'cream_cheese_s35_c80_c2o'})
# Add chocolate_pudding_s21 step44
candidates.insert(1, {'task': 'chocolate_pudding', 'state_id': 21, 'step': 44, 'candidate_id': 'chocolate_pudding_s21_c44_c2o'})

# Deduplicate and limit
seen = set(); unique = []
for c in candidates:
    key = (c['task'], c['state_id'], c['step'])
    if key not in seen: seen.add(key); unique.append(c)
candidates = unique[:MAX_CANDIDATES]
print('[%s] %d unique candidates' % (time.strftime('%H:%M:%S'), len(candidates)), flush=True)

# ── Walk to each candidate, freeze, quick screen ──
bank = []
task_counts = {}
for ci, c in enumerate(candidates):
    task = c['task']; state_id = c['state_id']; target_step = c['step']
    if task not in TASK_IDX: continue
    ti = TASK_IDX[task]

    print('[%s] %d/%d: %s_s%d step=%d...' % (time.strftime('%H:%M:%S'), ci+1, len(candidates), task, state_id, target_step), flush=True)
    try:
        task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
        bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
        instruction = task_obj.language

        env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, target_step + 20, num_steps_wait=10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)

        # Walk to target
        for step in range(target_step - 10):  # -10 for dummy wait already done
            obs['agentview_image']; img_uint8 = obs['agentview_image']
            img_pil = Image.fromarray(img_uint8)
            inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
            inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
            tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
            clean_a = decode_action(tids)
            env_a = postprocess_openvla_action_for_libero(clean_a, TARGET_OBJECT_GUESS_V4.get(task, task))
            obs, _, _, _ = env.step(env_a)

        # Freeze tensors at target step
        obs['agentview_image']; img_uint8 = obs['agentview_image'].copy()
        img_pil = Image.fromarray(img_uint8)
        inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
        input_ids_t = inputs['input_ids']
        pixel_values_t = inputs['pixel_values']

        # Quick screen
        open_rate = decode_repeated(input_ids_t, pixel_values_t, QUICK_N)
        close_rate = 1 - open_rate

        frame = {'task': task, 'state_id': state_id, 'step': target_step,
            'candidate_id': c['candidate_id'],
            'quick_open_rate': open_rate, 'quick_close_rate': close_rate,
            'instruction': instruction}

        if close_rate >= 0.8:
            # Save frozen tensors
            prefix = '%s_s%d_step%d' % (task, state_id, target_step)
            input_ids_cpu = input_ids_t.detach().cpu().float().numpy()
            pixel_values_cpu = pixel_values_t.detach().cpu().float().numpy()
            np.savez(os.path.join(OUT, 'frozen_%s.npz' % prefix),
                img_uint8=img_uint8, input_ids=input_ids_cpu, pixel_values=pixel_values_cpu,
                instruction=instruction, task=task, state_id=state_id, step=target_step)
            frame['frozen_path'] = 'frozen_%s.npz' % prefix
            frame['shortlisted'] = True
            task_counts[task] = task_counts.get(task, 0) + 1
            print('  SHORTLISTED: close=%.1f%% → frozen_%s.npz' % (100*close_rate, prefix), flush=True)
        else:
            frame['shortlisted'] = False
            print('  SKIP: close=%.1f%% (<80%%)' % (100*close_rate), flush=True)

        bank.append(frame)
        env.close()

    except Exception as e:
        print('  ERROR: %s' % str(e)[:100], flush=True)
        bank.append({'task': task, 'state_id': state_id, 'step': target_step, 'shortlisted': False, 'error': str(e)[:200]})

# ── Strict screen: ×30 for shortlisted ──
print('\n[%s] Strict screening (×%d) for %d shortlisted frames...' % (time.strftime('%H:%M:%S'), STRICT_N, sum(1 for f in bank if f.get('shortlisted'))), flush=True)
for frame in bank:
    if not frame.get('shortlisted'): continue
    fp = os.path.join(OUT, frame['frozen_path'])
    if not os.path.exists(fp): continue
    frozen = np.load(fp, allow_pickle=True)
    input_ids_t2 = torch.from_numpy(frozen['input_ids']).to(device)
    pixel_values_t2 = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)
    open_rate = decode_repeated(input_ids_t2, pixel_values_t2, STRICT_N)
    close_rate = 1 - open_rate
    frame['strict_open_rate'] = open_rate
    frame['strict_close_rate'] = close_rate
    if close_rate >= 0.90:
        frame['class'] = 'STABLE_CLOSE'
    elif close_rate >= 0.60:
        frame['class'] = 'NEAR_BOUNDARY_CLOSE'
    elif close_rate >= 0.20:
        frame['class'] = 'NUMERICALLY_UNSTABLE'
    else:
        frame['class'] = 'STABLE_OPEN'
    print('  %s: close=%.2f → %s' % (frame['frozen_path'], close_rate, frame['class']), flush=True)

# ── Summary ──
stable = [f for f in bank if f.get('class') == 'STABLE_CLOSE']
near = [f for f in bank if f.get('class') == 'NEAR_BOUNDARY_CLOSE']
print('\n=== Bank Summary ===', flush=True)
print('Total candidates: %d' % len(bank), flush=True)
print('Shortlisted: %d' % sum(1 for f in bank if f.get('shortlisted')), flush=True)
print('STABLE_CLOSE: %d' % len(stable), flush=True)
print('NEAR_BOUNDARY: %d' % len(near), flush=True)
print('Tasks covered: %s' % sorted(set(f['task'] for f in stable)), flush=True)

# Save bank manifest
with open(os.path.join(OUT, 'frozen_bank_manifest.json'), 'w') as f:
    json.dump(bank, f, indent=2, default=str)

print('\nOutput: %s/frozen_bank_manifest.json' % OUT, flush=True)
if len(stable) >= 12 and len(set(f['task'] for f in stable)) >= 3:
    print('GATE G1 PASSED: %d Stable CLOSE, %d tasks' % (len(stable), len(set(f['task'] for f in stable))), flush=True)
else:
    print('GATE G1 FAILED: need >=12 Stable CLOSE and >=3 tasks', flush=True)
