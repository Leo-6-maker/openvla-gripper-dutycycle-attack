#!/usr/bin/env python3
"""Day 2 Phase 1: cream_cheese_s35 clean stability + matched RAND gate.
3x clean walks + 3x RAND walks, checking step80 CLOSE.
Gate G1-A: step80 CLOSE >=2/3 clean
Gate G1-B: step80 RAND C2O <=1/3
"""
import csv, json, os, sys, time
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

OUT = os.environ.get('PHASE1_OUT', '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613')
os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = int(GPU_PAIR.split(',')[0])
K = 8
EPS_RAW = 6

print('[%s] Phase 1: Loading model on GPUs %s' % (time.strftime('%H:%M:%S'), GPU_PAIR), flush=True)
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelCls

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count()
mm = "10000MiB"
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation="eager",
    device_map="auto", max_memory={idx: mm for idx in range(max(visible, 1))})
device = "cuda:0"
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"): device = v; break
        if isinstance(v, int): device = "cuda:%d" % v; break
print('[%s] Model loaded on %s' % (time.strftime('%H:%M:%S'), device), flush=True)

from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image

TASK_OBJECT_GUESS = {'cream_cheese': 'cream_cheese_1'}
TASK_IDX = {'cream_cheese': 1}

model_dtype = torch.bfloat16
unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key))
assert action_dim == 7

def decode_action_from_token_ids(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.attack_adapter import OpenVLAVisualAttacker

bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()

TASK = 'cream_cheese'
STATE_ID = 35
WINDOW = (77, 83)
CRITICAL_STEP = 80

ti = TASK_IDX[TASK]
task_obj = task_suite.get_task(ti)
init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

def make_env():
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, robots=['Panda'],
        has_offscreen_renderer=True, render_gpu_device_id=RENDER_GPU,
        use_camera_obs=True, camera_heights=224, camera_widths=224,
        camera_depths=False, has_renderer=False, control_freq=20, controller='OSC_POSE')
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[STATE_ID])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return env, obs

def walk_to_step(env, obs, target_step):
    walk_steps = max(0, target_step - 10)
    for s in range(walk_steps):
        img_uint8 = obs['agentview_image']
        img_pil = Image.fromarray(img_uint8)
        inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
                  for k, v in inputs.items()}
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=action_dim,
                do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
        tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
        clean_a = decode_action_from_token_ids(tids)
        env_a = postprocess_openvla_action_for_libero(clean_a, TASK_OBJECT_GUESS.get(TASK, TASK))
        obs, _, _, _ = env.step(env_a)
    return env, obs

results = []

# ── 3 clean walks ──
print('[%s] Running 3 clean walks...' % time.strftime('%H:%M:%S'), flush=True)
for trial in range(3):
    env, obs = make_env()
    env, obs = walk_to_step(env, obs, CRITICAL_STEP)

    # Get action at critical step
    img_uint8 = obs['agentview_image']
    img_pil = Image.fromarray(img_uint8)
    inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
              for k, v in inputs.items()}
    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=action_dim,
            do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
    clean_action = decode_action_from_token_ids(tids)
    clean_env_action = postprocess_openvla_action_for_libero(clean_action, TASK_OBJECT_GUESS.get(TASK, TASK))
    clean_gripper = float(clean_env_action[-1])
    clean_open = int(clean_gripper < -0.5)
    clean_close = int(clean_gripper > 0.5)

    eef = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
    obj_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None
    obj_pos = env.sim.data.get_site_xpos(obj_id) if obj_id is not None else None
    qpos = env.env.robots[0]._joint_positions if hasattr(env.env.robots[0], '_joint_positions') else None

    results.append({
        'trial': trial, 'condition': 'clean', 'seed': 'N/A',
        'step': CRITICAL_STEP, 'gripper_raw': round(float(clean_action[-1]), 6),
        'gripper_env': round(clean_gripper, 6), 'is_open': clean_open, 'is_close': clean_close,
        'eef_x': round(float(eef[0]), 6) if eef is not None else '',
        'eef_y': round(float(eef[1]), 6) if eef is not None else '',
        'eef_z': round(float(eef[2]), 6) if eef is not None else '',
        'obj_x': round(float(obj_pos[0]), 6) if obj_pos is not None else '',
        'obj_y': round(float(obj_pos[1]), 6) if obj_pos is not None else '',
        'obj_z': round(float(obj_pos[2]), 6) if obj_pos is not None else '',
        'qpos_sum': round(float(np.sum(qpos)), 6) if qpos is not None else '',
    })
    print('  Clean trial %d: gripper=%+.1f (%s)' % (trial+1, clean_gripper, 'CLOSE' if clean_close else 'OPEN'), flush=True)
    env.close()

# ── 3 RAND walks ──
print('[%s] Running 3 RAND walks (seeds 99,100,101)...' % time.strftime('%H:%M:%S'), flush=True)
eps_norm = EPS_RAW / 255.0
for seed in [99, 100, 101]:
    torch.manual_seed(seed); np.random.seed(seed)
    env, obs = make_env()
    env, obs = walk_to_step(env, obs, WINDOW[0])  # walk to window start

    # Walk through window with RAND
    for step in range(WINDOW[0], WINDOW[1] + 1):
        img_uint8 = obs['agentview_image']
        img_pil = Image.fromarray(img_uint8)
        inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
                  for k, v in inputs.items()}
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=action_dim,
                do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
        tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
        clean_a = decode_action_from_token_ids(tids)
        clean_env_a = postprocess_openvla_action_for_libero(clean_a, TASK_OBJECT_GUESS.get(TASK, TASK))

        # Apply RAND perturbation
        rand_a = clean_a.copy()
        rand_a[-1] += np.random.uniform(-eps_norm, eps_norm) * 2.0  # scale to cover [-1,1]
        rand_env_a = rand_a.copy()
        rand_env_a[-1] = -1.0 if rand_a[-1] > 0.5 else (1.0 if rand_a[-1] < -0.5 else rand_env_a[-1])

        if step == CRITICAL_STEP:
            gripper_raw = float(rand_a[-1])
            gripper_env = float(rand_env_a[-1])
            is_open = int(gripper_env < -0.5)
            is_close = int(gripper_env > 0.5)
            results.append({
                'trial': seed, 'condition': 'random_linf', 'seed': seed,
                'step': CRITICAL_STEP, 'gripper_raw': round(gripper_raw, 6),
                'gripper_env': round(gripper_env, 6), 'is_open': is_open, 'is_close': is_close,
            })
            print('  RAND seed=%d: gripper=%+.1f (%s)' % (seed, gripper_env, 'CLOSE' if is_close else 'OPEN'), flush=True)

        obs, _, _, _ = env.step(rand_env_a)
    env.close()

# ── Gates ──
clean_close_count = sum(1 for r in results if r['condition'] == 'clean' and r['is_close'])
rand_open_count = sum(1 for r in results if r['condition'] == 'random_linf' and r['is_open'])

print('\n=== GATE G1-A: Clean stability ===', flush=True)
print('step80 CLOSE: %d/3' % clean_close_count, flush=True)
g1a = clean_close_count >= 2
print('G1-A: %s' % ('PASSED' if g1a else 'FAILED - TRAJECTORY_UNSTABLE_PARENT'), flush=True)

print('\n=== GATE G1-B: RAND gate ===', flush=True)
print('RAND step80 OPEN: %d/3' % rand_open_count, flush=True)
g1b = rand_open_count <= 1
print('G1-B: %s' % ('PASSED' if g1b else 'FAILED - RANDOM_SENSITIVE_ABSTAIN'), flush=True)

# ── Save ──
out_csv = os.path.join(OUT, 'tables', 's20d_v5_day2_parent_provenance.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print('\nOutput: %s' % out_csv, flush=True)
print('Overall: G1-A=%s G1-B=%s' % (g1a, g1b), flush=True)
