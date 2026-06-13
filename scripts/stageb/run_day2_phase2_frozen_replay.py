#!/usr/bin/env python3
"""Day 2 Phase 2: Frozen-state C2O replay on cream_cheese_s35.
Walks with smoke-matching timing (no dummy wait correction), captures first CLOSE state near step80.
Runs VIS seed99×5, VIS seed100/101×3, RAND seed99/100/101 on FROZEN observation.
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
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUT = os.environ.get('PHASE2_OUT', '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613')
os.makedirs(os.path.join(OUT, 'frozen_events'), exist_ok=True)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = 4; K = 8; EPS_RAW = 6; PGD_STEPS = 20

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

from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.attack_adapter import OpenVLAVisualAttacker, get_adv_inputs_from_attack_result

TASK_OBJECT_GUESS = {'cream_cheese': 'cream_cheese_1'}
TASK_IDX = {'cream_cheese': 1}

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
TASK = 'cream_cheese'; STATE_ID = 35; TARGET_STEP = 80

ti = TASK_IDX[TASK]; task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language
eps_norm = EPS_RAW / 255.0

# ── Step 1: Walk to target step (NO dummy wait correction → match smoke) ──
print('[%s] Walking to step %d (smoke-matched timing)...' % (time.strftime('%H:%M:%S'), TARGET_STEP), flush=True)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, robots=['Panda'],
    has_offscreen_renderer=True, render_gpu_device_id=RENDER_GPU,
    use_camera_obs=True, camera_heights=224, camera_widths=224,
    camera_depths=False, has_renderer=False, control_freq=20, controller='OSC_POSE')
env.seed(0); obs = env.reset(); obs = env.set_init_state(init_states[STATE_ID])

for s in range(TARGET_STEP):
    img_uint8 = obs['agentview_image']; img_pil = Image.fromarray(img_uint8)
    inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k, v in inputs.items()}
    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
    clean_a = decode_action_from_token_ids(tids)
    env_a = postprocess_openvla_action_for_libero(clean_a, TASK_OBJECT_GUESS.get(TASK, TASK))
    obs, _, _, _ = env.step(env_a)

# Get clean action at target step
obs['agentview_image']; img_uint8 = obs['agentview_image']; img_pil = Image.fromarray(img_uint8)
inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k, v in inputs.items()}
with torch.no_grad():
    gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
clean_action = decode_action_from_token_ids(tids)
clean_env_action = postprocess_openvla_action_for_libero(clean_action, TASK_OBJECT_GUESS.get(TASK, TASK))
clean_gripper = float(clean_env_action[-1])
clean_open = int(clean_gripper < -0.5); clean_close = int(clean_gripper > 0.5)
print('[%s] Clean at step %d: gripper=%+.1f (%s)' % (time.strftime('%H:%M:%S'), TARGET_STEP, clean_gripper, 'CLOSE' if clean_close else 'OPEN'), flush=True)

# Save frozen observation
frozen = {
    'img_uint8': img_uint8.copy(), 'instruction': instruction,
    'clean_action': clean_action.tolist(), 'clean_env_action': clean_env_action.tolist(),
    'clean_gripper': clean_gripper, 'clean_open': clean_open, 'clean_close': clean_close,
    'gen_out_sequences': gen_out.sequences[0].detach().cpu().numpy().tolist(),
}
np.savez(os.path.join(OUT, 'frozen_events', 'cream_cheese_s35_step80_frozen.npz'), **{k: v for k,v in frozen.items() if k != 'img_uint8'})
np.save(os.path.join(OUT, 'frozen_events', 'cream_cheese_s35_step80_rgb.npy'), img_uint8)
print('[%s] Frozen event saved' % time.strftime('%H:%M:%S'), flush=True)

# ── Step 2: Frozen replay ──
results = []
attacker_config_template = {
    'method': 'token_prefix_pgd', 'epsilon': eps_norm,
    'alpha': eps_norm / PGD_STEPS * 2.5, 'num_iter': PGD_STEPS,
    'token_label_source': 'prefix_locked_gripper_open_margin',
    'target_token_margin': 5, 'K_trigger': K,
    'use_restart': True, 'num_restarts': 1,
    'random_start': True, 'target_return_first': False,
}

for cond_name, seeds, is_rand in [('VIS_seed99', [99]*5, False), ('VIS_seed100', [100]*3, False), ('VIS_seed101', [101]*3, False),
                                    ('RAND_seed99', [99], True), ('RAND_seed100', [100], True), ('RAND_seed101', [101], True)]:
    for rep, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        try:
            if is_rand:
                # RAND: add uniform noise to clean action
                rand_a = np.array(frozen['clean_action']).copy()
                rand_a[-1] += np.random.uniform(-eps_norm, eps_norm) * 2.0
                adv_env_g = -1.0 if rand_a[-1] > 0.5 else (1.0 if rand_a[-1] < -0.5 else 0.0)
                adv_open = int(adv_env_g < -0.5)
                results.append({'condition': cond_name, 'rep': rep, 'seed': seed,
                    'clean_gripper': clean_gripper, 'clean_open': clean_open,
                    'adv_gripper': round(float(rand_a[-1]), 6), 'adv_env_gripper': round(float(adv_env_g), 6),
                    'adv_open': adv_open, 'c2o': int(clean_close and adv_open),
                    'interface': 'RAND', 'ce_drop': '', 'infra': 'ok'})
            else:
                attacker = OpenVLAVisualAttacker(model, processor, attacker_config_template, device=device)
                attack_result = attacker.attack(img_uint8, instruction,
                    np.array(frozen['clean_action']), np.array(frozen['clean_action']), gen_out, unnorm_key=unnorm_key)

                if attack_result is None:
                    results.append({'condition': cond_name, 'rep': rep, 'seed': seed,
                        'clean_gripper': clean_gripper, 'clean_open': clean_open,
                        'adv_gripper': '', 'adv_env_gripper': '', 'adv_open': '', 'c2o': '',
                        'interface': 'FAIL', 'ce_drop': '', 'infra': 'attack_result_none'})
                    continue

                adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                if adv_inputs is None or adv_inputs.get("input_ids") is None:
                    results.append({'condition': cond_name, 'rep': rep, 'seed': seed,
                        'clean_gripper': clean_gripper, 'clean_open': clean_open,
                        'adv_gripper': '', 'adv_env_gripper': '', 'adv_open': '', 'c2o': '',
                        'interface': 'FAIL', 'ce_drop': '', 'infra': 'no_adv_inputs'})
                    continue

                input_ids = adv_inputs["input_ids"].to(device)
                pixel_values = adv_inputs["pixel_values"].to(device=device, dtype=model_dtype)
                with torch.inference_mode():
                    adv_gen = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                        max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=False)
                adv_tids = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
                adv_action = decode_action_from_token_ids(adv_tids)
                adv_env_action = postprocess_openvla_action_for_libero(adv_action, TASK_OBJECT_GUESS.get(TASK, TASK))
                adv_g = float(adv_env_action[-1]); adv_open = int(adv_g < -0.5)
                dbg = getattr(attack_result, 'debug', {}) or {}
                ce_i = float(dbg.get('target_ce_initial', -1) or -1)
                ce_f = float(dbg.get('target_ce_final', -1) or -1)
                results.append({'condition': cond_name, 'rep': rep, 'seed': seed,
                    'clean_gripper': clean_gripper, 'clean_open': clean_open,
                    'adv_gripper': round(float(adv_action[-1]), 6), 'adv_env_gripper': round(adv_g, 6),
                    'adv_open': adv_open, 'c2o': int(clean_close and adv_open),
                    'interface': 'PASS', 'ce_drop': round(ce_i - ce_f, 4),
                    'infra': 'ok'})
        except Exception as e:
            results.append({'condition': cond_name, 'rep': rep, 'seed': seed,
                'clean_gripper': clean_gripper, 'clean_open': clean_open,
                'adv_gripper': '', 'adv_env_gripper': '', 'adv_open': '', 'c2o': '',
                'interface': 'FAIL', 'ce_drop': '', 'infra': str(e)[:80]})

        r = results[-1]
        print('  %s rep=%d: clean=%s adv=%s c2o=%s %s' % (cond_name, rep+1,
            'CLOSE' if clean_close else 'OPEN', 'OPEN' if r.get('adv_open') else ('CLOSE' if r.get('adv_env_gripper') else '?'),
            'C2O!' if r.get('c2o') else '-', r['infra']), flush=True)

env.close()

# ── Summary ──
print('\n=== Phase 2 Frozen Replay Results ===', flush=True)
vis99 = [r for r in results if 'VIS_seed99' in r['condition']]
vis100 = [r for r in results if 'VIS_seed100' in r['condition']]
vis101 = [r for r in results if 'VIS_seed101' in r['condition']]
rand_all = [r for r in results if 'RAND' in r['condition']]

vis99_c2o = sum(1 for r in vis99 if r.get('c2o'))
vis100_c2o = sum(1 for r in vis100 if r.get('c2o'))
vis101_c2o = sum(1 for r in vis101 if r.get('c2o'))
rand_c2o = sum(1 for r in rand_all if r.get('c2o'))

print('clean=%s (CLOSE=%s)' % (clean_gripper, clean_close), flush=True)
print('VIS seed99 C2O: %d/%d' % (vis99_c2o, len(vis99)), flush=True)
print('VIS seed100 C2O: %d/%d' % (vis100_c2o, len(vis100)), flush=True)
print('VIS seed101 C2O: %d/%d' % (vis101_c2o, len(vis101)), flush=True)
print('RAND C2O: %d/%d' % (rand_c2o, len(rand_all)), flush=True)

if clean_close:
    if vis99_c2o >= 4: label = 'LOCAL_C2O_DETERMINISTIC'
    elif vis99_c2o >= 2: label = 'LOCAL_C2O_SEED_SENSITIVE'
    else: label = 'LOCAL_C2O_NOT_REPRODUCED'
    if rand_c2o >= 3: label += ' + RANDOM_SENSITIVE'
else:
    label = 'CLEAN_NOT_CLOSE_NO_C2O_TESTABLE'

print('Label: %s' % label, flush=True)

out_csv = os.path.join(OUT, 'tables', 's20d_v5_day2_frozen_event_replay.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print('Output: %s' % out_csv, flush=True)
