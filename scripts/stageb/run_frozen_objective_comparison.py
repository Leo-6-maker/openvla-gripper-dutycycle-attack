#!/usr/bin/env python3
"""Frozen-input O0/O1/O2 objective comparison on a clean CLOSE observation.
O0: prefix_locked_gripper_open_margin (logsumexp region, margin=5)
O1: prefix_locked_gripper_top1_open_vs_close (max open vs max close, margin={0,0.5,1})
O2: prefix_locked_gripper_open_region_ce (region CE)
Runs: walk to first CLOSE state, freeze observation, test each objective×seed.
Reports: autoregressive C2O rate (NOT final loss).
"""
import csv, hashlib, json, os, sys, time
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

OUT = os.environ.get('OBJ_OUT', '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613')
os.makedirs(os.path.join(OUT, 'objective_sweep'), exist_ok=True)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = 4; K = 8; EPS_RAW = 6; PGD_STEPS = 20
SEEDS = [99, 100, 101]
WALK_MAX = 120  # steps to search for CLOSE

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
from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait, set_init_state,
    seed_everything, TARGET_OBJECT_GUESS_V4)

model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7
eps_norm = EPS_RAW / 255.0

def decode_action_from_token_ids(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

def run_tests(task, state_id, task_idx, obj_configs):
    """Walk to first CLOSE step, freeze observation, test all objectives."""
    ti = task_idx; task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    instruction = task_obj.language

    # Walk to find first CLOSE step
    seed_everything(0)
    env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, WALK_MAX, num_steps_wait=10)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)

    close_step = None; frozen_data = None
    for step in range(WALK_MAX):
        obs['agentview_image']; img_uint8 = obs['agentview_image']
        img_pil = Image.fromarray(img_uint8)
        inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
        inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
        tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
        clean_a = decode_action_from_token_ids(tids)
        env_a = postprocess_openvla_action_for_libero(clean_a, TARGET_OBJECT_GUESS_V4.get(task, task))

        if step >= 10 and float(env_a[-1]) > 0.5 and close_step is None:
            close_step = step
            frozen_data = {
                'img_uint8': img_uint8.copy(), 'instruction': instruction,
                'clean_action': clean_a.tolist(), 'clean_gripper': float(env_a[-1]),
                'gen_out': gen_out, 'step': step,
            }
            print('[%s] %s_s%d: CLOSE at step %d, freezing' % (time.strftime('%H:%M:%S'), task, state_id, step), flush=True)
            break
        obs, _, _, _ = env.step(env_a)
    env.close()

    if frozen_data is None:
        print('[%s] %s_s%d: NO CLOSE found in %d steps' % (time.strftime('%H:%M:%S'), task, state_id, WALK_MAX), flush=True)
        return []

    # Run all objectives on frozen observation
    results = []
    img_uint8 = frozen_data['img_uint8']
    clean_action = np.array(frozen_data['clean_action'])
    gen_out = frozen_data['gen_out']

    for obj_cfg in obj_configs:
        obj_name = obj_cfg['name']; margin = obj_cfg.get('margin', 5)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            try:
                attacker = OpenVLAVisualAttacker(model, processor, {
                    'method': 'token_prefix_pgd',
                    'epsilon': eps_norm, 'alpha': eps_norm / PGD_STEPS * 2.5,
                    'num_iter': PGD_STEPS,
                    'token_label_source': obj_cfg['token_label_source'],
                    'target_token_margin': margin,
                    'K_trigger': K,
                    'use_restart': True, 'num_restarts': 1,
                    'random_start': True, 'target_return_first': False,
                }, device=device)

                attack_result = attacker.attack(img_uint8, instruction, clean_action, clean_action, gen_out, unnorm_key=unnorm_key)

                row = {'task': task, 'state_id': state_id, 'close_step': close_step,
                       'objective': obj_name, 'margin': margin, 'seed': seed,
                       'clean_gripper': frozen_data['clean_gripper'], 'adv_gripper': '', 'adv_open': '', 'c2o': '',
                       'ce_initial': '', 'ce_final': '', 'infra': 'ok'}

                if attack_result is None:
                    row['infra'] = 'attack_result_none'
                else:
                    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                    if adv_inputs is None or adv_inputs.get("input_ids") is None:
                        row['infra'] = 'no_adv_inputs'
                    else:
                        input_ids = adv_inputs["input_ids"].to(device)
                        pixel_values = adv_inputs["pixel_values"].to(device=device, dtype=model_dtype)
                        with torch.inference_mode():
                            adv_gen = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                                max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=False)
                        adv_tids = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
                        adv_action = decode_action_from_token_ids(adv_tids)
                        adv_env_action = postprocess_openvla_action_for_libero(adv_action, TARGET_OBJECT_GUESS_V4.get(task, task))
                        adv_g = float(adv_env_action[-1])
                        adv_open = int(adv_g < -0.5)
                        c2o = int(frozen_data['clean_gripper'] > 0.5 and adv_open)
                        row['adv_gripper'] = round(float(adv_action[-1]), 6)
                        row['adv_open'] = adv_open; row['c2o'] = c2o
                        dbg = getattr(attack_result, 'debug', {}) or {}
                        row['ce_initial'] = round(float(dbg.get('target_ce_initial', -1) or -1), 4)
                        row['ce_final'] = round(float(dbg.get('target_ce_final', -1) or -1), 4)
                results.append(row)
                c2o_str = 'C2O!' if row.get('c2o') else ''
                print('  %s margin=%s seed=%d: clean=%+.1f adv=%+.1f c2o=%s ce=%.1f->%.1f %s' % (
                    obj_name, margin, seed, frozen_data['clean_gripper'],
                    round(float(row.get('adv_gripper', 0) or 0), 1), c2o_str,
                    row.get('ce_initial', 0) or 0, row.get('ce_final', 0) or 0, row['infra']), flush=True)
            except Exception as e:
                results.append({'task': task, 'state_id': state_id, 'close_step': close_step,
                    'objective': obj_name, 'margin': margin, 'seed': seed,
                    'clean_gripper': frozen_data['clean_gripper'], 'adv_gripper': '', 'adv_open': '', 'c2o': '',
                    'ce_initial': '', 'ce_final': '', 'infra': str(e)[:80]})
    return results

# ── Objective configs ──
OBJECTIVES = [
    {'name': 'O0_logsumexp_margin5', 'token_label_source': 'prefix_locked_gripper_open_margin', 'margin': 5},
    {'name': 'O1_top1_margin0', 'token_label_source': 'prefix_locked_gripper_top1_open_vs_close', 'margin': 0},
    {'name': 'O1_top1_margin0.5', 'token_label_source': 'prefix_locked_gripper_top1_open_vs_close', 'margin': 0.5},
    {'name': 'O1_top1_margin1', 'token_label_source': 'prefix_locked_gripper_top1_open_vs_close', 'margin': 1},
    {'name': 'O2_region_ce', 'token_label_source': 'prefix_locked_gripper_open_region_ce', 'margin': 0},
]

# ── Test candidates: cream_cheese + chocolate_pudding (smoke v2 clean_CLOSE) ──
CANDIDATES = [
    {'task': 'cream_cheese', 'state_id': 35, 'task_idx': 1},
    {'task': 'chocolate_pudding', 'state_id': 21, 'task_idx': 8},
    {'task': 'chocolate_pudding', 'state_id': 11, 'task_idx': 8},
    {'task': 'cream_cheese', 'state_id': 21, 'task_idx': 1},
]

all_results = []
for c in CANDIDATES:
    results = run_tests(c['task'], c['state_id'], c['task_idx'], OBJECTIVES)
    all_results.extend(results)
    if len(all_results) >= 60:  # cap at ~60 tests
        break

# ── Summary ──
out_csv = os.path.join(OUT, 'objective_sweep', 'frozen_objective_comparison.csv')
if all_results:
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

print('\n=== Objective Comparison Summary ===', flush=True)
for obj_name in sorted(set(r['objective'] for r in all_results)):
    obj_results = [r for r in all_results if r['objective'] == obj_name]
    c2o = sum(1 for r in obj_results if r.get('c2o'))
    total = len(obj_results)
    print('  %s: C2O %d/%d (%.0f%%)' % (obj_name, c2o, total, 100*c2o/max(total,1)), flush=True)
print('Output: %s' % out_csv, flush=True)
