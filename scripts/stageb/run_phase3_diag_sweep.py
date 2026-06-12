#!/usr/bin/env python3
"""Phase 3: focused objective/budget diagnostic on best near-miss candidate.
Tests chocolate_pudding_s11 center=53 (clean_open=0, ce 48.5→0.02).
Sweep: pgd_steps×eps×margin×seed
"""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

# ── Must set CUDA_VISIBLE_DEVICES BEFORE torch import! ──
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

OUT = os.environ.get('DIAG_OUT', '/data/liuyu/outputs/stageb_v5_critical_close_overnight_20260613_0100/diag')
os.makedirs(OUT, exist_ok=True)

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = int(GPU_PAIR.split(',')[0])
K = 8

# ── Model load ──
print('[%s] Loading model on GPUs %s...' % (time.strftime('%H:%M:%S'), GPU_PAIR), flush=True)
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

TASK_OBJECT_GUESS = {
    'ketchup': 'ketchup_1', 'tomato_sauce': 'tomato_sauce_1', 'milk': 'milk_1',
    'butter': 'butter_1', 'cream_cheese': 'cream_cheese_1', 'salad_dressing': 'salad_dressing_1',
    'bbq_sauce': 'bbq_sauce_1', 'alphabet_soup': 'alphabet_soup_1',
    'orange_juice': 'orange_juice_1', 'chocolate_pudding': 'chocolate_pudding_1',
}
TASK_IDX = {
    'ketchup': 4, 'tomato_sauce': 5, 'milk': 7, 'butter': 6,
    'cream_cheese': 1, 'salad_dressing': 2, 'bbq_sauce': 3,
    'alphabet_soup': 0, 'orange_juice': 9, 'chocolate_pudding': 8,
}

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

# ── Env ──
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.attack_adapter import OpenVLAVisualAttacker, get_adv_inputs_from_attack_result

bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()

# ── Diagnostic sweep config ──
CANDIDATE = {
    'task': 'chocolate_pudding', 'state_id': 11, 'step': 53,
    'candidate_id': 'chocolate_pudding_s11_w50_56_c53_close_streak',
}
SWEEP = []
for pgd_steps in [20, 40]:
    for eps in [6, 8]:
        for margin in [5, 10, 20]:
            SWEEP.append({'pgd_steps': pgd_steps, 'eps': eps, 'margin': margin})

SEEDS = [99, 199, 299]

print('[%s] Sweep: %d configs × %d seeds = %d tests' % (
    time.strftime('%H:%M:%S'), len(SWEEP), len(SEEDS), len(SWEEP)*len(SEEDS)), flush=True)

# ── Set up env ──
task = CANDIDATE['task']
state_id = CANDIDATE['state_id']
event_step = CANDIDATE['step']

ti = TASK_IDX[task]
task_obj = task_suite.get_task(ti)
init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

def create_env():
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, robots=['Panda'],
        has_offscreen_renderer=True, render_gpu_device_id=RENDER_GPU,
        use_camera_obs=True, camera_heights=224, camera_widths=224,
        camera_depths=False, has_renderer=False, control_freq=20, controller='OSC_POSE')
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[state_id])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return env, obs

all_rows = []
_out_csv = os.path.join(OUT, 'phase3_diag_sweep.csv')
_fieldnames = None

for si, sweep_cfg in enumerate(SWEEP):
    pgd_steps = sweep_cfg['pgd_steps']
    eps_raw = sweep_cfg['eps']
    margin = sweep_cfg['margin']
    eps_norm = eps_raw / 255.0

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tag = 'pgd%d_eps%d_margin%d_seed%d' % (pgd_steps, eps_raw, margin, seed)

        # Fresh env per test
        try:
            env, obs = create_env()
        except Exception as e:
            all_rows.append({'pgd_steps': pgd_steps, 'eps': eps_raw, 'margin': margin,
                'seed': seed, 'clean_env_gripper': '', 'adv_env_gripper': '',
                'clean_open': '', 'adv_open': '', 'c2o': '',
                'ce_initial': '', 'ce_final': '', 'loss_decrease': '',
                'adv_decode_path': '', 'infra_status': 'env_error: %s' % str(e)[:60]})
            continue

        # Walk to event_step
        walk_steps = max(0, event_step - 10)
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
            env_a = postprocess_openvla_action_for_libero(clean_a, TASK_OBJECT_GUESS.get(task, task))
            obs, _, _, _ = env.step(env_a)

        # Get clean action at event step
        obs['agentview_image']
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
        clean_env_action = postprocess_openvla_action_for_libero(clean_action, TASK_OBJECT_GUESS.get(task, task))
        clean_gripper = float(clean_env_action[-1])
        clean_open = int(clean_gripper < -0.5)

        # Run TokenPGD with sweep config
        row = {'pgd_steps': pgd_steps, 'eps': eps_raw, 'margin': margin,
            'seed': seed, 'clean_env_gripper': round(clean_gripper, 6),
            'clean_open': clean_open, 'adv_env_gripper': '', 'adv_open': '', 'c2o': '',
            'ce_initial': '', 'ce_final': '', 'loss_decrease': '',
            'adv_decode_path': '', 'infra_status': 'ok'}

        try:
            attacker = OpenVLAVisualAttacker(model, processor, {
                'method': 'token_prefix_pgd',
                'epsilon': eps_norm,
                'alpha': eps_norm / pgd_steps * 2.5,
                'num_iter': pgd_steps,
                'token_label_source': 'prefix_locked_gripper_open_margin',
                'target_token_margin': margin,
                'K_trigger': K,
                'use_restart': True, 'num_restarts': 1,
                'random_start': True, 'target_return_first': False,
            }, device=device)

            attack_result = attacker.attack(
                img_uint8, instruction, clean_action, clean_action, gen_out,
                unnorm_key=unnorm_key)

            if attack_result is None:
                row['infra_status'] = 'attack_result_none'
            else:
                adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                if adv_inputs is None or adv_inputs.get("input_ids") is None:
                    row['infra_status'] = 'no_adv_inputs'
                else:
                    input_ids = adv_inputs["input_ids"].to(device)
                    pixel_values = adv_inputs["pixel_values"].to(device=device, dtype=model_dtype)
                    with torch.inference_mode():
                        adv_gen = model.generate(
                            input_ids=input_ids, pixel_values=pixel_values,
                            max_new_tokens=action_dim, do_sample=False,
                            return_dict_in_generate=True, output_scores=False)
                    adv_tids = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
                    adv_action = decode_action_from_token_ids(adv_tids)
                    adv_env_action = postprocess_openvla_action_for_libero(
                        adv_action, TASK_OBJECT_GUESS.get(task, task))
                    adv_gripper = float(adv_env_action[-1])
                    adv_open = int(adv_gripper < -0.5)
                    c2o = int((not clean_open) and adv_open)

                    row['adv_env_gripper'] = round(adv_gripper, 6)
                    row['adv_open'] = adv_open
                    row['c2o'] = c2o
                    row['adv_decode_path'] = 'token_pgd_adv_inputs_generate'
                    dbg = getattr(attack_result, 'debug', {}) or {}
                    row['ce_initial'] = round(float(dbg.get('target_ce_initial', -1) or -1), 4)
                    row['ce_final'] = round(float(dbg.get('target_ce_final', -1) or -1), 4)
                    row['loss_decrease'] = round(row['ce_initial'] - row['ce_final'], 4)
        except Exception as e:
            row['infra_status'] = 'pgd_error: %s' % str(e)[:80]

        all_rows.append(row)
        # Save incrementally
        if _fieldnames is None:
            _fieldnames = list(all_rows[0].keys())
        with open(_out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_fieldnames)
            w.writeheader(); w.writerows(all_rows)
        c2o_str = 'C2O!' if row.get('c2o') else ''
        def safe_round(v, nd=1):
            try: return round(float(v), nd)
            except: return '?'
        print('[%s] %s clean=%+.1f(%s) adv=%s(%s) ce=%s→%s %s %s' % (
            time.strftime('%H:%M:%S'), tag,
            clean_gripper, 'O' if clean_open else 'C',
            safe_round(row['adv_env_gripper'], 1),
            'O' if row.get('adv_open') else ('C' if row['adv_env_gripper'] != '' else '?'),
            row.get('ce_initial', '?'), row.get('ce_final', '?'),
            c2o_str, row['infra_status']), flush=True)

        env.close()

# ── Save final results ──
if all_rows:
    with open(_out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_fieldnames)
        w.writeheader(); w.writerows(all_rows)

# Summary
c2o_count = sum(1 for r in all_rows if r.get('c2o'))
print('\n=== Phase 3 Diagnostic Summary ===', flush=True)
print('Total: %d, C2O: %d' % (len(all_rows), c2o_count), flush=True)
if c2o_count > 0:
    print('C2O FOUND! Configs:', flush=True)
    for r in all_rows:
        if r.get('c2o'):
            print('  pgd=%s eps=%s margin=%s seed=%s' % (r['pgd_steps'], r['eps'], r['margin'], r['seed']), flush=True)
else:
    print('NO C2O FOUND in any sweep config.', flush=True)
    # Still show if any config showed progress
    clean_c = sum(1 for r in all_rows if r.get('clean_open') == 0)
    print('clean_CLOSE reproduced: %d/%d' % (clean_c, len(all_rows)), flush=True)
print('Output: %s' % out_csv, flush=True)
