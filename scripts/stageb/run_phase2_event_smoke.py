#!/usr/bin/env python3
"""Phase 2: eps6 event-window smoke runner.
Reads candidate CSV, runs one-step TokenPGD per window step, records full telemetry.
Usage: python run_phase2_event_smoke.py --gpu 4 --candidates phase2_smoke_top30.csv --output_dir OUT/smoke/
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

# ── Args ──
ap = argparse.ArgumentParser()
ap.add_argument('--gpus', required=True, help='GPU pair e.g. "0,1" or "4,5"')
ap.add_argument('--candidates', required=True)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--max_candidates', type=int, default=15)
ap.add_argument('--offset', type=int, default=0)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=int, default=6)
ap.add_argument('--attack_seeds', default='99,199,299')
ap.add_argument('--model_path', default='/data/aviary/models/openvla/openvla-7b-finetuned-libero-object')
args = ap.parse_args()

gpu_pair = args.gpus.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.makedirs(args.output_dir, exist_ok=True)

MODEL_PATH = args.model_path
attack_seeds = [int(s) for s in args.attack_seeds.split(',')]

# Render GPU: use the first actual GPU index (EGL doesn't respect CUDA_VISIBLE_DEVICES remapping)
RENDER_GPU_ID = int(gpu_pair[0])

# ── Model load (v5-compatible: auto-distribute across GPU pair) ──
print('[%s] Loading model on GPUs %s...' % (time.strftime('%H:%M:%S'), args.gpus))
from transformers import AutoProcessor
from PIL import Image
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelCls

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count()
mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
max_memory = {idx: mm for idx in range(max(visible, 1))}
max_memory["cpu"] = "128GiB"
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
    device_map="auto", max_memory=max_memory)
device = "cuda:0"
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"):
            device = v; break
        if isinstance(v, int):
            device = "cuda:%d" % v; break
print('[%s] Model loaded on %s' % (time.strftime('%H:%M:%S'), device))

from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero

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
assert action_dim == 7, f'Unexpected action_dim={action_dim}'

# ── Env helpers ──
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.attack_adapter import OpenVLAVisualAttacker, get_adv_inputs_from_attack_result

bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()

def make_env(task_name, state_id, render_gpu):
    ti = TASK_IDX[task_name]
    task_obj = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        robots=['Panda'],
        has_offscreen_renderer=True,
        render_gpu_device_id=render_gpu,
        use_camera_obs=True,
        camera_heights=224, camera_widths=224,
        camera_depths=False,
        has_renderer=False,
        control_freq=20,
        controller='OSC_POSE',
    )
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[state_id])
    # V4 dummy wait
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)
    return env, obs, task_obj

def decode_action_from_token_ids(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi = np.array(stats["q99"]); lo = np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

# ── Read candidates ──
with open(args.candidates) as f:
    candidates = list(csv.DictReader(f))
print('[%s] Loaded %d candidates, running up to %d' % (
    time.strftime('%H:%M:%S'), len(candidates), args.max_candidates))
candidates = candidates[args.offset:args.offset + args.max_candidates]

# ── Results ──
all_rows = []
K_trigger = 8

for ci, cand in enumerate(candidates):
    cid = cand['candidate_id']
    task = cand['task']
    state_id = int(cand['state_id'])
    ws = int(cand['window_start'])
    we = int(cand['window_end'])
    phase = cand.get('phase', '')
    tier = cand.get('tier', '')

    print('[%s] Candidate %d/%d: %s ws=%d we=%d' % (
        time.strftime('%H:%M:%S'), ci+1, len(candidates), cid, ws, we))

    for seed in attack_seeds:
        seed_tag = '%s_s%d_w%d_%d_seed%d' % (task, state_id, ws, we, seed)
        print('  [%s] seed=%d' % (time.strftime('%H:%M:%S'), seed))

        # Fresh env per seed to avoid state pollution
        try:
            env, obs, task_obj = make_env(task, state_id, render_gpu=RENDER_GPU_ID)
        except Exception as e:
            print('    ERROR: env setup failed: %s' % str(e)[:80])
            for step in range(ws, we+1):
                all_rows.append({
                    'candidate_id': cid, 'task': task, 'state_id': state_id,
                    'attack_seed': seed, 'step': step, 'phase': phase, 'tier': tier,
                    'clean_env_gripper': '', 'adv_env_gripper': '',
                    'clean_open_bool': '', 'adv_open_bool': '',
                    'clean_close_bool': '', 'clean_close_to_adv_open': '',
                    'target_ce_initial': '', 'target_ce_final': '', 'loss_decrease': '',
                    'gripper_logit_margin_after': '',
                    'open_region_prob_mass_after': '', 'close_bin_prob_mass_after': '',
                    'pixel_budget_adv_inputs_linf': '',
                    'adv_decode_path': '', 'used_adv_inputs': '', 'fallback_adapter_used': '',
                    'infra_status': 'env_setup_failed',
                })
            continue

        instruction = task_obj.language

        # ── Walk to window_start with clean policy ──
        for step in range(ws):
            img_uint8 = obs['agentview_image']
            img_pil = Image.fromarray(img_uint8)
            inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
            inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
                      for k, v in inputs.items()}
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=action_dim,
                    do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
            token_ids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
            clean_action = decode_action_from_token_ids(token_ids)
            env_action = postprocess_openvla_action_for_libero(
                clean_action, TASK_OBJECT_GUESS.get(task, task), task)

            if step >= 10:
                obs, _, _, _ = env.step(env_action)
            else:
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

        # ── Per-step attack in window ──
        attacker = None
        for step in range(ws, we+1):
            obs['agentview_image']
            img_uint8 = obs['agentview_image']

            # Get clean action first
            inputs = processor(text=instruction, images=img_uint8, return_tensors='pt')
            inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype)
                      for k, v in inputs.items()}
            with torch.no_grad():
                gen_out = model.generate(**inputs, max_new_tokens=action_dim,
                    do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
            token_ids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
            clean_action, _ = decode_with_scores(token_ids, model, unnorm_key, use_v4_decode=True)
            clean_env_action = postprocess_openvla_action_for_libero(
                clean_action, TASK_OBJECT_GUESS.get(task, task), task)

            clean_gripper = float(clean_env_action[-1])
            clean_open = int(clean_gripper < -0.5)
            clean_close = int(clean_gripper > 0.5)

            # TokenPGD attack
            row = {
                'candidate_id': cid, 'task': task, 'state_id': state_id,
                'attack_seed': seed, 'step': step, 'phase': phase, 'tier': tier,
                'clean_env_gripper': round(clean_gripper, 6),
                'clean_open_bool': clean_open,
                'clean_close_bool': clean_close,
                'adv_env_gripper': '', 'adv_open_bool': '', 'clean_close_to_adv_open': '',
                'target_ce_initial': '', 'target_ce_final': '', 'loss_decrease': '',
                'gripper_logit_margin_after': '',
                'open_region_prob_mass_after': '', 'close_bin_prob_mass_after': '',
                'pixel_budget_adv_inputs_linf': '',
                'adv_decode_path': '', 'used_adv_inputs': False, 'fallback_adapter_used': False,
                'infra_status': 'ok',
            }

            try:
                # Set seed for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                # Init attacker per step (fresh PGD)
                eps_norm = args.eps_raw_pixels / 255.0
                attacker = OpenVLAVisualAttacker(model, processor, {
                    'method': 'token_prefix_pgd',
                    'epsilon': eps_norm,
                    'alpha': eps_norm / args.pgd_steps * 2.5,
                    'num_iter': args.pgd_steps,
                    'token_label_source': 'prefix_locked_gripper_open_margin',
                    'target_token_margin': 5,
                    'K_trigger': K_trigger,
                    'use_restart': True,
                    'num_restarts': 1,
                    'random_start': True,
                    'target_return_first': False,
                }, device=device, model_dtype=model_dtype)

                attack_result = attacker.attack(
                    img_uint8, instruction,
                    clean_action,  # target_action
                    clean_action,  # original_action
                    gen_out,
                    unnorm_key=unnorm_key)

                if attack_result is None:
                    row['infra_status'] = 'attack_result_none'
                else:
                    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                    if adv_inputs is None or len(adv_inputs) == 0:
                        row['infra_status'] = 'no_adv_inputs'
                    else:
                        # Decode from adv_inputs (v5 runner path)
                        adv_gen = model.generate(
                            inputs_embeds=adv_inputs.to(device=device, dtype=model_dtype),
                            max_new_tokens=action_dim, do_sample=False, num_beams=1,
                            return_dict_in_generate=True, output_scores=False)
                        adv_token_ids = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
                        adv_action = decode_action_from_token_ids(adv_token_ids)
                        adv_env_action = postprocess_openvla_action_for_libero(
                            adv_action, TASK_OBJECT_GUESS.get(task, task), task)

                        adv_gripper = float(adv_env_action[-1])
                        adv_open = int(adv_gripper < -0.5)
                        c2o = int(clean_close and adv_open)

                        row['adv_env_gripper'] = round(adv_gripper, 6)
                        row['adv_open_bool'] = adv_open
                        row['clean_close_to_adv_open'] = c2o
                        row['adv_decode_path'] = 'token_pgd_adv_inputs_generate'
                        row['used_adv_inputs'] = True

                        # Telemetry from debug
                        dbg = getattr(attack_result, 'debug', {}) or {}
                        row['target_ce_initial'] = round(float(dbg.get('target_ce_initial', -1) or -1), 6)
                        row['target_ce_final'] = round(float(dbg.get('target_ce_final', -1) or -1), 6)
                        row['loss_decrease'] = round(float(dbg.get('loss_decrease', 0) or 0), 6)
                        row['gripper_logit_margin_after'] = round(float(dbg.get('gripper_logit_margin_after', -1) or -1), 6)
                        row['open_region_prob_mass_after'] = round(float(dbg.get('open_region_prob_mass_after', -1) or -1), 6)
                        row['close_bin_prob_mass_after'] = round(float(dbg.get('close_bin_prob_mass_after', -1) or -1), 6)
                        row['pixel_budget_adv_inputs_linf'] = round(float(dbg.get('pixel_budget_adv_inputs_linf', -1) or -1), 8)
                        row['fallback_adapter_used'] = bool(dbg.get('fallback_adapter_used', False))

            except Exception as e:
                row['infra_status'] = 'pgd_error: %s' % str(e)[:80]

            all_rows.append(row)
            print('    step=%d clean=%+.1f(%d) adv=%+.1f(%d) c2o=%d ce_drop=%.4f infra=%s' % (
                step, clean_gripper, clean_open,
                float(row['adv_env_gripper']) if row['adv_env_gripper'] != '' else 0,
                int(row['adv_open_bool']) if row['adv_open_bool'] != '' else 0,
                int(row['clean_close_to_adv_open']) if row['clean_close_to_adv_open'] != '' else 0,
                float(row['loss_decrease']) if row['loss_decrease'] != '' else 0,
                row['infra_status']))

            # Step env with CLEAN action to advance to next step
            obs, _, _, _ = env.step(clean_env_action)

        env.close()

# ── Save ──
fieldnames = [
    'candidate_id', 'task', 'state_id', 'attack_seed', 'step', 'phase', 'tier',
    'clean_env_gripper', 'adv_env_gripper', 'clean_open_bool', 'adv_open_bool',
    'clean_close_bool', 'clean_close_to_adv_open',
    'target_ce_initial', 'target_ce_final', 'loss_decrease',
    'gripper_logit_margin_after', 'open_region_prob_mass_after',
    'close_bin_prob_mass_after', 'pixel_budget_adv_inputs_linf',
    'adv_decode_path', 'used_adv_inputs', 'fallback_adapter_used', 'infra_status',
]

out_path = os.path.join(args.output_dir, 'phase2_smoke_gpu%d.csv' % args.gpu)
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(all_rows)

# ── Summary ──
from collections import Counter
c2o_count = sum(1 for r in all_rows if int(r.get('clean_close_to_adv_open', 0) or 0) > 0)
infra_ok = sum(1 for r in all_rows if r['infra_status'] == 'ok')
infra_err = len(all_rows) - infra_ok
print('\n=== Phase 2 Smoke GPU%d Summary ===' % args.gpu)
print('Total rows: %d, c2o events: %d, infra errors: %d' % (len(all_rows), c2o_count, infra_err))
print('Decode paths: %s' % Counter(r['adv_decode_path'] for r in all_rows if r['adv_decode_path']))
print('Output: %s' % out_path)
