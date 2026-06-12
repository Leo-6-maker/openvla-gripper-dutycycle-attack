#!/usr/bin/env python3
"""Phase 2 minimal smoke: test TokenPGD at event-center steps only.
One attack per candidate per seed. Fast diagnostic before full per-step smoke.
Usage: python run_phase2_minimal_smoke.py --gpus 0,1 --candidates TOP30.csv --output_dir OUT/smoke/
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
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DISPLAY", "")

ap = argparse.ArgumentParser()
ap.add_argument('--gpus', required=True, help='GPU pair e.g. "0,1"')
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

# Render GPU uses actual GPU index (EGL not remapped by CUDA_VISIBLE_DEVICES)
render_gpu = int(gpu_pair[0])

# ── Model load ──
print('[%s] Loading model on GPUs %s...' % (time.strftime('%H:%M:%S'), args.gpus), flush=True)
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelCls

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count()
mm = "10000MiB"
max_memory = {idx: mm for idx in range(max(visible, 1))}
max_memory["cpu"] = "128GiB"
model = AutoModelCls.from_pretrained(
    args.model_path, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation="eager",
    device_map="auto", max_memory=max_memory)
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

# Decode helpers
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

def make_env(task_name, state_id):
    ti = TASK_IDX[task_name]
    task_obj = task_suite.get_task(ti)
    init_states = task_suite.get_task_init_states(ti)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, robots=['Panda'],
        has_offscreen_renderer=True, render_gpu_device_id=render_gpu,
        use_camera_obs=True, camera_heights=224, camera_widths=224,
        camera_depths=False, has_renderer=False, control_freq=20, controller='OSC_POSE')
    env.seed(0)
    obs = env.reset()
    obs = env.set_init_state(init_states[state_id])
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)
    return env, obs, task_obj

# ── Read candidates ──
with open(args.candidates) as f:
    candidates = list(csv.DictReader(f))
candidates = candidates[args.offset:args.offset + args.max_candidates]
attack_seeds = [int(s) for s in args.attack_seeds.split(',')]
print('[%s] Processing %d candidates' % (time.strftime('%H:%M:%S'), len(candidates)), flush=True)

all_rows = []
K_trigger = 8
eps_norm = args.eps_raw_pixels / 255.0

for ci, cand in enumerate(candidates):
    cid = cand['candidate_id']
    task = cand['task']
    state_id = int(cand['state_id'])
    event_center = int(cand['event_center_step'])
    phase = cand.get('phase', '')
    tier = cand.get('tier', '')
    clean_close_streak = cand.get('clean_close_streak', '?')
    clean_close_steps = cand.get('clean_close_steps', '?')

    print('[%s] %d/%d: %s center=%d' % (time.strftime('%H:%M:%S'), ci+1, len(candidates), cid, event_center), flush=True)

    for seed in attack_seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        try:
            env, obs, task_obj = make_env(task, state_id)
        except Exception as e:
            all_rows.append({'candidate_id': cid, 'task': task, 'state_id': state_id,
                'attack_seed': seed, 'step': event_center, 'phase': phase, 'tier': tier,
                'clean_close_streak': clean_close_streak, 'clean_close_steps': clean_close_steps,
                'clean_env_gripper': '', 'adv_env_gripper': '', 'clean_open_bool': '',
                'adv_open_bool': '', 'clean_close_to_adv_open': '',
                'target_ce_initial': '', 'target_ce_final': '', 'loss_decrease': '',
                'pixel_budget_adv_inputs_linf': '',
                'adv_decode_path': '', 'used_adv_inputs': False, 'fallback_adapter_used': False,
                'infra_status': 'env_setup_failed: %s' % str(e)[:60]})
            continue

        instruction = task_obj.language

        # Walk to event_center: make_env already did 10 dummy steps,
        # so walk event_center-10 policy steps to reach same state as trace step event_center
        walk_steps = max(0, event_center - 10)
        for step in range(walk_steps):
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
                clean_action, TASK_OBJECT_GUESS.get(task, task))
            obs, _, _, _ = env.step(env_action)

        # Now at event_center: get clean action, then attack
        obs['agentview_image']
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
        clean_env_action = postprocess_openvla_action_for_libero(
            clean_action, TASK_OBJECT_GUESS.get(task, task))
        clean_gripper = float(clean_env_action[-1])
        clean_open = int(clean_gripper < -0.5)
        clean_close = int(clean_gripper > 0.5)

        row = {'candidate_id': cid, 'task': task, 'state_id': state_id,
            'attack_seed': seed, 'step': event_center, 'phase': phase, 'tier': tier,
            'clean_close_streak': clean_close_streak, 'clean_close_steps': clean_close_steps,
            'clean_env_gripper': round(clean_gripper, 6),
            'clean_open_bool': clean_open,
            'adv_env_gripper': '', 'adv_open_bool': '', 'clean_close_to_adv_open': '',
            'target_ce_initial': '', 'target_ce_final': '', 'loss_decrease': '',
            'pixel_budget_adv_inputs_linf': '',
            'adv_decode_path': '', 'used_adv_inputs': False, 'fallback_adapter_used': False,
            'infra_status': 'ok'}

        try:
            attacker = OpenVLAVisualAttacker(model, processor, {
                'method': 'token_prefix_pgd',
                'epsilon': eps_norm,
                'alpha': eps_norm / args.pgd_steps * 2.5,
                'num_iter': args.pgd_steps,
                'token_label_source': 'prefix_locked_gripper_open_margin',
                'target_token_margin': 5,
                'K_trigger': K_trigger,
                'use_restart': True, 'num_restarts': 1,
                'random_start': True, 'target_return_first': False,
            }, device=device)

            attack_result = attacker.attack(
                img_uint8, instruction,
                clean_action, clean_action, gen_out,
                unnorm_key=unnorm_key)

            if attack_result is None:
                row['infra_status'] = 'attack_result_none'
            else:
                adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                if adv_inputs is None or len(adv_inputs) == 0:
                    row['infra_status'] = 'no_adv_inputs'
                else:
                    adv_gen = model.generate(
                        inputs_embeds=adv_inputs.to(device=device, dtype=model_dtype),
                        max_new_tokens=action_dim, do_sample=False, num_beams=1,
                        return_dict_in_generate=True, output_scores=False)
                    adv_token_ids = adv_gen.sequences[0, -action_dim:].detach().cpu().numpy()
                    adv_action = decode_action_from_token_ids(adv_token_ids)
                    adv_env_action = postprocess_openvla_action_for_libero(
                        adv_action, TASK_OBJECT_GUESS.get(task, task))
                    adv_gripper = float(adv_env_action[-1])
                    adv_open = int(adv_gripper < -0.5)
                    c2o = int(clean_close and adv_open)

                    row['adv_env_gripper'] = round(adv_gripper, 6)
                    row['adv_open_bool'] = adv_open
                    row['clean_close_to_adv_open'] = c2o
                    row['adv_decode_path'] = 'token_pgd_adv_inputs_generate'
                    row['used_adv_inputs'] = True

                    dbg = getattr(attack_result, 'debug', {}) or {}
                    row['target_ce_initial'] = round(float(dbg.get('target_ce_initial', -1) or -1), 6)
                    row['target_ce_final'] = round(float(dbg.get('target_ce_final', -1) or -1), 6)
                    row['loss_decrease'] = round(float(dbg.get('loss_decrease', 0) or 0), 6)
                    row['pixel_budget_adv_inputs_linf'] = round(float(dbg.get('pixel_budget_adv_inputs_linf', -1) or -1), 8)
                    row['fallback_adapter_used'] = bool(dbg.get('fallback_adapter_used', False))

        except Exception as e:
            row['infra_status'] = 'pgd_error: %s' % str(e)[:80]

        all_rows.append(row)
        print('  seed=%d clean=%+.1f(%s) adv=%s(%s) c2o=%s ce_drop=%s infra=%s' % (
            seed, clean_gripper, 'O' if clean_open else 'C',
            'O' if row['adv_open_bool'] else ('C' if row['adv_env_gripper'] != '' else '?'),
            row['adv_open_bool'] if row['adv_open_bool'] != '' else '?',
            row['clean_close_to_adv_open'] if row['clean_close_to_adv_open'] != '' else '?',
            round(row['loss_decrease'], 4) if row['loss_decrease'] != '' else '?',
            row['infra_status']), flush=True)

        env.close()

# ── Save ──
fieldnames = list(all_rows[0].keys()) if all_rows else []
out_path = os.path.join(args.output_dir, 'phase2_minimal_smoke_gpu%s_%s.csv' % (gpu_pair[0], gpu_pair[1]))
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(all_rows)

c2o = sum(1 for r in all_rows if int(r.get('clean_close_to_adv_open', 0) or 0) > 0)
infra_ok = sum(1 for r in all_rows if r['infra_status'] == 'ok')
print('\n=== Summary GPU%s ===' % args.gpus, flush=True)
print('Total: %d, C2O: %d, Infra OK: %d' % (len(all_rows), c2o, infra_ok), flush=True)
print('Output: %s' % out_path, flush=True)
