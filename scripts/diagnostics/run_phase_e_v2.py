#!/usr/bin/env python3
"""Phase E v2 — Phase-aligned low-budget canary with mechanism fields."""

import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6'

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.attack_adapter import (TokenPrefixPGDAttacker, prepare_openvla_image_for_attack, _prompt, get_adv_inputs_from_attack_result)

MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM = 'libero_object'
OUT = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'

TID = {'cream_cheese': 1, 'bbq_sauce': 3}

# Parent-start aligned (v1 centered [31,41]/[30,40] were natural-open)
CANARY_V2 = [
    ('cream_cheese', 4, 28, 45, 28, 38, 1),
    ('bbq_sauce',    5, 27, 44, 27, 37, 0),
]
EPS, STEPS, RESTARTS = 4, 10, 1
PHASE_ALIGNMENT = 'parent_start_aligned_manual_heuristic'

print("=== Phase E v2: Phase-Aligned Canary ===")
print(f"Alignment: {PHASE_ALIGNMENT}")
print(f"Budget: eps={EPS} steps={STEPS} restarts={RESTARTS}")
t0 = time.time()
mm = {0: '9GiB', 1: '9GiB', 'cpu': '2GiB'}
model = AutoModelForVision2Seq.from_pretrained(MODEL, attn_implementation='eager',
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    local_files_only=True, device_map='auto', max_memory=mm)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
dev = str(next(model.parameters()).device)
print(f"Model: {dev} in {time.time()-t0:.1f}s")

st = model.get_action_stats(UNNORM)
mask = np.asarray(st['mask'], dtype=bool)
q01 = np.asarray(st['q01'], np.float32); q99 = np.asarray(st['q99'], np.float32)
bins = np.asarray(model.bin_centers, np.float32)
vs = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
nb = len(bins)
eps_proc = EPS / 255.0

def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action

def read_gripper(env):
    try:
        jn = env.sim.model.joint_names
        for i, name in enumerate(jn):
            if 'gripper' in name.lower() and 'finger' in name.lower():
                adr = int(env.sim.model.jnt_qposadr[i])
                return float(env.sim.data.qpos[adr])
    except: pass
    return -999.0

bd = benchmark.get_benchmark_dict()
ts = bd['libero_object']()
res = []

for tk, sid, pws, pwe, ws, we, label in CANARY_V2:
    tid = TID[tk]
    ins = f"pick up the {tk.replace('_', ' ')} and place it in the basket"
    print(f"\n{tk}_s{sid} parent=[{pws},{pwe}] aligned=[{ws},{we}] label={label}")
    t_ep = time.time()

    try:
        task = ts.get_task(tid)
        bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
        env.seed(0); its = ts.get_task_init_states(tid)
        obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(its[sid % len(its)])

        step, done, vis_open, token_flips = 0, False, 0, 0
        gq_start = None; gq_min = 999
        arm_l2_list = []; raw_clean_g = []; raw_adv_g = []; env_adv_g = []

        while step < 300 and not done:
            img_np = np.array(Image.fromarray(obs['agentview_image'][::-1]))
            img = prepare_openvla_image_for_attack(img_np)
            inp = proc(_prompt(ins), img, return_tensors='pt'); inp.pop('attention_mask', None)
            iid = inp['input_ids'].to(dev)
            pv = inp['pixel_values'].to(device=dev, dtype=torch.bfloat16)

            with torch.no_grad():
                co = model.generate(iid, pixel_values=pv, max_new_tokens=7, do_sample=False, return_dict_in_generate=True, output_scores=True)
            ctoks = co.sequences[0, -7:].cpu().numpy()
            clean_act = np.zeros(7, np.float32)
            for d in range(7):
                if mask[d]: clean_act[d] = 0.5*(bins[min(int(vs-ctoks[d]-1), nb-1)]+1.0)*(q99[d]-q01[d])+q01[d]
            target_act = clean_act.copy(); target_act[-1] = 0.0

            gq = read_gripper(env)
            if step == ws and gq_start is None: gq_start = gq
            if ws <= step < we and gq >= 0: gq_min = min(gq_min, gq)
            raw_clean_g.append(float(clean_act[-1]))

            if ws <= step < we:
                config = {'attack_optimizer': {
                    'epsilon': float(eps_proc),
                    'step_size': float(max(eps_proc/STEPS, 1e-4)),
                    'num_steps': STEPS, 'random_start': False, 'temporal_init': 'none',
                    'objective': 'prefix_locked_gripper_open_margin',
                    'gripper_margin': 5.0, 'arm_preserve_weight': 0.1}}
                attacker = TokenPrefixPGDAttacker(model, proc, config, seed=0, device=dev)
                result = attacker.attack(img_np, instruction=ins, clean_action=clean_act, target_action=target_act, unnorm_key=UNNORM)
                adv_inputs = get_adv_inputs_from_attack_result(result)
                with torch.no_grad():
                    ao = model.generate(adv_inputs['input_ids'].to(dev),
                        pixel_values=adv_inputs['pixel_values'].to(device=dev, dtype=torch.bfloat16),
                        max_new_tokens=7, do_sample=False, return_dict_in_generate=True, output_scores=True)
                atoks = ao.sequences[0, -7:].cpu().numpy()
                adv_act = np.zeros(7, np.float32)
                for d in range(7):
                    if mask[d]: adv_act[d] = 0.5*(bins[min(int(vs-atoks[d]-1), nb-1)]+1.0)*(q99[d]-q01[d])+q01[d]
                raw_adv_g.append(float(adv_act[-1]))
                if float(adv_act[-1]) < 0.5: vis_open += 1
                if any(ctoks[d] != atoks[d] for d in range(7)): token_flips += 1
                env_action = invert_gripper_action(normalize_gripper_action(adv_act))
                env_adv_g.append(float(env_action[-1]))
                arm_l2_list.append(float(np.linalg.norm(adv_act[:6] - clean_act[:6])))
                obs, reward, done, info = env.step(env_action)
            else:
                env_action = invert_gripper_action(normalize_gripper_action(clean_act))
                obs, reward, done, info = env.step(env_action)
            step += 1

        env.close(); rt = time.time()-t_ep
        gq_delta = gq_min - gq_start if (gq_start >= 0 and gq_min < 999) else 0
        wlen = we - ws
        print(f"  done={done} steps={step} vis_open={vis_open}/{wlen} gq_start={gq_start:.4f} gq_min={gq_min:.4f} delta={gq_delta:.4f} arm_l2={np.mean(arm_l2_list) if arm_l2_list else 0:.4f} rt={rt:.1f}s")

        res.append(dict(task_key=tk, state_id=sid, full_vis_label=label,
            parent_window_start=pws, parent_window_end=pwe,
            window_start=ws, window_end=we, compressed_len=wlen,
            phase_alignment_source=PHASE_ALIGNMENT, alignment_score='',
            eps_raw_pixels=EPS, pgd_steps=STEPS, pgd_restarts=RESTARTS,
            objective='prefix_locked_gripper_open_margin',
            VIS_OPEN_count=vis_open, done=int(done), steps=step,
            qpos_opening_delta_mujoco=round(gq_delta,6),
            gripper_qpos_source_priority='mujoco_primary',
            env_adv_action_gripper_mean=round(np.mean(env_adv_g),4) if env_adv_g else '',
            raw_adv_action_gripper_mean=round(np.mean(raw_adv_g),4) if raw_adv_g else '',
            raw_clean_action_gripper_mean=round(np.mean(raw_clean_g),4) if raw_clean_g else '',
            arm_l2_mean=round(np.mean(arm_l2_list),6) if arm_l2_list else '',
            arm_l2_max=round(max(arm_l2_list),6) if arm_l2_list else '',
            token_flip_count=token_flips,
            runtime_sec=round(rt,2), gpu_pair='2,6',
            provenance_status='ok', denominator_status='low_budget_no_precheck',
            label_source='low_budget_vis', label_confidence='silver_candidate_pending_calibration',
            action_transform_version='official_normalize_then_invert_20260605',
            epsilon_calibration='script_direct_raw_div255'))
    except Exception as e:
        import traceback; traceback.print_exc()
        res.append(dict(task_key=tk, state_id=sid, full_vis_label=label,
            parent_window_start=pws, parent_window_end=pwe, window_start=ws, window_end=we,
            provenance_status=f'ERROR:{str(e)[:100]}', label_source='low_budget_vis',
            label_confidence='silver_candidate_pending_calibration'))

with open(f'{OUT}/tables/fast_vis_low_budget_canary_v2.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
    w.writeheader(); w.writerows(res)
print(f"\nPhase E v2 done: {len(res)} rows, {time.time()-t0:.1f}s total")
