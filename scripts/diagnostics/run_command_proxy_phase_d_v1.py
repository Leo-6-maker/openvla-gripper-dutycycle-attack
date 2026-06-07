#!/usr/bin/env python3
"""Phase D v1 — Command-Open Proxy with correct measurement + task-level outcomes.
8 calibration samples, full 300-step episodes, Mujoco sim qpos for gripper.
"""

import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6'

import torch, numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM = 'libero_object'
OUT = os.environ.get('OUT_DIR', '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605')
MEASUREMENT_VERSION = 'v1_mujoco_sim_qpos_20260605'
ACTION_INJECTION_VERSION = 'v1_env_step_pre_injection_20260605'

TID = {'alphabet_soup':0,'cream_cheese':1,'salad_dressing':2,'bbq_sauce':3,'ketchup':4,
       'tomato_sauce':5,'butter':6,'milk':7,'chocolate_pudding':8,'orange_juice':9}

print('=== Phase D v1: Command-Open Proxy ===')
print(f'GPU: 2,6 (CUDA_VISIBLE_DEVICES isolation)')
t0 = time.time()

mm = {0: '9GiB', 1: '9GiB', 'cpu': '2GiB'}
model = AutoModelForVision2Seq.from_pretrained(MODEL, attn_implementation='eager',
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    local_files_only=True, device_map='auto', max_memory=mm)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
dev = str(next(model.parameters()).device)
print(f'Model: {dev} in {time.time()-t0:.1f}s')

st = model.get_action_stats(UNNORM)
mask = np.asarray(st['mask'], dtype=bool)
q01 = np.asarray(st['q01'], np.float32); q99 = np.asarray(st['q99'], np.float32)
bins = np.asarray(model.bin_centers, np.float32)
vs = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
nb = len(bins)

bd = benchmark.get_benchmark_dict()
ts = bd['libero_object']()

with open(f'{OUT}/tables/fast_vis_calibration_candidates_v0.csv') as f:
    cands = list(csv.DictReader(f))

def read_gripper(env):
    """Read gripper qpos from Mujoco sim joints."""
    try:
        jn = env.sim.model.joint_names
        for i, name in enumerate(jn):
            if 'gripper' in name.lower() and 'finger' in name.lower():
                adr = int(env.sim.model.jnt_qposadr[i])
                return float(env.sim.data.qpos[adr]), f'mujoco:{name}'
    except: pass
    return -999.0, 'unavailable'

res = []
for i, c in enumerate(cands):
    tk = c['task_key'].strip(); sid = int(c['state_id'])
    ws = int(c['parent_window_start']); we = int(c['parent_window_end'])
    label = c.get('full_vis_label', '')
    phase = c.get('phase_bin_proxy', '')

    tid = TID.get(tk)
    if tid is None:
        res.append(dict(task_key=tk, state_id=sid, window_start=ws, window_end=we,
            phase_bin_proxy=phase, full_vis_label=label,
            provenance_status='ERROR:no_task_mapping'))
        continue

    ins = f"pick up the {tk.replace('_', ' ')} and place it in the basket"
    print(f'[{i+1}/8] {tk}_s{sid} [{ws},{we}] label={label}')
    t_ep = time.time()

    try:
        task = ts.get_task(tid)
        bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
        env.seed(0)
        init_states = ts.get_task_init_states(tid)
        obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(init_states[sid % len(init_states)])

        step = 0; done = False; foc = 0
        gq_start = None; gq_min_window = 999; gq_post_max = 0
        clean_gripper_act = None; gq_vals = []

        while step < 300 and not done:
            img = Image.fromarray(obs['agentview_image'][::-1])
            p = f"In: What action should the robot take to {ins.lower()}?\nOut:"
            inp = proc(p, img, return_tensors='pt'); inp.pop('attention_mask', None)
            iid = inp['input_ids'].to(dev)
            pv = inp['pixel_values'].to(device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                o = model.generate(iid, pixel_values=pv, max_new_tokens=7,
                    do_sample=False, return_dict_in_generate=True, output_scores=True)
            toks = o.sequences[0, -7:].cpu().numpy()
            act = np.zeros(7, np.float32)
            for d in range(7):
                if mask[d]:
                    bid = min(int(vs - toks[d] - 1), nb - 1)
                    act[d] = 0.5*(bins[bid]+1.0)*(q99[d]-q01[d])+q01[d]

            # Record clean gripper action at window start
            if step == ws and clean_gripper_act is None:
                clean_gripper_act = float(act[-1])

            # Measure gripper qpos BEFORE env.step
            gq, gq_src = read_gripper(env)
            gq_vals.append(gq)
            if step == ws and gq_start is None:
                gq_start = gq
            if ws <= step < we and gq >= 0:
                gq_min_window = min(gq_min_window, gq)

            # Force OPEN during window
            injected = 0
            if ws <= step < we:
                act[-1] = 1.0  # OPEN in env action space
                injected = 1
                foc += 1

            obs, reward, done, info = env.step(act)
            step += 1

        # Post-window gripper
        gq_end, _ = read_gripper(env) if step > 0 else (-999, '')
        if gq_end >= 0:
            gq_post_max = max(gq_post_max, gq_end)

        env.close()
        rt = time.time() - t_ep

        # Compute metrics
        gq_delta = gq_min_window - gq_start if (gq_start >= 0 and gq_min_window < 999) else 0
        gq_valid = [x for x in gq_vals if x >= 0]

        print(f'  done={done} steps={step} foc={foc} gq_start={gq_start:.4f} '
              f'gq_min_window={gq_min_window:.4f} delta={gq_delta:.4f} rt={rt:.1f}s')

        res.append(dict(
            task_key=tk, state_id=sid, window_start=ws, window_end=we,
            phase_bin_proxy=phase, full_vis_label=label,
            forced_open_count=foc, forced_open_value_used=1.0,
            clean_gripper_action=round(clean_gripper_act,6) if clean_gripper_act is not None else '',
            forced_gripper_action=1.0, post_transform_gripper_action=1.0,
            measurement_version=MEASUREMENT_VERSION,
            action_injection_version=ACTION_INJECTION_VERSION,
            gripper_qpos_source=gq_src,
            gripper_qpos_start=round(gq_start,6) if gq_start >= 0 else '',
            gripper_qpos_min_in_window=round(gq_min_window,6) if gq_min_window < 999 else '',
            gripper_qpos_delta=round(gq_delta,6),
            gripper_qpos_abs_after_max=round(gq_post_max,6) if gq_post_max > 0 else '',
            physical_response_class=('strong' if abs(gq_delta)>0.01 else ('weak' if abs(gq_delta)>0.001 else 'none')),
            done=int(done), steps=step, task_failure=int(not done),
            task_negative=int(done), gpu_pair='2,6',
            runtime_sec=round(rt,2), provenance_status='ok',
            denominator_status='command_proxy_no_denom_check',
            label_source='command_proxy', label_confidence='proxy'))
    except Exception as e:
        import traceback; traceback.print_exc()
        res.append(dict(task_key=tk, state_id=sid, window_start=ws, window_end=we,
            phase_bin_proxy=phase, full_vis_label=label,
            provenance_status=f'ERROR:{str(e)[:100]}', gpu_pair='2,6',
            measurement_version=MEASUREMENT_VERSION,
            label_source='command_proxy', label_confidence='proxy'))

# Write CSV
with open(f'{OUT}/tables/fast_vis_command_proxy_v1.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
    w.writeheader(); w.writerows(res)

total_time = time.time() - t0
pos = [r for r in res if str(r.get('full_vis_label','')) == '1']
neg = [r for r in res if str(r.get('full_vis_label','')) == '0']
pos_fail = sum(r.get('task_failure',0) or 0 for r in pos)
neg_fail = sum(r.get('task_failure',0) or 0 for r in neg)
print(f'\nPhase D v1 done: {len(res)} rows, {total_time:.1f}s')
print(f'Pos task_failure: {pos_fail}/{len(pos)}  Neg task_failure: {neg_fail}/{len(neg)}')
