#!/usr/bin/env python3
"""Command proxy semantics microcheck — verify gripper qpos measurement + action injection."""

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

print('=== Gripper Semantics Microcheck ===')
print('CUDA_VISIBLE_DEVICES=2,6 (physical isolation)')
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
TID = {'cream_cheese': 1, 'bbq_sauce': 3}

samples = [
    ('cream_cheese', 4, 28, 45, 'positive'),
    ('bbq_sauce', 5, 27, 44, 'negative'),
]

res = []
for tk, sid, ws, we, label in samples:
    tid = TID[tk]
    ins = f"pick up the {tk.replace('_', ' ')} and place it in the basket"
    print(f"\n=== {tk}_s{sid} [{ws},{we}] ({label}) ===")

    # Probe gripper observation keys
    task = ts.get_task(tid)
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
    env.seed(0)
    init_states = ts.get_task_init_states(tid)
    obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(init_states[sid % len(init_states)])

    gripper_keys = [k for k in obs.keys() if 'gripper' in k.lower()]
    jn = env.sim.model.joint_names
    gi = [(i, n) for i, n in enumerate(jn) if 'gripper' in n.lower()]
    print(f"  Gripper obs keys: {gripper_keys}")
    print(f"  Gripper sim joints: {gi}")
    if gi:
        gq0 = float(env.sim.data.qpos[gi[0][0]])
        print(f"  Initial gripper qpos (sim): {gq0:.4f}")
        if gripper_keys:
            gq_obs = float(np.mean(np.asarray(obs[gripper_keys[0]]).ravel()))
            print(f"  Initial gripper qpos (obs): {gq_obs:.4f}")
    env.close()

    # === CLEAN rollout ===
    print("  [clean]...")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
    env.seed(0)
    obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(init_states[sid % len(init_states)])

    c_gq = []; c_act = []; step = 0; done = False
    max_steps = min(we + 20, 250)
    while step < max_steps and not done:
        img = Image.fromarray(obs['agentview_image'][::-1])
        p = f"In: What action should the robot take to {ins.lower()}?\nOut:"
        inp = proc(p, img, return_tensors='pt'); inp.pop('attention_mask', None)
        iid = inp['input_ids'].to(dev)
        pv = inp['pixel_values'].to(device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            o = model.generate(iid, pixel_values=pv, max_new_tokens=7, do_sample=False,
                return_dict_in_generate=True, output_scores=True)
        toks = o.sequences[0, -7:].cpu().numpy()
        act = np.zeros(7, np.float32)
        for d in range(7):
            if mask[d]:
                bid = min(int(vs - toks[d] - 1), nb - 1)
                act[d] = 0.5*(bins[bid]+1.0)*(q99[d]-q01[d])+q01[d]
        c_act.append(float(act[-1]))

        # Use sim.data.qpos for gripper (obs gripper_qpos is always 0.0)
        gq = float(env.sim.data.qpos[gi[0][0]]) if gi else -999
        c_gq.append(gq)

        obs, reward, done, info = env.step(act)
        step += 1
    env.close()

    valid_gq = [x for x in c_gq if x != -999]
    cp = np.mean(valid_gq[:ws]) if len(valid_gq) > ws else float('nan')
    cw = np.mean(valid_gq[ws:we]) if len(valid_gq) > we else float('nan')
    ca = np.mean(valid_gq[we:]) if len(valid_gq) > we else float('nan')
    cact = np.mean(c_act[ws:we]) if len(c_act) > we else float('nan')
    print(f"  Clean: grip_pre={cp:.4f} grip_window={cw:.4f} grip_post={ca:.4f} decoded_gripper={cact:.4f}")

    # === FORCED OPEN rollout ===
    print("  [forced_open]...")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
    env.seed(0)
    obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(init_states[sid % len(init_states)])

    f_gq = []; f_act = []; f_inj = []; step = 0; done = False
    while step < max_steps and not done:
        img = Image.fromarray(obs['agentview_image'][::-1])
        p = f"In: What action should the robot take to {ins.lower()}?\nOut:"
        inp = proc(p, img, return_tensors='pt'); inp.pop('attention_mask', None)
        iid = inp['input_ids'].to(dev)
        pv = inp['pixel_values'].to(device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            o = model.generate(iid, pixel_values=pv, max_new_tokens=7, do_sample=False,
                return_dict_in_generate=True, output_scores=True)
        toks = o.sequences[0, -7:].cpu().numpy()
        act = np.zeros(7, np.float32)
        for d in range(7):
            if mask[d]:
                bid = min(int(vs - toks[d] - 1), nb - 1)
                act[d] = 0.5*(bins[bid]+1.0)*(q99[d]-q01[d])+q01[d]
        f_act.append(float(act[-1]))

        injected = 0
        if ws <= step < we:
            act[-1] = 1.0  # OPEN in env action space, injected BEFORE env.step()
            injected = 1
        f_inj.append(injected)

        obs, reward, done, info = env.step(act)

        # Use sim.data.qpos for gripper (obs gripper_qpos is always 0.0)
        gq = float(env.sim.data.qpos[gi[0][0]]) if gi else -999
        f_gq.append(gq)
        step += 1
    env.close()

    valid_fq = [x for x in f_gq if x != -999]
    fp = np.mean(valid_fq[:ws]) if len(valid_fq) > ws else float('nan')
    fw = np.mean(valid_fq[ws:we]) if len(valid_fq) > we else float('nan')
    fa = np.mean(valid_fq[we:]) if len(valid_fq) > we else float('nan')
    fd = fw - fp if not (np.isnan(fp) or np.isnan(fw)) else float('nan')
    n_inj = sum(f_inj)
    print(f"  Forced: grip_pre={fp:.4f} grip_window={fw:.4f} grip_post={fa:.4f} delta={fd:.4f} injected={n_inj}")

    res.append(dict(
        task_key=tk, state_id=sid, window_start=ws, window_end=we, label=label,
        clean_gripper_pre=round(cp, 6) if not np.isnan(cp) else '',
        clean_gripper_window=round(cw, 6) if not np.isnan(cw) else '',
        clean_gripper_post=round(ca, 6) if not np.isnan(ca) else '',
        clean_decoded_gripper=round(cact, 6) if not np.isnan(cact) else '',
        forced_gripper_pre=round(fp, 6) if not np.isnan(fp) else '',
        forced_gripper_window=round(fw, 6) if not np.isnan(fw) else '',
        forced_gripper_post=round(fa, 6) if not np.isnan(fa) else '',
        forced_gripper_delta=round(fd, 6) if not np.isnan(fd) else '',
        forced_injected_steps=n_inj,
        gripper_obs_keys=str(gripper_keys),
        measurement_valid=int(len(valid_fq) > 0),
        gpu_pair='2,6', provenance_status='ok',
        measurement_version='v1_obs_gripper_qpos',
        label_source='command_proxy_microcheck', label_confidence='proxy'))

with open(f'{OUT}/tables/command_proxy_semantics_microcheck_v0.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
    w.writeheader(); w.writerows(res)
print(f'\nMicrocheck done: {len(res)} rows, {time.time()-t0:.1f}s total')
