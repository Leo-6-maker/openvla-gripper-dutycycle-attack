#!/usr/bin/env python3
"""Export MuJoCo gripper qpos trace for Phase E calibration candidates.
Clean rollout only — no PGD, no VIS. GPU 2,6.
"""

import sys, os, time, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6'

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM = 'libero_object'
OUT = '/data/liuyu/outputs/phaseE_mujoco_qpos_cache_20260605'

TID = {t: i for i, t in enumerate([
    'alphabet_soup','cream_cheese','salad_dressing','bbq_sauce','ketchup',
    'tomato_sauce','butter','milk','chocolate_pudding','orange_juice'])}

def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize: action[..., -1] = np.sign(action[..., -1]); action[..., -1] = 1.0 if action[..., -1]==0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy(); action[..., -1] = -1.0 * action[..., -1]
    return action

def read_gripper(env):
    try:
        jn = env.sim.model.joint_names
        for i, name in enumerate(jn):
            if 'gripper' in name.lower() and 'finger' in name.lower():
                return float(env.sim.data.qpos[env.sim.model.jnt_qposadr[i]])
    except: pass
    return -999.0

def read_obs_gripper(obs):
    if 'robot0_gripper_qpos' in obs:
        arr = np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32).ravel()
        if arr.size > 0: return float(arr[0])
    return -999.0

print("=== Phase E MuJoCo qpos cache export ===")
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

bd = benchmark.get_benchmark_dict()
ts = bd['libero_object']()

# Read calibration candidates
cands = []
with open('/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/fast_vis_calibration_candidates_v0.csv') as f:
    cands = list(csv.DictReader(f))

os.makedirs(OUT, exist_ok=True)

for c in cands:
    tk = c['task_key'].strip(); sid = int(c['state_id'])
    tid = TID.get(tk)
    if tid is None: continue
    ins = f"pick up the {tk.replace('_', ' ')} and place it in the basket"

    ep_dir = os.path.join(OUT, f"{tk}_s{sid}")
    os.makedirs(ep_dir, exist_ok=True)
    out_csv = os.path.join(ep_dir, 'qpos_trace.csv')

    if os.path.exists(out_csv):
        print(f"SKIP {tk}_s{sid}: already cached")
        continue

    print(f"{tk}_s{sid}...")
    try:
        task = ts.get_task(tid)
        bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
        env.seed(0); its = ts.get_task_init_states(tid)
        obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(its[sid % len(its)])

        rows = []; step = 0; done = False
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

            gq_m = read_gripper(env)
            gq_o = read_obs_gripper(obs)
            env_act = invert_gripper_action(normalize_gripper_action(act))
            rows.append(dict(step=step, task_key=tk, state_id=sid,
                gripper_qpos_mujoco=round(gq_m, 8),
                robot0_gripper_qpos_obs=round(gq_o, 8),
                qpos_source='mujoco_trace',
                clean_raw_gripper=round(float(act[-1]), 6),
                clean_env_gripper_after_transform=round(float(env_act[-1]), 6),
                clean_open_flag=int(float(act[-1]) < 0.5),
                done=int(done), steps=step,
                provenance_status='ok', denominator_status='qpos_cache_only'))

            obs, reward, done, info = env.step(env_act)
            step += 1
        env.close()

        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  {len(rows)} steps -> {out_csv}")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"Done: {time.time()-t0:.1f}s")
