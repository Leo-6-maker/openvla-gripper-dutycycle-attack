#!/usr/bin/env python3
"""S7 Action-hidden full extraction — stable pool v2, pre-window only, clean forward."""
import csv, os, sys, json, hashlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--count', type=int, default=20)
ap.add_argument('--output_dir', default='/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_full')
args, _ = ap.parse_known_args()
# Do NOT override CUDA_VISIBLE_DEVICES — set by caller
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
visible = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')]
render_physical_gpu = visible[gpu_ids[1]] if len(visible) > gpu_ids[1] else gpu_ids[1]
os.makedirs(args.output_dir, exist_ok=True)

import numpy as np; import torch; from PIL import Image

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

# Load stable pool targets
STABLE = os.path.join(REPO, 'tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv')
KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']

targets = []
with open(STABLE) as f:
    for r in csv.DictReader(f):
        pk = r['parent']
        task = None
        for tk in KNOWN:
            if tk in pk: task = tk; break
        if not task: continue
        win = r.get('window','')
        parts = win.replace('_env','|').split('|')
        ws_we = parts[0].split('_')
        if len(ws_we) < 2: continue
        sid = int(parts[1]) if len(parts)>1 and parts[1].isdigit() else 0
        targets.append((pk[:50], task, sid, int(ws_we[0]), int(ws_we[1])))

# Dedup + slice
seen = set(); unique = []
for t in targets:
    key = (t[1], t[2], t[3], t[4])
    if key not in seen: seen.add(key); unique.append(t)
unique = unique[args.start:args.start+args.count]
print('Shard: start=%d count=%d -> %d windows' % (args.start, args.count, len(unique)))

# Load model BEFORE LIBERO
from transformers import AutoModelForVision2Seq, AutoProcessor
print('Loading model...')
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
UNNORM_KEY = 'libero_object'; action_dim = int(model.get_action_dim(UNNORM_KEY))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,'salad_dressing':3,'bbq_sauce':4,'milk':5,'alphabet_soup':6,'tomato_sauce':7,'orange_juice':8}
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

def decode_tokens(tids, VS, BC_NP, LO, HI, MK):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

results = []
for idx, (name, task_name, state_id, ws, we) in enumerate(unique):
    cfg = TASK_CFG.get(task_name)
    if cfg is None: continue
    task_obj = task_suite.get_task(cfg); init_states = task_suite.get_task_init_states(cfg)
    if state_id >= len(init_states): continue

    instruction = task_obj.language if hasattr(task_obj,'language') else task_name.replace('_',' ')
    bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, camera_names=['agentview'],
                             control_freq=20, render_gpu_device_id=render_physical_gpu)
    env.seed(state_id); env.reset(); env.set_init_state(init_states[state_id])
    prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()
    done = False; step = 0; max_steps = we + 10; step_records = []

    while not done and step < max_steps:
        img = env.sim.render(256, 256, camera_name='agentview')
        img_pil = Image.fromarray(img.astype(np.uint8)).rotate(180)
        inp = processor(prompt, img_pil)
        inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
               if isinstance(v,torch.Tensor) else v for k,v in inp.items()}
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True,
                                 output_hidden_states=True)
        # Only take last layer hidden to avoid OOM
        hidden_tup = gen.hidden_states[-1]  # last token's hidden states
        gripper_hidden = hidden_tup[-1][0, 0, :].cpu().float().numpy()  # last layer only
        token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
        action = decode_tokens(token_ids, VS, BC_NP, LO, HI, MK)
        raw_gripper = float(action[-1])
        env_action = action.copy(); env_action[-1] = -1.0 if raw_gripper>0.5 else 1.0
        obs, reward, done, info = env.step(env_action)
        step_records.append({'step':step, 'raw_gripper':round(raw_gripper,6),
            'env_gripper':env_action[-1], 'hidden_dim':len(gripper_hidden),
            'hidden_mean':float(np.mean(gripper_hidden)), 'hidden_std':float(np.std(gripper_hidden))})
        step += 1
    env.close(); torch.cuda.empty_cache()

    pre = [r for r in step_records if r['step'] < ws]
    if len(pre) < 5: continue
    hidden_dim = pre[0]['hidden_dim']
    # Last-step hidden and pre-window hidden stats
    last_hidden_data = step_records[ws-1] if ws > 0 and ws <= len(step_records) else None
    pre_rg = ','.join(str(r['raw_gripper']) for r in pre)
    pre_hash = hashlib.md5(pre_rg.encode()).hexdigest()[:12]

    results.append({
        'window':name,'task':task_name,'ws':ws,'we':we,'state_id':state_id,
        'n_pre':len(pre),'hidden_dim':hidden_dim,'pre_hash':pre_hash,'prompt':prompt,
        'online_safe':True,'feature_source':'pre_window_only',
        'h_mean_mean':round(np.mean([r['hidden_mean'] for r in pre]),6),
        'h_mean_std':round(np.std([r['hidden_mean'] for r in pre]),6),
        'h_std_mean':round(np.mean([r['hidden_std'] for r in pre]),6),
        'h_last_mean':round(last_hidden_data['hidden_mean'],6) if last_hidden_data else 0,
        'h_last_std':round(last_hidden_data['hidden_std'],6) if last_hidden_data else 0,
        'h_mean_slope':round(np.polyfit(range(len(pre)),[r['hidden_mean'] for r in pre],1)[0],6) if len(pre)>1 else 0,
    })
    print('[%d/%d] %s pre=%d dim=%d' % (idx+1, len(unique), task_name, len(pre), hidden_dim))

suffix = '_w%d' % args.start
out_csv = os.path.join(args.output_dir, 'action_hidden_full_features%s.csv' % suffix)
with open(out_csv,'w',newline='') as f:
    w = csv.writer(f)
    cols = ['window','task','ws','we','state_id','n_pre','hidden_dim','pre_hash','prompt',
            'online_safe','feature_source','h_mean_mean','h_mean_std','h_std_mean',
            'h_last_mean','h_last_std','h_mean_slope']
    w.writerow(cols)
    for r in results: w.writerow([r.get(c,'') for c in cols])
print('Saved %d rows -> %s' % (len(results), out_csv))
