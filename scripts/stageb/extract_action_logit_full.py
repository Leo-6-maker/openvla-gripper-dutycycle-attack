#!/usr/bin/env python3
"""Full action-logit extraction — stable pool v2 + confirmation windows. P1 online-safe."""
import csv, os, sys, json, hashlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--targets_csv', default='tables/action_logit_full_targets.csv')
ap.add_argument('--output_dir', default='/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full')
ap.add_argument('--start', type=int, default=0, help='Start index in targets (0-based)')
ap.add_argument('--count', type=int, default=0, help='Max windows to process (0=all)')
args, _ = ap.parse_known_args()
# Do NOT override CUDA_VISIBLE_DEVICES — it must be set by the caller
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
# Map CUDA index to physical GPU ID for EGL rendering
visible = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
render_physical_gpu = visible[gpu_ids[1]] if len(visible) > gpu_ids[1] else gpu_ids[1]

import numpy as np; import torch; from PIL import Image

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
os.makedirs(args.output_dir, exist_ok=True)

# Load targets
targets = []
with open(args.targets_csv) as f:
    for r in csv.DictReader(f):
        targets.append((r['window_id'], r['task'], int(r['state_id']), int(r['window_start']), int(r['window_end'])))
# Dedup
seen = set(); unique = []
for t in targets:
    key = (t[1], t[2], t[3], t[4])
    if key not in seen: seen.add(key); unique.append(t)
print('Targets: %d total, %d unique' % (len(targets), len(unique)))
targets = unique

# Slice for parallel workers
if args.count > 0:
    targets = targets[args.start:args.start + args.count]
    print('Slice: start=%d count=%d -> %d windows' % (args.start, args.count, len(targets)))

# Load model BEFORE LIBERO
from transformers import AutoModelForVision2Seq, AutoProcessor
print('Loading model...')
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
UNNORM_KEY = 'libero_object'
action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

gripper_dim = action_dim - 1
open_ids, close_ids = set(), set()
for tid in range(VS):
    disc = np.clip(VS - tid - 1, 0, len(BC_NP)-1)
    val = 0.5*(BC_NP[disc]+1)*(HI[gripper_dim]-LO[gripper_dim])+LO[gripper_dim]
    (open_ids if val > 0.5 else close_ids).add(tid)
open_ids = sorted(open_ids); close_ids = sorted(close_ids)

# Now import LIBERO — prevent TF from using GPU memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using GPU
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_CFG = {'ketchup':0,'butter':1,'cream_cheese':2,'salad_dressing':3,'bbq_sauce':4,'milk':5,'alphabet_soup':6,'tomato_sauce':7,'orange_juice':8}
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()

def decode_tokens(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

results = []
for idx, (name, task_name, state_id, ws, we) in enumerate(targets):
    print('[%d/%d] %s s=%d [%d,%d]' % (idx+1, len(targets), task_name, state_id, ws, we), end=' ')
    cfg = TASK_CFG.get(task_name)
    if cfg is None: print('SKIP unknown'); continue
    try:
        task_obj = task_suite.get_task(cfg)
        init_states = task_suite.get_task_init_states(cfg)
    except: print('SKIP no task'); continue
    if state_id >= len(init_states): print('SKIP OOB'); continue

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
                                 return_dict_in_generate=True, output_scores=True)
        token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
        action = decode_tokens(token_ids); raw_gripper = float(action[-1])
        gripper_scores = gen.scores[-1][0]
        gripper_probs = torch.softmax(gripper_scores.float(), dim=-1)
        open_prob = float(gripper_probs[open_ids].sum().cpu())
        close_prob = float(gripper_probs[close_ids].sum().cpu())
        total = open_prob+close_prob
        open_norm = open_prob/total if total>0 else 0.5
        logit_margin = float(gripper_scores[open_ids].max() - gripper_scores[close_ids].max())
        entropy = float(-(gripper_probs * torch.log(gripper_probs+1e-12)).sum().cpu())
        top2_vals, _ = torch.topk(gripper_scores.float(), 2)
        top2_margin = float(top2_vals[0]-top2_vals[1])

        env_action = action.copy(); env_action[-1] = -1.0 if raw_gripper>0.5 else 1.0
        obs, reward, done, info = env.step(env_action)
        step_records.append({'step':step,'raw_gripper':round(raw_gripper,6),'env_gripper':env_action[-1],
            'open_prob':round(open_prob,6),'close_prob':round(close_prob,6),
            'open_norm':round(open_norm,6),'logit_margin':round(logit_margin,6),
            'entropy':round(entropy,6),'top2_margin':round(top2_margin,6),'token_ids':token_ids.tolist()})
        step += 1
    env.close()
    torch.cuda.empty_cache()  # prevent OOM accumulation across windows

    pre = [r for r in step_records if r['step'] < ws]
    win = [r for r in step_records if ws <= r['step'] <= we]
    post = [r for r in step_records if r['step'] > we]
    if len(pre) < 5: print('SKIP too few pre'); continue

    pre_rg = ','.join(str(r['raw_gripper']) for r in pre)
    pre_env = ','.join(str(r['env_gripper']) for r in pre)
    pre_tok = ','.join(str(r['token_ids']) for r in pre)
    pre_hash = hashlib.md5((pre_rg+'|'+pre_env+'|'+pre_tok).encode()).hexdigest()[:12]

    def agg(fn, d=pre): return round(float(np.mean([r[fn] for r in d])), 6) if d else 0
    def aggl(fn, d=pre): return float(d[-1][fn]) if d else 0
    def agg_std(fn, d=pre): return round(float(np.std([r[fn] for r in d])), 6) if len(d)>1 else 0

    results.append({
        'window':name,'task':task_name,'ws':ws,'we':we,'state_id':state_id,
        'n_pre':len(pre),'n_win':len(win),'n_post':len(post),
        'online_safe':True,'feature_source':'pre_window_only','model_path':MODEL_PATH,
        'image_preprocess':'official_rot180_only','prompt':prompt,'pre_hash':pre_hash,
        'rg_mean':agg('raw_gripper'),'rg_std':agg_std('raw_gripper'),
        'rg_last':aggl('raw_gripper'),
        'rg_slope':round(np.polyfit(range(len(pre)),[r['raw_gripper'] for r in pre],1)[0],6) if len(pre)>1 else 0,
        'open_norm_mean':agg('open_norm'),'open_norm_last':aggl('open_norm'),
        'logit_margin_mean':agg('logit_margin'),'logit_margin_last':aggl('logit_margin'),
        'entropy_mean':agg('entropy'),'entropy_last':aggl('entropy'),
        'top2_margin_mean':agg('top2_margin'),'top2_margin_last':aggl('top2_margin'),
    })
    print('pre=%d open_norm=%.3f margin=%.1f entropy=%.3f' % (len(pre),
        results[-1]['open_norm_mean'],results[-1]['logit_margin_mean'],results[-1]['entropy_mean']))

# Save
suffix = '_w%d' % args.start if args.count > 0 else ''
out_csv = os.path.join(args.output_dir, 'action_logit_full_features%s.csv' % suffix)
with open(out_csv,'w',newline='') as f:
    w=csv.writer(f)
    cols=['window','task','ws','we','state_id','n_pre','n_win','n_post','online_safe','feature_source',
          'model_path','image_preprocess','prompt','pre_hash',
          'rg_mean','rg_std','rg_last','rg_slope',
          'open_norm_mean','open_norm_last','logit_margin_mean','logit_margin_last',
          'entropy_mean','entropy_last','top2_margin_mean','top2_margin_last']
    w.writerow(cols)
    for r in results: w.writerow([r.get(c,'') for c in cols])
print('Saved %d rows -> %s' % (len(results), out_csv))
