#!/usr/bin/env python3
"""Action-logit clean extraction smoke: 4 windows, GPU clean forward sidecar.
Extracts OpenVLA internal action-decoding signals without modifying main runner.
"""
import csv, os, sys, json, time, argparse
import numpy as np
import torch
from PIL import Image

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_smoke'
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    ('A1_milk[70,80]', 'milk', 0, 70, 80),
    ('A2_butter[80,90]', 'butter', 0, 80, 90),
    ('FP_tomato[55,65]', 'tomato_sauce', 0, 55, 65),
    ('FN_salad[70,80]', 'salad_dressing', 2, 70, 80),
]

# ── Step 1: Load OpenVLA BEFORE LIBERO (TF otherwise steals GPU memory) ──
ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
args, _ = ap.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_pair
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]

from transformers import AutoModelForVision2Seq, AutoProcessor
print('Loading OpenVLA (before LIBERO)...')
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
print('Model loaded, device=%s dtype=%s' % (model_device, model_dtype))

# ── Step 2: Now import LIBERO (TF will go to leftover GPU memory) ──
import gym
gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

UNNORM_KEY = 'libero_object'
action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)

s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32)
HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

# Precompute: which token IDs produce gripper > 0.5 vs < 0.5
gripper_dim = action_dim - 1
open_token_ids = set()
close_token_ids = set()
for tid in range(VS):
    disc = np.clip(VS - tid - 1, 0, len(BC_NP) - 1)
    val = 0.5 * (BC_NP[disc] + 1) * (HI[gripper_dim] - LO[gripper_dim]) + LO[gripper_dim]
    if val > 0.5: open_token_ids.add(tid)
    elif val < 0.5: close_token_ids.add(tid)

open_ids = sorted(open_token_ids)
close_ids = sorted(close_token_ids)
open_token_set = set(open_ids)
close_token_set = set(close_ids)
print('Gripper: %d open tokens, %d close tokens out of %d bins' % (len(open_ids), len(close_ids), VS))

TASK_CFG = {
    'ketchup': 0, 'butter': 1, 'cream_cheese': 2, 'salad_dressing': 3,
    'bbq_sauce': 4, 'milk': 5, 'alphabet_soup': 6, 'tomato_sauce': 7, 'orange_juice': 8,
}

# ── Setup env ──
bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()

def decode_tokens(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP) - 1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

# ── Run extraction ──
results = []
for name, task_name, state_id, ws, we in TARGETS:
    print('\n=== %s: %s s=%d [%d,%d] ===' % (name, task_name, state_id, ws, we))

    cfg = TASK_CFG.get(task_name)
    if cfg is None: print('  SKIP: unknown task'); continue
    task_obj = task_suite.get_task(cfg)
    init_states = task_suite.get_task_init_states(cfg)
    if state_id >= len(init_states):
        print('  SKIP: state_id OOB'); continue

    instruction = task_obj.language if hasattr(task_obj, 'language') else task_name.replace('_', ' ')
    bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, camera_names=['agentview'],
                             control_freq=20, render_gpu_device_id=gpu_ids[1])
    env.seed(state_id); env.reset()
    env.set_init_state(init_states[state_id])

    # Official prompt: same as run_stageb_vis_labeling.py
    prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()
    done = False; step = 0; max_steps = we + 10
    step_records = []

    while not done and step < max_steps:
        img = env.sim.render(256, 256, camera_name='agentview')
        img_pil = Image.fromarray(img.astype(np.uint8)).rotate(180)

        # Same processor call as official runner
        inp = processor(prompt, img_pil)
        inp = {k: v.to(model_device, dtype=torch.bfloat16 if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v.dtype)
               if isinstance(v, torch.Tensor) else v for k, v in inp.items()}

        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)

        token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
        action = decode_tokens(token_ids)
        raw_gripper = float(action[-1])

        # Extract action-logit features
        scores = gen.scores  # list of [1, vocab_size] tensors, one per generated token
        gripper_scores = scores[-1][0]  # last token = gripper, shape [vocab_size]
        gripper_probs = torch.softmax(gripper_scores.float(), dim=-1)

        open_prob = float(gripper_probs[list(open_ids)].sum().cpu())
        close_prob = float(gripper_probs[list(close_ids)].sum().cpu())
        total_prob = open_prob + close_prob
        open_norm = open_prob / total_prob if total_prob > 0 else 0.5
        close_norm = close_prob / total_prob if total_prob > 0 else 0.5
        logit_margin = float(gripper_scores[list(open_ids)].max() - gripper_scores[list(close_ids)].max())

        # Entropy over all vocab
        log_probs = torch.log(gripper_probs + 1e-12)
        entropy = float(-(gripper_probs * log_probs).sum().cpu())

        # top-2 margin
        top2_vals, _ = torch.topk(gripper_scores.float(), 2)
        top2_margin = float(top2_vals[0] - top2_vals[1])

        # Env step
        env_action = action.copy()
        env_action[-1] = -1.0 if raw_gripper > 0.5 else 1.0
        obs, reward, done, info = env.step(env_action)

        step_records.append({
            'step': step, 'in_window': ws <= step <= we,
            'raw_gripper': round(raw_gripper, 6),
            'env_gripper': -1.0 if raw_gripper > 0.5 else 1.0,
            'open_prob': round(open_prob, 6), 'close_prob': round(close_prob, 6),
            'open_norm': round(open_norm, 6), 'close_norm': round(close_norm, 6),
            'logit_margin': round(logit_margin, 6),
            'entropy': round(entropy, 6), 'top2_margin': round(top2_margin, 6),
            'token_ids': token_ids.tolist(),
        })
        step += 1

    env.close()

    # P0-fix: pre = step < ws only (NOT post-window)
    pre = [r for r in step_records if r['step'] < ws]
    win = [r for r in step_records if ws <= r['step'] <= we]
    post = [r for r in step_records if r['step'] > we]

    if len(pre) < 5:
        print('  SKIP: too few pre steps (%d)' % len(pre)); continue

    # Prefix provenance hash
    import hashlib
    pre_rg_str = ','.join(str(r['raw_gripper']) for r in pre)
    pre_hash = hashlib.md5(pre_rg_str.encode()).hexdigest()[:8]

    def agg(name, func, data=pre):
        vals = [r[name] for r in data]
        return round(func(vals), 6)

    r = {
        'window': name, 'task': task_name, 'ws': ws, 'we': we,
        'n_pre': len(pre), 'n_win': len(win), 'n_post': len(post),
        'prompt': prompt, 'pre_hash': pre_hash,
        'online_safe': True,  # all features from step < ws only
        'rg_mean': agg('raw_gripper', np.mean), 'rg_std': agg('raw_gripper', np.std),
        'rg_last': pre[-1]['raw_gripper'], 'rg_slope': agg('raw_gripper', lambda x: np.polyfit(range(len(x)), x, 1)[0]),
        'open_prob_mean': agg('open_prob', np.mean), 'open_prob_std': agg('open_prob', np.std),
        'close_prob_mean': agg('close_prob', np.mean),
        'open_norm_mean': agg('open_norm', np.mean), 'open_norm_last': pre[-1]['open_norm'],
        'logit_margin_mean': agg('logit_margin', np.mean), 'logit_margin_std': agg('logit_margin', np.std),
        'logit_margin_last': pre[-1]['logit_margin'],
        'entropy_mean': agg('entropy', np.mean), 'entropy_std': agg('entropy', np.std),
        'entropy_last': pre[-1]['entropy'],
        'top2_margin_mean': agg('top2_margin', np.mean), 'top2_margin_last': pre[-1]['top2_margin'],
    }

    # Print key diagnostics
    print('  pre=%d win=%d  rg=%.3f±%.3f  open_prob=%.4f  margin=%.3f  entropy=%.3f  top2=%.3f' % (
        len(pre), len(win), r['rg_mean'], r['rg_std'], r['open_norm_mean'],
        r['logit_margin_mean'], r['entropy_mean'], r['top2_margin_mean']))
    results.append(r)

# ── Compare FP vs FN ──
fp = [r for r in results if 'FP' in r['window']]
fn = [r for r in results if 'FN' in r['window']]
if fp and fn:
    fp = fp[0]; fn = fn[0]
    print('\n=== FP vs FN Action-Logit Comparison ===')
    feats = ['rg_mean','rg_std','rg_slope','open_norm_mean','open_norm_last',
             'logit_margin_mean','logit_margin_std','logit_margin_last',
             'entropy_mean','entropy_std','entropy_last','top2_margin_mean','top2_margin_last']
    print('%-25s %18s %18s %10s' % ('Feature','FP tomato[55,65]','FN salad[70,80]','Delta'))
    for feat in feats:
        d = fp[feat] - fn[feat]
        print('%-25s %18s %18s %+10s' % (feat, str(fp[feat]), str(fn[feat]), str(round(d, 6))))

# Save
with open(os.path.join(OUT_DIR, 'action_logit_smoke_features.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    cols = ['window','task','ws','we','n_pre','n_win','rg_mean','rg_std','rg_last','rg_slope',
            'open_prob_mean','open_prob_std','close_prob_mean','open_norm_mean','open_norm_last',
            'logit_margin_mean','logit_margin_std','logit_margin_last',
            'entropy_mean','entropy_std','entropy_last',
            'top2_margin_mean','top2_margin_last']
    w.writerow(cols)
    for r_ in results:
        w.writerow([r_[c] for c in cols])

print('\nSaved: %s' % os.path.join(OUT_DIR, 'action_logit_smoke_features.csv'))
