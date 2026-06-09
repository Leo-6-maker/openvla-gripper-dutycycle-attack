#!/usr/bin/env python3
"""Action-hidden clean extraction smoke — 4 windows, GPU clean forward sidecar."""
import csv, os, sys, json, hashlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='0,1')
ap.add_argument('--output_dir', default='/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_smoke_4w')
args, _ = ap.parse_known_args()
# Do NOT override CUDA_VISIBLE_DEVICES — must be set by caller
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
visible = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
render_physical_gpu = visible[gpu_ids[1]] if len(visible) > gpu_ids[1] else gpu_ids[1]
os.makedirs(args.output_dir, exist_ok=True)

import numpy as np; import torch; from PIL import Image

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

TARGETS = [
    ('A1_milk[70,80]','milk',0,70,80),
    ('A2_butter[80,90]','butter',0,80,90),
    ('FP_tomato[55,65]','tomato_sauce',0,55,65),
    ('FN_salad[70,80]','salad_dressing',2,70,80),
]

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

results = []
for name, task_name, state_id, ws, we in TARGETS:
    cfg = TASK_CFG.get(task_name)
    if cfg is None: continue
    task_obj = task_suite.get_task(cfg)
    init_states = task_suite.get_task_init_states(cfg)
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

        # Get last-layer hidden at each generation step
        hidden_states = gen.hidden_states  # tuple of tuples: (n_tokens, (n_layers, B, 1, hidden_dim))
        # Last token (gripper), last layer
        gripper_hidden = hidden_states[-1][-1][0, 0, :].cpu().float().numpy()  # shape [hidden_dim]

        # Decode action
        token_ids = gen.sequences[0, -action_dim:].cpu().numpy()
        VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
        s = model.get_action_stats(UNNORM_KEY)
        LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
        MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)
        def decode_tokens(tids):
            tids = np.asarray(tids, dtype=np.int64)
            disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
            return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)
        action = decode_tokens(token_ids); raw_gripper = float(action[-1])

        # Logit features
        gripper_scores = gen.scores[-1][0]
        gripper_probs = torch.softmax(gripper_scores.float(), dim=-1)
        gripper_dim = action_dim - 1
        open_ids = [tid for tid in range(VS) if 0.5*(BC_NP[np.clip(VS-tid-1,0,len(BC_NP)-1)]+1)*(HI[gripper_dim]-LO[gripper_dim])+LO[gripper_dim] > 0.5]
        close_ids = [tid for tid in range(VS) if 0.5*(BC_NP[np.clip(VS-tid-1,0,len(BC_NP)-1)]+1)*(HI[gripper_dim]-LO[gripper_dim])+LO[gripper_dim] < 0.5]
        open_prob = float(gripper_probs[open_ids].sum().cpu())
        close_prob = float(gripper_probs[close_ids].sum().cpu())
        total = open_prob+close_prob
        open_norm = open_prob/total if total>0 else 0.5
        logit_margin = float(gripper_scores[open_ids].max() - gripper_scores[close_ids].max())
        entropy = float(-(gripper_probs * torch.log(gripper_probs+1e-12)).sum().cpu())

        env_action = action.copy(); env_action[-1] = -1.0 if raw_gripper>0.5 else 1.0
        obs, reward, done, info = env.step(env_action)
        step_records.append({
            'step':step, 'raw_gripper':round(raw_gripper,6), 'env_gripper':env_action[-1],
            'open_norm':round(open_norm,6), 'logit_margin':round(logit_margin,6), 'entropy':round(entropy,6),
            'hidden_dim': len(gripper_hidden),
        })
        step += 1
    env.close(); torch.cuda.empty_cache()

    pre = [r for r in step_records if r['step'] < ws]
    if len(pre) < 5: continue
    pre_rg = ','.join(str(r['raw_gripper']) for r in pre)
    pre_hash = hashlib.md5(pre_rg.encode()).hexdigest()[:12]

    hidden_dim = step_records[0]['hidden_dim']
    # PCA-ready: save last pre-window hidden as raw vector
    last_hidden = None  # We need to re-extract since we didn't save per-step hidden
    print('%s: pre=%d hidden_dim=%d' % (name, len(pre), hidden_dim))
    results.append({
        'window':name,'task':task_name,'ws':ws,'we':we,'n_pre':len(pre),
        'hidden_dim':hidden_dim,'pre_hash':pre_hash,'prompt':prompt,
        'online_safe':True,'feature_source':'pre_window_only',
    })

# Save manifest
with open(os.path.join(args.output_dir, 'hidden_smoke_manifest.csv'),'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['window','task','ws','we','n_pre','hidden_dim','pre_hash','online_safe'])
    for r in results:
        w.writerow([r['window'],r['task'],r['ws'],r['we'],r['n_pre'],r['hidden_dim'],r['pre_hash'],r['online_safe']])
print('Saved %d windows, hidden_dim=%d' % (len(results), results[0]['hidden_dim'] if results else 0))
