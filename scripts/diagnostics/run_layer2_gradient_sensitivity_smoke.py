#!/usr/bin/env python3
"""Layer-2 clean-only gradient sensitivity audit — no attack rollout.
Measures: how sensitive is the gripper OPEN logit to input pixels vs arm logits?
Runs on idle GPU, no env.step, no VIS/RAND.
"""
import csv, os, sys, argparse, json, hashlib
from datetime import datetime
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', default='4,5')
ap.add_argument('--output_dir', default='/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_gradient_sensitivity_smoke')
args, _ = ap.parse_known_args()

import torch; from PIL import Image

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE: print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

from gripper_attack.openvla_libero_exec_spec import (
    OFFICIAL_UNNORM_KEY_LIBERO_OBJECT as UNNORM_KEY,
    official_prompt,
    get_libero_image_official,
)
from transformers import AutoModelForVision2Seq, AutoProcessor

print('[%s] Loading model...' % datetime.now().strftime('%H:%M:%S'))
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
model.eval()

VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32); HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASK_CFG = {'butter':1,'cream_cheese':2,'milk':5,'tomato_sauce':7}
bm_dict = benchmark.get_benchmark_dict(); task_suite = bm_dict['libero_object']()

# Target windows: task, sid, ws, pre-sample-steps (before window)
TARGETS = [
    ('milk', 0, 70, 80, [60, 65, 69]),
    ('butter', 0, 90, 100, [80, 85, 89]),
    ('cream_cheese', 0, 65, 75, [55, 60, 64]),
    ('tomato_sauce', 2, 165, 175, [155, 160, 164]),
]

os.makedirs(args.output_dir, exist_ok=True)
results = []

for task, sid, ws, we, pre_steps in TARGETS:
    cfg = TASK_CFG.get(task)
    if cfg is None: continue
    task_obj = task_suite.get_task(cfg); initial_states = task_suite.get_task_init_states(cfg)
    if sid >= len(initial_states): continue
    instruction = str(task_obj.language) if hasattr(task_obj,'language') else task.replace('_',' ')
    bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                             render_gpu_device_id=_render_gpu)
    env.seed(sid); obs = env.reset()
    env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(initial_states[sid])

    step = 0; max_s = max(pre_steps) + 1
    while step < max_s:
        img = get_libero_image_official(obs)
        pil = Image.fromarray(img.astype(np.uint8))

        if step in pre_steps:
            # Forward + backward for gripper OPEN logit
            text = official_prompt(instruction.lower())
            inp = processor(text, pil, return_tensors='pt')
            for k, v in list(inp.items()):
                if torch.is_floating_point(v):
                    inp[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    inp[k] = v.to(model_device)
            if not torch.all(inp['input_ids'][:, -1] == 29871):
                inp['input_ids'] = torch.cat((inp['input_ids'],
                    torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)

            # Enable gradients on pixel_values
            pv = inp['pixel_values'].clone().detach().to(dtype=torch.float32)
            pv.requires_grad_(True)
            inp['pixel_values'] = pv.to(dtype=model_dtype)
            # Also need to handle the dtype for the model — keep as model_dtype for forward
            # but grad needs float32. Use mixed: forward in bf16, grad computed on float32 copy

            try:
                with torch.enable_grad():
                    pv_f32 = inp['pixel_values'].detach().to(dtype=torch.float32, device=model_device)
                    pv_f32.requires_grad_(True)
                    inp_fwd = {k: (v.to(dtype=torch.float32) if k == 'pixel_values' and torch.is_floating_point(v) else v) for k,v in inp.items()}
                    inp_fwd['pixel_values'] = pv_f32

                    gen = model.generate(**inp_fwd, max_new_tokens=action_dim, do_sample=False,
                                         return_dict_in_generate=True, output_scores=True)
                scores = gen.scores  # list of tensors, one per generated token

                # Gripper token is the last generated token (action_dim-th)
                gripper_scores = scores[-1][0]  # [vocab_size]
                # Arm tokens are first 6
                arm_scores = torch.stack([scores[i][0] for i in range(6)], dim=0)

                # Gradient of gripper max logit w.r.t. pixel_values
                gripper_max_logit = gripper_scores.max()
                gripper_max_logit.backward(retain_graph=False)
                grad_gripper = pv_f32.grad.clone()
                gripper_grad_norm = float(torch.norm(grad_gripper).cpu())

                # Reset and compute arm gradient
                pv_f32.grad = None
                arm_max_logit = arm_scores.max()
                arm_max_logit.backward(retain_graph=False)
                grad_arm = pv_f32.grad.clone()
                arm_grad_norm = float(torch.norm(grad_arm).cpu())

                # Gradient ratio: high = gripper more sensitive than arm
                grad_ratio = gripper_grad_norm / arm_grad_norm if arm_grad_norm > 0 else 0

                # Gradient cosine similarity (spatial alignment)
                grad_cos = float(torch.nn.functional.cosine_similarity(
                    grad_gripper.flatten(), grad_arm.flatten(), dim=0).cpu())

                results.append({
                    'task': task, 'state_id': sid, 'ws': ws, 'we': we,
                    'pre_step': step,
                    'gripper_grad_norm': round(gripper_grad_norm, 6),
                    'arm_grad_norm': round(arm_grad_norm, 6),
                    'grad_ratio': round(grad_ratio, 4),
                    'grad_cosine': round(grad_cos, 4),
                    'infra': 'ok',
                })
                print('[%s] step=%d grip_grad=%.4f arm_grad=%.4f ratio=%.4f cos=%.4f' % (
                    task, step, gripper_grad_norm, arm_grad_norm, grad_ratio, grad_cos))

            except Exception as e:
                results.append({
                    'task': task, 'state_id': sid, 'ws': ws, 'we': we,
                    'pre_step': step,
                    'gripper_grad_norm': -1, 'arm_grad_norm': -1,
                    'grad_ratio': -1, 'grad_cosine': -1,
                    'infra': 'grad_error: %s' % str(e)[:80],
                })
                print('[%s] step=%d GRAD_ERROR: %s' % (task, step, str(e)[:80]))

            torch.cuda.empty_cache()

        # Step env forward (no attack)
        with torch.inference_mode():
            text = official_prompt(instruction.lower())
            inp = processor(text, pil, return_tensors='pt')
            for k, v in list(inp.items()):
                if torch.is_floating_point(v):
                    inp[k] = v.to(device=model_device, dtype=model_dtype)
                else:
                    inp[k] = v.to(model_device)
            gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False)
        tids = gen[0, -action_dim:].cpu().numpy()
        disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
        action = np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)
        raw_grip = float(action[-1])
        env_a = action.copy()
        env_a[-1] = float(np.where(raw_grip > 0.5, -1.0, 1.0))
        obs, reward, done, info = env.step(env_a)
        step += 1

    env.close(); torch.cuda.empty_cache()
    print('[%s] done' % task)

# Save
out_csv = os.path.join(args.output_dir, 'layer2_gradient_sensitivity_smoke.csv')
cols = ['task','state_id','ws','we','pre_step','gripper_grad_norm','arm_grad_norm','grad_ratio','grad_cosine','infra']
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in results: w.writerow(r)

# Summary
print()
print('=== Gradient Sensitivity Summary ===')
print('%-15s %8s %8s %8s %8s %s' % ('Task', 'grip_grad', 'arm_grad', 'ratio', 'cosine', 'interpretation'))
for task in ['milk','butter','cream_cheese','tomato_sauce']:
    tr = [r for r in results if r['task'] == task and r['infra'] == 'ok']
    if not tr: continue
    g_mean = np.mean([r['gripper_grad_norm'] for r in tr])
    a_mean = np.mean([r['arm_grad_norm'] for r in tr])
    r_mean = np.mean([r['grad_ratio'] for r in tr])
    c_mean = np.mean([r['grad_cosine'] for r in tr])
    interp = 'GOOD (grip sensitive, arm sep)' if (r_mean > 1.5 and c_mean < 0.5) else \
             'MIXED' if r_mean > 0.8 else 'WEAK (arm-dominated)'
    print('%-15s %8.4f %8.4f %8.4f %8.4f %s' % (task, g_mean, a_mean, r_mean, c_mean, interp))

print()
print('Saved: %s' % out_csv)
print('Note: clean-only mechanism analysis. No attack rollout. No selector claim.')
