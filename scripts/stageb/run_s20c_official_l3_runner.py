#!/usr/bin/env python3
"""S20c: Official-eval-aligned fixed-window Level-3 runner.
Uses prepare_openvla_image + official postprocess + env.check_success().
Based on v4_run_eval_openvla.py preprocessing path."""
import os, sys, argparse, json, csv, time
import numpy as np
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True, choices=['ketchup','tomato_sauce','milk','butter','cream_cheese','salad_dressing','bbq_sauce','alphabet_soup','orange_juice','chocolate_pudding'])
ap.add_argument('--state_id', type=int, default=0)
ap.add_argument('--state_ids', default='', help='comma-separated state ids, e.g. 0,1,2')
ap.add_argument('--window_start', type=int, default=70)
ap.add_argument('--window_end', type=int, default=80)
ap.add_argument('--condition', choices=['clean','vis_pgd','random_linf'], default='clean')
ap.add_argument('--attack_seed', type=int, default=0)
ap.add_argument('--pgd_steps', type=int, default=20)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--random_control_seed', type=int, default=None)
ap.add_argument('--job_id', type=int, default=0)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--max_steps', type=int, default=400)
ap.add_argument('--save_video_dir', default='')
ap.add_argument('--gpu_pair', default='0,1')
args = ap.parse_args()

_eps_eff = args.eps_raw_pixels / 255.0

# Resolve state_ids
if args.state_ids:
    state_ids = [int(x) for x in args.state_ids.split(',')]
else:
    state_ids = [args.state_id]

# Import official image preprocessing
REVIEWED = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REVIEWED); sys.path.insert(0, os.path.join(REVIEWED, 'src'))
from gripper_attack.openvla_preprocess import prepare_openvla_image

# Env
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import gym; gym.logger.set_level(40)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import torch; from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
visible = [int(x) for x in _VISIBLE.split(',')] if _VISIBLE else [0]
render_gpu = visible[gpu_ids[1]] if len(visible) > gpu_ids[1] else gpu_ids[1]

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', trust_remote_code=True)
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

UNNORM_KEY = 'libero_object'
action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s_stats = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s_stats['q01'], dtype=np.float32)
HI = np.asarray(s_stats['q99'], dtype=np.float32)
MK = np.asarray(s_stats.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)

NUM_STEPS_WAIT = 10

def official_decode(tids):
    tids = np.asarray(tids, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP)-1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

def official_postprocess(action):
    """Official normalize_gripper + invert for LIBERO."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    action[..., -1] = -1.0 * action[..., -1]
    return np.clip(action, -1.0, 1.0).astype(np.float32)

def generate_action(img_pil, instruction):
    prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()
    inp = processor(prompt, img_pil)
    inp = {k: v.to(model_device, dtype=model_dtype if isinstance(v,torch.Tensor) and v.dtype==torch.float32 else v.dtype)
           if isinstance(v,torch.Tensor) else v for k,v in inp.items()}
    if 'attention_mask' in inp: del inp['attention_mask']
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    tids = gen.sequences[0, -action_dim:].cpu().numpy()
    action = official_decode(tids)
    env_action = official_postprocess(action)
    return env_action, tids

# Attack setup
if args.condition == 'vis_pgd':
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
    attacker_config = {
        'epsilon': _eps_eff, 'step_size': _eps_eff / max(args.pgd_steps,1) * 1.5,
        'num_steps': args.pgd_steps, 'random_start': True,
        'objective': 'prefix_locked_gripper_open_margin',
        'arm_preserve_weight': 0.5, 'gripper_margin': 5.0,
    }
    attacker = TokenPrefixPGDAttacker(
        model=model, processor=processor, config=attacker_config, seed=args.attack_seed,
        device='cuda:%d' % gpu_ids[0],
        preprocess_kwargs={'libero_preprocess_backend': 'official_pil_lanczos',
                          'center_crop': False, 'resize_size': 224,
                          'postprocess_gripper': True})
    attacker._freeze_model()

# LIBERO task setup
TASK_IDX = {'ketchup':4,'tomato_sauce':5,'milk':7,'butter':6,'cream_cheese':1,'salad_dressing':2,'bbq_sauce':3,'alphabet_soup':0,'orange_juice':9,'chocolate_pudding':8}
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()
task_obj = task_suite.get_task(TASK_IDX[args.task])
init_states = task_suite.get_task_init_states(TASK_IDX[args.task])
instruction = task_obj.language
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

os.makedirs(args.output_dir, exist_ok=True)

# Run each state_id
for sid in state_ids:
    if sid >= len(init_states):
        print('state_id %d out of range (max %d)' % (sid, len(init_states)-1))
        continue

    print('[%s] %s s%d w[%d,%d] %s seed=%d' % (
        datetime.now().strftime('%H:%M:%S'), args.task, sid,
        args.window_start, args.window_end, args.condition, args.attack_seed))

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=['agentview'],
        control_freq=20, render_gpu_device_id=render_gpu,
        horizon=args.max_steps + NUM_STEPS_WAIT)
    env.seed(0); env.reset(); env.set_init_state(init_states[sid])

    # num_steps_wait
    dummy_action = np.zeros(action_dim, dtype=np.float32)
    dummy_action[-1] = 1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy_action)

    trace_rows = []; decoded_open_bools = []; qpos_history = []
    success_ever = False; success_step = -1; done_ever = False; done_step = -1
    infra_status = 'ok'
    ws = args.window_start; we = args.window_end

    for step in range(args.max_steps):
        img = env.sim.render(256, 256, camera_name='agentview')
        # Official image preprocessing
        img_pil = prepare_openvla_image(img, libero_preprocess_backend='official_pil_lanczos', center_crop=True, resize_size=224)

        env_action, tids = generate_action(img_pil, instruction)
        raw_action = env_action.copy()
        raw_gripper = float(env_action[-1])

        # Qpos before step
        try:
            qpos = env.sim.data.qpos.copy()
            gripper_qpos = float(qpos[7]) if len(qpos) > 7 else 0.0
        except:
            gripper_qpos = 0.0
        qpos_history.append(gripper_qpos)

        in_window = ws <= step < we
        attack_this_step = in_window and args.condition not in ('clean',)

        pgd_applied = 0; attacks_applied = 0
        random_seed_str = ''; random_seed_mode = 'n/a'
        perturbation_space = 'none'

        if attack_this_step:
            if args.condition == 'vis_pgd':
                try:
                    result = attacker.attack(observation=img, instruction=instruction.lower(),
                                              target_action=None, unnorm_key=UNNORM_KEY)
                    adv_inputs = get_adv_inputs_from_attack_result(result)
                    adv_pv = adv_inputs['pixel_values'].to(device=model_device, dtype=model_dtype)
                    adv_ids = adv_inputs['input_ids'].to(model_device)
                    with torch.no_grad():
                        gen = model.generate(input_ids=adv_ids, pixel_values=adv_pv,
                                             max_new_tokens=action_dim, do_sample=False,
                                             return_dict_in_generate=True, output_scores=False)
                    adv_tids = gen.sequences[0, -action_dim:].cpu().numpy()
                    adv_action = official_decode(adv_tids)
                    env_action = official_postprocess(adv_action)
                    pgd_applied = 1; attacks_applied = 1
                    perturbation_space = 'processor_pixel_values_linf_official'
                except Exception as e:
                    infra_status = 'pgd_error: %s' % str(e)[:60]

            elif args.condition == 'random_linf':
                try:
                    prompt = 'In: What action should the robot take to %s?\nOut:' % instruction.lower()
                    inp = processor(prompt, img_pil)
                    pv_clean = inp['pixel_values'].to(device=model_device, dtype=model_dtype)
                    if args.random_control_seed is not None:
                        random_seed_str = str(args.random_control_seed)
                        random_seed_mode = 'explicit_random_control_seed'
                    else:
                        random_seed_str = str(int(args.attack_seed) + args.job_id)
                        random_seed_mode = 'legacy_attack_seed_plus_job_id'
                    rand_gen = torch.Generator(device=pv_clean.device)
                    rand_gen.manual_seed(int(random_seed_str))
                    noise = (2 * torch.rand(pv_clean.shape, device=pv_clean.device,
                                           dtype=pv_clean.dtype, generator=rand_gen) - 1) * _eps_eff
                    rand_pv = torch.clamp(pv_clean + noise, pv_clean - _eps_eff, pv_clean + _eps_eff)
                    inp_rand = {k: (rand_pv if k=='pixel_values' else v) for k,v in inp.items()}
                    if 'attention_mask' in inp_rand: del inp_rand['attention_mask']
                    with torch.no_grad():
                        gen2 = model.generate(**inp_rand, max_new_tokens=action_dim, do_sample=False,
                                              return_dict_in_generate=True, output_scores=True)
                    rand_tids = gen2.sequences[0, -action_dim:].cpu().numpy()
                    rand_action = official_decode(rand_tids)
                    env_action = official_postprocess(rand_action)
                    attacks_applied = 1
                    perturbation_space = 'random_linf_official'
                except Exception as e:
                    infra_status = 'random_error: %s' % str(e)[:60]

        is_open = int(env_action[-1] < -0.5)
        obs, reward, done, info = env.step(env_action)
        decoded_open_bools.append(is_open)

        # Official success check
        try: sc = bool(env.check_success())
        except: sc = False
        if sc and not success_ever: success_ever = True; success_step = step
        if done and not done_ever: done_ever = True; done_step = step

        # Video frame
        video_frame_path = ''
        if args.save_video_dir:
            os.makedirs(args.save_video_dir, exist_ok=True)
            video_frame_path = os.path.join(args.save_video_dir, 'frame_%06d.png' % step)
            img_pil.save(video_frame_path)

        trace_rows.append({
            'step': step, 'in_window': int(in_window), 'attack_this_step': int(attack_this_step),
            'env_action_6': round(float(env_action[-1]), 6), 'decoded_open_bool': is_open,
            'gripper_qpos': round(gripper_qpos, 8),
            'pgd_applied': pgd_applied, 'attacks_applied': attacks_applied,
            'random_seed_str': random_seed_str, 'random_seed_mode': random_seed_mode,
            'perturbation_space': perturbation_space,
            'reward': round(float(reward), 6) if isinstance(reward, (int,float)) else 0,
            'done_step_flag': int(done), 'success_check': int(sc),
            'condition': args.condition, 'task': args.task,
            'state_id': sid, 'attack_seed': args.attack_seed,
        })

        if done or sc:
            break

    env.close(); torch.cuda.empty_cache()

    # Metrics
    open_count = sum(1 for i in range(ws, min(we, len(decoded_open_bools))) if decoded_open_bools[i])
    streak = max_streak = 0
    for i in range(ws, min(we, len(decoded_open_bools))):
        if decoded_open_bools[i]: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    pre_qpos = np.array(qpos_history[:ws]) if ws > 0 else np.array([0.0])
    baseline_qpos = float(np.median(pre_qpos)) if len(pre_qpos) > 0 else 0.0
    post_start = we; post_end = min(len(qpos_history), we + 40)
    post_qpos = np.array(qpos_history[post_start:post_end]) if post_end > post_start else np.array([])
    qpos_pos_area = float(np.sum(np.maximum(post_qpos - baseline_qpos, 0))) if len(post_qpos) > 0 else 0.0
    qpos_neg_area = float(np.sum(np.maximum(baseline_qpos - post_qpos, 0))) if len(post_qpos) > 0 else 0.0

    safe_pair = '%s_s%d_w%d_%d_s20c_%s_seed%d' % (args.task, sid, ws, we, args.condition, args.attack_seed)
    summary = {
        'job_id': args.job_id, 'task': args.task, 'state_id': sid,
        'window_start': ws, 'window_end': we, 'condition': args.condition,
        'attack_seed': args.attack_seed,
        'n_steps': step + 1, 'max_steps': args.max_steps,
        'official_success_check': success_ever, 'success_step': success_step,
        'official_done': done_ever, 'done_step': done_step,
        'num_steps_wait': NUM_STEPS_WAIT,
        'runner': 'official_eval_aligned_l3',
        'image_preprocess': 'prepare_openvla_image_official_pil_lanczos_224',
        'action_postprocess': 'normalize_gripper_invert_official',
        'decoded_open_count': open_count, 'max_open_streak': max_streak,
        'qpos_pos_area': round(qpos_pos_area, 8), 'qpos_neg_area': round(qpos_neg_area, 8),
        'qpos_baseline': round(baseline_qpos, 8),
        'infra_status': infra_status,
        'video_dir': args.save_video_dir,
    }

    out_json = os.path.join(args.output_dir, 'summary_%s_%s_job%d.json' % (safe_pair, args.condition, args.job_id))
    with open(out_json, 'w') as f: json.dump(summary, f)

    if trace_rows:
        out_trace = os.path.join(args.output_dir, 'trace_%s_%s_job%d.csv' % (safe_pair, args.condition, args.job_id))
        with open(out_trace, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            for r in trace_rows: w.writerow(r)

    print('[%s] Done: state=%d steps=%d open=%d streak=%d success=%s@%d done=%s@%d infra=%s' % (
        datetime.now().strftime('%H:%M:%S'), sid, step+1, open_count, max_streak,
        success_ever, success_step, done_ever, done_step, infra_status))
