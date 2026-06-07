#!/usr/bin/env python3
"""Clean trace recovery: run clean rollouts for missing windows, extract online features.

NO VIS PGD. NO random. NO oracle. NO detector training.
Only clean forward passes + online-legal feature extraction.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_clean_trace_recovery.py \
    --gpu_pair 0,1 --shard 0 --shard_total 3
"""
import csv, os, sys, argparse, json
from datetime import datetime
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
from gripper_attack.openvla_libero_exec_spec import env_gripper_is_open, env_gripper_is_close
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_DIR = '/data/liuyu/outputs/clean_trace_recovery_20260607'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--shard_total', type=int, default=3)
ap.add_argument('--max_steps', type=int, default=320)
args = ap.parse_args()

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]

# GPU swap for pair 0,1 (GPU1 primary)
if _physical and _physical == [0, 1]:
    gpu_ids = [1, 0]
    _render_gpu = 1
    log('GPU swap: using GPU1 as primary')

log('GPU: physical=%s logical=%s render=%d' % (_VISIBLE, ','.join(map(str, gpu_ids)), _render_gpu))

# ── Identify missing windows ─────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    all_labels = list(csv.DictReader(f))

# Check which have clean traces already
EXISTING_TRACE_DIR = '/data/liuyu/outputs/proprionostep_shadow_calib_20260607'
def has_trace(task, sid):
    import glob
    return bool(glob.glob(os.path.join(EXISTING_TRACE_DIR, 'vis_%s_s%s_clean_*_trace.csv' % (task, sid))))

missing = []
for r in all_labels:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    if not has_trace(task, sid):
        missing.append(r)
        log('  MISSING: %s s%s [%s,%s]' % (task, sid, r['window_start'], r['window_end']))

log('Total missing: %d windows' % len(missing))

# Shard
total = args.shard_total
shard_size = (len(missing) + total - 1) // total
start = args.shard * shard_size
end = min(start + shard_size, len(missing))
my_windows = missing[start:end]
log('Shard %d/%d: %d windows [%d:%d]' % (args.shard, total, len(my_windows), start, end))

# ── Load model ──────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor

log('Loading model...')
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto',
    max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
action_dim = int(model.get_action_dim(UNNORM_KEY))
log('Model loaded, action_dim=%d' % action_dim)

VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC_NP = np.asarray(model.bin_centers, dtype=np.float32)
s = model.get_action_stats(UNNORM_KEY)
LO = np.asarray(s['q01'], dtype=np.float32)
HI = np.asarray(s['q99'], dtype=np.float32)
MK = np.asarray(s.get('mask', np.ones_like(LO, dtype=bool)), dtype=bool)
UNNORM_KEY = 'libero_object'
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

def decode_tokens_to_action(tids_1d):
    tids = np.asarray(tids_1d, dtype=np.int64)
    disc = np.clip(VS - tids - 1, 0, len(BC_NP) - 1)
    return np.where(MK, 0.5*(BC_NP[disc]+1)*(HI-LO)+LO, BC_NP[disc]).astype(np.float32)

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

def prompt_fn(text):
    return 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: What action should the robot take to %s? ASSISTANT:' % text

def make_inputs(pil_image, instruction_text):
    text = prompt_fn(instruction_text.lower())
    inp = processor(text, pil_image, return_tensors='pt')
    for k, v in list(inp.items()):
        if torch.is_floating_point(v):
            inp[k] = v.to(device=model_device, dtype=model_dtype)
        else:
            inp[k] = v.to(model_device)
    if not torch.all(inp['input_ids'][:, -1] == 29871):
        inp['input_ids'] = torch.cat((inp['input_ids'],
            torch.tensor([[29871]], dtype=torch.long, device=model_device)), dim=1)
    return inp

# TASK CFG
TASK_CFG = {
    'ketchup': {'task_id': 0}, 'butter': {'task_id': 1},
    'cream_cheese': {'task_id': 2}, 'salad_dressing': {'task_id': 3},
    'bbq_sauce': {'task_id': 4}, 'milk': {'task_id': 5},
    'alphabet_soup': {'task_id': 6}, 'tomato_sauce': {'task_id': 7},
    'orange_juice': {'task_id': 8},
}

# ── Extract gripper logits (online-legal) ────────────────────────
def extract_gripper_logits(full_ids, pixel_values):
    """Extract gripper-dimension logits from a clean forward pass."""
    with torch.inference_mode():
        out = model(input_ids=full_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
    logits = out.logits[0, -1, :]  # [vocab_size] — next-token logits
    return logits

# ── Process ──────────────────────────────────────────────────────
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MANIFEST_ROWS = []
WINDOW_FEATURE_ROWS = []
STEP_FEATURE_ROWS = []
UNNORM_KEY = 'libero_object'

for idx, r in enumerate(my_windows):
    task = r['task_key'].strip(); sid = int(r['state_id'])
    ws = int(r['window_start']); we = int(r['window_end'])
    label_status = r.get('label_status', '?').strip()
    taxonomy = r.get('taxonomy', '?').strip()
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
    log('[%d/%d] %s (%s)' % (idx+1, len(my_windows), cid, label_status))

    cfg = TASK_CFG.get(task)
    if cfg is None:
        log('  SKIP: unknown task')
        MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                              'window_start': str(ws), 'window_end': str(we),
                              'status': 'SKIP_UNKNOWN_TASK', 'error': 'task not in TASK_CFG'})
        continue

    try:
        bm_dict = benchmark.get_benchmark_dict()
        task_suite = bm_dict['libero_object']()
        task_obj = task_suite.get_task(cfg['task_id'])
        initial_states = task_suite.get_task_init_states(cfg['task_id'])
        if sid >= len(initial_states):
            log('  SKIP: state OOB (max=%d)' % len(initial_states))
            MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                                  'window_start': str(ws), 'window_end': str(we),
                                  'status': 'SKIP_STATE_OOB', 'error': 'state_id >= n_initial_states'})
            continue
        instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else task.replace('_', ' ')
        bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 has_renderer=False, has_offscreen_renderer=True,
                                 use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                                 render_gpu_device_id=_render_gpu)
        env.seed(0); obs = env.reset()
        env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(initial_states[sid])
    except Exception as e:
        log('  INFRA: env init error: %s' % str(e)[:80])
        MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                              'window_start': str(ws), 'window_end': str(we),
                              'status': 'INFRA_ENV_INIT', 'error': str(e)[:120]})
        continue

    # ── Run clean rollout ────────────────────────────────────────
    trace_rows = []
    current_step = 0; done = False; step_error = ''
    try:
        while not done and current_step < min(max(we + 5, args.max_steps), 400):
            img = obs['agentview_image']
            pil = Image.fromarray(img.astype(np.uint8))
            inputs = make_inputs(pil, instruction)
            clean_pv = inputs['pixel_values']; clean_ids = inputs['input_ids']

            # Decode action
            with torch.inference_mode():
                gen = model.generate(input_ids=clean_ids, pixel_values=clean_pv,
                                     max_new_tokens=action_dim, do_sample=False,
                                     return_dict_in_generate=True, output_scores=False)
            tids = gen.sequences[0, -action_dim:].cpu().numpy()
            action = decode_tokens_to_action(tids)
            raw_gripper = float(action[-1])
            env_action = normalize_gripper_action(action.copy(), binarize=True)
            env_action = invert_gripper_action(env_action)
            env_gripper = float(env_action[-1])

            # Gripper qpos
            qpos = env.sim.data.qpos
            gripper_qpos = float((qpos[-2] + qpos[-1]) / 2.0)

            # EEF position
            eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_center')]

            # In-window flag
            in_window = 1 if ws <= current_step <= we else 0

            trace_row = {
                'step': str(current_step),
                'in_window': str(in_window),
                'raw_gripper': str(round(raw_gripper, 6)),
                'env_gripper': str(round(env_gripper, 1)),
                'gripper_qpos': str(round(gripper_qpos, 6)),
                'eef_x': str(round(float(eef_pos[0]), 6)),
                'eef_y': str(round(float(eef_pos[1]), 6)),
                'eef_z': str(round(float(eef_pos[2]), 6)),
                'arm_action_0': str(round(float(env_action[0]), 6)),
                'arm_action_1': str(round(float(env_action[1]), 6)),
                'arm_action_2': str(round(float(env_action[2]), 6)),
                'arm_action_3': str(round(float(env_action[3]), 6)),
                'arm_action_4': str(round(float(env_action[4]), 6)),
                'arm_action_5': str(round(float(env_action[5]), 6)),
                'done': str(int(done)),
            }
            trace_rows.append(trace_row)

            obs, reward, done, info = env.step(env_action)
            current_step += 1
    except Exception as e:
        step_error = str(e)[:120]
        log('  INFRA: step error at step %d: %s' % (current_step, step_error))

    env.close()

    if step_error and len(trace_rows) < ws + 1:
        MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                              'window_start': str(ws), 'window_end': str(we),
                              'status': 'CLEAN_FAIL', 'error': step_error,
                              'steps_recovered': str(len(trace_rows))})
        continue

    # ── Extract window-level features ────────────────────────────
    window_rows = [r2 for r2 in trace_rows if int(r2['in_window']) == 1]
    pre_rows = [r2 for r2 in trace_rows if int(r2['step']) < ws][-20:]

    if len(window_rows) < 2:
        MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                              'window_start': str(ws), 'window_end': str(we),
                              'status': 'CLEAN_FAIL', 'error': 'window too short (%d frames)' % len(window_rows),
                              'steps_recovered': str(len(trace_rows))})
        continue

    def safe_f(v, default=0.0):
        try: return float(v)
        except: return default

    # Gripper qpos
    qpos_vals = np.array([safe_f(r['gripper_qpos']) for r in window_rows])
    # Gripper action
    env_grip_vals = np.array([safe_f(r['env_gripper']) for r in window_rows])
    raw_grip_vals = np.array([safe_f(r['raw_gripper']) for r in window_rows])
    # EEF
    eef_xs = np.array([safe_f(r['eef_x']) for r in window_rows])
    eef_ys = np.array([safe_f(r['eef_y']) for r in window_rows])
    eef_zs = np.array([safe_f(r['eef_z']) for r in window_rows])
    # Pre-window
    pre_qpos = np.array([safe_f(r['gripper_qpos']) for r in pre_rows]) if pre_rows else qpos_vals[:1]
    pre_grip = np.array([safe_f(r['env_gripper']) for r in pre_rows]) if pre_rows else env_grip_vals[:1]

    n = len(window_rows)
    n_total = len(trace_rows)

    # Gripper qpos features
    gripper_qpos_mean = float(np.mean(qpos_vals))
    gripper_qpos_std = float(np.std(qpos_vals))
    gripper_qpos_min = float(np.min(qpos_vals))
    gripper_qpos_max = float(np.max(qpos_vals))
    gripper_qpos_at_start = float(qpos_vals[0])
    gripper_qpos_range = float(np.max(qpos_vals) - np.min(qpos_vals))
    gripper_is_closed = 1.0 if gripper_qpos_mean < 0.03 else 0.0
    gripper_is_open = 1.0 if gripper_qpos_mean > 0.035 else 0.0

    # Gripper action
    grip_open_count = int(np.sum([env_gripper_is_open(v) for v in env_grip_vals]))
    grip_close_count = int(np.sum([env_gripper_is_close(v) for v in env_grip_vals]))
    grip_open_rate = float(grip_open_count / max(n, 1))
    grip_action_mean = float(np.mean(env_grip_vals))
    grip_action_std = float(np.std(env_grip_vals))
    grip_action_switches = int(np.sum(np.abs(np.diff(np.sign(env_grip_vals))) > 0))
    streak = 0; max_streak = 0
    for g in env_grip_vals:
        if env_gripper_is_open(g): streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    clean_longest_open_streak = int(max_streak)

    # Raw gripper
    raw_grip_mean = float(np.mean(raw_grip_vals))
    raw_grip_std = float(np.std(raw_grip_vals))

    # EEF
    eef_displacement = float(np.linalg.norm([eef_xs[-1]-eef_xs[0], eef_ys[-1]-eef_ys[0], eef_zs[-1]-eef_zs[0]]))
    eef_velocity_mean = float(np.mean(np.sqrt(np.diff(eef_xs)**2 + np.diff(eef_ys)**2 + np.diff(eef_zs)**2))) if n > 1 else 0.0
    eef_z_mean = float(np.mean(eef_zs))
    eef_z_std = float(np.std(eef_zs))
    eef_z_trend = float(eef_zs[-1] - eef_zs[0]) if n > 1 else 0.0

    # Temporal
    qpos_delta_from_pre = float(np.mean(qpos_vals) - np.mean(pre_qpos))
    grip_action_delta_from_pre = float(np.mean(env_grip_vals) - np.mean(pre_grip))

    # Window position
    window_start_frac = float(ws / max(n_total, 1))
    window_center_frac = float((ws + we) / 2 / max(n_total, 1))
    window_len_steps = we - ws + 1
    window_len_frac = float(window_len_steps / max(n_total, 1))
    steps_remaining = n_total - we
    step_at_start = ws

    # Arm action statistics
    arm_actions = np.array([[safe_f(r['arm_action_%d' % i]) for r in window_rows] for i in range(6)])
    arm_displacement = float(np.linalg.norm(arm_actions[:, -1] - arm_actions[:, 0])) if n > 1 else 0.0
    arm_mean_velocity = float(np.mean([np.linalg.norm(arm_actions[:, i+1] - arm_actions[:, i])
                                       for i in range(n-1)])) if n > 1 else 0.0

    wf = {
        'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
        'window_start': str(ws), 'window_end': str(we),
        'label_status': label_status, 'taxonomy': taxonomy,
        'provenance': 'clean_trace_recovery_20260607_shard%d' % args.shard,
        'trace_available': 'yes',
        'window_len_steps': str(window_len_steps),
        'n_trace_steps': str(n_total),
        'n_window_frames': str(n),
        # Gripper qpos
        'gripper_qpos_mean': str(round(gripper_qpos_mean, 6)),
        'gripper_qpos_std': str(round(gripper_qpos_std, 6)),
        'gripper_qpos_min': str(round(gripper_qpos_min, 6)),
        'gripper_qpos_max': str(round(gripper_qpos_max, 6)),
        'gripper_qpos_at_start': str(round(gripper_qpos_at_start, 6)),
        'gripper_qpos_range': str(round(gripper_qpos_range, 6)),
        'gripper_is_closed': str(round(gripper_is_closed, 4)),
        'gripper_is_open': str(round(gripper_is_open, 4)),
        # Gripper action
        'grip_open_count': str(grip_open_count),
        'grip_close_count': str(grip_close_count),
        'grip_open_rate': str(round(grip_open_rate, 4)),
        'grip_action_mean': str(round(grip_action_mean, 4)),
        'grip_action_std': str(round(grip_action_std, 4)),
        'grip_action_switches': str(grip_action_switches),
        'clean_longest_open_streak': str(clean_longest_open_streak),
        # Raw gripper
        'raw_grip_mean': str(round(raw_grip_mean, 6)),
        'raw_grip_std': str(round(raw_grip_std, 6)),
        # EEF
        'eef_displacement': str(round(eef_displacement, 6)),
        'eef_velocity_mean': str(round(eef_velocity_mean, 6)),
        'eef_z_mean': str(round(eef_z_mean, 4)),
        'eef_z_std': str(round(eef_z_std, 4)),
        'eef_z_trend': str(round(eef_z_trend, 4)),
        # Arm action
        'arm_displacement': str(round(arm_displacement, 6)),
        'arm_mean_velocity': str(round(arm_mean_velocity, 6)),
        # Temporal
        'qpos_delta_from_pre': str(round(qpos_delta_from_pre, 6)),
        'grip_action_delta_from_pre': str(round(grip_action_delta_from_pre, 4)),
        # Position
        'window_start_frac': str(round(window_start_frac, 4)),
        'window_center_frac': str(round(window_center_frac, 4)),
        'window_len_frac': str(round(window_len_frac, 4)),
        'step_at_start': str(step_at_start),
        'steps_remaining': str(steps_remaining),
        # VIS labels (for audit only, NOT features)
        'vis_open_count': r.get('vis_open_count', ''),
        'label_physical_response': r.get('label_physical_response', ''),
        'qpos_label': r.get('qpos_label', ''),
    }
    WINDOW_FEATURE_ROWS.append(wf)

    # Per-step rows
    for tr in trace_rows:
        tr['candidate_id'] = cid; tr['task_key'] = task; tr['state_id'] = str(sid)
        tr['window_start'] = str(ws); tr['window_end'] = str(we)
        tr['label_status'] = label_status; tr['taxonomy'] = taxonomy
        tr['provenance'] = 'clean_trace_recovery_20260607_shard%d' % args.shard
        STEP_FEATURE_ROWS.append(tr)

    # Manifest
    MANIFEST_ROWS.append({'candidate_id': cid, 'task_key': task, 'state_id': str(sid),
                          'window_start': str(ws), 'window_end': str(we),
                          'status': 'CLEAN_OK', 'error': '',
                          'steps_recovered': str(n_total),
                          'n_window_frames': str(n)})
    log('  OK: %d steps, %d window frames' % (n_total, n))

# ── Write outputs ────────────────────────────────────────────────
tag = 'shard%d' % args.shard
if MANIFEST_ROWS:
    with open(os.path.join(OUT_DIR, 'manifest_%s.csv' % tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_ROWS[0].keys()))
        w.writeheader(); w.writerows(MANIFEST_ROWS)
    log('Wrote %d manifest rows' % len(MANIFEST_ROWS))

if WINDOW_FEATURE_ROWS:
    with open(os.path.join(OUT_DIR, 'window_features_%s.csv' % tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(WINDOW_FEATURE_ROWS[0].keys()))
        w.writeheader(); w.writerows(WINDOW_FEATURE_ROWS)
    log('Wrote %d window feature rows' % len(WINDOW_FEATURE_ROWS))

if STEP_FEATURE_ROWS:
    with open(os.path.join(OUT_DIR, 'step_features_%s.csv' % tag), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(STEP_FEATURE_ROWS[0].keys()))
        w.writeheader(); w.writerows(STEP_FEATURE_ROWS)
    log('Wrote %d step feature rows' % len(STEP_FEATURE_ROWS))

log('DONE')
