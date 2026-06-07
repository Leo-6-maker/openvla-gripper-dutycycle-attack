#!/usr/bin/env python3
"""Bounded PGD budget diagnostic on 11 disagreement rows.

Runs PGD3 (baseline), PGD10, PGD20 no-env on same frames.
No env.step during probe. No full rollout.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/run_active_probe_v1_pgd_budget_diagnostic.py \
    --gpu_pair 0,1 --shard 0 --shard_total 2
"""
import csv, os, sys, argparse
from datetime import datetime
import numpy as np
import torch

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _VISIBLE:
    print('FATAL: CUDA_VISIBLE_DEVICES must be set'); sys.exit(1)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'
UNNORM_KEY = 'libero_object'
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

ap = argparse.ArgumentParser()
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--shard_total', type=int, default=2)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--n_frames', type=int, default=3, help='frames per window (default 3 for speed)')
ap.add_argument('--skip_pgd20', action='store_true', help='skip slow PGD20')
args = ap.parse_args()

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()]
_render_gpu = _physical[0] if _physical else gpu_ids[0]
log('GPU: physical=%s logical=%s render=%d' % (_VISIBLE, args.gpu_pair, _render_gpu))

# ── Load 11 disagreement rows ───────────────────────────────────
with open(os.path.join(REPO, 'tables', 'active_probe_v1_disagreement_review_queue.csv')) as f:
    queue = list(csv.DictReader(f))

# Load full31 features for frame positions
with open(os.path.join(OUT_DIR, 'window_features_full31_merged.csv')) as f:
    full31_features = { (r['task_key'], r['state_id'],
                         int(r['window_start']), int(r['window_end'])): r
                       for r in csv.DictReader(f) }

# Load step features to know which frames were probed
step_features = []
for s in ['0', '1', '2']:
    sf = os.path.join(OUT_DIR, 'step_features_full31_shard%s.csv' % s)
    if os.path.exists(sf):
        with open(sf) as f:
            step_features.extend(list(csv.DictReader(f)))

# Index step features by (task, state, window_start, window_end, step)
step_by_window = {}
for r in step_features:
    k = (r['task_key'], r['state_id'], r['window_start'], r['window_end'], r['step'])
    step_by_window[k] = r

# Shard
total = args.shard_total
shard_size = (len(queue) + total - 1) // total
start = args.shard * shard_size
end = min(start + shard_size, len(queue))
my_queue = queue[start:end]
log('Shard %d/%d: %d rows [%d:%d]' % (args.shard, total, len(my_queue), start, end))

# ── Load model ──────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor
from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result

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

def get_env_gripper(action):
    a = normalize_gripper_action(action.copy(), binarize=True)
    a = invert_gripper_action(a)
    return float(a[-1])

def get_full_action(action):
    a = normalize_gripper_action(action.copy(), binarize=True)
    a = invert_gripper_action(a)
    return a

model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

# ── Create attackers for each budget ─────────────────────────────
_eps_eff = args.eps_raw_pixels / 255.0

def make_attacker(pgd_steps):
    cfg = {
        'epsilon': _eps_eff,
        'step_size': _eps_eff / max(pgd_steps, 1) * 1.5,
        'num_steps': pgd_steps,
        'random_start': True,
        'objective': 'prefix_locked_gripper_open_margin',
        'arm_preserve_weight': 0.5,
        'gripper_margin': 5.0,
    }
    att = TokenPrefixPGDAttacker(
        model=model, processor=processor, config=cfg, seed=0,
        device='cuda:%d' % gpu_ids[0],
        preprocess_kwargs={'postprocess_gripper': True})
    att._freeze_model()
    return att

attackers = {
    'pgd3': make_attacker(3),
    'pgd10': make_attacker(10),
}
if not args.skip_pgd20:
    attackers['pgd20'] = make_attacker(20)
budgets = list(attackers.keys())
log('Attackers ready: %s (PGD20 %s)' % (', '.join(budgets), 'SKIPPED' if args.skip_pgd20 else 'enabled'))

# ── Prompt ───────────────────────────────────────────────────────
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

def decode_gripper_from_inputs(inp):
    with torch.inference_mode():
        gen = model.generate(**inp, max_new_tokens=action_dim, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    tids = gen.sequences[0, -action_dim:].cpu().numpy()
    return get_env_gripper(decode_tokens_to_action(tids)), decode_tokens_to_action(tids)

def random_linf_perturb(pv, epsilon):
    noise = (2 * torch.rand_like(pv) - 1) * epsilon
    pv_perturbed = torch.clamp(pv + noise, pv - epsilon, pv + epsilon)
    return pv_perturbed

# TASK CFG
TASK_CFG = {
    'ketchup': {'task_id': 0}, 'butter': {'task_id': 1},
    'cream_cheese': {'task_id': 2}, 'salad_dressing': {'task_id': 3},
    'bbq_sauce': {'task_id': 4}, 'milk': {'task_id': 5},
    'alphabet_soup': {'task_id': 6}, 'tomato_sauce': {'task_id': 7},
    'orange_juice': {'task_id': 8},
}

# ── Process ──────────────────────────────────────────────────────
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

diagnostic_rows = []

for idx, q in enumerate(my_queue):
    task = q['task_key'].strip(); sid = q['state_id'].strip()
    ws = int(q['window_start']); we = int(q['window_end'])
    dtype = q['disagreement_type'].strip()
    log('[%d/%d] %s s%s [%d,%d] %s' % (idx+1, len(my_queue), task, sid, ws, we, dtype))

    cfg = TASK_CFG.get(task)
    if cfg is None: log('  SKIP: unknown task'); continue

    # Get frame positions from full31 step features
    window_key_prefix = (task, sid, str(ws), str(we))
    probe_steps = sorted([int(r['step']) for r in step_features
                         if (r['task_key'], r['state_id'], r['window_start'], r['window_end']) == window_key_prefix])
    if not probe_steps:
        # Fallback: evenly spaced
        window_len = we - ws + 1
        n_probe = min(args.n_frames, window_len)
        probe_steps = sorted(set([ws + int(i*(window_len-1)/(n_probe-1)) for i in range(n_probe)])) if n_probe > 1 else [ws]
    else:
        # Use same frames but subsample to n_frames
        if len(probe_steps) > args.n_frames:
            step = len(probe_steps) // args.n_frames
            probe_steps = [probe_steps[i] for i in range(0, len(probe_steps), step)][:args.n_frames]
    n_frames = len(probe_steps)
    log('  frames: %s' % probe_steps[:5] + ('...' if len(probe_steps) > 5 else ''))

    try:
        bm_dict = benchmark.get_benchmark_dict()
        task_suite = bm_dict['libero_object']()
        task_obj = task_suite.get_task(cfg['task_id'])
        initial_states = task_suite.get_task_init_states(cfg['task_id'])
        if int(sid) >= len(initial_states):
            log('  SKIP: state OOB'); continue
        instruction = str(task_obj.language) if hasattr(task_obj, 'language') and task_obj.language else task.replace('_', ' ')
        bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 has_renderer=False, has_offscreen_renderer=True,
                                 use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                                 render_gpu_device_id=_render_gpu)
        env.seed(0); obs = env.reset()
        env.sim.data.qvel[:] = 0; env.sim.forward()
        env.set_init_state(initial_states[int(sid)])
    except Exception as e:
        log('  SKIP: env error %s' % str(e)[:80]); continue

    # Step to window_start
    current_step = 0; done = False
    while not done and current_step < ws:
        img = obs['agentview_image']
        pil = Image.fromarray(img.astype(np.uint8))
        inputs = make_inputs(pil, instruction)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=False)
        tids = gen.sequences[0, -action_dim:].cpu().numpy()
        env_action = get_full_action(decode_tokens_to_action(tids))
        obs, reward, done, info = env.step(env_action)
        current_step += 1

    # Collect per-frame results for each budget
    results_by_budget = {b: {'grips': [], 'actions': [], 'l2_deltas': []} for b in ['clean', 'random', 'pgd3', 'pgd10', 'pgd20']}

    frame_idx = 0
    while not done and current_step <= we + 1:
        if current_step in probe_steps:
            img = obs['agentview_image']
            pil = Image.fromarray(img.astype(np.uint8))
            inputs = make_inputs(pil, instruction)
            clean_pv = inputs['pixel_values'].clone()
            clean_ids = inputs['input_ids'].clone()

            # 1) Clean decode
            clean_grip, clean_action = decode_gripper_from_inputs({'input_ids': clean_ids, 'pixel_values': clean_pv})
            results_by_budget['clean']['grips'].append(clean_grip)
            results_by_budget['clean']['actions'].append(clean_action)

            # 2) Random Linf
            rand_pv = random_linf_perturb(clean_pv.clone(), _eps_eff)
            rand_grip, rand_action = decode_gripper_from_inputs({'input_ids': clean_ids, 'pixel_values': rand_pv})
            results_by_budget['random']['grips'].append(rand_grip)
            results_by_budget['random']['actions'].append(rand_action)
            rand_l2 = float(np.linalg.norm((rand_action - clean_action).reshape(-1)))
            results_by_budget['random']['l2_deltas'].append(rand_l2)

            # 3) PGD3 decode
            try:
                result3 = attackers['pgd3'].attack(observation=pil, instruction=instruction.lower(),
                                                    target_action=clean_action, unnorm_key=UNNORM_KEY)
                adv3 = get_adv_inputs_from_attack_result(result3)
                pgd3_grip, pgd3_action = decode_gripper_from_inputs(
                    {'input_ids': adv3['input_ids'].to(model_device),
                     'pixel_values': adv3['pixel_values'].to(device=model_device, dtype=model_dtype)})
                pgd3_l2 = float(np.linalg.norm((pgd3_action - clean_action).reshape(-1)))
            except Exception as e:
                pgd3_grip = float('nan'); pgd3_action = clean_action; pgd3_l2 = float('nan')
            results_by_budget['pgd3']['grips'].append(pgd3_grip)
            results_by_budget['pgd3']['actions'].append(pgd3_action)
            results_by_budget['pgd3']['l2_deltas'].append(pgd3_l2)

            # 4) PGD10 decode
            try:
                result10 = attackers['pgd10'].attack(observation=pil, instruction=instruction.lower(),
                                                      target_action=clean_action, unnorm_key=UNNORM_KEY)
                adv10 = get_adv_inputs_from_attack_result(result10)
                pgd10_grip, pgd10_action = decode_gripper_from_inputs(
                    {'input_ids': adv10['input_ids'].to(model_device),
                     'pixel_values': adv10['pixel_values'].to(device=model_device, dtype=model_dtype)})
                pgd10_l2 = float(np.linalg.norm((pgd10_action - clean_action).reshape(-1)))
            except Exception as e:
                pgd10_grip = float('nan'); pgd10_action = clean_action; pgd10_l2 = float('nan')
            results_by_budget['pgd10']['grips'].append(pgd10_grip)
            results_by_budget['pgd10']['actions'].append(pgd10_action)
            results_by_budget['pgd10']['l2_deltas'].append(pgd10_l2)

            # 5) PGD20 decode (optional)
            if not args.skip_pgd20:
                try:
                    result20 = attackers['pgd20'].attack(observation=pil, instruction=instruction.lower(),
                                                          target_action=clean_action, unnorm_key=UNNORM_KEY)
                    adv20 = get_adv_inputs_from_attack_result(result20)
                    pgd20_grip, pgd20_action = decode_gripper_from_inputs(
                        {'input_ids': adv20['input_ids'].to(model_device),
                         'pixel_values': adv20['pixel_values'].to(device=model_device, dtype=model_dtype)})
                    pgd20_l2 = float(np.linalg.norm((pgd20_action - clean_action).reshape(-1)))
                except Exception as e:
                    pgd20_grip = float('nan'); pgd20_action = clean_action; pgd20_l2 = float('nan')
                results_by_budget['pgd20']['grips'].append(pgd20_grip)
                results_by_budget['pgd20']['actions'].append(pgd20_action)
                results_by_budget['pgd20']['l2_deltas'].append(pgd20_l2)

            frame_idx += 1

        # Step env with CLEAN action only
        if current_step <= we:
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                     return_dict_in_generate=True, output_scores=False)
            tids = gen.sequences[0, -action_dim:].cpu().numpy()
            env_action = get_full_action(decode_tokens_to_action(tids))
            obs, reward, done, info = env.step(env_action)
        current_step += 1

    env.close()

    if not results_by_budget['clean']['grips']:
        log('  SKIP: no data'); continue

    # ── Aggregate per-budget ────────────────────────────────────
    def compute_stats(grips):
        opens = [1 if g > 0 else 0 for g in grips if not np.isnan(g)]
        open_count = sum(opens)
        open_rate = round(open_count / max(len(opens), 1), 4)
        streak = 0; max_streak = 0
        for o in opens:
            if o: streak += 1; max_streak = max(max_streak, streak)
            else: streak = 0
        return open_count, open_rate, max_streak

    c_cnt, c_rate, c_streak = compute_stats(results_by_budget['clean']['grips'])
    r_cnt, r_rate, r_streak = compute_stats(results_by_budget['random']['grips'])
    p3_cnt, p3_rate, p3_streak = compute_stats(results_by_budget['pgd3']['grips'])
    p10_cnt, p10_rate, p10_streak = compute_stats(results_by_budget['pgd10']['grips'])
    p20_cnt, p20_rate, p20_streak = compute_stats(results_by_budget['pgd20']['grips'])

    # gripper selectivity: targeted arm delta vs random arm delta
    def compute_selectivity(adv_grips, adv_actions, rand_actions, clean_actions):
        if not adv_actions or not rand_actions: return None, None
        adv_l2s = [float(np.linalg.norm((a - c).reshape(-1))) for a, c in zip(adv_actions, clean_actions)]
        rand_l2s = [float(np.linalg.norm((a - c).reshape(-1))) for a, c in zip(rand_actions, clean_actions)]
        mean_adv_l2 = np.mean(adv_l2s) if adv_l2s else float('nan')
        mean_rand_l2 = np.mean(rand_l2s) if rand_l2s else float('nan')
        # selectivity = (targeted gripper delta) / (targeted arm L2 + eps)
        # But simpler: compare mean L2 for targeted vs random
        # If targeted L2 is similar to random L2 but gripper changes more -> selective
        grip_delta_adv = abs(np.mean([g for g in adv_grips if not np.isnan(g)]) - np.mean([g for g in results_by_budget['clean']['grips'] if not np.isnan(g)])) if adv_grips else 0
        grip_delta_rand = abs(np.mean([g for g in results_by_budget['random']['grips'] if not np.isnan(g)]) - np.mean([g for g in results_by_budget['clean']['grips'] if not np.isnan(g)]))
        selectivity = grip_delta_adv / max(mean_adv_l2, 1e-9) - grip_delta_rand / max(mean_rand_l2, 1e-9)
        return round(selectivity, 4), round(mean_adv_l2, 6)

    p3_sel, p3_arm_l2 = compute_selectivity(
        results_by_budget['pgd3']['grips'], results_by_budget['pgd3']['actions'],
        results_by_budget['random']['actions'], results_by_budget['clean']['actions'])
    p10_sel, p10_arm_l2 = compute_selectivity(
        results_by_budget['pgd10']['grips'], results_by_budget['pgd10']['actions'],
        results_by_budget['random']['actions'], results_by_budget['clean']['actions'])
    p20_sel, p20_arm_l2 = compute_selectivity(
        results_by_budget['pgd20']['grips'], results_by_budget['pgd20']['actions'],
        results_by_budget['random']['actions'], results_by_budget['clean']['actions'])

    ceiling = 1 if c_rate >= 0.8 else 0

    # ── Load VIS label data ──────────────────────────────────────
    with open('/data/liuyu/outputs/shared_detector_v25_inputs_20260606/object_phase_response_labels_v2.csv') as f:
        labels = { (r['task_key'].strip(), r['state_id'].strip(),
                    int(r['window_start']), int(r['window_end'])): r
                  for r in csv.DictReader(f) }
    lbl = labels.get((task, sid, ws, we), {})
    vis_open = int(lbl.get('vis_open_count', 0) or 0)
    batch = lbl.get('source_batch', '?').strip()

    row = {
        'candidate_id': '%s_s%s_w%d_%d' % (task, sid, ws, we),
        'task_key': task, 'state_id': sid,
        'window_start': str(ws), 'window_end': str(we),
        'label_type': lbl.get('label_status', '?'),
        'disagreement_type': dtype,
        'n_frames': str(n_frames),
        'source_batch': batch,
        # Clean
        'clean_open_count': str(c_cnt), 'clean_open_rate': str(c_rate),
        # Random
        'random_open_count': str(r_cnt), 'random_open_rate': str(r_rate),
        # PGD3
        'targeted_open_count_pgd3': str(p3_cnt), 'targeted_open_rate_pgd3': str(p3_rate),
        'targeted_longest_streak_pgd3': str(p3_streak),
        'targeted_minus_clean_pgd3': str(p3_cnt - c_cnt),
        'targeted_minus_random_pgd3': str(p3_cnt - r_cnt),
        # PGD10
        'targeted_open_count_pgd10': str(p10_cnt), 'targeted_open_rate_pgd10': str(p10_rate),
        'targeted_longest_streak_pgd10': str(p10_streak),
        'targeted_minus_clean_pgd10': str(p10_cnt - c_cnt),
        'targeted_minus_random_pgd10': str(p10_cnt - r_cnt),
        # PGD20
        'targeted_open_count_pgd20': str(p20_cnt), 'targeted_open_rate_pgd20': str(p20_rate),
        'targeted_longest_streak_pgd20': str(p20_streak),
        'targeted_minus_clean_pgd20': str(p20_cnt - c_cnt),
        'targeted_minus_random_pgd20': str(p20_cnt - r_cnt),
        # Selectivity
        'gripper_selectivity_pgd3': str(p3_sel) if p3_sel is not None else 'nan',
        'gripper_selectivity_pgd10': str(p10_sel) if p10_sel is not None else 'nan',
        'gripper_selectivity_pgd20': str(p20_sel) if p20_sel is not None else 'nan',
        'arm_l2_pgd3': str(p3_arm_l2) if p3_arm_l2 is not None else 'nan',
        'arm_l2_pgd10': str(p10_arm_l2) if p10_arm_l2 is not None else 'nan',
        'arm_l2_pgd20': str(p20_arm_l2) if p20_arm_l2 is not None else 'nan',
        # Ceiling
        'ceiling_flag': str(ceiling),
        # VIS
        'vis_trace_open_count': str(vis_open),
        'vis_pgd_budget': lbl.get('source_batch', '?'),
        # Convention
        'open_convention_match': 'YES (both decoded +1=OPEN after normalize+invert)',
        'window_alignment_status': 'OK (same ws,we)',
        # Diagnosis (to be filled after all results)
        'diagnosis': '',
    }
    diagnostic_rows.append(row)

    log('  PGD3: open=%d rate=%.2f t-c=%d | PGD10: open=%d rate=%.2f t-c=%d | PGD20: open=%d rate=%.2f t-c=%d | VIS open=%d' % (
        p3_cnt, p3_rate, p3_cnt - c_cnt, p10_cnt, p10_rate, p10_cnt - c_cnt, p20_cnt, p20_rate, p20_cnt - c_cnt, vis_open))

# ── Write ────────────────────────────────────────────────────────
tag = 'pgd_budget_shard%d' % args.shard
if diagnostic_rows:
    DIA_CSV = os.path.join(OUT_DIR, 'pgd_budget_diagnostic_%s.csv' % tag)
    with open(DIA_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(diagnostic_rows[0].keys()))
        w.writeheader(); w.writerows(diagnostic_rows)
    log('Wrote %d rows to %s' % (len(diagnostic_rows), DIA_CSV))

log('DONE')
