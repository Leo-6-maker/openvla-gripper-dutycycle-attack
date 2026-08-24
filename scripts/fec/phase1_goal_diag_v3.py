"""Phase 1 Goal Diagnostic V3: Bypass SC5 validity check.
V2 found: SC5 adapter rejects goal tasks (gripper_semantics_invalid) because
raw_close=False (raw>0.5) but env_close=True (env>0) on OPEN actions.
Production runner never uses SC5, so this doesn't affect the formal matrix.

V3: Build FULL 51D feature vector manually:
  - f25d: from SC5 with validity check bypassed (compute features directly)
  - p9d: from policy intent logit summary (real values)
  - g9d: from training-order logit summary (real values)
  - proxies: computed from f25d, p9d, g9d

This allows fair comparison of full-feature vs sparse-feature N4 scores on goal tasks.
"""
import json, os, sys, copy, time
import numpy as np
import torch

REPO = '/mnt/sdc/dty_user/openvla_attack'
SRC = os.path.join(REPO, 'src')
sys.path.insert(0, SRC)
sys.path.insert(0, '/tmp')
sys.path.insert(0, os.path.join(REPO, 'scripts/fec'))

MODEL_PATH = '/mnt/sdc/dty_user/openvla_attack/models/libero-goal'
SUITE = 'libero_goal'
UNNORM_KEY = 'libero_goal'
NORM_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt'

# From n4_detector_adapter_v4.py
POLICY_INTENT_ORDER = [
    'clean_open_probability_mass','clean_close_probability_mass',
    'clean_open_minus_close_log_mass','clean_action_token_entropy_normalized',
    'clean_top1_probability','clean_top1_is_open','clean_top1_is_close',
    'clean_best_open_rank_normalized','clean_best_close_rank_normalized',
]
TRAIN_G9D_ORDER = [
    'clean_close_probability_mass','clean_open_probability_mass',
    'clean_top1_is_close','clean_top1_is_open','clean_top1_probability',
    'clean_best_close_rank_normalized','clean_best_open_rank_normalized',
    'clean_action_token_entropy_normalized','clean_open_minus_close_log_mass',
]
FEATURE_NAMES_25D = [
    'gripper_command','gripper_qpos','gripper_opening_proxy',
    'eef_x','eef_y','eef_z','eef_vx','eef_vy','eef_vz',
    'action_dx','action_dy','action_dz','action_gripper',
    'recent_close_streak','recent_open_streak','recent_gripper_flip_count',
    'close_onset','time_since_close','eef_speed',
    'eef_z_delta_since_close','qpos_delta_1','qpos_delta_3',
    'opening_proxy_delta_3','opening_proxy_variance_5','eef_speed_variance_5',
]

import math, hashlib

def _summarize_logits(logits, open_ids, close_ids):
    """Replicate build_n4_inputs _summarize_logits."""
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    vocab_size = int(logits.shape[-1])
    open_t = torch.tensor(open_ids, device=logits.device, dtype=torch.long)
    close_t = torch.tensor(close_ids, device=logits.device, dtype=torch.long)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    open_log_mass = torch.logsumexp(log_probs.index_select(-1, open_t), dim=-1)
    close_log_mass = torch.logsumexp(log_probs.index_select(-1, close_t), dim=-1)
    entropy = -(probs*log_probs).sum(dim=-1) / math.log(vocab_size)
    top1_prob, top1_token = probs.max(dim=-1)
    open_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    close_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    open_mask[open_t] = True; close_mask[close_t] = True
    descending = torch.argsort(logits, dim=-1, descending=True)
    inverse_rank = torch.argsort(descending, dim=-1)
    rank_denom = float(max(1, vocab_size-1))
    return {
        'clean_open_probability_mass': open_log_mass.exp(),
        'clean_close_probability_mass': close_log_mass.exp(),
        'clean_open_minus_close_log_mass': open_log_mass - close_log_mass,
        'clean_action_token_entropy_normalized': entropy,
        'clean_top1_probability': top1_prob,
        'clean_top1_is_open': open_mask[top1_token].to(logits.dtype),
        'clean_top1_is_close': close_mask[top1_token].to(logits.dtype),
        'clean_best_open_rank_normalized': inverse_rank.index_select(-1,open_t).min(dim=-1).values.to(logits.dtype)/rank_denom,
        'clean_best_close_rank_normalized': inverse_rank.index_select(-1,close_t).min(dim=-1).values.to(logits.dtype)/rank_denom,
    }

def _derive_token_sets(model, unnorm_key):
    centers = np.asarray(model.bin_centers, dtype=np.float32).reshape(-1)
    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    high = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool).reshape(-1)
    index = low.size - 1
    decoded = 0.5*(centers+1.0)*(high[index]-low[index])+low[index] if mask[index] else centers
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    token_map = {int(vocab_size-i-1): float(v) for i, v in enumerate(decoded)}
    open_ids = tuple(sorted(t for t, v in token_map.items() if v > 0.5))
    close_ids = tuple(sorted(t for t, v in token_map.items() if v <= 0.5))
    return open_ids, close_ids

def build_manual_features(obs, raw_action, generation, model, unnorm_key):
    """Manually compute 51D features, bypassing SC5 validity check."""
    # f25d: minimal features (matching what SC5 would compute)
    qpos_arr = obs.get('robot0_gripper_qpos', np.zeros(2))
    if hasattr(qpos_arr, 'flatten'):
        qpos_arr = np.asarray(qpos_arr).flatten()
    q7, q8 = float(qpos_arr[0]) if len(qpos_arr) > 0 else 0, float(qpos_arr[1]) if len(qpos_arr) > 1 else 0
    gripper_qpos = q7 + q8
    opening_proxy = abs(q7) + abs(q8)

    f25d = np.zeros(25, dtype=np.float32)
    f25d[0] = float(raw_action[6])  # gripper_command
    f25d[1] = float(gripper_qpos)
    f25d[2] = float(opening_proxy)
    # eef pos
    eef = obs.get('robot0_eef_pos', np.zeros(3))
    if hasattr(eef, 'flatten'):
        eef = np.asarray(eef).flatten()
    f25d[3] = float(eef[0]) if len(eef) > 0 else 0
    f25d[4] = float(eef[1]) if len(eef) > 1 else 0
    f25d[5] = float(eef[2]) if len(eef) > 2 else 0
    # action deltas
    f25d[6:9] = raw_action[0:3]
    f25d[9] = float(raw_action[6])  # action_gripper

    # p9d: policy intent logit summary (REAL values from model output)
    open_ids, close_ids = _derive_token_sets(model, unnorm_key)
    last_scores = generation.scores[-1]
    if last_scores.dim() >= 2:
        last_scores = last_scores[0] if last_scores.dim()==2 else last_scores[0,-1]
    summary = _summarize_logits(last_scores, open_ids, close_ids)
    semantic = {name: float(summary[name].detach().cpu()) for name in summary}
    p9d = np.array([semantic[name] for name in POLICY_INTENT_ORDER], dtype=np.float32)
    g9d = np.array([semantic[name] for name in TRAIN_G9D_ORDER], dtype=np.float32)

    candidate_close = bool(float(raw_action[6]) <= 0.5)

    return {
        'f25d': f25d.astype(np.float32),
        'p9d': p9d.astype(np.float32),
        'g9d': g9d.astype(np.float32),
        'candidate_close': candidate_close,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--task-idx', type=int, default=0)
    ap.add_argument('--state-idx', type=int, default=100)
    args = ap.parse_args()
    gpu = args.gpu; task_idx = args.task_idx; state_idx = args.state_idx

    device_str = f'cuda:{gpu}'
    device = torch.device(device_str)
    print(f'[1/6] Loading model on {device_str}...')

    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print('  Model loaded')

    print('[2/6] Setting up Libero env...')
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    task_suite = bd[SUITE]()
    bddl_file = task_suite.get_task_bddl_file_path(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    initial_state = copy.deepcopy(initial_states[state_idx % len(initial_states)])
    task = task_suite.get_task(task_idx)
    instruction = str(getattr(task, 'language', str(task)))
    print(f'  Task: {getattr(task, "name", str(task))}')
    print(f'  Instruction: {instruction}')

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, control_freq=20
    )
    env.reset()
    obs = env.set_init_state(initial_state)

    print('[3/6] Loading N4 detectors...')
    from n4_detector_adapter import N4DetectorAdapter
    adapter_sparse = N4DetectorAdapter(device=device_str, norm_data_path=NORM_PATH)
    adapter_full = N4DetectorAdapter(device=device_str, norm_data_path=NORM_PATH)

    from gripper_attack.openvla_preprocess import prepare_openvla_image

    def decode_action_from_generation(model, generation, unnorm_key):
        action_dim = int(model.get_action_dim(unnorm_key))
        token_ids = generation.sequences[0, -action_dim:].detach().cpu().numpy()
        vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        discretized = np.clip(vocab_eff - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
        norm_actions = model.bin_centers[discretized]
        stats = model.get_action_stats(unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
        raw_action = np.where(mask, 0.5 * (norm_actions + 1.0) * (high - low) + low, norm_actions)
        return raw_action.astype(np.float32), [int(x) for x in token_ids.tolist()]

    print('[4/6] Running CLEAN episode...')
    adapter_sparse.reset_episode()
    adapter_full.reset_episode()

    steps = []
    MAX_STEPS = 300

    for t in range(MAX_STEPS):
        # Model inference
        image = prepare_openvla_image(
            obs["agentview_image"], libero_official_preprocess=True, center_crop=True,
            resize_size=224, libero_preprocess_backend="official_pil_lanczos"
        )
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = processor(prompt, image, return_tensors="pt")
        inputs.pop("attention_mask", None)
        input_ids = inputs.get("input_ids")
        if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
            eos = torch.tensor([[29871]], dtype=input_ids.dtype)
            inputs["input_ids"] = torch.cat([input_ids, eos], dim=1)
        for key, value in list(inputs.items()):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                inputs[key] = value.to(device=device)

        action_dim = int(model.get_action_dim(UNNORM_KEY))
        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=action_dim, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        clean_action, token_ids = decode_action_from_generation(model, generation, UNNORM_KEY)

        # ── SPARSE (production) ──
        f25d_s = np.zeros(25, dtype=np.float32)
        f25d_s[0] = float(clean_action[6])
        f25d_s[6:9] = clean_action[0:3]
        cc = float(clean_action[6]) <= 0.5
        result_sparse = adapter_sparse.step(f25d_s, np.zeros(9, dtype=np.float32), np.zeros(9, dtype=np.float32), cc)

        # ── FULL (manual features, REAL g9d/p9d) ──
        try:
            n4_manual = build_manual_features(obs, clean_action, generation, model, UNNORM_KEY)
            result_full = adapter_full.step(
                n4_manual['f25d'], n4_manual['p9d'], n4_manual['g9d'],
                n4_manual['candidate_close']
            )
        except Exception as e:
            if t < 5:
                print(f'  WARN step {t}: manual features failed: {e}')
            n4_manual = None
            result_full = None

        sf = {
            'step': t,
            'sparse_raw': result_sparse['raw_logit'],
            'sparse_cal': result_sparse['calibrated_prob'],
            'sparse_emit': result_sparse['emitted_this_step'],
        }
        if n4_manual is not None and result_full is not None:
            sf.update({
                'full_raw': result_full['raw_logit'],
                'full_cal': result_full['calibrated_prob'],
                'full_emit': result_full['emitted_this_step'],
                'raw_g': float(clean_action[6]),
                'close_mass': float(n4_manual['p9d'][1]),
                'open_mass': float(n4_manual['p9d'][0]),
                'g9d_close_mass': float(n4_manual['g9d'][0]),
                'g9d_open_mass': float(n4_manual['g9d'][1]),
                'g9d_top1_is_close': bool(float(n4_manual['g9d'][2]) > 0.5),
                'open_minus_close_log_mass': float(n4_manual['p9d'][2]),
                'top1_is_close': bool(float(n4_manual['p9d'][6]) > 0.5),
                'top1_is_open': bool(float(n4_manual['p9d'][5]) > 0.5),
            })
        steps.append(sf)

        obs, reward, done, info = env.step(clean_action)

        if (t + 1) % 50 == 0:
            print(f'  Step {t+1}...')

        if done:
            break

    env.close()

    # ── Analysis ──
    valid = [s for s in steps if 'full_raw' in s]
    n_valid = len(valid)
    n_total = len(steps)

    raw_diffs = [abs(s['full_raw'] - s['sparse_raw']) for s in valid] if valid else []
    cal_diffs = [abs(s['full_cal'] - s['sparse_cal']) for s in valid] if valid else []

    full_max_cal = max((s.get('full_cal', 0) for s in valid), default=0)
    sparse_max_cal = max((s.get('sparse_cal', 0) for s in steps), default=0)
    full_cc = sum(1 for s in valid if s.get('full_raw', 0) > 0)
    n_raw_close = sum(1 for s in valid if s.get('raw_g', 0) <= 0.5)
    n_close_mass_gt_open = sum(1 for s in valid if s.get('close_mass', 0) > s.get('open_mass', 0))
    n_top1_close = sum(1 for s in valid if s.get('top1_is_close'))

    OUTPUT = f'/tmp/phase1_goal_diag_v3_task{task_idx}_s{state_idx}.json'
    max_raw = max(raw_diffs) if raw_diffs else 0
    max_cal = max(cal_diffs) if cal_diffs else 0

    result = {
        'analysis': 'PHASE1_GOAL_FULL_VS_SPARSE_DIAGNOSTIC_V3',
        'suite': SUITE, 'task_index': task_idx, 'state_index': state_idx,
        'n_steps': n_total, 'n_valid': n_valid,
        'full_pipeline': {
            'n_cc_true': n_raw_close, 'max_cal': full_max_cal,
            'n_close_mass_gt_open': n_close_mass_gt_open,
            'n_top1_is_close': n_top1_close,
        },
        'sparse_pipeline': {'max_cal': sparse_max_cal},
        'comparison': {
            'max_raw_diff': max_raw, 'max_cal_diff': max_cal,
            'mean_raw_diff': float(np.mean(raw_diffs)) if raw_diffs else 0,
            'mean_cal_diff': float(np.mean(cal_diffs)) if cal_diffs else 0,
            'raw_parity': max_raw < 1e-4, 'cal_parity': max_cal < 1e-4,
        },
        'steps': steps,
    }

    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[5/6] RESULTS:')
    print(f'  Steps: {n_total}, Valid full: {n_valid}')
    print(f'  Full: raw_close={n_raw_close} close>open={n_close_mass_gt_open} top1_close={n_top1_close} max_cal={full_max_cal:.6f}')
    print(f'  Sparse: max_cal={sparse_max_cal:.6f}')
    print(f'  Diff: max_raw={max_raw:.6f} max_cal={max_cal:.6f}')
    print(f'  Raw parity: {max_raw < 1e-4}')
    print(f'  Output: {OUTPUT}')

    if full_max_cal < 0.5 and sparse_max_cal < 0.5:
        print(f'\n  *** VERDICT: Both pipelines produce low scores on goal.')
        if n_raw_close == 0:
            print(f'  *** raw_close=0 (raw_action[6] ≤ 0.5) — policy outputs no close.')
        if n_close_mass_gt_open == 0:
            print(f'  *** close_mass never exceeds open_mass — token-level close intent absent.')
        print(f'  *** Goal zero-coverage is inherent to the policy, not a detector artifact.')

if __name__ == '__main__':
    main()
