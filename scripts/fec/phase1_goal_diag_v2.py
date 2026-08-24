"""Phase 1 Goal Diagnostic V2: Shadow-observer wrapper reusing production runner infra.
Runs ONE CLEAN episode with the production pipeline, but ALSO calls build_n4_inputs()
as a shadow observer to compare full-feature vs sparse-feature N4 detector scores.

Reuses run_gpu_smoke.py's model loading, env setup, image preprocessing, and action decoding.
The N4 detector is run TWICE per step: once with sparse features (production), once with full features (shadow).
"""
import json, os, sys, copy, time, hashlib
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
TASK_IDX = 0
STATE_IDX = 100

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--task-idx', type=int, default=TASK_IDX)
    ap.add_argument('--state-idx', type=int, default=STATE_IDX)
    args = ap.parse_args()
    gpu = args.gpu; task_idx = args.task_idx; state_idx = args.state_idx

    device_str = f'cuda:{gpu}'
    device = torch.device(device_str)
    print(f'[1/6] Loading model on {device_str}...')

    # ── Reuse production model loading ──
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print('  Model loaded')

    # ── Reuse production env setup ──
    print('[2/6] Setting up Libero env...')
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    task_suite = bd[SUITE]()
    bddl_file = task_suite.get_task_bddl_file_path(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    initial_state = copy.deepcopy(initial_states[state_idx % len(initial_states)])
    task = task_suite.get_task(task_idx)
    task_name = task.name if hasattr(task, 'name') else str(task)
    instruction = str(getattr(task, 'language', task_name))
    print(f'  Task: {task_name}')
    print(f'  Instruction: {instruction}')

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, control_freq=20
    )
    env.reset()
    obs = env.set_init_state(initial_state)

    # ── Load N4 detectors ──
    print('[3/6] Loading N4 detectors...')
    from n4_detector_adapter import N4DetectorAdapter
    adapter_sparse = N4DetectorAdapter(device=device_str, norm_data_path=NORM_PATH)
    adapter_full = N4DetectorAdapter(device=device_str, norm_data_path=NORM_PATH)
    from n4_detector_adapter_v4 import build_n4_inputs

    # ── Image preprocessing (reuse production) ──
    from gripper_attack.openvla_preprocess import prepare_openvla_image

    # ── Action decoding (inlined from run_gpu_smoke.py) ──
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
    print('  Action decoder inlined')

    print('[4/6] Running CLEAN episode...')
    adapter_sparse.reset_episode()
    adapter_full.reset_episode()

    steps_full = []
    steps_sparse = []
    done = False
    MAX_STEPS = 300
    DUMMY_WAIT = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    for t in range(MAX_STEPS):
        # ── Model inference (reuse production pipeline) ──
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
        env_action = clean_action.copy()

        # ── SPARSE pipeline (production) ──
        f25d_s = np.zeros(25, dtype=np.float32)
        f25d_s[0] = float(clean_action[6])
        f25d_s[6:9] = clean_action[0:3]
        p9d_s = np.zeros(9, dtype=np.float32)
        g9d_s = np.zeros(9, dtype=np.float32)
        cc = float(clean_action[6]) <= 0.5
        result_sparse = adapter_sparse.step(f25d_s, p9d_s, g9d_s, cc)

        # ── FULL pipeline (shadow observer) ──
        try:
            n4_full = build_n4_inputs(
                obs=obs, clean_raw_action=clean_action, clean_env_action=env_action,
                clean_model_output=generation, policy_step=t, suite=SUITE,
                unnorm_key=UNNORM_KEY, model=model, processor=processor
            )
            result_full = adapter_full.step(
                n4_full['f25d'], n4_full['p9d'], n4_full['g9d'], n4_full['candidate_close']
            )
        except Exception as e:
            if t < 5:
                print(f'  WARN step {t}: build_n4_inputs failed: {e}')
            n4_full = None
            result_full = None

        # Collect
        sf = {
            'step': t,
            'sparse_raw': result_sparse['raw_logit'],
            'sparse_cal': result_sparse['calibrated_prob'],
            'sparse_cc': result_sparse['candidate_close'],
            'sparse_emit': result_sparse['emitted_this_step'],
        }
        if n4_full is not None and result_full is not None:
            sf.update({
                'full_raw': result_full['raw_logit'],
                'full_cal': result_full['calibrated_prob'],
                'full_cc': result_full['candidate_close'],
                'full_emit': result_full['emitted_this_step'],
                'raw_gripper': float(n4_full['f25d'][0]),
                'gripper_qpos': float(n4_full['f25d'][1]),
                'g9d_close_mass': float(n4_full['g9d'][0]),
                'g9d_open_mass': float(n4_full['g9d'][1]),
                'g9d_top1_is_close': bool(float(n4_full['g9d'][2]) > 0.5),
                'p9d_close_mass': float(n4_full['p9d'][1]),
                'p9d_open_mass': float(n4_full['p9d'][0]),
                'p9d_top1_is_close': bool(float(n4_full['p9d'][6]) > 0.5),
            })
        steps_full.append(sf)

        # Step env
        obs, reward, done, info = env.step(clean_action)

        if (t + 1) % 50 == 0:
            print(f'  Step {t+1}...')

        if done:
            break

    env.close()

    # ── Compare ──
    valid = [s for s in steps_full if 'full_raw' in s]
    n = len(valid)
    raw_diffs = [abs(s['full_raw'] - s['sparse_raw']) for s in valid]
    cal_diffs = [abs(s['full_cal'] - s['sparse_cal']) for s in valid]

    full_max_cal = max((s.get('full_cal', 0) for s in valid), default=0)
    sparse_max_cal = max((s.get('sparse_cal', 0) for s in steps_full), default=0)
    full_cc_count = sum(1 for s in valid if s.get('full_cc'))
    sparse_cc_count = sum(1 for s in steps_full if s.get('sparse_cc'))
    full_g9d_close_gt_open = sum(1 for s in valid if s.get('g9d_close_mass', 0) > s.get('g9d_open_mass', 0))
    full_p9d_top1_close = sum(1 for s in valid if s.get('p9d_top1_is_close'))

    OUTPUT = f'/tmp/phase1_goal_diag_v2_task{task_idx}_s{state_idx}.json'
    max_raw = max(raw_diffs) if raw_diffs else 0
    max_cal = max(cal_diffs) if cal_diffs else 0

    result = {
        'analysis': 'PHASE1_GOAL_FULL_VS_SPARSE_DIAGNOSTIC_V2',
        'suite': SUITE, 'task_index': task_idx, 'state_index': state_idx, 'gpu': gpu,
        'n_steps': len(steps_full), 'n_valid_full': n,
        'full_pipeline': {
            'n_cc_true': full_cc_count, 'max_cal': full_max_cal,
            'g9d_close_gt_open': full_g9d_close_gt_open,
            'p9d_top1_close': full_p9d_top1_close,
        },
        'sparse_pipeline': {'n_cc_true': sparse_cc_count, 'max_cal': sparse_max_cal},
        'comparison': {
            'n': n, 'max_raw_diff': max_raw, 'max_cal_diff': max_cal,
            'mean_raw_diff': float(np.mean(raw_diffs)) if raw_diffs else 0,
            'mean_cal_diff': float(np.mean(cal_diffs)) if cal_diffs else 0,
            'raw_parity': max_raw < 1e-4, 'cal_parity': max_cal < 1e-4,
        },
        'steps': steps_full,
    }

    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[6/6] RESULTS:')
    print(f'  Steps: {len(steps_full)}, Full-valid: {n}')
    print(f'  Full: cc={full_cc_count} max_cal={full_max_cal:.6f} g9d_close>open={full_g9d_close_gt_open} p9d_top1_close={full_p9d_top1_close}')
    print(f'  Sparse: cc={sparse_cc_count} max_cal={sparse_max_cal:.6f}')
    print(f'  Diff: max_raw={max_raw:.6f} max_cal={max_cal:.6f}')
    print(f'  Raw parity: {max_raw < 1e-4}')
    print(f'  Output: {OUTPUT}')

    if full_max_cal < 0.5 and sparse_max_cal < 0.5:
        print(f'\n  *** VERDICT: Both pipelines produce low scores.')
        if full_cc_count == 0:
            print(f'  *** raw_close=0 in FULL pipeline — policy genuinely outputs no close on goal.')
        print(f'  *** Score is low regardless of feature quality. Not a sparse-feature artifact.')

if __name__ == '__main__':
    main()
