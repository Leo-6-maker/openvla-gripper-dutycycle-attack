"""Phase 1 Goal Diagnostic: Run one goal task with FULL build_n4_inputs pipeline.
Compares full-feature scores against sparse-feature scores (production runner).
gpu 6, single CLEAN arm only.
"""
import json, os, sys, time, hashlib
import numpy as np
import torch

REPO = '/mnt/sdc/dty_user/openvla_attack'
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, '/tmp')
sys.path.insert(0, os.path.join(REPO, 'scripts/fec'))

from n4_detector_adapter_v4 import build_n4_inputs
from n4_detector_adapter import N4DetectorAdapter

# ── Config ──
MODEL_PATH = '/mnt/sdc/dty_user/openvla_attack/models/libero-goal'
SUITE = 'libero_goal'
UNNORM_KEY = 'libero_goal'
NORM_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/fec_implementation_v1/n4_norms_o0i0.pt'
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0, help='gpu ID (after CUDA_VISIBLE_DEVICES remap)')
    ap.add_argument('--task-idx', type=int, default=0)
    ap.add_argument('--state-idx', type=int, default=100)
    args_inner = ap.parse_args()

    gpu = args_inner.gpu
    task_idx = args_inner.task_idx
    state_idx = args_inner.state_idx
    device = torch.device(f'cuda:{gpu}')
    print(f'[1/5] Loading OpenVLA from {MODEL_PATH}...')
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f'  Model loaded on {device}')

    print('[2/5] Loading Libero env...')
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[SUITE]()
    task_name = [t for t in task_suite.get_task_names() if f'task_{task_idx:02d}' in t or f'task{task_idx}' in t]
    if not task_name:
        all_tasks = list(task_suite.get_task_names())
        task_name = all_tasks[task_idx] if task_idx < len(all_tasks) else None
        if not task_name:
            print(f'ERROR: cannot find task {task_idx}')
            return
    else:
        task_name = task_name[0]
    print(f'  Task: {task_name}')

    # Get env init state (matching production runner)
    import copy
    from libero.libero.envs import OffScreenRenderEnv
    bddl_file = task_suite.get_task_bddl_file_path(task_idx)
    initial_states = task_suite.get_task_init_states(task_idx)
    initial_state = copy.deepcopy(initial_states[state_idx % len(initial_states)])
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=True, control_freq=20)
    env.reset()
    obs = env.set_init_state(initial_state)

    print('[3/5] Initializing N4 Detector (full pipeline + sparse comparison)...')
    adapter_full = N4DetectorAdapter(device=str(device), norm_data_path=NORM_PATH)
    adapter_sparse = N4DetectorAdapter(device=str(device), norm_data_path=NORM_PATH)

    print('[4/5] Running CLEAN episode...')
    obs = env.reset()
    adapter_full.reset_episode()
    adapter_sparse.reset_episode()

    steps_full = []
    steps_sparse = []
    policy_step = 0
    done = False
    max_steps = 300

    for t in range(max_steps):
        # Model inference
        model_inputs = processor(obs, task=env.task_embodiment)
        model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()}

        with torch.no_grad():
            clean_output = model.generate(
                **{k: v for k, v in model_inputs.items() if isinstance(v, torch.Tensor)},
                output_scores=True, return_dict_in_generate=True,
                max_new_tokens=10, do_sample=False
            )

        clean_action = model.decode_action(clean_output.scores[-1], UNNORM_KEY)
        env_action = clean_action.copy() if hasattr(clean_action, 'copy') else clean_action

        # ── FULL pipeline ──
        try:
            n4_full = build_n4_inputs(
                obs=obs, clean_raw_action=clean_action,
                clean_env_action=env_action,
                clean_model_output=clean_output,
                policy_step=policy_step, suite=SUITE,
                unnorm_key=UNNORM_KEY, model=model,
                processor=processor
            )
            result_full = adapter_full.step(
                n4_full['f25d'], n4_full['p9d'], n4_full['g9d'],
                n4_full['candidate_close']
            )
        except Exception as e:
            print(f'  WARN: full pipeline step {t}: {e}')
            n4_full = None
            result_full = None

        # ── SPARSE pipeline (matching production runner) ──
        f25d_sparse = np.zeros(25, dtype=np.float32)
        f25d_sparse[0] = float(clean_action[6])
        f25d_sparse[6:9] = clean_action[0:3]
        p9d_sparse = np.zeros(9, dtype=np.float32)
        g9d_sparse = np.zeros(9, dtype=np.float32)
        cc = float(clean_action[6]) <= 0.5
        result_sparse = adapter_sparse.step(f25d_sparse, p9d_sparse, g9d_sparse, cc)

        # Collect step data
        if n4_full is not None and result_full is not None:
            step_full = {
                'step': t, 'policy_step': policy_step,
                'raw_gripper': float(n4_full['f25d'][0]),
                'env_gripper': float(env_action[6]),
                'gripper_qpos': float(n4_full['f25d'][1]),
                'raw_close': bool(n4_full['candidate_close']),
                'env_close': bool(float(env_action[6]) > 0),
                # g9d fields
                'g9d_close_mass': float(n4_full['g9d'][0]),
                'g9d_open_mass': float(n4_full['g9d'][1]),
                'g9d_top1_is_close': bool(float(n4_full['g9d'][2]) > 0.5),
                'g9d_top1_is_open': bool(float(n4_full['g9d'][3]) > 0.5),
                'g9d_entropy': float(n4_full['g9d'][7]),
                'g9d_open_minus_close_log_mass': float(n4_full['g9d'][8]),
                # p9d fields
                'p9d_open_mass': float(n4_full['p9d'][0]),
                'p9d_close_mass': float(n4_full['p9d'][1]),
                'p9d_top1_is_close': bool(float(n4_full['p9d'][6]) > 0.5),
                'p9d_top1_is_open': bool(float(n4_full['p9d'][5]) > 0.5),
                # Detector (full)
                'full_raw_logit': result_full['raw_logit'],
                'full_calibrated_prob': result_full['calibrated_prob'],
                'full_candidate_close': result_full['candidate_close'],
                'full_emitted': result_full['emitted_this_step'],
            }
            steps_full.append(step_full)

        # Sparse detector
        step_sparse = {
            'step': t,
            'sparse_raw_logit': result_sparse['raw_logit'],
            'sparse_calibrated_prob': result_sparse['calibrated_prob'],
            'sparse_candidate_close': result_sparse['candidate_close'],
            'sparse_emitted': result_sparse['emitted_this_step'],
        }
        steps_sparse.append(step_sparse)

        # Step environment
        obs, reward, done, info = env.step(clean_action)
        policy_step += 1

        if done:
            break

    env.close()

    # ── Comparison ──
    n = min(len(steps_full), len(steps_sparse))
    raw_diffs = []
    cal_diffs = []
    for i in range(n):
        sf = steps_full[i]
        ss = steps_sparse[i]
        raw_diffs.append(abs(sf['full_raw_logit'] - ss['sparse_raw_logit']))
        cal_diffs.append(abs(sf['full_calibrated_prob'] - ss['sparse_calibrated_prob']))

    max_raw_diff = max(raw_diffs) if raw_diffs else 0
    max_cal_diff = max(cal_diffs) if cal_diffs else 0
    mean_raw_diff = np.mean(raw_diffs) if raw_diffs else 0
    mean_cal_diff = np.mean(cal_diffs) if cal_diffs else 0

    # Goal-specific metrics
    full_cc_true = sum(1 for s in steps_full if s.get('raw_close'))
    sparse_cc_true = sum(1 for s in steps_sparse if s.get('sparse_candidate_close'))
    full_max_cal = max((s.get('full_calibrated_prob', 0) for s in steps_full), default=0)
    sparse_max_cal = max((s.get('sparse_calibrated_prob', 0) for s in steps_sparse), default=0)
    g9d_close_gt_open = sum(1 for s in steps_full if s.get('g9d_close_mass', 0) > s.get('g9d_open_mass', 0))
    p9d_top1_close = sum(1 for s in steps_full if s.get('p9d_top1_is_close'))
    g9d_any_close_gt_open = sum(1 for s in steps_full if s.get('g9d_close_mass', 0) > 0.5)

    output = {
        'analysis': 'PHASE1_GOAL_FULL_VS_SPARSE_DIAGNOSTIC_V1',
        'suite': SUITE, 'task_index': task_idx, 'state_index': state_idx,
        'gpu': gpu,
        'n_steps': len(steps_full),
        'full_pipeline': {
            'n_cc_true': full_cc_true,
            'max_cal': full_max_cal,
            'p9d_top1_close_count': p9d_top1_close,
            'g9d_close_gt_open_count': g9d_close_gt_open,
            'g9d_close_gt_05_count': g9d_any_close_gt_open,
        },
        'sparse_pipeline': {
            'n_cc_true': sparse_cc_true,
            'max_cal': sparse_max_cal,
        },
        'comparison': {
            'max_raw_diff': max_raw_diff,
            'max_cal_diff': max_cal_diff,
            'mean_raw_diff': mean_raw_diff,
            'mean_cal_diff': mean_cal_diff,
            'raw_diff_parity': max_raw_diff < 1e-4,
            'cal_diff_parity': max_cal_diff < 1e-4,
        },
        'steps_full': steps_full,
        'steps_sparse': steps_sparse,
    }

    OUTPUT = f'/tmp/phase1_goal_diag_task{task_idx}_s{state_idx}.json'
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  Steps: {len(steps_full)}')
    print(f'  Full pipeline: cc_true={full_cc_true} max_cal={full_max_cal:.6f} p9d_top1_close={p9d_top1_close} g9d_close>open={g9d_close_gt_open}')
    print(f'  Sparse pipeline: cc_true={sparse_cc_true} max_cal={sparse_max_cal:.6f}')
    print(f'  Diff: max_raw={max_raw_diff:.6f} max_cal={max_cal_diff:.6f} mean_raw={mean_raw_diff:.6f}')
    print(f'  Raw parity: {max_raw_diff < 1e-4}')
    print(f'  Output: {OUTPUT}')

    # Critical finding
    if full_max_cal < 0.5 and sparse_max_cal < 0.5:
        print(f'\n  *** VERDICT: Both pipelines produce low scores on goal.')
        print(f'  *** Full max_cal={full_max_cal:.4f}, Sparse max_cal={sparse_max_cal:.4f}')
        if full_cc_true == 0:
            print(f'  *** raw_close=0 in FULL pipeline too — policy genuinely outputs no close on goal')
        print(f'  *** This confirms: low score is NOT caused by sparse features.')
        print(f'  *** Remaining hypotheses: (a) Teacher has no critical windows on goal, OR (b) Student generalization failure.')

if __name__ == '__main__':
    main()
