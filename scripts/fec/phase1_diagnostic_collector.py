"""Phase 1 Diagnostic Data Collector.
Runs CLEAN episodes (no attack) through the FULL build_n4_inputs() pipeline,
collecting all per-step fields for root cause analysis.

Contrast with production runner: the production runner feeds sparse features
(zeros for g9d/p9d). This collector uses build_n4_inputs() V4 which runs the
full SC5 streaming feature pipeline and extracts real logit summaries.

Output: one JSONL per episode with per-step records containing all fields
listed in Phase 1.1 of the N5 roadmap.
"""
import json, os, sys, time, argparse, hashlib
import numpy as np
import torch

# Server paths
REPO_ROOT = '/mnt/sdc/dty_user/openvla_attack'
SRC_ROOT = '/mnt/sdc/dty_user/openvla_attack/src'
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts/fec'))

from n4_detector_adapter_v4 import build_n4_inputs, N4DetectorAdapter

# Libero task suites
SUITE_TASKS = {
    'libero_10': list(range(10)),
    'libero_goal': list(range(10)),
    'libero_object': list(range(10)),
    'libero_spatial': list(range(10)),
}

SUITE_HORIZONS = {
    'libero_10': 520, 'libero_goal': 300,
    'libero_object': 280, 'libero_spatial': 220,
}

NORM_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1/o0_i0/normalization.pt'

def collect_diagnostic_episode(env, model, processor, unnorm_key, suite, task_idx, state_idx,
                                adapter, device, max_steps):
    """Run one CLEAN episode and collect all per-step diagnostic fields."""
    from libero.libero.envs import OffScreenRenderEnv
    obs = env.reset()
    adapter.reset_episode()

    steps = []
    policy_step = 0

    for t in range(max_steps):
        # Prepare model inputs
        if hasattr(processor, '__call__'):
            model_inputs = processor(obs, task=env.task_embodiment)
        else:
            model_inputs = obs

        # Get clean model output
        with torch.no_grad():
            clean_output = model.generate(
                **{k: v.to(device) for k, v in model_inputs.items()},
                output_scores=True, return_dict_in_generate=True,
                max_new_tokens=10, do_sample=False
            )

        # Decode action
        clean_action_raw = model.decode_action(clean_output.scores[-1], unnorm_key)

        # Get env action (for env_gripper)
        clean_env_action = model.decode_action(clean_output.scores[-1], unnorm_key)

        # Step environment
        obs, reward, done, info = env.step(clean_action_raw)

        # Build full N4 inputs via canonical provider
        try:
            n4_inputs = build_n4_inputs(
                obs=obs, clean_raw_action=clean_action_raw,
                clean_env_action=clean_env_action,
                clean_model_output=clean_output,
                policy_step=policy_step, suite=suite,
                unnorm_key=unnorm_key, model=model,
                processor=processor
            )
        except Exception as e:
            print(f'  WARN: build_n4_inputs failed at step {t}: {e}')
            n4_inputs = None

        # Run through V4 detector
        if n4_inputs is not None:
            detector_result = adapter.step(
                n4_inputs['f25d'], n4_inputs['p9d'], n4_inputs['g9d'],
                n4_inputs['candidate_close']
            )
        else:
            detector_result = None

        # Collect all Phase 1.1 fields
        record = {
            'episode_step': t,
            'policy_step': policy_step,
            'suite': suite,
            'task_index': task_idx,
            'state_index': state_idx,
        }

        if n4_inputs is not None:
            f25d = n4_inputs['f25d']
            p9d = n4_inputs['p9d']
            g9d = n4_inputs['g9d']

            record.update({
                'raw_gripper': float(f25d[0]),
                'env_gripper': float(clean_env_action[6]) if clean_env_action is not None else 0,
                'gripper_command': float(f25d[0]),
                'gripper_qpos': float(f25d[1]),
                'gripper_opening_proxy': float(f25d[2]),
                'qpos_delta_1': float(f25d[21]),
                'qpos_delta_3': float(f25d[22]),
                'raw_close': bool(float(f25d[0]) <= 0.5),
                'env_close': bool(float(clean_env_action[6]) > 0) if clean_env_action is not None else False,
                'open_mass': float(p9d[0]),
                'close_mass': float(p9d[1]),
                'open_minus_close_log_mass': float(p9d[2]),
                'entropy': float(p9d[3]),
                'top1_prob': float(p9d[4]),
                'top1_is_open': bool(float(p9d[5]) > 0.5),
                'top1_is_close': bool(float(p9d[6]) > 0.5),
                'best_open_rank': float(p9d[7]),
                'best_close_rank': float(p9d[8]),
                'g9d_close_mass': float(g9d[0]),
                'g9d_open_mass': float(g9d[1]),
                'g9d_top1_is_close': bool(float(g9d[2]) > 0.5),
                'g9d_top1_is_open': bool(float(g9d[3]) > 0.5),
                'g9d_top1_prob': float(g9d[4]),
                'g9d_best_close_rank': float(g9d[5]),
                'g9d_best_open_rank': float(g9d[6]),
                'g9d_entropy': float(g9d[7]),
                'g9d_open_minus_close_log_mass': float(g9d[8]),
            })

        if detector_result is not None:
            record.update({
                'v4_raw_logit': detector_result['raw_logit'],
                'v4_calibrated_prob': detector_result['calibrated_prob'],
                'v4_candidate_close': detector_result['candidate_close'],
                'v4_persistence_counter': detector_result['persistence_counter'],
                'v4_emitted_this_step': detector_result['emitted_this_step'],
            })

        # Physical state from obs
        qpos = obs.get('robot0_gripper_qpos', [0, 0])
        if hasattr(qpos, '__len__') and len(qpos) >= 2:
            record['physical_gripper_q7'] = float(qpos[0])
            record['physical_gripper_q8'] = float(qpos[1])
        eef = obs.get('robot0_eef_pos', [0, 0, 0])
        if hasattr(eef, '__len__') and len(eef) >= 3:
            record['eef_x'] = float(eef[0])
            record['eef_y'] = float(eef[1])
            record['eef_z'] = float(eef[2])

        record['done'] = bool(done)
        record['success'] = bool(info.get('success', False)) if isinstance(info, dict) else False

        steps.append(record)
        policy_step += 1

        if done:
            break

    return steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, choices=list(SUITE_TASKS.keys()))
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--state-index', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--model-path', default='/mnt/sdc/dty_user/openvla_attack/models/openvla-7b')
    parser.add_argument('--output-dir', default='/tmp/phase1_diagnostics')
    parser.add_argument('--max-states', type=int, default=3,
                        help='Number of initial states to sample per task')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load OpenVLA model
    print(f'Loading OpenVLA from {args.model_path}...')
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Determine unnorm key (Libero uses the suite name)
    unnorm_key = args.suite

    # Initialize V4 detector adapter
    adapter = N4DetectorAdapter(device=str(device), norm_data_path=NORM_PATH)

    # Import Libero
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_names = [t for t in task_suite.get_task_names()
                  if f'task_{args.task_index:02d}' in t or f'task{args.task_index}' in t]

    if not task_names:
        # Try by index
        all_tasks = list(task_suite.get_task_names())
        if args.task_index < len(all_tasks):
            task_names = [all_tasks[args.task_index]]
        else:
            print(f'ERROR: task_index {args.task_index} out of range ({len(all_tasks)} tasks)')
            sys.exit(1)

    task_name = task_names[0]
    print(f'Task: {task_name}')

    max_steps = SUITE_HORIZONS.get(args.suite, 300)

    # Collect diagnostics
    all_episodes = []
    for si in range(args.max_states):
        state_idx = args.state_index + si
        print(f'  State {state_idx} ({si+1}/{args.max_states})...')
        try:
            env_args = {
                'task_name': task_name,
                'benchmark_name': args.suite,
                'task_init_id': state_idx,
                'device': str(device) if torch.cuda.is_available() else 'cpu',
            }
            if hasattr(task_suite, 'get_task_init_states'):
                env = task_suite.get_task_init_states(task_name)[state_idx]
            else:
                try:
                    from libero.libero.envs import OffScreenRenderEnv
                    env = OffScreenRenderEnv(**env_args)
                except:
                    print(f'  SKIP state {state_idx}: cannot create env')
                    continue

            steps = collect_diagnostic_episode(
                env, model, processor, unnorm_key, args.suite,
                args.task_index, state_idx, adapter, device, max_steps
            )
            all_episodes.append({'state_index': state_idx, 'steps': steps})
            print(f'    {len(steps)} steps collected')
        except Exception as e:
            print(f'  ERROR state {state_idx}: {e}')
            import traceback
            traceback.print_exc()

    # Save output
    output_file = os.path.join(
        args.output_dir,
        f'diag_{args.suite}_task{args.task_index:02d}_s{args.state_index}_{int(time.time())}.json'
    )
    with open(output_file, 'w') as f:
        json.dump({
            'suite': args.suite,
            'task_index': args.task_index,
            'state_indices': [args.state_index + i for i in range(args.max_states)],
            'gpu_id': args.gpu_id,
            'n_episodes': len(all_episodes),
            'total_steps': sum(len(ep['steps']) for ep in all_episodes),
            'episodes': all_episodes,
        }, f, indent=2)

    print(f'\nSaved {len(all_episodes)} episodes, {sum(len(ep["steps"]) for ep in all_episodes)} total steps')
    print(f'Output: {output_file}')

    # Quick summary
    for ep in all_episodes:
        max_cal = max((s.get('v4_calibrated_prob', 0) for s in ep['steps']), default=0)
        n_cc = sum(1 for s in ep['steps'] if s.get('raw_close', False))
        n_env_close = sum(1 for s in ep['steps'] if s.get('env_close', False))
        n_close_mass = sum(1 for s in ep['steps'] if s.get('close_mass', 0) > s.get('open_mass', 0))
        print(f'  State {ep["state_index"]}: steps={len(ep["steps"])} max_cal={max_cal:.4f} '
              f'raw_close={n_cc} env_close={n_env_close} close>open={n_close_mass}')

    env.close() if hasattr(env, 'close') else None

if __name__ == '__main__':
    main()
