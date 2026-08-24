"""C2R0.2: task_09 First-Divergence Forensics.

A: Termination alignment — compare original vs replay done/success/reason
B: First-divergence state capture — full state at step 135 pre/post
C: Repeatability — replay 3x from same canonical state

Reports numerical_parity and init_hash_status separately.
"""
import json, os, sys, time, hashlib, copy
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(DIR)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))

from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait,
)
from v22_production_v2 import parse_sidecar

CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
C2R02_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2cr02_task09_forensics'
os.makedirs(C2R02_OUT, exist_ok=True)

NUM_STEPS_WAIT = 10
IDENT = ('libero_10', 'task_09', 'state_00')
DIVERGENCE_STEP = 135  # step BEFORE first observed EEF divergence


def load_original(ident):
    suite, task, state = ident
    sidecar_path = os.path.join(CS200, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    step_path = os.path.join(CS200, suite, task, state, 'step_records.jsonl')
    meta_path = os.path.join(CS200, suite, task, state, 'episode_metadata.json')

    original = parse_sidecar(sidecar_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    with open(step_path) as f:
        actions = [json.loads(l)['action_env'] for l in f if l.strip()]

    return original['steps'], actions, metadata


def build_env_and_init(suite, task_idx, state_id, max_steps):
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict

    benchmark = get_benchmark(suite)(0)
    task_obj = benchmark.get_task(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                            task_obj.problem_folder, task_obj.bddl_file)

    suite_dict = get_benchmark_dict()
    suite_obj = suite_dict[suite]()
    init_states = suite_obj.get_task_init_states(task_idx)
    canonical_state = init_states[state_id]

    env, obs = build_v4_exact_env(
        bddl_path, render_gpu_device_id=-1,
        max_steps=max_steps, num_steps_wait=NUM_STEPS_WAIT,
    )
    obs = env.set_init_state(canonical_state)
    env, obs = apply_dummy_wait(env, obs, NUM_STEPS_WAIT)
    return env, obs


def capture_full_state(env, step_label):
    """Capture complete sim state at current step."""
    sim = env.sim
    model = sim.model

    state = {
        'step': step_label,
        'sim_time': float(sim.data.time),
        'qpos': sim.data.qpos.copy().tolist(),
        'qvel': sim.data.qvel.copy().tolist(),
        'ctrl': sim.data.ctrl.copy().tolist() if hasattr(sim.data, 'ctrl') else [],
        'act': sim.data.act.copy().tolist() if hasattr(sim.data, 'act') else [],
        'ncon': int(sim.data.ncon),
        'eef_pos': sim.data.site_xpos[model.site_name2id('gripper0_grip_site')].tolist(),
    }

    # Contact details
    contacts = []
    for ci in range(sim.data.ncon):
        c = sim.data.contact[ci]
        g1 = model.geom(c.geom1).name
        g2 = model.geom(c.geom2).name
        if g1 and g2:
            contacts.append({
                'geom1': g1, 'geom2': g2,
                'dist': float(c.dist),
                'pos': c.pos.copy().tolist() if hasattr(c, 'pos') else [],
            })
    state['contacts'] = contacts

    # Fixture sites (microwave, desk_caddy etc.)
    fixture_sites = {}
    for site_name in sim.model.site_names:
        if any(k in site_name for k in ['heating', 'cook', 'contain', 'caddy']):
            sid = model.site_name2id(site_name)
            fixture_sites[site_name] = {
                'xpos': sim.data.site_xpos[sid].tolist(),
                'xmat': sim.data.site_xmat[sid].tolist(),
            }
    state['fixture_sites'] = fixture_sites

    return state


def compare_step(replay_state, orig_step, t):
    """Compare replay state with original sidecar step. Returns issues list."""
    issues = []
    r_eef = replay_state.get('eef_pos', [0,0,0])
    o_eef = orig_step.get('robot0_eef_pos', [0,0,0])
    if len(r_eef) >= 3 and len(o_eef) >= 3:
        d = abs(r_eef[0]-o_eef[0]) + abs(r_eef[1]-o_eef[1]) + abs(r_eef[2]-o_eef[2])
        if d > 0.0001:
            issues.append(f'step_{t}_eef_L1={d:.6f}')

    r_qpos = replay_state.get('gripper_qpos', [0,0])
    o_qpos = orig_step.get('robot0_gripper_qpos', [0,0])
    if len(r_qpos) >= 2 and len(o_qpos) >= 2:
        d = abs(r_qpos[0]-o_qpos[0]) + abs(r_qpos[1]-o_qpos[1])
        if d > 0.0001:
            issues.append(f'step_{t}_qpos_L1={d:.6f}')

    return issues


def replay_one_run(env, obs, actions, orig_steps, run_label, capture_state=False):
    """Single replay run. Returns (parity_issues, termination_info, captured_states)."""
    issues = []
    termination = {'original_done_step': None, 'replay_done_step': None,
                   'original_success': None}
    captured = {}

    for t in range(min(len(actions), len(orig_steps))):
        # Record pre-action state
        sim = env.sim
        qpos = obs.get('robot0_gripper_qpos', [0,0])
        if hasattr(qpos, 'flatten'): qpos = qpos.flatten()
        eef = obs.get('robot0_eef_pos', [0,0,0])
        if hasattr(eef, 'flatten'): eef = eef.flatten()

        # Capture full state at divergence boundary
        if capture_state and t in [DIVERGENCE_STEP - 1, DIVERGENCE_STEP, DIVERGENCE_STEP + 1]:
            captured[f'{run_label}_pre_action_step_{t}'] = capture_full_state(env, f'{run_label}_pre_{t}')

        # Compare
        step_issues = compare_step(
            {'eef_pos': eef, 'gripper_qpos': qpos},
            orig_steps[t], t)
        issues.extend(step_issues)

        # Check original termination
        orig_done = orig_steps[t].get('env_done')
        if orig_done and termination['original_done_step'] is None:
            termination['original_done_step'] = t

        # Execute action
        action = actions[t]
        obs, reward, done, info = env.step(action)

        if done and termination['replay_done_step'] is None:
            termination['replay_done_step'] = t

        # Capture post-action state
        if capture_state and t in [DIVERGENCE_STEP - 1, DIVERGENCE_STEP, DIVERGENCE_STEP + 1]:
            captured[f'{run_label}_post_action_step_{t}'] = capture_full_state(env, f'{run_label}_post_{t}')

    return issues, termination, captured


def main():
    print('=' * 60)
    print('C2R0.2: task_09 First-Divergence Forensics')
    print('=' * 60)

    suite, task, state_id_str = IDENT
    task_idx = int(task.replace('task_', ''))
    state_id = int(state_id_str.replace('state_', ''))
    ident_str = f'{suite}/{task}/{state_id_str}'

    print(f'\nTarget: {ident_str}')
    orig_steps, actions, metadata = load_original(IDENT)
    print(f'Sidecar steps: {len(orig_steps)}, Actions: {len(actions)}')

    # A: Termination alignment
    print(f'\n--- A: Termination Alignment ---')
    orig_done_steps = [i for i, s in enumerate(orig_steps) if s.get('env_done')]
    print(f'Original done at steps: {orig_done_steps}')
    print(f'Original success: {metadata.get("success")}')
    print(f'Original termination: {metadata.get("termination_reason")}')

    # B+C: Repeatability test (3 runs)
    print(f'\n--- B+C: 3-Run Repeatability + First-Divergence Capture ---')
    all_runs = []

    for run_idx in range(3):
        print(f'\nRun {run_idx + 1}/3...', end=' ', flush=True)
        env, obs = build_env_and_init(suite, task_idx, state_id,
                                      max(500, len(actions) + 50))
        try:
            # Capture initial state
            init_full = capture_full_state(env, 'after_init')

            issues, term, captured = replay_one_run(
                env, obs, actions, orig_steps,
                f'run{run_idx+1}',
                capture_state=(run_idx == 0),  # Full capture only on first run
            )

            if run_idx == 0:
                all_captured = captured
                all_captured['initial'] = init_full

            # Find first divergence
            first_div = None
            for iss in issues:
                step_num = int(iss.split('_')[1])
                if first_div is None or step_num < first_div:
                    first_div = step_num

            print(f'issues={len(issues)}, first_div_step={first_div}, '
                  f'replay_done={term["replay_done_step"]}')

            all_runs.append({
                'run': run_idx + 1,
                'n_issues': len(issues),
                'first_divergence_step': first_div,
                'termination': term,
                'issue_sample': issues[:5],
            })
        finally:
            env.close()

    # Report
    print(f'\n--- B: First-Divergence State ---')
    n_div_steps = set(r['first_divergence_step'] for r in all_runs if r['first_divergence_step'])
    print(f'First divergence steps across runs: {n_div_steps}')

    # Show key diff at step 135
    if 'run1_pre_action_step_135' in all_captured:
        s = all_captured['run1_pre_action_step_135']
        o = orig_steps[135] if len(orig_steps) > 135 else {}
        print(f'\nStep 135 pre-action:')
        print(f'  EEF: replay={s["eef_pos"]} orig={o.get("robot0_eef_pos","?")}')
        print(f'  ncon: replay={s["ncon"]}')

    print(f'\n--- C: Repeatability ---')
    first_divs = [r['first_divergence_step'] for r in all_runs]
    same_div = len(set(first_divs)) == 1
    print(f'First-div steps: {first_divs}')
    print(f'Same divergence point: {same_div}')
    print(f'Interpretation: ', end='')
    if same_div:
        print('SYSTEMATIC — hidden state/config difference')
    else:
        print('NON_DETERMINISTIC or contact-instability')

    # Summary
    print(f'\n{"=" * 60}')
    print('C2R0.2 Summary:')
    print(f'  numerical_parity: {min(r["n_issues"] for r in all_runs)} issues')
    print(f'  init_hash_status: HASH_CANONICALIZATION_UNRESOLVED')
    print(f'  first_divergence: {first_divs}')
    print(f'  repeatability: {"SYSTEMATIC" if same_div else "VARIABLE"}')

    # Write report
    report = {
        'gate': 'C2R0.2_TASK09_FORENSICS',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'identity': ident_str,
        'numerical_parity_status': 'PARTIAL' if any(r['n_issues'] > 0 for r in all_runs) else 'PASS',
        'init_hash_status': 'HASH_CANONICALIZATION_UNRESOLVED',
        'repeatability_same_divergence': same_div,
        'first_divergence_steps': first_divs,
        'runs': all_runs,
        'state_at_divergence_summary': {
            k: {'ncon': v.get('ncon'), 'eef_pos': v.get('eef_pos')}
            for k, v in all_captured.items() if '135' in k or '136' in k
        },
    }

    rp = os.path.join(C2R02_OUT, 'C2R02_REPORT.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nReport: {rp}')

    for r in all_runs:
        if r['n_issues'] == 0:
            sys.exit(0)
    sys.exit(5)


if __name__ == '__main__':
    main()
