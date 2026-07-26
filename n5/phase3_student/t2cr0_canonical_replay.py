"""T2R-C2R0: Canonical Init-State Recovery and Exact Replay.

Uses the original CS200 collector protocol:
  build_v4_exact_env → set_init_state(init_states[state_id]) → 10 dummy waits → replay

Verifies replay parity with original sidecar step-by-step.
"""
import json, os, sys, time, hashlib, copy
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(DIR)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))

from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait, DUMMY_OPEN_ACTION,
)
from v22_production_v2 import parse_sidecar, get_object_slices_for_task

CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
C1_REGISTRY = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_full_registry/per_task'
C2R0_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2cr0_canonical_replay'
os.makedirs(C2R0_OUT, exist_ok=True)

NUM_STEPS_WAIT = 10
# Parity tolerances (tight, for exact replay)
QPOS_TOL = 0.0005
EEF_TOL = 0.001
OBJ_TOL = 0.002


def hash_init_state(init_state_obj):
    """Hash canonical init state using array bytes (matches CS200 protocol)."""
    # init_state is typically a numpy array or dict of arrays
    if isinstance(init_state_obj, np.ndarray):
        return hashlib.sha256(init_state_obj.tobytes()).hexdigest()
    if isinstance(init_state_obj, dict):
        data = []
        for k in sorted(init_state_obj.keys()):
            v = init_state_obj[k]
            if isinstance(v, np.ndarray):
                data.append(v.tobytes())
            else:
                data.append(str(v).encode())
        return hashlib.sha256(b''.join(data)).hexdigest()
    return hashlib.sha256(str(init_state_obj).encode()).hexdigest()


def replay_one(ident):
    """Replay one episode with exact collector protocol. Returns (result, parity_ok, error)."""
    suite, task, state_id_str = ident.split('/')
    task_idx = int(task.replace('task_', ''))
    state_id = int(state_id_str.replace('state_', ''))

    # Load original data
    sidecar_path = os.path.join(CS200, suite, task, state_id_str, 'privileged_teacher_sidecar.jsonl')
    step_path = os.path.join(CS200, suite, task, state_id_str, 'step_records.jsonl')
    meta_path = os.path.join(CS200, suite, task, state_id_str, 'episode_metadata.json')

    if not all(os.path.isfile(p) for p in [sidecar_path, step_path, meta_path]):
        return None, False, 'Missing CS200 files'

    original = parse_sidecar(sidecar_path)
    orig_steps = original['steps']

    with open(meta_path) as f:
        metadata = json.load(f)
    expected_init_sha = metadata.get('initial_state_sha256', '')

    with open(step_path) as f:
        recorded = [json.loads(l)['action_env'] for l in f if l.strip()]
    n_actions = len(recorded)

    # Create environment using exact collector protocol
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(suite)(0)
    task_obj = benchmark.get_task(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                            task_obj.problem_folder, task_obj.bddl_file)

    # Get canonical init states
    bm = get_benchmark.__self__ if hasattr(get_benchmark, '__self__') else None
    from libero.libero.benchmark import get_benchmark_dict
    suite_dict = get_benchmark_dict()
    suite_obj = suite_dict[suite]()
    init_states = suite_obj.get_task_init_states(task_idx)

    if state_id >= len(init_states):
        return None, False, f'state_id {state_id} out of range (max {len(init_states)-1})'

    canonical_state = init_states[state_id]
    init_sha = hash_init_state(canonical_state)
    init_sha_match = (init_sha == expected_init_sha)

    env = None
    try:
        # Exact collector protocol
        env, obs = build_v4_exact_env(
            bddl_path, render_gpu_device_id=-1,
            max_steps=max(500, n_actions + 50),
            num_steps_wait=NUM_STEPS_WAIT,
        )
        obs = env.set_init_state(canonical_state)
        env, obs = apply_dummy_wait(env, obs, NUM_STEPS_WAIT)

        # Now replay recorded actions
        telemetry = []
        parity_issues = []

        for t in range(min(n_actions, len(orig_steps))):
            action = recorded[t]
            obs, reward, done, info = env.step(action)

            sim = env.sim

            # Collect world-frame positions
            frame = {'step': t}

            # EEF
            eef = obs.get('robot0_eef_pos', [0,0,0])
            if hasattr(eef, 'flatten'): eef = eef.flatten()
            frame['eef_pos'] = [float(x) for x in eef[:3]]

            # Gripper qpos
            qpos = obs.get('robot0_gripper_qpos', [0,0])
            if hasattr(qpos, 'flatten'): qpos = qpos.flatten()
            frame['gripper_qpos'] = [float(x) for x in qpos[:2]]

            # Parity check vs original
            orig = orig_steps[t]
            orig_qpos = orig.get('robot0_gripper_qpos', [0,0])
            orig_eef = orig.get('robot0_eef_pos', [0,0,0])

            if len(qpos) >= 2 and len(orig_qpos) >= 2:
                qd = abs(qpos[0]-orig_qpos[0]) + abs(qpos[1]-orig_qpos[1])
                if qd > QPOS_TOL:
                    parity_issues.append({'step': t, 'field': 'gripper_qpos',
                                         'diff': float(qd), 'tol': QPOS_TOL})

            if len(eef) >= 3 and len(orig_eef) >= 3:
                ed = abs(eef[0]-orig_eef[0]) + abs(eef[1]-orig_eef[1]) + abs(eef[2]-orig_eef[2])
                if ed > EEF_TOL:
                    parity_issues.append({'step': t, 'field': 'eef_pos',
                                         'diff': float(ed), 'tol': EEF_TOL})

            # Object state parity
            obj_state_orig = orig.get('object_state', [])
            obj_state_replay = obs.get('object_state', [])
            if hasattr(obj_state_replay, 'flatten'):
                obj_state_replay = obj_state_replay.flatten()
            if len(obj_state_orig) > 0 and len(obj_state_replay) >= len(obj_state_orig):
                obj_diff = sum(abs(float(obj_state_replay[i]) - float(obj_state_orig[i]))
                              for i in range(min(len(obj_state_orig), len(obj_state_replay))))
                if obj_diff > OBJ_TOL * len(obj_state_orig):
                    parity_issues.append({'step': t, 'field': 'object_state',
                                         'diff': float(obj_diff)})

            # Contacts
            contacts = []
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]
                g1 = sim.model.geom(c.geom1).name
                g2 = sim.model.geom(c.geom2).name
                if g1 and g2:
                    contacts.append([g1, g2])
            frame['contacts'] = contacts

            telemetry.append(frame)

            if done and t < n_actions - 5:
                parity_issues.append({'step': t, 'type': 'early_termination',
                                     'expected': n_actions})
                break

        parity_ok = len(parity_issues) == 0

        result = {
            'identity': ident,
            'init_sha_match': init_sha_match,
            'expected_init_sha': expected_init_sha[:16] + '...',
            'actual_init_sha': init_sha[:16] + '...',
            'n_steps_original': len(orig_steps),
            'n_actions': n_actions,
            'n_replayed': len(telemetry),
            'parity_ok': parity_ok,
            'n_parity_issues': len(parity_issues),
            'parity_sample': parity_issues[:5],
        }
        return result, parity_ok and init_sha_match, None

    except Exception as e:
        return None, False, str(e)
    finally:
        if env is not None:
            try: env.close()
            except: pass


def main():
    print('=' * 60)
    print('C2R0: Canonical Init-State Recovery')
    print('=' * 60)

    # Test on 4 diverse episodes
    test_eps = [
        ('libero_10', 'task_00', 'state_00'),
        ('libero_10', 'task_09', 'state_00'),   # microwave fixture
        ('libero_object', 'task_00', 'state_04'),
        ('libero_spatial', 'task_00', 'state_00'),
    ]

    results = []
    for suite, task, state in test_eps:
        ident = f'{suite}/{task}/{state}'
        print(f'\n{ident}...', end=' ', flush=True)
        r, ok, err = replay_one(ident)
        if err:
            print(f'ERROR: {err[:120]}')
        elif ok:
            print(f'PARITY_OK  init_sha_match={r["init_sha_match"]}  issues=0')
        else:
            print(f'PARTIAL  init_sha_match={r["init_sha_match"]}  issues={r["n_parity_issues"]}')
            for p in r.get('parity_sample', [])[:3]:
                print(f'    step={p["step"]} {p["field"]}: diff={p["diff"]:.6f}')
        if r:
            results.append(r)

    n_ok = sum(1 for r in results if r.get('parity_ok') and r.get('init_sha_match'))
    print(f'\n{"=" * 60}')
    print(f'C2R0: {n_ok}/{len(results)} fully matched')
    for r in results:
        print(f'  {r["identity"]}: init_sha={r["init_sha_match"]} parity={r["parity_ok"]} issues={r["n_parity_issues"]}')

    # Write report
    report = {'gate': 'C2R0_CANONICAL_REPLAY', 'results': results}
    rp = os.path.join(C2R0_OUT, 'C2R0_REPORT.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nReport: {rp}')

    if n_ok == len(results):
        sys.exit(0)
    else:
        sys.exit(5)


if __name__ == '__main__':
    main()
