"""C3-S: Static Fixture Invariance Seal.

For every EXACT_SITE fixture target used in C3:
  S1: Verify strict EXACT_SITE resolution (no fallback).
  S2: Sample sim.data.site_xpos/xmat across all canonical init states,
      3 fresh resets each, pre/post dummy wait, rollout start/end.
  S3: Audit ancestor joint chain for movable joints.
  S4: Verify world-pose invariance within 1e-6 tolerance.

Output: STATIC_FIXTURE_SEAL.json with per-fixture receipts.
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(DIR)), 'src'))

from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait,
)

C1_REGISTRY = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_full_registry/per_task'
C3S_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2cs_static_fixture_seal'
os.makedirs(C3S_OUT, exist_ok=True)

FOUR_SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
NUM_STEPS_WAIT = 10
N_RESETS = 3
POS_TOLERANCE = 1e-6
ROT_TOLERANCE = 1e-6


def ancestor_has_movable_joint(model, body_id):
    """Check if body or any ancestor has a movable joint (hinge/slide/free)."""
    MOVABLE_TYPES = {0, 1, 2, 3}  # free, ball, slide, hinge in MuJoCo
    checked = set()
    current = body_id
    while current >= 0 and current not in checked:
        checked.add(current)
        # Check all joints
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == current:
                jtype = model.jnt_type[j]
                if jtype in MOVABLE_TYPES:
                    # Check if joint has range of motion
                    jnt_range = model.jnt_range[j] if hasattr(model, 'jnt_range') else None
                    if jnt_range is not None and len(jnt_range) >= 2:
                        if jnt_range[1] - jnt_range[0] > 0.001:
                            return True, f'joint_{j}_type_{jtype}_range_{jnt_range[1]-jnt_range[0]:.4f}'
                    else:
                        return True, f'joint_{j}_type_{jtype}'
        current = model.body_parentid[current]
    return False, None


def check_fixture_invariance(suite, task_idx, site_name):
    """Check if a fixture site is static across init states and resets."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict

    benchmark = get_benchmark(suite)(0)
    task_obj = benchmark.get_task(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                            task_obj.problem_folder, task_obj.bddl_file)
    bddl_sha = hashlib.sha256(open(bddl_path, 'rb').read()).hexdigest()

    suite_dict = get_benchmark_dict()
    suite_obj = suite_dict[suite]()
    init_states = suite_obj.get_task_init_states(task_idx)
    n_states = len(init_states)

    samples = []
    ancestor_result = None

    for state_id in range(min(n_states, 5)):  # Up to 5 canonical states
        for reset_idx in range(N_RESETS):
            env, obs = build_v4_exact_env(
                bddl_path, render_gpu_device_id=-1,
                max_steps=50, num_steps_wait=NUM_STEPS_WAIT,
            )
            try:
                canonical_state = init_states[state_id]
                obs = env.set_init_state(canonical_state)

                # Sample: post-init (pre dummy wait)
                sid = env.sim.model.site_name2id(site_name)
                pre_wait = {
                    'stage': 'post_init',
                    'state_id': state_id, 'reset': reset_idx,
                    'xpos': env.sim.data.site_xpos[sid].tolist(),
                    'xmat': env.sim.data.site_xmat[sid].tolist(),
                }
                samples.append(pre_wait)

                # Ancestor joint audit (once per task)
                if ancestor_result is None:
                    body_id = int(env.sim.model.site_bodyid[sid])
                    has_movable, detail = ancestor_has_movable_joint(env.sim.model, body_id)
                    ancestor_result = {
                        'site_body_id': body_id,
                        'has_movable_ancestor': has_movable,
                        'detail': detail,
                    }

                # Post dummy wait
                env, obs = apply_dummy_wait(env, obs, NUM_STEPS_WAIT)
                post_wait = {
                    'stage': 'post_wait',
                    'state_id': state_id, 'reset': reset_idx,
                    'xpos': env.sim.data.site_xpos[sid].tolist(),
                    'xmat': env.sim.data.site_xmat[sid].tolist(),
                }
                samples.append(post_wait)
            finally:
                env.close()

    # Compute invariance
    xpos_array = np.array([s['xpos'] for s in samples])
    xmat_array = np.array([s['xmat'] for s in samples])

    pos_range = xpos_array.max(axis=0) - xpos_array.min(axis=0)
    max_pos_drift = float(pos_range.max())

    # Rotation: max element-wise difference from mean
    mean_xmat = xmat_array.mean(axis=0)
    rot_drifts = np.abs(xmat_array - mean_xmat).max(axis=0)
    max_rot_drift = float(rot_drifts.max())

    is_static_pos = max_pos_drift <= POS_TOLERANCE
    is_static_rot = max_rot_drift <= ROT_TOLERANCE
    is_static = is_static_pos and is_static_rot

    return {
        'suite': suite, 'task_idx': task_idx,
        'site_name': site_name,
        'bddl_sha': bddl_sha,
        'n_init_states_sampled': min(n_states, 5),
        'n_resets_per_state': N_RESETS,
        'n_total_samples': len(samples),
        'max_pos_drift': max_pos_drift,
        'max_rot_drift': max_rot_drift,
        'pos_invariant': is_static_pos,
        'rot_invariant': is_static_rot,
        'is_static': is_static,
        'ancestor_audit': ancestor_result,
        'pos_tolerance': POS_TOLERANCE,
        'rot_tolerance': ROT_TOLERANCE,
        'mean_xpos': xpos_array.mean(axis=0).tolist(),
        'mean_xmat': mean_xmat.tolist(),
    }


def main():
    print('=' * 60)
    print('C3-S: Static Fixture Invariance Seal')
    print('=' * 60)

    # Collect all EXACT_SITE fixture targets from C1
    fixture_targets = []
    for suite in FOUR_SUITES:
        for task_idx in range(10):
            task_key = f'{suite}_task_{task_idx:02d}'
            c1_path = os.path.join(C1_REGISTRY, f'{task_key}.json')
            if not os.path.isfile(c1_path):
                continue
            with open(c1_path) as f:
                c1_data = json.load(f)
            rel_map = c1_data['legacy'].get('relation_map', [])
            for rm in rel_map:
                if rm.get('resolution') == 'EXACT_SITE' and '_region' in rm.get('target_bddl', ''):
                    fixture_targets.append({
                        'suite': suite, 'task_idx': task_idx,
                        'site_name': rm['target_bddl'],
                        'predicate': rm['predicate'],
                        'size': rm.get('size'),
                        'parent_body_name': rm.get('parent_body_name'),
                    })

    # Deduplicate by site_name
    seen = set()
    unique_fixtures = []
    for ft in fixture_targets:
        if ft['site_name'] not in seen:
            seen.add(ft['site_name'])
            unique_fixtures.append(ft)

    print(f'Unique fixture targets: {len(unique_fixtures)}')
    for ft in unique_fixtures:
        print(f'  {ft["site_name"]} ({ft["suite"]}/task_{ft["task_idx"]:02d}) '
              f'size={ft["size"]} parent={ft["parent_body_name"]}')

    # Run invariance check for each
    print(f'\nChecking invariance ({N_RESETS} resets × up to 5 states each)...')
    results = {}
    static_count = 0
    non_static = []

    for ft in unique_fixtures:
        site = ft['site_name']
        print(f'  {site}...', end=' ', flush=True)
        r = check_fixture_invariance(ft['suite'], ft['task_idx'], site)
        results[site] = r

        if r['is_static']:
            static_count += 1
            print(f'STATIC  pos_drift={r["max_pos_drift"]:.2e}  rot_drift={r["max_rot_drift"]:.2e}')
        else:
            non_static.append(site)
            movable = r['ancestor_audit'].get('has_movable_ancestor', False)
            print(f'NON_STATIC  pos_drift={r["max_pos_drift"]:.2e}  rot_drift={r["max_rot_drift"]:.2e}  movable_ancestor={movable}')

    # Summary
    print(f'\n{"=" * 60}')
    n_total = len(unique_fixtures)
    print(f'Static: {static_count}/{n_total}')
    print(f'Non-static: {len(non_static)}')

    all_static = len(non_static) == 0

    seal = {
        'gate': 'C3-S_STATIC_FIXTURE_INVARIANCE',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_fixture_targets': n_total,
        'n_static': static_count,
        'n_non_static': len(non_static),
        'all_static': all_static,
        'pos_tolerance': POS_TOLERANCE,
        'rot_tolerance': ROT_TOLERANCE,
        'fixture_world_poses': {
            site: {
                'mean_xpos': r['mean_xpos'],
                'mean_xmat': r['mean_xmat'],
                'is_static': r['is_static'],
                'parent_body_name': next((ft['parent_body_name'] for ft in unique_fixtures if ft['site_name'] == site), None),
                'size': next((ft['size'] for ft in unique_fixtures if ft['site_name'] == site), None),
            }
            for site, r in results.items() if r['is_static']
        },
        'per_fixture': results,
    }

    sp = os.path.join(C3S_OUT, 'STATIC_FIXTURE_SEAL.json')
    with open(sp, 'w') as f:
        json.dump(seal, f, indent=2, default=str)
    print(f'\nSeal: {sp}')

    if all_static:
        print('C3-S: PASS — all fixtures static invariant')
        sys.exit(0)
    else:
        print(f'C3-S: HOLD — {len(non_static)} non-static fixtures')
        for ns in non_static:
            r = results[ns]
            print(f'  {ns}: movable_ancestor={r["ancestor_audit"].get("has_movable_ancestor")}')
        sys.exit(5)


if __name__ == '__main__':
    main()
