"""Verify D2.1.2 unknown propagation on server."""
import sys, json
sys.path.insert(0, '/tmp')
from v5_physics import _candidate_close, derive_episode_rows, parse_bddl_task_role
from action_contract import CanonicalActionState
from copy import deepcopy

protocol = {
    'schema': 'DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL',
    'fixed_constants': {
        'relative_position_scale_m': 0.1, 'relative_quaternion_scale': 0.1,
        'lift_scale_m': 0.02, 'target_progress_scale_m': 0.05,
        'minimum_stable_grasp_dwell': 10, 'tier2_min_stable_grasp': 0.5,
        'tier2_max_release_risk': 0.6, 'tier2_max_regrasp_risk': 0.6,
        'tier1_min_utility': 0.4, 'tier3_min_stable_grasp': 0.3,
    },
    'candidate_close': {'close_threshold': 0.5},
    'history': {'score_window_steps': 10, 'minimum_stable_grasp_dwell': 10},
    'window_policy': {'loader_preserve_candidate_segment': True},
}
bddl = '(define (problem test)\n(:objects\n  obj_1 - cube\n  target_1 - plate\n)\n(:init\n  (On obj_1 floor_region)\n)\n(:goal\n  (And (In obj_1 target_1_contain_region))\n)\n)\n'
role = parse_bddl_task_role(bddl, suite='libero_object', task_idx=0, object_names=['obj_1', 'target_1'])
print('Role: applic={} status={} manipulated={}'.format(role.applicable, role.status, role.manipulated_objects))

steps = []
sidecars = []
for i in range(12):
    if i < 4: raw_val = 0.0
    elif i < 8: raw_val = 0.5  # boundary
    else: raw_val = 1.0  # open
    steps.append({'clean_action_raw_7d': [0,0,0,0,0,0,raw_val], 'valid': True})
    sidecars.append({
        'object_state': [0.1*i/11,0,0.03*i/11,1,0,0,0, 0.1,0,0,1,0,0,0],
        'robot0_eef_pos': [0.1*i/11,0,0.03*i/11],
        'robot0_gripper_qpos': [0,0],
        'mujoco_contact_pairs': [['obj_1_g1','gripper0_finger1_pad_collision']],
    })

rows, _ = derive_episode_rows(steps, sidecars, role,
    {'obj_1': {'pos':[0,3],'quat':[3,7],'to_eef_pos':[7,10],'to_eef_quat':[10,14]}}, protocol)

# Check boundary steps (indices 4-7): action_known=False
for i in range(4, 8):
    r = rows[i]
    state = steps[i].get('_action_state')
    print('step {}: cc={} known={} phase={} intent={}'.format(
        i, r['candidate_close'], r['known_mask'], r['phase_name'],
        state.action_intent if state else 'NONE'))
    assert not r['candidate_close'], 'boundary must not be close'
    assert not r['known_mask'], 'boundary must have known_mask=False'
    assert r['phase_name'] == 'UNKNOWN', 'boundary must be UNKNOWN'

# Check known CLOSE steps (indices 0-3)
for i in range(0, 4):
    r = rows[i]
    state = steps[i].get('_action_state')
    print('step {}: cc={} known={} intent={}'.format(
        i, r['candidate_close'], r['known_mask'], state.action_intent if state else 'NONE'))
    assert r['candidate_close'], 'raw=0.0 must be candidate close'
    assert r['known_mask'], 'known CLOSE must have known_mask=True'

# Check OPEN steps (indices 8-11)
for i in range(8, 12):
    r = rows[i]
    state = steps[i].get('_action_state')
    print('step {}: cc={} known={} intent={}'.format(
        i, r['candidate_close'], r['known_mask'], state.action_intent if state else 'NONE'))
    assert not r['candidate_close'], 'raw=1.0 must not be candidate close'
    assert r['known_mask'], 'known OPEN must have known_mask=True'

print('ALL CHECKS PASSED')
