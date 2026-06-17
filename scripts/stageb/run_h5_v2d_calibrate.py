#!/usr/bin/env python3
"""V2-D: Gripper calibration + object binding + CLEAN reference on GPU(2,6)."""
import json, os, sys, numpy as np
sys.path.insert(0, '/data/liuyu/worktrees/l3_h5_v2_telemetry_d/src')
sys.path.insert(0, '/data/liuyu/worktrees/l3_h5_v2_telemetry_d/scripts')
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path

OUT = '/data/liuyu/outputs/l3_h5_v2_telemetry_d_r1'
os.makedirs(OUT, exist_ok=True)

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(6); init_states = suite.get_task_init_states(6)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

env, obs = build_v4_exact_env(bddl, 6, 400, 10)
obs = env.set_init_state(init_states[11])
env, obs = apply_dummy_wait(env, obs, 10)

calib = {}

# ── Gripper joint mapping ──
gripper_addrs = env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[7:9, 0]]
calib['gripper_joint_indices'] = [int(i) for i in env.sim.model.actuator_trnid[7:9, 0]]
calib['gripper_qpos_addrs'] = [int(a) for a in gripper_addrs]

# ── Object binding ──
obj_name = 'butter_1'
if obj_name in env.sim.model.body_names:
    calib['object_name'] = obj_name
    calib['object_body_id'] = int(env.sim.model.body_name2id(obj_name))
else:
    calib['object_name'] = 'NOT_FOUND'
    calib['object_body_id'] = -1

# ── OPEN/CLOSE calibration ──
results = {}
for cmd_val, label in [(1.0, 'CLOSE'), (-1.0, 'OPEN')]:
    env, obs = build_v4_exact_env(bddl, 6, 400, 10)
    obs = env.set_init_state(init_states[11])
    env, obs = apply_dummy_wait(env, obs, 10)
    for _ in range(20):
        env.sim.data.ctrl[env.sim.model.actuator_trnid[7:9, 0]] = cmd_val
        env.sim.step()
    q7 = float(env.sim.data.qpos[gripper_addrs[0]])
    q8 = float(env.sim.data.qpos[gripper_addrs[1]])
    results[label] = {'qpos_7': q7, 'qpos_8': q8, 'qpos_sum': q7 + q8}
    env.close()

calib['close_state'] = results['CLOSE']
calib['open_state'] = results['OPEN']
calib['close_qpos_sum'] = results['CLOSE']['qpos_sum']
calib['open_qpos_sum'] = results['OPEN']['qpos_sum']

# Direction: negative = OPEN (qpos decreases when opening)
calib['qpos_open_direction'] = 'negative' if calib['open_qpos_sum'] < calib['close_qpos_sum'] else 'positive'
calib['qpos_range'] = abs(calib['open_qpos_sum'] - calib['close_qpos_sum'])

print('Calibration:')
print('  Close qpos_sum: {:.4f}'.format(calib['close_qpos_sum']))
print('  Open qpos_sum: {:.4f}'.format(calib['open_qpos_sum']))
print('  Direction: {}'.format(calib['qpos_open_direction']))
print('  Range: {:.4f}'.format(calib['qpos_range']))
print('  Object: {} id={}'.format(calib['object_name'], calib['object_body_id']))

with open(os.path.join(OUT, 'calibration.json'), 'w') as f:
    json.dump(calib, f, indent=2)

# ── Contact proxy thresholds (from initial obs) ──
env, obs = build_v4_exact_env(bddl, 6, 400, 10)
obs = env.set_init_state(init_states[11])
env, obs = apply_dummy_wait(env, obs, 10)
obj_xyz = env.sim.data.body_xpos[calib['object_body_id']]
calib['object_initial_xyz'] = [float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])]
calib['lift_threshold'] = 0.02  # 2cm above initial
calib['detach_threshold'] = 0.05  # 5cm EEF-object distance
env.close()

print('  Object initial z: {:.4f}'.format(calib['object_initial_xyz'][2]))

with open(os.path.join(OUT, 'calibration.json'), 'w') as f:
    json.dump(calib, f, indent=2)
print('V2-D calibration saved to {}'.format(OUT))
