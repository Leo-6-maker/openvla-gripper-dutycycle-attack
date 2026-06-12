#!/usr/bin/env python3
"""Apply P0/P1 fixes to v5 runner per code review."""
import re

path = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py'
with open(path) as f:
    src = f.read()

fixes_applied = []

# P0-2: Add dummy wait after set_init_state, before max_steps
old = '    max_steps = args.max_steps_override'
new = '''    # V4-aligned dummy wait (P0 fix: was missing)
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(dummy_action)

    max_steps = args.max_steps_override'''
if old in src and new not in src:
    src = src.replace(old, new)
    fixes_applied.append('P0-2: dummy_wait')

# P0-3: Fix success metric logic (was inverted)
old_succ = "success_primary_now = success_done if args.success_metric == 'check_success' else success_check"
new_succ = "success_primary_now = success_done if args.success_metric == 'done' else success_check"
if old_succ in src:
    src = src.replace(old_succ, new_succ)
    fixes_applied.append('P0-3: success_metric')

# P0-8: adv_inputs missing -> hard fail instead of silent continue
old_hf = """                            infra_status = 'v5_token_pgd_no_adv_inputs'
                            v5_telemetry['fallback_adapter_used'] = True"""
new_hf = """                            infra_status = 'v5_token_pgd_no_adv_inputs'
                            raise RuntimeError('V5 HARD FAIL: adv_inputs missing for token_pgd')"""
if old_hf in src:
    src = src.replace(old_hf, new_hf)
    fixes_applied.append('P0-8: adv_inputs_hard_fail')

# P1-9: Fix EEF/object before-after order
# Remove eef_after that was read before env.step, add after
old_eef = """        eef_before = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        eef_after = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        obj_before_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None
        obj_before = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None
        obj_after = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None"""
new_eef = """        eef_before = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        obj_before_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None
        obj_before = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None"""
if old_eef in src:
    src = src.replace(old_eef, new_eef)
    fixes_applied.append('P1-9: eef_before_only')

# P1-9b: Add eef_after/obj_after after env.step
old_after = """        gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))
        is_open = 1 if env_action[-1] < -0.5 else 0

        total_decoded_open += is_open
        if is_open: current_streak += 1
        else: current_streak = 0
        max_streak = max(max_streak, current_streak)


        success_done = info.get('success_done', 0) if isinstance(info, dict) else 0"""
new_after = """        gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))
        eef_after = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        obj_after = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None
        is_open = 1 if env_action[-1] < -0.5 else 0

        total_decoded_open += is_open
        if is_open: current_streak += 1
        else: current_streak = 0
        max_streak = max(max_streak, current_streak)

        success_check = bool(env.check_success())
        success_done = info.get('success_done', 0) if isinstance(info, dict) else 0"""
if old_after in src:
    src = src.replace(old_after, new_after)
    fixes_applied.append('P1-9b: eef_after_order + success_check')

with open(path, 'w') as f:
    f.write(src)

for fix in fixes_applied:
    print(f'  APPLIED: {fix}')
print(f'  Total fixes: {len(fixes_applied)}')
