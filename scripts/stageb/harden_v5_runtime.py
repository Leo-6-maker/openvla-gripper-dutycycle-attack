#!/usr/bin/env python3
"""Runtime hardening for v5 scripts. Fixes P0-A through P0-F from code review of a838b6e."""
import os, sys

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
fixes = []

# ═══ P0-A: Fix one-step smoke unnorm_key order ═══
path = f'{REPO}/scripts/stageb/run_s20d_v5_token_pgd_one_step_smoke.py'
with open(path) as f: src = f.read()
old = "model_dtype = torch.bfloat16\naction_dim = int(model.get_action_dim(unnorm_key))\nunnorm_key = 'libero_object'"
new = "model_dtype = torch.bfloat16\nunnorm_key = 'libero_object'\naction_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7, f'Unexpected action_dim={action_dim}'"
if old in src:
    src = src.replace(old, new); fixes.append('P0-A: one-step smoke unnorm_key order')
with open(path, 'w') as f: f.write(src)

# ═══ P0-B: Fix window smoke action_dim ═══
path = f'{REPO}/scripts/stageb/run_s20d_v5_token_pgd_window_smoke.py'
with open(path) as f: src = f.read()
old = "model_dtype = torch.bfloat16; action_dim = model.config.pad_to_multiple_of"
new = "model_dtype = torch.bfloat16; unnorm_key = 'libero_object'; action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7, f'Unexpected action_dim={action_dim}'"
if old in src:
    src = src.replace(old, new); fixes.append('P0-B: window smoke action_dim')
with open(path, 'w') as f: f.write(src)

# ═══ P0-C: Fix audit script missing Path import ═══
path = f'{REPO}/scripts/stageb/audit_vis_runner_attack_method.py'
with open(path) as f: src = f.read()
if 'from pathlib import Path' not in src:
    src = src.replace("import csv, os, ast, re", "import csv, os, ast, re\nfrom pathlib import Path")
    fixes.append('P0-C: audit Path import')
with open(path, 'w') as f: f.write(src)

# ═══ P0-D: Fix v5 runner EEF/object before-after ordering ═══
path = f'{REPO}/scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py'
with open(path) as f: src = f.read()

# Fix: ensure eef_before/obj_before read BEFORE env.step
# Current broken order: eef_before AFTER env.step
old_order = """        env_action = clean_env_action.copy()
        executed_action = clean_action.copy()

        if attack_this_step:
            if args.condition == 'vis_pgd' and attacker is not None:
                try:
                    attack_result = attacker.attack(
                        img_uint8, instruction,
                        clean_action,
                        clean_action,
                        gen_out,
                        unnorm_key=unnorm_key)"""
new_order = """        env_action = clean_env_action.copy()
        executed_action = clean_action.copy()

        # Read EEF/object BEFORE env.step (P0-D fix)
        eef_before = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None
        obj_before_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None
        obj_before = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None

        if attack_this_step:
            if args.condition == 'vis_pgd' and attacker is not None:
                try:
                    attack_result = attacker.attack(
                        img_uint8, instruction,
                        clean_action,
                        clean_action,
                        gen_out,
                        unnorm_key=unnorm_key)"""

if old_order in src:
    src = src.replace(old_order, new_order)
    fixes.append('P0-D: eef_before BEFORE env.step')

# Fix: remove stale eef_before/obj_before that was AFTER env.step (if still present)
# The old duplicate should be gone now
stale_eef = "eef_before = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None\n        obj_before_id = env.env.object_sites[0] if hasattr(env.env, 'object_sites') and env.env.object_sites else None\n        obj_before = env.sim.data.get_site_xpos(obj_before_id) if obj_before_id is not None else None\n\n        obs, reward, done, info = env.step(env_action)"
# This should NOT be in the file anymore after the above fix

# Fix: ensure eef_after/obj_after defined after env.step
old_after2 = "gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))\n        eef_after = env.env.robots[0]._hand_pos"
new_after2 = "gripper_qpos_after = float(np.sum(gripper_phys_after.get('qpos', [0.0])))\n        eef_after = env.env.robots[0]._hand_pos if hasattr(env.env.robots[0], '_hand_pos') else None"
if old_after2 in src:
    src = src.replace(old_after2, new_after2)

# P0-E: Fix success_check to use env.check_success()
old_succ = "success_check = bool(env.check_success())\n        success_done = info.get('success_done', 0) if isinstance(info, dict) else 0\n        success_check_now = info.get('success_check', 0) if isinstance(info, dict) else 0"
# This might not exist. Check simpler pattern:
old_succ2 = "success_check_now = info.get('success_check', 0) if isinstance(info, dict) else 0"
if old_succ2 in src:
    src = src.replace(old_succ2, "success_check_now = bool(env.check_success())")
    fixes.append('P0-E: success_check via env.check_success()')

# P0-F: Fix eps_norm scope — define globally
old_eps = "# ── Attack setup (v5: FIXED) ──"
new_eps = "# ── Attack setup (v5: FIXED) ──\neps_norm = args.eps_raw_pixels / 255.0  # P0-F: global scope"
if old_eps in src and "eps_norm = args.eps_raw_pixels / 255.0  # P0-F" not in src:
    src = src.replace(old_eps, new_eps)
    # Remove eps_norm from inside vis_pgd block
    src = src.replace("    eps_norm = args.eps_raw_pixels / 255.0\n    attacker_config = {", "    attacker_config = {")
    fixes.append('P0-F: eps_norm global scope')

# P1 telemetry: add per-step v5 fields to trace
old_trace = """            'infra_status': infra_status, 'window_start': ws, 'window_end': we,
        })"""
new_trace = """            'infra_status': infra_status, 'window_start': ws, 'window_end': we,
            # v5 per-step telemetry
            'attack_method': v5_telemetry['attack_method'],
            'token_label_source': v5_telemetry['token_label_source'],
            'target_ce_initial': round(v5_telemetry['target_ce_initial'], 6),
            'target_ce_final': round(v5_telemetry['target_ce_final'], 6),
            'loss_decrease': round(v5_telemetry['loss_decrease'], 6),
            'gripper_logit_margin_after': round(v5_telemetry['gripper_logit_margin_after'], 6),
            'open_region_prob_mass_after': round(v5_telemetry['open_region_prob_mass_after'], 6),
            'close_bin_prob_mass_after': round(v5_telemetry['close_bin_prob_mass_after'], 6),
            'corrected_open_token_count': v5_telemetry['corrected_open_token_count'],
            'pixel_budget_adv_inputs_linf': round(v5_telemetry['pixel_budget_adv_inputs_linf'], 8),
            'adv_decode_path': v5_telemetry['adv_decode_path'],
            'used_adv_inputs': v5_telemetry['used_adv_inputs'],
            'fallback_adapter_used': v5_telemetry['fallback_adapter_used'],
            'adv_gripper_raw': round(float(executed_action[-1]), 6) if pgd_applied else '',
            'adv_env_gripper': round(float(env_action[-1]), 6) if pgd_applied else '',
            'adv_open_bool': is_open if pgd_applied else '',
        })"""
if old_trace in src:
    src = src.replace(old_trace, new_trace)
    fixes.append('P1: per-step v5 telemetry in trace')

with open(path, 'w') as f: f.write(src)

# ═══ Delete fix_v5_p0_p1.py ═══
fix_path = f'{REPO}/scripts/stageb/fix_v5_p0_p1.py'
if os.path.exists(fix_path):
    os.remove(fix_path)
    fixes.append('Deleted fix_v5_p0_p1.py')

for f in fixes:
    print(f'  {f}')
print(f'  Total: {len(fixes)} fixes')
