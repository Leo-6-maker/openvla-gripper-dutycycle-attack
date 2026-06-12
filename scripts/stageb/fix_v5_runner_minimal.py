#!/usr/bin/env python3
"""Apply minimal correct v5 fixes to reverted runner. Avoids harden regressions."""
import re

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
path = f'{REPO}/scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py'

with open(path) as f:
    src = f.read()

fixes = []

# 1. Fix action_dim (line ~84)
old = 'action_dim = model.config.pad_to_multiple_of'
new = "unnorm_key = 'libero_object'\naction_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7, f'Unexpected action_dim={action_dim}'"
if old in src:
    src = src.replace(old, new)
    fixes.append('1. action_dim = int(model.get_action_dim())')

# 2. Add V4 dummy wait (before max_steps, after env init)
old = "    max_steps = min(args.max_steps_override, init_states.shape[1] if init_states.ndim == 2 else args.max_steps_override)"
new = "    # V4-aligned dummy wait\n    dummy_action = [0, 0, 0, 0, 0, 0, -1]\n    for _ in range(args.num_steps_wait):\n        obs, _, _, _ = env.step(dummy_action)\n\n    max_steps = min(args.max_steps_override, init_states.shape[1] if init_states.ndim == 2 else args.max_steps_override)"
if old in src:
    src = src.replace(old, new)
    fixes.append('2. V4 dummy wait')

# 3. Fix success metrics
old = "        success_done = info.get('success_done', 0) if isinstance(info, dict) else 0"
new = "        success_done = bool(done)"
if old in src:
    src = src.replace(old, new)
    fixes.append('3a. success_done = bool(done)')

old = "        success_check = info.get('success_check', 0) if isinstance(info, dict) else 0"
new = "        success_check = bool(env.check_success())"
if old in src:
    src = src.replace(old, new)
    fixes.append('3b. success_check = bool(env.check_success())')

# Fix the ternary logic: committed has WRONG order
old = "        success_primary_now = success_done if args.success_metric == 'check_success' else success_check"
new = "        success_primary_now = success_done if args.success_metric == 'done' else success_check"
if old in src:
    src = src.replace(old, new)
    fixes.append('3c. success_primary_now ternary corrected')

# 4. Add attack_result None → RuntimeError (the else branch)
old = """                    if attack_result is not None:
                        adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                        if adv_inputs is not None and len(adv_inputs) > 0:"""
new = """                    if attack_result is None:
                        raise RuntimeError('V5 HARD FAIL: attack_result is None for token_pgd')
                    else:
                        adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                        if adv_inputs is not None and len(adv_inputs) > 0:"""
if old in src:
    src = src.replace(old, new)
    fixes.append('4. RuntimeError on attack_result None')

# 5. Add adv_inputs missing → RuntimeError
old = """                        else:
                            infra_status = '%s_no_adv_inputs' % args.condition"""
new = """                        else:
                            raise RuntimeError('V5 HARD FAIL: adv_inputs missing for token_pgd')"""
if old in src:
    src = src.replace(old, new)
    fixes.append('5. RuntimeError on adv_inputs missing')

# 6. Fix except block to re-raise for VIS
old = """                except Exception as e:
                    infra_status = 'vis_error: %s' % str(e)[:80]"""
new = """                except Exception as e:
                    raise"""
if old in src:
    src = src.replace(old, new)
    fixes.append('6. VIS except re-raises')

# 7. Add eps_norm global scope (before attacker init)
old = "# ── Attack setup (v5: FIXED) ──"
new = "# ── Attack setup (v5: FIXED) ──\neps_norm = args.eps_raw_pixels / 255.0"
if old in src and 'eps_norm = args' not in src.split(old)[0].split('\n')[-5:]:
    src = src.replace(old, new)
    # Also remove eps_norm from inside vis_pgd block (duplicate)
    src = src.replace("\n\n    eps_norm = args.eps_raw_pixels / 255.0", "")
    fixes.append('7. eps_norm global scope')

# 8. Add step_v5 per-step init
old = "        pgd_applied = 0; perturbation_space = 'none'"
new = """        pgd_applied = 0; perturbation_space = 'none'
        step_v5 = {'attack_method': '', 'token_label_source': '',
            'target_ce_initial': '', 'target_ce_final': '', 'loss_decrease': '',
            'gripper_logit_margin_after': '', 'open_region_prob_mass_after': '',
            'close_bin_prob_mass_after': '', 'corrected_open_token_count': '',
            'pixel_budget_adv_inputs_linf': '', 'adv_decode_path': '',
            'used_adv_inputs': False, 'fallback_adapter_used': False,
            'adv_gripper_raw': '', 'adv_env_gripper': '', 'adv_open_bool': ''}"""
if old in src:
    src = src.replace(old, new)
    fixes.append('8. step_v5 per-step init')

# 9. Add per-step v5 telemetry recording in the PGD success branch
old_pgd = "                            perturbation_space = 'token_prefix_pgd_adv_inputs_v5'"
new_pgd = """                            perturbation_space = 'token_prefix_pgd_adv_inputs_v5'

                            step_v5['adv_decode_path'] = 'token_pgd_adv_inputs_generate'
                            step_v5['adv_gripper_raw'] = float(executed_action[-1])
                            step_v5['adv_env_gripper'] = float(env_action[-1])
                            step_v5['adv_open_bool'] = int(env_action[-1] < -0.5)
                            step_v5['used_adv_inputs'] = True"""
if old_pgd in src:
    src = src.replace(old_pgd, new_pgd)
    fixes.append('9a. step_v5 per-step PGD success fields')

# Also switch telemetry writes from v5_telemetry to step_v5
old_tel = "                            v5_telemetry['attack_method']"
new_tel = "                            step_v5['attack_method']"
if old_tel in src:
    src = src.replace(old_tel, new_tel)
    fixes.append('9b. telemetry: attack_method → step_v5')

old_tel = "                            v5_telemetry['token_label_source']"
new_tel = "                            step_v5['token_label_source']"
if old_tel in src:
    src = src.replace(old_tel, new_tel)
    fixes.append('9c. telemetry: token_label_source → step_v5')

old_tel = "                            v5_telemetry['target_ce_initial']"
new_tel = "                            step_v5['target_ce_initial']"
if old_tel in src:
    src = src.replace(old_tel, new_tel)
    fixes.append('9d. telemetry: target_ce_initial → step_v5')

old_tel = "                            v5_telemetry['target_ce_final']"
new_tel = "                            step_v5['target_ce_final']"
if old_tel in src:
    src = src.replace(old_tel, new_tel)
    fixes.append('9e. telemetry: target_ce_final → step_v5')

# Add remaining step_v5 field writes from debug info
# These are currently written to v5_telemetry but should also go to step_v5
# Actually, let me check what v5_telemetry fields are set from debug
# The committed version writes these to v5_telemetry dict from dbg
# For the trace, we need them per-step. Let me add step_v5 writes after the existing v5_telemetry writes

old = "                            v5_telemetry['loss_decrease']"
new = "                            step_v5['loss_decrease']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9f. telemetry: loss_decrease → step_v5')

old = "                            v5_telemetry['open_region_prob_mass_after']"
new = "                            step_v5['open_region_prob_mass_after']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9g. telemetry: open_region_prob_mass_after → step_v5')

old = "                            v5_telemetry['close_bin_prob_mass_after']"
new = "                            step_v5['close_bin_prob_mass_after']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9h. telemetry: close_bin_prob_mass_after → step_v5')

old = "                            v5_telemetry['gripper_logit_margin_after']"
new = "                            step_v5['gripper_logit_margin_after']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9i. telemetry: gripper_logit_margin_after → step_v5')

old = "                            v5_telemetry['corrected_open_token_count']"
new = "                            step_v5['corrected_open_token_count']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9j. telemetry: corrected_open_token_count → step_v5')

old = "                            v5_telemetry['pixel_budget_adv_inputs_linf']"
new = "                            step_v5['pixel_budget_adv_inputs_linf']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9k. telemetry: pixel_budget_adv_inputs_linf → step_v5')

old = "                            v5_telemetry['fallback_adapter_used']"
new = "                            step_v5['fallback_adapter_used']"
if old in src:
    src = src.replace(old, new)
    fixes.append('9l. telemetry: fallback_adapter_used → step_v5')

# 10. Add v5 per-step fields to trace CSV row
# The trace row currently doesn't have v5 fields. Add them.
old_trace = "            'infra_status': infra_status, 'window_start': ws, 'window_end': we,\n        })"
new_trace = """            'infra_status': infra_status, 'window_start': ws, 'window_end': we,
            # v5 per-step telemetry
            'attack_method': step_v5.get('attack_method', '') if pgd_applied else '',
            'token_label_source': step_v5.get('token_label_source', '') if pgd_applied else '',
            'target_ce_initial': round(float(step_v5.get('target_ce_initial', '') or 0), 6) if pgd_applied else '',
            'target_ce_final': round(float(step_v5.get('target_ce_final', '') or 0), 6) if pgd_applied else '',
            'loss_decrease': round(float(step_v5.get('loss_decrease', '') or 0), 6) if pgd_applied else '',
            'gripper_logit_margin_after': round(float(step_v5.get('gripper_logit_margin_after', '') or 0), 6) if pgd_applied else '',
            'open_region_prob_mass_after': round(float(step_v5.get('open_region_prob_mass_after', '') or 0), 6) if pgd_applied else '',
            'close_bin_prob_mass_after': round(float(step_v5.get('close_bin_prob_mass_after', '') or 0), 6) if pgd_applied else '',
            'corrected_open_token_count': step_v5.get('corrected_open_token_count', '') if pgd_applied else '',
            'pixel_budget_adv_inputs_linf': round(float(step_v5.get('pixel_budget_adv_inputs_linf', '') or 0), 8) if pgd_applied else '',
            'adv_decode_path': step_v5.get('adv_decode_path', '') if pgd_applied else '',
            'used_adv_inputs': step_v5.get('used_adv_inputs', False) if pgd_applied else False,
            'fallback_adapter_used': step_v5.get('fallback_adapter_used', False) if pgd_applied else False,
            'adv_gripper_raw': round(float(executed_action[-1]), 6) if pgd_applied else '',
            'adv_env_gripper': round(float(env_action[-1]), 6) if pgd_applied else '',
            'adv_open_bool': is_open if pgd_applied else '',
        })"""

if old_trace in src:
    src = src.replace(old_trace, new_trace)
    fixes.append('10. v5 per-step fields in trace')

with open(path, 'w') as f:
    f.write(src)

print('Applied %d fixes:' % len(fixes))
for f in fixes:
    print('  %s' % f)
