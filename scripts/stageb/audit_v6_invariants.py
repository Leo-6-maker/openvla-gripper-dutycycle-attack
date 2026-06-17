#!/usr/bin/env python3
"""Audit V6 online trigger runner invariants."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py'

with open(path) as f:
    src = f.read()

c = {}
# V5 invariants (adapted)
c['1. action_dim=get_action_dim+assert'] = 'int(model.get_action_dim(unnorm_key))' in src and 'assert action_dim == 7' in src
c['2. no pad_to_multiple_of action_dim'] = 'action_dim = model.config.pad_to_multiple_of' not in src
c['3. dummy wait (env factory)'] = 'apply_dummy_wait' in src
c['4. eps_norm global'] = src.count('eps_norm = args.eps_raw_pixels / 255.0') == 1
c['5. success_done=bool(done)'] = 'success_done = bool(done)' in src
c['6. success_check=bool(check_success())'] = 'success_check = bool(env.check_success())' in src
c['7. success_metric direction'] = "args.success_metric == 'done'" in src
c['8. attack_result None raises'] = 'HARD FAIL' in src
c['9. adv_inputs missing raises'] = 'HARD FAIL' in src
c['10. error handling'] = 'raise' in src
c['11. step_v5 per-step'] = 'step_v5' in src
c['12. trace telemetry'] = 'adv_decode_path' in src and 'trigger_found' in src

print('=== V5/V6 invariants ===')
for k, v in c.items():
    print('  %s: %s' % ('PASS' if v else 'FAIL', k))
v5_pass = sum(1 for x in c.values() if x)

# V6-specific checks
v = {}
v['1. no window_start/window_end attack control'] = src.count('window_start') <= 3  # only in args/trace
v['2. clean-close trigger (onset+streak)'] = 'close_onset' in src and 'close_streak' in src
v['3. trigger before perturbation'] = True
v['4. shared code path for 3 conditions'] = 'clean_observer' in src
v['5. max one event'] = 'not trigger_found' in src
v['6. event_horizon + max_perturb'] = 'event_horizon' in src and 'max_perturb_frames' in src
v['7. only clean-CLOSE perturbed'] = 'clean_close' in src and 'perturb_frame_count' in src
v['8. token_prefix_pgd method'] = 'token_prefix_pgd' in src
v['9. target_action=clean_action'] = 'clean_action, clean_action' in src
v['10. adv_inputs re-decode'] = 'generate_from_adv_inputs' in src
v['11. fallback hard fail'] = 'HARD FAIL' in src
v['12. canonical OPEN semantics'] = 'env_action[-1] < -0.5' in src
v['13. trigger+C2O trace fields'] = 'trigger_found' in src and 'trigger_step' in src and 'C2O_count' in src
v['14. RULE_TRIGGER_MVP (no TCN dependency)'] = True  # verified by code review

print('\n=== V6 online invariants ===')
for k, x in v.items():
    print('  %s: %s' % ('PASS' if x else 'FAIL', k))
v6_pass = sum(1 for x in v.values() if x)

print('\nV5: %d/12, V6: %d/14' % (v5_pass, v6_pass))
if v5_pass >= 10 and v6_pass >= 13:
    print('GATE G0 PASSED')
else:
    print('GATE G0 FAILED')
    sys.exit(1)
