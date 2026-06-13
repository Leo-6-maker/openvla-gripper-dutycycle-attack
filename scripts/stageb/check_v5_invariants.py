#!/usr/bin/env python3
"""Verify 12 v5 runner invariants."""
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py'

with open(path) as f:
    src = f.read()

checks = {}
checks['1. get_action_dim+assert7'] = 'int(model.get_action_dim(unnorm_key))' in src and 'assert action_dim == 7' in src
checks['2. no pad_to_multiple_of'] = 'action_dim = model.config.pad_to_multiple_of' not in src
checks['3. V4 dummy wait'] = 'dummy_action = [0, 0, 0, 0, 0, 0, -1]' in src and 'num_steps_wait' in src
checks['4. eps_norm global'] = src.count('eps_norm = args.eps_raw_pixels / 255.0') == 1
checks['5. success_done=bool(done)'] = 'success_done = bool(done)' in src
checks['6. success_check=bool(env.check_success())'] = 'success_check = bool(env.check_success())' in src
checks['7. success_metric direction'] = "success_primary_now = success_done if args.success_metric == 'done' else success_check" in src
checks['8. attack_result None raises'] = 'attack_result is None' in src and 'RuntimeError' in src
checks['9. adv_inputs missing raises'] = 'adv_inputs' in src and 'RuntimeError' in src
checks['10. VIS except re-raises'] = 'except Exception' in src and 'raise' in src.split('if args.condition')[1] if 'if args.condition' in src else False
checks['11. step_v5 per-step'] = "step_v5 = {'attack_method':" in src or "step_v5 = {\n            'attack_method':" in src
checks['12. trace v5 fields'] = 'adv_decode_path' in src and 'used_adv_inputs' in src and 'adv_open_bool' in src

passed = sum(1 for v in checks.values() if v)
for k, v in checks.items():
    print("  %s: %s" % ("PASS" if v else "FAIL", k))

print("\n%d/12 invariants PASS" % passed)
if passed == 12:
    print("GATE G0 PASSED")
else:
    print("GATE G0 FAILED")
    sys.exit(1)
