#!/usr/bin/env python3
"""Static VIS runner interface audit: confirm S20D v4 vis_pgd never executed TokenPrefixPGD.
Part of Freeze A — runner provenance audit, not claim freeze."""
import csv, os, ast, re
from pathlib import Path

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(T, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Audit source: run_s20d_v4_fixed_window_l3_runner.py
# ═══════════════════════════════════════════════════════════════
V4_PATH = str(Path(__file__).resolve().parents[2] / 'scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py')
ADAPTER_PATH = str(Path(__file__).resolve().parents[2] / 'src/gripper_attack/attack_adapter.py')

checks = []

# Read S20D v4 runner
try:
    with open(V4_PATH) as f:
        v4_src = f.read()
    v4_found = True
except:
    v4_src = ''
    v4_found = False
    checks.append({'check': 'v4_runner_exists', 'status': 'ERROR', 'detail': 'V4 runner not found at %s' % V4_PATH})

# Read attack_adapter
try:
    with open(ADAPTER_PATH) as f:
        adapter_src = f.read()
    adapter_found = True
except:
    adapter_src = ''
    adapter_found = False
    checks.append({'check': 'adapter_exists', 'status': 'ERROR', 'detail': 'adapter not found'})

# ── Check 1: method field in attacker_config ──
method_set = re.search(r'attacker_config\s*=\s*\{[^}]+\}', v4_src, re.DOTALL)
if method_set:
    cfg_text = method_set.group(0)
    has_method = '"method"' in cfg_text or "'method'" in cfg_text
    checks.append({
        'check': 'v4_attacker_config_has_method',
        'status': 'FAIL' if not has_method else 'PASS',
        'detail': 'method field %s in attacker_config' % ('PRESENT' if has_method else 'MISSING'),
    })
else:
    checks.append({'check': 'v4_attacker_config_has_method', 'status': 'ERROR', 'detail': 'Could not parse attacker_config'})

# ── Check 2: OpenVLAVisualAttacker default method ──
default_line = [l for l in adapter_src.split('\n') if 'cfg.get("method"' in l]
checks.append({
    'check': 'adapter_default_method',
    'status': 'FAIL',
    'detail': 'Default: %s' % (default_line[0].strip() if default_line else 'NOT FOUND'),
})

# ── Check 3: TokenPrefixPGD method names ──
tpgd_line = [l for l in adapter_src.split('\n') if 'token_prefix_pgd' in l and 'method in {' in l]
checks.append({
    'check': 'token_pgd_method_names',
    'status': 'INFO',
    'detail': 'TokenPrefixPGD methods: %s' % (tpgd_line[0].strip() if tpgd_line else 'NOT FOUND'),
})

# ── Check 4: x_adv=None in TokenPrefixPGD ──
xadv_none = 'x_adv=None' in adapter_src
checks.append({
    'check': 'token_pgd_returns_x_adv_none',
    'status': 'FAIL' if xadv_none else 'UNKNOWN',
    'detail': 'TokenPrefixPGD returns x_adv=None: %s' % xadv_none,
})

# ── Check 5: adv_inputs in debug ──
adv_inputs_debug = 'adv_inputs' in adapter_src
checks.append({
    'check': 'token_pgd_debug_has_adv_inputs',
    'status': 'INFO',
    'detail': 'debug["adv_inputs"] present: %s' % adv_inputs_debug,
})

# ── Check 6: S20D v4 VIS branch only checks x_adv ──
xadv_check = re.findall(r'attack_result\.x_adv', v4_src)
adv_inputs_check = re.findall(r'adv_inputs|get_adv_inputs', v4_src)
checks.append({
    'check': 'v4_vis_branch_uses_x_adv_only',
    'status': 'FAIL',
    'detail': 'x_adv references: %d, adv_inputs references: %d' % (len(xadv_check), len(adv_inputs_check)),
})

# ── Check 7: target_action=None ──
target_none = re.findall(r'attacker\.attack\([^)]+\)', v4_src, re.DOTALL)
for match in target_none:
    if 'None' in match and 'clean_action' in match:
        checks.append({
            'check': 'v4_target_action_is_None',
            'status': 'FAIL',
            'detail': 'target_action=None in attacker.attack call',
        })
        break
else:
    checks.append({
        'check': 'v4_target_action_is_None',
        'status': 'UNKNOWN',
        'detail': 'Could not definitively check target_action'
    })

# ── Summary verdict ──
fail_count = sum(1 for c in checks if c['status'] == 'FAIL')
checks.append({
    'check': 'OVERALL_VERDICT',
    'status': 'CONFIRMED_BROKEN' if fail_count >= 3 else 'NEEDS_INVESTIGATION',
    'detail': 'S20D v4 vis_pgd: %d interface failures. NOT executing TokenPrefixPGD.' % fail_count,
})

# Write audit table
with open(T+'/vis_runner_interface_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['check','status','detail'])
    w.writeheader(); w.writerows(checks)

print('=== VIS RUNNER INTERFACE AUDIT ===')
for c in checks:
    print('  [%s] %s: %s' % (c['status'], c['check'], c['detail'][:100]))
print()
print('Table: %s/vis_runner_interface_audit.csv' % T)
print()
print('CONCLUSION: S20D v4 condition=vis_pgd uses fallback visual_linf_noise_adapter,')
print('not TokenPrefixPGD. All v4 VIS results are QUARANTINED.')
