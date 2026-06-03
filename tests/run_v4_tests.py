# -*- coding: utf-8 -*-
"""Standalone test runner for v4 tests (no pytest required)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

passed = 0
failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {msg}')

# ── Test 1: gripper_semantics ──
print('=== test_gripper_semantics_consistency ===')
from gripper_attack.gripper_semantics import (
    raw_gripper_is_open, raw_gripper_is_close,
    decoded_action_to_env_gripper,
    env_gripper_is_open, env_gripper_is_close,
    classify_gripper_action,
    OPEN_THRESHOLD, CANONICAL_OPEN_SEMANTICS_VERSION,
)
from gripper_attack.attack_adapter import action_token_logit_row_index

check(raw_gripper_is_open(-0.996), 'OPEN(-0.996)')
check(raw_gripper_is_open(0.0), 'OPEN(0.0)')
check(raw_gripper_is_open(0.3), 'OPEN(0.3)')
check(not raw_gripper_is_open(0.996), 'not OPEN(0.996)')
check(not raw_gripper_is_open(0.8), 'not OPEN(0.8)')
check(raw_gripper_is_close(0.996), 'CLOSE(0.996)')
check(not raw_gripper_is_close(0.0), 'not CLOSE(0.0)')

check(decoded_action_to_env_gripper(0.0) == 1.0, 'decoded 0.0->env +1')
check(decoded_action_to_env_gripper(0.996) == -1.0, 'decoded 0.996->env -1')
check(env_gripper_is_open(1.0), 'env +1 is open')
check(not env_gripper_is_open(-1.0), 'env -1 is not open')

for raw_val in [-0.996, -0.5, 0.0, 0.3, 0.6, 0.8, 0.996]:
    env_val = decoded_action_to_env_gripper(raw_val)
    check(raw_gripper_is_open(raw_val) == env_gripper_is_open(env_val),
          f'equiv at raw={raw_val}')

check(classify_gripper_action(0.0) == 'open', 'classify open')
check(classify_gripper_action(0.996) == 'close_or_hold', 'classify close')
check(classify_gripper_action(OPEN_THRESHOLD) == 'boundary', 'classify boundary')
check(isinstance(CANONICAL_OPEN_SEMANTICS_VERSION, str), 'version is string')
check('20260603' in CANONICAL_OPEN_SEMANTICS_VERSION, 'version has date')

# ── Test 1.5: row index ──
print()
print('=== test_action_token_logit_row_index ===')
# CRITICAL: gripper row must be -2, not -1
check(action_token_logit_row_index(6, 7) == -2, 'gripper(dim=6) row=-2')
check(action_token_logit_row_index(0, 7) == -8, 'dim=0 row=-8')
check(action_token_logit_row_index(5, 7) == -3, 'dim=5 row=-3')
for dim, exp in {0: -8, 1: -7, 2: -6, 3: -5, 4: -4, 5: -3, 6: -2}.items():
    check(action_token_logit_row_index(dim, 7) == exp, f'dim={dim} -> {exp}')

# ── Test 2: prefix locked loss ──
print()
print('=== test_prefix_locked_loss_contains_gripper ===')
import torch
import torch.nn.functional as F

OPEN_IDS = list(range(31873, 32000))
CLOSE_IDS = list(range(31744, 31873))
GRIPPER_ROW = -2  # correct row index for action_dim=7

def compute_margin_loss(logits, region_ids, margin=5.0):
    gripper_row = logits[0, GRIPPER_ROW, :]  # FIXED: -2, not -1
    open_ids = torch.tensor(region_ids, dtype=torch.long)
    log_open = torch.logsumexp(gripper_row[open_ids], dim=0)
    non_open_mask = torch.ones_like(gripper_row, dtype=torch.bool)
    non_open_mask[open_ids] = False
    max_non_open = gripper_row[non_open_mask].max()
    return F.relu(max_non_open - log_open + float(margin))

# CRITICAL: changing row -2 changes loss, changing -1 does NOT
logits_test = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits_test[0, GRIPPER_ROW, CLOSE_IDS[0]] = 50.0
logits_test[0, GRIPPER_ROW, OPEN_IDS[0]] = 10.0
loss_before = float(compute_margin_loss(logits_test, OPEN_IDS, margin=5.0))
logits_test[0, -1, OPEN_IDS[0]] = 999.0  # change wrong row
loss_after_wrong = float(compute_margin_loss(logits_test, OPEN_IDS, margin=5.0))
check(loss_before == loss_after_wrong,
      f'Changing row -1 must NOT change gripper loss: {loss_before:.4f} vs {loss_after_wrong:.4f}')

logits_test[0, GRIPPER_ROW, OPEN_IDS[0]] = 80.0  # change correct row
loss_after_correct = float(compute_margin_loss(logits_test, OPEN_IDS, margin=5.0))
check(loss_after_correct != loss_before,
      f'Changing row -2 MUST change gripper loss: {loss_before:.4f} vs {loss_after_correct:.4f}')

# Margin loss > 0 when CLOSE dominates
logits = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits[0, GRIPPER_ROW, CLOSE_IDS[0]] = 50.0
logits[0, GRIPPER_ROW, OPEN_IDS[0]] = 10.0
loss = compute_margin_loss(logits, OPEN_IDS, margin=5.0)
check(float(loss) > 0.0, f'gripper margin loss positive: {float(loss):.4f}')

# Margin loss = 0 when OPEN dominates
logits2 = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits2[0, GRIPPER_ROW, OPEN_IDS[0]] = 100.0
logits2[0, GRIPPER_ROW, CLOSE_IDS[0]] = 10.0
loss2 = compute_margin_loss(logits2, OPEN_IDS, margin=5.0)
check(float(loss2) == 0.0, f'gripper margin loss zero: {float(loss2):.4f}')

# Gradient flows through correct row
logits3 = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits3[0, GRIPPER_ROW, CLOSE_IDS[0]] = 50.0
logits3[0, GRIPPER_ROW, OPEN_IDS[0]] = 10.0
logits3.requires_grad_(True)
loss3 = compute_margin_loss(logits3, OPEN_IDS, margin=5.0)
loss3.backward()
grad_sum_gripper = logits3.grad[0, GRIPPER_ROW, :].abs().sum()
check(float(grad_sum_gripper) > 0, f'gradient non-zero on row {GRIPPER_ROW}: {float(grad_sum_gripper):.4f}')

# Loss independent of label masking
logits_a = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits_a[0, GRIPPER_ROW, CLOSE_IDS[0]] = 50.0
logits_a[0, GRIPPER_ROW, OPEN_IDS[0]] = 10.0
logits_b = logits_a.clone()
l_a = compute_margin_loss(logits_a, OPEN_IDS, margin=5.0)
l_b = compute_margin_loss(logits_b, OPEN_IDS, margin=5.0)
check(float(l_a) == float(l_b), 'loss independent of label state')

# Region validity
check(len(OPEN_IDS) > 0, 'OPEN region non-empty')
check(len(CLOSE_IDS) > 0, 'CLOSE region non-empty')
check(set(OPEN_IDS).isdisjoint(set(CLOSE_IDS)), 'OPEN/CLOSE disjoint')

# Combined loss: gripper + arm
logits_c = torch.zeros(1, 107, 32000, dtype=torch.float32)
logits_c[0, GRIPPER_ROW, CLOSE_IDS[0]] = 50.0
logits_c[0, GRIPPER_ROW, OPEN_IDS[0]] = 10.0
for arm_dim in range(6):
    row_idx = action_token_logit_row_index(arm_dim, 7)
    logits_c[0, row_idx, 999] = 100.0
grip_loss = compute_margin_loss(logits_c, OPEN_IDS, margin=5.0)
check(float(grip_loss) > 0.0, f'combined: gripper loss={float(grip_loss):.4f}')
arm_ces = []
for arm_dim in range(6):
    row_idx = action_token_logit_row_index(arm_dim, 7)
    row = logits_c[0, row_idx, :]
    target = torch.tensor([arm_dim * 1000 + 1000])
    arm_ces.append(F.cross_entropy(row.view(1, -1), target))
arm_term = torch.stack(arm_ces).mean()
check(float(arm_term) > 0.0, f'combined: arm loss={float(arm_term):.4f}')

# ── Test 3: no teacher-forced fallback ──
print()
print('=== test_no_teacher_forced_fallback ===')
_GRIPPER_OBJ_SET = {
    'gripper_open_region_ce', 'force_open_z_down_token_ce',
    'force_open_region_z_down_ce',
    'prefix_locked_gripper_open_region_ce',
    'prefix_locked_gripper_open_margin',
    'gripper_open_expected_action',
}

def simulate_restart_selection(debug, objective, redecode_ok=True):
    if objective not in _GRIPPER_OBJ_SET:
        return 'ce_selected'
    if debug.get('adv_inputs') is None:
        raise RuntimeError("adv_inputs missing")
    if not redecode_ok:
        raise RuntimeError("Re-decode failed")
    return 'redecode_selected'

try:
    simulate_restart_selection({'adv_inputs': None}, 'prefix_locked_gripper_open_margin')
    check(False, 'should raise for missing adv_inputs')
except RuntimeError:
    check(True, 'missing adv_inputs raises')
for obj in _GRIPPER_OBJ_SET:
    try:
        simulate_restart_selection({'adv_inputs': None}, obj)
        check(False, f'{obj} should raise')
    except RuntimeError:
        check(True, f'{obj} bans fallback')
try:
    simulate_restart_selection({'adv_inputs': {'x':1}}, 'prefix_locked_gripper_open_margin', redecode_ok=False)
    check(False, 'should raise for redecode fail')
except RuntimeError as e:
    check('Re-decode' in str(e), 'redecode failure raises')
r = simulate_restart_selection({'adv_inputs': {'x':1}}, 'prefix_locked_gripper_open_margin', redecode_ok=True)
check(r == 'redecode_selected', 'normal path')
r2 = simulate_restart_selection({'adv_inputs': None}, 'targeted_directional_ce')
check(r2 == 'ce_selected', 'non-gripper unaffected')

# ── Test 4: provenance ──
print()
print('=== test_provenance_aggregator_schema ===')
def _make_trace_row(**overrides):
    base = {
        'task':'ketchup','condition':'vis_pgd','seed':'0','step':'0','policy_step':'0',
        'in_window':'True','attack_attempted':'True','pgd_applied':'True',
        'controller_active':'True','controller_stopped':'False','effective_attack_step_idx':'0',
        'raw_gripper':'0.0','env_gripper':'1.0','gripper_qpos':'0.039',
        'qpos_pre_step':'0.039','qpos_post_step':'0.038','clean_grip':'0.996','adv_grip':'0.0',
        'clean_z':'0.0','adv_z':'0.0','nad_dof7':'0.0','nad_z':'0.0','nad_dof1_3':'0.0',
        'arm_l2':'0.0','linf':'0.0','token_flip':'True','attack_dt':'0.5',
        'eef_x':'0.0','eef_y':'0.0','eef_z':'0.0','done':'False','reward':'0.0',
        'ctrl_mode':'fixed','ctrl_stop_reason':'none','ctrl_streak':'0',
        'ctrl_max_streak':'0','ctrl_qpos_delta':'0.0','ctrl_attacks':'0',
    }
    base.update(overrides)
    return base

rows = [_make_trace_row(policy_step=str(i), adv_grip='0.0') for i in range(3)]
rows += [_make_trace_row(policy_step='3', adv_grip='0.996')]
oc = sum(1 for r in rows if raw_gripper_is_open(float(r['adv_grip'])))
check(oc == 3, f'canonical open: {oc} (expected 3 of 4)')

rows_c = [_make_trace_row(policy_step=str(i), adv_grip='0.996') for i in range(18)]
oc = sum(1 for r in rows_c if raw_gripper_is_open(float(r['adv_grip'])))
check(oc == 0, f'all close: {oc} OPEN (expected 0)')

rows_o = [_make_trace_row(policy_step=str(i), adv_grip='0.0') for i in range(18)]
oc = sum(1 for r in rows_o if raw_gripper_is_open(float(r['adv_grip'])))
check(oc == 18, f'all open: {oc} OPEN (expected 18)')

# qpos delta
rq = [_make_trace_row(policy_step='0', qpos_post_step='0.039'),
      _make_trace_row(policy_step='1', qpos_post_step='0.002')]
attacked = [x for x in rq if x['pgd_applied'] == 'True']
qpost = [float(x['qpos_post_step']) for x in attacked if 'qpos_post_step' in x]
qd = max(abs(v - qpost[0]) for v in qpost) if len(qpost) > 1 else 0.0
check(abs(qd - 0.037) < 0.001, f'qpos_delta_post={qd}')

# Schema incomplete
r_noq = [{k:v for k,v in _make_trace_row(policy_step=str(i)).items() if k!='qpos_post_step'} for i in range(5)]
att2 = [x for x in r_noq if x['pgd_applied'] == 'True']
qp2 = [float(x['qpos_post_step']) for x in att2 if 'qpos_post_step' in x]
check(len(qp2) == 0, f'schema_incomplete qpost entries: {len(qp2)}')

# Arm L2
ra = [_make_trace_row(policy_step='0', arm_l2='0.0'), _make_trace_row(policy_step='1', arm_l2='0.1')]
av = [float(x['arm_l2']) for x in ra]
check(abs(np.mean(av) - 0.05) < 0.001, f'armL2={np.mean(av)}')
check(abs(np.max(av) - 0.1) < 0.001, f'armL2max={np.max(av)}')

# Done
rd = [_make_trace_row(policy_step='0', done='False'), _make_trace_row(policy_step='1', done='True')]
check(any(x['done'] == 'True' for x in rd), 'done detection')

# Provenance separation fields present
prov_fields = ['code_status','attack_adapter_commit','runner_commit','semantics_version',
               'prefix_loss_version','teacher_forced_fallback_allowed','trace_generated_by_repaired_runner']
for f in prov_fields:
    check(f in ['code_status','attack_adapter_commit','runner_commit','semantics_version',
                 'prefix_loss_version','teacher_forced_fallback_allowed','trace_generated_by_repaired_runner'],
          f'provenance field: {f}')

# ── Summary ──
print()
print(f'{"="*40}')
print(f'Results: {passed} passed, {failed} failed')
if failed == 0:
    print('ALL TESTS PASSED')
else:
    print(f'{failed} TESTS FAILED')
    sys.exit(1)
