"""Phase A: Five-arm implementation tests. No LIBERO rollout."""
import sys, os, json, numpy as np, torch

sys.path.insert(0, '/tmp')
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')

from n4_detector_adapter import N4DetectorAdapter, PLATT_A, PLATT_B, TAU, D_PERSIST
from fec_runner import (RandomTimeSampler, K10BudgetController, FECTelemetry,
                         build_step_record, validate_rand_matched, validate_oracle_override)
K10 = 10

E = '/mnt/sdc/dty_user/openvla_attack_evidence'
IMPL_DIR = E + '/fec_implementation_v1'

FAILURES = []

def check(condition, name):
    if not condition:
        FAILURES.append(name)
        print('  FAIL: {}'.format(name))
    else:
        print('  PASS: {}'.format(name))

print('=== FEC FIVE-ARM IMPLEMENTATION TESTS ===')
print()

# ── TEST 1: N4 Adapter ──
print('1. N4 Detector Adapter')
adapter = N4DetectorAdapter(device='cpu', norm_data_path=IMPL_DIR + '/n4_norms_o0i0.pt')

# Test reset
adapter.reset_episode()
check(adapter._t == 0, 'reset_t_zero')
check(adapter.latch == False, 'reset_latch_false')
check(adapter.persistence_counter == 0, 'reset_counter_zero')
check(adapter.emit_step is None, 'reset_emit_none')

# Test single step with dummy data
f25d = np.random.randn(25).astype(np.float32)
p9d = np.random.randn(9).astype(np.float32)
g9d = np.random.randn(9).astype(np.float32)
result = adapter.step(f25d, p9d, g9d, True)
check('raw_logit' in result, 'step_has_raw_logit')
check('calibrated_prob' in result, 'step_has_cal_prob')
check('candidate_close' in result, 'step_has_cc')
check('persistence_counter' in result, 'step_has_pcounter')
check('latch' in result, 'step_has_latch')
check('emitted_this_step' in result, 'step_has_emitted')
check(isinstance(result['raw_logit'], float), 'raw_logit_float')
check(isinstance(result['calibrated_prob'], float), 'cal_prob_float')
check(0.0 <= result['calibrated_prob'] <= 1.0, 'cal_prob_range')

# Test trajectory accumulation
traj = adapter.get_trajectory()
check(len(traj['raw_logits']) == 1, 'traj_length_1')
check(len(traj['cal_probs']) == 1, 'traj_cal_length_1')

# Test persistence logic: verify counter resets on threshold miss
adapter.reset_episode()
# Feed steps that produce known calibrated prob values
# The model output depends on weights; test counter logic by observing behavior
f25d_z = np.zeros(25, dtype=np.float32)
p9d_z = np.zeros(9, dtype=np.float32)
g9d_z = np.zeros(9, dtype=np.float32)

# Step 0: with cc=True, observe counter
r0 = adapter.step(f25d_z, p9d_z, g9d_z, True)
cal0 = r0['calibrated_prob']
# Counter should be 1 if cal0 >= TAU, 0 otherwise
expected_counter = 1 if cal0 >= TAU else 0
check(r0['persistence_counter'] == expected_counter,
      'counter_logic_t0_cal{:.4f}_counter={}'.format(cal0, r0['persistence_counter']))

# Step 1: with cc=False, counter should RESET to 0
r1 = adapter.step(f25d_z, p9d_z, g9d_z, False)
check(r1['persistence_counter'] == 0,
      'counter_reset_on_no_cc_counter={}'.format(r1['persistence_counter']))

# Verify emit_step tracking works (even if model didn't emit for dummy data)
check(isinstance(adapter.emit_step, (int, type(None))), 'emit_step_type_valid')
check(adapter.latch in (True, False), 'latch_is_bool')
print('  (Note: model output with zero input: cal_prob={:.4f})'.format(cal0))

# Test one-shot latch property: if latch is True, further steps don't change emit_step
# (This test verifies the TYPE correctness even if dummy data doesn't trigger emit)
adapter.reset_episode()
# Run 20 steps — if model emits, test latch; if not, verify no false emit
emitted_ever = False
for t in range(20):
    r = adapter.step(np.random.randn(25).astype(np.float32)*5,
                     np.zeros(9, dtype=np.float32),
                     np.zeros(9, dtype=np.float32), True)
    if r['emitted_this_step']:
        emitted_ever = True
        pre_emit = adapter.emit_step
        # After emit, run more steps and verify emit_step unchanged
        for s in range(3):
            r2 = adapter.step(np.random.randn(25).astype(np.float32)*5,
                             np.zeros(9, dtype=np.float32),
                             np.zeros(9, dtype=np.float32), True)
        check(adapter.emit_step == pre_emit, 'one_shot_latch_preserved')
        check(adapter.latch == True, 'latch_remains_true')
        break
if not emitted_ever:
    print('  (Note: model did not emit with dummy data — latch test deferred to GPU smoke)')
check(adapter.emit_step is not None or not emitted_ever, 'emit_consistent')

# ── TEST 2: K10 Budget Controller ──
print()
print('2. K10 Budget Controller')
budget = K10BudgetController()
check(budget.is_active(0) == False, 'not_active_before_emit')
check(budget.frame_index(0) == -1, 'frame_idx_before_emit')

budget.bind_emit(50)
check(budget.is_active(49) == False, 'not_active_before_start')
check(budget.is_active(50) == True, 'active_at_start')
check(budget.is_active(55) == True, 'active_mid_window')
check(budget.is_active(59) == True, 'active_at_end')
check(budget.is_active(60) == False, 'not_active_after_window')
check(budget.frame_index(50) == 0, 'frame_idx_0')
check(budget.frame_index(55) == 5, 'frame_idx_5')
check(budget.frame_index(59) == 9, 'frame_idx_9')
check(budget.frame_index(60) == -1, 'frame_idx_after')

summary = budget.get_summary()
check(summary['k10_planned'] == 10, 'k10_planned_10')

# ── TEST 3: RANDOM_TIME Sampler ──
print()
print('3. RANDOM_TIME Sampler')
sampler1 = RandomTimeSampler(seed=42)
sampler2 = RandomTimeSampler(seed=42)
sampler3 = RandomTimeSampler(seed=99)

# Same seed → same time
t1 = sampler1.sample(300)
t2 = sampler2.sample(300)
check(t1 == t2, 'same_seed_same_time')

# Different seed → different time (high probability)
t3 = sampler3.sample(300)
check(t1 != t3, 'diff_seed_diff_time')

# Time within valid range
check(t1 >= sampler1.min_step, 'time_above_min')
check(t1 <= int(300 * sampler1.max_step_ratio) - K10 + 1, 'time_below_max')

# K10 executable: start + K10 <= max_t
check(t1 + K10 <= int(300 * sampler1.max_step_ratio), 'k10_executable')

# Short episode: no valid time
t_short = sampler1.sample(10)
check(t_short is None, 'short_episode_no_time')

# ── TEST 4: Telemetry Schema ──
print()
print('4. Telemetry Schema')
tlm = FECTelemetry('parent_01', 'TRUE_T10', 42, 43, 44)
rec = build_step_record(0, None, [0.1]*7, [0.1]*7,
                        {'raw_logit': 1.5, 'calibrated_prob': 0.9, 'candidate_close': True,
                         'persistence_counter': 3, 'latch': False, 'emitted_this_step': False,
                         'emit_step': None},
                        budget, False, -1, 'TRUE_T10')

required_fields = ['step', 'arm', 'raw_logit', 'calibrated_prob', 'candidate_close',
                   'persistence_counter', 'latch', 'emitted_this_step', 'emit_step',
                   'attack_active', 'attack_frame_idx']
for field in required_fields:
    check(field in rec, 'telemetry_has_{}'.format(field))

check(rec['attack_active'] == False, 'telemetry_attack_inactive')
check(rec['attack_frame_idx'] == -1, 'telemetry_frame_idx')

# Active attack step
budget2 = K10BudgetController()
budget2.bind_emit(10)
rec2 = build_step_record(12, None, [0.1]*7, [0.2]*7,
                         {'raw_logit': 2.0, 'calibrated_prob': 0.95, 'candidate_close': True,
                          'persistence_counter': 6, 'latch': True, 'emitted_this_step': True,
                          'emit_step': 10},
                         budget2, True, 2, 'TRUE_T10')
check(rec2['attack_active'] == True, 'telemetry_attack_active')
check(rec2['attack_frame_idx'] == 2, 'telemetry_frame_idx_2')
check(rec2['attack_start'] == 10, 'telemetry_attack_start')

# ── TEST 5: ORACLE Validator ──
print()
print('5. ORACLE Validator')
# Create synthetic telemetry with ORACLE arm preservation
tlm_oracle = FECTelemetry('p1', 'COMMAND_OPEN_ORACLE', 42, 43, 44)
# Clean and final actions where arm dims match but gripper differs
clean_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]
final_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
rec_o = build_step_record(0, None, clean_a, final_a,
                          {'raw_logit': 1.0, 'calibrated_prob': 0.8, 'candidate_close': True,
                           'persistence_counter': 1, 'latch': False, 'emitted_this_step': False,
                           'emit_step': None},
                          budget, False, -1, 'COMMAND_OPEN_ORACLE')
tlm_oracle.record_step(rec_o)
ok, issues = validate_oracle_override([tlm_oracle])
check(ok, 'oracle_arm_preserved')
check(len(issues) == 0, 'oracle_no_issues')

# Arm violation
final_bad = [0.11, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
rec_bad = build_step_record(0, None, clean_a, final_bad,
                            {'raw_logit': 1.0, 'calibrated_prob': 0.8, 'candidate_close': True,
                             'persistence_counter': 1, 'latch': False, 'emitted_this_step': False,
                             'emit_step': None},
                            budget, False, -1, 'COMMAND_OPEN_ORACLE')
tlm_bad = FECTelemetry('p2', 'COMMAND_OPEN_ORACLE', 42, 43, 44)
tlm_bad.record_step(rec_bad)
ok2, issues2 = validate_oracle_override([tlm_bad])
check(not ok2, 'oracle_detects_arm_violation')

# ── Summary ──
print()
print('='*50)
if FAILURES:
    print('IMPLEMENTATION TESTS: FAIL ({} failures)'.format(len(FAILURES)))
else:
    print('IMPLEMENTATION TESTS: ALL PASS')
print()
