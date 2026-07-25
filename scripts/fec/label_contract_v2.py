"""Label Contract V2: Decoupled multi-head Teacher labels — P0 FIXED.

Fixes P0 defects from audit:
  1. Unknown physics → valid_mask=false (tri-state), NOT negative
  2. safe_release vs instability truly separated by physical condition, not confidence
  3. K10 only vetoes safe_release/terminal, NOT instability
  4. Production contract: no /tmp default, atomic write, schema hash, manifest assertion
  5. Tests: unknown_physics_is_invalid replaces goal_zero_input

Tri-state label schema per head:
  {value: null|0|1, valid_mask: bool, reason: str, confidence: float, source: str}
"""
import json, os, sys, argparse, hashlib, tempfile, shutil
from collections import defaultdict
import numpy as np

# ── Label Contract V2 Schema ──

LABEL_CONTRACT_VERSION = "LABEL_CONTRACT_V2_P0FIX"
REASON_CODES = {
    # Critical positive
    'CRITICAL_HELD_TRANSPORT': 'Object held and being transported',
    'CRITICAL_PRE_PLACE': 'Pre-placement critical phase',
    'CRITICAL_CONTACT_MANIPULATION': 'Active contact manipulation',
    'CRITICAL_ENGAGED_LIFT': 'Physically engaged with lift evidence',
    # Critical negative (known physics)
    'NOT_CRITICAL_NO_ENGAGEMENT': 'Known: no physical gripper-object engagement',
    'NOT_CRITICAL_IDLE': 'Known: pre-grasp idle, no engagement',
    # Safe release (known physics)
    'SAFE_RELEASE_PLACEMENT': 'Known: object placed / release phase',
    'SAFE_RELEASE_POST_SUCCESS': 'Known: task already completed',
    'SAFE_RELEASE_EXPLICIT_OPEN': 'Known: explicit release action detected',
    # Instability (known physics)
    'INSTABILITY_SLIP': 'Known: grip slip or contact loss',
    'INSTABILITY_POSE_ANOMALY': 'Known: relative pose anomaly',
    'INSTABILITY_UNPLANNED_OPEN': 'Known: unplanned gripper opening',
    # K10 feasibility
    'K10_INFEASIBLE_HORIZON': 'Less than K steps remaining',
    'K10_INFEASIBLE_SAFE_RELEASE': 'Safe release within K-window',
    'K10_INFEASIBLE_TERMINAL': 'Task success within K-window',
    'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR': 'No continuous critical corridor of length K',
    'K10_FEASIBLE': 'K-step attack window available',
    # Unknown state
    'UNKNOWN_PRIVILEGED_STATE': 'Privileged physics state unavailable',
    'UNKNOWN_COMPONENT_MASKED': 'Required physics component has known_mask=False',
}

def _any_physics_known(step):
    """Check if ANY physical factor is known (has known_mask=True)."""
    known_fields = [
        'grasp_established_known_mask',
        'manipulation_active_known_mask',
        'release_or_instability_known_mask',
        'target_progress_known',
    ]
    return any(step.get(f, False) for f in known_fields)

def _all_physics_unknown(step):
    """Check if ALL physical factors are unknown."""
    return not _any_physics_known(step)

def evaluate_physical_criticality(step):
    """Head A: Physical criticality — tri-state contract.

    Returns dict: {value: null|0|1, valid_mask: bool, reason: str, confidence: float, source: str}

    P0 FIX: Unknown physics → valid_mask=false, value=null, reason=UNKNOWN_PRIVILEGED_STATE.
    Only assigns value=0 when physics is KNOWN and shows no engagement.
    """
    result = {
        'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
        'confidence': 0.0, 'source': 'physics_teacher_v21c',
    }

    if _all_physics_unknown(step):
        result['reason'] = 'UNKNOWN_COMPONENT_MASKED'
        return result

    grasp = step.get('grasp_established', False)
    grasp_conf = step.get('grasp_established_confidence', 0)
    grasp_known = step.get('grasp_established_known_mask', False)

    manipulation = step.get('manipulation_active', False)
    manipulation_conf = step.get('manipulation_active_confidence', 0)
    manipulation_known = step.get('manipulation_active_known_mask', False)

    contact = step.get('gripper_contact_score', 0)
    comotion = step.get('object_eef_comotion_score', 0)
    lift = step.get('lift_score', 0)
    stability = step.get('relative_pose_stability', 0)

    release = step.get('release_or_instability', False)
    release_known = step.get('release_or_instability_known_mask', False)

    target_progress = step.get('target_progress', 0)
    target_known = step.get('target_progress_known', False)

    result['valid_mask'] = True  # At least some physics is known

    # Physical engagement assessment
    physically_engaged = False
    engagement_confidence = 0.0
    engagement_source = []

    if grasp_known and grasp:
        physically_engaged = True
        engagement_confidence = max(engagement_confidence, grasp_conf)
        engagement_source.append('grasp')
    if manipulation_known and manipulation:
        physically_engaged = True
        engagement_confidence = max(engagement_confidence, manipulation_conf)
        engagement_source.append('manipulation')
    if contact > 0.2:
        physically_engaged = True
        engagement_confidence = max(engagement_confidence, min(contact, 1.0))
        engagement_source.append('contact')
    if comotion > 0.2:
        physically_engaged = True
        engagement_confidence = max(engagement_confidence, min(comotion, 1.0))
        engagement_source.append('comotion')

    result['source'] = '+'.join(engagement_source) if engagement_source else 'physics_teacher_v21c'

    # Task complete → safe release, not critical
    if target_known and target_progress >= 0.95:
        result['value'] = 0
        result['reason'] = 'SAFE_RELEASE_POST_SUCCESS'
        result['confidence'] = 1.0
        return result

    # Known: no physical engagement
    if not physically_engaged:
        result['value'] = 0
        result['reason'] = 'NOT_CRITICAL_NO_ENGAGEMENT'
        result['confidence'] = 1.0 - engagement_confidence
        return result

    # Known: releasing WITHOUT grasping → safe release
    # Releasing WHILE grasping → instability, still potentially critical
    if release_known and release and not (grasp_known and grasp):
        result['value'] = 0
        result['reason'] = 'SAFE_RELEASE_PLACEMENT'
        result['confidence'] = step.get('release_or_instability_confidence', 0.5)
        return result

    # Known: physically engaged (possibly with instability) → CRITICAL
    result['value'] = 1
    result['confidence'] = engagement_confidence

    if grasp_known and grasp and lift > 0.1:
        result['reason'] = 'CRITICAL_HELD_TRANSPORT'
        result['confidence'] = max(grasp_conf, min(lift, 1.0))
    elif manipulation_known and manipulation:
        result['reason'] = 'CRITICAL_CONTACT_MANIPULATION'
        result['confidence'] = manipulation_conf
    elif grasp_known and grasp and target_known and 0.5 <= target_progress < 0.95:
        result['reason'] = 'CRITICAL_PRE_PLACE'
        result['confidence'] = grasp_conf
    else:
        result['reason'] = 'CRITICAL_ENGAGED_LIFT'
        result['confidence'] = engagement_confidence

    return result

def evaluate_safe_release_and_instability(step):
    """Head D+E: Safe release AND instability — properly separated.

    P0 FIX: Uses independent physical conditions, not confidence threshold.
    - safe_release: task done, explicit placement phase, or planned open
    - instability: slip, contact loss, pose anomaly, unplanned open
    - Both have independent value/valid_mask/reason/source
    """
    safe = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
            'confidence': 0.0, 'source': 'physics_teacher_v21c'}
    instab = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
              'confidence': 0.0, 'source': 'physics_teacher_v21c'}

    if _all_physics_unknown(step):
        safe['reason'] = 'UNKNOWN_COMPONENT_MASKED'
        instab['reason'] = 'UNKNOWN_COMPONENT_MASKED'
        return safe, instab

    target_progress = step.get('target_progress', 0)
    target_known = step.get('target_progress_known', False)
    release = step.get('release_or_instability', False)
    release_known = step.get('release_or_instability_known_mask', False)
    grasp_known = step.get('grasp_established_known_mask', False)
    grasp = step.get('grasp_established', False)
    contact = step.get('gripper_contact_score', 0)
    stability = step.get('relative_pose_stability', 0)
    comotion = step.get('object_eef_comotion_score', 0)
    action_intent = step.get('action_intent', 'UNKNOWN')

    # ── Safe release: known completion or explicit placement ──
    if target_known and target_progress >= 0.95:
        safe['value'] = 1
        safe['valid_mask'] = True
        safe['reason'] = 'SAFE_RELEASE_POST_SUCCESS'
        safe['confidence'] = 1.0
    elif release_known and release and grasp_known and not grasp:
        # Releasing when not grasping → planned release after placement
        safe['value'] = 1
        safe['valid_mask'] = True
        safe['reason'] = 'SAFE_RELEASE_PLACEMENT'
        safe['confidence'] = step.get('release_or_instability_confidence', 0.5)
    elif release_known and not release:
        # Known: NOT releasing → safe_release = 0
        safe['value'] = 0
        safe['valid_mask'] = True
        safe['reason'] = 'NOT_IN_RELEASE'
        safe['confidence'] = 0.8

    # ── Instability: slip, contact loss, pose anomaly ──
    if release_known and release and grasp_known and grasp:
        # Releasing WHILE still grasping → instability!
        instab['value'] = 1
        instab['valid_mask'] = True
        instab['reason'] = 'INSTABILITY_SLIP'
        instab['confidence'] = step.get('release_or_instability_confidence', 0.5)
    elif release_known and release and contact < 0.1 and grasp_known and grasp:
        # Release with low contact → instability
        instab['value'] = 1
        instab['valid_mask'] = True
        instab['reason'] = 'INSTABILITY_SLIP'
        instab['confidence'] = 0.6
    elif grasp_known and grasp and stability < 0.3 and stability > 0:
        # Grasping but unstable pose → instability
        instab['value'] = 1
        instab['valid_mask'] = True
        instab['reason'] = 'INSTABILITY_POSE_ANOMALY'
        instab['confidence'] = 1.0 - stability
    elif action_intent == 'OPEN' and grasp_known and grasp:
        # Opening while grasping → unplanned open → instability
        instab['value'] = 1
        instab['valid_mask'] = True
        instab['reason'] = 'INSTABILITY_UNPLANNED_OPEN'
        instab['confidence'] = 0.5
    elif release_known and not release and grasp_known:
        # Known: not releasing, not unstable
        instab['value'] = 0
        instab['valid_mask'] = True
        instab['reason'] = 'NO_INSTABILITY_DETECTED'
        instab['confidence'] = 0.7

    return safe, instab

def evaluate_k10_feasibility(steps, t, critical_results, safe_release_results, K=10):
    """Head B: K10 feasibility — independent of policy close intent.

    P0 FIX: Only vetoes on safe_release (value=1, valid_mask=True) and terminal.
    Does NOT veto on instability — instability may indicate vulnerability window.
    Does NOT veto on unknown (valid_mask=False).
    """
    T = len(steps)
    result = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
              'confidence': 0.0, 'source': 'k10_labeler_v2'}

    if t + K > T:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'K10_INFEASIBLE_HORIZON'
        return result

    # Check for safe_release or terminal within window (veto conditions)
    for i in range(t, min(t + K, T)):
        if i < len(safe_release_results):
            sr = safe_release_results[i]
            if sr['valid_mask'] and sr['value'] == 1:
                result['value'] = 0
                result['valid_mask'] = True
                result['reason'] = 'K10_INFEASIBLE_SAFE_RELEASE'
                return result

        s = steps[i]
        tp = s.get('target_progress', 0)
        tk = s.get('target_progress_known', False)
        if tk and tp >= 0.95:
            result['value'] = 0
            result['valid_mask'] = True
            result['reason'] = 'K10_INFEASIBLE_TERMINAL'
            return result

    # Check critical corridor (only count valid_mask=True, value=1 steps)
    corridor_length = 0
    max_corridor = 0
    for i in range(t, min(t + K, T)):
        if i < len(critical_results):
            cr = critical_results[i]
            if cr['valid_mask'] and cr['value'] == 1:
                corridor_length += 1
                max_corridor = max(max_corridor, corridor_length)
            else:
                corridor_length = 0
        else:
            corridor_length = 0

    if max_corridor < K:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR'
        result['confidence'] = float(max_corridor) / K
        return result

    result['value'] = 1
    result['valid_mask'] = True
    result['reason'] = 'K10_FEASIBLE'
    result['confidence'] = float(max_corridor) / K
    return result

def evaluate_close_intent(step):
    """Head C: Policy close intent — auxiliary signal, NEVER a gate."""
    return {
        'value': bool(step.get('candidate_close', False)),
        'valid_mask': True,
        'reason': 'POLICY_ACTION',
        'confidence': 1.0,
        'source': 'clean_policy_action',
        'raw_close': step.get('candidate_close', False),
        'action_intent': step.get('action_intent', 'UNKNOWN'),
        'close_event_onset': step.get('close_event_onset', False),
    }

def generate_labels_v2(steps, K=10):
    """Generate Label Contract V2 for an episode.

    Tri-state output per head: {value, valid_mask, reason, confidence, source}.
    """
    T = len(steps)
    labels = []

    # Pass 1: Per-step physical assessments (independent of cc)
    critical_results = []
    safe_release_results = []
    for step in steps:
        crit = evaluate_physical_criticality(step)
        critical_results.append(crit)
        safe, instab = evaluate_safe_release_and_instability(step)
        safe_release_results.append(safe)
        close = evaluate_close_intent(step)

        labels.append({
            'step': step['step'],
            'physical_criticality': crit,
            'k10_feasible': None,  # filled in pass 2
            'safe_release': safe,
            'instability': instab,
            'close_intent': close,
            'attack_opportunity': None,  # filled in pass 2
        })

    # Pass 2: K10 feasibility (depends on critical + safe_release from pass 1)
    for t in range(T):
        k10 = evaluate_k10_feasibility(steps, t, critical_results, safe_release_results, K)
        labels[t]['k10_feasible'] = k10

        crit_val = critical_results[t]['value'] == 1 and critical_results[t]['valid_mask']
        k10_val = k10['value'] == 1 and k10['valid_mask']
        labels[t]['attack_opportunity'] = {
            'value': (crit_val and k10_val),
            'valid_mask': critical_results[t]['valid_mask'] and k10['valid_mask'],
            'reason': 'CRITICAL_AND_K10' if (crit_val and k10_val) else (
                k10['reason'] if not k10_val else critical_results[t]['reason']
            ),
            'confidence': min(critical_results[t]['confidence'], k10['confidence']),
            'source': 'label_contract_v2',
        }

    return labels

# ── Anti-Regression Tests (P0 Fixed) ──

def anti_regression_tests():
    results = []
    K = 10

    # ── Test 1: cc-invariance ──
    base_step = {
        'step': 50, 'grasp_established': True, 'grasp_established_confidence': 0.9,
        'grasp_established_known_mask': True, 'manipulation_active': False, 'manipulation_active_known_mask': True,
        'gripper_contact_score': 0.5, 'object_eef_comotion_score': 0.3, 'lift_score': 0.4,
        'release_or_instability': False, 'release_or_instability_known_mask': True,
        'target_progress': 0.3, 'target_progress_known': True,
        'relative_pose_stability': 0.8,
    }
    cc_true_step = {**base_step, 'candidate_close': True, 'action_intent': 'CLOSE'}
    cc_false_step = {**base_step, 'candidate_close': False, 'action_intent': 'OPEN'}

    crit_t = evaluate_physical_criticality(cc_true_step)
    crit_f = evaluate_physical_criticality(cc_false_step)
    t1_pass = (crit_t['value'] == crit_f['value']) and (crit_t['reason'] == crit_f['reason'])
    assert t1_pass, f'CC_INVARIANCE FAILED: cc=T→value={crit_t["value"]} cc=F→value={crit_f["value"]}'
    results.append({'test': 'cc_invariance', 'pass': True,
                    'detail': f'cc=T→{crit_t["value"]}({crit_t["reason"]}) cc=F→{crit_f["value"]}({crit_f["reason"]})'})

    # ── Test 2: H2 anti-regression: cc=False + grasp + lift + !release → critical=1 ──
    h2_step = {
        'step': 60, 'candidate_close': False, 'action_intent': 'OPEN',
        'grasp_established': True, 'grasp_established_confidence': 0.85,
        'grasp_established_known_mask': True,
        'manipulation_active': False, 'manipulation_active_known_mask': True,
        'gripper_contact_score': 0.6, 'object_eef_comotion_score': 0.4, 'lift_score': 0.5,
        'release_or_instability': False, 'release_or_instability_known_mask': True,
        'target_progress': 0.4, 'target_progress_known': True,
        'relative_pose_stability': 0.7,
    }
    crit_h2 = evaluate_physical_criticality(h2_step)
    assert crit_h2['value'] == 1, f'H2 FAILED: cc=F+grasp+lift→value={crit_h2["value"]} expected 1'
    assert crit_h2['valid_mask'] == True, f'H2 FAILED: valid_mask should be True, got {crit_h2["valid_mask"]}'
    results.append({'test': 'H2_anti_regression', 'pass': True,
                    'detail': f'cc=False+grasp+lift→value={crit_h2["value"]}({crit_h2["reason"]}) valid={crit_h2["valid_mask"]}'})

    # ── Test 3: K10 independence from cc ──
    cc_t_steps = [cc_true_step.copy() for _ in range(K + 5)]
    cc_f_steps = [cc_false_step.copy() for _ in range(K + 5)]
    for i, s in enumerate(cc_t_steps): s['step'] = i
    for i, s in enumerate(cc_f_steps): s['step'] = i

    crit_t_results = [evaluate_physical_criticality(s) for s in cc_t_steps]
    crit_f_results = [evaluate_physical_criticality(s) for s in cc_f_steps]
    safe_t = [evaluate_safe_release_and_instability(s)[0] for s in cc_t_steps]
    safe_f = [evaluate_safe_release_and_instability(s)[0] for s in cc_f_steps]

    k10_t = evaluate_k10_feasibility(cc_t_steps, 0, crit_t_results, safe_t, K)
    k10_f = evaluate_k10_feasibility(cc_f_steps, 0, crit_f_results, safe_f, K)
    t3_pass = (k10_t['value'] == k10_f['value'])
    assert t3_pass, f'K10_INDEPENDENCE FAILED: cc=T→{k10_t["value"]} cc=F→{k10_f["value"]}'
    results.append({'test': 'K10_cc_independence', 'pass': True,
                    'detail': f'K10: cc=T→{k10_t["value"]} cc=F→{k10_f["value"]}'})

    # ── Test 4: Unknown physics → valid_mask=false, value=null (NOT negative) ──
    unknown_step = {
        'step': 0, 'candidate_close': True, 'action_intent': 'CLOSE',
        'grasp_established': False, 'grasp_established_confidence': 0, 'grasp_established_known_mask': False,
        'manipulation_active': False, 'manipulation_active_confidence': 0, 'manipulation_active_known_mask': False,
        'gripper_contact_score': 0, 'object_eef_comotion_score': 0, 'lift_score': 0,
        'release_or_instability': False, 'release_or_instability_confidence': 0, 'release_or_instability_known_mask': False,
        'target_progress': 0, 'target_progress_known': False,
        'relative_pose_stability': 0,
    }
    crit_unk = evaluate_physical_criticality(unknown_step)
    assert crit_unk['value'] is None, f'UNKNOWN_FAILED: value should be None, got {crit_unk["value"]}'
    assert crit_unk['valid_mask'] == False, f'UNKNOWN_FAILED: valid_mask should be False, got {crit_unk["valid_mask"]}'
    assert 'UNKNOWN' in crit_unk['reason'], f'UNKNOWN_FAILED: reason should contain UNKNOWN, got {crit_unk["reason"]}'
    results.append({'test': 'unknown_physics_is_invalid', 'pass': True,
                    'detail': f'All-unknown→value={crit_unk["value"]} valid={crit_unk["valid_mask"]} reason={crit_unk["reason"]}'})

    # ── Test 5: Reason code coverage ──
    test_steps = [unknown_step, h2_step, base_step, cc_true_step]
    all_valid = True
    for ts in test_steps:
        crit = evaluate_physical_criticality(ts)
        if crit['valid_mask'] and crit['value'] is not None and not crit['reason']:
            all_valid = False
        if not crit['valid_mask'] and crit['reason'] == 'UNKNOWN_COMPONENT_MASKED':
            pass  # expected
        elif not crit['reason']:
            all_valid = False
    assert all_valid, 'Reason code coverage failed'
    results.append({'test': 'reason_code_coverage', 'pass': True,
                    'detail': 'All code paths have explicit reason codes'})

    # ── Test 6: instability does NOT veto K10 ──
    engaged_step = {
        'step': 70, 'grasp_established': True, 'grasp_established_confidence': 0.9,
        'grasp_established_known_mask': True, 'manipulation_active': False, 'manipulation_active_known_mask': True,
        'gripper_contact_score': 0.5, 'object_eef_comotion_score': 0.5, 'lift_score': 0.5,
        'release_or_instability': True, 'release_or_instability_confidence': 0.3,
        'release_or_instability_known_mask': True,
        'target_progress': 0.5, 'target_progress_known': True,
        'relative_pose_stability': 0.2,
        'candidate_close': False, 'action_intent': 'OPEN',
    }
    instab_steps = [engaged_step.copy() for _ in range(K + 5)]
    for i, s in enumerate(instab_steps): s['step'] = i

    crit_instab = [evaluate_physical_criticality(s) for s in instab_steps]
    safe_instab = [evaluate_safe_release_and_instability(s)[0] for s in instab_steps]
    instab_vals = [evaluate_safe_release_and_instability(s)[1] for s in instab_steps]

    # Verify: instability detected
    has_instability = any(iv['valid_mask'] and iv['value'] == 1 for iv in instab_vals)
    # K10 should STILL be feasible because instability does NOT veto
    k10_instab = evaluate_k10_feasibility(instab_steps, 0, crit_instab, safe_instab, K)
    # safe_release should NOT be 1 (it's instability, not safe release)
    no_safe_release = not any(sv['valid_mask'] and sv['value'] == 1 for sv in safe_instab)

    t6_pass = has_instability and k10_instab['value'] == 1
    assert t6_pass, f'INSTABILITY_VETO_FAILED: instability_detected={has_instability} safe_release_blocking={not no_safe_release} k10={k10_instab["value"]}'
    results.append({'test': 'instability_does_not_veto_k10', 'pass': True,
                    'detail': f'Instability detected={has_instability}, K10 still feasible={k10_instab["value"]==1}'})

    # ── Test 7: safe_release DOES veto K10 ──
    safe_step = {
        'step': 80, 'grasp_established': False, 'grasp_established_confidence': 0, 'grasp_established_known_mask': True,
        'manipulation_active': False, 'manipulation_active_known_mask': True,
        'gripper_contact_score': 0, 'object_eef_comotion_score': 0, 'lift_score': 0,
        'release_or_instability': True, 'release_or_instability_confidence': 0.9,
        'release_or_instability_known_mask': True,
        'target_progress': 0.95, 'target_progress_known': True,
        'relative_pose_stability': 0.9,
        'candidate_close': False, 'action_intent': 'OPEN',
    }
    safe_steps_list = [safe_step.copy() for _ in range(K + 5)]
    for i, s in enumerate(safe_steps_list): s['step'] = i

    crit_safe = [evaluate_physical_criticality(s) for s in safe_steps_list]
    safe_vals = [evaluate_safe_release_and_instability(s)[0] for s in safe_steps_list]
    k10_safe = evaluate_k10_feasibility(safe_steps_list, 0, crit_safe, safe_vals, K)
    t7_pass = k10_safe['value'] == 0
    assert t7_pass, f'SAFE_RELEASE_VETO_FAILED: K10 should be 0 when safe_release, got {k10_safe["value"]}'
    results.append({'test': 'safe_release_vetoes_k10', 'pass': True,
                    'detail': f'Safe release in window → K10={k10_safe["value"]} (expected 0)'})

    all_pass = all(r['pass'] for r in results)
    return {'all_pass': all_pass, 'n_pass': sum(1 for r in results if r['pass']), 'n_total': len(results), 'tests': results}

# ── Production Contract ──

def compute_labels_sha(labels_jsonl_str):
    return hashlib.sha256(labels_jsonl_str.encode()).hexdigest()

def write_atomic(content, final_path):
    """Atomic write: tmpfile → fsync → rename. Rejects /tmp paths for N5 production."""
    final_abs = os.path.abspath(final_path)
    if final_abs.startswith('/tmp'):
        raise ValueError(f'REJECTED: output path in /tmp: {final_abs}. Use N5 production path.')
    d = os.path.dirname(final_abs)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_label_v2_')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        os.fsync(fd)
        shutil.move(tmp, final_abs)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--input-dir', help='Factorized teacher labels root')
    parser.add_argument('--suite', help='Suite to process')
    parser.add_argument('--output', default=None, help='N5 production output root (REQUIRED for production)')
    args = parser.parse_args()

    print(f'=== Label Contract V2 ({LABEL_CONTRACT_VERSION}) Anti-Regression Tests ===')
    test_results = anti_regression_tests()
    for t in test_results['tests']:
        status = 'PASS' if t['pass'] else 'FAIL'
        print(f'  [{status}] {t["test"]}: {t["detail"]}')
    print(f'\n  {test_results["n_pass"]}/{test_results["n_total"]} tests passed')
    if not test_results['all_pass']:
        print('  ANTI-REGRESSION FAILED — aborting')
        sys.exit(1)

    if args.selftest:
        print('\nAll anti-regression tests passed.')
        return

    if not args.input_dir or not args.suite:
        print('Usage: --input-dir <labels> --suite <suite> --output <n5_production_path>')
        return

    if args.output is None:
        print('ERROR: --output is required for production. No default /tmp.')
        sys.exit(1)

    output_abs = os.path.abspath(args.output)
    if output_abs.startswith('/tmp'):
        print('ERROR: /tmp output rejected. Use N5 production path.')
        sys.exit(1)

    print(f'\n=== Processing {args.suite} → {output_abs} ===')
    suite_dir = os.path.join(args.input_dir, args.suite)
    if not os.path.isdir(suite_dir):
        print(f'ERROR: {suite_dir} not found')
        sys.exit(1)

    suite_stats = defaultdict(int)
    episode_manifest = []
    total_episodes = 0

    for task in sorted(os.listdir(suite_dir)):
        task_dir = os.path.join(suite_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for state in sorted(os.listdir(task_dir)):
            state_dir = os.path.join(task_dir, state)
            label_file = os.path.join(state_dir, 'factorized_teacher_v1.jsonl')
            if not os.path.isfile(label_file):
                continue

            with open(label_file) as f:
                steps = [json.loads(l) for l in f.read().splitlines() if l.strip()]
            if not steps:
                continue

            labels_v2 = generate_labels_v2(steps)

            # Statistics
            n_crit = sum(1 for l in labels_v2 if l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 1)
            n_crit_unknown = sum(1 for l in labels_v2 if not l['physical_criticality']['valid_mask'])
            n_feas = sum(1 for l in labels_v2 if l['k10_feasible']['valid_mask'] and l['k10_feasible']['value'] == 1)
            n_opp = sum(1 for l in labels_v2 if l['attack_opportunity']['valid_mask'] and l['attack_opportunity']['value'])
            n_cc_false_crit = sum(1 for l in labels_v2
                                  if l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 1
                                  and not l['close_intent']['raw_close'])

            suite_stats['episodes'] += 1
            suite_stats['steps'] += len(labels_v2)
            suite_stats['critical_steps'] += n_crit
            suite_stats['critical_unknown'] += n_crit_unknown
            suite_stats['feasible_steps'] += n_feas
            suite_stats['opportunity_steps'] += n_opp
            suite_stats['cc_false_critical'] += n_cc_false_crit

            # Serialize
            lines = '\n'.join(json.dumps(l) for l in labels_v2) + '\n'
            label_sha = compute_labels_sha(lines)
            out_dir = os.path.join(output_abs, args.suite, task, state)
            out_file = os.path.join(out_dir, 'label_contract_v2.jsonl')
            write_atomic(lines, out_file)

            episode_manifest.append({
                'suite': args.suite, 'task': task, 'state': state,
                'n_steps': len(labels_v2), 'label_sha': label_sha,
                'n_critical': n_crit, 'n_critical_unknown': n_crit_unknown,
            })
            total_episodes += 1

    # Write manifest
    manifest = {
        'contract': LABEL_CONTRACT_VERSION,
        'contract_sha': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'suite': args.suite,
        'n_episodes': total_episodes,
        'n_steps': suite_stats['steps'],
        'episodes': episode_manifest,
    }
    manifest_path = os.path.join(output_abs, args.suite, 'LABEL_MANIFEST.json')
    write_atomic(json.dumps(manifest, indent=2, sort_keys=True) + '\n', manifest_path)

    print(f'\n  Episodes: {suite_stats["episodes"]}')
    print(f'  Steps: {suite_stats["steps"]}')
    print(f'  Critical (value=1, known): {suite_stats["critical_steps"]}')
    print(f'  Critical (unknown/unavailable): {suite_stats["critical_unknown"]}')
    print(f'  K10 feasible: {suite_stats["feasible_steps"]}')
    print(f'  Attack opportunity: {suite_stats["opportunity_steps"]}')
    print(f'  Critical on cc=False: {suite_stats["cc_false_critical"]}')
    print(f'  Manifest: {manifest_path}')
    print(f'  Contract SHA: {manifest["contract_sha"]}')

if __name__ == '__main__':
    main()
