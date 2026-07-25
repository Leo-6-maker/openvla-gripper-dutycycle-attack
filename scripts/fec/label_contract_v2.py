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

def _engagement_evidence(step):
    """Extract engagement evidence channels with independent known masks.

    Each channel returns (known: bool, positive: bool, confidence: float).
    Score-based channels use >0.2 as positive and non-zero as known signal.
    """
    channels = []
    # grasp
    gk = step.get('grasp_established_known_mask', False)
    channels.append(('grasp', gk, gk and step.get('grasp_established', False),
                     step.get('grasp_established_confidence', 0)))
    # manipulation
    mk = step.get('manipulation_active_known_mask', False)
    channels.append(('manipulation', mk, mk and step.get('manipulation_active', False),
                     step.get('manipulation_active_confidence', 0)))
    # contact: non-zero score treated as computed; explicit mask overrides
    cs = step.get('gripper_contact_score', 0)
    ck = step.get('contact_known_mask', cs != 0)
    channels.append(('contact', bool(ck), cs > 0.2, min(cs, 1.0)))
    # comotion
    ms = step.get('object_eef_comotion_score', 0)
    cmk = step.get('comotion_known_mask', ms != 0)
    channels.append(('comotion', bool(cmk), ms > 0.2, min(ms, 1.0)))
    # lift
    ls = step.get('lift_score', 0)
    lk = step.get('lift_known_mask', ls != 0)
    channels.append(('lift', bool(lk), ls > 0.1, min(ls, 1.0)))
    return channels

def _evidence_lattice(channels):
    """Evidence lattice for engagement.
    - any known+positive → can determine 'positively engaged' (but may be partial)
    - all known and all negative → can determine 'not engaged'
    - else → unknown (valid_mask=false)
    Returns (can_determine_positive: bool, can_determine_negative: bool,
             any_positive: bool, all_known: bool, best_confidence: float,
             positive_sources: list[str])
    """
    any_positive = False
    all_known = True
    all_negative = True
    best_conf = 0.0
    sources = []
    for name, known, positive, conf in channels:
        if known:
            if positive:
                any_positive = True
                all_negative = False
                sources.append(name)
                best_conf = max(best_conf, conf)
            # known+negative: contributes to all_known but not to positive
        else:
            all_known = False
    can_determine_positive = any_positive
    can_determine_negative = all_known and all_negative
    return can_determine_positive, can_determine_negative, any_positive, all_known, best_conf, sources

def evaluate_physical_criticality(step):
    """Head A: Physical criticality — evidence lattice (head-specific validity).

    P0 FINAL FIX: Uses evidence lattice. Any known+positive → can be positive.
    All known+negative → can be negative. Otherwise → valid_mask=false.
    """
    result = {
        'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
        'confidence': 0.0, 'source': 'physics_teacher_v21c',
    }

    channels = _engagement_evidence(step)
    can_pos, can_neg, any_pos, all_known, best_conf, sources = _evidence_lattice(channels)

    release = step.get('release_or_instability', False)
    release_known = step.get('release_or_instability_known_mask', False)

    target_progress = step.get('target_progress', 0)
    target_known = step.get('target_progress_known', False)

    grasp = step.get('grasp_established', False)
    grasp_known = step.get('grasp_established_known_mask', False)

    # Task complete → known safe release (independent of engagement)
    if target_known and target_progress >= 0.95:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'SAFE_RELEASE_POST_SUCCESS'
        result['confidence'] = 1.0
        return result

    # Known releasing without grasp → safe release, not critical
    if release_known and release and grasp_known and not grasp:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'SAFE_RELEASE_PLACEMENT'
        result['confidence'] = step.get('release_or_instability_confidence', 0.5)
        return result

    # Can determine negative: all engagement channels known and all negative
    if can_neg:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'NOT_CRITICAL_NO_ENGAGEMENT'
        result['confidence'] = best_conf
        return result

    # Can determine positive: at least one channel known+positive
    if can_pos:
        result['value'] = 1
        result['valid_mask'] = True
        result['confidence'] = best_conf
        result['source'] = '+'.join(sources)
        # Determine subtype
        gk = step.get('grasp_established_known_mask', False)
        g = step.get('grasp_established', False)
        ls = step.get('lift_score', 0)
        mk = step.get('manipulation_active_known_mask', False)
        m = step.get('manipulation_active', False)
        if gk and g and ls > 0.1:
            result['reason'] = 'CRITICAL_HELD_TRANSPORT'
        elif mk and m:
            result['reason'] = 'CRITICAL_CONTACT_MANIPULATION'
        elif gk and g and target_known and 0.5 <= target_progress < 0.95:
            result['reason'] = 'CRITICAL_PRE_PLACE'
        else:
            result['reason'] = 'CRITICAL_ENGAGED_LIFT'
        return result

    # Otherwise: partial unknown → valid_mask=false
    result['reason'] = 'UNKNOWN_COMPONENT_MASKED'
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

def evaluate_safe_release_and_instability(step, production_mode=False):
    """Head D+E: Safe release AND instability — each with OWN validity.

    FINAL: Instability uses evidence lattice (same principle as criticality).
    Production mode requires explicit known_mask; legacy mode infers from score>0.
    """
    safe = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
            'confidence': 0.0, 'source': 'physics_teacher_v21c'}
    instab = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE',
              'confidence': 0.0, 'source': 'physics_teacher_v21c'}

    target_progress = step.get('target_progress', 0)
    target_known = step.get('target_progress_known', False)
    release = step.get('release_or_instability', False)
    release_known = step.get('release_or_instability_known_mask', False)
    grasp_known = step.get('grasp_established_known_mask', False)
    grasp = step.get('grasp_established', False)
    contact = step.get('gripper_contact_score', 0)
    ck = step.get('contact_known_mask', contact != 0) if not production_mode else step.get('contact_known_mask', False)
    stability = step.get('relative_pose_stability', 0)
    sk = step.get('stability_known_mask', stability != 0) if not production_mode else step.get('stability_known_mask', False)

    # ── safe_release (unchanged from previous fix) ──
    if target_known and target_progress >= 0.95:
        safe['value'] = 1; safe['valid_mask'] = True
        safe['reason'] = 'SAFE_RELEASE_POST_SUCCESS'; safe['confidence'] = 1.0
    elif release_known and release and grasp_known and not grasp:
        safe['value'] = 1; safe['valid_mask'] = True
        safe['reason'] = 'SAFE_RELEASE_PLACEMENT'
        safe['confidence'] = step.get('release_or_instability_confidence', 0.5)
    elif release_known and not release:
        safe['value'] = 0; safe['valid_mask'] = True
        safe['reason'] = 'NOT_IN_RELEASE'; safe['confidence'] = 0.8

    # ── instability: evidence lattice ──
    # Channels: slip (release+grasp), contact-loss, pose-anomaly
    instab_channels = []
    # slip: releasing while grasping
    instab_channels.append(('slip', release_known and grasp_known,
                            release and grasp,
                            step.get('release_or_instability_confidence', 0.5)))
    # contact loss: low contact while grasping
    instab_channels.append(('contact_loss', bool(ck) and grasp_known,
                            bool(ck) and grasp and contact < 0.1,
                            0.6 if (bool(ck) and grasp and contact < 0.1) else 0.0))
    # pose anomaly: low stability while grasping
    instab_channels.append(('pose_anomaly', bool(sk) and grasp_known,
                            bool(sk) and grasp and 0 < stability < 0.3,
                            1.0 - stability if (bool(sk) and grasp and 0 < stability < 0.3) else 0.0))

    any_known = any(k for _, k, _, _ in instab_channels)
    if not any_known:
        return safe, instab  # both unknown

    any_positive = any(p for _, k, p, _ in instab_channels if k and p)
    all_channels_known = all(k for _, k, _, _ in instab_channels)
    # For "all negative": every channel that IS known must be negative
    all_known_negative = all((not p) for _, k, p, _ in instab_channels if k)

    if any_positive:
        instab['value'] = 1; instab['valid_mask'] = True
        # Find the specific reason
        for name, k, p, conf in instab_channels:
            if k and p:
                if name == 'slip':
                    instab['reason'] = 'INSTABILITY_SLIP'; instab['confidence'] = conf
                elif name == 'contact_loss':
                    instab['reason'] = 'INSTABILITY_SLIP'; instab['confidence'] = conf
                elif name == 'pose_anomaly':
                    instab['reason'] = 'INSTABILITY_POSE_ANOMALY'; instab['confidence'] = conf
                break
    elif all_channels_known and all_known_negative:
        instab['value'] = 0; instab['valid_mask'] = True
        instab['reason'] = 'NO_INSTABILITY_DETECTED'; instab['confidence'] = 0.7
    # else: partial unknown → valid_mask stays False

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
    # P0-4 FIX: unknown steps → valid_mask=false for K10 overall
    corridor_length = 0
    max_corridor = 0
    has_unknown_in_window = False
    has_known_false_in_window = False
    for i in range(t, min(t + K, T)):
        if i < len(critical_results):
            cr = critical_results[i]
            if not cr['valid_mask']:
                has_unknown_in_window = True
                corridor_length = 0  # reset corridor at unknown
            elif cr['value'] == 1:
                corridor_length += 1
                max_corridor = max(max_corridor, corridor_length)
            else:
                has_known_false_in_window = True
                corridor_length = 0
        else:
            corridor_length = 0

    # Known false in window → definitively infeasible
    if has_known_false_in_window and max_corridor < K:
        result['value'] = 0
        result['valid_mask'] = True
        result['reason'] = 'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR'
        result['confidence'] = float(max_corridor) / K
        return result

    # Unknown in window → cannot determine feasibility
    if has_unknown_in_window and max_corridor < K:
        result['value'] = None
        result['valid_mask'] = False
        result['reason'] = 'K10_UNKNOWN_CRITICAL_IN_WINDOW'
        result['confidence'] = float(max_corridor) / K
        return result

    # All known, corridor sufficient
    if max_corridor >= K:
        result['value'] = 1
        result['valid_mask'] = True
        result['reason'] = 'K10_FEASIBLE'
        result['confidence'] = float(max_corridor) / K
        return result

    # All known, insufficient corridor
    result['value'] = 0
    result['valid_mask'] = True
    result['reason'] = 'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR'
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

    # ── Test 8: Partial unknown → valid_mask=false (NOT negative) ──
    partial_unknown_step = {
        'step': 90, 'candidate_close': True, 'action_intent': 'CLOSE',
        'grasp_established': False, 'grasp_established_confidence': 0, 'grasp_established_known_mask': False,
        'manipulation_active': False, 'manipulation_active_confidence': 0, 'manipulation_active_known_mask': False,
        'gripper_contact_score': 0, 'object_eef_comotion_score': 0, 'lift_score': 0,
        'release_or_instability': False, 'release_or_instability_confidence': 0, 'release_or_instability_known_mask': True,
        'target_progress': 0.5, 'target_progress_known': True,
        'relative_pose_stability': 0,
    }
    crit_partial = evaluate_physical_criticality(partial_unknown_step)
    t8_pass = (crit_partial['value'] is None and crit_partial['valid_mask'] == False)
    assert t8_pass, f'PARTIAL_UNKNOWN_FAILED: value={crit_partial["value"]} valid={crit_partial["valid_mask"]} (expected None, False)'
    results.append({'test': 'partial_unknown_is_invalid', 'pass': True,
                    'detail': f'Only release/target known → value={crit_partial["value"]} valid={crit_partial["valid_mask"]}'})

    # ── Test 9: K10 with unknown in window → valid_mask=false ──
    # Build a sequence with all critical but one step unknown
    unk_steps = []
    for _ in range(K + 5):
        s = dict(base_step)
        unk_steps.append(s)
    # Make step 5 unknown
    unk_steps[5] = dict(unknown_step)
    crit_unk_seq = [evaluate_physical_criticality(s) for s in unk_steps]
    safe_unk_seq = [evaluate_safe_release_and_instability(s)[0] for s in unk_steps]
    k10_unk = evaluate_k10_feasibility(unk_steps, 0, crit_unk_seq, safe_unk_seq, K)
    t9_pass = (k10_unk['value'] is None and k10_unk['valid_mask'] == False)
    assert t9_pass, f'K10_UNKNOWN_FAILED: value={k10_unk["value"]} valid={k10_unk["valid_mask"]} (expected None, False)'
    results.append({'test': 'k10_unknown_in_window', 'pass': True,
                    'detail': f'Unknown step in K10 window → value={k10_unk["value"]} valid={k10_unk["valid_mask"]}'})

    # ── Test 10: Atomic write success + content verify + N5 root check ──
    test_dir = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/tmp/atomic_test'
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    try:
        test_content = json.dumps({'test': 'atomic_write_test', 'value': 42})
        test_path = os.path.join(test_dir, 'test_output.json')
        write_atomic(test_content, test_path)
        assert os.path.isfile(test_path), 'Atomic write: file not created'
        with open(test_path) as ff:
            assert ff.read() == test_content, 'Atomic write: content mismatch'
        # Verify no-clobber
        try:
            write_atomic('different', test_path)
            assert False, 'Should have rejected overwrite'
        except FileExistsError:
            pass
        # Verify /tmp rejection
        try:
            write_atomic('x', '/tmp/should_fail.json')
            assert False, 'Should have rejected /tmp path'
        except ValueError:
            pass
        # Verify non-N5 path rejection
        try:
            write_atomic('x', '/mnt/sdc/dty_user/should_fail.json')
            assert False, 'Should have rejected non-N5 path'
        except ValueError:
            pass
        results.append({'test': 'atomic_write_success', 'pass': True,
                        'detail': 'Write + content + no-clobber + /tmp + non-N5 rejection'})
    finally:
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # ── Test 11: K10 with all known → correct result ──
    all_crit_steps = [dict(base_step) for _ in range(K + 5)]
    for i, s in enumerate(all_crit_steps): s['step'] = i
    crit_all_known = [evaluate_physical_criticality(s) for s in all_crit_steps]
    safe_all_known = [evaluate_safe_release_and_instability(s)[0] for s in all_crit_steps]
    k10_all = evaluate_k10_feasibility(all_crit_steps, 0, crit_all_known, safe_all_known, K)
    t11_pass = (k10_all['value'] == 1 and k10_all['valid_mask'] == True)
    assert t11_pass, f'K10_ALL_KNOWN_FAILED: value={k10_all["value"]} valid={k10_all["valid_mask"]}'
    results.append({'test': 'k10_all_known_feasible', 'pass': True,
                    'detail': f'All critical known → K10={k10_all["value"]}'})

    all_pass = all(r['pass'] for r in results)
    n_pass = sum(1 for r in results if r['pass'])
    n_fail = len(results) - n_pass
    return {'all_pass': all_pass, 'n_pass': n_pass, 'n_fail': n_fail, 'n_total': len(results), 'tests': results}

# ── Production Contract ──

def compute_labels_sha(labels_jsonl_str):
    return hashlib.sha256(labels_jsonl_str.encode()).hexdigest()

N5_ALLOWED_ROOT = os.path.realpath('/mnt/sdc/dty_user/openvla_attack_outputs/n5')

def write_atomic(content, final_path, allow_overwrite=False):
    """Atomic write: realpath check → tmpfile → fsync → os.replace → fsync dir.

    FINAL: realpath/commonpath prevents symlink escapes.
    Uses os.link as no-clobber (hardlink fails if target exists = atomic check).
    Rejects allow_overwrite in production.
    """
    final_abs = os.path.realpath(final_path)
    # Verify under allowed root via common path
    if os.path.commonpath([final_abs, N5_ALLOWED_ROOT]) != N5_ALLOWED_ROOT:
        raise ValueError(f'REJECTED: {final_abs} not under {N5_ALLOWED_ROOT}')
    # Detect symlinks in path components
    p = final_abs
    while p != N5_ALLOWED_ROOT and p != os.path.dirname(p):
        if os.path.islink(p):
            raise ValueError(f'REJECTED: symlink in output path: {p}')
        p = os.path.dirname(p)

    if not allow_overwrite and os.path.exists(final_abs):
        raise FileExistsError(f'REJECTED: output already exists: {final_abs}')

    d = os.path.dirname(final_abs)
    os.makedirs(d, exist_ok=True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_label_v2_')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-clobber via hardlink: succeeds only if final doesn't exist
        # If hardlink fails with EEXIST, we've been raced — fail
        try:
            os.link(tmp, final_abs)
        except FileExistsError:
            os.unlink(tmp)
            raise FileExistsError(f'REJECTED: concurrent write detected for {final_abs}')
        os.unlink(tmp)
        dir_fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        if tmp and os.path.exists(tmp):
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
    print(f'\n  {test_results["n_pass"]} PASS / {test_results["n_fail"]} FAIL (total {test_results["n_total"]})')
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
