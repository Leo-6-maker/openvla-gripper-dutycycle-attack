"""T2R1: Relation-Aware Placement Patch for V22 Teacher.

Adds goal-support contact detection and relation-aware placement
by monkeypatching v22_production_v2.compute_placement_state.

To use: import t2r1_placement_patch before calling process_episode.
"""
import json, os, sys
import numpy as np

# Import the original module
from v22_production_v2 import (
    _slice_vector, _dist, _finite_vector, _contact_flags,
    T2_PLACEMENT_TOLERANCE,
)

T2_PLACEMENT_CONTACT_DWELL = 3
_original_compute_placement_state = None


def _normalize_name(name):
    """Normalize BDDL/MuJoCo object names for fuzzy matching."""
    n = name.replace('_contain_region', '').replace('_init_region', '')
    # Remove common suffixes that MuJoCo strips
    for suffix in ['_back', '_front', '_top', '_bottom', '_left', '_right',
                   '_main', '_body', '_link', '_visual']:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    return n


def _check_object_goal_support_contact(steps_data, t, manipulated_objects, goal_support_names):
    """Check if manipulated object contacts goal support at step t.

    Uses fuzzy name matching: normalizes BDDL names and MuJoCo geoms
    by stripping region/suffix qualifiers.
    """
    pairs = steps_data[t].get('mujoco_contact_pairs', [])
    if not goal_support_names:
        return False, None
    # Pre-normalize all names
    gs_norms = [_normalize_name(gs) for gs in goal_support_names]
    manip_norms = [_normalize_name(m) for m in manipulated_objects]
    for pair in pairs:
        pair_norms = [_normalize_name(str(item)) for item in pair]
        for gs_base in gs_norms:
            for pn in pair_norms:
                # Check if goal support name appears as substring of contact pair entry
                if gs_base in pn or pn in gs_base:
                    # Check if a manipulated object is also in this pair
                    for mn in manip_norms:
                        for pn2 in pair_norms:
                            if mn in pn2 or pn2 in mn:
                                # Found: object + goal support in same contact pair
                                return True, gs_base
    return False, None


def compute_placement_state_t2r1(steps, grasp_results, manipulated_objects, object_slices,
                                  target_names, goal_support_names=None, goal_relations=None):
    """Relation-aware placement detection.

    Evidence sources (in priority order):
      1. object-goal_support_contact (direct physical evidence via MuJoCo pairs)
      2. semantic near target (point distance to target object position)
      3. terminal proximity (weakest, only used as last-resort fallback)

    placement = prior_transport AND goal_relation_evidence AND gripper_independent
    """
    T = len(steps)
    results = []
    can_compute = len(manipulated_objects) > 0 and len(object_slices) > 0
    if goal_support_names is None:
        goal_support_names = []
    if goal_relations is None:
        goal_relations = []

    def _had_evidence(idx):
        if idx < 0 or idx >= len(grasp_results):
            return False
        g = grasp_results[idx]
        return g.get('grasp_established', False) or g.get('contact_established', False)

    ever_had_evidence = any(_had_evidence(t) for t in range(T))

    # Pre-compute goal-support contact dwell
    gs_contact_dwell = 0
    gs_dwell_by_step = [0] * T
    for t in range(T):
        has_gs, _ = _check_object_goal_support_contact(
            steps, t, manipulated_objects, goal_support_names)
        if has_gs:
            gs_contact_dwell += 1
        else:
            gs_contact_dwell = 0
        gs_dwell_by_step[t] = gs_contact_dwell

    # Pre-compute gripper opening events
    gripper_opening = [False] * T
    for t in range(1, T):
        q_curr = _finite_vector(steps[t].get('robot0_gripper_qpos'), 2)
        q_prev = _finite_vector(steps[t-1].get('robot0_gripper_qpos'), 2)
        if q_curr is not None and q_prev is not None:
            w_curr = abs(q_curr[0]) + abs(q_curr[1])
            w_prev = abs(q_prev[0]) + abs(q_prev[1])
            if (w_curr - w_prev) > 0.01:
                gripper_opening[t] = True

    for t in range(T):
        placed = False
        known = can_compute and grasp_results[t]['grasp_known_mask']
        conf = 0.0
        region = None
        evidence_type = None

        if not known:
            results.append({
                'object_placed': False, 'placement_known_mask': False,
                'placement_confidence': 0.0, 'placement_region': None,
            })
            continue

        prior_transport = _had_evidence(max(0, t-1))

        # Release event detection
        is_release = False
        if t > 0:
            had_prior = _had_evidence(t-1)
            has_current = _had_evidence(t)
            if had_prior and not has_current:
                is_release = True

        gripper_independent = is_release or gripper_opening[t]

        # Evidence tier 1: goal-support contact (strongest)
        has_gs_contact = gs_dwell_by_step[t] >= T2_PLACEMENT_CONTACT_DWELL
        if has_gs_contact and prior_transport:
            placed = True
            conf = 0.80
            evidence_type = 'GOAL_SUPPORT_CONTACT'

        # Evidence tier 2: object near target position (point distance)
        if not placed:
            for name in manipulated_objects:
                spec = object_slices.get(name)
                if spec is None: continue
                obj_pos = _slice_vector(steps[t].get('object_state', []), spec, 'pos')
                if obj_pos is None: continue
                for tname in target_names:
                    tspec = object_slices.get(tname)
                    if tspec is None:
                        for fk in sorted(object_slices.keys()):
                            if fk in tname or tname.startswith(fk):
                                tspec = object_slices.get(fk)
                                break
                    if tspec is None: continue
                    tpos = _slice_vector(steps[t].get('object_state', []), tspec, 'pos')
                    if tpos is None: continue
                    if _dist(obj_pos, tpos) < T2_PLACEMENT_TOLERANCE and prior_transport:
                        placed = True
                        conf = 0.65
                        evidence_type = 'NEAR_TARGET'
                        region = tname
                        break
                if placed: break

        # Evidence tier 3: terminal proximity (weakest, last resort)
        if not placed and t >= T - 5 and ever_had_evidence and gripper_independent:
            for name in manipulated_objects:
                spec = object_slices.get(name)
                if spec is None: continue
                obj_pos = _slice_vector(steps[t].get('object_state', []), spec, 'pos')
                if obj_pos is None: continue
                for tname in target_names:
                    tspec = object_slices.get(tname)
                    if tspec is None:
                        for fk in sorted(object_slices.keys()):
                            if fk in tname or tname.startswith(fk):
                                tspec = object_slices.get(fk)
                                break
                    if tspec is None: continue
                    tpos = _slice_vector(steps[t].get('object_state', []), tspec, 'pos')
                    if tpos is None: continue
                    if _dist(obj_pos, tpos) < max(T2_PLACEMENT_TOLERANCE, 0.35):
                        placed = True
                        conf = 0.45
                        evidence_type = 'TERMINAL_PROXIMITY'
                        region = tname
                        break
                if placed: break

        if placed and not gripper_independent and not (t >= T - 3):
            # Need release/opening evidence for non-terminal steps
            # But allow terminal steps to count without explicit release
            placed = (t >= T - 3)

        results.append({
            'object_placed': bool(placed),
            'placement_known_mask': bool(known),
            'placement_confidence': float(conf),
            'placement_region': region,
            'placement_evidence_type': evidence_type,
        })

    return results


def patch():
    """Apply T2R1 patch globally."""
    global _original_compute_placement_state
    import v22_production_v2
    _original_compute_placement_state = v22_production_v2.compute_placement_state
    v22_production_v2.compute_placement_state = compute_placement_state_t2r1


def unpatch():
    """Restore original function."""
    global _original_compute_placement_state
    import v22_production_v2
    if _original_compute_placement_state is not None:
        v22_production_v2.compute_placement_state = _original_compute_placement_state
