"""Regression tests for corrected known-mask metric contract.
Validates that head-level replay respects known_mask across all contract rules.
"""
from __future__ import annotations

import json
from collections import defaultdict


# ── Replay engine (same logic as analysis script, standalone for testing) ──

def head_emit(step, head, threshold=0.5):
    """A head emits on a step when all conditions are met."""
    if not step.get('route_supported', True):
        return False
    km = step.get(f'{head}_known_mask', False)
    prob = step.get(f'{head}_prob', 0.0)
    return bool(km) and prob >= threshold


def any_head_emit(step, thresholds=None):
    """Union of known heads only."""
    if thresholds is None:
        thresholds = {'grasp': 0.5, 'manipulation': 0.5, 'release': 0.5}
    result = False
    for head in ['grasp', 'manipulation', 'release']:
        if head_emit(step, head, thresholds.get(head, 0.5)):
            result = True
    return result


def count_background_emits(steps, thresholds=None):
    """Count background steps where any known head emits."""
    if thresholds is None:
        thresholds = {'grasp': 0.5, 'manipulation': 0.5, 'release': 0.5}
    bg_known = 0
    bg_emit = 0
    for s in steps:
        if s.get('event_id', -1) < 0 and s.get('route_supported', True):
            if any(s.get(f'{h}_known_mask', False) for h in ['grasp', 'manipulation', 'release']):
                bg_known += 1
            if any_head_emit(s, thresholds):
                bg_emit += 1
    return bg_known, bg_emit


def longest_streak(steps, thresholds=None):
    """Find longest consecutive any-head emit streak within each episode."""
    if thresholds is None:
        thresholds = {'grasp': 0.5, 'manipulation': 0.5, 'release': 0.5}
    bg_eps = defaultdict(list)
    for s in steps:
        if s.get('event_id', -1) < 0 and s.get('route_supported', True):
            bg_eps[s['canonical_parent_key']].append(s)

    max_streak = 0
    for ep_key, ep_steps in bg_eps.items():
        ep_sorted = sorted(ep_steps, key=lambda x: x['step_index'])
        seq = 0
        prev_idx = None
        for s in ep_sorted:
            curr_idx = s['step_index']
            # Valid-mask gap: non-consecutive indices break streak
            if prev_idx is not None and curr_idx != prev_idx + 1:
                seq = 0
            if any_head_emit(s, thresholds):
                seq += 1
                max_streak = max(max_streak, seq)
            else:
                seq = 0
            prev_idx = curr_idx
    return max_streak


def episode_false_starts(steps, thresholds=None, persistence=1):
    """Count episodes with at least one false start."""
    if thresholds is None:
        thresholds = {'grasp': 0.5, 'manipulation': 0.5, 'release': 0.5}
    bg_eps = defaultdict(list)
    for s in steps:
        if s.get('event_id', -1) < 0 and s.get('route_supported', True):
            bg_eps[s['canonical_parent_key']].append(s)

    false_eps = 0
    for ep_key, ep_steps in bg_eps.items():
        ep_sorted = sorted(ep_steps, key=lambda x: x['step_index'])
        seq = 0
        triggered = False
        for s in ep_sorted:
            if any_head_emit(s, thresholds):
                seq += 1
                if seq >= persistence and not triggered:
                    false_eps += 1
                    triggered = True
            else:
                seq = 0
    return false_eps, len(bg_eps)


# ── Test fixtures ──

def make_step(event_id, route_supported=True, **head_kwargs):
    """Create a step dict with per-head prob/known_mask/target fields."""
    step = {
        'canonical_parent_key': 'test/suite/0/state0',
        'step_index': 0,
        'event_id': event_id,
        'route_supported': route_supported,
        'mechanism_route': 'single_object_pick_place',
    }
    for head in ['grasp', 'manipulation', 'release']:
        kw = head_kwargs.get(head, {})
        step[f'{head}_prob'] = kw.get('prob', 0.0)
        step[f'{head}_known_mask'] = kw.get('known_mask', True)
        step[f'{head}_target'] = kw.get('target', False)
    return step


# ── Tests ──

def test_manipulation_unknown_never_emits():
    """manipulation_known_mask=False must not contribute to any_head_emit."""
    s = make_step(-1, manipulation={'prob': 0.99, 'known_mask': False},
                  grasp={'prob': 0.0, 'known_mask': True},
                  release={'prob': 0.0, 'known_mask': True})
    assert not any_head_emit(s), "manipulation unknown should not cause emit"


def test_unknown_step_never_in_denominator():
    """Step where all heads have known_mask=False should not be counted in denominator."""
    steps = [
        make_step(-1, grasp={'prob': 0.0, 'known_mask': False},
                  manipulation={'prob': 0.0, 'known_mask': False},
                  release={'prob': 0.0, 'known_mask': False}),
    ]
    bg_known, bg_emit = count_background_emits(steps)
    assert bg_known == 0, f"all-unknown step should not be in bg_known: got {bg_known}"
    assert bg_emit == 0, f"all-unknown step should not emit: got {bg_emit}"


def test_unknown_step_breaks_streak():
    """A step with all heads unknown should break the streak."""
    steps = []
    for i in range(3):
        s = make_step(-1, grasp={'prob': 0.9, 'known_mask': True},
                      manipulation={'prob': 0.0, 'known_mask': False},
                      release={'prob': 0.0, 'known_mask': True})
        s['step_index'] = i
        steps.append(s)
    # Insert an all-unknown step at position 1
    gap = make_step(-1, grasp={'prob': 0.9, 'known_mask': False},
                    manipulation={'prob': 0.0, 'known_mask': False},
                    release={'prob': 0.0, 'known_mask': False})
    gap['step_index'] = 1
    # Shift indices
    steps[1]['step_index'] = 0
    steps[2]['step_index'] = 2
    all_steps = [steps[0], gap, steps[1], steps[2]]
    streak = longest_streak(all_steps)
    assert streak < 3, f"all-unknown step should break streak, got max={streak}"


def test_episode_boundary_breaks_streak():
    """Different canonical_parent_key should break streak."""
    steps = []
    for i in range(5):
        s = make_step(-1, grasp={'prob': 0.9, 'known_mask': True},
                      manipulation={'prob': 0.0, 'known_mask': False},
                      release={'prob': 0.0, 'known_mask': True})
        s['step_index'] = i
        s['canonical_parent_key'] = f'ep{i//3}'  # 3 in ep0, 2 in ep1
        steps.append(s)
    streak = longest_streak(steps)
    assert streak <= 3, f"episode boundary should break streak, got max={streak}"


def test_unsupported_route_never_emits():
    """route_supported=False should not contribute."""
    s = make_step(-1, route_supported=False,
                  grasp={'prob': 0.9, 'known_mask': True})
    assert not any_head_emit(s), "unsupported route should not emit"


def test_per_step_p50_fpr_consistency():
    """If P50 >= threshold, then FPR must be >= 0.5."""
    probs = [0.1, 0.3, 0.6, 0.7, 0.9]
    median = sorted(probs)[len(probs)//2]
    fpr = sum(1 for p in probs if p >= 0.5) / len(probs)
    if median >= 0.5:
        assert fpr >= 0.5, f"P50={median} >= 0.5 but FPR={fpr} < 0.5"
    # This is a sanity check on arithmetic, not data-dependent


def test_deterministic_output():
    """Same input twice should give same result."""
    steps = [
        make_step(-1, grasp={'prob': 0.3, 'known_mask': True},
                  release={'prob': 0.6, 'known_mask': True}),
        make_step(-1, grasp={'prob': 0.8, 'known_mask': True},
                  release={'prob': 0.1, 'known_mask': True}),
    ]
    r1 = count_background_emits(steps)
    r2 = count_background_emits(steps)
    assert r1 == r2, f"non-deterministic: {r1} != {r2}"


def test_old_buggy_behavior_reproduces_error():
    """Reproduce the old bug: any_head without known_mask."""
    s = make_step(-1, manipulation={'prob': 0.99, 'known_mask': False},
                  grasp={'prob': 0.0, 'known_mask': True},
                  release={'prob': 0.0, 'known_mask': True})
    # OLD BUGGY behavior: max() without known_mask check
    old_buggy_any = max(s['grasp_prob'], s['manipulation_prob'], s['release_prob'])
    assert old_buggy_any >= 0.5, "old bug would have emitted"
    # NEW behavior: known_mask required
    assert not any_head_emit(s), "new implementation correctly rejects"


def test_valid_mask_gap_breaks_streak():
    """Non-consecutive step_index should break streak."""
    steps = []
    for i in [0, 1, 5, 6]:  # gap between 1 and 5
        s = make_step(-1, grasp={'prob': 0.9, 'known_mask': True},
                      release={'prob': 0.0, 'known_mask': True})
        s['step_index'] = i
        steps.append(s)
    streak = longest_streak(steps)
    assert streak <= 2, f"index gap should break streak, got max={streak}"


def test_any_head_only_unions_known_heads():
    """If only grasp is known, any_head = grasp emit."""
    s = make_step(-1,
                  grasp={'prob': 0.7, 'known_mask': True},
                  manipulation={'prob': 0.9, 'known_mask': False},
                  release={'prob': 0.0, 'known_mask': False})
    assert head_emit(s, 'grasp'), "grasp should emit"
    assert not head_emit(s, 'manipulation'), "manip unknown should not emit"
    assert any_head_emit(s), "any_head should be True (grasp emits)"
    assert not head_emit(s, 'release'), "release unknown should not emit"
