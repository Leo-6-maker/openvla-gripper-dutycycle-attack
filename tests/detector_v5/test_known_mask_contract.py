"""Regression tests for known-mask metric contract.
Validates: manipulation unknown never emits, unknown breaks streak,
episode boundary resets, unsupported route excluded, determinism,
P50/FPR consistency, old-bug reproduction.
"""
from __future__ import annotations

from collections import defaultdict


def _make_step(event_id=-1, route_supported=True, canonical="test/id/0", step=0, **heads):
    s = {
        "canonical_parent_key": canonical, "step_index": step,
        "event_id": event_id, "route_supported": route_supported,
        "mechanism_route": "single_object_pick_place",
    }
    for h in ["grasp", "manipulation", "release"]:
        kw = heads.get(h, {})
        s[f"{h}_prob"] = kw.get("prob", 0.0)
        s[f"{h}_known_mask"] = kw.get("known_mask", True)
        s[f"{h}_target"] = kw.get("target", False)
    return s


def head_emit(step, head, threshold=0.5):
    if not step.get("route_supported", True):
        return False
    return bool(step.get(f"{head}_known_mask", False)) and step.get(f"{head}_prob", 0) >= threshold


def any_head_emit(step, thresholds=None):
    if thresholds is None:
        thresholds = {"grasp": 0.5, "manipulation": 0.5, "release": 0.5}
    for h in ["grasp", "manipulation", "release"]:
        if head_emit(step, h, thresholds.get(h, 0.5)):
            return True
    return False


def longest_streak(steps, thresholds=None):
    if thresholds is None:
        thresholds = {"grasp": 0.5, "manipulation": 0.5, "release": 0.5}
    bg_eps = defaultdict(list)
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported", True):
            bg_eps[s["canonical_parent_key"]].append(s)
    max_streak = 0
    for ep_steps in bg_eps.values():
        ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
        seq = 0; prev_idx = None
        for s in ep_sorted:
            curr = s["step_index"]
            if prev_idx is not None and curr != prev_idx + 1:
                seq = 0
            if any_head_emit(s, thresholds):
                seq += 1
                max_streak = max(max_streak, seq)
            else:
                seq = 0
            prev_idx = curr
    return max_streak


# ── Tests ──

def test_manipulation_unknown_never_emits():
    s = _make_step(-1, manipulation={"prob": 0.99, "known_mask": False},
                   grasp={"prob": 0.0}, release={"prob": 0.0})
    assert not any_head_emit(s)


def test_all_unknown_not_in_denominator():
    s = _make_step(-1, grasp={"prob": 0.0, "known_mask": False},
                   manipulation={"prob": 0.0, "known_mask": False},
                   release={"prob": 0.0, "known_mask": False})
    assert not any_head_emit(s)


def test_unknown_breaks_streak():
    steps = []
    for i in range(3):
        steps.append(_make_step(-1, step=i, grasp={"prob": 0.9}))
    gap = _make_step(-1, step=1, grasp={"prob": 0.9, "known_mask": False},
                     release={"prob": 0.0, "known_mask": False})
    all_s = [_make_step(-1, step=0, grasp={"prob": 0.9}),
             gap,
             _make_step(-1, step=1, grasp={"prob": 0.9}),
             _make_step(-1, step=2, grasp={"prob": 0.9})]
    for i, s in enumerate(all_s):
        s["step_index"] = i
    streak = longest_streak(all_s)
    assert streak < 3, f"unknown should break streak, got {streak}"


def test_episode_boundary_breaks_streak():
    steps = []
    for i in range(5):
        s = _make_step(-1, canonical=f"ep{i//3}", step=i, grasp={"prob": 0.9})
        steps.append(s)
    streak = longest_streak(steps)
    assert streak <= 3, f"ep boundary should break streak, got {streak}"


def test_unsupported_route_never_emits():
    s = _make_step(-1, route_supported=False, grasp={"prob": 0.9})
    assert not any_head_emit(s)


def test_deterministic():
    steps = [_make_step(-1, step=0, grasp={"prob": 0.3}, release={"prob": 0.6}),
             _make_step(-1, step=1, grasp={"prob": 0.8}, release={"prob": 0.1})]
    r1 = any_head_emit(steps[0])
    r2 = any_head_emit(steps[0])
    assert r1 == r2


def test_old_bug_reproduces_error():
    s = _make_step(-1, manipulation={"prob": 0.99, "known_mask": False},
                   grasp={"prob": 0.0}, release={"prob": 0.0})
    old_buggy = max(s["grasp_prob"], s["manipulation_prob"], s["release_prob"])
    assert old_buggy >= 0.5
    assert not any_head_emit(s)


def test_valid_mask_gap_breaks_streak():
    steps = []
    for idx in [0, 1, 5, 6]:
        s = _make_step(-1, step=idx, grasp={"prob": 0.9}, release={"prob": 0.0})
        steps.append(s)
    streak = longest_streak(steps)
    assert streak <= 2, f"gap should break streak, got {streak}"


def test_any_head_only_unions_known():
    s = _make_step(-1, grasp={"prob": 0.7, "known_mask": True},
                   manipulation={"prob": 0.9, "known_mask": False},
                   release={"prob": 0.0, "known_mask": False})
    assert any_head_emit(s)


def test_p50_fpr_consistency():
    probs = [0.1, 0.3, 0.6, 0.7, 0.9]
    median = sorted(probs)[len(probs)//2]
    fpr = sum(1 for p in probs if p >= 0.5) / len(probs)
    if median >= 0.5:
        assert fpr >= 0.5
