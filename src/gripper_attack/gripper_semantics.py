# -*- coding: utf-8 -*-
"""Canonical gripper OPEN/CLOSE semantics for VIS attack pipeline.

ALL code that classifies a gripper action as OPEN or CLOSE MUST use the helpers
in this module.  Direct comparisons against 0.5 (``< 0.5`` or ``> 0.5``) outside
this module are forbidden — they caused the semantic-inconsistency bug audited in
Issue A of the VIS prefix-margin repair (2026-06-03).

Semantic ground truth
---------------------
The production pipeline is (see ``scripts/vis_rollout_adaptive_v3.py``):

    raw_action  →  normalize_gripper_action(binarize=True)  →  invert_gripper_action
    (model decoded)   (sign → {−1,+1} with 0→+1)              (× −1 for MuJoCo)

    * raw_action ≈  0.996  →  env_action = −1  →  CLOSE  (gripper holds / closes)
    * raw_action ≈  0.0    →  env_action = +1  →  OPEN   (gripper releases / opens)

The decoded-action OPEN region (``get_gripper_region_by_decoded_action``) uses
``decoded_action < 0.5``, which is equivalent to ``env_val > 0`` after the full
normalize→invert pipeline.  This equivalence is assertion-guarded.
"""

from __future__ import annotations

import numpy as np

# ── Canonical threshold in decoded (raw) action space ──
# All bins with decoded_action < OPEN_THRESHOLD decode to OPEN after the full
# production pipeline.  This threshold defaults to 0.5 and MUST match the value
# used by ``get_gripper_region_by_decoded_action(open_threshold=...)``.
OPEN_THRESHOLD: float = 0.5

# ── Semantic version string — bump when semantics change ──
CANONICAL_OPEN_SEMANTICS_VERSION = "v1.0_decoded_action_lt_0.5_is_open_20260603"


def raw_gripper_is_open(raw_gripper: float, *, threshold: float = OPEN_THRESHOLD) -> bool:
    """Return True if *raw_gripper* (model-decoded action[-1]) is OPEN.

    raw_gripper is the value produced by the OpenVLA decode pipeline BEFORE
    normalize_gripper_action / invert_gripper_action.  For the standard
    LIBERO-Object q01/q99 stats this is the bin-center directly.
    """
    return float(raw_gripper) < float(threshold)


def raw_gripper_is_close(raw_gripper: float, *, threshold: float = OPEN_THRESHOLD) -> bool:
    """Return True if *raw_gripper* is CLOSE (the complement of OPEN)."""
    return not raw_gripper_is_open(raw_gripper, threshold=threshold)


def decoded_action_to_env_gripper(raw_gripper: float, *, binarize: bool = True) -> float:
    """Apply the production normalize→invert pipeline to a raw gripper value.

    This is the SAME computation as ::

        normalize_gripper_action(raw, binarize=True)
        invert_gripper_action(normalized)

    Returns:
        +1.0  →  OPEN  (gripper releases / opens in MuJoCo)
        −1.0  →  CLOSE (gripper holds / closes in MuJoCo)
    """
    val = float(raw_gripper)
    # Step 1: normalize_gripper_action with binarize=True
    val = 2.0 * val - 1.0
    if binarize:
        val = np.sign(val)
        val = 1.0 if val == 0 else val
    # Step 2: invert_gripper_action
    val = -1.0 * val
    return float(val)


def env_gripper_is_open(env_gripper: float) -> bool:
    """Return True if *env_gripper* (post-normalize+invert) is OPEN.

    env_gripper is the value stored in rollout trace column ``env_gripper``,
    i.e. the value passed to ``env.step()`` for the gripper dimension.
    """
    return float(env_gripper) > 0.0


def env_gripper_is_close(env_gripper: float) -> bool:
    """Return True if *env_gripper* is CLOSE."""
    return float(env_gripper) < 0.0


def classify_gripper_action(raw_gripper: float, *, threshold: float = OPEN_THRESHOLD) -> str:
    """Classify a raw (decoded) gripper action.

    Returns one of ``"open"``, ``"close_or_hold"``, ``"boundary"``.

    ``"boundary"`` is reserved for values within 1e-4 of the threshold.
    """
    val = float(raw_gripper)
    eps = 1e-4
    if abs(val - float(threshold)) < eps:
        return "boundary"
    if raw_gripper_is_open(val, threshold=threshold):
        return "open"
    return "close_or_hold"


# ── Self-consistency assertions (evaluated at import time) ──
# These guard against future edits that break the equivalence.

def _self_check():
    """Verify that raw_gripper_is_open ⇔ env_gripper_is_open for key values."""
    test_values = {
        -0.996094: "open",   # most-extreme OPEN bin
        -0.5:      "open",
         0.0:      "open",   # neutral → OPEN after pipeline
         0.496:    "open",   # just below threshold
         0.5:      "close_or_hold",  # exactly at threshold → boundary
         0.504:    "close_or_hold",  # just above threshold
         0.996094: "close_or_hold",  # most-extreme CLOSE bin (saturation tokens 31744/31745)
    }
    for raw_val, expected in test_values.items():
        got = classify_gripper_action(raw_val)
        assert got == expected, (
            f"Self-check failed: raw={raw_val} expected={expected} got={got}. "
            f"env_gripper={decoded_action_to_env_gripper(raw_val)}"
        )
    # env_gripper_is_open must agree with raw_gripper_is_open for all non-boundary values
    for raw_val in [-0.996, -0.5, 0.0, 0.3, 0.6, 0.8, 0.996]:
        raw_open = raw_gripper_is_open(raw_val)
        env_open = env_gripper_is_open(decoded_action_to_env_gripper(raw_val))
        assert raw_open == env_open, (
            f"Equivalence failure: raw={raw_val} raw_is_open={raw_open} env_is_open={env_open}"
        )
    # Verify the env values directly
    assert decoded_action_to_env_gripper(0.996094) == -1.0, "CLOSE token must map to env=-1"
    assert decoded_action_to_env_gripper(0.0) == 1.0, "OPEN/neutral token must map to env=+1"


_self_check()
