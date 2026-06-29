#!/usr/bin/env python3
"""Shared score-only FSM (legacy_v1) for detector evaluation.

Single source of truth used by both train F1 checkpoint selection and
formal evaluator. Accepts pre-computed corridor_p/release_p/phase_names arrays.

FSM: IDLE → (phase=="stable_carry" AND corridor_p > tau_c) → ARMED
     ARMED → (step >= arm_step + guard AND corridor_p > tau_c AND release_p < tau_r) → EMITTED
     EMITTED: one-shot latch, never transitions again.
"""
from __future__ import annotations


def run_fsm_legacy_v1(corridor_p, release_p, phase_names,
                       tau_corridor=0.3, tau_release=0.3, guard=5):
    """Run legacy_v1 FSM on pre-computed scores.

    Args:
        corridor_p: array-like of corridor sigmoid scores
        release_p: array-like of release sigmoid scores
        phase_names: list of phase name strings
        tau_corridor: corridor threshold (default 0.3)
        tau_release: release threshold (default 0.3)
        guard: guard steps after arming (default 5)

    Returns:
        (emitted: bool, emit_step: int)  emit_step=-1 if not emitted
    """
    state = "IDLE"
    arm_step = -1
    n = len(corridor_p)
    for step in range(n):
        if state == "IDLE":
            if phase_names[step] == "stable_carry" and corridor_p[step] > tau_corridor:
                state = "ARMED"
                arm_step = step
        elif state == "ARMED":
            if (step >= arm_step + guard and
                    corridor_p[step] > tau_corridor and
                    release_p[step] < tau_release):
                return True, step
    return False, -1


def model_to_scores(model, features_tensor):
    """Convert model output to (corridor_p, release_p, phase_names).

    Args:
        model: SC5MLPV1 instance in eval mode
        features_tensor: torch.Tensor (n_steps, 25) already normalized

    Returns:
        (corridor_p: np.ndarray, release_p: np.ndarray, phase_names: list[str])
    """
    import torch
    import numpy as np
    from gripper_attack.sc5mlp_v1 import SC5_PHASES

    with torch.no_grad():
        out = model(features_tensor)
    cp = torch.sigmoid(out["corridor_logit"]).squeeze(-1).numpy()
    rp = torch.sigmoid(out["release_logit"]).squeeze(-1).numpy()
    phase_idx = out["phase_logits"].argmax(dim=-1).numpy()
    phase_names = [SC5_PHASES[p] for p in phase_idx]
    return cp, rp, phase_names


# ── Regression test: fixed traces for equivalence with authority runtime ──

def _smoke_test():
    """Verify FSM on synthetic traces. Not a scientific result — engineering only."""
    cp = [0.1, 0.1, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 0.1]
    rp = [0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.5, 0.5]
    phases = ["approach"] * 3 + ["stable_carry"] * 6 + ["release_safe"] * 3
    emitted, step = run_fsm_legacy_v1(cp, rp, phases)
    # Should arm at step 2 (first stable_carry with cp>0.3)
    # Should emit at step 7 (arm_step 2 + guard 5 = step 7, cp>0.3, rp<0.3)
    assert emitted, "Should emit on synthetic trace"
    assert step == 8, f"Expected emit at step 8 (arm=3 + guard=5), got {step}"
    return True


if __name__ == "__main__":
    if _smoke_test():
        print("score_fsm_legacy_v1: smoke test PASS")
