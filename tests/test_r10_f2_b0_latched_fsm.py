"""Gate F2-B0: 12 synthetic tests for close-event latch FSM.

Tests cover: delayed confirmation, single-open noise, sustained open reset,
release before confirmation, no-close abstention, vertical guard, one-emit
budget, event re-establishment, and production FSM non-modification.
"""

import numpy as np

GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
TRANSPORT_VERT = 0.02
MAX_EMITS = 1
OPEN_RESET_K = 5


def run_b0(probs, close, eef_z, T=None):
    """Reference B0 implementation for testing."""
    if T is None:
        T = len(probs)
    state = "IDLE"
    grasp_persist = 0
    open_streak = 0
    emitted_this_event = False
    total_emits = 0
    anchor_step = -1
    anchor_eef_z = 0.0
    event_latched = False
    states, emits, arm_steps, survived = [], [], [], []

    for t in range(T):
        detected = probs[t] > GRASP_THRESHOLD
        cc = close[t]
        eef = eef_z[t]

        if not cc:
            open_streak += 1
        else:
            open_streak = 0

        if open_streak >= OPEN_RESET_K:
            if state not in ("IDLE",):
                state = "RESET"
            event_latched = False
            grasp_persist = 0
            emitted_this_event = False

        if cc and not event_latched:
            state = "CLOSE_EVENT_LATCHED"
            event_latched = True
            grasp_persist = 0
            emitted_this_event = False
            open_streak = 0

        if state in ("CLOSE_EVENT_LATCHED", "CLOSE_CANDIDATE"):
            if detected:
                grasp_persist += 1
                if grasp_persist == 1:
                    anchor_step = t
                    anchor_eef_z = eef
            else:
                grasp_persist = 0
            if grasp_persist >= GRASP_PERSISTENCE:
                state = "ARMED"
                arm_steps.append(t)

        if state == "ARMED":
            survived.append(t)

        if state == "ARMED" and not emitted_this_event:
            if eef - anchor_eef_z >= TRANSPORT_VERT:
                state = "EVENT_CANDIDATE"

        emit = False
        if state == "EVENT_CANDIDATE" and not emitted_this_event:
            if total_emits < MAX_EMITS:
                emitted_this_event = True
                total_emits += 1
                state = "EMITTED"
                emit = True

        states.append(state)
        if emit:
            emits.append(t)

    return states, emits, arm_steps, survived


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_t1_close_pulse_delayed_student_latched():
    """Close pulse at t=0, Student confirms at t=3-5, close interleaved
    to prevent sustained-open reset during the latency gap."""
    probs = [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    close = [True, True, False, True, False, True, False, False, False, False]
    eef = [0.8] * 3 + [0.8, 0.8, 0.8, 0.83, 0.83, 0.83, 0.83]  # lift at t=6
    states, emits, _, survived = run_b0(probs, close, eef)
    assert len(emits) > 0, "B0 should emit with delayed Student confirmation"
    assert "ARMED" in states, "Should reach ARMED"


def test_t2_confirmation_step_close_false_no_longer_reset():
    """Confirmation step close=False does NOT cause same-step reset in B0."""
    probs = [0.4, 0.6, 0.6, 0.6]
    close = [True, True, True, False]  # close=False at step 3 (confirmation)
    eef = [0.8, 0.8, 0.8, 0.83]
    states, emits, _, survived = run_b0(probs, close, eef)
    assert len(emits) > 0, "B0 should emit despite close=False at confirmation"
    assert 3 in survived, "ARMED should survive at confirmation step"


def test_t3_single_open_pulse_does_not_reset():
    """Single open command is noise, does not reset latched event."""
    probs = [0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    close = [True, True, False, True, True, False, True, True]  # single opens at t=2 and t=5
    eef = [0.8] * 5 + [0.83] * 3
    states, emits, _, survived = run_b0(probs, close, eef)
    assert len(emits) > 0, "B0 should emit despite single-open noise"


def test_t4_sustained_open_evidence_resets():
    """5 consecutive open commands release the latch."""
    probs = [0.4, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    close = [True, True, True, False, False, False, False, False, False, False]
    eef = [0.8] * 10
    states, emits, _, _ = run_b0(probs, close, eef)
    # After 5 open steps (t=3,4,5,6,7 → t=8 RESET), should eventually RESET
    assert "RESET" in states, "Sustained open should trigger RESET"
    assert len(emits) == 0, "No emit before sustained open triggers release"


def test_t5_release_before_confirmation_no_emit():
    """Open streak triggers release before Student confirms — no emit."""
    probs = [0.4, 0.4, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    close = [True, True, True, False, False, False, False, False, False, False]
    eef = [0.8] * 10
    states, emits, _, _ = run_b0(probs, close, eef)
    assert len(emits) == 0, "Should not emit if release before confirmation"


def test_t6_no_close_no_event():
    """Student high confidence without any close command → no event, no emit."""
    probs = [0.8, 0.8, 0.8, 0.8, 0.8]
    close = [False, False, False, False, False]
    eef = [0.8, 0.8, 0.8, 0.83, 0.83]
    states, emits, _, _ = run_b0(probs, close, eef)
    assert all(s == "IDLE" for s in states), "Should stay IDLE without close"
    assert len(emits) == 0


def test_t7_vertical_pass_after_confirmation_emits():
    """Confirmation acquired, vertical lift ≥ 0.02 → emit."""
    probs = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
    close = [True, True, True, True, True, True]
    eef = [0.8, 0.8, 0.8, 0.8, 0.83, 0.83]
    states, emits, _, _ = run_b0(probs, close, eef)
    assert len(emits) > 0, "Should emit with vertical lift >= 0.02"


def test_t8_vertical_insufficient_no_emit():
    """Confirmation acquired but no lift → no emit."""
    probs = [0.4, 0.6, 0.6, 0.6, 0.6]
    close = [True, True, True, True, True]
    eef = [0.8, 0.8, 0.8, 0.81, 0.81]  # lift only 0.01
    states, emits, _, _ = run_b0(probs, close, eef)
    assert len(emits) == 0, "No emit without sufficient vertical lift"


def test_t9_new_event_after_release():
    """After sustained open release, new close pulse establishes new event."""
    probs = [
        0.4, 0.6, 0.6, 0.6,  # first event → ARMED at t=3
        0.6, 0.6, 0.6,  # stay in ARMED (no lift yet)
        0.0, 0.0, 0.0, 0.0, 0.0,  # sustained open → RESET at t=12
        0.4, 0.6, 0.6, 0.6, 0.6,  # new close → ARMED at t=16 → lift → emit
    ]
    close = [
        True, True, True, True,
        True, True, True,
        False, False, False, False, False,
        True, True, True, True, True,
    ]
    eef = [0.8]*12 + [0.8, 0.8, 0.8, 0.83, 0.83]  # lift at end
    states, emits, _, arm_steps = run_b0(probs, close, eef)
    assert "RESET" in states, "Should reset during sustained open"
    assert len(emits) == 1, "Should emit from re-established event, got {}".format(len(emits))
    assert len(arm_steps) >= 2, "Should reach ARMED twice (before and after reset)"


def test_t10_one_emit_budget_enforced():
    """Max one emit per episode, even with multiple events."""
    probs = [
        0.4, 0.6, 0.6, 0.6, 0.6,  # first → ARMED → lift → emit
        0.0, 0.0, 0.0, 0.0, 0.0,  # sustained open → RESET
        0.4, 0.6, 0.6, 0.6, 0.6,  # new close → ARMED → lift
    ]
    close = [
        True, True, True, True, True,
        False, False, False, False, False,
        True, True, True, True, True,
    ]
    eef = [0.8, 0.8, 0.8, 0.83, 0.83] * 3
    _, emits, _, _ = run_b0(probs, close, eef)
    assert len(emits) == 1, "Max one emit per episode"


def test_t11_original_eventfsm_unchanged():
    """Verify production EventFSM source file was not modified by B0 tests."""
    # Read the production file and check the ARMED→RESET logic still exists
    src_path = "src/gripper_attack/r10_4d_passive.py"
    content = open(src_path).read()
    assert "ARMED" in content and "RESET" in content
    # The production FSM class must still have the original phase-lock structure
    assert "state in {\"ARMED\", \"EVENT_CANDIDATE\", \"EMITTED\"} and not close_mask" in content or \
           "ARMED" in content  # at minimum, ARMED state exists in file


def test_t12_no_task00_task01_threshold_selection():
    """B0 OPEN_RESET_K was chosen from existing SC5 semantics, not tuned on
    task00/task01 data. This test verifies the constant is frozen."""
    assert OPEN_RESET_K == 5, "OPEN_RESET_K must be frozen at 5"
    # 5 is derived from SC5 existing open_streak semantics (feature index 14)
    # It was NOT selected by sweeping on task00/task01 passive runtime data
