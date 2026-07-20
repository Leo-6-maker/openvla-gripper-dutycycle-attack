#!/usr/bin/env python3
"""Gate F2-B0: Close-event latch FSM — minimal phase-lock repair.

Only change from production FSM:
  - Close pulse establishes a latched event (does not require close at
    confirmation step).
  - Single open command = noise, does not reset.
  - Sustained open streak (>= OPEN_RESET_K) releases the latch → RESET.

Everything else FROZEN:
  - Student probability > 0.5, persistence = 3
  - First-positive anchor
  - Vertical guard = 0.02 m
  - Max 1 emit per episode
  - Route support

CPU only. Does NOT modify production EventFSM in r10_4d_passive.py.
"""

import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Frozen constants ─────────────────────────────────────────────────────────
GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
TRANSPORT_VERT = 0.02
OPEN_RESET_K = 5  # sustained open streak to release latch
MAX_EMITS = 1

# ── Paths ────────────────────────────────────────────────────────────────────
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels"
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"
BUNDLE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(25, 64, 2, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def load_student(device):
    ckpt_path = BUNDLE / "full_fit_deploy.pt"
    if not ckpt_path.is_file():
        raise SystemExit("CHECKPOINT_MISSING")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = StudentModel().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def jsonl(path):
    if not path.is_file():
        raise SystemExit("FILE_MISSING:{}".format(path))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("FILE_EMPTY:{}".format(path))
    return [json.loads(l) for l in lines]


# ═══════════════════════════════════════════════════════════════════════════════
# Production FSM (exact copy)
# ═══════════════════════════════════════════════════════════════════════════════

def run_production_fsm(probs, close_masks, eef_z, T):
    state = "IDLE"
    grasp_persist = 0
    emitted_this_event = False
    total_emits = 0
    anchor_step = -1
    anchor_eef_z = 0.0
    states, emits, arm_steps, survived = [], [], [], []

    for t in range(T):
        detected = probs[t] > GRASP_THRESHOLD
        cc = close_masks[t]
        eef = eef_z[t]

        if state == "IDLE" and cc:
            state = "CLOSE_CANDIDATE"
            grasp_persist = 0
            emitted_this_event = False

        if state == "CLOSE_CANDIDATE":
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

        reset_this_step = False
        if state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not cc:
            state = "RESET"
            reset_this_step = True

        if not reset_this_step and state == "ARMED":
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

        if state == "RESET" and cc:
            state = "CLOSE_CANDIDATE"
            grasp_persist = 0
            emitted_this_event = False

        states.append(state)
        if emit:
            emits.append(t)

    return states, emits, arm_steps, survived


# ═══════════════════════════════════════════════════════════════════════════════
# F2-B0: Close-event latch FSM
# ═══════════════════════════════════════════════════════════════════════════════
# Only change: close pulse latches event. Single open = noise. Sustained open
# streak resets. Everything else identical to production.

def run_b0_latched_fsm(probs, close_masks, eef_z, T, open_reset_k=OPEN_RESET_K):
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
        cc = close_masks[t]
        eef = eef_z[t]

        # Track open streak for sustained-release detection
        if not cc:
            open_streak += 1
        else:
            open_streak = 0

        # Sustained open → release latch
        if open_streak >= open_reset_k:
            if state not in ("IDLE",):
                state = "RESET"
            event_latched = False
            grasp_persist = 0
            emitted_this_event = False

        # Close pulse establishes event latch
        if cc and not event_latched:
            state = "CLOSE_EVENT_LATCHED"
            event_latched = True
            grasp_persist = 0
            emitted_this_event = False
            open_streak = 0

        # Student persistence accumulation (in LATCHED or CLOSE_CANDIDATE)
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

        # B0: NO same-step reset on single open
        # ARMED survives unless sustained open streak triggers release

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
# Comparison on fold-0 data
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu")
    model = load_student(device)

    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]

    # Fail-closed: verify all files exist
    for identity in val_ids:
        parts = identity.split("/")
        tp = TEACHER_ROOT / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        sp = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        if not tp.is_file():
            raise SystemExit("TEACHER_MISSING:{}".format(identity))
        if not sp.is_file():
            raise SystemExit("S1_MISSING:{}".format(identity))

    prod_total_emits = 0
    b0_total_emits = 0
    prod_eps_emit = 0
    b0_eps_emit = 0
    prod_survived_armed = 0
    b0_survived_armed = 0
    prod_armed = 0
    b0_armed = 0

    for identity in val_ids:
        parts = identity.split("/")
        suite, task, state = parts

        teacher_recs = jsonl(TEACHER_ROOT / suite / task / state / "physics_teacher_v21.jsonl")
        s1_recs = jsonl(S1_ROOT / suite / task / state / "student_input_records.jsonl")
        T = len(s1_recs)
        if len(teacher_recs) != T:
            raise SystemExit("LENGTH_MISMATCH:{}".format(identity))

        features_list = [s1_recs[t]["features_25d"] for t in range(T)]
        with torch.no_grad():
            hidden = torch.zeros(2, 1, 64, device=device)
            probs = []
            for feats in features_list:
                x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                _, hidden = model.encoder(x, hidden)
                logit = model.head_multi(hidden[-1]).item()
                probs.append(1.0 / (1.0 + math.exp(-logit)))
            probs = np.array(probs, dtype=np.float64)

        close_masks = [float(s1_recs[t]["features_25d"][0]) <= 0.5 for t in range(T)]
        eef_z = np.array([float(s1_recs[t]["features_25d"][5]) for t in range(T)], dtype=np.float64)

        # Production
        p_states, p_emits, p_arm, p_surv = run_production_fsm(probs, close_masks, eef_z, T)
        prod_total_emits += len(p_emits)
        prod_armed += len(p_arm)
        prod_survived_armed += len(p_surv)
        if p_emits:
            prod_eps_emit += 1

        # B0
        b_states, b_emits, b_arm, b_surv = run_b0_latched_fsm(probs, close_masks, eef_z, T)
        b0_total_emits += len(b_emits)
        b0_armed += len(b_arm)
        b0_survived_armed += len(b_surv)
        if b_emits:
            b0_eps_emit += 1

    print("=" * 70)
    print("F2-B0: Close-Event Latch FSM vs Production FSM")
    print("  {} episodes, full-FIT checkpoint (qualified)".format(len(val_ids)))
    print("=" * 70)
    print("{:<20s} {:>12s} {:>12s}".format("", "Production", "B0-Latch"))
    print("-" * 46)
    print("{:<20s} {:>12d} {:>12d}".format("Total ARMED steps", prod_armed, b0_armed))
    print("{:<20s} {:>12d} {:>12d}".format("Survived ARMED", prod_survived_armed, b0_survived_armed))
    print("{:<20s} {:>12d} {:>12d}".format("Total emits", prod_total_emits, b0_total_emits))
    print("{:<20s} {:>12d} {:>12d}".format("Episodes with emit", prod_eps_emit, b0_eps_emit))
    print("-" * 46)

    phase_lock_relieved = b0_survived_armed - prod_survived_armed
    emit_gain = b0_total_emits - prod_total_emits
    print("Phase-lock relief: {} ARMED → survived".format(phase_lock_relieved))
    print("Emit gain: {}".format(emit_gain))
    print()

    # Waterfall comparison
    print("Production waterfall: P3={} → P4={} → P7={}".format(prod_armed, prod_survived_armed, prod_total_emits))
    print("B0 waterfall:         P3={} → P4={} → P7={}".format(b0_armed, b0_survived_armed, b0_total_emits))
    gap_armed = b0_armed - b0_survived_armed
    if gap_armed > 0:
        print("  B0 still loses {} ARMED→EMIT (vertical guard)".format(gap_armed))


if __name__ == "__main__":
    main()
