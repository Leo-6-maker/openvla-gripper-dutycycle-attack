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

F2-B0.1 additions:
  - Event-level metrics (entries vs occupancy)
  - Per-emit safety audit (open_streak, qpos, Teacher membership)
  - 50-episode primary reason classification
  - Production EventFSM NOT modified.

CPU only.
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
OPEN_RESET_K = 5  # F2-B0 proposed release parameter (not yet frozen in SC5)
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
# F2-B0: Close-event latch FSM (shared, importable)
# ═══════════════════════════════════════════════════════════════════════════════

def run_b0_latched_fsm(probs, close_masks, eef_z, T, open_reset_k=OPEN_RESET_K):
    """Close-event latch FSM — minimal phase-lock repair.

    Returns:
      states: list[str] per-step state
      emits: list[int] emit step indices
      events: list[dict] per-event summary
    """
    state = "IDLE"
    grasp_persist = 0
    open_streak = 0
    emitted_this_event = False
    total_emits = 0
    anchor_step = -1
    anchor_eef_z = 0.0
    event_latched = False
    states, emits_out = [], []
    current_event = None
    events = []

    def _close_event():
        d = {
            "state": "RESET",
            "emit": False,
            "armed_entry": False,
            "armed_survived": False,
            "vertical_pass": False,
            "reset_reason": None,
        }
        return d

    for t in range(T):
        detected = probs[t] > GRASP_THRESHOLD
        cc = close_masks[t]
        eef = eef_z[t]

        if not cc:
            open_streak += 1
        else:
            open_streak = 0

        if open_streak >= open_reset_k:
            if state not in ("IDLE",) and current_event is not None:
                current_event["reset_reason"] = "SUSTAINED_OPEN_K{}_at_t{}".format(open_streak, t)
                current_event["state"] = "RESET"
            state = "RESET"
            event_latched = False
            grasp_persist = 0
            emitted_this_event = False
            current_event = None

        if cc and not event_latched:
            state = "CLOSE_EVENT_LATCHED"
            event_latched = True
            grasp_persist = 0
            emitted_this_event = False
            open_streak = 0
            current_event = _close_event()
            current_event["latch_step"] = t
            events.append(current_event)

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
                if current_event is not None:
                    current_event["armed_entry"] = True
                    current_event["armed_step"] = t
                    current_event["armed_open_streak"] = open_streak

        if state == "ARMED":
            if current_event is not None:
                current_event["armed_survived"] = True

        if state == "ARMED" and not emitted_this_event:
            if eef - anchor_eef_z >= TRANSPORT_VERT:
                state = "EVENT_CANDIDATE"
                if current_event is not None:
                    current_event["vertical_pass"] = True
                    current_event["vertical_step"] = t
                    current_event["vertical_open_streak"] = open_streak

        emit = False
        if state == "EVENT_CANDIDATE" and not emitted_this_event:
            if total_emits < MAX_EMITS:
                emitted_this_event = True
                total_emits += 1
                state = "EMITTED"
                emit = True
                if current_event is not None:
                    current_event["emit"] = True
                    current_event["emit_step"] = t
                    current_event["emit_open_streak"] = open_streak
                    current_event["state"] = "EMITTED"

        states.append(state)
        if emit:
            emits_out.append(t)

    return states, emits_out, events


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison + safety audit on fold-0 data
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu")
    model = load_student(device)

    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]

    # Fail-closed verify
    for identity in val_ids:
        parts = identity.split("/")
        tp = TEACHER_ROOT / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        sp = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        if not tp.is_file():
            raise SystemExit("TEACHER_MISSING:{}".format(identity))
        if not sp.is_file():
            raise SystemExit("S1_MISSING:{}".format(identity))

    total_emits = 0
    eps_emit = 0
    emit_details = []
    ep_causes = defaultdict(int)

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

        # B0 FSM
        states, emits, events = run_b0_latched_fsm(probs, close_masks, eef_z, T)

        if emits:
            eps_emit += 1
            total_emits += len(emits)

        # Per-emit safety audit
        for e_idx, e_step in enumerate(emits):
            gripper_qpos = float(s1_recs[e_step]["features_25d"][1]) if e_step < T else 0
            opening_proxy = float(s1_recs[e_step]["features_25d"][2]) if e_step < T else 0
            open_streak_at_emit = float(s1_recs[e_step]["features_25d"][14]) if e_step < T else 0
            close_streak_at_emit = float(s1_recs[e_step]["features_25d"][13]) if e_step < T else 0

            # Find the emitting event
            emitting_event = None
            for ev in events:
                if ev.get("emit") and ev.get("emit_step") == e_step:
                    emitting_event = ev
                    break

            # Physics Teacher positive segment check
            teacher_labels = []
            for t in range(T):
                tr = teacher_recs[t]
                cc = bool(tr.get("candidate_close", False))
                valid = bool(tr.get("student_valid", True))
                known = bool(tr.get("known_mask", True))
                sg = float(tr.get("stable_grasp_score", 0))
                teacher_labels.append(cc and valid and known and sg >= 0.3)

            in_positive = teacher_labels[e_step] if e_step < T else False

            detail = {
                "identity": identity,
                "emit_step": e_step,
                "armed_open_streak": emitting_event.get("armed_open_streak") if emitting_event else None,
                "vertical_open_streak": emitting_event.get("vertical_open_streak") if emitting_event else None,
                "emit_open_streak": emitting_event.get("emit_open_streak") if emitting_event else None,
                "gripper_qpos": round(gripper_qpos, 6),
                "opening_proxy": round(opening_proxy, 6),
                "student_prob": round(float(probs[e_step]), 6) if e_step < T else 0,
                "in_positive_segment": in_positive,
                "close_mask_at_emit": close_masks[e_step] if e_step < T else False,
                "s1_close_streak": float(close_streak_at_emit),
                "s1_open_streak": float(open_streak_at_emit),
            }
            emit_details.append(detail)

        # Episode primary reason classification
        n_latched = len(events)
        n_armed = sum(1 for ev in events if ev.get("armed_entry"))
        n_armed_survived = sum(1 for ev in events if ev.get("armed_survived"))
        n_vertical = sum(1 for ev in events if ev.get("vertical_pass"))
        n_emit_events = sum(1 for ev in events if ev.get("emit"))
        n_released = sum(1 for ev in events if ev.get("reset_reason"))

        if n_emit_events > 0:
            ep_causes["EMIT"] += 1
        elif n_latched == 0:
            ep_causes["NO_CLOSE_LATCH"] += 1
        elif n_armed_survived == 0 and n_armed > 0:
            ep_causes["NO_STUDENT_PERSISTENCE_WHILE_LATCHED"] += 1
        elif n_armed_survived > 0 and n_vertical == 0:
            ep_causes["ARMED_NO_VERTICAL_PASS"] += 1
        elif n_released > 0 and n_armed == 0:
            ep_causes["LATCH_RELEASED_BEFORE_CONFIRMATION"] += 1
        else:
            ep_causes["UNCLASSIFIED"] += 1

    # ── Report ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("F2-B0.1: Emit Safety Audit + Episode Classification")
    print("  {} episodes, full-FIT checkpoint".format(len(val_ids)))
    print("=" * 70)

    print("\nEvent-level metrics (B0):")
    print("  Total emits: {}".format(total_emits))
    print("  Episodes with emit: {}".format(eps_emit))

    print("\nPer-episode primary reason:")
    for cause in ["EMIT", "ARMED_NO_VERTICAL_PASS", "NO_STUDENT_PERSISTENCE_WHILE_LATCHED",
                  "NO_CLOSE_LATCH", "LATCH_RELEASED_BEFORE_CONFIRMATION", "UNCLASSIFIED"]:
        if cause in ep_causes:
            print("  {}: {}".format(cause, ep_causes[cause]))

    total_classified = sum(ep_causes.values())
    print("  TOTAL: {} (must equal {})".format(total_classified, len(val_ids)))

    print("\nPer-emit safety audit ({} emits):".format(len(emit_details)))
    if emit_details:
        print("  {:>40s} {:>6s} {:>8s} {:>8s} {:>8s} {:>10s} {:>8s} {:>6s} {:>10s}".format(
            "identity", "step", "arm_os", "vert_os", "emit_os", "qpos", "opening", "pos?", "close?"))
        print("  " + "-" * 114)
        for d in emit_details[:20]:  # First 20
            print("  {:>40s} {:>6d} {:>8s} {:>8s} {:>8s} {:>10.4f} {:>8.4f} {:>6s} {:>10s}".format(
                d["identity"][-40:], d["emit_step"],
                str(d["armed_open_streak"]), str(d["vertical_open_streak"]),
                str(d["emit_open_streak"]),
                d["gripper_qpos"], d["opening_proxy"],
                "Y" if d["in_positive_segment"] else "N",
                "Y" if d["close_mask_at_emit"] else "N"))

    # Safety summary
    emits_with_open = sum(1 for d in emit_details if d["emit_open_streak"] is not None and d["emit_open_streak"] > 0)
    emits_in_positive = sum(1 for d in emit_details if d["in_positive_segment"])
    emits_with_close = sum(1 for d in emit_details if d["close_mask_at_emit"])
    print("\nSafety summary:")
    print("  Emits with open_streak > 0 at emit: {}".format(emits_with_open))
    print("  Emits inside Teacher positive segment: {}".format(emits_in_positive))
    print("  Emits at close_mask=True step: {}".format(emits_with_close))


if __name__ == "__main__":
    main()
