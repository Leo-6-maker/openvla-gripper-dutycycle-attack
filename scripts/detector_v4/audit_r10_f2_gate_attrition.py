#!/usr/bin/env python3
"""Gate F2-A: Gate attrition waterfall on fold-0 validation identities.

F2-A1 input binding (qualified):
  Student checkpoint: full-FIT deployment bundle (only available checkpoint)
    → OOF by initial state (validation uses states 00-04, training uses 05-19)
    → NOT OOF by task (all 10 tasks seen during training)
    → LIMITATION: not true fold-specific OOF checkpoint
  Teacher event stream: Physics Teacher V2.1 via R10.1A derive_per_step_labels()
  Fold manifest: OFFICIAL_V3_FIT_FOLDS_V1_d31187f, fold-0 validation
  Feature contract: 25D SC5 order, SHA256 3d1101d2...

CPU only. Reads from frozen artifacts. No training. No FSM changes.

For each Teacher target event, outputs:
  - close segments and streak lengths
  - Student probability timeline
  - First persistence-3 confirmation
  - Confirmation close_mask state
  - Production FSM state transitions
  - Anchor and motion diagnostics
  - Emit or no-emit reason
"""

import json, math, sys, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels"
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"
BUNDLE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")

GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
TRANSPORT_VERT = 0.02
TRANSPORT_LAT = 0.015
TRANSPORT_PATH = 0.03


# ── Student model (must match RoutedGraspDetector) ──────────────────────────
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(25, 64, 2, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def load_student(device):
    ckpt = torch.load(str(BUNDLE / "full_fit_deploy.pt"), map_location=device, weights_only=False)
    model = StudentModel().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── Frozen Teacher: grasp_established from R10.1A contract ───────────────────
def derive_grasp_established(teacher_rec):
    cc = bool(teacher_rec.get("candidate_close", False))
    valid = bool(teacher_rec.get("student_valid", True))
    known = bool(teacher_rec.get("known_mask", True))
    sg = float(teacher_rec.get("stable_grasp_score", 0))
    dwell = int(teacher_rec.get("stable_grasp_dwell", 0))
    return cc and valid and known and sg >= 0.3


# ── Production FSM (exact match to r10_4d_passive.py EventFSM) ───────────────
def run_production_fsm(student_probs, close_masks, eef_xyz, T):
    """Run exact production FSM: IDLE→CLOSE_CANDIDATE→ARMED→EVENT_CANDIDATE→EMIT→RESET."""
    state = "IDLE"
    grasp_persist = 0
    emitted_this_event = False
    total_emits = 0
    event_id = 0
    anchor_step = -1
    anchor_eef_z = 0.0
    close_onset = -1
    close_onset_eef = np.zeros(3)
    arm_steps = []

    states = []  # per-step state
    emits = []

    for t in range(T):
        detected = student_probs[t] > GRASP_THRESHOLD
        cc = close_masks[t]
        eef = eef_xyz[t]

        if state == "IDLE" and cc:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            grasp_persist = 0
            emitted_this_event = False
            close_onset = t
            close_onset_eef = eef.copy()

        if state == "CLOSE_CANDIDATE":
            if detected:
                grasp_persist += 1
                if grasp_persist == 1:
                    anchor_step = t
                    anchor_eef_z = eef[2]
            else:
                grasp_persist = 0
            if grasp_persist >= GRASP_PERSISTENCE:
                state = "ARMED"
                arm_steps.append(t)

        if state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not cc:
            state = "RESET"

        if state == "ARMED" and not emitted_this_event:
            if eef[2] - anchor_eef_z >= TRANSPORT_VERT:
                state = "EVENT_CANDIDATE"

        emit = False
        if state == "EVENT_CANDIDATE" and not emitted_this_event:
            if total_emits < 1:
                emitted_this_event = True
                total_emits += 1
                state = "EMITTED"
                emit = True

        if state == "RESET" and cc:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            grasp_persist = 0
            emitted_this_event = False
            close_onset = t
            close_onset_eef = eef.copy()

        states.append(state)
        if emit:
            emits.append(t)

    return states, emits, arm_steps


def analyze_event_attrition(teacher_labels, probs, close_masks, eef_xyz, features, teacher_recs_raw):
    """Per-event attrition analysis."""
    T = len(probs)
    fsm_states, fsm_emits, fsm_arm_steps = run_production_fsm(probs, close_masks, eef_xyz, T)

    # Extract Teacher events (continuous grasp_established runs)
    events = []
    in_event = False
    start = -1
    for t in range(T):
        if teacher_labels[t] and not in_event:
            start = t
            in_event = True
        elif not teacher_labels[t] and in_event:
            events.append({"start": start, "end": t - 1})
            in_event = False
    if in_event:
        events.append({"start": start, "end": T - 1})

    event_rows = []
    for eid, event in enumerate(events):
        es, ee = event["start"], event["end"]
        row = {"event_id": eid, "teacher_start": es, "teacher_end": ee, "duration": ee - es + 1}

        # Close segments within event
        close_segs = []
        in_close = False
        cs = -1
        for t in range(max(0, es - 10), min(T, ee + 1)):
            if close_masks[t] and not in_close:
                cs = t
                in_close = True
            elif not close_masks[t] and in_close:
                close_segs.append((cs, t - 1))
                in_close = False
        if in_close:
            close_segs.append((cs, min(T - 1, ee)))
        row["close_segments"] = close_segs
        row["close_segment_count"] = len(close_segs)
        row["close_max_streak"] = max((e - s + 1 for s, e in close_segs), default=0)

        # Student persistence within event
        persistence_runs = []
        in_run = False
        rs = -1
        for t in range(es, ee + 1):
            if probs[t] > GRASP_THRESHOLD and not in_run:
                rs = t
                in_run = True
            elif probs[t] <= GRASP_THRESHOLD and in_run:
                run_len = t - rs
                if run_len >= GRASP_PERSISTENCE:
                    persistence_runs.append({"start": rs, "end": t - 1, "length": run_len})
                in_run = False
        if in_run:
            run_len = ee + 1 - rs
            if run_len >= GRASP_PERSISTENCE:
                persistence_runs.append({"start": rs, "end": ee, "length": run_len})

        row["persistence_runs"] = persistence_runs
        row["has_persistence"] = len(persistence_runs) > 0
        if persistence_runs:
            first = persistence_runs[0]
            row["first_persistence_start"] = first["start"]
            row["first_persistence_end"] = first["end"]
            row["first_confirmation_step"] = first["start"] + GRASP_PERSISTENCE - 1
            conf_step = row["first_confirmation_step"]
            row["confirmation_close_mask"] = close_masks[conf_step] if conf_step < T else False
            row["confirmation_fsm_state"] = fsm_states[conf_step] if conf_step < T else "?"
            row["confirmation_student_prob"] = float(probs[conf_step]) if conf_step < T else 0.0

            # Check if close occurred before confirmation
            close_before = any(s < conf_step for s, e in close_segs)
            row["close_before_confirmation"] = close_before

            # Phase-lock check: confirmation step close_mask
            row["phase_lock_at_confirmation"] = not row["confirmation_close_mask"]

            # Motion from first-positive anchor
            if conf_step < T:
                first_pos_anchor = first["start"]
                anchor_z = eef_xyz[first_pos_anchor][2]
                max_vert_first = max(eef_xyz[t][2] - anchor_z for t in range(first_pos_anchor, ee + 1))
                row["max_vert_first_positive_anchor"] = float(max_vert_first)
                row["vert_pass_first_positive"] = max_vert_first >= TRANSPORT_VERT

            # Motion from close-onset anchor (first close before first persistence)
            co_step = None
            for s, e in close_segs:
                if s < conf_step:
                    co_step = s
                    break
            if co_step is not None:
                co_z = eef_xyz[co_step][2]
                max_vert_co = max(eef_xyz[t][2] - co_z for t in range(co_step, ee + 1))
                max_lat_co = max(
                    np.linalg.norm(eef_xyz[t][:2] - eef_xyz[co_step][:2])
                    for t in range(co_step, ee + 1))
                row["close_onset_step"] = co_step
                row["max_vert_close_onset_anchor"] = float(max_vert_co)
                row["max_lat_close_onset_anchor"] = float(max_lat_co)
                row["vert_pass_close_onset"] = max_vert_co >= TRANSPORT_VERT
                row["transport_3d_pass"] = (max_vert_co >= TRANSPORT_VERT or
                                            max_lat_co >= TRANSPORT_LAT)
            else:
                row["close_onset_step"] = -1
                row["max_vert_close_onset_anchor"] = 0.0
                row["max_lat_close_onset_anchor"] = 0.0
                row["vert_pass_close_onset"] = False
                row["transport_3d_pass"] = False

        else:
            row["first_confirmation_step"] = -1
            row["confirmation_close_mask"] = False
            row["confirmation_fsm_state"] = "NO_PERSISTENCE"
            row["phase_lock_at_confirmation"] = False

        # FSM state: did it reach ARMED within this event?
        row["fsm_armed_in_event"] = any(
            es <= arm_step <= ee for arm_step in fsm_arm_steps)
        row["fsm_emit_in_event"] = any(
            es <= emit_step <= ee for emit_step in fsm_emits)

        # Check if FSM emitted anywhere (not just this event)
        row["fsm_any_emit"] = len(fsm_emits) > 0
        row["fsm_emit_steps"] = fsm_emits

        # Classify root cause
        row["root_cause"] = classify_event(row)

        event_rows.append(row)

    return event_rows


def classify_event(row):
    """Primary root cause classification per event."""
    if row["fsm_any_emit"] and row["fsm_emit_in_event"]:
        return "CURRENT_FSM_EMIT"

    if row["close_segment_count"] == 0:
        return "NO_CANONICAL_CLOSE_EVIDENCE"

    if not row["has_persistence"]:
        return "NO_OOF_STUDENT_PERSISTENCE"

    conf = row.get("first_confirmation_step", -1)
    if conf < 0:
        return "NO_CONFIRMATION_STEP"

    if not row.get("close_before_confirmation", False):
        return "NO_CLOSE_BEFORE_CONFIRMATION"

    if row.get("phase_lock_at_confirmation", False):
        return "PHASE_LOCK_RESET_AT_CONFIRMATION"

    if not row.get("fsm_armed_in_event", False):
        return "FSM_DID_NOT_REACH_ARMED"

    if not row.get("vert_pass_first_positive", False):
        if row.get("vert_pass_close_onset", False):
            return "VERTICAL_PASS_CLOSE_ONSET_BUT_FSM_USES_FIRST_POSITIVE"
        return "VERTICAL_GUARD_BLOCK_BOTH_ANCHORS"

    if not row.get("fsm_emit_in_event", False) and row.get("fsm_any_emit", False):
        return "EMIT_OUTSIDE_TARGET_EVENT"

    return "UNCLASSIFIABLE"


def main():
    device = torch.device("cpu")
    print("Loading Student checkpoint (full-FIT, qualified OOF by initial state)...")
    model = load_student(device)
    print("Student loaded: {} params".format(sum(p.numel() for p in model.parameters())))

    # Load fold-0 validation identities
    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]
    print("Fold-0 val multi_object: {} episodes".format(len(val_ids)))

    all_event_rows = []
    episode_stats = []
    errors = 0

    for idx, identity in enumerate(val_ids):
        parts = identity.split("/")
        suite, task, state = parts

        teacher_path = TEACHER_ROOT / suite / task / state / "physics_teacher_v21.jsonl"
        s1_path = S1_ROOT / suite / task / state / "student_input_records.jsonl"

        if not teacher_path.is_file() or not s1_path.is_file():
            errors += 1
            continue

        teacher_recs_raw = jsonl(teacher_path)
        s1_recs = jsonl(s1_path)
        T = min(len(teacher_recs_raw), len(s1_recs))
        if T < 10:
            errors += 1
            continue

        # Teacher labels
        teacher_labels = [derive_grasp_established(teacher_recs_raw[t]) for t in range(T)]

        # Student probabilities
        features = [s1_recs[t]["features_25d"] for t in range(T)]
        with torch.no_grad():
            hidden = torch.zeros(2, 1, 64, device=device)
            probs = []
            for feats in features:
                x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                _, hidden = model.encoder(x, hidden)
                logit = model.head_multi(hidden[-1]).item()
                prob = 1.0 / (1.0 + math.exp(-logit))
                probs.append(prob)
            probs = np.array(probs, dtype=np.float64)

        # Close masks (production: raw_action[-1] <= 0.5)
        close_masks = [float(s1_recs[t]["features_25d"][0]) <= 0.5 for t in range(T)]

        # EEF positions
        eef_xyz = np.array([
            [float(s1_recs[t]["features_25d"][3]),
             float(s1_recs[t]["features_25d"][4]),
             float(s1_recs[t]["features_25d"][5])]
            for t in range(T)
        ], dtype=np.float64)

        event_rows = analyze_event_attrition(
            teacher_labels, probs, close_masks, eef_xyz, features, teacher_recs_raw)

        for row in event_rows:
            row["identity"] = identity
            row["T"] = T
            all_event_rows.append(row)

        # Episode-level statistics
        n_teacher = len(event_rows)
        n_emit = sum(1 for r in event_rows if r["root_cause"] == "CURRENT_FSM_EMIT")
        n_phase_lock = sum(1 for r in event_rows if r["root_cause"] == "PHASE_LOCK_RESET_AT_CONFIRMATION")
        n_no_persist = sum(1 for r in event_rows if r["root_cause"] == "NO_OOF_STUDENT_PERSISTENCE")
        episode_stats.append({
            "identity": identity, "T": T,
            "n_teacher_events": n_teacher, "n_emit": n_emit,
            "n_phase_lock": n_phase_lock,
            "n_no_persistence": n_no_persist,
        })

    # ── Attrition Waterfall ──────────────────────────────────────────────────
    T0 = len(all_event_rows)
    T1 = sum(1 for r in all_event_rows if r["close_segment_count"] > 0)
    T2 = sum(1 for r in all_event_rows if r["has_persistence"])
    T3 = sum(1 for r in all_event_rows if r.get("close_before_confirmation", False))
    T4 = sum(1 for r in all_event_rows
             if r.get("has_persistence") and not r.get("phase_lock_at_confirmation", True)
             and r.get("first_confirmation_step", -1) >= 0)
    T5 = sum(1 for r in all_event_rows if r.get("fsm_armed_in_event", False))
    T6 = sum(1 for r in all_event_rows if r.get("vert_pass_first_positive", False))
    T7 = sum(1 for r in all_event_rows if r.get("vert_pass_close_onset", False))
    T8 = sum(1 for r in all_event_rows if r.get("transport_3d_pass", False))
    T9 = sum(1 for r in all_event_rows if r.get("fsm_emit_in_event", False))
    T10 = 0  # strict K10 not available in this scope

    # Root cause counts
    cause_counts = defaultdict(int)
    for r in all_event_rows:
        cause_counts[r["root_cause"]] += 1

    print("\n" + "=" * 70)
    print("Gate F2-A: Attrition Waterfall (Fold-0 val, {} episodes)".format(len(val_ids)))
    print("  {} errors (missing Teacher/S1)".format(errors))
    print("=" * 70)
    print("T0   Eligible Teacher target events:       {}".format(T0))
    print("T1   Has canonical close:                  {} ({:.1f}%)".format(T1, 100*T1/max(T0,1)))
    print("T2   Has Student persistence-3:             {} ({:.1f}%)".format(T2, 100*T2/max(T0,1)))
    print("T3   Close before confirmation:             {} ({:.1f}%)".format(T3, 100*T3/max(T0,1)))
    print("T4   Confirmation step close=True (no lock): {} ({:.1f}%)".format(T4, 100*T4/max(T0,1)))
    print("T5   FSM reached ARMED:                     {} ({:.1f}%)".format(T5, 100*T5/max(T0,1)))
    print("T6   Vertical pass (first-pos anchor):      {} ({:.1f}%)".format(T6, 100*T6/max(T0,1)))
    print("T7   Vertical pass (close-onset anchor):    {} ({:.1f}%)".format(T7, 100*T7/max(T0,1)))
    print("T8   3D transport pass:                     {} ({:.1f}%)".format(T8, 100*T8/max(T0,1)))
    print("T9   FSM emit inside Teacher event:         {} ({:.1f}%)".format(T9, 100*T9/max(T0,1)))
    print("-" * 70)

    print("\nRoot Cause Distribution:")
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print("  {}: {}".format(cause, count))

    # Write output
    output = {
        "schema": "R10_F2_GATE_ATTRITION_WATERFALL_V1",
        "fold": 0,
        "n_episodes": len(val_ids),
        "errors": errors,
        "waterfall": {"T0": T0, "T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6, "T7": T7, "T8": T8, "T9": T9},
        "root_cause_counts": dict(cause_counts),
        "episode_stats": episode_stats,
        "input_binding": {
            "student_checkpoint": "full_fit_deploy (qualified OOF by initial state)",
            "teacher": "Physics Teacher V2.1 via R10.1A contract",
            "fold_manifest": str(FOLD_ROOT),
            "fold_id": 0,
            "feature_contract_sha256": "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366",
        },
    }
    print("\n" + json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
