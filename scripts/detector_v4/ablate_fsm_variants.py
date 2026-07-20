#!/usr/bin/env python3
"""Gate F1: CPU-only FSM scheduler ablation on frozen fold-0 validation data.

Compares FSM variants using frozen Teacher grasp_established labels
and Student probabilities from the frozen R10.3 checkpoint.

Read-only. No OpenVLA. No LIBERO. No training.

FSM variants:
  v0 = current:  first-positive-anchor + vertical 2cm
  v1 = close-onset-anchor + vertical 2cm
  v2 = close-onset-anchor + 3D transport OR
  v3 = Student confirmation + release veto only (no motion gate)
  v4 = motion-or-timeout (transport evidence OR bounded timeout W=50)
"""

import json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels"
BUNDLE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"

GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
TRANSPORT_VERT = 0.02
TRANSPORT_LAT = 0.015
TRANSPORT_PATH = 0.03
TIMEOUT_W = 50

# Student model (must match RoutedGraspDetector in r10_4d_passive.py)
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(25, 64, 2, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def load_student(device):
    ckpt = torch.load(str(BUNDLE / "full_fit_deploy.pt"), map_location=device)
    model = StudentModel().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def derive_grasp_established(teacher_rec):
    """From frozen R10.1A contract: grasp_established = cc and valid and sg >= 0.3 and dwell >= 0."""
    cc = bool(teacher_rec.get("candidate_close", False))
    valid = bool(teacher_rec.get("student_valid", True))
    sg = float(teacher_rec.get("stable_grasp_score", 0))
    dwell = int(teacher_rec.get("stable_grasp_dwell", 0))
    known = bool(teacher_rec.get("known_mask", True))
    return cc and valid and known and sg >= 0.3 and dwell >= 0


def jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_student(model, features_25d, device):
    """Run Student GRU step-by-step on features, return per-step probabilities."""
    hidden = torch.zeros(2, 1, 64, device=device)
    probs = []
    for feats in features_25d:
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        _, hidden = model.encoder(x, hidden)
        logit = model.head_multi(hidden[-1]).item()
        prob = 1.0 / (1.0 + math.exp(-logit))
        probs.append(prob)
    return np.array(probs, dtype=np.float64)


def evaluate_fsm(teacher_labels, student_probs, close_masks, eef_xyz, variant):
    """Run FSM variant on one episode, return emit steps and violations."""
    T = len(student_probs)
    emits = []
    anchor_step = -1
    anchor_eef = np.zeros(3)
    close_onset = -1
    close_onset_eef = np.zeros(3)

    state = "IDLE"
    grasp_persist = 0
    emitted_this_event = False
    total_emits = 0
    event_id = 0

    violations = []
    pre_confirm_min_z = None

    use_close_mask = variant not in ("v5", "v5h", "v5c", "v5ch")
    hysteresis_exit = 0.35 if variant in ("v5h", "v5ch") else GRASP_THRESHOLD
    prev_detected = False

    for t in range(T):
        # Hysteresis: enter at GRASP_THRESHOLD, stay above hysteresis_exit
        raw_prob = student_probs[t]
        if prev_detected:
            detected = raw_prob > hysteresis_exit
        else:
            detected = raw_prob > GRASP_THRESHOLD
        prev_detected = detected

        cc = close_masks[t] if t < len(close_masks) else False
        eef = eef_xyz[t] if t < len(eef_xyz) else np.zeros(3)

        # ── FSM state machine ──
        if variant in ("v5c", "v5ch"):
            # close_onset gate (feature index 17, from 25D features)
            # close_onset stays True from first close_streak>=3 until open_streak>=3
            co = bool(close_masks[t])  # close_onset stored in close_masks slot for these variants
            enter_condition = co
            exit_condition = False  # reset handled differently below
        else:
            enter_condition = cc if use_close_mask else True
            exit_condition = (not cc) if use_close_mask else False

        if state == "IDLE" and enter_condition:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            grasp_persist = 0
            emitted_this_event = False
            close_onset = t
            close_onset_eef = eef.copy()
            pre_confirm_min_z = eef[2]

        if state == "CLOSE_CANDIDATE":
            pre_confirm_min_z = min(pre_confirm_min_z, eef[2]) if pre_confirm_min_z is not None else eef[2]
            if detected:
                grasp_persist += 1
                if grasp_persist == 1:
                    anchor_step = t
                    anchor_eef = eef.copy()
            else:
                grasp_persist = 0
            if grasp_persist >= GRASP_PERSISTENCE:
                anchor_step = t - GRASP_PERSISTENCE + 1
                anchor_eef = eef_xyz[anchor_step] if anchor_step < len(eef_xyz) else eef.copy()
                state = "CONFIRMED"

        if state in ("CONFIRMED", "ELIGIBLE", "EMITTED") and exit_condition:
            state = "RESET"

        if state == "CONFIRMED":
            eligible = False
            if variant in ("v0", "v5", "v5h"):
                # current / Student-only: first-positive anchor + vertical
                anchor_z = anchor_eef[2]
                if eef[2] - anchor_z >= TRANSPORT_VERT:
                    eligible = True
                if variant in ("v5", "v5h"):
                    # Student-only: also eligible immediately if no motion detected
                    # (transport evidence OR immediate, since motion is not required)
                    eligible = True
            elif variant == "v1":
                anchor_z = close_onset_eef[2]
                if eef[2] - anchor_z >= TRANSPORT_VERT:
                    eligible = True
            elif variant == "v2":
                anchor_xyz = close_onset_eef.copy()
                vert = abs(eef[2] - anchor_xyz[2])
                lat = np.linalg.norm(eef[:2] - anchor_xyz[:2])
                path_len = np.linalg.norm(eef - anchor_xyz)
                if vert >= TRANSPORT_VERT or lat >= TRANSPORT_LAT or path_len >= TRANSPORT_PATH:
                    eligible = True
            elif variant == "v3":
                eligible = True
            elif variant == "v4":
                confirmed_age = t - (anchor_step if anchor_step >= 0 else t)
                anchor_xyz = close_onset_eef.copy()
                vert = abs(eef[2] - anchor_xyz[2])
                lat = np.linalg.norm(eef[:2] - anchor_xyz[:2])
                path_len = np.linalg.norm(eef - anchor_xyz)
                transport = vert >= TRANSPORT_VERT or lat >= TRANSPORT_LAT or path_len >= TRANSPORT_PATH
                if transport or confirmed_age >= TIMEOUT_W:
                    eligible = True
            else:
                eligible = False

            if eligible:
                state = "ELIGIBLE"

        emit = False
        if state == "ELIGIBLE" and not emitted_this_event:
            if total_emits < 1:
                emitted_this_event = True
                total_emits += 1
                state = "EMITTED"
                emit = True

        if state == "RESET" and enter_condition:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            grasp_persist = 0
            emitted_this_event = False
            close_onset = t
            close_onset_eef = eef.copy()
            pre_confirm_min_z = eef[2]

        if emit:
            emits.append(t)
            # Check structural violations
            if teacher_labels and t < len(teacher_labels) and not teacher_labels[t]:
                violations.append("emit_outside_teacher_grasp_t={}".format(t))
            if total_emits > 1:
                violations.append("duplicate_emit_t={}".format(t))
            if anchor_step >= 0 and t < anchor_step:
                violations.append("pre_anchor_emit_t={}".format(t))

    return {
        "emits": emits,
        "n_emits": len(emits),
        "violations": violations,
        "events_seen": event_id,
    }


def compute_metrics(all_results):
    """Aggregate metrics across all episodes."""
    n_eps = len(all_results)
    n_teacher_events = sum(r["n_teacher_events"] for r in all_results)
    n_emits = sum(r["fsm_result"]["n_emits"] for r in all_results)
    n_violations = sum(len(r["fsm_result"]["violations"]) for r in all_results)
    eps_with_emit = sum(1 for r in all_results if r["fsm_result"]["n_emits"] > 0)

    # Emit inside Teacher: what fraction of emits are inside Teacher grasp intervals
    total_emits_inside = 0
    total_teacher_steps = 0
    for r in all_results:
        for e_step in r["fsm_result"]["emits"]:
            if e_step < len(r["teacher_labels"]) and r["teacher_labels"][e_step]:
                total_emits_inside += 1
        total_teacher_steps += sum(r["teacher_labels"])

    precision = total_emits_inside / max(n_emits, 1)
    opportunity_recall = n_emits / max(n_teacher_events, 1) if n_teacher_events > 0 else 0.0

    return {
        "n_episodes": n_eps,
        "n_teacher_events": n_teacher_events,
        "n_emits": n_emits,
        "eps_with_emit": eps_with_emit,
        "emit_precision": round(precision, 4),
        "opportunity_recall": round(opportunity_recall, 4),
        "n_violations": n_violations,
    }


def main():
    device = torch.device("cpu")
    print("Loading Student checkpoint...")
    model = load_student(device)
    print("Student loaded: {} params".format(sum(p.numel() for p in model.parameters())))

    # Load fold-0 validation identities (multi_object only)
    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]
    print("Fold-0 validation multi_object episodes: {}".format(len(val_ids)))

    # Limit to first 50 for speed
    val_ids = val_ids[:50]

    # Process each episode
    all_results = {v: [] for v in ["v0", "v1", "v2", "v3", "v4", "v5", "v5h", "v5c", "v5ch"]}

    for idx, identity in enumerate(val_ids):
        parts = identity.split("/")
        suite, task, state = parts

        # Load Teacher labels
        teacher_path = TEACHER_ROOT / suite / task / state / "physics_teacher_v21.jsonl"
        s1_path = S1_ROOT / suite / task / state / "student_input_records.jsonl"

        if not teacher_path.is_file() or not s1_path.is_file():
            continue

        teacher_recs = jsonl(teacher_path)
        s1_recs = jsonl(s1_path)
        T = min(len(teacher_recs), len(s1_recs))

        if T < 10:
            continue

        # Derive Teacher grasp_established
        teacher_labels = [derive_grasp_established(teacher_recs[t]) for t in range(T)]

        # Student features and probabilities
        features = [s1_recs[t]["features_25d"] for t in range(T)]
        with torch.no_grad():
            probs = run_student(model, features, device)

        # close_mask from raw action (feature index 0 = gripper_command)
        close_masks = [float(s1_recs[t].get("features_25d", [0]*25)[0]) <= 0.5 for t in range(T)]
        # close_onset from features (feature index 17)
        close_onsets = [bool(float(s1_recs[t].get("features_25d", [0]*25)[17])) for t in range(T)]

        # EEF positions from features (indices 3,4,5)
        eef_xyz = np.array([
            [float(s1_recs[t]["features_25d"][3]),
             float(s1_recs[t]["features_25d"][4]),
             float(s1_recs[t]["features_25d"][5])]
            for t in range(T)
        ], dtype=np.float64)

        # Count Teacher events (continuous grasp_established runs)
        teacher_events = 0
        in_event = False
        for tl in teacher_labels:
            if tl and not in_event:
                teacher_events += 1
                in_event = True
            elif not tl:
                in_event = False

        if idx < 3:
            print("  {}: T={} teacher_events={}".format(identity, T, teacher_events))

        # Run each FSM variant
        for variant in ["v0", "v1", "v2", "v3", "v4", "v5", "v5h", "v5c", "v5ch"]:
            masks = close_onsets if variant in ("v5c", "v5ch") else close_masks
            fsm_result = evaluate_fsm(teacher_labels, probs, masks, eef_xyz, variant)
            all_results[variant].append({
                "identity": identity,
                "T": T,
                "n_teacher_events": teacher_events,
                "fsm_result": fsm_result,
                "teacher_labels": teacher_labels,
            })

    # Report
    print("\n" + "=" * 70)
    print("FSM Variant Comparison (Fold-0 val, multi_object, n={})".format(
        len(all_results["v0"])))
    print("=" * 70)
    print("{:<6s} {:>6s} {:>6s} {:>8s} {:>12s} {:>12s} {:>6s}".format(
        "Variant", "Episodes", "Emits", "EpsEmit", "Precision", "OppRecall", "Viol"))
    print("-" * 70)
    for variant in ["v0", "v1", "v2", "v3", "v4", "v5", "v5h", "v5c", "v5ch"]:
        m = compute_metrics(all_results[variant])
        print("{:<6s} {:>6d} {:>6d} {:>8d} {:>12.4f} {:>12.4f} {:>6d}".format(
            variant, m["n_episodes"], m["n_emits"], m["eps_with_emit"],
            m["emit_precision"], m["opportunity_recall"], m["n_violations"]))
    print("-" * 70)

    # Per-variant violation details
    for variant in ["v0", "v1", "v2", "v3", "v4", "v5", "v5h", "v5c", "v5ch"]:
        violations = []
        for r in all_results[variant]:
            violations.extend(r["fsm_result"]["violations"])
        if violations:
            print("{} violations: {}".format(variant, violations[:5]))


if __name__ == "__main__":
    main()
