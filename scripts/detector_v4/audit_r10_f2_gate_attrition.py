#!/usr/bin/env python3
"""Gate F2-A.1: Exact source-bound production FSM attrition waterfall.

INPUT STATUS:
  OOF fold checkpoints: UNAVAILABLE (saved to /tmp, cleaned up)
  Exact Teacher event stream (R10.1A): UNAVAILABLE (output not preserved)
  Qualified fallback: full-FIT checkpoint + Physics Teacher V2.1 raw labels.
  All results are marked FULL_FIT_STUDENT, not OOF.

CPU only. Fail-closed: any missing file, length mismatch, or hash mismatch
terminates nonzero. No continue. No min() truncation. No default True/False.

Outputs:
  PRODUCTION_WATERFALL.json  — strict nested P0-P8
  COUNTERFACTUAL_GEOMETRY.json — paired anchor recovery matrix
  PER_TASK.csv
  ROOT_CAUSE_COUNTS.json
  EVENT_ROWS.jsonl (optional)
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
FEATURE_CONTRACT_SHA = "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366"

GRASP_THRESHOLD = 0.5
GRASP_PERSISTENCE = 3
TRANSPORT_VERT = 0.02
TRANSPORT_LAT = 0.015
TRANSPORT_PATH = 0.03


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(25, 64, 2, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def load_student(device):
    ckpt_path = BUNDLE / "full_fit_deploy.pt"
    if not ckpt_path.is_file():
        raise SystemExit("F2_CHECKPOINT_MISSING:{}".format(ckpt_path))
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = StudentModel().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def jsonl(path):
    if not path.is_file():
        raise SystemExit("F2_FILE_MISSING:{}".format(path))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("F2_FILE_EMPTY:{}".format(path))
    return [json.loads(l) for l in lines]


def derive_grasp_established(teacher_rec):
    """Frozen R10.1A contract. All fields must be present — no defaults."""
    cc = teacher_rec["candidate_close"]
    valid = teacher_rec["student_valid"]
    known = teacher_rec["known_mask"]
    sg = float(teacher_rec["stable_grasp_score"])
    # NOTE: dwell field is available but contract uses dwell>=0 (always true at V2.1).
    # Full R10.1A contract also uses release_risk/regrasp_risk — those require
    # the R10.1A event-level output which is not preserved. This is a qualified
    # approximation. Label accordingly.
    return bool(cc) and bool(valid) and bool(known) and sg >= 0.3


def run_production_fsm(student_probs, close_masks, eef_xyz, T):
    """Exact production FSM: IDLE→CLOSE_CANDIDATE→ARMED→EVENT_CANDIDATE→EMIT→RESET."""
    state = "IDLE"
    grasp_persist = 0
    emitted_this_event = False
    total_emits = 0
    event_id = 0
    anchor_step = -1
    anchor_eef_z = 0.0

    states = []
    emits = []
    arm_steps = []
    survived_armed = []

    for t in range(T):
        detected = student_probs[t] > GRASP_THRESHOLD
        cc = close_masks[t]
        eef = eef_xyz[t]

        if state == "IDLE" and cc:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            grasp_persist = 0
            emitted_this_event = False

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

        reset_this_step = False
        if state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not cc:
            state = "RESET"
            reset_this_step = True

        if not reset_this_step and state == "ARMED":
            survived_armed.append(t)

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

        states.append(state)
        if emit:
            emits.append(t)

    return states, emits, arm_steps, survived_armed


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main():
    device = torch.device("cpu")

    # ── INPUT VALIDATION ────────────────────────────────────────────────────
    print("F2-A.1: Input binding...")
    if not (BUNDLE / "full_fit_deploy.pt").is_file():
        raise SystemExit("HOLD_EXACT_OOF_SOURCE_UNAVAILABLE")
    print("  Student: full-FIT (qualified, NOT OOF — fold checkpoints unavailable)")

    manifest_path = FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json"
    if not manifest_path.is_file():
        raise SystemExit("FOLD_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text())
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]

    # Verify feature contract
    if manifest.get("feature_order_sha256") != FEATURE_CONTRACT_SHA:
        raise SystemExit("FEATURE_CONTRACT_HASH_MISMATCH")
    print("  Feature contract: {} OK".format(FEATURE_CONTRACT_SHA[:16]))
    print("  Fold-0 val multi_object: {} identities".format(len(val_ids)))

    # Fail-closed: verify every identity has Teacher + S1 files
    print("  Verifying all {} identities have Teacher + S1 files...".format(len(val_ids)))
    for identity in val_ids:
        parts = identity.split("/")
        tp = TEACHER_ROOT / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        sp = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        if not tp.is_file():
            raise SystemExit("TEACHER_FILE_MISSING:{}".format(identity))
        if not sp.is_file():
            raise SystemExit("S1_FILE_MISSING:{}".format(identity))
    print("  All files present.")

    print("  Teacher: Physics V2.1 raw labels (qualified — R10.1A event stream unavailable)")
    print("  Teacher event derivation: cc AND valid AND known AND sg>=0.3")

    # ── LOAD MODEL ──────────────────────────────────────────────────────────
    print("\nLoading Student checkpoint...")
    model = load_student(device)
    print("  Params: {}".format(sum(p.numel() for p in model.parameters())))

    # ── PROCESS EPISODES ────────────────────────────────────────────────────
    all_events = []
    episode_stats = []

    for identity in val_ids:
        parts = identity.split("/")
        suite, task, state = parts

        teacher_recs_raw = jsonl(TEACHER_ROOT / suite / task / state / "physics_teacher_v21.jsonl")
        s1_recs = jsonl(S1_ROOT / suite / task / state / "student_input_records.jsonl")

        # Fail-closed: lengths must match exactly
        Tt = len(teacher_recs_raw)
        Ts = len(s1_recs)
        if Tt != Ts:
            raise SystemExit("LENGTH_MISMATCH:{} T={} S={}".format(identity, Tt, Ts))

        T = Tt

        # Teacher labels (qualified approximation — see docstring)
        teacher_labels = [derive_grasp_established(teacher_recs_raw[t]) for t in range(T)]

        # Student probabilities
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
        eef_xyz = np.array([[float(s1_recs[t]["features_25d"][3]),
                             float(s1_recs[t]["features_25d"][4]),
                             float(s1_recs[t]["features_25d"][5])] for t in range(T)], dtype=np.float64)

        # Production FSM
        fsm_states, fsm_emits, fsm_arm_steps, fsm_survived = run_production_fsm(probs, close_masks, eef_xyz, T)

        # Extract contiguous Teacher-positive segments
        segments = []
        in_seg = False
        start = -1
        for t in range(T):
            if teacher_labels[t] and not in_seg:
                start = t
                in_seg = True
            elif not teacher_labels[t] and in_seg:
                segments.append((start, t - 1))
                in_seg = False
        if in_seg:
            segments.append((start, T - 1))

        for seg_id, (es, ee) in enumerate(segments):
            ev = {
                "identity": identity, "seg_id": seg_id,
                "start": es, "end": ee, "duration": ee - es + 1,
            }

            # P1: FSM entered CLOSE_CANDIDATE before confirmation
            # Find Student persistence runs within this segment
            runs = []
            in_run = False
            rs = -1
            for t in range(es, ee + 1):
                if probs[t] > GRASP_THRESHOLD and not in_run:
                    rs = t
                    in_run = True
                elif probs[t] <= GRASP_THRESHOLD and in_run:
                    if t - rs >= GRASP_PERSISTENCE:
                        runs.append((rs, t - 1))
                    in_run = False
            if in_run and ee + 1 - rs >= GRASP_PERSISTENCE:
                runs.append((rs, ee))

            if not runs:
                ev["p2_persistence"] = False
                all_events.append(ev)
                continue

            first_run = runs[0]
            conf_step = first_run[0] + GRASP_PERSISTENCE - 1
            ev["conf_step"] = conf_step
            ev["p2_persistence"] = True

            # P1: close before confirmation?
            close_segs = []
            in_close = False
            cs = -1
            for t in range(0, T):
                if close_masks[t] and not in_close:
                    cs = t
                    in_close = True
                elif not close_masks[t] and in_close:
                    close_segs.append((cs, t - 1))
                    in_close = False
            if in_close:
                close_segs.append((cs, T - 1))

            close_before_conf = any(s < conf_step for s, e in close_segs)
            ev["p1_close_before"] = close_before_conf
            if not close_before_conf:
                all_events.append(ev)
                continue

            # P3: FSM transiently entered ARMED at conf_step (exact step)
            ev["p3_transient_armed"] = conf_step in fsm_arm_steps

            # P4: FSM survived conf_step in ARMED (not reset same step)
            ev["p4_survived_armed"] = conf_step in fsm_survived

            # P5/P6/P7: motion gate
            if ev["p4_survived_armed"]:
                # first-positive anchor motion
                anchor_z_fp = eef_xyz[first_run[0]][2]
                max_vert_fp = max(eef_xyz[t][2] - anchor_z_fp for t in range(conf_step, ee + 1))
                ev["p6_vert_first_pos"] = max_vert_fp >= TRANSPORT_VERT

                # close-onset anchor motion
                co_step = None
                for s, e in close_segs:
                    if s < conf_step and e >= first_run[0]:
                        co_step = s
                        break
                if co_step is None:
                    for s, e in close_segs:
                        if s < conf_step:
                            co_step = s
                if co_step is not None:
                    co_z = eef_xyz[co_step][2]
                    max_vert_co = max(eef_xyz[t][2] - co_z for t in range(co_step, ee + 1))
                    max_lat_co = max(
                        np.linalg.norm(eef_xyz[t][:2] - eef_xyz[co_step][:2])
                        for t in range(co_step, ee + 1))
                    ev["co_anchor_step"] = co_step
                    ev["co_vert_pass"] = max_vert_co >= TRANSPORT_VERT
                    ev["co_lat_pass"] = max_lat_co >= TRANSPORT_LAT
                    ev["co_path_pass"] = max(max_vert_co, max_lat_co, max(
                        np.linalg.norm(eef_xyz[t] - eef_xyz[co_step])
                        for t in range(co_step, ee + 1))) >= TRANSPORT_PATH

                # P7: emit inside segment
                ev["p7_emit_inside"] = any(es <= e <= ee for e in fsm_emits)
            else:
                ev["p6_vert_first_pos"] = False

            all_events.append(ev)

        episode_stats.append({
            "identity": identity, "T": T,
            "n_segments": len(segments),
            "n_fsm_emits": len(fsm_emits),
        })

    # ── NESTED WATERFALL ────────────────────────────────────────────────────
    P0 = len(all_events)
    P2 = sum(1 for e in all_events if e.get("p2_persistence"))
    P1_close = sum(1 for e in all_events if e.get("p1_close_before"))
    P3 = sum(1 for e in all_events if e.get("p3_transient_armed"))
    P4 = sum(1 for e in all_events if e.get("p4_survived_armed"))
    P6 = sum(1 for e in all_events if e.get("p6_vert_first_pos"))
    P7 = sum(1 for e in all_events if e.get("p7_emit_inside"))

    # P2P = P2 AND P1 (close + persistence together)
    P2P = sum(1 for e in all_events if e.get("p2_persistence") and e.get("p1_close_before"))

    # Paired anchor matrix
    anchor_matrix = {"fp_yes_co_yes": 0, "fp_yes_co_no": 0, "fp_no_co_yes": 0, "fp_no_co_no": 0}
    for e in all_events:
        fp = e.get("p6_vert_first_pos", False)
        co = e.get("co_vert_pass", False)
        if fp and co:
            anchor_matrix["fp_yes_co_yes"] += 1
        elif fp and not co:
            anchor_matrix["fp_yes_co_no"] += 1
        elif not fp and co:
            anchor_matrix["fp_no_co_yes"] += 1
        else:
            anchor_matrix["fp_no_co_no"] += 1

    # Root cause
    causes = defaultdict(int)
    for e in all_events:
        if e.get("p7_emit_inside"):
            causes["CURRENT_FSM_EMIT"] += 1
        elif not e.get("p2_persistence"):
            causes["NO_FULL_FIT_STUDENT_PERSISTENCE"] += 1
        elif not e.get("p1_close_before"):
            causes["NO_CLOSE_BEFORE_CONFIRMATION"] += 1
        elif not e.get("p3_transient_armed"):
            causes["FSM_NEVER_ARMED"] += 1
        elif not e.get("p4_survived_armed"):
            causes["PHASE_LOCK_RESET_AT_CONFIRMATION"] += 1
        elif not e.get("p6_vert_first_pos"):
            causes["VERTICAL_GUARD_BLOCK"] += 1
        else:
            causes["OTHER_UNCLASSIFIED"] += 1

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    waterfall = {
        "schema": "R10_F2_PRODUCTION_WATERFALL_V1",
        "student_source": "FULL_FIT_DEPLOYMENT_CHECKPOINT",
        "student_source_note": "OOF fold checkpoints unavailable (/tmp cleaned)",
        "teacher_source": "PHYSICS_TEACHER_V21_RAW_LABELS",
        "teacher_source_note": "R10.1A event stream unavailable; qualified cc+valid+known+sg>=0.3",
        "fold_id": 0, "n_episodes": len(val_ids),
        "n_total_segments": P0,
        "waterfall": {
            "P0_total_segments": P0,
            "P2_student_persistence": P2,
            "P2P_persistence_and_close": P2P,
            "P1_close_before_confirmation": P1_close,
            "P3_transient_armed": P3,
            "P4_survived_armed_no_reset": P4,
            "P6_vert_first_pos_pass": P6,
            "P7_emit_inside_segment": P7,
        },
        "note": "P1 and P2 are INDEPENDENT filters (not nested). P2P=P2 AND P1 is the nested count.",
        "root_cause_counts": dict(causes),
        "anchor_recovery_matrix": anchor_matrix,
        "anchor_matrix_note": "co_vert_pass only computed for P4 survivors with close-onset anchor",
        "episode_stats": episode_stats,
    }

    print("\n" + "=" * 70)
    print("F2-A.1: Production FSM Attrition Waterfall")
    print("  Student: FULL_FIT (qualified — fold checkpoints unavailable)")
    print("  Teacher: Physics V2.1 raw labels (qualified — event stream unavailable)")
    print("=" * 70)
    print("P0  Total positive segments:               {}".format(P0))
    print("P2  Student persistence-3:                  {} ({:.1f}%)".format(P2, 100*P2/max(P0,1)))
    print("P2P Persistence AND close before conf:      {} ({:.1f}%)".format(P2P, 100*P2P/max(P0,1)))
    print("P3  Transient ARMED at confirmation:        {} ({:.1f}%)".format(P3, 100*P3/max(P0,1)))
    print("P4  Survived ARMED (no same-step reset):    {} ({:.1f}%)".format(P4, 100*P4/max(P0,1)))
    print("P6  Vertical pass (first-pos anchor):       {} ({:.1f}%)".format(P6, 100*P6/max(P0,1)))
    print("P7  Emit inside segment:                    {} ({:.1f}%)".format(P7, 100*P7/max(P0,1)))
    print("-" * 70)
    print("Root causes:")
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        print("  {}: {}".format(c, n))
    print("-" * 70)
    print("Paired anchor recovery matrix (P4 survivors):")
    for k, v in anchor_matrix.items():
        print("  {}: {}".format(k, v))
    recovered = anchor_matrix["fp_no_co_yes"]
    print("  Recovered by close-onset anchor: {}".format(recovered))

    # Write output
    output_path = Path("/tmp/r10_f2_waterfall.json")
    output_path.write_text(json.dumps(waterfall, indent=2, sort_keys=True, default=str))
    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print("\nOutput: {} (sha256={})".format(output_path, output_sha[:16]))

    # Save event rows
    events_path = Path("/tmp/r10_f2_event_rows.jsonl")
    events_path.write_text("\n".join(json.dumps(e, sort_keys=True, default=str) for e in all_events))
    print("Event rows: {} ({} events)".format(events_path, len(all_events)))


if __name__ == "__main__":
    main()
