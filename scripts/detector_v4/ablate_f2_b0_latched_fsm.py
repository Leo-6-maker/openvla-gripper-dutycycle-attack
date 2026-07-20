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
    episode_stats = []

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
            cause = "EMIT"
        elif n_latched == 0:
            cause = "NO_CLOSE_LATCH"
        elif n_armed == 0:
            cause = "LATCHED_NO_STUDENT_CONFIRMATION"
        elif n_released > 0 and n_armed_survived == 0:
            cause = "LATCH_RELEASED_BEFORE_CONFIRMATION"
        elif n_armed_survived > 0 and n_vertical == 0:
            cause = "ARMED_NO_VERTICAL_PASS"
        else:
            cause = "UNCLASSIFIED"
        ep_causes[cause] += 1
        episode_stats.append({"identity": identity, "primary_cause": cause,
                              "n_latched": n_latched, "n_armed": n_armed,
                              "n_vertical": n_vertical, "n_emit": n_emit_events})

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

    # ── 4-part ledger + trace windows ──────────────────────────────────────
    trace_windows = []
    ledger = {"A_command_open_emits": [], "B_close_but_mask_negative": [],
              "C_qualified_positive": [], "D_armed_no_vertical": []}

    for d in emit_details:
        e_step = d["emit_step"]
        identity = d["identity"]
        # Find episode data
        parts = identity.split("/")
        ep_teacher = jsonl(TEACHER_ROOT / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl")
        ep_s1 = jsonl(S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl")
        T_ep = len(ep_s1)

        # Trace window: e_step-5 to e_step+5
        window = []
        for t in range(max(0, e_step - 5), min(T_ep, e_step + 6)):
            tr = ep_teacher[t]
            sr = ep_s1[t]
            f = sr["features_25d"]
            window.append({
                "step": t, "is_emit": t == e_step,
                "raw_gripper": float(f[0]),
                "raw_close": float(f[0]) <= 0.5,
                "gripper_qpos": float(f[1]),
                "opening_proxy": float(f[2]),
                "eef_z": float(f[5]),
                "close_streak": float(f[13]),
                "open_streak": float(f[14]),
                "close_onset": bool(float(f[16])),
                "qpos_delta_1": float(f[20]),
                "qpos_delta_3": float(f[21]),
                "opening_delta_3": float(f[22]),
                "opening_var_5": float(f[23]),
                "student_prob": float(probs[t]) if t < len(probs) else 0,
                "stable_grasp_score": float(tr.get("stable_grasp_score", -1)),
                "stable_grasp_dwell": int(tr.get("stable_grasp_dwell", -1)),
                "release_risk": float(tr.get("release_risk", -1)),
                "regrasp_risk": float(tr.get("regrasp_or_instability_risk", -1)),
                "candidate_close": bool(tr.get("candidate_close", False)),
            })
        trace_windows.append({"identity": identity, "emit_step": e_step, "window": window})

        # Classify into ledger
        entry = dict(d)
        entry["trace_window"] = window
        if d["emit_open_streak"] is not None and d["emit_open_streak"] > 0:
            ledger["A_command_open_emits"].append(entry)
        elif not d["in_positive_segment"] and d["close_mask_at_emit"]:
            ledger["B_close_but_mask_negative"].append(entry)
        elif d["in_positive_segment"]:
            ledger["C_qualified_positive"].append(entry)
        else:
            ledger["B_close_but_mask_negative"].append(entry)

    # ARMED-no-vertical episodes: collect first ARMED event per episode
    for ep_stat in episode_stats:
        if ep_stat.get("primary_cause") == "ARMED_NO_VERTICAL_PASS":
            ledger["D_armed_no_vertical"].append({"identity": ep_stat["identity"]})

    # Print ledger summary
    print("\n4-Part Emit Ledger:")
    for ledger_name in ["A_command_open_emits", "B_close_but_mask_negative",
                        "C_qualified_positive", "D_armed_no_vertical"]:
        entries = ledger[ledger_name]
        print("  {}: {} entries".format(ledger_name, len(entries)))
        if ledger_name != "D_armed_no_vertical":
            for e in entries[:2]:
                print("    {} step={} open_streak={} qpos={:.4f} opening={:.4f} pos={}".format(
                    e["identity"][-30:], e["emit_step"], e.get("emit_open_streak"),
                    e["gripper_qpos"], e["opening_proxy"], e["in_positive_segment"]))

    # Print detailed trace for first emit in each ledger
    for ledger_name in ["A_command_open_emits", "B_close_but_mask_negative",
                        "C_qualified_positive"]:
        entries = ledger[ledger_name]
        if entries:
            e = entries[0]
            print("\n  Trace: {} emit_step={} (ledger {})".format(e["identity"][-30:], e["emit_step"], ledger_name[0]))
            print("  {:>4s} {:>8s} {:>6s} {:>8s} {:>8s} {:>6s} {:>6s} {:>8s} {:>7s} {:>6s}".format(
                "step", "raw_cmd", "close", "qpos", "opening", "c_str", "o_str", "d_qpos3", "sg_score", "emit?"))
            for w in e["trace_window"]:
                marker = " <<<" if w["is_emit"] else ""
                print("  {:>4d} {:>8.4f} {:>6s} {:>8.4f} {:>8.4f} {:>6.1f} {:>6.1f} {:>8.4f} {:>7.3f}{:>6s}".format(
                    w["step"], w["raw_gripper"], "Y" if w["raw_close"] else "N",
                    w["gripper_qpos"], w["opening_proxy"],
                    w["close_streak"], w["open_streak"],
                    w["qpos_delta_3"], w["stable_grasp_score"], marker))

    # Write machine-readable traces
    traces_path = Path("/tmp/r10_f2_b0_emit_traces.jsonl")
    traces_path.write_text("\n".join(json.dumps(tw, sort_keys=True, default=str) for tw in trace_windows))
    print("\nTrace windows: {} ({} emits)".format(traces_path, len(trace_windows)))


if __name__ == "__main__":
    main()
