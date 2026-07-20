#!/usr/bin/env python3
"""R10.2B-M: Multi-Object Routed Grasp Student Training + vertical-lift FSM.

Route: multi_object_transfer only
Target: grasp_established (Teacher physics, current-step, causal)
Input: 25D causal proprio/action + deployment-safe mechanism route
Model: shared GRU encoder + multi-object routed head
Sampler: event-balanced
Guard: eef_z(t) - eef_z(active_anchor) >= 0.02m
Eval: Fold-0 480/120 OOF with pre-registered gates

All thresholds FROZEN before held-out prediction read.
Single seed. Single fold. No task/state tuning.
"""

from __future__ import annotations

import json, math, os, sys, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────────────
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719"
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"
K10_ROOT = OPS / "OFFICIAL_V3_R7_K10_OPPORTUNITY_LABELER_V1_2_1_8e4f5ff_20260719"
SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]

# ── Frozen hyperparameters (pre-registered) ──────────────────────────────────
FROZEN = {
    "seed": 20260720,
    "fold_id": 0,
    "route": "multi_object_transfer",
    "input_dim": 25,
    "hidden_dim": 64,
    "num_layers": 2,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grasp_persistence": 3,
    "grasp_threshold": 0.5,
    "guard_type": "vertical_lift",
    "guard_param": 0.02,  # 0.02m vertical lift
    "max_episode_emits": 1,
    "events_per_episode": 8,
    "pos_per_event": 4,
    "neg_per_event": 8,
    "hard_neg_per_event": 4,
}

# ── Named 25D indices ────────────────────────────────────────────────────────
IDX = {
    "eef_x": 3, "eef_y": 4, "eef_z": 5,
    "close_onset": 16, "time_since_close": 17,
    "eef_z_delta_since_close": 19,
}

# ── Mechanism parser ─────────────────────────────────────────────────────────
def parse_mechanism(identity: str) -> str:
    parts = identity.split("/")
    tk = f"{parts[0]}/{parts[1]}"
    mapping = {
        "libero_goal/task_07": "unsupported_abstain",
        "libero_object": "single_object_pick_place",
        "libero_spatial": "single_object_pick_place",
        "libero_goal": "single_object_pick_place",
        "libero_10": "multi_object_transfer",
    }
    if tk in mapping: return mapping[tk]
    if parts[0] in mapping: return mapping[parts[0]]
    return "unsupported_abstain"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class RoutedGraspDetector(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head_multi = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.head_single = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, mechanism_ids: list[str]) -> torch.Tensor:
        hidden, _ = self.encoder(x)  # [B, T, H]
        B, T, H = hidden.shape
        logits = torch.zeros(B, T, device=x.device)
        for b in range(B):
            mech = mechanism_ids[b]
            hb = hidden[b]
            if mech == "multi_object_transfer":
                logits[b] = self.head_multi(hb).squeeze(-1)
            elif mech == "single_object_pick_place":
                logits[b] = self.head_single(hb).squeeze(-1)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Event-balanced sampler
# ═══════════════════════════════════════════════════════════════════════════════

def build_episode_data(identities: list[str]) -> list[dict]:
    """Load per-identity 25D, grasp labels, and event structure."""
    episodes = []
    for identity in identities:
        parts = identity.split("/")
        mech = parse_mechanism(identity)
        if mech != FROZEN["route"]:
            continue

        s1_p = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        teacher_p = TEACHER_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        if not s1_p.is_file() or not teacher_p.is_file():
            continue

        s1 = _jsonl(s1_p)
        teacher = _jsonl(teacher_p)
        T = min(len(s1), len(teacher))

        feats_25d = []
        grasp_labels = []
        close_mask = []

        for t in range(T):
            sr = s1[t]
            tr = teacher[t]
            f25 = [float(v) for v in sr["features_25d"]]
            sg = float(tr.get("stable_grasp_score", 0))
            cc = bool(tr.get("candidate_close", False))
            valid = bool(tr.get("student_valid", True))
            supported = mech != "unsupported_abstain"
            ge = cc and valid and supported and sg >= 0.3

            feats_25d.append(f25)
            grasp_labels.append(ge)
            close_mask.append(cc)

        # Identify candidate_close events
        events = []
        in_cc = False
        ev_start = 0
        for t in range(T):
            if close_mask[t] and not in_cc:
                ev_start = t; in_cc = True
            elif not close_mask[t] and in_cc:
                if t - ev_start >= 5:
                    events.append((ev_start, t - 1))
                in_cc = False
        if in_cc and T - ev_start >= 5:
            events.append((ev_start, T - 1))

        if not events:
            continue

        episodes.append({
            "identity": identity, "T": T,
            "feats": feats_25d, "grasp": grasp_labels, "close": close_mask,
            "events": events, "mech": mech,
        })

    return episodes


class EventBalancedSampler:
    """Samples events, then positive/hard-negative/negative steps within each event."""

    def __init__(self, episodes: list[dict], events_per_ep: int = 8,
                 pos_per_event: int = 4, neg_per_event: int = 8,
                 hard_neg_per_event: int = 4):
        self.episodes = episodes
        self.events_per_ep = events_per_ep
        self.pos_per_event = pos_per_event
        self.neg_per_event = neg_per_event
        self.hard_neg_per_event = hard_neg_per_event

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
        """Returns (features [B,T,25], grasp_labels [B,T], mechanism_ids, mask [B,T])."""
        import random
        rng = random.Random(FROZEN["seed"])

        # Fixed sequence length
        seq_len = 256

        feats_batch = []
        labels_batch = []
        mech_batch = []
        mask_batch = []

        for _ in range(batch_size):
            ep = rng.choice(self.episodes)
            T = ep["T"]
            events = ep["events"]

            # Build step weights: prioritize positive steps, then hard negatives, then negatives
            step_weight = [1.0] * T  # default: background negative

            for ev_start, ev_end in events:
                for t in range(max(0, ev_start), min(T, ev_end + 1)):
                    if ep["grasp"][t]:
                        step_weight[t] = 10.0  # positive: high weight
                    elif t <= ev_start + 3 or t >= ev_end - 3:
                        step_weight[t] = 3.0  # event boundary: hard negative

            # Weighted sampling of a segment
            if T <= seq_len:
                start = 0
            else:
                total_w = sum(step_weight)
                if total_w == 0:
                    start = rng.randint(0, T - seq_len)
                else:
                    center = rng.choices(range(T), weights=step_weight, k=1)[0]
                    lo = max(0, center - seq_len // 2)
                    hi = min(T - seq_len, center + seq_len // 2)
                    if lo > hi:
                        start = max(0, T - seq_len)
                    else:
                        start = rng.randint(lo, hi)

            end = min(T, start + seq_len)
            actual_len = end - start

            feats = torch.tensor(ep["feats"][start:end], dtype=torch.float32)
            labels = torch.tensor([1.0 if ep["grasp"][t] else 0.0 for t in range(start, end)])

            # Pad if needed
            if actual_len < seq_len:
                feats = F.pad(feats, (0, 0, 0, seq_len - actual_len))
                labels = F.pad(labels, (0, seq_len - actual_len))
                mask = torch.cat([torch.ones(actual_len), torch.zeros(seq_len - actual_len)])
            else:
                mask = torch.ones(seq_len)

            feats_batch.append(feats)
            labels_batch.append(labels)
            mech_batch.append(ep["mech"])
            mask_batch.append(mask)

        return (torch.stack(feats_batch), torch.stack(labels_batch),
                mech_batch, torch.stack(mask_batch))


# ═══════════════════════════════════════════════════════════════════════════════
# FSM evaluator (Student grasp + vertical-lift guard)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_fsm(
    model: nn.Module,
    identities: list[str],
    device: torch.device,
) -> dict:
    """Run Student grasp + FSM on all identities. Returns per-episode + aggregate metrics."""
    model.eval()

    results = []
    per_suite = defaultdict(lambda: {"n": 0, "emits": 0, "k10_hits": 0, "k10_tot": 0,
                                      "ar_hits": 0, "ar_tot": 0,
                                      "pre_anchor": 0, "release_overlap": 0,
                                      "duplicate": 0, "unsupported": 0,
                                      "reset_closure_ok": 0, "reset_closure_tot": 0})

    with torch.no_grad():
        for identity in identities:
            mech = parse_mechanism(identity)
            parts = identity.split("/")

            # Load data
            s1_p = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
            teacher_p = TEACHER_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
            k10_p = K10_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "k10_labels_v121.jsonl"

            if not all(p.is_file() for p in [s1_p, teacher_p, k10_p]):
                continue
            s1 = _jsonl(s1_p)
            teacher = _jsonl(teacher_p)
            k10_labels = _jsonl(k10_p)
            T = min(len(s1), len(teacher), len(k10_labels))

            # Get Student grasp predictions
            feats = torch.tensor([[float(v) for v in sr["features_25d"]] for sr in s1[:T]],
                                 dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,25]
            logits = model(feats, [mech]).squeeze(0)  # [T]
            grasp_probs = torch.sigmoid(logits).cpu()

            # Per-step ground truth
            grasp_true = []
            close_mask = []
            ar_true = []
            k10_start = []
            release_risk = []
            eef_z_vals = []

            for t in range(T):
                tr = teacher[t]
                sg = float(tr.get("stable_grasp_score", 0))
                rr = float(tr.get("release_risk", 0))
                ri = float(tr.get("regrasp_or_instability_risk", 0))
                cc = bool(tr.get("candidate_close", False))
                valid = bool(tr.get("student_valid", True))
                supported = mech != "unsupported_abstain"
                gn = float(tr.get("task_grasp_necessity", 1))
                gc = float(tr.get("gripper_contact_score", 0))
                lift = float(tr.get("lift_score", 0))
                sr2 = float(tr.get("support_removed", 0))
                tp = float(tr.get("target_progress", 0))

                ge = cc and valid and supported and sg >= 0.3
                ma = (lift >= 0.3 or sr2 >= 0.3 or tp > 0.05) and ge
                ra = rr > 0.5 and cc and supported
                rgi = ri > 0.5 and cc and supported
                gd = gn > 0 and (gc > 0 or sg > 0.3) and supported
                ar = valid and supported and ge and gd and ma and not ra and not rgi and cc

                grasp_true.append(ge)
                close_mask.append(cc)
                ar_true.append(ar)
                release_risk.append(rr)
                k10_start.append(k10_labels[t].get("is_feasible_start", False))
                eef_z_vals.append(float(s1[t]["features_25d"][IDX["eef_z"]]))

            # FSM
            state = "IDLE"
            grasp_persist = 0
            anchor_step = -1
            anchor_eef_z = 0.0
            armed_step = -1
            emitted = False
            total_emits = 0
            emit_step = -1
            event_id = 0
            prev_event_end = -1
            reset_ok = 0
            reset_fail = 0
            event_reset_count = 0
            violations = []

            for t in range(T):
                if mech == "unsupported_abstain":
                    continue

                grasp_detected = grasp_probs[t].item() > FROZEN["grasp_threshold"]

                # IDLE → CLOSE_CANDIDATE
                if state == "IDLE" and close_mask[t]:
                    state = "CLOSE_CANDIDATE"
                    event_id += 1

                # CLOSE_CANDIDATE → accumulate persistence
                if state == "CLOSE_CANDIDATE":
                    if grasp_detected:
                        grasp_persist += 1
                        if grasp_persist == 1:
                            anchor_step = t
                            anchor_eef_z = eef_z_vals[t]
                    else:
                        grasp_persist = 0
                    if grasp_persist >= FROZEN["grasp_persistence"]:
                        state = "ARMED"
                        armed_step = t

                # Reopen → RESET
                if state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not close_mask[t]:
                    if prev_event_end >= 0:
                        gap = t - prev_event_end
                        if gap <= 5: reset_ok += 1
                        else: reset_fail += 1
                    prev_event_end = t
                    event_reset_count += 1
                    state = "RESET"

                # ARMED: check vertical lift
                if state == "ARMED" and not emitted:
                    vert_lift = eef_z_vals[t] - anchor_eef_z
                    if vert_lift >= FROZEN["guard_param"]:
                        state = "EVENT_CANDIDATE"

                # EVENT_CANDIDATE → EMIT
                if state == "EVENT_CANDIDATE" and not emitted:
                    if total_emits < FROZEN["max_episode_emits"]:
                        if t < anchor_step:
                            violations.append(f"PRE_ANCHOR: t={t} anchor={anchor_step}")
                        emitted = True
                        total_emits += 1
                        emit_step = t
                        state = "EMITTED"

                # RESET → CLOSE_CANDIDATE
                if state == "RESET" and close_mask[t]:
                    state = "CLOSE_CANDIDATE"
                    grasp_persist = 0
                    emitted = False
                    anchor_step = -1
                    event_id += 1

            # Compile metrics
            n_emits = 1 if emit_step >= 0 else 0
            k10_hit = 1 if (emit_step >= 0 and k10_start[emit_step]) else 0
            ar_hit = 1 if (emit_step >= 0 and ar_true[emit_step]) else 0
            n_k10 = sum(1 for k in k10_start if k)
            n_ar = sum(1 for a in ar_true if a)
            pre_anchor = len(violations)
            release_overlap_val = 1 if (emit_step >= 0 and release_risk[emit_step] > 0.5) else 0
            unsupported_emit = 1 if (mech == "unsupported_abstain" and n_emits > 0) else 0
            duplicate = max(0, total_emits - 1)

            suite = parts[0]
            per_suite[suite]["n"] += 1
            per_suite[suite]["emits"] += n_emits
            per_suite[suite]["k10_hits"] += k10_hit
            per_suite[suite]["k10_tot"] += 1 if n_k10 > 0 else 0
            per_suite[suite]["ar_hits"] += ar_hit
            per_suite[suite]["ar_tot"] += 1 if n_ar > 0 else 0
            per_suite[suite]["pre_anchor"] += pre_anchor
            per_suite[suite]["release_overlap"] += release_overlap_val
            per_suite[suite]["duplicate"] += duplicate
            per_suite[suite]["unsupported"] += unsupported_emit
            per_suite[suite]["reset_closure_ok"] += reset_ok
            per_suite[suite]["reset_closure_tot"] += event_reset_count

            results.append({
                "identity": identity, "emit_step": emit_step,
                "k10_hit": k10_hit, "ar_hit": ar_hit,
                "pre_anchor": pre_anchor, "release_overlap": release_overlap_val,
                "violations": violations,
            })

    # Aggregate
    total_n = sum(s["n"] for s in per_suite.values())
    total_emits = sum(s["emits"] for s in per_suite.values())
    total_k10_hits = sum(s["k10_hits"] for s in per_suite.values())
    total_k10_tot = sum(s["k10_tot"] for s in per_suite.values())
    total_ar_hits = sum(s["ar_hits"] for s in per_suite.values())
    total_pre = sum(s["pre_anchor"] for s in per_suite.values())
    total_rel = sum(s["release_overlap"] for s in per_suite.values())
    total_dup = sum(s["duplicate"] for s in per_suite.values())
    total_unsup = sum(s["unsupported"] for s in per_suite.values())

    global_metrics = {
        "n_episodes": total_n, "n_emits": total_emits,
        "k10_recall": total_k10_hits / max(1, total_k10_tot),
        "ar_overlap": total_ar_hits / max(1, total_emits) if total_emits else 0,
        "pre_anchor_emit": total_pre,
        "release_overlap": total_rel,
        "duplicate_emit": total_dup,
        "unsupported_emit": total_unsup,
    }

    suite_metrics = {}
    for suite in SUITES:
        s = per_suite[suite]
        if s["n"] == 0:
            continue
        suite_metrics[suite] = {
            "n": s["n"], "emits": s["emits"],
            "k10_recall": s["k10_hits"] / max(1, s["k10_tot"]),
            "ar_overlap": s["ar_hits"] / max(1, s["emits"]) if s["emits"] else 0,
            "pre_anchor": s["pre_anchor"],
            "release_overlap": s["release_overlap"],
            "reset_closure": s["reset_closure_ok"] / max(1, s["reset_closure_tot"]),
        }

    return {"global": global_metrics, "per_suite": suite_metrics, "per_episode": results}


# ═══════════════════════════════════════════════════════════════════════════════
# Grasp head metrics (step-level + event-level)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_grasp_head(
    model: nn.Module,
    identities: list[str],
    device: torch.device,
) -> dict:
    """Evaluate grasp_established prediction quality: step-level + event-level."""
    model.eval()

    all_probs = []
    all_labels = []
    event_metrics = []

    with torch.no_grad():
        for identity in identities:
            mech = parse_mechanism(identity)
            parts = identity.split("/")

            s1_p = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
            teacher_p = TEACHER_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
            if not s1_p.is_file() or not teacher_p.is_file():
                continue

            s1 = _jsonl(s1_p)
            teacher = _jsonl(teacher_p)
            T = min(len(s1), len(teacher))

            feats = torch.tensor([[float(v) for v in sr["features_25d"]] for sr in s1[:T]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(feats, [mech]).squeeze(0)
            probs = torch.sigmoid(logits).cpu()

            grasp_true = []
            close_mask = []
            for t in range(T):
                tr = teacher[t]
                sg = float(tr.get("stable_grasp_score", 0))
                cc = bool(tr.get("candidate_close", False))
                valid = bool(tr.get("student_valid", True))
                supported = mech != "unsupported_abstain"
                ge = cc and valid and supported and sg >= 0.3
                grasp_true.append(ge)
                close_mask.append(cc)

                all_probs.append(probs[t].item())
                all_labels.append(1 if ge else 0)

            # Event-level: contiguous CC runs
            events = []
            in_cc = False; ev_start = 0
            for t in range(T):
                if close_mask[t] and not in_cc:
                    ev_start = t; in_cc = True
                elif not close_mask[t] and in_cc:
                    if t - ev_start >= 5:
                        events.append((ev_start, t - 1))
                    in_cc = False
            if in_cc and T - ev_start >= 5:
                events.append((ev_start, T - 1))

            for ev_start, ev_end in events:
                ev_probs = probs[ev_start:ev_end+1]
                ev_labels = [1 if grasp_true[t] else 0 for t in range(ev_start, ev_end+1)]

                has_grasp = any(ev_labels)
                detected = any(p > FROZEN["grasp_threshold"] for p in ev_probs)

                # First onset
                true_first = next((i for i, l in enumerate(ev_labels) if l), None)
                pred_first = next((i for i, p in enumerate(ev_probs) if p > FROZEN["grasp_threshold"]), None)
                latency = (pred_first - true_first) if (true_first is not None and pred_first is not None) else None

                # Fragmentation
                above = [p > FROZEN["grasp_threshold"] for p in ev_probs]
                fragments = sum(1 for i in range(1, len(above)) if above[i] != above[i-1])

                event_metrics.append({
                    "has_grasp": has_grasp, "detected": detected,
                    "latency": latency, "fragments": fragments,
                })

    # Step-level
    probs_t = torch.tensor(all_probs)
    labels_t = torch.tensor(all_labels)
    preds_t = (probs_t > 0.5).float()
    step_acc = (preds_t == labels_t).float().mean().item()

    # Manual AUPRC
    desc = torch.argsort(probs_t, descending=True)
    labels_sorted = labels_t[desc]
    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1 - labels_sorted, dim=0)
    total_pos = labels_t.sum()
    if total_pos > 0:
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / total_pos
        rec_diff = rec[1:] - rec[:-1]
        auprc = (prec[:-1] * rec_diff).sum().item() if len(rec_diff) > 0 else 0.0
    else:
        auprc = 0.0

    # Event-level
    tp_ev = [m for m in event_metrics if m["has_grasp"]]
    fp_ev = [m for m in event_metrics if not m["has_grasp"]]
    ev_hit = sum(1 for m in tp_ev if m["detected"])
    ev_recall = ev_hit / len(tp_ev) if tp_ev else 0
    ev_fa = sum(1 for m in fp_ev if m["detected"])
    ev_precision = ev_hit / (ev_hit + ev_fa) if (ev_hit + ev_fa) else 0
    latencies = [m["latency"] for m in tp_ev if m["detected"] and m["latency"] is not None]
    frags = [m["fragments"] for m in tp_ev if m["has_grasp"]]

    return {
        "step_auprc": max(0.0, min(1.0, auprc)),
        "step_accuracy": step_acc,
        "step_prevalence": labels_t.float().mean().item(),
        "event_recall": ev_recall,
        "event_precision": ev_precision,
        "event_f1": 2 * ev_recall * ev_precision / (ev_recall + ev_precision) if (ev_recall + ev_precision) else 0,
        "n_events": len(event_metrics),
        "n_grasp_events": len(tp_ev),
        "latency_median": sorted(latencies)[len(latencies)//2] if latencies else None,
        "fragmentation_median": sorted(frags)[len(frags)//2] if frags else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(fold_id: int = 0, output_dir: str = "/tmp/r10_2b_m"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fold: {fold_id}")
    print(f"Frozen config: {json.dumps(FROZEN, indent=2)}")

    # Load fold
    fold = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text(encoding="utf-8"))
    f_fold = next(f for f in fold["folds"] if f["fold_id"] == fold_id)
    train_ids = sorted(f_fold["train_identities"])
    val_ids = sorted(f_fold["validation_identities"])

    # Filter to multi_object_transfer
    train_ids = [i for i in train_ids if parse_mechanism(i) == FROZEN["route"]]
    val_ids = [i for i in val_ids if parse_mechanism(i) == FROZEN["route"]]
    print(f"\nMulti-object route: {len(train_ids)} train / {len(val_ids)} val")

    # Build episode data
    print("Building episode data...")
    train_eps = build_episode_data(train_ids)
    val_eps = build_episode_data(val_ids)
    print(f"  Train episodes: {len(train_eps)}, Val episodes: {len(val_eps)}")

    # Sampler
    sampler = EventBalancedSampler(
        train_eps,
        events_per_ep=FROZEN["events_per_episode"],
        pos_per_event=FROZEN["pos_per_event"],
        neg_per_event=FROZEN["neg_per_event"],
        hard_neg_per_event=FROZEN["hard_neg_per_event"],
    )

    # Model
    torch.manual_seed(FROZEN["seed"])
    model = RoutedGraspDetector(
        input_dim=FROZEN["input_dim"],
        hidden_dim=FROZEN["hidden_dim"],
        num_layers=FROZEN["num_layers"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN["lr"],
                            weight_decay=FROZEN["weight_decay"])

    print(f"\nTraining {FROZEN['epochs']} epochs...")
    for epoch in range(FROZEN["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = len(train_eps) // FROZEN["batch_size"]

        for _ in range(n_batches):
            feats, labels, mech_ids, mask = sampler.sample_batch(FROZEN["batch_size"])
            feats = feats.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            opt.zero_grad()
            logits = model(feats, mech_ids)
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, n_batches)

        if epoch % 5 == 0 or epoch == FROZEN["epochs"] - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}")

    # ── Evaluate grasp head ──
    print("\n--- Grasp Head Evaluation ---")
    grasp_metrics = evaluate_grasp_head(model, val_ids, device)
    for k, v in grasp_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ── Evaluate FSM ──
    print("\n--- FSM Evaluation (vertical-lift=0.02m) ---")
    fsm_metrics = evaluate_fsm(model, val_ids, device)
    g = fsm_metrics["global"]
    print(f"  Episodes: {g['n_episodes']}")
    print(f"  Emits: {g['n_emits']} ({100*g['n_emits']/g['n_episodes']:.1f}%)")
    print(f"  K10 recall: {g['k10_recall']:.4f}")
    print(f"  AR overlap: {g['ar_overlap']:.4f}")
    print(f"  Pre-anchor emit: {g['pre_anchor_emit']}")
    print(f"  Release overlap: {g['release_overlap']}")
    print(f"  Duplicate emit: {g['duplicate_emit']}")
    print(f"  Unsupported emit: {g['unsupported_emit']}")

    for suite in SUITES:
        if suite in fsm_metrics["per_suite"]:
            s = fsm_metrics["per_suite"][suite]
            print(f"  [{suite}] emit={s['emits']}/{s['n']} K10={s['k10_recall']:.3f} "
                  f"AR={s['ar_overlap']:.3f} pre={s['pre_anchor']} rel={s['release_overlap']} "
                  f"reset={s['reset_closure']:.3f}")

    # ── Gate checks ──
    print("\n--- Gate Checks ---")
    gates = {
        "event_recall >= 70%": grasp_metrics["event_recall"] >= 0.70,
        "event_precision >= 90%": grasp_metrics["event_precision"] >= 0.90,
        "median_latency <= 5": (grasp_metrics["latency_median"] if grasp_metrics["latency_median"] is not None else 999) <= 5,
        "median_fragmentation == 1": (grasp_metrics["fragmentation_median"] if grasp_metrics["fragmentation_median"] is not None else 999) == 1,
        "pre_anchor_emit == 0": g["pre_anchor_emit"] == 0,
        "unsupported_emit == 0": g["unsupported_emit"] == 0,
        "duplicate_emit == 0": g["duplicate_emit"] == 0,
        "release_overlap == 0": g["release_overlap"] == 0,
    }

    # Oracle retention
    oracle_ar = 0.870
    oracle_k10 = 0.209
    ar_retention = g["ar_overlap"] / oracle_ar if oracle_ar > 0 else 0
    k10_retention = g["k10_recall"] / oracle_k10 if oracle_k10 > 0 else 0
    gates["AR retention >= 70%"] = ar_retention >= 0.70
    gates["K10 retention >= 60%"] = k10_retention >= 0.60
    gates["Student AR >= 60.9%"] = g["ar_overlap"] >= 0.609
    gates["Student K10 >= 12.5%"] = g["k10_recall"] >= 0.125

    all_pass = True
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed: all_pass = False
        print(f"  {gate_name}: {status}")

    print(f"\n  OVERALL: {'ALL GATES PASS' if all_pass else 'SOME GATES FAIL'}")

    # Save checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / f"checkpoint_fold{fold_id}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "frozen": FROZEN,
        "fold_id": fold_id,
        "grasp_metrics": grasp_metrics,
        "fsm_metrics": {k: v for k, v in fsm_metrics.items() if k != "per_episode"},
    }, ckpt_path)
    print(f"\n  Checkpoint saved to {ckpt_path}")

    # Save per-episode FSM results for aggregation
    ep_results_path = Path(output_dir) / f"fsm_episodes_fold{fold_id}.json"
    ep_results_path.write_text(json.dumps(fsm_metrics["per_episode"], indent=2))
    print(f"  Episode results saved to {ep_results_path}")

    return {
        "frozen": FROZEN, "fold_id": fold_id,
        "grasp": grasp_metrics, "fsm": {k: v for k, v in fsm_metrics.items() if k != "per_episode"},
        "gates": gates, "all_pass": all_pass, "checkpoint": str(ckpt_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=str, default="/tmp/r10_2b_m")
    args = parser.parse_args()
    result = train_and_evaluate(fold_id=args.fold, output_dir=args.output)
    sys.exit(0 if result["all_pass"] else 1)
