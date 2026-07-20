#!/usr/bin/env python3
"""R10.2B-M Matched-Fold Audit: Teacher vs Student on identical 50 val episodes.

Runs Teacher grasp + FSM and Student grasp + FSM on the EXACT same Fold-0 val
identities. Reports paired counts, corrected fragmentation semantics,
and bootstrap CI for Student − Teacher difference.

No training. CPU or GPU. Requires trained checkpoint from train_r10_2b_multi_object_grasp.py.
"""

from __future__ import annotations

import json, math, os, sys, random
from collections import defaultdict, Counter
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

IDX = {"eef_z": 5}
SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]

# Frozen from R10.2B-M
FROZEN = {
    "route": "multi_object_transfer",
    "grasp_persistence": 3,
    "grasp_threshold": 0.5,
    "guard_type": "vertical_lift",
    "guard_param": 0.02,
    "max_episode_emits": 1,
}

def parse_mechanism(identity: str) -> str:
    parts = identity.split("/")
    tk = f"{parts[0]}/{parts[1]}"
    mapping = {"libero_goal/task_07": "unsupported_abstain", "libero_object": "single_object_pick_place",
               "libero_spatial": "single_object_pick_place", "libero_goal": "single_object_pick_place",
               "libero_10": "multi_object_transfer"}
    if tk in mapping: return mapping[tk]
    if parts[0] in mapping: return mapping[parts[0]]
    return "unsupported_abstain"

def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# Model (must match train_r10_2b)
# ═══════════════════════════════════════════════════════════════════════════════

class RoutedGraspDetector(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, mechanism_ids: list[str]) -> torch.Tensor:
        hidden, _ = self.encoder(x)
        B, T, H = hidden.shape
        logits = torch.zeros(B, T, device=x.device)
        for b in range(B):
            mech = mechanism_ids[b]
            if mech == "multi_object_transfer":
                logits[b] = self.head_multi(hidden[b]).squeeze(-1)
            elif mech == "single_object_pick_place":
                logits[b] = self.head_single(hidden[b]).squeeze(-1)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# FSM (identical for Teacher and Student, only grasp source differs)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fsm(
    grasp_signal: list[bool],  # True where grasp detected
    close_mask: list[bool],
    eef_z_vals: list[float],
    mechanism: str,
    k10_start: list[bool],
    ar_true: list[bool],
    release_risk: list[float],
) -> dict:
    """Run FSM with given grasp signal. Returns per-episode metrics."""
    T = len(grasp_signal)
    state = "IDLE"
    grasp_persist = 0
    anchor_step = -1
    anchor_eef_z = 0.0
    emitted = False
    total_emits = 0
    emit_step = -1
    event_id = 0
    violations = []
    positive_runs = []  # [number of contiguous positive runs per matched event]
    total_events = 0

    for t in range(T):
        if mechanism == "unsupported_abstain":
            continue

        grasp_detected = grasp_signal[t]

        if state == "IDLE" and close_mask[t]:
            state = "CLOSE_CANDIDATE"
            event_id += 1
            total_events += 1
            current_event_positive_runs = 0
            in_positive_run = False

        if state == "CLOSE_CANDIDATE":
            if grasp_detected:
                grasp_persist += 1
                if grasp_persist == 1:
                    anchor_step = t
                    anchor_eef_z = eef_z_vals[t]
                if not in_positive_run:
                    in_positive_run = True
            else:
                grasp_persist = 0
                if in_positive_run:
                    current_event_positive_runs += 1
                    in_positive_run = False
            if grasp_persist >= FROZEN["grasp_persistence"]:
                state = "ARMED"
                # Record fragmentation for this event
                if in_positive_run:
                    current_event_positive_runs += 1
                    in_positive_run = False
                positive_runs.append(current_event_positive_runs)

        if state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not close_mask[t]:
            state = "RESET"

        if state == "ARMED" and not emitted:
            vert_lift = eef_z_vals[t] - anchor_eef_z
            if vert_lift >= FROZEN["guard_param"]:
                state = "EVENT_CANDIDATE"

        if state == "EVENT_CANDIDATE" and not emitted:
            if total_emits < FROZEN["max_episode_emits"]:
                if t < anchor_step:
                    violations.append(f"PRE_ANCHOR")
                emitted = True
                total_emits += 1
                emit_step = t
                state = "EMITTED"

        if state == "RESET" and close_mask[t]:
            state = "CLOSE_CANDIDATE"
            grasp_persist = 0
            emitted = False
            anchor_step = -1
            event_id += 1
            total_events += 1
            current_event_positive_runs = 0
            in_positive_run = False

    n_emits = 1 if emit_step >= 0 else 0
    k10_hit = 1 if (emit_step >= 0 and k10_start[emit_step]) else 0
    ar_hit = 1 if (emit_step >= 0 and ar_true[emit_step]) else 0
    n_k10 = sum(1 for k in k10_start if k)
    n_ar = sum(1 for a in ar_true if a)
    pre_anchor = len(violations)
    release_overlap_val = 1 if (emit_step >= 0 and release_risk[emit_step] > 0.5) else 0
    dup = max(0, total_emits - 1)

    return {
        "emit_step": emit_step, "n_emits": n_emits,
        "k10_hit": k10_hit, "n_k10": n_k10,
        "ar_hit": ar_hit, "n_ar": n_ar,
        "pre_anchor": pre_anchor, "release_overlap": release_overlap_val,
        "duplicate": dup, "total_events": total_events,
        "positive_runs": positive_runs,
    }


# ═══════════════════════════════════════════════════════════════════════════════

def load_episode(identity: str) -> dict | None:
    parts = identity.split("/")
    s1_p = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
    teacher_p = TEACHER_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
    k10_p = K10_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "k10_labels_v121.jsonl"
    if not all(p.is_file() for p in [s1_p, teacher_p, k10_p]):
        return None

    s1 = _jsonl(s1_p)
    teacher = _jsonl(teacher_p)
    k10_labels = _jsonl(k10_p)
    T = min(len(s1), len(teacher), len(k10_labels))

    feats_25d = []
    grasp_true = []
    close_mask = []
    eef_z_vals = []
    k10_start = []
    ar_true = []
    release_risk = []

    for t in range(T):
        tr = teacher[t]
        sr = s1[t]
        sg = float(tr.get("stable_grasp_score", 0))
        rr = float(tr.get("release_risk", 0))
        ri = float(tr.get("regrasp_or_instability_risk", 0))
        cc = bool(tr.get("candidate_close", False))
        valid = bool(tr.get("student_valid", True))
        mech = parse_mechanism(identity)
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

        feats_25d.append([float(v) for v in sr["features_25d"]])
        grasp_true.append(ge)
        close_mask.append(cc)
        eef_z_vals.append(float(sr["features_25d"][IDX["eef_z"]]))
        k10_start.append(k10_labels[t].get("is_feasible_start", False))
        ar_true.append(ar)
        release_risk.append(rr)

    return {
        "identity": identity, "T": T,
        "feats": feats_25d, "grasp_true": grasp_true, "close": close_mask,
        "eef_z": eef_z_vals, "k10_start": k10_start, "ar_true": ar_true,
        "release_risk": release_risk, "mech": mech,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = Path("/tmp/r10_2b_m_checkpoint.pt")
    if not ckpt_path.is_file():
        print("ERROR: Run train_r10_2b_multi_object_grasp.py first to generate checkpoint")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = RoutedGraspDetector(
        input_dim=ckpt["frozen"]["input_dim"],
        hidden_dim=ckpt["frozen"]["hidden_dim"],
        num_layers=ckpt["frozen"]["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Load Fold-0 val identities (multi_object_transfer only)
    fold = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text(encoding="utf-8"))
    f0 = next(f for f in fold["folds"] if f["fold_id"] == 0)
    val_ids = sorted(f0["validation_identities"])
    val_ids = [i for i in val_ids if parse_mechanism(i) == "multi_object_transfer"]
    print(f"Val identities: {len(val_ids)}")

    # Run Teacher + Student on each identity
    paired = []
    with torch.no_grad():
        for identity in val_ids:
            ep = load_episode(identity)
            if ep is None:
                continue

            # Teacher FSM
            teacher_grasp = ep["grasp_true"]
            teacher_result = run_fsm(
                teacher_grasp, ep["close"], ep["eef_z"], ep["mech"],
                ep["k10_start"], ep["ar_true"], ep["release_risk"],
            )

            # Student FSM
            feats = torch.tensor(ep["feats"], dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(feats, [ep["mech"]]).squeeze(0)
            probs = torch.sigmoid(logits).cpu()
            student_grasp = [p.item() > FROZEN["grasp_threshold"] for p in probs]

            student_result = run_fsm(
                student_grasp, ep["close"], ep["eef_z"], ep["mech"],
                ep["k10_start"], ep["ar_true"], ep["release_risk"],
            )

            paired.append({
                "identity": identity,
                "T": ep["T"],
                "teacher": teacher_result,
                "student": student_result,
                "has_k10": ep["k10_start"].count(True) > 0,
                "has_ar": ep["ar_true"].count(True) > 0,
            })

    # ── Paired comparison ──
    print(f"\n{'='*70}")
    print("PAIRED TEACHER vs STUDENT (N={} episodes)".format(len(paired)))
    print(f"{'='*70}")

    # K10 hit contingency
    t_k10_hit = sum(1 for p in paired if p["teacher"]["k10_hit"])
    s_k10_hit = sum(1 for p in paired if p["student"]["k10_hit"])
    both_k10 = sum(1 for p in paired if p["teacher"]["k10_hit"] and p["student"]["k10_hit"])
    t_only_k10 = sum(1 for p in paired if p["teacher"]["k10_hit"] and not p["student"]["k10_hit"])
    s_only_k10 = sum(1 for p in paired if not p["teacher"]["k10_hit"] and p["student"]["k10_hit"])
    neither_k10 = sum(1 for p in paired if not p["teacher"]["k10_hit"] and not p["student"]["k10_hit"])

    n_k10_total = sum(1 for p in paired if p["has_k10"])

    print(f"\nK10 hit contingency (episodes with K10: {n_k10_total}):")
    print(f"  Both hit:     {both_k10}")
    print(f"  Teacher only: {t_only_k10}")
    print(f"  Student only: {s_only_k10}")
    print(f"  Neither:      {neither_k10}")
    print(f"  Teacher recall: {t_k10_hit}/{n_k10_total} = {t_k10_hit/max(1,n_k10_total):.4f}")
    print(f"  Student recall: {s_k10_hit}/{n_k10_total} = {s_k10_hit/max(1,n_k10_total):.4f}")

    # AR hit contingency
    n_emit_t = sum(1 for p in paired if p["teacher"]["n_emits"] > 0)
    n_emit_s = sum(1 for p in paired if p["student"]["n_emits"] > 0)
    t_ar = sum(1 for p in paired if p["teacher"]["ar_hit"])
    s_ar = sum(1 for p in paired if p["student"]["ar_hit"])

    print(f"\nAR overlap (Teacher emits: {n_emit_t}, Student emits: {n_emit_s}):")
    print(f"  Teacher AR overlap: {t_ar}/{n_emit_t} = {t_ar/max(1,n_emit_t):.4f}")
    print(f"  Student AR overlap: {s_ar}/{n_emit_s} = {s_ar/max(1,n_emit_s):.4f}")

    # Student-only hit analysis
    if s_only_k10 > 0:
        print(f"\nStudent-only K10 hits ({s_only_k10}):")
        for p in paired:
            if not p["teacher"]["k10_hit"] and p["student"]["k10_hit"]:
                t_emit = p["teacher"]["emit_step"]
                s_emit = p["student"]["emit_step"]
                print(f"  {p['identity']}: T_emit={t_emit} S_emit={s_emit} shift={s_emit - t_emit if t_emit >=0 else 'N/A'}")

    if t_only_k10 > 0:
        print(f"\nTeacher-only K10 hits ({t_only_k10}):")
        for p in paired:
            if p["teacher"]["k10_hit"] and not p["student"]["k10_hit"]:
                t_emit = p["teacher"]["emit_step"]
                s_emit = p["student"]["emit_step"]
                print(f"  {p['identity']}: T_emit={t_emit} S_emit={s_emit}")

    # ── Fragmentation analysis ──
    print(f"\n{'='*70}")
    print("FRAGMENTATION SEMANTICS (positive_runs_per_matched_event)")
    print(f"{'='*70}")

    all_teacher_runs = []
    all_student_runs = []
    for p in paired:
        all_teacher_runs.extend(p["teacher"]["positive_runs"])
        all_student_runs.extend(p["student"]["positive_runs"])

    def frag_stats(runs: list[int], label: str):
        if not runs:
            return
        c = Counter(runs)
        median = sorted(runs)[len(runs)//2]
        n_total = len(runs)
        n_1 = c.get(1, 0)
        n_0 = c.get(0, 0)
        n_ge_2 = sum(v for k, v in c.items() if k >= 2)
        excess = [max(0, r - 1) for r in runs]
        median_excess = sorted(excess)[len(excess)//2]
        print(f"  [{label}] n_events={n_total} median_runs={median} median_excess={median_excess}")
        print(f"    runs=1: {n_1} ({100*n_1/n_total:.1f}%)")
        print(f"    runs=0: {n_0} ({100*n_0/n_total:.1f}%)")
        print(f"    runs>=2: {n_ge_2} ({100*n_ge_2/n_total:.1f}%)")
        print(f"    distribution: {dict(c.most_common(5))}")

    frag_stats(all_teacher_runs, "Teacher")
    frag_stats(all_student_runs, "Student")

    # ── Safety gates ──
    print(f"\n{'='*70}")
    print("SAFETY GATES")
    print(f"{'='*70}")
    s_pre = sum(1 for p in paired if p["student"]["pre_anchor"] > 0)
    s_dup = sum(1 for p in paired if p["student"]["duplicate"] > 0)
    s_rel = sum(1 for p in paired if p["student"]["release_overlap"] > 0)

    for name, val, expected in [
        ("pre-anchor emit", s_pre, 0), ("duplicate emit", s_dup, 0),
        ("release overlap", s_rel, 0),
    ]:
        print(f"  {name}: {val} (expected {expected}) {'PASS' if val == expected else 'FAIL'}")

    # ── Bootstrap CI ──
    print(f"\n{'='*70}")
    print("BOOTSTRAP CI (Student − Teacher, 10000 resamples)")
    print(f"{'='*70}")
    random.seed(20260720)
    B = 10000
    n = len(paired)
    k10_diffs = []
    ar_diffs = []
    emit_diffs = []

    for p in paired:
        t_k10 = 1 if p["teacher"]["k10_hit"] else 0
        s_k10 = 1 if p["student"]["k10_hit"] else 0
        t_ar = 1 if p["teacher"]["ar_hit"] else 0
        s_ar = 1 if p["student"]["ar_hit"] else 0
        t_emit = 1 if p["teacher"]["n_emits"] > 0 else 0
        s_emit = 1 if p["student"]["n_emits"] > 0 else 0
        k10_diffs.append(s_k10 - t_k10)
        ar_diffs.append(s_ar - t_ar)
        emit_diffs.append(s_emit - t_emit)

    def bootstrap_ci(diffs, B=10000):
        means = []
        for _ in range(B):
            sample = [diffs[random.randint(0, n-1)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        return means[B//40], means[B//2], means[B - B//40]  # 95% CI

    for name, diffs, denom_name, denom_val in [
        ("K10 hit rate diff", k10_diffs, "K10 episodes", n_k10_total),
        ("AR overlap diff", ar_diffs, "Total episodes", n),
        ("Emit rate diff", emit_diffs, "Total episodes", n),
    ]:
        lo, med, hi = bootstrap_ci(diffs)
        raw_mean = sum(diffs) / len(diffs)
        print(f"  {name}: mean={raw_mean:.4f} 95% CI=[{lo:.4f}, {hi:.4f}] per {denom_name}={denom_val}")

    print(f"\n{'='*70}")
    print("MATCHED AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
