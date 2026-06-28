#!/usr/bin/env python3
"""
Phase B Held-Out Evaluator V1 — implements LOTO_METRIC_SCHEMA_V2.

Runs frozen SC5DetectorRuntime on held-out feature CSV + materialized labels.
Computes all primary and secondary metrics per the metric preregistration.
Testable on Fold 00 (pilot only — already opened). NEVER run on Folds 01-09
before LOTO_TEST_OPEN_EVENT_V1.json exists and is verified.
"""
import argparse, csv, hashlib, json, math, os, sys, numpy as np
from collections import defaultdict
from datetime import timezone, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3; GUARD = 5; K_SC5 = 10
EVALUATOR_VERSION = "V1"


def load_labels(path):
    """Load materialized held-out labels. Fail-closed on duplicates."""
    labels = {}
    tasks = set()
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            lab = json.loads(line)
            key = (lab["task_idx"], lab["state_id"], lab["step_idx"])
            assert key not in labels, f"Duplicate label: {key}"
            labels[key] = lab
            tasks.add(lab["task_idx"])
    return labels, tasks


def load_heldout_csv(path):
    """Load held-out feature CSV, group by episode."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    episodes = defaultdict(list)
    for r in rows:
        episodes[(int(r["task_idx"]), int(r["state_id"]))].append(r)
    for k in episodes:
        episodes[k] = sorted(episodes[k], key=lambda x: int(x["step"]))
    return episodes


def compute_corridor_active_set(ep_labels):
    """Compute corridor_active_at_t from teacher labels using frozen SC5 protocol.
    Uses valid-start set membership (NOT range interval)."""
    from gripper_attack.v2_privileged_teacher import (
        find_sc5_anchor_v2, compute_sc5_valid_start_corridor
    )
    sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD)
    if sc5["valid"]:
        return set(compute_sc5_valid_start_corridor(ep_labels, sc5["anchor"], K=K_SC5)["corridor_active_at_t"]), sc5["anchor"]
    else:
        return set(), sc5["anchor"]


def evaluate_fold(checkpoint_path, heldout_csv_path, heldout_labels_path, output_dir):
    """Run evaluation on one fold's held-out data. Returns per-episode results."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    # Load
    ckpt_path = Path(checkpoint_path)
    with open(ckpt_path, "rb") as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    labels, label_tasks = load_labels(heldout_labels_path)
    episodes = load_heldout_csv(heldout_csv_path)

    ep_keys = sorted(episodes.keys())
    ep_label_keys = set((lab["task_idx"], lab["state_id"]) for lab in labels.values())
    assert ep_label_keys == set(ep_keys), \
        f"Keyset mismatch: episodes={len(ep_keys)} labels={len(ep_label_keys)}"

    # Per-episode evaluation
    episode_results = []
    for ep_key in ep_keys:
        rows = episodes[ep_key]
        t, s = ep_key
        task_name = str(t)  # Will be enriched from labels

        # Collect labels for this episode
        ep_labels = []
        for r in rows:
            step = int(r["step"])
            lab = labels.get((t, s, step))
            if lab is None:
                raise KeyError(f"Missing label t{t} s{s} step{step}")
            ep_labels.append(lab)
            task_name = lab.get("task_name", str(t))

        corridor_active, teacher_anchor = compute_corridor_active_set(ep_labels)
        teacher_positive = len(corridor_active) > 0
        n_label_rows = len(ep_labels)

        # Run detector
        rt = SC5DetectorRuntime(str(ckpt_path), tau_corridor=TAU_CORRIDOR,
                                tau_release=TAU_RELEASE, guard=GUARD)
        rt.reset()
        emit_step = -1; armed = False
        corridor_ps = []; all_steps = []
        for r in rows:
            step = int(r["step"])
            features = {name: float(r["f_" + name]) for name in SC5_FEATURES}
            for v in features.values():
                assert not (math.isnan(v) or math.isinf(v)), \
                    f"NaN/Inf feature at t{t} s{s} step{step}"
            result = rt.update(features, step)
            cp = result.get("corridor_p")
            if cp is not None:
                corridor_ps.append(float(cp))
                all_steps.append(step)
            if result.get("state") == "ARMED": armed = True
            if emit_step < 0 and result.get("state") == "EMITTED":
                emit_step = step

        emitted = emit_step >= 0

        # --- Classify episode ---
        if teacher_positive:
            if emitted:
                if emit_step in corridor_active:
                    category = "CW"  # correct window
                elif emit_step < min(corridor_active):
                    category = "FE"  # false early
                else:
                    category = "LI"  # late or invalid
            else:
                category = "MP"  # missed positive
        else:
            if emitted:
                category = "FPn"  # false positive (negative episode)
            else:
                category = "TN"

        # --- Timing ---
        signed_err = None; abs_err = None
        if teacher_positive and emitted:
            signed_err = emit_step - teacher_anchor
            abs_err = abs(signed_err)

        # --- Frame-level FPR ---
        frame_fpr_pos = 0; frame_fpr_neg = 0
        n_corr_pos = 0; n_corr_neg = 0
        cp_pos_list = []; cp_neg_list = []
        for r, lab in zip(rows, ep_labels):
            step = int(r["step"])
            lab_step = lab["step_idx"]
            cp_val = None
            for s, cp in zip(all_steps, corridor_ps):
                if s == step:
                    cp_val = cp; break
            if cp_val is None: continue
            if lab_step in corridor_active:
                n_corr_pos += 1
                cp_pos_list.append(cp_val)
                if cp_val > TAU_CORRIDOR: frame_fpr_pos += 1
            else:
                n_corr_neg += 1
                cp_neg_list.append(cp_val)
                if cp_val > TAU_CORRIDOR: frame_fpr_neg += 1

        episode_results.append({
            "task_idx": t, "task_name": task_name, "state_id": s,
            "teacher_positive": teacher_positive,
            "teacher_anchor": teacher_anchor,
            "corridor_active_steps": sorted(corridor_active),
            "n_corridor_pos_frames": n_corr_pos,
            "n_corridor_neg_frames": n_corr_neg,
            "emitted": emitted, "emit_step": emit_step,
            "armed": armed,
            "category": category,
            "signed_error": signed_err, "absolute_error": abs_err,
            "corridor_p_max": max(corridor_ps) if corridor_ps else None,
            "corridor_p_median": float(np.median(corridor_ps)) if corridor_ps else None,
            "corridor_p_p99": float(np.percentile(corridor_ps, 99)) if len(corridor_ps) > 1 else (max(corridor_ps) if corridor_ps else None),
            "cp_pos_list": cp_pos_list, "cp_neg_list": cp_neg_list,
        })

    # --- Aggregate metrics ---
    n_ep = len(episode_results)
    tp_eps = [e for e in episode_results if e["teacher_positive"]]
    tn_eps = [e for e in episode_results if not e["teacher_positive"]]
    n_pos = len(tp_eps); n_neg = len(tn_eps)

    cw = sum(1 for e in tp_eps if e["category"] == "CW")
    fe = sum(1 for e in tp_eps if e["category"] == "FE")
    li = sum(1 for e in tp_eps if e["category"] == "LI")
    mp = sum(1 for e in tp_eps if e["category"] == "MP")
    fpn = sum(1 for e in tn_eps if e["category"] == "FPn")
    tn = sum(1 for e in tn_eps if e["category"] == "TN")

    coverage = cw / n_pos if n_pos > 0 else None
    k10_containment = cw / n_pos if n_pos > 0 else None  # Under V2 schema, CW == K10 containment

    emitted_pos = [e for e in tp_eps if e["emitted"]]
    timing_errors = [e["signed_error"] for e in emitted_pos if e["signed_error"] is not None]
    abs_errors = [e["absolute_error"] for e in emitted_pos if e["absolute_error"] is not None]

    # Frame-level FPR (no-corridor frames only) — Definition A: per-frame score FPR
    # Counts ALL evaluable no-corridor frames where corridor_p > tau_corridor.
    # Independent of state-machine emission. Frames after emission that still have
    # corridor_p values ARE included. This is a per-frame score discrimination metric,
    # NOT an emission-based trigger rate.
    all_cp_neg = []
    for e in episode_results:
        all_cp_neg.extend(e["cp_neg_list"])
    all_cp_pos = []
    for e in episode_results:
        all_cp_pos.extend(e["cp_pos_list"])
    n_no_corridor_frames = len(all_cp_neg)
    n_corridor_pos_frames = len(all_cp_pos)
    n_evaluable_frames = n_no_corridor_frames + n_corridor_pos_frames
    fpr_numerator = sum(1 for v in all_cp_neg if v > TAU_CORRIDOR)
    frame_fpr = fpr_numerator / n_no_corridor_frames if n_no_corridor_frames > 0 else None

    # Score saturation
    sat_pos = sum(1 for v in all_cp_pos if v >= 0.99) / len(all_cp_pos) if all_cp_pos else None
    sat_neg = sum(1 for v in all_cp_neg if v >= 0.99) / len(all_cp_neg) if all_cp_neg else None

    zero_emission = sum(1 for e in episode_results if not e["emitted"]) / n_ep

    summary = {
        "n_episodes": n_ep, "n_positive": n_pos, "n_negative": n_neg,
        "n_emitted": sum(1 for e in episode_results if e["emitted"]),
        "CW": cw, "FE": fe, "LI": li, "MP": mp, "FPn": fpn, "TN": tn,
        "coverage": coverage,
        "K10_containment": k10_containment,
        "n_evaluable_frames": n_evaluable_frames,
        "n_corridor_pos_frames": n_corridor_pos_frames,
        "n_no_corridor_frames": n_no_corridor_frames,
        "no_corridor_frame_FPR": frame_fpr,
        "no_corridor_frame_FPR_numerator": fpr_numerator,
        "no_corridor_frame_FPR_denominator": n_no_corridor_frames,
        "no_corridor_frame_FPR_definition": "per-frame score FPR (Definition A): fraction of all evaluable no-corridor frames where corridor_p > tau_corridor. Independent of state-machine emission logic. NOT an episode-level false-trigger rate.",
        "episode_false_trigger_rate": fpn / n_neg if n_neg > 0 else None,
        "episode_false_trigger_rate_definition": "Fraction of teacher-negative episodes where the detector state machine emitted. This is an episode-level metric distinct from the frame-level no_corridor_frame_FPR.",
        "zero_emission_rate": zero_emission,
        "early_trigger_rate": fe / n_pos if n_pos > 0 else None,
        "late_trigger_rate": li / n_pos if n_pos > 0 else None,
        "median_signed_error": float(np.median(timing_errors)) if timing_errors else None,
        "median_absolute_error": float(np.median(abs_errors)) if abs_errors else None,
        "mean_signed_error": float(np.mean(timing_errors)) if timing_errors else None,
        "mean_absolute_error": float(np.mean(abs_errors)) if abs_errors else None,
        "score_saturation_pos": sat_pos,
        "score_saturation_neg": sat_neg,
        "corridor_p_median_pos": float(np.median(all_cp_pos)) if all_cp_pos else None,
        "corridor_p_median_neg": float(np.median(all_cp_neg)) if all_cp_neg else None,
        "checkpoint_sha256": ckpt_sha,
        "evaluator_version": EVALUATOR_VERSION,
    }

    # Write outputs
    ep_path = out / "HELDOUT_EPISODE_RESULTS.json"
    with open(ep_path, "w") as f:
        json.dump([{k: v for k, v in e.items() if k not in ("cp_pos_list", "cp_neg_list")}
                   for e in episode_results], f, indent=2)

    sum_path = out / "HELDOUT_SEED_SUMMARY.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Row-level diagnostics CSV
    diag_path = out / "HELDOUT_ROW_DIAGNOSTICS.csv"
    with open(diag_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_idx","task_name","state_id","teacher_positive","teacher_anchor",
                     "emitted","emit_step","category","signed_error","absolute_error",
                     "corridor_p_max","corridor_p_median","corridor_p_p99","n_corridor_pos",
                     "n_corridor_neg"])
        for e in episode_results:
            w.writerow([e["task_idx"],e["task_name"],e["state_id"],e["teacher_positive"],
                       e["teacher_anchor"],e["emitted"],e["emit_step"],e["category"],
                       e["signed_error"],e["absolute_error"],e["corridor_p_max"],
                       e["corridor_p_median"],e["corridor_p_p99"],e["n_corridor_pos_frames"],
                       e["n_corridor_neg_frames"]])

    # Atomic COMPLETE
    complete = {
        "gate": "PHASE_B_HELDOUT_EVALUATION_COMPLETE",
        "checkpoint_sha256": ckpt_sha,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {k: v for k, v in summary.items() if k not in ("cp_pos_list", "cp_neg_list")},
    }
    with open(out / "COMPLETE.json", "w") as f:
        json.dump(complete, f, indent=2)

    return episode_results, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--heldout_csv", required=True, help="FOLDxx_HELDOUT_FEATURE_DATASET.csv")
    ap.add_argument("--heldout_labels", required=True, help="FOLDxx_teacher_labels_heldout.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fold_id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)

    print(f"Phase B Evaluator V1: Fold {args.fold_id} Seed {args.seed}")
    ep_results, summary = evaluate_fold(
        args.checkpoint, args.heldout_csv, args.heldout_labels, args.output_dir)

    n_pos = summary["n_positive"]
    print(f"  Episodes: {summary['n_episodes']} (pos={n_pos} neg={summary['n_negative']})")
    print(f"  Coverage: {summary['coverage']:.4f}" if summary["coverage"] is not None else "  Coverage: NO_POSITIVES")
    print(f"  K10 containment: {summary['K10_containment']:.4f}" if summary["K10_containment"] is not None else "  K10: NO_POSITIVES")
    print(f"  No-corridor FPR: {summary['no_corridor_frame_FPR']:.4f}" if summary["no_corridor_frame_FPR"] is not None else "  FPR: NO_DATA")
    print(f"  Med signed error: {summary['median_signed_error']}" if summary["median_signed_error"] is not None else "  Med signed error: N/A")
    print(f"  Score saturation (pos): {summary['score_saturation_pos']:.4f}" if summary["score_saturation_pos"] is not None else "  Sat pos: NO_DATA")
    print(f"  Score saturation (neg): {summary['score_saturation_neg']:.4f}" if summary["score_saturation_neg"] is not None else "  Sat neg: NO_DATA")
    print(f"  Zero emission rate: {summary['zero_emission_rate']:.4f}")
    print(f"\n  Complete: {args.output_dir}/COMPLETE.json")


if __name__ == "__main__":
    main()
