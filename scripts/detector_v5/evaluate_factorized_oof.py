#!/usr/bin/env python3
"""Evaluate Factorized Student OOF held-out predictions.

Reads 24 prediction shards, computes all event-level metrics per the
frozen OOF eval protocol, and produces the gate decision.

Input: prediction base directory with 24 subdirectories
       (e.g. OFFICIAL_V3_FACTORIZED_STUDENT_OOF_PREDICTIONS_V1/predict_25D9D_fold0_seed42/)
Output: sealed evaluation root with all metric files + gate decision
"""
import argparse, hashlib, json, os, sys, uuid, platform
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    with t.open("x") as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {"SHA256SUMS", "SHA256SUMS.sha256"}
    fs = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in fs)
    _atomic_text(root / "SHA256SUMS", c)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


THRESHOLD = 0.5
SUPPORTED_ROUTES = ["single_object_pick_place", "multi_object_transfer"]


def load_event_predictions(pred_dir):
    """Return list of event prediction dicts from a sealed prediction shard."""
    ep_file = pred_dir / "heldout_event_predictions.jsonl"
    if not ep_file.is_file():
        raise FileNotFoundError(f"missing event predictions: {ep_file}")
    return [json.loads(l) for l in ep_file.read_text().splitlines() if l.strip()]


def load_step_predictions(pred_dir):
    """Return list of step prediction dicts from a sealed prediction shard."""
    sp_file = pred_dir / "heldout_step_predictions.jsonl"
    if not sp_file.is_file():
        raise FileNotFoundError(f"missing step predictions: {sp_file}")
    return [json.loads(l) for l in sp_file.read_text().splitlines() if l.strip()]


def compute_event_recall(events, head, threshold=THRESHOLD):
    """Compute recall: TP / (TP + FN) for events where Teacher says positive."""
    score_key = f"{head}_score_max"
    emit_key = f"{head}_emit"
    target_key = f"{head}_target"

    positives = [e for e in events if e[target_key]]
    if not positives:
        return None, 0
    tp = sum(1 for e in positives if e[score_key] >= threshold)
    return tp / len(positives), len(positives)


def compute_precision(events, head, threshold=THRESHOLD):
    """Precision: TP / (TP + FP) for events where Student emits."""
    emit_key = f"{head}_emit"
    target_key = f"{head}_target"

    emissions = [e for e in events if e[emit_key]]
    if not emissions:
        return None, 0
    tp = sum(1 for e in emissions if e[target_key])
    return tp / len(emissions), len(emissions)


def compute_route_metrics(events, head, route, threshold=THRESHOLD):
    """Per-route recall for a specific head."""
    route_events = [e for e in events if e["mechanism_route"] == route]
    if not route_events:
        return None, 0
    return compute_event_recall(route_events, head, threshold)


def compute_safety_metrics(step_records):
    """Compute safety emit rates from step-level predictions."""
    total_known = 0
    release_overlap = 0
    background_emit = 0
    unsupported_emit = 0
    unsupported_known = 0

    for s in step_records:
        if not s["route_supported"]:
            if s["grasp_prob"] > 0 or s["manipulation_prob"] > 0 or s["release_prob"] > 0:
                unsupported_emit += 1
            continue

        g_km = s["grasp_known_mask"]
        m_km = s["manipulation_known_mask"]
        r_km = s["release_known_mask"]

        # Release-overlap: both grasp and release emit on same step where both known
        if g_km and r_km and s["grasp_prob"] >= THRESHOLD and s["release_prob"] >= THRESHOLD:
            release_overlap += 1

        # Background false emit: any head emits on background step (event_id == -1)
        if s["event_id"] < 0:
            if g_km and s["grasp_prob"] >= THRESHOLD:
                background_emit += 1
            elif m_km and s["manipulation_prob"] >= THRESHOLD:
                background_emit += 1
            elif r_km and s["release_prob"] >= THRESHOLD:
                background_emit += 1

        # Count known steps for rate denominator
        if g_km or m_km or r_km:
            total_known += 1

    return {
        "release_overlap_emit_count": release_overlap,
        "release_overlap_emit_rate": release_overlap / max(1, total_known),
        "background_false_emit_count": background_emit,
        "background_false_emit_rate": background_emit / max(1, total_known),
        "unsupported_route_emit_count": unsupported_emit,
        "total_known_steps": total_known,
    }


def compute_onset_latency(step_records, threshold=THRESHOLD):
    """Per-event onset latency: steps from event start to first detection."""
    # Group steps by (identity, event_id)
    event_steps = defaultdict(list)
    for s in step_records:
        if s["event_id"] < 0:
            continue
        event_steps[(s["canonical_parent_key"], s["event_id"])].append(s)

    latencies = {"grasp": [], "manipulation": [], "release": []}
    for (identity, eid), steps in event_steps.items():
        steps_sorted = sorted(steps, key=lambda s: s["step_index"])
        for head in ["grasp", "manipulation", "release"]:
            prob_key = f"{head}_prob"
            km_key = f"{head}_known_mask"
            first_detect = None
            event_start = steps_sorted[0]["step_index"]
            for s in steps_sorted:
                if s[km_key] and s[prob_key] >= threshold:
                    first_detect = s["step_index"] - event_start
                    break
            if first_detect is not None:
                latencies[head].append(first_detect)

    return {head: {"mean": mean(vals) if vals else None, "count": len(vals)}
            for head, vals in latencies.items()}


def compute_event_coverage(step_records, threshold=THRESHOLD):
    """Fraction of known steps within each event where prob >= threshold."""
    event_steps = defaultdict(list)
    for s in step_records:
        if s["event_id"] < 0:
            continue
        event_steps[(s["canonical_parent_key"], s["event_id"])].append(s)

    coverages = {"grasp": [], "manipulation": [], "release": []}
    for (identity, eid), steps in event_steps.items():
        for head in ["grasp", "manipulation", "release"]:
            prob_key = f"{head}_prob"
            km_key = f"{head}_known_mask"
            known_steps = [s for s in steps if s[km_key]]
            if not known_steps:
                continue
            detected = sum(1 for s in known_steps if s[prob_key] >= threshold)
            coverages[head].append(detected / len(known_steps))

    return {head: {"mean": mean(vals) if vals else None, "count": len(vals)}
            for head, vals in coverages.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-base", type=Path, required=True,
                    help="Directory containing all 24 prediction shard directories")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--eval-protocol", type=Path,
                    default=ROOT / "configs/DETECTOR_V5_FACTORIZED_OOF_EVAL_PROTOCOL_V1.json")
    args = ap.parse_args()

    pred_base = args.predictions_base.resolve()
    out = args.output_root.resolve()
    protocol = json.loads(args.eval_protocol.read_text())

    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # ── Load all prediction shards ──
    print("Loading predictions...")
    model_types = ["25D9D", "25D"]
    folds = [0, 1, 2, 3]
    seeds = [42, 123, 456]

    all_event_preds = {mt: {} for mt in model_types}  # mt -> {(fold,seed): [events]}
    all_step_preds = {mt: {} for mt in model_types}

    for mt in model_types:
        for fold in folds:
            for seed in seeds:
                shard_dir = pred_base / f"predict_{mt}_fold{fold}_seed{seed}"
                key = (fold, seed)
                try:
                    all_event_preds[mt][key] = load_event_predictions(shard_dir)
                    all_step_preds[mt][key] = load_step_predictions(shard_dir)
                except Exception as e:
                    print(f"  WARNING: {shard_dir}: {e}")
                    all_event_preds[mt][key] = []
                    all_step_preds[mt][key] = []

    # ── Compute per-run metrics ──
    print("Computing per-run metrics...")
    per_run_metrics = {mt: {} for mt in model_types}
    for mt in model_types:
        for (fold, seed), events in all_event_preds[mt].items():
            steps = all_step_preds[mt][(fold, seed)]
            run_key = f"{mt}_fold{fold}_seed{seed}"

            metrics = {}
            for head in ["grasp", "manipulation", "release"]:
                recall, n_pos = compute_event_recall(events, head)
                precision, n_emit = compute_precision(events, head)
                metrics[f"{head}_recall"] = recall if recall is not None else None
                metrics[f"{head}_precision"] = precision if precision is not None else None
                metrics[f"{head}_positive_events"] = n_pos

            # Per-route
            for route in SUPPORTED_ROUTES:
                for head in ["grasp", "manipulation", "release"]:
                    recall, n = compute_route_metrics(events, head, route)
                    metrics[f"{route}_{head}_recall"] = recall if recall is not None else None

            # Later event
            later = [e for e in events if e["is_later_event"]]
            first = [e for e in events if not e["is_later_event"]]
            for head in ["grasp", "manipulation", "release"]:
                lr, ln = compute_event_recall(later, head)
                fr, fn = compute_event_recall(first, head)
                metrics[f"{head}_later_event_recall"] = lr if lr is not None else None
                metrics[f"{head}_first_event_recall"] = fr if fr is not None else None

            # Safety
            safety = compute_safety_metrics(steps)
            metrics.update(safety)

            # Onset latency
            onset = compute_onset_latency(steps)
            for head in ["grasp", "manipulation", "release"]:
                metrics[f"{head}_onset_latency_mean"] = onset[head]["mean"]

            # Event coverage
            coverage = compute_event_coverage(steps)
            for head in ["grasp", "manipulation", "release"]:
                metrics[f"{head}_coverage_mean"] = coverage[head]["mean"]

            metrics["total_events"] = len(events)
            metrics["total_steps"] = len(steps)
            per_run_metrics[mt][run_key] = metrics

    # ── Head-level aggregation (pooled across folds, per seed) ──
    print("Computing head-level aggregates...")
    per_head_metrics = {mt: {} for mt in model_types}
    for mt in model_types:
        for head in ["grasp", "manipulation", "release"]:
            # Pool all events of this model type across all folds/seeds
            all_events = []
            for (fold, seed), events in all_event_preds[mt].items():
                all_events.extend(events)

            recall, n = compute_event_recall(all_events, head)
            precision, ne = compute_precision(all_events, head)

            # Per-seed stats
            seed_recalls = defaultdict(list)
            for (fold, seed), events in all_event_preds[mt].items():
                r, _ = compute_event_recall(events, head)
                if r is not None:
                    seed_recalls[seed].append(r)

            seed_stats = {}
            for seed, recalls in seed_recalls.items():
                if recalls:
                    seed_stats[f"seed{seed}"] = {
                        "mean": mean(recalls), "stdev": stdev(recalls) if len(recalls) > 1 else 0,
                        "per_fold": recalls,
                    }

            per_head_metrics[mt][head] = {
                "pooled_recall": recall, "pooled_precision": precision,
                "total_positive_events": n,
                "seed_aggregation": seed_stats,
            }

    # ── Route-level aggregation ──
    print("Computing route-level aggregates...")
    per_route_metrics = {mt: {} for mt in model_types}
    for mt in model_types:
        for route in SUPPORTED_ROUTES:
            per_route_metrics[mt][route] = {}
            route_events = []
            for (fold, seed), events in all_event_preds[mt].items():
                route_events.extend([e for e in events if e["mechanism_route"] == route])

            for head in ["grasp", "manipulation", "release"]:
                recall, n = compute_event_recall(route_events, head)
                per_route_metrics[mt][route][head] = {
                    "recall": recall, "positive_events": n,
                }

        # Macro over routes
        per_route_metrics[mt]["macro"] = {}
        for head in ["grasp", "manipulation", "release"]:
            route_recalls = []
            for route in SUPPORTED_ROUTES:
                r = per_route_metrics[mt][route][head]["recall"]
                if r is not None:
                    route_recalls.append(r)
            per_route_metrics[mt]["macro"][head] = {
                "recall": mean(route_recalls) if route_recalls else None,
                "route_recalls": route_recalls,
            }

    # ── Later-event metrics ──
    print("Computing later-event metrics...")
    later_event_metrics = {mt: {} for mt in model_types}
    for mt in model_types:
        all_later = []
        all_first = []
        for (fold, seed), events in all_event_preds[mt].items():
            all_later.extend([e for e in events if e["is_later_event"]])
            all_first.extend([e for e in events if not e["is_later_event"]])

        for head in ["grasp", "manipulation", "release"]:
            lr, ln = compute_event_recall(all_later, head)
            fr, fn_ = compute_event_recall(all_first, head)
            later_event_metrics[mt][head] = {
                "first_event_recall": fr, "first_event_count": fn_,
                "later_event_recall": lr, "later_event_count": ln,
            }

    # ── Safety emit metrics ──
    print("Computing safety metrics...")
    safety_metrics = {mt: {} for mt in model_types}
    for mt in model_types:
        all_steps = []
        for (fold, seed), steps in all_step_preds[mt].items():
            all_steps.extend(steps)
        safety_metrics[mt] = compute_safety_metrics(all_steps)

    # ── Model comparison: 25D9D vs 25D ──
    print("Computing model comparison...")
    model_comparison = {}
    for head in ["grasp", "manipulation", "release"]:
        comparison = {}
        for mt in model_types:
            comparison[mt] = per_head_metrics[mt][head]["pooled_recall"]
        diff_25d9d_minus_25d = (
            (comparison.get("25D9D") or 0) - (comparison.get("25D") or 0)
        )
        model_comparison[head] = {
            "25D9D_recall": comparison.get("25D9D"),
            "25D_recall": comparison.get("25D"),
            "delta": diff_25d9d_minus_25d,
        }

    # For later event
    for head in ["grasp", "manipulation", "release"]:
        cmp = {}
        for mt in model_types:
            cmp[mt] = later_event_metrics[mt][head].get("later_event_recall")
        model_comparison[f"{head}_later_event"] = {
            "25D9D_later_recall": cmp.get("25D9D"),
            "25D_later_recall": cmp.get("25D"),
            "delta": (cmp.get("25D9D") or 0) - (cmp.get("25D") or 0),
        }

    # ── OOF Gate Decision ──
    print("Computing gate decision...")
    gate_thresholds = protocol["oof_gate_thresholds"]
    gate_checks = {}
    catastrophic = False
    any_fail = False

    for mt in model_types:
        gate_checks[mt] = {}
        macro = per_route_metrics[mt]["macro"]

        # Macro grasp recall
        g_recall = macro["grasp"]["recall"] or 0
        g_min = gate_thresholds["macro_grasp_event_recall"]["min"]
        g_cat = gate_thresholds["macro_grasp_event_recall"]["catastrophic"]
        g_pass = g_recall >= g_min
        g_cat_flag = g_recall < g_cat
        gate_checks[mt]["macro_grasp_recall"] = {"value": g_recall, "min": g_min, "pass": g_pass, "catastrophic": g_cat_flag}
        if not g_pass: any_fail = True
        if g_cat_flag: catastrophic = True

        # Macro manipulation recall
        m_recall = macro["manipulation"]["recall"] or 0
        m_min = gate_thresholds["macro_manipulation_event_recall"]["min"]
        m_cat = gate_thresholds["macro_manipulation_event_recall"]["catastrophic"]
        m_pass = m_recall >= m_min
        m_cat_flag = m_recall < m_cat
        gate_checks[mt]["macro_manipulation_recall"] = {"value": m_recall, "min": m_min, "pass": m_pass, "catastrophic": m_cat_flag}
        if not m_pass: any_fail = True
        if m_cat_flag: catastrophic = True

        # Per-route recall minimum
        for route in SUPPORTED_ROUTES:
            route_min_recall = min(
                per_route_metrics[mt][route][h]["recall"] or 0
                for h in ["grasp", "manipulation", "release"]
            )
            r_min = gate_thresholds["per_route_recall_min"]["min"]
            r_cat = gate_thresholds["per_route_recall_min"]["catastrophic"]
            r_pass = route_min_recall >= r_min
            r_cat_flag = route_min_recall < r_cat
            gate_checks[mt][f"{route}_min_recall"] = {"value": route_min_recall, "min": r_min, "pass": r_pass, "catastrophic": r_cat_flag}
            if not r_pass: any_fail = True
            if r_cat_flag: catastrophic = True

        # Later-event recall
        for head in ["grasp", "manipulation", "release"]:
            lr = later_event_metrics[mt][head].get("later_event_recall") or 0
            le_min = gate_thresholds["later_event_recall"]["min"]
            le_cat = gate_thresholds["later_event_recall"]["catastrophic"]
            le_pass = lr >= le_min if later_event_metrics[mt][head]["later_event_count"] > 0 else True
            le_cat_flag = lr < le_cat and later_event_metrics[mt][head]["later_event_count"] > 0
            gate_checks[mt][f"{head}_later_event_recall"] = {"value": lr, "min": le_min, "pass": le_pass, "catastrophic": le_cat_flag}
            if not le_pass: any_fail = True
            if le_cat_flag: catastrophic = True

        # Release-overlap emit rate
        ro_rate = safety_metrics[mt]["release_overlap_emit_rate"]
        ro_max = gate_thresholds["release_overlap_emit_rate"]["max"]
        ro_cat = gate_thresholds["release_overlap_emit_rate"]["catastrophic"]
        ro_pass = ro_rate <= ro_max
        ro_cat_flag = ro_rate > ro_cat
        gate_checks[mt]["release_overlap_emit_rate"] = {"value": ro_rate, "max": ro_max, "pass": ro_pass, "catastrophic": ro_cat_flag}
        if not ro_pass: any_fail = True
        if ro_cat_flag: catastrophic = True

        # Unsupported-route emit
        us_count = safety_metrics[mt]["unsupported_route_emit_count"]
        us_pass = us_count == 0
        gate_checks[mt]["unsupported_route_emit"] = {"value": us_count, "max": 0, "pass": us_pass, "catastrophic": not us_pass}
        if not us_pass: any_fail = True; catastrophic = True

        # Background false emit rate
        bg_rate = safety_metrics[mt]["background_false_emit_rate"]
        bg_max = gate_thresholds["background_false_emit_rate"]["max"]
        bg_cat = gate_thresholds["background_false_emit_rate"]["catastrophic"]
        bg_pass = bg_rate <= bg_max
        bg_cat_flag = bg_rate > bg_cat
        gate_checks[mt]["background_false_emit_rate"] = {"value": bg_rate, "max": bg_max, "pass": bg_pass, "catastrophic": bg_cat_flag}
        if not bg_pass: any_fail = True
        if bg_cat_flag: catastrophic = True

    # ── Gate outcome ──
    if catastrophic:
        gate_outcome = "CATASTROPHIC_OOF_FAILURE"
    elif any_fail:
        gate_outcome = "HOLD_OOF_STUDENT"
    else:
        gate_outcome = "PASS_OOF_STUDENT"

    gate_decision = {
        "outcome": gate_outcome,
        "catastrophic": catastrophic,
        "any_failure": any_fail,
        "per_model_type": gate_checks,
        "frozen_thresholds": gate_thresholds,
    }

    # ── Write all artifacts ──
    print(f"Gate outcome: {gate_outcome}")
    print("Writing artifacts...")

    _atomic_text(staging / "per_run_metrics.json", json.dumps(per_run_metrics, indent=2))
    _atomic_text(staging / "per_head_metrics.json", json.dumps(per_head_metrics, indent=2))
    _atomic_text(staging / "per_route_metrics.json", json.dumps(per_route_metrics, indent=2))
    _atomic_text(staging / "later_event_metrics.json", json.dumps(later_event_metrics, indent=2))
    _atomic_text(staging / "safety_emit_metrics.json", json.dumps(safety_metrics, indent=2))
    _atomic_text(staging / "model_comparison_25d9d_vs_25d.json", json.dumps(model_comparison, indent=2))
    _atomic_text(staging / "oof_gate_decision.json", json.dumps(gate_decision, indent=2))

    # Evaluation protocol SHA binding
    proto_sha = sha256_file(args.eval_protocol)
    _atomic_text(staging / "evaluation_protocol.json", json.dumps({
        "protocol_path": str(args.eval_protocol),
        "protocol_sha": proto_sha,
        "protocol_schema": protocol["schema"],
    }, indent=2))

    # Checkpoint inventory
    checkpoint_list = []
    for mt in model_types:
        for fold in folds:
            for seed in seeds:
                checkpoint_list.append(f"{mt}/fold{fold}_seed{seed}")
    _atomic_text(staging / "checkpoint_inventory.json", json.dumps({
        "count": len(checkpoint_list), "checkpoints": checkpoint_list,
    }, indent=2))

    _atomic_text(staging / "source_binding.json", json.dumps({
        "predictions_base": str(pred_base),
        "eval_protocol_sha": proto_sha,
    }, indent=2))

    _atomic_text(staging / "environment.json", json.dumps({
        "python_version": platform.python_version(),
        "host": platform.node(),
    }, indent=2))

    write_seal(staging)
    os.replace(staging, out)

    print(f"\nEvaluation complete: {out}")
    print(f"Gate: {gate_outcome}")
    print("\n=== Gate Summary ===")
    for mt in model_types:
        print(f"\n{mt}:")
        for check_name, check in gate_checks[mt].items():
            status = "PASS" if check["pass"] else ("CATASTROPHIC" if check.get("catastrophic") else "FAIL")
            val = check.get("value", "?")
            if isinstance(val, float):
                val = f"{val:.4f}"
            print(f"  [{status}] {check_name}: {val}")


if __name__ == "__main__":
    main()
