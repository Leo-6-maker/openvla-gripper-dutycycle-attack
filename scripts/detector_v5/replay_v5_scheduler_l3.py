#!/usr/bin/env python3
"""V5 scheduler L3 replay for exact-W32 inner-CV predictions.

Status: TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY

Blockers for authoritative L3:
  RUNTIME_CANDIDATE_CLOSE  = BLOCKED (policy raw_gripper not in prediction bundle)
  UTILITY_REGRASP_SEMANTICS = BLOCKED (utility=grasp, regrasp=manipulation unvalidated)
  PROPER_INNER_TRAIN_PLATT  = BLOCKED (inner-train inference not generated)
  K10_CONTAINMENT           = NOT_MEASURED (no frozen K10 label in prediction)

Do NOT interpret output as authoritative false-start rate.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def __sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            d.update(chunk)
    return d.hexdigest()


def _sig(z):
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ── Data loading ──

def load_steps(prediction_dir):
    path = Path(prediction_dir) / "heldout_step_predictions.jsonl"
    steps = []
    with open(path) as f:
        for line in f:
            steps.append(json.loads(line))
    return steps


def group_episodes(steps):
    eps = defaultdict(list)
    for s in steps:
        eps[s["canonical_parent_key"]].append(s)
    return dict(eps)


def load_manifest(manifest_path):
    if manifest_path is None:
        return None
    return json.loads(Path(manifest_path).read_text())


# ── Opportunity corridors ──

def find_teacher_event_corridors(ep_steps, min_length=8):
    """Find contiguous blocks of route_supported AND event_id >= 0 steps.

    This is TEACHER_EVENT_CORRIDOR_PROXY — NOT a frozen K10 label.
    """
    ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
    corridors = []
    i = 0
    while i < len(ep_sorted):
        s = ep_sorted[i]
        if s.get("route_supported") and s.get("event_id", -1) >= 0:
            start = s["step_index"]
            j = i
            while (j < len(ep_sorted)
                   and ep_sorted[j].get("route_supported")
                   and ep_sorted[j].get("event_id", -1) >= 0
                   and ep_sorted[j]["step_index"] == start + (j - i)):
                j += 1
            length = j - i
            if length >= min_length:
                corridors.append({
                    "start": start, "end": ep_sorted[j - 1]["step_index"],
                    "length": length,
                    "source": "teacher_event_proxy",
                })
            i = j
        else:
            i += 1
    return corridors


# ── Scheduler replay ──

def replay_scheduler(episodes, calibrator, candidate_close_fn):
    """Replay V5 scheduler.

    calibrator: dict with optional 'grasp_a','grasp_b','release_a','release_b'
                If None, uses raw probabilities (a=1, b=0).
    candidate_close_fn: callable(step) -> bool, runtime-causal candidate gate.
    """
    config = V5SchedulerConfig()

    if calibrator is None:
        a_g, b_g = 1.0, 0.0
        a_r, b_r = 1.0, 0.0
    else:
        a_g = calibrator.get("grasp_a", 1.0)
        b_g = calibrator.get("grasp_b", 0.0)
        a_r = calibrator.get("release_a", 1.0)
        b_r = calibrator.get("release_b", 0.0)

    results = {}
    for ep_key, ep_steps in sorted(episodes.items()):
        ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
        scheduler = V5OneShotScheduler(config)
        final = {}

        for s in ep_sorted:
            cc = candidate_close_fn(s)
            gp = s.get("grasp_prob", 0)
            rp = s.get("release_prob", 0)
            mp = s.get("manipulation_prob", 0)
            if calibrator is not None:
                if 0 < gp < 1: gp = _sig(a_g * math.log(gp / (1 - gp)) + b_g)
                if 0 < rp < 1: rp = _sig(a_r * math.log(rp / (1 - rp)) + b_r)

            final = scheduler.update(
                step=s["step_index"],
                candidate_close=cc,
                valid=s["route_supported"],
                utility_probability=gp,
                release_probability=rp,
                regrasp_probability=mp,
                uncertainty_probability=0.0,
            )

        results[ep_key] = {
            "emitted": final["one_shot_emitted"],
            "emit_step": final["emit_step"],
            "final_state": final["state"],
        }
    return results


# ── Metric computation ──

def compute_l3_metrics(episodes, scheduler_results, min_corridor_len=8):
    """Decomposed L3 metrics with strict numerator/denominator separation."""
    n_total = len(episodes)

    n_negative = 0       # no valid corridor
    n_positive = 0       # has >=1 valid corridor

    neg_emits = 0
    pos_on_corridor = 0
    pos_off_corridor = 0
    pos_abstain = 0

    for ep_key, ep_steps in sorted(episodes.items()):
        result = scheduler_results.get(ep_key, {})
        emitted = result.get("emitted", False)
        emit_step = result.get("emit_step", -1)
        corridors = find_teacher_event_corridors(ep_steps, min_corridor_len)

        if not corridors:
            n_negative += 1
            if emitted:
                neg_emits += 1
        else:
            n_positive += 1
            if emitted:
                on_corridor = any(c["start"] <= emit_step <= c["end"] for c in corridors)
                if on_corridor:
                    pos_on_corridor += 1
                else:
                    pos_off_corridor += 1
            else:
                pos_abstain += 1

    total_emitted = neg_emits + pos_on_corridor + pos_off_corridor

    # Verify: total_emitted must equal sum of all emit counts
    emitted_from_results = sum(1 for v in scheduler_results.values() if v.get("emitted"))
    assert total_emitted == emitted_from_results, \
        f"total_emitted={total_emitted} != emitted_from_results={emitted_from_results}"

    return {
        "total_episodes": n_total,
        "negative_episodes": n_negative,
        "positive_episodes": n_positive,
        "negative_episode_emits": neg_emits,
        "positive_on_corridor_emits": pos_on_corridor,
        "positive_off_corridor_emits": pos_off_corridor,
        "positive_abstentions": pos_abstain,
        "total_emitted_episodes": total_emitted,

        "negative_episode_false_start_rate":
            neg_emits / max(1, n_negative) if n_negative > 0 else 0.0,
        "positive_episode_off_corridor_rate":
            pos_off_corridor / max(1, n_positive) if n_positive > 0 else 0.0,
        "valid_opportunity_recall":
            pos_on_corridor / max(1, n_positive) if n_positive > 0 else 0.0,
        "emit_precision":
            pos_on_corridor / max(1, total_emitted) if total_emitted > 0 else 0.0,
        "abstention_rate":
            pos_abstain / max(1, n_positive) if n_positive > 0 else 0.0,
        "invalid_emit_fraction":
            (neg_emits + pos_off_corridor) / max(1, total_emitted) if total_emitted > 0 else 0.0,
        "overall_emit_rate":
            total_emitted / max(1, n_total),
    }


# ── Known-mask-aware background emit (3 variants) ──

def compute_legacy_background_emit(steps, tau=0.5):
    """A. Legacy: matches original evaluator — any known head on bg steps."""
    bg_total = 0
    bg_emit = 0
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported"):
            gk = s.get("grasp_known_mask", False)
            mk = s.get("manipulation_known_mask", False)
            rk = s.get("release_known_mask", False)
            if gk or mk or rk:
                bg_total += 1
            if ((gk and s.get("grasp_prob", 0) >= tau) or
                (mk and s.get("manipulation_prob", 0) >= tau) or
                (rk and s.get("release_prob", 0) >= tau)):
                bg_emit += 1
    return bg_emit / max(1, bg_total), bg_total


def compute_head_conditional_emit(steps, head, tau=0.5):
    """B. Head-conditional: denominator = background AND this head known."""
    known_count = 0
    emit_count = 0
    km_key = f"{head}_known_mask"
    prob_key = f"{head}_prob"
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported"):
            if s.get(km_key, False):
                known_count += 1
                if s.get(prob_key, 0) >= tau:
                    emit_count += 1
    return emit_count / max(1, known_count), known_count


def compute_any_known_head_emit(steps, tau=0.5):
    """C. Any-known-head union: denominator = bg AND at least one head known."""
    bg_any_known = 0
    bg_emit = 0
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported"):
            gk = s.get("grasp_known_mask", False)
            mk = s.get("manipulation_known_mask", False)
            rk = s.get("release_known_mask", False)
            any_k = gk or mk or rk
            if any_k:
                bg_any_known += 1
                if ((gk and s.get("grasp_prob", 0) >= tau) or
                    (mk and s.get("manipulation_prob", 0) >= tau) or
                    (rk and s.get("release_prob", 0) >= tau)):
                    bg_emit += 1
    return bg_emit / max(1, bg_any_known), bg_any_known


# ── Manifest validation ──

def validate_calibration_manifest(calib_fit_manifest, heldout_manifest):
    """Fail-closed if calibration fit identities intersect heldout identities."""
    if calib_fit_manifest is None or heldout_manifest is None:
        return True  # can't validate without manifests — caller should warn
    fit_ids = set(calib_fit_manifest.get("fit_identities", []))
    heldout_ids = set(heldout_manifest.get("heldout_identities", []))
    if not fit_ids.isdisjoint(heldout_ids):
        violators = fit_ids & heldout_ids
        raise RuntimeError(
            f"CALIBRATION_LEAKAGE: {len(violators)} identities in both fit and heldout: {sorted(violators)[:5]}..."
        )
    return True


# ── Main ──

def main():
    ap = argparse.ArgumentParser(
        description="V5 scheduler L3 replay (TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY)")
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--prediction-manifest", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--calibration-bundle", type=Path, default=None)
    ap.add_argument("--calibration-fit-manifest", type=Path, default=None)
    ap.add_argument("--opportunity-source", type=str, default="teacher_event_proxy",
                    choices=["teacher_event_proxy"],
                    help="Source for valid opportunity corridors")
    ap.add_argument("--min-corridor-length", type=int, default=8)
    ap.add_argument("--scheduler-config", type=Path, default=None)
    args = ap.parse_args()

    pred_root = args.prediction_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate calibration manifest isolation
    heldout_manifest = load_manifest(args.prediction_manifest)
    calib_fit_manifest = load_manifest(args.calibration_fit_manifest)
    if calib_fit_manifest is not None and heldout_manifest is not None:
        validate_calibration_manifest(calib_fit_manifest, heldout_manifest)

    # Load calibrator
    calibrator = None
    calib_bundle_sha = None
    calib_fit_hash = None
    if args.calibration_bundle is not None:
        cb = Path(args.calibration_bundle)
        if not cb.is_file():
            raise SystemExit(f"Calibration bundle not found: {cb}")
        calib_data = json.loads(cb.read_text())
        calib_bundle_sha = _sha256_file(cb)
        calibrator = {}
        for entry in calib_data:
            h = entry.get("head", "")
            if h == "grasp":
                calibrator["grasp_a"] = entry.get("a", 1.0)
                calibrator["grasp_b"] = entry.get("b", 0.0)
            elif h == "release":
                calibrator["release_a"] = entry.get("a", 1.0)
                calibrator["release_b"] = entry.get("b", 0.0)
        if calib_fit_manifest is not None:
            calib_fit_hash = _sha256_file(Path(args.calibration_fit_manifest))

    # Artifact identities
    script_sha = _sha256_file(Path(__file__))
    scheduler_sha = _sha256_file(ROOT / "src/gripper_attack/v5_scheduler.py")
    source_commit = None
    try:
        import subprocess
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass

    pred_manifest_sha = _sha256_file(Path(args.prediction_manifest)) if args.prediction_manifest else None

    # ── Runtime-causal candidate_close ──
    # BLOCKED: policy raw_gripper not in prediction bundle.
    # Using TEACHER_EVENT_PROXY as placeholder.
    def candidate_close_fn(step):
        return step.get("route_supported", False) and step.get("event_id", -1) >= 0

    candidate_close_status = "TEACHER_EVENT_PROXY_NOT_RUNTIME_CAUSAL"

    # Discover splits
    splits = sorted(
        d.name for d in pred_root.iterdir()
        if d.is_dir() and d.name.startswith("predict_V2B_EXACT_")
    )
    if not splits:
        raise SystemExit(f"No predict_V2B_EXACT_* directories in {pred_root}")

    header = f"{'split':<10} {'neg_ep':>6} {'pos_ep':>6} {'neg_emit':>8} {'pos_on':>7} {'pos_off':>7} {'abstain':>7} {'total_emit':>10} {'neg_fpr':>8} {'pos_off':>8} {'recall':>8} {'prec':>8} {'abst':>8} {'inv_emit':>8}"
    print(f"V5 Scheduler L3 Replay — {candidate_close_status}")
    print(f"  opportunity_source: {args.opportunity_source}")
    print(f"  scheduler SHA: {scheduler_sha[:16]}")
    print(f"  script SHA: {script_sha[:16]}")
    print(header)
    print("-" * len(header))

    all_metrics = {}
    csv_rows = []

    for sl in splits:
        pred_dir = pred_root / sl
        pred_file = pred_dir / "heldout_step_predictions.jsonl"
        if not pred_file.is_file():
            continue

        steps = load_steps(pred_dir)
        episodes = group_episodes(steps)
        short = sl.replace("predict_V2B_EXACT_W32_H64_D0.1_WD1e-4_", "")

        sched_results = replay_scheduler(episodes, calibrator, candidate_close_fn)
        m = compute_l3_metrics(episodes, sched_results, args.min_corridor_length)
        all_metrics[short] = m

        print(f"{short:<10} {m['negative_episodes']:>6} {m['positive_episodes']:>6} "
              f"{m['negative_episode_emits']:>8} {m['positive_on_corridor_emits']:>7} "
              f"{m['positive_off_corridor_emits']:>7} {m['positive_abstentions']:>7} "
              f"{m['total_emitted_episodes']:>10} "
              f"{m['negative_episode_false_start_rate']:>8.4f} "
              f"{m['positive_episode_off_corridor_rate']:>8.4f} "
              f"{m['valid_opportunity_recall']:>8.4f} "
              f"{m['emit_precision']:>8.4f} "
              f"{m['abstention_rate']:>8.4f} "
              f"{m['invalid_emit_fraction']:>8.4f}")

        csv_rows.append({
            "split": short,
            "negative_episodes": m["negative_episodes"],
            "positive_episodes": m["positive_episodes"],
            "negative_episode_emits": m["negative_episode_emits"],
            "positive_on_corridor_emits": m["positive_on_corridor_emits"],
            "positive_off_corridor_emits": m["positive_off_corridor_emits"],
            "positive_abstentions": m["positive_abstentions"],
            "total_emitted_episodes": m["total_emitted_episodes"],
            "negative_false_start_rate": round(m["negative_episode_false_start_rate"], 6),
            "positive_off_corridor_rate": round(m["positive_episode_off_corridor_rate"], 6),
            "valid_opportunity_recall": round(m["valid_opportunity_recall"], 6),
            "emit_precision": round(m["emit_precision"], 6),
            "abstention_rate": round(m["abstention_rate"], 6),
            "invalid_emit_fraction": round(m["invalid_emit_fraction"], 6),
            "overall_emit_rate": round(m["overall_emit_rate"], 6),
        })

    # Write outputs
    manifest = {
        "replay_status": "TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY",
        "authoritative_l3": False,
        "candidate_close_status": candidate_close_status,
        "utility_mapping_status": "UNVALIDATED_SEMANTIC_PROXY (grasp_prob)",
        "release_mapping_status": "DIRECT (release_prob)",
        "regrasp_mapping_status": "UNVALIDATED_SEMANTIC_PROXY (manipulation_prob)",
        "opportunity_label_status": f"TEACHER_EVENT_PROXY (source={args.opportunity_source})",
        "calibration_status": "NON_AUTHORITATIVE" if calibrator is None else "NON_AUTHORITATIVE",
        "k10_containment_status": "NOT_MEASURED",
        "script_sha256": script_sha,
        "scheduler_sha256": scheduler_sha,
        "scheduler_source": "src/gripper_attack/v5_scheduler.py",
        "prediction_manifest_sha256": pred_manifest_sha,
        "calibration_bundle_sha256": calib_bundle_sha,
        "calibration_fit_identity_hash": calib_fit_hash,
        "source_commit": source_commit,
        "opportunity_source": args.opportunity_source,
        "min_corridor_length": args.min_corridor_length,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_splits": len(splits),
        "blockers": [
            "RUNTIME_CANDIDATE_CLOSE: policy raw_gripper not in prediction bundle",
            "UTILITY_REGRASP_SEMANTICS: utility=grasp, regrasp=manipulation unvalidated",
            "PROPER_INNER_TRAIN_PLATT: inner-train inference not generated",
            "K10_CONTAINMENT: no frozen K10 label in prediction",
        ],
    }

    with open(out_root / "v5_scheduler_l3_replay.json", "w") as f:
        json.dump({**manifest, "per_split_metrics": all_metrics}, f, indent=2)

    with open(out_root / "v5_scheduler_l3_per_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)

    # Also write known-mask denominator audit for first split
    if splits:
        steps = load_steps(pred_root / splits[0])
        leg_rate, leg_n = compute_legacy_background_emit(steps)
        g_cond_rate, g_known_n = compute_head_conditional_emit(steps, "grasp")
        m_cond_rate, m_known_n = compute_head_conditional_emit(steps, "manipulation")
        r_cond_rate, r_known_n = compute_head_conditional_emit(steps, "release")
        any_rate, any_n = compute_any_known_head_emit(steps)

        known_denom_audit = {
            "legacy_background_emit_rate": {"rate": leg_rate, "denominator": leg_n,
                                             "description": "A. Original Stage-1 evaluator: all supported bg steps"},
            "head_conditional_grasp": {"rate": g_cond_rate, "denominator": g_known_n,
                                        "description": "B. bg AND grasp_known"},
            "head_conditional_manipulation": {"rate": m_cond_rate, "denominator": m_known_n,
                                               "description": "B. bg AND manipulation_known"},
            "head_conditional_release": {"rate": r_cond_rate, "denominator": r_known_n,
                                          "description": "B. bg AND release_known"},
            "any_known_head": {"rate": any_rate, "denominator": any_n,
                                "description": "C. bg AND at least one head known"},
        }
        with open(out_root / "known_mask_denominator_audit.json", "w") as f:
            json.dump(known_denom_audit, f, indent=2)

    print(f"\nOutputs: {out_root}")
    print(f"Status: {manifest['replay_status']}")


if __name__ == "__main__":
    main()
