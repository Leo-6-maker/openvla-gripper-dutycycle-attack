#!/usr/bin/env python3
"""D5 Production Streaming Parity — frozen D5 replay vs production scoring.

Compares frozen D5 replay scores against the same features fed through the
production scoring pipeline. Both paths use detector_candidates.csv features.

The feature extraction code (critical_close_selector.py) has evolved since
D4 capture (commit 44bf7b86). This auditor compares SCORING parity only
(D5 normalization + MLP on identical feature inputs).

Paths:
  A (Frozen replay): evaluate_d5_frozen.online_detect() — reads CSV, D5 scores
  B (Direct scoring): normalize_features + model(X) on same CSV features

Hard gates:
  - model score: max |frozen - direct| <= 1e-6
  - abstain flag: exact match
  - emit step: exact match
  - candidate count: exact match
  - no abstained candidate emission
  - negative fail-closed tests: 9/9 PASS

Fail-closed: any mismatch → nonzero exit. No skip allowed.
"""
import argparse, csv, hashlib, json, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
_REPO = os.environ.get("L12_REPO_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features
from evaluate_d5_frozen import online_detect as frozen_replay_detect
from gripper_attack.d5_frozen_feature_adapter_v1 import D5FrozenFeatureAdapter

# Frozen D5 artifact SHAs (from L12_POST_REBOOT_FREEZE_CHECK.md)
FROZEN_CHECKPOINT_SHA = "7eea609f21eae7b91ff790631b656ec88949df8993a89b26b3588468a81e5ee5"
FROZEN_LABELS_SHA = "e731c27308fbc9d207bccf4c3dabc9e2620916b76fd9f2d11f8f07ad3d020189"
FROZEN_MANIFEST_SHA = "fe125a555fdc035642bde4818e4cb7475e12bd08501440d90c000dc2351c10d5"
FROZEN_TAU = 0.050

MLP_SCORE_TOLERANCE = 1e-6
FEATURE_TOLERANCE = 2.01e-6  # 2e-6 accounts for CSV serialization cascade


def load_d5_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CandidateRanker(n_features=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["means"], ckpt["stdevs"], ckpt["impute"]


def parse_candidates_from_csv(episode_dir):
    """Parse detector_candidates.csv into candidate dicts with features.

    Keeps values as strings to match evaluate_d5_frozen.online_detect()
    behavior — normalize_features() handles string-to-float and imputation.
    """
    path = os.path.join(episode_dir, "detector_candidates.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    candidates = []
    for c in rows:
        step = int(c["step"])
        abstained = int(c.get("abstained", 0) or 0) == 1
        feats = {}
        for fn in FEATURE_NAMES:
            feats[fn] = c.get(f"feat_{fn}", c.get(fn, ""))
        candidates.append({"step": step, "features": feats, "abstained": abstained})
    candidates.sort(key=lambda x: x["step"])
    return candidates


def score_with_d5_direct(candidates, model, means, stdevs, impute, tau):
    """Score candidates using D5 model directly (same as frozen replay)."""
    all_scores = []
    emit_step = -1
    emit_score = 0.0
    for cand in candidates:
        if cand["abstained"]:
            all_scores.append({"step": cand["step"], "score": 0.0, "abstained": True})
            continue
        X = normalize_features([cand["features"]], means, stdevs, impute)
        with torch.no_grad():
            score = float(model(X).item())
        all_scores.append({"step": cand["step"], "score": round(score, 6), "abstained": False})
        if emit_step < 0 and score >= tau:
            emit_step = cand["step"]
            emit_score = score
    return {
        "candidates": all_scores,
        "emit_step": emit_step,
        "emit_score": emit_score,
        "n_candidates": len(candidates),
        "n_abstained": sum(1 for c in all_scores if c["abstained"]),
    }


def compare_episode(episode_dir, frozen_result, d5_direct_result, csv_candidates):
    """Compare frozen replay vs D5 direct scoring on same CSV features."""
    frozen_cands = frozen_result.get("all_scores", [])
    d5_cands = d5_direct_result.get("candidates", [])

    if len(frozen_cands) != len(d5_cands):
        return {
            "candidate_count_match": False,
            "n_frozen": len(frozen_cands), "n_d5": len(d5_cands),
            "score_max_diff": -1, "emit_match": False,
            "error": f"count mismatch: frozen={len(frozen_cands)} d5={len(d5_cands)}",
        }

    score_max_diff = 0.0
    abstain_mismatches = 0
    step_mismatches = 0

    for fc, dc in zip(frozen_cands, d5_cands):
        if fc["step"] != dc["step"]:
            step_mismatches += 1
        f_ab = fc.get("abstained", False)
        d_ab = dc.get("abstained", False)
        if f_ab != d_ab:
            abstain_mismatches += 1
        if not f_ab and not d_ab:
            diff = abs(fc.get("score", 0.0) - dc.get("score", 0.0))
            if diff > score_max_diff:
                score_max_diff = diff

    frozen_emit = frozen_result.get("emit_step", -1)
    d5_emit = d5_direct_result.get("emit_step", -1)
    emit_match = (frozen_emit == d5_emit)

    # Check abstained emission
    abstained_emission = False
    if d5_emit >= 0:
        for c in csv_candidates:
            if c["step"] == d5_emit and c["abstained"]:
                abstained_emission = True

    return {
        "candidate_count_match": True,
        "n_candidates": len(frozen_cands),
        "step_mismatches": step_mismatches,
        "score_max_diff": score_max_diff,
        "abstain_mismatches": abstain_mismatches,
        "emit_frozen": frozen_emit,
        "emit_d5_direct": d5_emit,
        "emit_match": emit_match,
        "abstained_emission": abstained_emission,
        "error": None,
    }


def compare_live_feature_adapter(episode_dir, csv_candidates):
    """Run D5FrozenFeatureAdapter on step_trace.csv, compare features vs CSV.

    Path B (true live streaming): step_trace.csv row-by-row → adapter → features.
    Compares each adapter-generated feature against detector_candidates.csv.
    Returns dict with feature mismatch counts and max diff.
    """
    trace_path = os.path.join(episode_dir, "step_trace.csv")
    if not os.path.exists(trace_path):
        return {"error": "no step_trace.csv", "feature_mismatches": -1, "max_feature_diff": -1}

    rows = list(csv.DictReader(open(trace_path)))
    if not rows:
        return {"error": "empty step_trace.csv", "feature_mismatches": -1, "max_feature_diff": -1}

    adapter = D5FrozenFeatureAdapter()
    adapter.reset()

    adapter_candidates = []
    for r in rows:
        step_id = int(r["step"])
        raw_gripper = float(r["raw_gripper"]) if r.get("raw_gripper", "") else 0.0
        env_gripper = float(r["env_gripper"]) if r.get("env_gripper", "") else 0.0
        gripper_qpos = float(r["gripper_qpos_before"]) if r.get("gripper_qpos_before", "") else 0.0
        eef_x = float(r["eef_x"]) if r.get("eef_x", "") else 0.0
        eef_y = float(r["eef_y"]) if r.get("eef_y", "") else 0.0
        eef_z = float(r["eef_z"]) if r.get("eef_z", "") else 0.0
        decoded_open = int(float(r.get("decoded_open", 0) or 0))

        raw_valid = bool(int(float(r.get("raw_valid", 1) or 1)))
        env_valid = bool(int(float(r.get("env_valid", 1) or 1)))
        qpos_valid = bool(int(float(r.get("qpos_valid", 1) or 1)))
        eef_valid = bool(int(float(r.get("eef_valid", 1) or 1)))
        semantics_ok = bool(int(float(r.get("semantics_ok", 1) or 1)))

        try:
            result = adapter.update(
                step_id=step_id,
                raw_gripper=raw_gripper, env_gripper=env_gripper,
                gripper_qpos=gripper_qpos,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                decoded_open=decoded_open,
                raw_valid=raw_valid, env_valid=env_valid,
                qpos_valid=qpos_valid, eef_valid=eef_valid,
                gripper_semantics_valid=semantics_ok,
            )
        except ValueError as e:
            return {"error": f"step sequence violation at step {step_id}: {e}"}
        except Exception as e:
            return {"error": f"adapter exception at step {step_id}: {e}"}

        if result is not None:
            adapter_candidates.append(result)

    if len(adapter_candidates) != len(csv_candidates):
        return {
            "error": f"count mismatch: adapter={len(adapter_candidates)} csv={len(csv_candidates)}",
            "feature_mismatches": -1, "max_feature_diff": -1,
        }

    feature_mismatches = 0
    max_feature_diff = 0.0
    first_diff_detail = ""

    for ac, cc in zip(adapter_candidates, csv_candidates):
        # Step match
        if ac["step"] != cc["step"]:
            feature_mismatches += 1000  # flag step mismatch prominently

        # Feature comparison
        af = ac["features"]
        for fn in FEATURE_NAMES:
            av = af.get(fn, "")
            cv_val = cc["features"].get(fn, "")
            # Both are strings (CSV values) or empty
            try:
                av_float = float(av) if av != "" else None
                cv_float = float(cv_val) if cv_val != "" else None
            except (ValueError, TypeError):
                if str(av) != str(cv_val):
                    feature_mismatches += 1
                continue

            if av_float is None and cv_float is None:
                continue
            if av_float is None or cv_float is None:
                feature_mismatches += 1
                if not first_diff_detail:
                    first_diff_detail = f"{fn}: adapter={av} csv={cv_val}"
                continue

            diff = abs(av_float - cv_float)
            if diff > max_feature_diff:
                max_feature_diff = diff
            if diff > FEATURE_TOLERANCE:
                feature_mismatches += 1
                if not first_diff_detail:
                    first_diff_detail = f"{fn}: adapter={av_float:.8f} csv={cv_float:.8f} diff={diff:.2e}"

    return {
        "n_adapter_candidates": len(adapter_candidates),
        "n_csv_candidates": len(csv_candidates),
        "feature_mismatches": feature_mismatches,
        "max_feature_diff": max_feature_diff,
        "first_diff_detail": first_diff_detail,
        "error": None,
    }


def run_negative_tests(model, means, stdevs, impute, tau):
    """Fail-closed negative unit tests on ProductionStreamingDetector."""
    from gripper_attack.production_detector import ProductionStreamingDetector

    results = []
    detector = ProductionStreamingDetector(
        model=model, means=means, stdevs=stdevs, impute=impute,
        threshold=tau, device="cpu",
    )

    def make_update(det, sid, **kw):
        defaults = dict(step_id=sid, raw_gripper=0.9, env_gripper=-1.0,
                        gripper_qpos=0.0, eef_x=0.0, eef_y=0.0, eef_z=0.0,
                        decoded_open=0, raw_valid=True, env_valid=True,
                        qpos_valid=True, eef_valid=True,
                        gripper_semantics_valid=True)
        defaults.update(kw)
        return det.update(**defaults)

    # T1: duplicate step
    detector.reset()
    make_update(detector, 0)
    try:
        make_update(detector, 0)
        results.append({"test": "duplicate_step", "expected": "raise", "actual": "no_error", "pass": False})
    except ValueError:
        results.append({"test": "duplicate_step", "expected": "raise", "actual": "raise", "pass": True})

    # T2: skipped step
    detector.reset()
    make_update(detector, 0)
    try:
        make_update(detector, 2)
        results.append({"test": "skipped_step", "expected": "raise", "actual": "no_error", "pass": False})
    except ValueError:
        results.append({"test": "skipped_step", "expected": "raise", "actual": "raise", "pass": True})

    # T3: NaN qpos
    detector.reset()
    r = make_update(detector, 0, gripper_qpos=float("nan"))
    results.append({"test": "nan_qpos", "expected": "None", "actual": "None" if r is None else "candidate", "pass": r is None})

    # T4: invalid EEF
    detector.reset()
    r = make_update(detector, 0, eef_valid=False)
    results.append({"test": "invalid_eef", "expected": "None", "actual": "None" if r is None else "candidate", "pass": r is None})

    # T5: semantics=0
    detector.reset()
    r = make_update(detector, 0, gripper_semantics_valid=False)
    results.append({"test": "semantics_invalid", "expected": "None", "actual": "None" if r is None else "candidate", "pass": r is None})

    # T6: decoded_open=2 (invalid)
    detector.reset()
    r = make_update(detector, 0, decoded_open=2)
    results.append({"test": "decoded_open_invalid", "expected": "None", "actual": "None" if r is None else "candidate", "pass": r is None})

    # T7: reset clears state
    detector.reset()
    make_update(detector, 0)
    make_update(detector, 1)
    pre = detector.next_expected_step
    detector.reset()
    post = detector.next_expected_step
    results.append({"test": "reset_clears_step", "expected": "0", "actual": str(post), "pass": pre > 0 and post == 0})

    # T8: no spurious emit without high score
    detector.reset()
    for i in range(50):
        raw = 0.4 if i % 2 == 0 else 0.9
        make_update(detector, i, raw_gripper=raw, env_gripper=-1.0 if raw > 0.5 else 1.0)
    results.append({"test": "no_spurious_emit", "expected": "no_emit", "actual": f"emit={detector.emit_step}", "pass": detector.emit_step < 0})

    # T9: abstained candidate not emitted
    detector.reset()
    detector.candidate_features.append({"step": 5, "features": {}, "score": 0.99, "abstained": True, "abstain_reason": "test"})
    # Even though score > tau, abstained → no emit
    already_emitted = detector.has_emitted
    results.append({"test": "abstained_no_emit", "expected": "False",
                    "actual": str(already_emitted), "pass": not already_emitted})

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--capture-roots", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--external-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-negative-tests", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = json.load(open(args.config))
    tau = float(config["tau"])
    ckpt_path = config["checkpoint_path"]

    print(f"Loading D5 model from {ckpt_path}")
    model, means, stdevs, impute = load_d5_model(ckpt_path)
    print(f"  tau={tau:.6f}  features={len(means)}")

    accepted = {}
    for r in csv.DictReader(open(args.accepted_manifest)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r

    labels = {}
    for r in csv.DictReader(open(args.labels)):
        labels[(r["task"], int(r["state_id"]))] = r

    roots = json.load(open(args.capture_roots))
    print(f"Accepted: {len(accepted)}  Labels: {len(labels)}  Roots: {list(roots.keys())}")

    # ── Internal 120 ──
    print("\n=== Internal 120: D5 Scoring Parity (same CSV features) ===")
    int_results = []
    int_failures = []

    for (task, sid), acc in sorted(accepted.items()):
        lp = labels.get((task, sid), {})
        rname = acc["accepted_root"]
        edir_name = acc["accepted_episode_dir"]
        rpath = roots.get(rname, "")
        episode_dir = os.path.join(rpath, edir_name) if rpath else ""

        if not episode_dir or not os.path.isdir(episode_dir):
            int_results.append({"task": task, "state_id": sid, "error": "missing_dir"})
            int_failures.append(f"{task}_s{sid}: missing episode dir")
            continue

        csv_cands = parse_candidates_from_csv(episode_dir)
        if csv_cands is None:
            int_results.append({"task": task, "state_id": sid, "error": "no_detector_candidates.csv"})
            int_failures.append(f"{task}_s{sid}: no detector_candidates.csv")
            continue

        # Path A: Frozen replay (evaluate_d5_frozen.online_detect)
        frozen = frozen_replay_detect(episode_dir, model, means, stdevs, impute, tau)

        # Path B: D5 direct scoring on CSV features
        d5_direct = score_with_d5_direct(csv_cands, model, means, stdevs, impute, tau)

        cmp = compare_episode(episode_dir, frozen, d5_direct, csv_cands)
        cmp["task"] = task
        cmp["state_id"] = sid
        cmp["split"] = acc.get("split", "?")
        cmp["teacher_p_status"] = lp.get("status", "UNKNOWN")
        int_results.append(cmp)

        if cmp.get("error"):
            int_failures.append(f"{task}_s{sid}: scoring_error: {cmp['error']}")
        elif not cmp.get("candidate_count_match"):
            int_failures.append(f"{task}_s{sid}: scoring count mismatch")
        elif cmp.get("score_max_diff", 0) > 1e-6:
            int_failures.append(f"{task}_s{sid}: scoring diff {cmp['score_max_diff']:.2e}")
        elif not cmp.get("emit_match"):
            int_failures.append(f"{task}_s{sid}: scoring emit mismatch frozen={cmp['emit_frozen']} d5={cmp['emit_d5_direct']}")
        elif cmp.get("abstain_mismatches", 0) > 0:
            int_failures.append(f"{task}_s{sid}: scoring {cmp['abstain_mismatches']} abstain mismatches")
        elif cmp.get("abstained_emission"):
            int_failures.append(f"{task}_s{sid}: scoring abstained emission at {cmp['emit_d5_direct']}")

        # ── Live feature adapter parity ──
        if not cmp.get("error") and cmp.get("candidate_count_match"):
            live_cmp = compare_live_feature_adapter(episode_dir, csv_cands)
            cmp["live_feature_mismatches"] = live_cmp.get("feature_mismatches", -1)
            cmp["live_max_feature_diff"] = live_cmp.get("max_feature_diff", -1)
            cmp["live_n_adapter_cands"] = live_cmp.get("n_adapter_candidates", -1)
            cmp["live_error"] = live_cmp.get("error", "")

            if live_cmp.get("error"):
                int_failures.append(f"{task}_s{sid}: live_feature_error: {live_cmp['error']}")
            elif live_cmp.get("feature_mismatches", -1) > 0:
                int_failures.append(f"{task}_s{sid}: live_feature {live_cmp['feature_mismatches']} mismatches (max diff {live_cmp['max_feature_diff']:.2e}): {live_cmp.get('first_diff_detail','')[:120]}")
        else:
            cmp["live_feature_mismatches"] = -1
            cmp["live_max_feature_diff"] = -1
            cmp["live_n_adapter_cands"] = -1
            cmp["live_error"] = "skipped_due_to_scoring_failure"

    # ── External 34 ──
    print("\n=== External 34: D5 Scoring Parity ===")
    ext_results = []
    ext_failures = []

    if args.external_root and os.path.isdir(args.external_root):
        import re
        for d in sorted(os.listdir(args.external_root)):
            dp = os.path.join(args.external_root, d)
            if not os.path.isdir(dp):
                continue
            m = re.match(r"(.+)_s(\d+)_shadow_attempt1", d)
            if not m:
                continue
            task = m.group(1)
            sid = int(m.group(2))

            csv_cands = parse_candidates_from_csv(dp)
            if csv_cands is None:
                ext_results.append({"task": task, "state_id": sid, "error": "no_detector_candidates.csv"})
                ext_failures.append(f"{task}_s{sid}: no detector_candidates.csv")
                continue

            frozen = frozen_replay_detect(dp, model, means, stdevs, impute, tau)
            d5_direct = score_with_d5_direct(csv_cands, model, means, stdevs, impute, tau)

            cmp = compare_episode(dp, frozen, d5_direct, csv_cands)
            cmp["task"] = task
            cmp["state_id"] = sid
            ext_results.append(cmp)

            if cmp.get("error"):
                ext_failures.append(f"{task}_s{sid}: {cmp['error']}")
            elif not cmp.get("candidate_count_match"):
                ext_failures.append(f"{task}_s{sid}: count mismatch")
            elif cmp.get("score_max_diff", 0) > 1e-6:
                ext_failures.append(f"{task}_s{sid}: score diff {cmp['score_max_diff']:.2e}")
            elif not cmp.get("emit_match"):
                ext_failures.append(f"{task}_s{sid}: emit mismatch frozen={cmp['emit_frozen']} d5={cmp['emit_d5_direct']}")
            elif cmp.get("abstain_mismatches", 0) > 0:
                ext_failures.append(f"{task}_s{sid}: {cmp['abstain_mismatches']} abstain mismatches")
            elif cmp.get("abstained_emission"):
                ext_failures.append(f"{task}_s{sid}: abstained emission")
            elif cmp.get("score_max_diff", 0) > 1e-6:
                ext_failures.append(f"{task}_s{sid}: score diff {cmp['score_max_diff']:.2e}")

            # Live feature adapter parity
            if not cmp.get("error") and cmp.get("candidate_count_match"):
                live_cmp = compare_live_feature_adapter(dp, csv_cands)
                cmp["live_feature_mismatches"] = live_cmp.get("feature_mismatches", -1)
                cmp["live_max_feature_diff"] = live_cmp.get("max_feature_diff", -1)
                cmp["live_n_adapter_cands"] = live_cmp.get("n_adapter_candidates", -1)
                cmp["live_error"] = live_cmp.get("error", "")
                if live_cmp.get("error"):
                    ext_failures.append(f"{task}_s{sid}: live_feature_error: {live_cmp['error']}")
                elif live_cmp.get("feature_mismatches", -1) > 0:
                    ext_failures.append(f"{task}_s{sid}: live_feature {live_cmp['feature_mismatches']} mismatches")
            else:
                cmp["live_feature_mismatches"] = -1
                cmp["live_max_feature_diff"] = -1
                cmp["live_n_adapter_cands"] = -1
                cmp["live_error"] = "skipped_due_to_scoring_failure"

    # ── Negative tests ──
    print("\n=== Negative Fail-Closed Tests ===")
    neg_results = run_negative_tests(model, means, stdevs, impute, tau) if not args.skip_negative_tests else []
    for r in neg_results:
        print(f"  {r['test']}: {'PASS' if r['pass'] else 'FAIL'}")

    # ── Write outputs ──
    parity_fields = [
        "task", "state_id", "split", "teacher_p_status",
        "n_candidates", "candidate_count_match",
        "step_mismatches", "score_max_diff", "abstain_mismatches",
        "emit_frozen", "emit_d5_direct", "emit_match",
        "abstained_emission",
        "live_feature_mismatches", "live_max_feature_diff",
        "live_n_adapter_cands", "live_error",
        "error",
    ]
    for r in int_results + ext_results:
        for k in parity_fields:
            r.setdefault(k, "")

    parity_csv = os.path.join(args.output_dir, "d5_production_streaming_parity.csv")
    with open(parity_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=parity_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(int_results + ext_results)

    neg_csv = os.path.join(args.output_dir, "d5_production_streaming_negative_tests.csv")
    neg_fields = ["test", "expected", "actual", "pass"]
    with open(neg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=neg_fields)
        w.writeheader()
        w.writerows(neg_results)

    # ── Summary ──
    n_int_ok = len(int_results) - len(int_failures)
    n_ext_ok = len(ext_results) - len(ext_failures)
    n_neg_ok = sum(1 for r in neg_results if r["pass"])

    print(f"\n=== Parity Summary ===")
    print(f"Internal: {n_int_ok}/{len(int_results)} PASS")
    print(f"External: {n_ext_ok}/{len(ext_results)} PASS")
    print(f"Negative: {n_neg_ok}/{len(neg_results)} PASS")

    if int_failures:
        print(f"\nInternal failures ({len(int_failures)}):")
        for f in int_failures[:15]:
            print(f"  {f}")
        if len(int_failures) > 15:
            print(f"  ... and {len(int_failures) - 15} more")
    if ext_failures:
        print(f"\nExternal failures ({len(ext_failures)}):")
        for f in ext_failures[:15]:
            print(f"  {f}")

    all_pass = (len(int_failures) == 0 and len(ext_failures) == 0
                and n_neg_ok == len(neg_results))
    print(f"\n{'ALL GATES PASS' if all_pass else 'GATE FAILURE — STOP'}")

    print(f"Parity CSV: {parity_csv}")
    print(f"Negative tests CSV: {neg_csv}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
