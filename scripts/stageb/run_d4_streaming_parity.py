#!/usr/bin/env python3
"""D4.2a: Streaming first-trigger parity validation.

Step-by-step replay: reads one row at a time, maintains internal state,
detects CLOSE candidates causally, computes 16 features, scores with MLP,
and fires at τ=0.236312. Must match D4.1 batch results exactly.

CPU only. No future rows. No committed candidate table at inference.
Teacher-P only used for final comparison (not decision).
"""

import argparse, csv, hashlib, json, os, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features, TIE_TOLERANCE
from run_l12_e4c2b_repair import sha256_file
from run_d2_fresh_confirm import select_eligible_multi_traces

FROZEN_CHECKPOINT_SHA = "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
FROZEN_TAU = 0.236312


class StreamingDetector:
    """Causal streaming state machine for first-trigger detection."""

    def __init__(self, model, means, stdevs, impute, threshold=FROZEN_TAU):
        self.model = model
        self.means = means
        self.stdevs = stdevs
        self.impute = impute
        self.threshold = threshold

        # Internal state
        self.history = []  # list of record dicts (cumulative)
        self.prev_raw = None
        self.prev_gripper_valid = True
        self.close_streak = 0
        self.close_steps = []  # steps where CLOSE events detected
        self.open_steps = []  # steps where OPEN detected
        self.emit_step = -1
        self.emit_idx = -1
        self.candidate_features = []  # (step, features_dict) for audit

    def update(self, record):
        """Process one step. Returns None (no candidate) or (candidate_step, features_dict, mlp_score)."""
        self.history.append(record)
        step = len(self.history) - 1

        # Track OPEN
        if int(float(record.get("decoded_open_bool", 0))) == 1:
            self.open_steps.append(step)

        # Gripper validity
        gripper_valid = str(record.get("gripper_semantics_valid", "1")) not in ("0", "False", "false")
        raw_now = float(record.get("clean_gripper_raw", record.get("clean_gripper_raw_proxy", 0.5)))
        raw_valid = True  # assume valid for streaming

        # Raw crossing detection
        raw_crossing = False
        if self.prev_raw is not None and self.prev_gripper_valid and gripper_valid and raw_valid:
            if self.prev_raw > 0.5 and raw_now <= 0.5:
                raw_crossing = True

        # Close streak
        clean_close = int(float(record.get("clean_close", 0)))
        close_onset = int(float(record.get("close_onset", 0)))
        if clean_close:
            self.close_streak += 1
        else:
            self.close_streak = 0

        is_candidate = raw_crossing or bool(close_onset) or self.close_streak == 1

        self.prev_raw = raw_now
        self.prev_gripper_valid = gripper_valid

        if not is_candidate:
            return None

        # Record candidate
        self.close_steps.append(step)

        # Compute 16 features causally (same as rule_based_close_predictor at step)
        from gripper_attack.critical_close_selector import rule_based_close_predictor, PREDICTION_HORIZON
        preds = rule_based_close_predictor(self.history, horizon=PREDICTION_HORIZON, teacher_anchor=-1)
        pred = preds[step]

        features = {}
        for fn in FEATURE_NAMES:
            if fn == "total_score":
                features[fn] = round(pred.get("score", 0), 4)
            elif fn == "raw_crossing_bonus":
                features[fn] = pred.get("raw_crossing_bonus", "")
            elif fn == "close_streak_bonus":
                features[fn] = pred.get("close_streak_bonus", "")
            elif fn == "close_onset_qpos_bonus":
                features[fn] = pred.get("close_onset_qpos_bonus", "")
            elif fn == "eef_deceleration_bonus":
                features[fn] = pred.get("eef_deceleration_bonus", "")
            elif fn == "qpos_ready_bonus":
                features[fn] = pred.get("qpos_ready_bonus", "")
            elif fn == "eef_speed_now":
                features[fn] = pred.get("eef_speed_now", "")
            elif fn == "eef_speed_prev":
                features[fn] = pred.get("eef_speed_prev", "")
            elif fn == "eef_deceleration_delta":
                sn = pred.get("eef_speed_now", ""); sp = pred.get("eef_speed_prev", "")
                features[fn] = round(float(sn) - float(sp), 6) if sn != "" and sp != "" else ""
            elif fn == "close_streak":
                features[fn] = pred.get("close_streak_value", "")
            elif fn == "raw_crossing":
                features[fn] = int(pred.get("raw_open_to_close_crossing", 0))
            elif fn == "close_onset":
                features[fn] = int(pred.get("close_onset", 0))
            elif fn == "qpos":
                features[fn] = pred.get("qpos", "")
            elif fn == "time_since_prev_close":
                prevs = [s for s in self.close_steps[:-1]]  # exclude current
                features[fn] = step - max(prevs) if prevs else ""
            elif fn == "time_since_last_open":
                priors = [s for s in self.open_steps if s < step]
                features[fn] = step - max(priors) if priors else ""
            elif fn == "candidate_index":
                features[fn] = len(self.close_steps) - 1  # index before current

        # MLP score
        X = normalize_features([features], self.means, self.stdevs, self.impute)
        with torch.no_grad():
            score = float(self.model(X).item())

        # First-trigger check
        if self.emit_step < 0 and score >= self.threshold:
            self.emit_step = step
            self.emit_idx = len(self.close_steps) - 1

        self.candidate_features.append((step, features, round(score, 6)))
        return (step, features, round(score, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table-test", required=True)
    ap.add_argument("--candidate-table-fresh", required=True)
    ap.add_argument("--trace-status", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-fresh", default="")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--norm-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Verify checkpoint
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load D4.1 results for comparison
    d4_test_model = list(csv.DictReader(open("tables/d4_replay/d4_test21_model_predictions.csv")))
    d4_fresh_model = list(csv.DictReader(open("tables/d4_replay/d4_fresh25_model_predictions.csv")))

    # Load manifest for trace file paths
    manifest_map = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}
    fresh_manifest = args.manifest_fresh or args.manifest
    if fresh_manifest and os.path.exists(fresh_manifest):
        fr = list(csv.DictReader(open(fresh_manifest)))
        if "filename" in fr[0]:
            for r in fr:
                manifest_map[r["filename"].replace(".csv", "")] = r

    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}

    # Load batch candidates for comparison
    batch_test_cands = list(csv.DictReader(open(args.candidate_table_test)))
    batch_fresh_cands = list(csv.DictReader(open(args.candidate_table_fresh)))
    status_rows = list(csv.DictReader(open(args.trace_status)))
    fresh25_ids = set(select_eligible_multi_traces(batch_fresh_cands, status_rows).keys())

    # Build trace sets
    test_ids = {tid for tid in split if split[tid] == "test"}

    all_trace_ids = sorted(test_ids | fresh25_ids)
    print(f"Streaming replay: {len(all_trace_ids)} traces ({len(test_ids)} test + {len(fresh25_ids)} fresh)")

    parity_results = []
    all_match = True

    for tid in all_trace_ids:
        if tid not in manifest_map:
            print(f"  SKIP {tid}: not in manifest")
            continue

        fp = manifest_map[tid]["source_path"]
        from remap_v4_trace_for_l12 import remap_v4_to_l12
        rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)

        detector = StreamingDetector(model, means, stdevs, impute)
        streaming_candidates = []
        for step, record in enumerate(rows):
            result = detector.update(record)
            if result is not None:
                streaming_candidates.append(result)

        # D4.1 batch result
        if tid in test_ids:
            d4_row = next(r for r in d4_test_model if r["trace_id"] == tid)
        else:
            d4_row = next(r for r in d4_fresh_model if r["trace_id"] == tid)

        # Compare
        batch_emit = int(d4_row["emit_step"]) if d4_row["classification"] != "ABSTAIN" else -1
        stream_emit = detector.emit_step

        emit_match = (batch_emit == stream_emit)
        n_cand_match = (len(streaming_candidates) == int(d4_row["n_candidates"]))

        # Feature comparison
        batch_cands = [c for c in (batch_test_cands + batch_fresh_cands) if c["trace_id"] == tid]
        batch_cands_sorted = sorted(batch_cands, key=lambda x: int(x["candidate_step"]))

        feature_mismatches = 0
        for i, (s_step, s_feats, s_score) in enumerate(streaming_candidates):
            if i < len(batch_cands_sorted):
                bc = batch_cands_sorted[i]
                for fn in FEATURE_NAMES:
                    bv = bc.get(fn, ""); sv = str(s_feats.get(fn, ""))
                    try:
                        if abs(float(bv) - float(sv)) > 0.001: feature_mismatches += 1
                    except:
                        if bv != sv: feature_mismatches += 1

        ok = emit_match and n_cand_match and feature_mismatches == 0
        if not ok:
            all_match = False

        parity_results.append({
            "trace_id": tid, "task_key": rows[0].get("task_key", d4_row.get("task_key", "")) if rows else "",
            "stream_emit_step": stream_emit, "batch_emit_step": batch_emit,
            "emit_match": emit_match, "n_cand_match": n_cand_match,
            "stream_n_cands": len(streaming_candidates),
            "batch_n_cands": int(d4_row["n_candidates"]),
            "feature_mismatches": feature_mismatches,
        })

        status = "OK" if ok else "MISMATCH"
        print(f"  {tid}: {status} (emit={emit_match} cands={n_cand_match} feat_errs={feature_mismatches})")

    n_ok = sum(1 for r in parity_results if r["emit_match"] and r["n_cand_match"] and r["feature_mismatches"] == 0)
    print(f"\nParity: {n_ok}/{len(parity_results)} traces match")

    # Write
    with open(out / "d4_streaming_parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parity_results[0].keys())); w.writeheader(); w.writerows(parity_results)

    with open(out / "d4_streaming_run_log.txt", "w") as f:
        f.write(f"D4.2a RUN LOG\n")
        f.write(f"checkpoint_sha: {actual_ckpt}\n")
        f.write(f"threshold: {FROZEN_TAU}\n")
        f.write(f"n_traces: {len(parity_results)}\n")
        f.write(f"n_full_match: {n_ok}\n")
        f.write(f"all_match: {all_match}\n")
        f.write(f"runner_sha: {sha256_file(__file__)}\n")

    print(f"Output: {out}")
    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()
