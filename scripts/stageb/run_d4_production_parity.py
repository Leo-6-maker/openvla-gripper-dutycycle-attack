#!/usr/bin/env python3
"""D4.2c: Production streaming adapter parity verification (fail-closed repair).

Feeds RAW trace fields (step_id, raw/env gripper, qpos, EEF xyz, decoded_open)
into ProductionStreamingDetector. Compares candidate_step, 16 raw features,
16 normalized features, MLP score, and emit_step with D4.1 batch results.

Hard gates:
  - normalized-feature parity:  max |prod - batch| <= 1e-7
  - MLP score parity:           max |prod - batch| <= 1e-6
  - step, emit, candidate count: exact match
  - no skipped traces
  - output directory new/empty

Fail-closed: any mismatch → nonzero exit. No skip allowed.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from train_d1b_detector import (
    CandidateRanker,
    FEATURE_NAMES,
    normalize_features,
    TIE_TOLERANCE,
)
from run_l12_e4c2b_repair import sha256_file
from run_d2_fresh_confirm import select_eligible_multi_traces
from gripper_attack.production_detector import ProductionStreamingDetector

FROZEN_CHECKPOINT_SHA = (
    "cdd3cbe4f42592dab81590d84f5a8ff67b9fc3b7326f691742b9a438f1174858"
)
FROZEN_TAU = 0.236312

# Parity tolerances
NORM_FEAT_TOLERANCE = 1e-7
MLP_SCORE_TOLERANCE = 1e-6
RAW_FEAT_TOLERANCE = 0.001


def _compute_batch_mlp_score(batch_features: dict, model, means, stdevs, impute,
                              device) -> float:
    """Normalize batch candidate features and score with frozen MLP."""
    X = normalize_features([batch_features], means, stdevs, impute)
    X = X.to(device)
    with torch.no_grad():
        return float(model(X).item())


def _compute_batch_norm_vec(batch_features: dict, means, stdevs, impute) -> list:
    """Normalize batch candidate features, return 16-element list."""
    X = normalize_features([batch_features], means, stdevs, impute)
    return [round(float(v), 10) for v in X[0].cpu().tolist()]


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
    ap.add_argument("--d4-predictions-dir", default="tables/d4_replay")
    args = ap.parse_args()

    out = Path(args.output_dir)

    # ── Output directory must be new and empty ──
    if out.exists():
        contents = list(out.iterdir())
        assert len(contents) == 0, (
            f"Output directory must be empty: {out}  (found {len(contents)} entries)"
        )
    out.mkdir(parents=True, exist_ok=True)

    # ── Verify checkpoint ──
    actual_ckpt = sha256_file(args.checkpoint)
    assert actual_ckpt == FROZEN_CHECKPOINT_SHA, (
        f"Checkpoint SHA mismatch: got {actual_ckpt[:16]}..., "
        f"expected {FROZEN_CHECKPOINT_SHA[:16]}..."
    )
    print(f"Checkpoint: {actual_ckpt[:16]}... VERIFIED")

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load manifests ──
    manifest_map = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}
    fresh_manifest = args.manifest_fresh or args.manifest
    if fresh_manifest and os.path.exists(fresh_manifest):
        fr = list(csv.DictReader(open(fresh_manifest)))
        if "filename" in fr[0]:
            for r in fr:
                manifest_map[r["filename"].replace(".csv", "")] = r

    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}
    status_rows = list(csv.DictReader(open(args.trace_status)))
    fresh25_ids = set(select_eligible_multi_traces(
        list(csv.DictReader(open(args.candidate_table_fresh))), status_rows).keys())
    test_ids = {tid for tid in split if split[tid] == "test"}
    all_ids = sorted(test_ids | fresh25_ids)

    # ── Load D4.1 batch predictions ──
    d4_dir = Path(args.d4_predictions_dir)
    d4_test = list(csv.DictReader(open(d4_dir / "d4_test21_model_predictions.csv")))
    d4_fresh = list(csv.DictReader(open(d4_dir / "d4_fresh25_model_predictions.csv")))

    # ── Load batch candidate tables ──
    batch_all_cands = (
        list(csv.DictReader(open(args.candidate_table_test)))
        + list(csv.DictReader(open(args.candidate_table_fresh)))
    )

    print(f"Production parity: {len(all_ids)} traces")

    parity_rows = []
    total_step_mismatch = 0
    total_feat_mismatch = 0
    total_norm_mismatch = 0
    max_norm_diff = 0.0
    total_mlp_mismatch = 0
    max_mlp_diff = 0.0
    total_emit_mismatch = 0
    n_skipped = 0

    for tid in all_ids:
        # ── No skip allowed ──
        assert tid in manifest_map, f"MISSING_MANIFEST: trace_id={tid} not found in manifest"
        fp = manifest_map[tid]["source_path"]

        # Remap trace to get raw fields
        from remap_v4_trace_for_l12 import remap_v4_to_l12

        rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)

        # D4.1 reference
        if tid in test_ids:
            d4_row = next(r for r in d4_test if r["trace_id"] == tid)
        else:
            d4_row = next(r for r in d4_fresh if r["trace_id"] == tid)

        # Batch candidates for this trace
        batch_cands = [c for c in batch_all_cands if c["trace_id"] == tid]
        batch_cands.sort(key=lambda x: int(x["candidate_step"]))

        # ── Production detector ──
        detector = ProductionStreamingDetector(
            model, means, stdevs, impute, threshold=FROZEN_TAU,
        )
        prod_results = []
        for step_idx, r in enumerate(rows):
            raw = float(r.get("clean_gripper_raw", r.get("clean_gripper_raw_proxy", 0.5)))
            env_val = float(r.get("clean_gripper_env", 0))
            qpos = float(r.get("gripper_qpos_before", 0))
            eef_x = float(r.get("eef_x", 0))
            eef_y = float(r.get("eef_y", 0))
            eef_z = float(r.get("eef_z", 0))
            dec_open = int(float(r.get("decoded_open_bool", 0)))
            sem_valid = (
                str(r.get("gripper_semantics_valid", "1")) not in ("0", "False", "false")
            )
            raw_ok = (
                r.get("clean_gripper_raw", "") != ""
                or r.get("clean_gripper_raw_proxy", "") != ""
            )
            env_ok = r.get("clean_gripper_env", "") != ""
            qpos_ok = r.get("gripper_qpos_before", "") != ""
            eef_ok = all(r.get(f"eef_{a}", "") != "" for a in "xyz")

            result = detector.update(
                step_idx, raw, env_val, qpos, eef_x, eef_y, eef_z,
                dec_open, raw_ok, env_ok, qpos_ok, eef_ok, sem_valid,
            )
            if result is not None:
                prod_results.append(result)

        # ── Compare ──
        step_mismatch = 0
        feat_mismatch = 0
        norm_mismatches = 0
        max_step_norm_diff = 0.0
        mlp_mismatches = 0
        max_step_mlp_diff = 0.0

        for i in range(min(len(prod_results), len(batch_cands))):
            pr = prod_results[i]
            bc = batch_cands[i]

            # Candidate step match
            if pr["step"] != int(bc["candidate_step"]):
                step_mismatch += 1

            # Raw feature comparison
            for fn in FEATURE_NAMES:
                bv = bc.get(fn, "")
                sv = str(pr["features"].get(fn, ""))
                try:
                    if abs(float(bv) - float(sv)) > RAW_FEAT_TOLERANCE:
                        feat_mismatch += 1
                except Exception:
                    if bv != sv:
                        feat_mismatch += 1

            # ── Normalized-feature parity (batch-normalize → compare) ──
            batch_norm_vec = _compute_batch_norm_vec(bc, means, stdevs, impute)
            prod_norm_vec = pr["normalized_features"]
            for j, (bn, pn) in enumerate(zip(batch_norm_vec, prod_norm_vec)):
                diff = abs(bn - pn)
                if diff > NORM_FEAT_TOLERANCE:
                    norm_mismatches += 1
                max_step_norm_diff = max(max_step_norm_diff, diff)

            # ── MLP score parity (batch-normalize + batch-MLP vs production MLP) ──
            batch_mlp_score = _compute_batch_mlp_score(
                bc, model, means, stdevs, impute, device,
            )
            prod_mlp_score = pr["score"]
            mlp_diff = abs(batch_mlp_score - prod_mlp_score)
            if mlp_diff > MLP_SCORE_TOLERANCE:
                mlp_mismatches += 1
            max_step_mlp_diff = max(max_step_mlp_diff, mlp_diff)

        # ── Emit match ──
        if d4_row["classification"] != "ABSTAIN":
            batch_emit = int(d4_row["emit_step"])
        else:
            batch_emit = -1
        emit_match = (
            detector.emit_step == batch_emit
            and len(prod_results) == len(batch_cands)
        )

        # Accumulate
        total_step_mismatch += step_mismatch
        total_feat_mismatch += feat_mismatch
        total_norm_mismatch += norm_mismatches
        max_norm_diff = max(max_norm_diff, max_step_norm_diff)
        total_mlp_mismatch += mlp_mismatches
        max_mlp_diff = max(max_mlp_diff, max_step_mlp_diff)
        if not emit_match:
            total_emit_mismatch += 1

        ok = (
            step_mismatch == 0
            and feat_mismatch == 0
            and norm_mismatches == 0
            and mlp_mismatches == 0
            and emit_match
        )
        parity_rows.append({
            "trace_id": tid,
            "prod_n_cands": len(prod_results),
            "batch_n_cands": len(batch_cands),
            "step_mismatch": step_mismatch,
            "feat_mismatch": feat_mismatch,
            "norm_feat_mismatches": norm_mismatches,
            "max_norm_feat_diff": round(max_step_norm_diff, 12),
            "mlp_score_mismatches": mlp_mismatches,
            "max_mlp_score_diff": round(max_step_mlp_diff, 12),
            "emit_match": emit_match,
            "prod_emit": detector.emit_step,
            "batch_emit": batch_emit,
            "status": "OK" if ok else "FAIL",
        })
        status_str = (
            "OK" if ok else
            f"FAIL (step={step_mismatch} feat={feat_mismatch} "
            f"norm={norm_mismatches} mlp={mlp_mismatches} emit={emit_match})"
        )
        print(f"  {tid}: {status_str}")

    # ── Hard gates ──
    print(f"\n=== GATES ===")
    print(f"  step_mismatches:       {total_step_mismatch}")
    print(f"  feature_mismatches:    {total_feat_mismatch}")
    print(f"  norm_feat_mismatches:  {total_norm_mismatch}")
    print(f"  max_norm_feat_diff:    {max_norm_diff:.2e}")
    print(f"  mlp_score_mismatches:  {total_mlp_mismatch}")
    print(f"  max_mlp_score_diff:    {max_mlp_diff:.2e}")
    print(f"  emit_mismatches:       {total_emit_mismatch}")
    print(f"  skipped:               {n_skipped}")
    print(f"  n_traces:              {len(parity_rows)}")

    passed = (
        total_step_mismatch == 0
        and total_feat_mismatch == 0
        and total_norm_mismatch == 0
        and total_mlp_mismatch == 0
        and total_emit_mismatch == 0
        and n_skipped == 0
        and len(parity_rows) == 46
    )

    print(f"  ALL GATES: {'PASS' if passed else 'FAIL'}")

    # ── Write outputs ──
    parity_path = out / "d4_production_parity.csv"
    with open(parity_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(parity_rows[0].keys()))
        w.writeheader()
        w.writerows(parity_rows)

    log_path = out / "d4_production_run_log.txt"
    runner_sha = sha256_file(__file__)
    detector_sha = sha256_file(
        os.path.join(
            PIPELINE_ROOT, "src", "gripper_attack", "production_detector.py",
        )
    )
    with open(log_path, "w") as f:
        f.write(f"D4.2c RUN LOG\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        f.write(f"checkpoint_sha: {actual_ckpt}\n")
        f.write(f"threshold: {FROZEN_TAU}\n")
        f.write(f"n_traces: {len(parity_rows)}\n")
        f.write(f"step_mismatches: {total_step_mismatch}\n")
        f.write(f"feature_mismatches: {total_feat_mismatch}\n")
        f.write(f"norm_feat_mismatches: {total_norm_mismatch}\n")
        f.write(f"max_norm_feat_diff: {max_norm_diff:.6e}\n")
        f.write(f"norm_feat_tolerance: {NORM_FEAT_TOLERANCE:.1e}\n")
        f.write(f"mlp_score_mismatches: {total_mlp_mismatch}\n")
        f.write(f"max_mlp_score_diff: {max_mlp_diff:.6e}\n")
        f.write(f"mlp_score_tolerance: {MLP_SCORE_TOLERANCE:.1e}\n")
        f.write(f"emit_mismatches: {total_emit_mismatch}\n")
        f.write(f"skipped: {n_skipped}\n")
        f.write(f"all_gates_pass: {passed}\n")
        f.write(f"runner_sha: {runner_sha}\n")
        f.write(f"detector_sha: {detector_sha}\n")

    # ── Output artifact hashes ──
    hash_path = out / "d4_production_artifact_hashes.csv"
    with open(hash_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        w.writerow(["parity_csv", sha256_file(str(parity_path))])
        w.writerow(["run_log", sha256_file(str(log_path))])
        w.writerow(["runner_sha", runner_sha])
        w.writerow(["detector_sha", detector_sha])
        w.writerow(["checkpoint_sha", actual_ckpt])

    print(f"Output: {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
