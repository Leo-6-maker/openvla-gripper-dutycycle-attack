#!/usr/bin/env python3
"""Freeze Factorized calibrators from C Student predictions and C Teacher labels.

Reads only calibrator-fit identities (C). Never reads P predictions, H
identities, H predictions, A identities, or attack outcomes.

Produces a sealed FACTORIZED_CALIBRATOR_FREEZE_V1.json with per-split
per-head method, parameters, metrics, and provenance binds.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from fit_factorized_calibrators import (
    HEADS, load_json, sha256_file, verify_sealed_directory, sigmoid,
    fit_raw, fit_intercept, fit_platt, validate_record,
    check_logit_prob_consistency,
)

FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
SELF_SHA = None

# Deterministic method selection rule (pre-registered):
# 1. PLATT preferred if PLATT.valid AND n_fit_pos >= 10 AND n_fit_neg >= 10
# 2. INTERCEPT_ONLY if INTERCEPT.valid AND n_fit_pos >= 5 AND n_fit_neg >= 5
# 3. RAW if RAW.valid
# 4. HOLD_INSUFFICIENT_DATA otherwise


def _identity_set(manifest: dict[str, Any], role: str, split_key: str) -> set[str]:
    if "identities" in manifest:
        return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    if split_key in splits:
        sd = splits[split_key]
        if isinstance(sd, list): return set(sd)
        if isinstance(sd, dict): return set(sd.get(role, []))
    if role in manifest:
        rd = manifest[role]
        if isinstance(rd, list): return set(rd)
    return set()


def load_teacher_labels(bundle_root: Path, split_key: str) -> list[dict[str, Any]]:
    path = bundle_root / split_key / "factorized_teacher_v1.jsonl"
    if not path.is_file():
        raise SystemExit(f"TEACHER_LABELS_MISSING: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
            dups: list[str] = []
            def hook(pairs):
                s = set(); r = {}
                for k, v in pairs:
                    if k in s: dups.append(k)
                    s.add(k)
                    r[k] = v
                return r
            r = json.loads(line, object_pairs_hook=hook)
            if dups:
                raise SystemExit(f"TEACHER_DUP_KEY: {path}:{ln}")
            ep = r.get("canonical_parent_key")
            step = r.get("step")
            if not isinstance(ep, str) or not ep:
                raise SystemExit(f"TEACHER_EP_INVALID: {path}:{ln}")
            if isinstance(step, bool) or not isinstance(step, int):
                raise SystemExit(f"TEACHER_STEP_INVALID: {path}:{ln} step={step!r}")
            key = (ep, step)
            if key in seen:
                raise SystemExit(f"TEACHER_DUP: {path}:{ln} {key}")
            seen.add(key)
            rows.append(r)
    return rows


def load_student_predictions(bundle_root: Path, split_key: str) -> list[dict[str, Any]]:
    path = bundle_root / split_key / "predictions.jsonl"
    if not path.is_file():
        raise SystemExit(f"PREDICTIONS_MISSING: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
            dups: list[str] = []
            def hook(pairs):
                s = set(); r = {}
                for k, v in pairs:
                    if k in s: dups.append(k)
                    s.add(k)
                    r[k] = v
                return r
            r = json.loads(line, object_pairs_hook=hook)
            if dups:
                raise SystemExit(f"PRED_DUP_KEY: {path}:{ln}")
            ep = r.get("canonical_parent_key")
            step = r.get("step")
            if not isinstance(ep, str) or not ep:
                raise SystemExit(f"PRED_EP_INVALID: {path}:{ln}")
            key = (ep, step)
            if key in seen:
                raise SystemExit(f"PRED_DUP: {path}:{ln} {key}")
            seen.add(key)
            rows.append(r)
    return rows


def fit_all_methods(records: list[dict[str, Any]], head: str) -> list[dict[str, Any]]:
    return [fit_raw(records, head), fit_intercept(records, head), fit_platt(records, head)]


def select_method(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in results if r.get("method_valid")]
    if not candidates:
        return results[0]  # return failed RAW as HOLD evidence

    def priority(r: dict[str, Any]) -> tuple[int, int, int]:
        method = r.get("method", "")
        n_pos = r.get("n_fit_pos", 0)
        n_neg = r.get("n_fit_neg", 0)
        if method == "PLATT" and n_pos >= 10 and n_neg >= 10:
            return 0, n_pos + n_neg, 0
        if method == "INTERCEPT_ONLY" and n_pos >= 5 and n_neg >= 5:
            return 1, n_pos + n_neg, 0
        if method == "RAW":
            return 2, n_pos + n_neg, 0
        return 3, 0, 0

    candidates.sort(key=priority)
    return candidates[0]


def _seal_output(output_root: Path, files: dict[str, str]) -> str:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    data = sorted(p for p in staging.iterdir() if p.is_file())
    sums_content = "".join(f"{sha256_file(p)}  {p.name}\n" for p in data)
    (staging / "SHA256SUMS").write_text(sums_content)
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, output_root)
    return seal


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--calibration-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--calibration-teacher-bundle-root", type=Path, required=True)
    ap.add_argument("--phase-b-receipt", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-receipt", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--checkpoint-training-ledger", type=Path, required=True)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]

    # Load and validate receipts
    phase_b = load_json(args.phase_b_receipt)
    if not phase_b.get("cp_inference_authorized"):
        raise SystemExit("PHASE_B_CP_NOT_AUTHORIZED")
    cp_val = load_json(args.cp_prediction_validation_receipt)
    if not cp_val.get("cp_predictions_ready"):
        raise SystemExit("CP_PREDICTIONS_NOT_READY")

    # Load manifests
    fit_manifest = load_json(args.calibrator_fit_manifest)
    training_ledger = load_json(args.checkpoint_training_ledger)
    feature_sha = sha256_file(args.feature_order_contract)
    norm_sha = sha256_file(args.normalization_contract)

    # Verify bundle seals
    c_pred_root = args.calibration_prediction_bundle_root.resolve()
    c_teacher_root = args.calibration_teacher_bundle_root.resolve()
    verify_sealed_directory(c_pred_root)
    verify_sealed_directory(c_teacher_root)

    per_split_calibrators: dict[str, dict[str, Any]] = {}
    fit_metrics: dict[str, Any] = {}
    identity_receipt: dict[str, dict[str, Any]] = {}
    all_valid = True
    hold_reasons: list[str] = []

    for sk in expected:
        # Extract C identities
        c_ids = _identity_set(fit_manifest, "calibrator_fit", sk)
        t_ids = _identity_set(training_ledger, "checkpoint_training", sk)

        # Verify C ∩ T = ∅
        overlap = c_ids & t_ids
        if overlap:
            raise SystemExit(f"CAL_TRAIN_OVERLAP: {sk} n={len(overlap)}")

        # Load predictions and Teacher labels
        pred_rows = load_student_predictions(c_pred_root, sk)
        teacher_rows = load_teacher_labels(c_teacher_root, sk)

        pred_ids = {r["canonical_parent_key"] for r in pred_rows}
        teacher_ids = {r["canonical_parent_key"] for r in teacher_rows}

        if pred_ids != c_ids:
            raise SystemExit(f"CAL_ID_CLOSURE_FAIL: {sk} missing={sorted(c_ids - pred_ids)} extra={sorted(pred_ids - c_ids)}")
        if teacher_ids != c_ids:
            raise SystemExit(f"CAL_TEACHER_CLOSURE_FAIL: {sk} missing={sorted(c_ids - teacher_ids)} extra={sorted(teacher_ids - c_ids)}")

        # Exact episode-step join
        pred_keys = {(r["canonical_parent_key"], r["step"]) for r in pred_rows}
        teacher_keys = {(r["canonical_parent_key"], r["step"]) for r in teacher_rows}
        if pred_keys != teacher_keys:
            raise SystemExit(f"CAL_JOIN_FAIL: {sk} pred_only={len(pred_keys - teacher_keys)} teacher_only={len(teacher_keys - pred_keys)}")

        identity_receipt[sk] = {"c_identity_count": len(c_ids), "c_identities": sorted(c_ids)}

        # Build calibration records for fitting
        cal_records: list[dict[str, Any]] = []
        pred_by_key = {(r["canonical_parent_key"], r["step"]): r for r in pred_rows}
        for t_row in teacher_rows:
            key = (t_row["canonical_parent_key"], t_row["step"])
            p_row = pred_by_key[key]
            record = {
                "episode": t_row["canonical_parent_key"],
                "step": t_row["step"],
            }
            for head in HEADS:
                record[f"{head}_logit"] = p_row[f"{head}_logit"]
                record[f"{head}_probability"] = p_row[f"{head}_probability"]
                record[f"{head}_known_mask"] = t_row.get(f"grasp_established_known_mask" if head == "grasp"
                    else "manipulation_active_known_mask" if head == "manipulation"
                    else "release_or_instability_known_mask", False)
                record[f"{head}_target"] = t_row.get(f"grasp_established" if head == "grasp"
                    else "manipulation_active" if head == "manipulation"
                    else "release_or_instability", False)
            cal_records.append(record)

        # Fit all methods and select best per head
        split_result: dict[str, Any] = {}
        for head in HEADS:
            all_results = fit_all_methods(cal_records, head)
            selected = select_method(all_results)
            for r in all_results:
                r["checkpoint_sha256"] = pred_rows[0].get("checkpoint_sha256", "")
                r["split"] = sk
            selected["all_candidates"] = all_results
            if not selected.get("method_valid"):
                all_valid = False
                hold_reasons.append(f"{sk}/{head}: {selected.get('method_status')}")
            split_result[head] = selected

        per_split_calibrators[sk] = split_result

        # Compute fit metrics
        fit_metrics[sk] = {}
        for head in HEADS:
            sel = split_result[head]
            fit_metrics[sk][head] = {
                "method": sel["method"],
                "method_valid": sel.get("method_valid", False),
                "n_fit_pos": sel.get("n_fit_pos", 0),
                "n_fit_neg": sel.get("n_fit_neg", 0),
                "a": sel.get("a", 1.0),
                "b": sel.get("b", 0.0),
                "method_status": sel.get("method_status", "HOLD"),
            }

    # Build freeze contract
    freeze_contract = {
        "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
        "status": "COMPLETE" if all_valid else "HOLD_INSUFFICIENT_DATA",
        "all_heads_frozen": all_valid,
        "freeze_bindings": {
            "phase_b_receipt_sha256": sha256_file(args.phase_b_receipt),
            "cp_prediction_validation_receipt_sha256": sha256_file(args.cp_prediction_validation_receipt),
            "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
            "calibration_prediction_bundle_sha256": sha256_file(c_pred_root / "SHA256SUMS"),
            "calibration_teacher_bundle_sha256": sha256_file(c_teacher_root / "SHA256SUMS"),
            "checkpoint_manifest_root": str(args.checkpoint_manifest_root.resolve()),
            "feature_order_sha256": feature_sha,
            "normalization_sha256": norm_sha,
            "freeze_code_sha256": SELF_SHA,
        },
        "selection_rule": "PLATT(n>=10)→INTERCEPT(n>=5)→RAW; deterministic tie-break by n_fit_pos+n_fit_neg",
        "per_split": {},
        "attack_authorized": False,
        "heldout_l3_authorized": False,
        "full_fit_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for sk in expected:
        freeze_contract["per_split"][sk] = {}
        for head in HEADS:
            sel = per_split_calibrators[sk][head]
            freeze_contract["per_split"][sk][head] = {
                "method": sel["method"],
                "a": sel.get("a", 1.0),
                "b": sel.get("b", 0.0),
                "method_valid": sel.get("method_valid", False),
                "n_fit_pos": sel.get("n_fit_pos", 0),
                "n_fit_neg": sel.get("n_fit_neg", 0),
            }

    files = {
        "FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(freeze_contract, indent=2, sort_keys=True) + "\n",
        "FACTORIZED_CALIBRATOR_FIT_METRICS_V1.json": json.dumps(fit_metrics, indent=2, sort_keys=True) + "\n",
        "FACTORIZED_CALIBRATOR_IDENTITY_RECEIPT_V1.json": json.dumps(identity_receipt, indent=2, sort_keys=True) + "\n",
    }
    _seal_output(out_root, files)
    print(f"Calibrator Freeze Complete: {out_root} all_valid={all_valid}")
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
