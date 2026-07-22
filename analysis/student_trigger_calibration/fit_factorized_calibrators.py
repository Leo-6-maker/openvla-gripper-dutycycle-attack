#!/usr/bin/env python3
"""Fit per-checkpoint calibrators on inner-train raw logits.

FAIL-CLOSED: missing fields, NaN/Inf, logit/prob mismatch → reject.
Methods: RAW, INTERCEPT_ONLY, PLATT.
Provenance: INDEPENDENT vs TRAIN_RESUBSTITUTION.
No silent fallback.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOGIT_PROB_TOLERANCE = 0.01


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))


def validate_record(r, head, idx):
    """Fail-closed field validation. Returns (logit, prob, is_known, is_target)."""
    for fld in ["episode", "step_index",
                f"{head}_logit", f"{head}_prob",
                f"{head}_known_mask", f"{head}_target"]:
        if fld not in r:
            raise SystemExit(f"FIELD_MISSING: record {idx} missing {fld}")

    logit = r[f"{head}_logit"]
    prob = r[f"{head}_prob"]
    km = r[f"{head}_known_mask"]
    tk = r[f"{head}_target"]

    if not isinstance(logit, (int, float)) or math.isnan(logit) or math.isinf(logit):
        raise SystemExit(f"LOGIT_INVALID: record {idx} {head}_logit={logit}")
    if not isinstance(prob, (int, float)) or math.isnan(prob) or math.isinf(prob):
        raise SystemExit(f"PROB_INVALID: record {idx} {head}_prob={prob}")
    if prob < 0 or prob > 1:
        raise SystemExit(f"PROB_OUT_OF_RANGE: record {idx} {head}_prob={prob}")
    if not isinstance(km, bool):
        raise SystemExit(f"KNOWN_MASK_NOT_BOOL: record {idx} {head}_known_mask={km}")
    if not isinstance(tk, bool):
        raise SystemExit(f"TARGET_NOT_BOOL: record {idx} {head}_target={tk}")
    if not isinstance(r["step_index"], int) or r["step_index"] < 0:
        raise SystemExit(f"STEP_INVALID: record {idx} step_index={r['step_index']}")

    return float(logit), float(prob), km, tk


def load_and_validate(bundle_dir, split_key):
    p = Path(bundle_dir) / split_key / "inner_train_logits.jsonl"
    if not p.is_file():
        raise SystemExit(f"LOGITS_MISSING: {p}")
    seen = set()
    records = []
    with open(p) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            key = (r.get("episode"), r.get("step_index"))
            if key in seen:
                raise SystemExit(f"DUPLICATE_KEY: record {idx} key={key}")
            seen.add(key)
            records.append(r)
    return records


def check_logit_prob_consistency(records, head):
    """Verify sigmoid(logit) ≈ prob for ALL records. Returns (ok, max_err)."""
    max_err = 0.0
    for i, r in enumerate(records):
        if r.get(f"{head}_known_mask"):
            logit = float(r[f"{head}_logit"])
            prob = float(r[f"{head}_prob"])
            expected = sigmoid(logit)
            err = abs(expected - prob)
            if err > max_err:
                max_err = err
    ok = max_err <= LOGIT_PROB_TOLERANCE
    return ok, max_err


def fit_raw(records, head):
    km = f"{head}_known_mask"; tk = f"{head}_target"
    n_pos = sum(1 for r in records if r[km] and r[tk])
    n_neg = sum(1 for r in records if r[km] and not r[tk])
    lp_ok, lp_err = check_logit_prob_consistency(records, head)
    return {"head": head, "method": "RAW", "a": 1.0, "b": 0.0,
            "n_fit_pos": n_pos, "n_fit_neg": n_neg, "method_valid": lp_ok,
            "logit_prob_max_error": round(lp_err, 8)}


def fit_intercept(records, head):
    logit_key = f"{head}_logit"; km = f"{head}_known_mask"; tk = f"{head}_target"
    lp_ok, lp_err = check_logit_prob_consistency(records, head)
    if not lp_ok:
        return {"head": head, "method": "INTERCEPT_ONLY", "a": 1.0, "b": 0.0,
                "n_fit_pos": 0, "n_fit_neg": 0, "method_valid": False,
                "method_status": "LOGIT_PROBABILITY_BINDING_FAIL",
                "logit_prob_max_error": round(lp_err, 8)}

    pos_z, neg_z = [], []
    for r in records:
        if not r[km]: continue
        z = float(r[logit_key])
        if r[tk]: pos_z.append(z)
        else: neg_z.append(z)

    if len(pos_z) < 5 or len(neg_z) < 5:
        return {"head": head, "method": "INTERCEPT_ONLY", "a": 1.0, "b": 0.0,
                "n_fit_pos": len(pos_z), "n_fit_neg": len(neg_z),
                "method_valid": False,
                "method_status": "HOLD_INSUFFICIENT_SAMPLES",
                "failure_reason": f"pos={len(pos_z)} neg={len(neg_z)}"}

    best_b, best_loss = 0.0, float("inf")
    for i in range(61):
        b = -3.0 + i * 0.1; loss = 0.0
        for z in pos_z: loss -= math.log(max(1e-7, sigmoid(z + b)))
        for z in neg_z: loss -= math.log(max(1e-7, 1 - sigmoid(z + b)))
        loss /= (len(pos_z) + len(neg_z))
        if loss < best_loss: best_loss, best_b = loss, b

    return {"head": head, "method": "INTERCEPT_ONLY", "a": 1.0, "b": round(best_b, 6),
            "n_fit_pos": len(pos_z), "n_fit_neg": len(neg_z),
            "class_prevalence": len(pos_z)/max(1,len(pos_z)+len(neg_z)),
            "fit_loss": round(best_loss, 6), "method_valid": True}


def fit_platt(records, head):
    logit_key = f"{head}_logit"; km = f"{head}_known_mask"; tk = f"{head}_target"
    lp_ok, lp_err = check_logit_prob_consistency(records, head)
    if not lp_ok:
        return {"head": head, "method": "PLATT", "a": 1.0, "b": 0.0,
                "n_fit_pos": 0, "n_fit_neg": 0, "method_valid": False,
                "method_status": "LOGIT_PROBABILITY_BINDING_FAIL",
                "logit_prob_max_error": round(lp_err, 8)}

    pos_z, neg_z = [], []
    for r in records:
        if not r[km]: continue
        z = float(r[logit_key])
        if r[tk]: pos_z.append(z)
        else: neg_z.append(z)

    if len(pos_z) < 5 or len(neg_z) < 5:
        return {"head": head, "method": "PLATT", "a": 1.0, "b": 0.0,
                "n_fit_pos": len(pos_z), "n_fit_neg": len(neg_z),
                "method_valid": False,
                "method_status": "HOLD_INSUFFICIENT_SAMPLES",
                "failure_reason": f"pos={len(pos_z)} neg={len(neg_z)}"}

    best_a, best_b, best_loss = 1.0, 0.0, float("inf")
    for ai in range(15):
        a = 0.2 + ai * 0.2
        for bi in range(61):
            b = -3.0 + bi * 0.1; loss = 0.0
            for z in pos_z: loss -= math.log(max(1e-7, sigmoid(a * z + b)))
            for z in neg_z: loss -= math.log(max(1e-7, 1 - sigmoid(a * z + b)))
            loss /= (len(pos_z) + len(neg_z))
            if loss < best_loss: best_loss, best_a, best_b = loss, a, b

    return {"head": head, "method": "PLATT", "a": round(best_a, 6), "b": round(best_b, 6),
            "n_fit_pos": len(pos_z), "n_fit_neg": len(neg_z),
            "class_prevalence": len(pos_z)/max(1,len(pos_z)+len(neg_z)),
            "fit_loss": round(best_loss, 6), "method_valid": True}


def validate_fit_heldout_disjoint(fit_manifest, heldout_manifest):
    fit_ids = set(fit_manifest.get("fit_identities", []))
    ho_ids = set(heldout_manifest.get("heldout_identities", []))
    if not fit_ids: raise ValueError("fit_identities is empty")
    if not ho_ids: raise ValueError("heldout_identities is empty")
    overlap = fit_ids & ho_ids
    if overlap:
        raise ValueError(f"CALIBRATION_LEAKAGE: {len(overlap)} identities in both")


def classify_provenance(fit_manifest, checkpoint_manifest):
    fit_ids = set(fit_manifest.get("fit_identities", []))
    train_ids = set(checkpoint_manifest.get("training_identities", []))
    if not train_ids:
        return "UNKNOWN_NO_TRAINING_IDENTITY_LIST"
    if fit_ids & train_ids:
        return "TRAIN_RESUBSTITUTION_CALIBRATION"
    return "INDEPENDENT_CALIBRATION"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-train-bundle-root", type=Path, required=True)
    ap.add_argument("--inner-train-manifest", type=Path, required=True)
    ap.add_argument("--heldout-manifest", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", type=str, required=True)
    ap.add_argument("--method", choices=["RAW","INTERCEPT_ONLY","PLATT"], default="PLATT")
    ap.add_argument("--checkpoint-sha256", type=str, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    fit_manifest = json.loads(args.inner_train_manifest.read_text())
    heldout_manifest = json.loads(args.heldout_manifest.read_text())
    validate_fit_heldout_disjoint(fit_manifest, heldout_manifest)

    provenance = "UNKNOWN"
    if args.checkpoint_manifest:
        ckpt_manifest = json.loads(Path(args.checkpoint_manifest).read_text())
        provenance = classify_provenance(fit_manifest, ckpt_manifest)

    records = load_and_validate(args.inner_train_bundle_root, args.split)

    # Validate each record
    for head in ["grasp", "manipulation", "release"]:
        for i, r in enumerate(records):
            validate_record(r, head, i)

    calibrators = []
    all_valid = True
    for head in ["grasp", "manipulation", "release"]:
        if args.method == "RAW": cal = fit_raw(records, head)
        elif args.method == "INTERCEPT_ONLY": cal = fit_intercept(records, head)
        else: cal = fit_platt(records, head)

        cal["checkpoint_sha256"] = args.checkpoint_sha256
        cal["split"] = args.split
        cal["provenance"] = provenance
        cal["formal_selection_eligible"] = False
        if not cal["method_valid"]: all_valid = False
        calibrators.append(cal)

    authoritative = (provenance == "INDEPENDENT_CALIBRATION" and all_valid)

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_CONTRACT_V1",
        "split": args.split, "method": args.method,
        "checkpoint_sha256": args.checkpoint_sha256,
        "student_source_commit": "401f79a05753d970ecc803bb96abc64ce132df42",
        "fit_manifest_sha256": sha256_file(args.inner_train_manifest),
        "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
        "provenance": provenance,
        "authoritative": authoritative,
        "all_heads_valid": all_valid,
        "fit_identity_count": len(fit_manifest.get("fit_identities", [])),
        "heldout_identity_count": len(heldout_manifest.get("heldout_identities", [])),
        "calibrators": calibrators,
        "formal_selection_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (staging / "calibration_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    sums = {}
    for f in staging.rglob("*"):
        if f.is_file() and f.name not in ("SHA256SUMS","SHA256SUMS.sha256"):
            sums[f.relative_to(staging).as_posix()] = sha256_file(f)
    (staging / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")

    os.replace(staging, out_root)
    print(f"Calibration sealed: {out_root}")
    print(f"Provenance: {provenance}  Authoritative: {authoritative}")


if __name__ == "__main__":
    main()
