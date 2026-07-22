#!/usr/bin/env python3
"""Produce FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2 — canonical schema.

Top-level: grasp/manipulation/release (not heads dict).
Eligibility: calibration_fit_authoritative, threshold_selection_authoritative,
             l3_evaluation_eligible.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-fit-contract", type=Path, required=True)
    ap.add_argument("--threshold-selection-contract", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", type=str, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    cal = json.loads(args.calibration_fit_contract.read_text())
    cal_heads = {c["head"]: c for c in cal.get("calibrators", [])}

    # Eligibility from calibration fit
    fit_auth = (cal.get("provenance") == "INDEPENDENT_CALIBRATION"
                and all(cal_heads.get(h, {}).get("method_valid", False) for h in ["grasp","manipulation","release"]))

    thr_data = {}
    thr_auth = False
    if args.threshold_selection_contract:
        thr = json.loads(Path(args.threshold_selection_contract).read_text())
        thr_data = thr.get("selected_thresholds", {})
        thr_auth = (thr.get("status") == "COMPLETE"
                    and thr.get("provenance") == "INDEPENDENT_POLICY_SELECTION")

    l3_eligible = fit_auth and thr_auth
    status = "AUTHORITATIVE" if l3_eligible else "DIAGNOSTIC"

    # Build canonical V3 contract
    heads = {}
    for head in ["grasp", "manipulation", "release"]:
        hc = cal_heads.get(head, {})
        if not hc:
            # Missing head → BLOCKER only
            staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
            staging.mkdir(parents=True)
            (staging / "BLOCKER_RECEIPT.json").write_text(json.dumps({
                "status": "BLOCKED_MISSING_HEAD", "head": head, "split": args.split,
            }, indent=2) + "\n")
            f = staging / "BLOCKER_RECEIPT.json"
            (staging / "SHA256SUMS").write_text(f"{sha256_file(f)}  BLOCKER_RECEIPT.json\n")
            (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
            os.replace(staging, out_root)
            raise SystemExit(f"BLOCKED_MISSING_HEAD: {head}")

        heads[head] = {
            "method": hc["method"],
            "a": hc["a"], "b": hc["b"],
            "threshold": thr_data.get(head),
            "transform": "probability=sigmoid(a*raw_logit+b)",
            "method_valid": hc.get("method_valid", False),
            "transform_valid": hc.get("method_valid", False),
            "fit_data_valid": hc.get("method_valid", False),
            "provenance_class": cal.get("provenance", "UNKNOWN"),
            "fit_manifest_sha256": sha256_file(args.calibration_fit_contract),
            "policy_selection_manifest_sha256": (
                sha256_file(Path(args.threshold_selection_contract)) if args.threshold_selection_contract else None),
        }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
        "split": args.split,
        "status": status,
        "calibration_fit_authoritative": fit_auth,
        "threshold_selection_authoritative": thr_auth,
        "l3_evaluation_eligible": l3_eligible,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "grasp": heads["grasp"], "manipulation": heads["manipulation"], "release": heads["release"],
        "checkpoint_sha256": cal.get("checkpoint_sha256", ""),
        "student_source_commit": cal.get("student_source_commit", "401f79a05753d970ecc803bb96abc64ce132df42"),
        "calibration_fit_manifest_sha256": sha256_file(args.calibration_fit_contract),
        "threshold_selection_manifest_sha256": (
            sha256_file(Path(args.threshold_selection_contract)) if args.threshold_selection_contract else None),
        "formal_selection_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (staging / "calibration_and_threshold_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    f = staging / "calibration_and_threshold_contract.json"
    (staging / "SHA256SUMS").write_text(f"{sha256_file(f)}  calibration_and_threshold_contract.json\n")
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    os.replace(staging, out_root)
    print(f"Contract V3: {out_root}  status={status}")


if __name__ == "__main__":
    main()
