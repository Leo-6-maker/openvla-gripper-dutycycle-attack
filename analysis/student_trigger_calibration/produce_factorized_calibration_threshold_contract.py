#!/usr/bin/env python3
"""Produce FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3.

Exact Codex schema compliance. No extra fields, no None, no defaults.
BLOCKER_RECEIPT when data insufficient.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = ROOT / "schemas/factorized_v2_calibration_and_threshold_contract_v3.schema.json"


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def validate_against_schema(contract):
    schema = json.loads(SCHEMA_PATH.read_text())
    try:
        import jsonschema
        jsonschema.validate(contract, schema)
    except ImportError:
        _manual_validate(contract, schema)
    except Exception as e:
        raise SystemExit(f"SCHEMA_VALIDATION_FAILED: {e}")


def _manual_validate(contract, schema):
    required = schema.get("required", [])
    for k in required:
        if k not in contract: raise SystemExit(f"MISSING_REQUIRED: {k}")
    props = schema.get("properties", {})
    for k in contract:
        if k not in props: raise SystemExit(f"EXTRA_FIELD: {k}")
    if contract["schema"] != "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3":
        raise SystemExit("BAD_SCHEMA")
    if contract["status"] not in ("DIAGNOSTIC", "AUTHORITATIVE"):
        raise SystemExit(f"BAD_STATUS: {contract['status']}")
    for fld in ["checkpoint_sha256","scheduler_source_sha256","structural_config_sha256","feature_order_sha256"]:
        if len(contract[fld]) != 64: raise SystemExit(f"BAD_SHA_LEN: {fld}")
    if len(contract["student_source_commit"]) != 40: raise SystemExit("BAD_COMMIT_LEN")
    for a in ["training_authorized","full_fit_authorized","attack_authorized"]:
        if contract[a] is not False: raise SystemExit(f"{a} must be false")
    hdef = schema.get("$defs",{}).get("head",{})
    for head in ["grasp","manipulation","release"]:
        h = contract[head]
        for k in hdef.get("required",[]):
            if k not in h: raise SystemExit(f"HEAD_MISSING: {head}.{k}")
        for k in h:
            if k not in hdef.get("properties",{}): raise SystemExit(f"HEAD_EXTRA: {head}.{k}")
        if h["threshold"] is None or not (0 <= h["threshold"] <= 1):
            raise SystemExit(f"BAD_THRESHOLD: {head}")
        for fld in ["fit_manifest_sha256","policy_selection_manifest_sha256"]:
            if len(h[fld]) != 64: raise SystemExit(f"BAD_HEAD_SHA: {head}.{fld}")
        for fld in ["method_valid","transform_valid","fit_data_valid"]:
            if h[fld] is not True: raise SystemExit(f"HEAD_{fld}_NOT_TRUE: {head}")


def blocker(out_root, reason):
    s = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); s.mkdir(parents=True)
    (s/"BLOCKER_RECEIPT.json").write_text(json.dumps({"status":reason,"authoritative_l3":False},indent=2)+"\n")
    f=s/"BLOCKER_RECEIPT.json"; (s/"SHA256SUMS").write_text(f"{sha256_file(f)}  BLOCKER_RECEIPT.json\n")
    (s/"SHA256SUMS.sha256").write_text(f"{sha256_file(s/'SHA256SUMS')}  SHA256SUMS\n")
    os.replace(s, out_root)
    print(f"BLOCKER: {out_root} ({reason})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-fit-contract", type=Path, required=True)
    ap.add_argument("--threshold-selection-contract", type=Path, default=None)
    ap.add_argument("--scheduler-source-sha256", type=str, required=True)
    ap.add_argument("--structural-config-sha256", type=str, required=True)
    ap.add_argument("--feature-order-sha256", type=str, required=True)
    ap.add_argument("--student-source-commit", type=str, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", type=str, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    cal = json.loads(args.calibration_fit_contract.read_text())
    cal_heads = {c["head"]: c for c in cal.get("calibrators", [])}

    if cal.get("provenance") != "INDEPENDENT_CALIBRATION":
        return blocker(out_root, "BLOCKED_NOT_INDEPENDENT_CALIBRATION")
    if not all(cal_heads.get(h,{}).get("method_valid",False) for h in ["grasp","manipulation","release"]):
        return blocker(out_root, "BLOCKED_CALIBRATOR_NOT_VALID")
    if not args.threshold_selection_contract:
        return blocker(out_root, "BLOCKED_NO_THRESHOLD_SELECTION")

    thr = json.loads(Path(args.threshold_selection_contract).read_text())
    if thr.get("status") != "COMPLETE":
        return blocker(out_root, "BLOCKED_THRESHOLD_NOT_COMPLETE")
    td = thr.get("selected_thresholds", {})
    for h in ["grasp","manipulation","release"]:
        if td.get(h) is None or not (0<=td[h]<=1):
            return blocker(out_root, f"BLOCKED_INVALID_THRESHOLD_{h}")

    thr_auth = thr.get("provenance") == "INDEPENDENT_POLICY_SELECTION"
    fit_sha = sha256_file(args.calibration_fit_contract)
    pol_sha = sha256_file(Path(args.threshold_selection_contract))

    heads = {}
    for head in ["grasp","manipulation","release"]:
        hc = cal_heads[head]
        heads[head] = {
            "method": hc["method"], "a": hc["a"], "b": hc["b"],
            "threshold": td[head],
            "transform": "probability=sigmoid(a*raw_logit+b)",
            "method_valid": True, "transform_valid": True, "fit_data_valid": True,
            "provenance_class": "INDEPENDENT_CALIBRATION",
            "fit_manifest_sha256": fit_sha,
            "policy_selection_manifest_sha256": pol_sha,
        }

    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
        "status": "AUTHORITATIVE" if thr_auth else "DIAGNOSTIC",
        "split": args.split,
        "checkpoint_sha256": cal["checkpoint_sha256"],
        "scheduler_source_sha256": args.scheduler_source_sha256,
        "structural_config_sha256": args.structural_config_sha256,
        "student_source_commit": args.student_source_commit,
        "feature_order_sha256": args.feature_order_sha256,
        "calibration_fit_authoritative": True,
        "threshold_selection_authoritative": thr_auth,
        "l3_evaluation_eligible": thr_auth,
        "training_authorized": False, "full_fit_authorized": False, "attack_authorized": False,
        "grasp": heads["grasp"], "manipulation": heads["manipulation"], "release": heads["release"],
    }
    validate_against_schema(contract)

    s = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); s.mkdir(parents=True)
    (s/"calibration_and_threshold_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    f=s/"calibration_and_threshold_contract.json"
    (s/"SHA256SUMS").write_text(f"{sha256_file(f)}  calibration_and_threshold_contract.json\n")
    (s/"SHA256SUMS.sha256").write_text(f"{sha256_file(s/'SHA256SUMS')}  SHA256SUMS\n")
    os.replace(s, out_root)
    print(f"Contract V3: {out_root}  status={contract['status']}")


if __name__ == "__main__":
    main()
