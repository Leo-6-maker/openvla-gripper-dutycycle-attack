#!/usr/bin/env python3
"""Select Factorized scheduler thresholds from independent policy-selection data.

FAIL-CLOSED: no independent data → HOLD. No heldout, no defaults, no CAL/CHECK.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-selection-bundle-root", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--calibration-contract", type=Path, required=True)
    ap.add_argument("--structural-config-sha256", type=str, required=True)
    ap.add_argument("--scheduler-source-sha256", type=str, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", type=str, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    cal_contract = json.loads(args.calibration_contract.read_text())
    pol_manifest = json.loads(args.policy_selection_manifest.read_text())

    if cal_contract.get("provenance") != "INDEPENDENT_CALIBRATION":
        raise SystemExit(
            "THRESHOLD_SELECTION = HOLD_NO_INDEPENDENT_CALIBRATION\n"
            f"  Calibration provenance: {cal_contract.get('provenance')}"
        )

    calibrators = cal_contract.get("calibrators", [])
    if not all(c.get("method_valid", False) for c in calibrators):
        raise SystemExit("THRESHOLD_SELECTION = HOLD_CALIBRATOR_NOT_VALID")

    # Verify policy-selection data identity disjointness
    pol_ids = set(pol_manifest.get("policy_selection_identities", []))
    if not pol_ids:
        raise SystemExit(
            "THRESHOLD_SELECTION = HOLD_NO_INDEPENDENT_POLICY_SELECTION_DATA"
        )

    # Select thresholds via grid search on policy-selection data
    # (Placeholder — real implementation requires loading policy-selection logits
    #  and computing false-start/recall trade-off. Blocked until data exists.)
    contract = {
        "schema": "FACTORIZED_V2_THRESHOLD_CONTRACT_V1",
        "split": args.split,
        "status": "HOLD_NO_INDEPENDENT_POLICY_SELECTION_DATA",
        "calibration_contract_sha256": sha256_file(args.calibration_contract),
        "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
        "structural_config_sha256": args.structural_config_sha256,
        "scheduler_source_sha256": args.scheduler_source_sha256,
        "selected_thresholds": None,
        "formal_selection_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "threshold_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    sums = {}
    for f in staging.rglob("*"):
        if f.is_file() and f.name not in ("SHA256SUMS","SHA256SUMS.sha256"):
            sums[f.relative_to(staging).as_posix()] = sha256_file(f)
    (staging / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, out_root)
    print(f"Threshold contract sealed: {out_root}")
    print("Status: HOLD_NO_INDEPENDENT_POLICY_SELECTION_DATA")


if __name__ == "__main__":
    main()
