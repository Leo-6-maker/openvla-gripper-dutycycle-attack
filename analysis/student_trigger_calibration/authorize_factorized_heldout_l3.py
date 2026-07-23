#!/usr/bin/env python3
"""Independent heldout-L3 authorization validator.

This is a SEPARATE gate from the Phase B validator. Phase B validator never
sets heldout_l3_inference_authorized=true. This validator checks ALL
prerequisites — Phase B receipt, CP prediction validation, calibrator freeze,
scheduler freeze, freeze independence, and that H has not yet run — and only
then authorizes EXACTLY ONE heldout-L3 evaluation run.

FAIL-CLOSED: any HOLD or error causes non-zero exit. Authorization scope is
single-run with explicit binding to all artifact SHAs.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()

def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)

def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    dups: list[str] = []
    def hook(pairs):
        seen = set(); result = {}
        for k, v in pairs:
            if k in seen: dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE: {path} {e}")
    if dups:
        raise SystemExit(f"{label}_DUP_KEYS: {path}")
    return value


def verify_bundle_seal(root: Path, label: str) -> None:
    bp = root.resolve()
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED")
    expected = sha256_file(sums)
    actual = sidecar.read_text().strip().split()
    if not actual or actual[0] != expected:
        raise SystemExit(f"{label}_SIDECAR_BROKEN")
    listed: set[str] = set()
    with open(sums) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2:
                raise SystemExit(f"{label}_SEAL_PARSE")
            file_sha, rel = parts[0], " ".join(parts[1:])
            if not is_64char_hex(file_sha):
                raise SystemExit(f"{label}_SEAL_SHA_INVALID: {rel}")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise SystemExit(f"{label}_SEAL_ESCAPE: {rel}")
            if rel in listed:
                raise SystemExit(f"{label}_SEAL_DUP: {rel}")
            listed.add(rel)
            target = bp / rel_path
            if not target.is_file() or sha256_file(target) != file_sha:
                raise SystemExit(f"{label}_SEAL_MISMATCH: {rel}")
    for p in bp.rglob("*"):
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            if p.relative_to(bp).as_posix() not in listed:
                raise SystemExit(f"{label}_SEAL_EXTRA: {p.relative_to(bp).as_posix()}")


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


FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-receipt", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-receipt", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-receipt", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-validation-receipt", type=Path, required=True)
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-prediction-root", type=Path, required=True,
                    help="INTENDED output root for H predictions — must NOT exist yet")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)
    if len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS:
        raise SystemExit(f"SPLIT_ENFORCEMENT: requires exactly 12 splits")

    errors: list[str] = []
    checks: dict[str, Any] = {}

    # ── 1. Phase B receipt ──────────────────────────────────────────
    phase_b = load_strict_json(args.phase_b_receipt, "PHASE_B")
    checks["phase_b_receipt"] = {
        "sha256": sha256_file(args.phase_b_receipt),
        "verdict": phase_b.get("verdict", "UNKNOWN"),
        "identity_disjointness": phase_b.get("identity_disjointness", "FAIL"),
        "heldout_teacher_closure": phase_b.get("heldout_teacher_closure", "HOLD"),
        "heldout_l3_data_ready": phase_b.get("heldout_l3_data_ready", False),
        "k10_contract_parity": phase_b.get("k10_contract_parity", "FAIL"),
    }

    # Phase B validator must have identity closure
    if phase_b.get("identity_disjointness") != "PASS":
        errors.append("PHASE_B_IDENTITY_NOT_PASS")
    # H teacher closure must pass
    if phase_b.get("heldout_teacher_closure") != "PASS":
        errors.append("PHASE_B_HELDOUT_TEACHER_NOT_PASS")
    # Data ready flag
    if not phase_b.get("heldout_l3_data_ready"):
        errors.append("PHASE_B_HELDOUT_DATA_NOT_READY")
    # K10 parity
    if phase_b.get("k10_contract_parity") != "PASS":
        errors.append("PHASE_B_K10_NOT_PASS")
    # Phase B validator must NOT have authorized L3 inference (our job)
    if phase_b.get("heldout_l3_inference_authorized") is not False:
        errors.append("PHASE_B_IMPROPERLY_AUTHORIZED_L3")

    # ── 2. CP prediction validation receipt ────────────────────────
    cp_val = load_strict_json(args.cp_prediction_validation_receipt, "CP_VAL")
    checks["cp_prediction_validation"] = {
        "sha256": sha256_file(args.cp_prediction_validation_receipt),
        "cp_predictions_ready": cp_val.get("cp_predictions_ready", False),
    }
    if not cp_val.get("cp_predictions_ready"):
        errors.append("CP_PREDICTIONS_NOT_READY")

    # ── 3. Calibrator freeze ────────────────────────────────────────
    cf_root = args.calibrator_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    cf_path = cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"
    cf_contract = load_strict_json(cf_path, "CAL_FREEZE")
    checks["calibrator_freeze"] = {
        "sha256": sha256_file(cf_path),
        "schema": cf_contract.get("schema"),
        "status": cf_contract.get("status"),
        "all_heads_frozen": cf_contract.get("all_heads_frozen", False),
    }
    if cf_contract.get("schema") != "FACTORIZED_CALIBRATOR_FREEZE_V1":
        errors.append("CAL_FREEZE_SCHEMA_INVALID")
    if not cf_contract.get("all_heads_frozen"):
        errors.append("CAL_FREEZE_NOT_ALL_HEADS_FROZEN")
    if cf_contract.get("attack_authorized") is not False:
        errors.append("CAL_FREEZE_ATTACK_AUTHORIZED")
    if cf_contract.get("heldout_l3_authorized") is not False:
        errors.append("CAL_FREEZE_HELDOUT_NOT_FALSE")

    # Calibrator freeze validation receipt
    cf_val = load_strict_json(args.calibrator_freeze_validation_receipt, "CAL_FREEZE_VAL")
    if cf_val.get("status") != "PASS":
        errors.append("CAL_FREEZE_VALIDATION_NOT_PASS")

    # ── 4. Scheduler freeze ─────────────────────────────────────────
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    sf_path = sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"
    sf_contract = load_strict_json(sf_path, "SCHED_FREEZE")
    checks["scheduler_freeze"] = {
        "sha256": sha256_file(sf_path),
        "schema": sf_contract.get("schema"),
        "status": sf_contract.get("status"),
    }
    if sf_contract.get("schema") != "FACTORIZED_SCHEDULER_FREEZE_V1":
        errors.append("SCHED_FREEZE_SCHEMA_INVALID")
    if sf_contract.get("status") != "COMPLETE":
        errors.append("SCHED_FREEZE_NOT_COMPLETE")
    if sf_contract.get("attack_authorized") is not False:
        errors.append("SCHED_FREEZE_ATTACK_AUTHORIZED")
    if sf_contract.get("heldout_l3_authorized") is not False:
        errors.append("SCHED_FREEZE_HELDOUT_NOT_FALSE")

    # Worst-split false-start must be <= 0.10
    ws_false = sf_contract.get("worst_split_false_start")
    if ws_false is None or ws_false > 0.10:
        errors.append(f"SCHED_FREEZE_WORST_FALSE_START: {ws_false}")

    # Scheduler freeze validation receipt
    sf_val = load_strict_json(args.scheduler_freeze_validation_receipt, "SCHED_FREEZE_VAL")
    if sf_val.get("status") != "PASS":
        errors.append("SCHED_FREEZE_VALIDATION_NOT_PASS")

    # ── 5. Freeze independence: C ∩ P ∩ H closure ─────────────────
    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")

    for sk in expected:
        c_ids = _identity_set(cal_manifest, "calibrator_fit", sk)
        p_ids = _identity_set(pol_manifest, "policy_selection", sk)
        h_ids = _identity_set(held_manifest, "heldout_l3", sk)

        if c_ids & h_ids:
            errors.append(f"C_H_OVERLAP: {sk} n={len(c_ids & h_ids)}")
        if p_ids & h_ids:
            errors.append(f"P_H_OVERLAP: {sk} n={len(p_ids & h_ids)}")
        if c_ids & p_ids:
            errors.append(f"C_P_OVERLAP: {sk} n={len(c_ids & p_ids)}")

    # ── 6. H not yet run ────────────────────────────────────────────
    h_pred_root = args.heldout_l3_prediction_root.resolve()
    checks["heldout_prediction_root"] = str(h_pred_root)
    if h_pred_root.exists():
        # Fail-closed: output root already exists
        errors.append("HELDOUT_PREEXISTING_OUTPUT")
        checks["preexisting_output"] = True
    else:
        checks["preexisting_output"] = False

    # ── Build authorization receipt ──────────────────────────────────
    all_pass = len(errors) == 0

    receipt = {
        "schema": "FACTORIZED_HELDOUT_L3_AUTHORIZATION_RECEIPT_V1",
        "authorization_code_sha256": SELF_SHA,
        "status": "AUTHORIZED" if all_pass else "HOLD",
        "heldout_l3_inference_authorized": all_pass,
        "authorization_scope": "EXACTLY_ONE_RUN",
        "authorized_h_manifest_sha256": sha256_file(args.heldout_l3_manifest),
        "authorized_checkpoint_manifest_root": str(args.checkpoint_manifest_root.resolve()),
        "authorized_calibrator_freeze_sha256": sha256_file(cf_path),
        "authorized_scheduler_freeze_sha256": sha256_file(sf_path),
        "authorized_output_root": str(h_pred_root),
        "authorized_phase_b_receipt_sha256": sha256_file(args.phase_b_receipt),
        "authorized_cp_validation_sha256": sha256_file(args.cp_prediction_validation_receipt),
        "attack_authorized": False,
        "canary_authorized": False,
        "heldout_l3_completed": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": errors,
        "checks": checks,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FACTORIZED_HELDOUT_L3_AUTHORIZATION_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, out_root)

    print(f"Heldout-L3 Authorization: {'AUTHORIZED' if all_pass else 'HOLD'}")
    for e in errors:
        print(f"  ERROR: {e}")
    print(f"  Output: {out_root}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
