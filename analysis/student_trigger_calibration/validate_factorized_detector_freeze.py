#!/usr/bin/env python3
"""Validate a FACTORIZED_DETECTOR_FREEZE_V1.json contract.

Checks: schema valid, seal valid, all bindings present, attack_authorized=false,
canary_authorized=false, all prerequisite artifact SHAs are valid 64-char hex.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, uuid
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


REQUIRED_BINDINGS = (
    "phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256",
    "calibrator_freeze_sha256", "calibrator_freeze_validation_sha256",
    "scheduler_freeze_sha256", "scheduler_freeze_validation_sha256",
    "heldout_authorization_receipt_sha256", "heldout_l3_run_receipt_sha256",
    "feature_order_sha256", "normalization_sha256",
    "structural_config_sha256", "scheduler_source_sha256",
    "runtime_adapter_source_sha256", "freeze_builder_code_sha256",
)


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-contract-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    freeze_root = args.freeze_contract_root.resolve()
    verify_bundle_seal(freeze_root, "FREEZE")

    contract_path = freeze_root / "FACTORIZED_DETECTOR_FREEZE_V1.json"
    contract = load_strict_json(contract_path, "FREEZE_CONTRACT")

    errors: list[str] = []

    if contract.get("schema") != "FACTORIZED_DETECTOR_FREEZE_V1":
        errors.append("SCHEMA_INVALID")

    # Authorization must be false
    if contract.get("attack_authorized") is not False:
        errors.append("ATTACK_AUTHORIZED_NOT_FALSE")
    if contract.get("canary_authorized") is not False:
        errors.append("CANARY_AUTHORIZED_NOT_FALSE")

    # All bindings must be valid 64-char hex
    bindings = contract.get("bindings", {})
    for key in REQUIRED_BINDINGS:
        val = bindings.get(key, "")
        if not is_64char_hex(val):
            errors.append(f"BINDING_INVALID: {key} value={str(val)[:32]}")

    # Heldout gate status
    h_gate = contract.get("heldout_l3_gate", {})
    if h_gate.get("gate_pass") is not True:
        errors.append("HELDOUT_L3_GATE_NOT_PASS")

    # Thresholds must be present
    thresholds = contract.get("selected_thresholds", {})
    for head in ("grasp", "manipulation", "release"):
        if not isinstance(thresholds.get(head), (int, float)):
            errors.append(f"THRESHOLD_MISSING: {head}")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    validation = {
        "schema": "FACTORIZED_DETECTOR_FREEZE_VALIDATION_V1",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "errors": errors,
        "freeze_contract_sha256": sha256_file(contract_path),
        "attack_authorized": False,
        "canary_authorized": False,
        "mode": args.mode,
    }
    (staging / "FACTORIZED_DETECTOR_FREEZE_VALIDATION_V1.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, out_root)

    print(f"Detector Freeze Validation: {validation['status']}")
    for e in errors:
        print(f"  ERROR: {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
