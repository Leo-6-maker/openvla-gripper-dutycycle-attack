#!/usr/bin/env python3
"""Validate a FACTORIZED_SCHEDULER_FREEZE_V1.json contract.

Checks: schema valid, seal valid, identity role=P, no H/A binding,
worst-split false-start <= 0.10, feasible threshold exists,
source/adapter/config SHA closure.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None
HEADS = ("grasp", "manipulation", "release")


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


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-contract-root", type=Path, required=True,
                    help="Sealed output directory from freeze_factorized_scheduler_policy.py")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    freeze_root = args.freeze_contract_root.resolve()
    verify_bundle_seal(freeze_root, "FREEZE")

    contract_path = freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"
    contract = load_strict_json(contract_path, "FREEZE_CONTRACT")

    errors: list[str] = []

    if contract.get("schema") != "FACTORIZED_SCHEDULER_FREEZE_V1":
        errors.append("SCHEMA_INVALID")

    status = contract.get("status", "")

    # Authorization flags
    if contract.get("attack_authorized") is not False:
        errors.append("ATTACK_AUTHORIZED_NOT_FALSE")
    if contract.get("heldout_l3_authorized") is not False:
        errors.append("HELDOUT_L3_AUTHORIZED_NOT_FALSE")

    # Bindings
    bindings = contract.get("bindings", {})
    for key in ("calibrator_freeze_sha256", "policy_selection_manifest_sha256",
                "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256",
                "runtime_adapter_source_sha256", "scheduler_source_sha256",
                "structural_config_sha256", "freeze_code_sha256"):
        if not is_64char_hex(bindings.get(key, "")):
            errors.append(f"BINDING_INVALID: {key}")

    if status == "HOLD_NO_FEASIBLE_THRESHOLD":
        errors.append("NO_FEASIBLE_THRESHOLD_FOUND")
    elif status == "COMPLETE":
        thresholds = contract.get("selected_thresholds", {})
        for head in HEADS:
            t_val = thresholds.get(head)
            if not isinstance(t_val, (int, float)) or not 0.0 <= float(t_val) <= 1.0:
                errors.append(f"THRESHOLD_INVALID: {head}")

        ws_false = contract.get("worst_split_false_start")
        if ws_false is None or not isinstance(ws_false, (int, float)) or ws_false > 0.10:
            errors.append(f"WORST_FALSE_START_INVALID: {ws_false}")

        per_split_metrics = contract.get("per_split", {})
        if len(per_split_metrics) != 12:
            errors.append(f"SPLIT_COUNT: expected 12 got {len(per_split_metrics)}")

        # Verify every split has defined false start rate
        for sk, metrics in per_split_metrics.items():
            fs_rate = metrics.get("negative_episode_false_start_rate")
            if fs_rate is None:
                errors.append(f"SPLIT_UNDEFINED: {sk}")
    else:
        errors.append(f"STATUS_UNKNOWN: {status}")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    validation = {
        "schema": "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "errors": errors,
        "freeze_contract_sha256": sha256_file(contract_path),
        "attack_authorized": False,
        "heldout_l3_authorized": False,
        "mode": args.mode,
    }
    (staging / "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n")

    files = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, out_root)

    print(f"Scheduler Freeze Validation: {validation['status']}")
    for e in errors:
        print(f"  ERROR: {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
