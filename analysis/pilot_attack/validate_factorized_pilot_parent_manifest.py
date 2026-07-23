#!/usr/bin/env python3
"""B1: Validate pilot parent manifest — all parents in RESERVED_FEC, disjoint from T/C/P/H/A."""
from __future__ import annotations

import argparse, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from pilot_integrity import sha256_file, load_strict_json, seal_output_dir, is_64char_hex

SELF_SHA = None
REQUIRED_PARENT_FIELDS = (
    "parent_id", "suite", "task", "clean_success", "detector_emitted",
    "remaining_horizon", "selection_rank",
)


def _identity_set(manifest: dict[str, Any], role: str, split_key: str = "") -> set[str]:
    if "identities" in manifest: return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    for key in splits:
        sd = splits[key]
        if isinstance(sd, list): return set(sd) | set().union(*(set(splits.get(k, [])) if isinstance(splits.get(k), list) else set() for k in splits))
        if isinstance(sd, dict): return set(sd.get(role, [])) | set().union(*(set(splits.get(k, {}).get(role, [])) for k in splits))
    if role in manifest:
        rd = manifest[role]
        if isinstance(rd, list): return set(rd)
    return set()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-parent-manifest", type=Path, required=True)
    ap.add_argument("--reserved-fec-manifest", type=Path, required=True)
    ap.add_argument("--t-manifest", type=Path, required=True)
    ap.add_argument("--c-manifest", type=Path, required=True)
    ap.add_argument("--p-manifest", type=Path, required=True)
    ap.add_argument("--h-manifest", type=Path, required=True)
    ap.add_argument("--a-manifest", type=Path, required=True)
    ap.add_argument("--pilot-detector-config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    parent_manifest = load_strict_json(args.pilot_parent_manifest, "PARENTS")
    fec_manifest = load_strict_json(args.reserved_fec_manifest, "FEC")
    detector = load_strict_json(args.pilot_detector_config, "DETECTOR")

    errors: list[str] = []
    parents = parent_manifest.get("parents", parent_manifest.get("identities", []))
    if not isinstance(parents, list) or not parents:
        errors.append("NO_PARENTS")

    fec_ids = _identity_set(fec_manifest, "identities") or _identity_set(fec_manifest, "reserved_fec")
    t_ids = _identity_set(load_strict_json(args.t_manifest, "T"), "checkpoint_training")
    c_ids = _identity_set(load_strict_json(args.c_manifest, "C"), "calibrator_fit")
    p_ids = _identity_set(load_strict_json(args.p_manifest, "P"), "policy_selection")
    h_ids = _identity_set(load_strict_json(args.h_manifest, "H"), "heldout_l3")
    a_ids = _identity_set(load_strict_json(args.a_manifest, "A"), "attack_eval")

    parent_ids: set[str] = set()
    for item in parents:
        if not isinstance(item, dict): errors.append(f"PARENT_NOT_OBJECT: {item}"); continue
        pid = item.get("parent_id", "")
        if not pid: errors.append("PARENT_NO_ID"); continue
        if pid in parent_ids: errors.append(f"PARENT_DUP: {pid}"); continue
        parent_ids.add(pid)

        # Check parent in RESERVED_FEC
        if pid not in fec_ids:
            errors.append(f"PARENT_NOT_IN_FEC: {pid}")

        # Check disjoint from T/C/P/H/A
        for label, ids in [("T", t_ids), ("C", c_ids), ("P", p_ids), ("H", h_ids), ("A", a_ids)]:
            if pid in ids:
                errors.append(f"PARENT_IN_{label}: {pid}")

        # Check required fields
        for fld in REQUIRED_PARENT_FIELDS:
            if fld not in item:
                errors.append(f"PARENT_MISSING_{fld}: {pid}")

        # Check clean_success
        if not item.get("clean_success", False):
            errors.append(f"PARENT_NOT_CLEAN: {pid}")

        # Check horizon
        horizon = item.get("remaining_horizon", 0)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 10:
            errors.append(f"PARENT_HORIZON_SHORT: {pid} horizon={horizon}")

        # Check attack outcome not used
        if item.get("attack_outcome") is not None or item.get("attack_eval_consumed"):
            errors.append(f"PARENT_ATTACK_CONSUMED: {pid}")

    # Detector config
    if detector.get("paper_authoritative") is not False:
        errors.append("DETECTOR_PAPER_AUTHORITATIVE")
    if detector.get("attack_eval_consumed") is not False:
        errors.append("DETECTOR_ATTACK_EVAL_CONSUMED")

    receipt = {
        "schema": "PILOT_PARENT_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_parents": len(parent_ids),
        "n_errors": len(errors),
        "fec_size": len(fec_ids),
        "errors": errors,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_PARENT_VALIDATION_V0.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)
    print(f"Pilot Parent Validation: {receipt['status']} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
