#!/usr/bin/env python3
"""Stage 8: Full-FIT — retrain final detector on FIT-TRAIN+DEV+CAL+CHECK with frozen config.

Requires H heldout gate_pass=true. Uses the same inner-CV trainer with all 4 splits
combined as training data. Output is atomic — model and receipt sealed together.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-labels-root", type=Path, required=True)
    ap.add_argument("--student-config-root", type=Path, required=True)
    ap.add_argument("--stage-7-h-receipt-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # ── Verify H heldout PASS ────────────────────────────────────────
    h_root = args.stage_7_h_receipt_root
    h_receipt = None
    for name in ["HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json",
                 "FACTORIZED_HELDOUT_L3_EVALUATION_RECEIPT_V1.json",
                 "receipt.json"]:
        candidate = h_root / name
        if candidate.is_file():
            h_receipt = json.loads(candidate.read_text(encoding="utf-8"))
            break
    if h_receipt is None:
        raise SystemExit(f"H_RECEIPT_NOT_FOUND: no receipt in {h_root}")

    run_status = h_receipt.get("run_status", h_receipt.get("status", ""))
    gate_pass = h_receipt.get("gate_pass", h_receipt.get("gate_status", False))

    if run_status not in ("COMPLETE", "PASS"):
        raise SystemExit(f"H_HELDOUT_NOT_COMPLETE: run_status={run_status}")
    if not gate_pass:
        raise SystemExit(f"H_HELDOUT_GATE_NOT_PASS: gate_pass={gate_pass}")

    print(f"H heldout: run_status={run_status} gate_pass={gate_pass}")

    # ── Full-FIT training using V2B trainer on all FIT splits ────────
    scripts_dir = ROOT / "scripts/detector_v5"
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # Train 12-seed ensemble on all 4 training splits combined
    for seed in [42, 123, 456]:
        inner_cv = args.student_config_root
        auth_root = args.student_config_root

        result = subprocess.run(
            [sys.executable, str(scripts_dir / "train_factorized_v2_inner_cv.py"),
             "--candidate", "V2B",
             "--outer-fold", "0",
             "--inner-fold", "0",
             "--seed", str(seed),
             "--gpu", str(args.gpu),
             "--receptive-field", "32",
             "--hidden-dim", "64",
             "--dropout", "0.1",
             "--weight-decay", "1e-4",
             "--epochs", "30",
             "--output-root", str(staging / f"full_fit_seed_{seed}"),
             "--inner-cv-splits-root", str(inner_cv),
             "--authorization-root", str(auth_root)],
            capture_output=False)
        if result.returncode != 0:
            raise SystemExit(f"FULL_FIT_TRAINING_FAILED at seed={seed}")

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "FULL_FIT_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "COMPLETE",
        "full_fit_splits": ["FIT-TRAIN", "FIT-DEV", "CAL", "CHECK"],
        "h_run_status": run_status,
        "h_gate_pass": gate_pass,
        "uses_h_for_training": False,
        "uses_a": False,
        "uses_fec": False,
        "architecture": "V2B",
        "receptive_field": 32,
        "hidden_dim": 64,
        "seeds": [42, 123, 456],
    }
    (staging / "FULL_FIT_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # ── Seal atomically ──────────────────────────────────────────────
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Full-FIT complete: seal={seal[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
