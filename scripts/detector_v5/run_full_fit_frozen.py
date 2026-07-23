#!/usr/bin/env python3
"""Stage 8: Full-FIT — retrain final detector on FIT-TRAIN+DEV+CAL+CHECK with frozen config.

Does NOT use H, A, or FEC states. Frozen architecture/hyperparameters from Stage 3.
Only runs if Stage 7 (H heldout) PASS receipt is present.
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
    ap.add_argument("--teacher-labels-root", type=Path, required=True,
                    help="Stage 1 output root (contains fit_train, fit_dev, cal, check)")
    ap.add_argument("--student-config-root", type=Path, required=True,
                    help="Stage 3 output root (contains frozen checkpoint, model config, architecture)")
    ap.add_argument("--stage-7-h-receipt-root", type=Path, required=True,
                    help="Stage 7b H heldout evaluation receipt (must be PASS)")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # ── Verify H receipt is PASS ─────────────────────────────────────
    h_receipt = json.loads((args.stage_7_h_receipt_root / "receipt.json").read_text())
    if h_receipt.get("status") != "PASS":
        raise SystemExit("H_HELDOUT_NOT_PASS: Full-FIT requires PASS on H heldout evaluation")

    # ── Load frozen config from Stage 3 ──────────────────────────────
    config_file = args.student_config_root / "frozen_config.json"
    if not config_file.is_file():
        raise SystemExit(f"FROZEN_CONFIG_MISSING: {config_file}")
    frozen_config = json.loads(config_file.read_text())

    # Build Full-FIT data: FIT-TRAIN + FIT-DEV + CAL + CHECK
    fit_splits = ["fit_train", "fit_dev", "cal", "check"]
    teacher_roots = [str(args.teacher_labels_root / s) for s in fit_splits]

    print(f"Full-FIT: training on {'+'.join(fit_splits)} with frozen config from Stage 3")
    print(f"  Architecture: {frozen_config.get('architecture', 'UNKNOWN')}")
    print(f"  H receipt: PASS")

    # ── Execute Full-FIT training ────────────────────────────────────
    scripts_dir = ROOT / "scripts/detector_v5"
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "train_factorized_v2_recommended_canary.py"),
         "--teacher-labels-roots", *teacher_roots,
         "--frozen-config", str(config_file),
         "--full-fit-mode", "true",
         "--output-root", str(out_root / "checkpoint")],
        capture_output=False)
    if result.returncode != 0:
        raise SystemExit("FULL_FIT_TRAINING_FAILED")

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "FULL_FIT_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "COMPLETE",
        "full_fit_splits": fit_splits,
        "frozen_config_sha256": sha256_file(config_file),
        "h_receipt_status": h_receipt.get("status"),
        "uses_h_for_training": False,
        "uses_a": False,
        "uses_fec": False,
        "architecture_frozen": True,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FULL_FIT_RECEIPT_V1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print("Full-FIT complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
