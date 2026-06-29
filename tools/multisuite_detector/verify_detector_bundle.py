#!/usr/bin/env python3
"""Fail-closed verification of frozen detector bundle."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

REQUIRED_FILES = [
    "detector_config.json", "feature_contract.json",
    "checkpoint.pt", "checkpoint_metadata.json",
    "FILES.json", "SHA256SUMS.txt",
]


def main():
    ap = argparse.ArgumentParser(description="Verify frozen detector bundle")
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--fail_fast", action="store_true")
    args = ap.parse_args()

    bundle = Path(args.bundle_dir)
    errors = []

    for name in REQUIRED_FILES:
        if not (bundle / name).exists():
            errors.append(f"MISSING: {name}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)

    files_json = json.loads((bundle / "FILES.json").read_text())
    sums_txt = (bundle / "SHA256SUMS.txt").read_text().strip().split("\n")
    declared = {}
    for line in sums_txt:
        sha, name = line.strip().split("  ", 1)
        declared[name] = sha

    for name, expected_sha in files_json.get("files", {}).items():
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual != expected_sha:
            errors.append(f"SHA MISMATCH: {name} expected={expected_sha[:16]} actual={actual[:16]}")
        if name in declared and declared[name] != expected_sha:
            errors.append(f"SUMS MISMATCH: {name}")

    import torch
    try:
        ckpt = torch.load(str(bundle / "checkpoint.pt"), map_location="cpu", weights_only=False)
        required_keys = ["model_state", "mean", "std", "feature_names", "phase_classes"]
        for k in required_keys:
            if k not in ckpt:
                errors.append(f"CHECKPOINT MISSING: {k}")
        if ckpt.get("split_mode") != "frozen":
            errors.append(f"split_mode={ckpt.get('split_mode')}, expected 'frozen'")
        if ckpt.get("normalization_source") != "train_only" and ckpt.get("normalization_source") is not None:
            errors.append(f"normalization_source={ckpt.get('normalization_source')}, expected 'train_only'")
    except Exception as e:
        errors.append(f"CHECKPOINT LOAD FAILED: {e}")

    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("VERIFICATION PASSED")
        json.dump({"gate": "BUNDLE_VERIFICATION_PASS", "errors": 0}, sys.stdout)


if __name__ == "__main__":
    main()
