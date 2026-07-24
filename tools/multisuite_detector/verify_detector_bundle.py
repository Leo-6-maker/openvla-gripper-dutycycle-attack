#!/usr/bin/env python3
"""Fail-closed verification of frozen detector bundle."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

REQUIRED_FILES = [
    "detector_config.json", "feature_contract.json",
    "data_contract.json", "normalization.json",
    "checkpoint.pt", "checkpoint_metadata.json",
    "FILES.json", "SHA256SUMS.txt",
]


def load_sums(path: Path) -> dict[str, str]:
    declared = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            sha, name = line.strip().split("  ", 1)
        except ValueError:
            declared[line] = ""
            continue
        declared[name] = sha
    return declared


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
    files = files_json.get("files", {})
    declared = load_sums(bundle / "SHA256SUMS.txt")

    for name in REQUIRED_FILES:
        if name not in files:
            errors.append(f"FILES MISSING ENTRY: {name}")
        if name not in declared:
            errors.append(f"SUMS MISSING ENTRY: {name}")

    for name in sorted(set(files) | set(declared)):
        p = bundle / name
        if not p.exists():
            errors.append(f"DECLARED FILE MISSING: {name}")
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        expected_sha = files.get(name)
        if expected_sha is None:
            errors.append(f"FILES MISSING ENTRY: {name}")
        elif actual != expected_sha:
            errors.append(f"SHA MISMATCH: {name} expected={expected_sha[:16]} actual={actual[:16]}")
        declared_sha = declared.get(name)
        if declared_sha is None:
            errors.append(f"SUMS MISSING ENTRY: {name}")
        elif actual != declared_sha:
            errors.append(f"SHA256SUMS MISMATCH: {name} expected={declared_sha[:16]} actual={actual[:16]}")
        if expected_sha and declared_sha and declared_sha != expected_sha:
            errors.append(f"FILES/SUMS DISAGREE: {name}")

    try:
        import torch
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
