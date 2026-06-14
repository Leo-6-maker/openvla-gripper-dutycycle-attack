#!/usr/bin/env python3
"""D1b.2: Fail-closed artifact SHA verification.

Reads expected SHAs from d1b_artifact_hashes.csv and compares against
runtime files. Any mismatch → nonzero exit. Called by training and
evaluation runners before any computation.
"""

import csv, hashlib, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "tables/deepseek_detector/d1b_artifact_hashes.csv"


def sha256_file(path):
    if not os.path.isfile(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all(root=None):
    """Compare runtime SHAs against frozen expected values.
    Returns (pass: bool, failures: list[str], results: dict)."""
    if root is None:
        root = ROOT
    manifest_path = os.path.join(str(root), MANIFEST)
    if not os.path.isfile(manifest_path):
        return False, [f"manifest not found: {manifest_path}"], {}

    expected = {}
    with open(manifest_path, "r") as f:
        for r in csv.DictReader(f):
            expected[r["artifact"]] = r["sha256"].strip()

    # Map artifact names to file paths (relative to repo root)
    path_map = {
        "training_manifest": "tables/deepseek_detector/d1b_training_manifest.csv",
        "split_manifest": "tables/deepseek_detector/d1b_split_manifest.csv",
        "split_summary": "tables/deepseek_detector/d1b_split_summary.csv",
        "leakage_audit": "tables/deepseek_detector/d1b_leakage_audit.csv",
        "feature_normalization": "tables/deepseek_detector/d1b_feature_normalization.csv",
        "candidate_table": "tables/e4c_audit/l12_e4c2b_close_candidates.csv",
        "training_config": "configs/d1b_detector_training.yaml",
        "train_runner": "scripts/stageb/train_d1b_detector.py",
        "eval_runner": "scripts/stageb/evaluate_d1b_detector.py",
        "protocol_tests": "tests/stageb/test_d1b_detector_protocol.py",
    }

    failures = []
    results = {}
    for name, rel_path in path_map.items():
        expected_sha = expected.get(name, "")
        if not expected_sha:
            failures.append(f"{name}: no expected SHA in manifest")
            continue
        abs_path = os.path.join(str(root), rel_path)
        actual = sha256_file(abs_path)
        results[name] = actual
        if actual == "MISSING":
            failures.append(f"{name}: file missing at {abs_path}")
        elif actual.lower() != expected_sha.lower():
            failures.append(
                f"{name}: SHA MISMATCH expected={expected_sha[:16]} actual={actual[:16]}"
            )

    # Check all expected artifacts were verified
    for name in expected:
        if name not in path_map:
            failures.append(f"{name}: in manifest but no path mapping (unchecked)")

    return len(failures) == 0, failures, results


def main():
    ok, failures, results = verify_all()
    print("=== D1b ARTIFACT VERIFICATION ===")
    for name, sha in results.items():
        print(f"  {name}: {sha[:16]}... OK")
    if not ok:
        print(f"\nFATAL: {len(failures)} SHA verification failure(s):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"  ALL {len(results)} ARTIFACTS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
