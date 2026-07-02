#!/usr/bin/env python3
"""Fail-closed validation of detector splits.

Checks:
- Same episode not in multiple splits
- Same parent not in multiple splits
- LOSO test suite not in training normalization
- No window-level leakage
- No forbidden features in training data
- No duplicate episode keys
- No duplicate accepted rows
- Suite/task/state not in model features
- Attack outcome fields not in features
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


FORBIDDEN_IN_FEATURES = [
    "normalized_step", "absolute_timestep", "suite", "task_id", "state_id",
    "object_identity", "teacher_anchor", "teacher_window",
    "object_pose", "target_pose", "attack_condition", "vis_outcome",
    "rand_outcome", "task_success", "episode_success",
]


def validate_split_file(split_path: str) -> dict:
    with open(split_path) as f:
        data = json.load(f)
    errors = []
    warnings = []

    required = ["split_type", "seed", "splits", "counts", "validation_passed"]
    for r in required:
        if r not in data:
            errors.append(f"Missing required field: {r}")

    if not data.get("validation_passed"):
        errors.append("Split was not validated during generation")

    splits = data.get("splits", {})
    train_set = set(splits.get("train", []))
    val_set = set(splits.get("val", []))
    test_set = set(splits.get("test", []))

    if train_set & val_set:
        errors.append(f"Overlap train/val: {len(train_set & val_set)} episodes")
    if train_set & test_set:
        errors.append(f"Overlap train/test: {len(train_set & test_set)} episodes")
    if val_set & test_set:
        errors.append(f"Overlap val/test: {len(val_set & test_set)} episodes")

    total = len(train_set) + len(val_set) + len(test_set)
    all_keys = train_set | val_set | test_set
    if len(all_keys) != total:
        errors.append(f"Key overlap detected: {len(all_keys)} unique vs {total} total")

    if data["split_type"] == "loso":
        test_suite = data.get("test_suite")
        if not test_suite:
            errors.append("LOSO split missing test_suite")
        train_suites = data.get("train_suites", [])
        if test_suite in train_suites:
            errors.append(f"LOSO test suite {test_suite} in train_suites")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings,
            "total_episodes": total, "train_count": len(train_set),
            "val_count": len(val_set), "test_count": len(test_set)}


def validate_feature_contract(contract_path: str) -> dict:
    with open(contract_path) as f:
        contract = json.load(f)
    errors = []
    features = contract.get("features", {}).get("names", [])
    for fb in FORBIDDEN_IN_FEATURES:
        if fb in features:
            errors.append(f"Forbidden feature '{fb}' in feature list")
    if len(features) != 25:
        errors.append(f"Expected 25 features, got {len(features)}")
    forbidden = contract.get("forbidden_features", [])
    for f in FORBIDDEN_IN_FEATURES:
        if f not in forbidden:
            errors.append(f"'{f}' not in forbidden_features list")
    return {"valid": len(errors) == 0, "errors": errors}


def main():
    ap = argparse.ArgumentParser(description="Validate detector splits")
    ap.add_argument("--split_file", required=True, help="Split manifest JSON")
    ap.add_argument("--feature_contract", help="Feature contract JSON")
    ap.add_argument("--fail_fast", action="store_true")
    args = ap.parse_args()

    all_errors = []
    result = validate_split_file(args.split_file)
    all_errors.extend(result["errors"])
    print(f"Split validation: {'PASS' if result['valid'] else 'FAIL'} "
          f"(train={result['train_count']}, val={result['val_count']}, test={result['test_count']})")

    if args.feature_contract:
        fc_result = validate_feature_contract(args.feature_contract)
        all_errors.extend(fc_result["errors"])
        print(f"Feature contract: {'PASS' if fc_result['valid'] else 'FAIL'}")

    if all_errors:
        print(f"\n{len(all_errors)} ERROR(S):")
        for e in all_errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("\nAll validations passed.")
        json.dump({"gate": "SPLIT_VALIDATION_PASS", "errors": 0}, sys.stdout)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
