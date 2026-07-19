"""Build a deterministic 4-suite x 10-task x 2-identity FIT subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from gripper_attack.v5_dataset import load_fit_registry
from gripper_attack.b3_training_protocol import load_fit_fold_bundle


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(registry_csv: Path, fold_root: Path, fold_id: int, output_root: Path) -> dict[str, object]:
    rows = load_fit_registry(registry_csv.resolve())
    fold = load_fit_fold_bundle(fold_root.resolve())
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == fold_id)
    train_keys = set(fold_row["train_identities"])
    selected: list[str] = []
    for suite in sorted({row["suite"] for row in rows}):
        for task in range(10):
            candidates = sorted(row["canonical_parent_key"] for row in rows if row["suite"] == suite and int(row["task_idx"]) == task and row["canonical_parent_key"] in train_keys)
            ranked = sorted(candidates, key=lambda key: hashlib.sha256(f"20260717:{key}".encode()).hexdigest())
            selected.extend(ranked[:2])
    if len(selected) != 80 or len(set(selected)) != 80:
        raise ValueError("stratified V5 smoke subset must contain 80 unique identities")
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        manifest = {
            "schema": "DETECTOR_V5_STRATIFIED_SMOKE_SUBSET_V1",
            "identity_count": 80,
            "per_suite_task_count": 2,
            "seed": 20260717,
            "fold_id": fold_id,
            "source_train_identity_sha256": fold_row["train_identity_sha256"],
            "identities": selected,
            "identity_sha256": _sha(selected),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "identities.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        from gripper_attack.b3_training_protocol import seal_directory
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.registry_csv, args.fold_root, args.fold_id, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
