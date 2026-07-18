#!/usr/bin/env python3
"""Build a sealed, fold-isolated V4 normalization bundle.

This command is deliberately boring: it reads only the selected FIT train
identities, recomputes the normalization from those episodes, and writes a
non-overwriting bundle.  It never reads FIT-DEV/CAL/CHECK or any attack root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from gripper_attack.v4_contract import (
    FEATURE_ORDER_SHA256,
    identity_sha,
    json_sha,
    measured_git_binding,
    sha256_file,
    verify_checksum_manifest,
)
from gripper_attack.v4_dataset import (
    FIT_STATES,
    SUITES,
    compute_v4_fold_normalization,
    load_v4_episode,
    select_fold_episodes,
)


def _seal(root: Path) -> dict[str, str]:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256_file(p)}  {str(p.relative_to(root)).replace(os.sep, '/') }\n"
            for p in payloads
        ),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "sidecar_sha256": sha256_file(sidecar)}


def _load_fold(fold_root: Path, fold_id: int) -> tuple[list[str], list[str], str]:
    seal = verify_checksum_manifest(fold_root)
    candidates = [
        fold_root / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json",
        fold_root / "fold_manifest.json",
        fold_root / "manifest.json",
    ]
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if manifest_path is None:
        raise ValueError(f"no fold manifest in {fold_root}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = value.get("folds", [])
    row = next((item for item in folds if int(item.get("fold_id", -1)) == fold_id), None)
    if row is None:
        raise ValueError(f"fold {fold_id} missing from {manifest_path}")
    train = sorted(set(str(v) for v in row.get("train_identities", [])))
    valid = sorted(set(str(v) for v in row.get("validation_identities", [])))
    if len(train) != 600 or len(valid) != 200 or set(train) & set(valid):
        raise ValueError(f"invalid fold cardinality/disjointness for fold {fold_id}")
    if value.get("feature_order_sha256") not in (None, FEATURE_ORDER_SHA256):
        raise ValueError("fold feature-order SHA mismatch")
    return train, valid, seal["sha256sums_sha256"]


def build_normalization(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    s1_seal = verify_checksum_manifest(args.s1_root)
    teacher_seal = verify_checksum_manifest(args.teacher_root)
    train_ids, valid_ids, fold_seal = _load_fold(args.fold_root, args.fold_id)
    if set(train_ids) | set(valid_ids) != {
        f"{suite}/task_{task:02d}/state_{state:02d}"
        for suite in SUITES for task in range(10) for state in FIT_STATES
    }:
        raise ValueError("fold manifest is not the frozen 800-identity FIT universe")
    episodes = []
    for identity in train_ids:
        suite, task_name, state_name = identity.split("/")
        ep = load_v4_episode(
            args.s1_root, args.teacher_root, suite,
            int(task_name.split("_", 1)[1]), int(state_name.split("_", 1)[1]), args.view,
        )
        if ep is None:
            raise ValueError(f"missing episode for {identity}")
        episodes.append(ep)
    if len(episodes) != 600:
        raise ValueError(f"expected 600 train episodes, got {len(episodes)}")
    normalization = compute_v4_fold_normalization(episodes, args.view)
    runner = measured_git_binding(args.runner_repo, [args.runner_script])
    runner_sha = json_sha(runner)
    train_sha = identity_sha(train_ids)
    staging = args.output_root.parent / f".{args.output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        payload = {
            "schema": "DETECTOR_V4_NORMALIZATION_V2",
            "view": args.view,
            "candidate": args.candidate,
            "fold_id": args.fold_id,
            "feature_order_sha256": FEATURE_ORDER_SHA256,
            "normalization": normalization.to_dict(),
            "normalization_semantic_sha256": normalization.sha256,
            "registry_sha256": args.registry_sha256,
            "s1_root_sha256s_sha256": s1_seal["sha256sums_sha256"],
            "teacher_root_sha256s_sha256": teacher_seal["sha256sums_sha256"],
            "fold_bundle_sha256": fold_seal,
            "train_identity_sha256": train_sha,
            "validation_identity_sha256": identity_sha(valid_ids),
            "runner_binding": runner,
            "runner_binding_sha256": runner_sha,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        (staging / "normalization.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "source_manifest.json").write_text(
            json.dumps({"train_identities": train_ids, "validation_identities": valid_ids}, indent=2) + "\n",
            encoding="utf-8",
        )
        # The bundle digest is the final SHA256SUMS digest, not a field inside a
        # payload (which would create a self-referential seal).  No payload is
        # written after this point.
        _seal(staging)
        os.replace(staging, args.output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, choices=range(4), required=True)
    parser.add_argument("--view", choices=["A", "B", "C"], required=True)
    parser.add_argument("--candidate", choices=["C0", "C1", "C2", "C3"], required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--runner-script", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_normalization(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
