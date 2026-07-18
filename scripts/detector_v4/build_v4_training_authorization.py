#!/usr/bin/env python3
"""Machine-build the corrected V4 FIT-fold training authorization.

The command accepts paths, not caller-provided digests.  It opens and verifies
the sealed roots, validates the frozen 800/600/200 identity geometry, measures
the current clean Git checkout, and only then writes a non-overwriting auth
bundle.  A hand-written JSON cannot satisfy this entry point.
"""

from __future__ import annotations

import argparse
import csv
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
from gripper_attack.v4_dataset import SUITES, FIT_STATES
from gripper_attack.v4_formal import V4Normalization


AUTH_SCHEMA = "DETECTOR_V4_TRAINING_AUTHORIZATION_V2"
VALID_CANDIDATES = {"C0", "C1", "C2", "C3"}


def _file_sha(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing input file: {path}")
    return sha256_file(path)


def _root_sha(path: Path) -> str:
    verify_checksum_manifest(path)
    return sha256_file(path / "SHA256SUMS")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_in_root(path: Path, root: Path, label: str) -> None:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside its sealed root") from exc
    listed = {line.split(maxsplit=1)[1].lstrip("*") for line in (root / "SHA256SUMS").read_text().splitlines() if line.strip()}
    if str(rel).replace(os.sep, "/") not in listed:
        raise ValueError(f"{label} is not covered by its root checksum manifest")


def _registry_rows(registry_csv: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(registry_csv.open(encoding="utf-8", newline="")))
    if len(rows) != 800:
        raise ValueError(f"formal FIT registry must have 800 rows, got {len(rows)}")
    identities = []
    for row in rows:
        identity = str(row.get("canonical_parent_key", ""))
        suite = str(row.get("suite", ""))
        task = int(row.get("task_idx", -1))
        state = int(row.get("state_id", -1))
        if state not in FIT_STATES or suite not in SUITES or task not in range(10):
            raise ValueError(f"registry row is outside FIT: {identity}")
        expected = f"{suite}/task_{task:02d}/state_{state:02d}"
        if identity != expected:
            raise ValueError(f"registry identity columns disagree: {identity}")
        if str(row.get("formal_selected", "")).lower() not in {"true", "1", "yes"}:
            raise ValueError(f"registry row not formally selected: {identity}")
        identities.append(identity)
    if len(set(identities)) != 800:
        raise ValueError("formal FIT registry has duplicate identities")
    if any(sum(1 for row in rows if row["suite"] == suite) != 200 for suite in SUITES):
        raise ValueError("formal FIT suite quotas are not 200 each")
    if any(sum(1 for row in rows if row["suite"] == suite and int(row["task_idx"]) == task) != 20 for suite in SUITES for task in range(10)):
        raise ValueError("formal FIT task quotas are not 20 each")
    return rows


def _fold_identities(fold_root: Path, fold_id: int) -> tuple[list[str], list[str]]:
    candidates = [
        fold_root / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json",
        fold_root / "fold_manifest.json",
        fold_root / "manifest.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError("fold manifest is missing")
    value = _json(path)
    row = next((v for v in value.get("folds", []) if int(v.get("fold_id", -1)) == fold_id), None)
    if row is None:
        raise ValueError(f"fold {fold_id} is missing")
    train = sorted(set(row.get("train_identities", [])))
    valid = sorted(set(row.get("validation_identities", [])))
    if len(train) != 600 or len(valid) != 200 or set(train) & set(valid):
        raise ValueError("fold must contain disjoint 600 train and 200 validation identities")
    return train, valid


def _verify_normalization(root: Path, fold_id: int, candidate: str, registry_sha: str, s1_sha: str, train_sha: str) -> dict[str, Any]:
    _root_sha(root)
    path = root / "normalization.json"
    payload = _json(path)
    if payload.get("schema") != "DETECTOR_V4_NORMALIZATION_V2":
        raise ValueError("wrong V4 normalization schema")
    if payload.get("fold_id") != fold_id or payload.get("view") not in {"A", "B", "C"}:
        raise ValueError("normalization fold/view mismatch")
    if payload.get("candidate", candidate) != candidate:
        raise ValueError("normalization candidate mismatch")
    norm = V4Normalization.from_dict(payload["normalization"])
    if payload.get("normalization_semantic_sha256") != norm.sha256:
        raise ValueError("normalization semantic SHA mismatch")
    if payload.get("feature_order_sha256") != FEATURE_ORDER_SHA256:
        raise ValueError("normalization feature-order SHA mismatch")
    if payload.get("registry_sha256") != registry_sha or payload.get("s1_root_sha256s_sha256") != s1_sha:
        raise ValueError("normalization source root mismatch")
    if payload.get("train_identity_sha256") != train_sha:
        raise ValueError("normalization train identity mismatch")
    binding = payload.get("runner_binding")
    if not isinstance(binding, dict) or binding.get("status") != "PASS" or binding.get("runner_worktree_clean") is not True:
        raise ValueError("normalization runner binding is not measured clean PASS")
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute_formal:
        raise ValueError("authorization generation requires --execute-formal")
    if args.candidate not in VALID_CANDIDATES:
        raise ValueError("unknown V4 candidate")
    for root in (args.registry_root, args.s1_root, args.fold_root, args.normalization_root):
        _root_sha(root)
    registry_sha = _root_sha(args.registry_root)
    s1_sha = _root_sha(args.s1_root)
    fold_sha = _root_sha(args.fold_root)
    registry_rows = _registry_rows(args.registry_csv)
    _assert_in_root(args.registry_csv, args.registry_root, "registry CSV")
    _assert_in_root(args.registry_summary, args.registry_root, "registry summary")
    summary = _json(args.registry_summary)
    if summary.get("formal_fit_ready") is not True or summary.get("formal_training_authorized") is not False:
        raise ValueError("registry summary is not a sealed formal FIT-ready preparation result")
    train_ids, valid_ids = _fold_identities(args.fold_root, args.fold_id)
    registry_ids = {row["canonical_parent_key"] for row in registry_rows}
    if set(train_ids) | set(valid_ids) != registry_ids:
        raise ValueError("fold universe differs from formal FIT registry")
    train_sha = identity_sha(train_ids)
    normalization = _verify_normalization(args.normalization_root, args.fold_id, args.candidate, registry_sha, s1_sha, train_sha)

    s1_audit = _json(args.s1_audit)
    teacher_aggregate = _json(args.teacher_aggregate)
    if s1_audit.get("status") != "PASS" or int(s1_audit.get("identity_count", 0)) != 800:
        raise ValueError("independent S1 root audit is not PASS for 800 identities")
    if teacher_aggregate.get("status") != "PASS" or int(teacher_aggregate.get("identity_count", teacher_aggregate.get("actual_identity_count", 0))) != 800:
        raise ValueError("Teacher aggregate is not PASS for 800 identities")
    _assert_in_root(args.s1_audit, args.s1_root, "S1 root audit")
    _assert_in_root(args.teacher_aggregate, args.s1_root, "Teacher aggregate")
    for path in (args.training_protocol, args.source_contract, args.protocol, args.feature_protocol, args.teacher_protocol):
        _file_sha(path)
    runner_paths = [args.runner_script, args.runner_config, str(Path(__file__).resolve().relative_to(args.runner_repo.resolve()).as_posix())]
    runner = measured_git_binding(args.runner_repo, runner_paths)
    input_snapshots = {
        "formal_fit_registry_sha256": _file_sha(args.registry_csv),
        "formal_registry_summary_sha256": _file_sha(args.registry_summary),
        "formal_registry_root_sha256": registry_sha,
        "s1_root_sha256": s1_sha,
        "s1_root_audit_sha256": _file_sha(args.s1_audit),
        "teacher_aggregate_sha256": _file_sha(args.teacher_aggregate),
        "training_protocol_sha256": _file_sha(args.training_protocol),
        "source_contract_sha256": _file_sha(args.source_contract),
        "protocol_sha256": _file_sha(args.protocol),
        "feature_protocol_sha256": _file_sha(args.feature_protocol),
        "teacher_protocol_sha256": _file_sha(args.teacher_protocol),
        "normalization_bundle_sha256": _root_sha(args.normalization_root),
        "normalization_sha256": normalization["normalization_semantic_sha256"],
        "fold_manifest_sha256": fold_sha,
    }
    payload = {
        "schema": AUTH_SCHEMA,
        "authorization_status": "PASS",
        "formal_fit_ready": True,
        "s1_materialization_status": "PASS",
        "teacher_aggregate_status": "PASS",
        "formal_training_authorized": True,
        "formal_attack_authorized": False,
        "model_selection_authorized": False,
        "candidate": args.candidate,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "fit_scope": "FIT_FOLD",
        "train_identity_sha256": train_sha,
        "validation_identity_sha256": identity_sha(valid_ids),
        "input_snapshots": input_snapshots,
        "runner_binding": runner,
        "authorization_generator": {"script": str(Path(__file__).resolve()), "script_sha256": _file_sha(Path(__file__).resolve())},
        "verification": {
            "status": "PASS",
            "registry_quota": "800=4x200=40x20",
            "fold_quota": "600_train+200_validation",
            "s1_root_audit": "PASS",
            "teacher_aggregate": "PASS",
            "normalization_recomputed_from_train_only": True,
            "protected_splits_read": False,
        },
    }
    payload["authorization_payload_sha256"] = json_sha(payload)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    staging = args.output_root.parent / f".{args.output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "authorization.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "input_snapshots.json").write_text(json.dumps(input_snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _seal(staging)
        os.replace(staging, args.output_root)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _seal(root: Path) -> None:
    payloads = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}), key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(p)}  {str(p.relative_to(root)).replace(os.sep, '/')}\n" for p in payloads), encoding="utf-8")
    value = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{value}  SHA256SUMS\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--execute-formal", action="store_true")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--candidate", choices=sorted(VALID_CANDIDATES), required=True)
    p.add_argument("--fold-id", type=int, choices=range(4), required=True)
    p.add_argument("--seed", type=int, choices=[20260717, 20260718, 20260719], required=True)
    p.add_argument("--registry-csv", type=Path, required=True)
    p.add_argument("--registry-summary", type=Path, required=True)
    p.add_argument("--registry-root", type=Path, required=True)
    p.add_argument("--s1-root", type=Path, required=True)
    p.add_argument("--s1-audit", type=Path, required=True)
    p.add_argument("--teacher-aggregate", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, required=True)
    p.add_argument("--normalization-root", type=Path, required=True)
    p.add_argument("--training-protocol", type=Path, required=True)
    p.add_argument("--source-contract", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--feature-protocol", type=Path, required=True)
    p.add_argument("--teacher-protocol", type=Path, required=True)
    p.add_argument("--runner-repo", type=Path, required=True)
    p.add_argument("--runner-script", required=True)
    p.add_argument("--runner-config", required=True)
    args = p.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
