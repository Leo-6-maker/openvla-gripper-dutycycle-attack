"""Sealed, preparation-only contracts for Official V3 training.

This module builds the immutable inputs used by the later trainer.  It does
not read live CLEAN data by itself and it never authorizes an attack.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .b3_formal import B3Normalization, B3_FEATURES_25D, json_sha


FOLD_VALIDATION_RANGES = {0: (0, 5), 1: (5, 10), 2: (10, 15), 3: (15, 20)}
VIABILITY_SEEDS = (20260717, 20260718, 20260719)
CHECKPOINT_STATUSES = (
    "ENGINEERING_SMOKE_ONLY",
    "FIT_FOLD_TRAINED_CANDIDATE",
    "FIT_VIABILITY_PASS",
    "FULL_FIT_REFIT_CANDIDATE",
    "FIT_DEV_SELECTED",
    "CALIBRATED",
    "CHECK_PASS",
    "ATTACK_CANARY_PASS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdefABCDEF" for c in value)


def _atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
    _atomic_text(path.with_name(path.name + ".sha256"), f"{sha256_file(path)}  {path.name}\n")


def seal_directory(root: Path) -> None:
    """Write an exact, non-recursive file closure for a flat contract root."""

    if (root / "SHA256SUMS").exists() or (root / "SHA256SUMS.sha256").exists():
        raise FileExistsError(f"checksum files already exist: {root}")
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    _atomic_text(root / "SHA256SUMS", "".join(f"{sha256_file(root / name)}  {name}\n" for name in names))
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def verify_sealed_directory(root: Path) -> None:
    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"sealed contract root is incomplete: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"SHA256SUMS sidecar mismatch: {root}")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not _is_sha(digest) or not name:
            raise ValueError(f"invalid checksum row: {root}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in listed:
            raise ValueError(f"unsafe/duplicate checksum path: {name}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest.lower():
            raise ValueError(f"checksum mismatch: {name}")
        listed[relative.as_posix()] = digest.lower()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.sha256"}
    if actual != expected:
        raise ValueError(f"sealed file-set mismatch: extra={sorted(actual - expected)} missing={sorted(expected - actual)}")


def _validate_fit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 800:
        raise ValueError(f"Official V3 FIT fold input must contain 800 rows, got {len(rows)}")
    expected: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("canonical_parent_key", ""))
        parts = key.split("/")
        if len(parts) != 3 or not parts[1].startswith("task_") or not parts[2].startswith("state_"):
            raise ValueError(f"invalid canonical identity: {key}")
        try:
            task = int(row["task_idx"])
            state = int(row["state_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid identity columns: {key}") from exc
        if key != f"{parts[0]}/task_{task:02d}/state_{state:02d}" or not 0 <= task < 10 or not 0 <= state < 20:
            raise ValueError(f"identity columns do not match canonical key: {key}")
        if row.get("split") != "FIT_TRAIN":
            raise ValueError(f"fold input contains non-FIT row: {key}")
        if key in expected:
            raise ValueError(f"duplicate FIT identity: {key}")
        expected.add(key)
        normalized.append(dict(row, task_idx=task, state_id=state, canonical_parent_key=key))
    return sorted(normalized, key=lambda item: item["canonical_parent_key"])


def build_fit_fold_manifest(rows: Sequence[Mapping[str, Any]], *, registry_sha256: str) -> dict[str, Any]:
    if not _is_sha(registry_sha256):
        raise ValueError("registry_sha256 must be a SHA-256 digest")
    normalized = _validate_fit_rows(rows)
    folds: list[dict[str, Any]] = []
    for fold_id, (start, end) in FOLD_VALIDATION_RANGES.items():
        valid = [row["canonical_parent_key"] for row in normalized if start <= row["state_id"] < end]
        train = [row["canonical_parent_key"] for row in normalized if row["canonical_parent_key"] not in set(valid)]
        if len(train) != 600 or len(valid) != 200:
            raise ValueError(f"fold {fold_id} count mismatch: train={len(train)} valid={len(valid)}")
        folds.append({
            "fold_id": fold_id,
            "validation_state_range": [start, end - 1],
            "train_state_ids": [state for state in range(20) if not start <= state < end],
            "validation_state_ids": list(range(start, end)),
            "train_identity_count": len(train),
            "validation_identity_count": len(valid),
            "train_identity_sha256": json_sha(train),
            "validation_identity_sha256": json_sha(valid),
            "train_identities": train,
            "validation_identities": valid,
        })
    return {
        "schema": "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1",
        "registry_sha256": registry_sha256,
        "feature_order_sha256": json_sha(list(B3_FEATURES_25D)),
        "fit_identity_count": 800,
        "fold_count": 4,
        "folds": folds,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def write_fit_fold_bundle(output_root: Path, manifest: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    if manifest.get("schema") != "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1" or manifest.get("fold_count") != 4:
        raise ValueError("invalid fold manifest")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json", dict(manifest))
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_fit_fold_bundle(root: Path) -> dict[str, Any]:
    verify_sealed_directory(root)
    path = root / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1":
        raise ValueError("unexpected fold manifest schema")
    # Rebuild the count/hash invariants from the sealed lists.
    all_keys: set[str] = set()
    universe: set[str] | None = None
    for fold in value.get("folds", []):
        train = list(fold.get("train_identities", []))
        valid = list(fold.get("validation_identities", []))
        if len(train) != 600 or len(valid) != 200 or set(train) & set(valid):
            raise ValueError("fold train/validation closure failed")
        if json_sha(train) != fold.get("train_identity_sha256") or json_sha(valid) != fold.get("validation_identity_sha256"):
            raise ValueError("fold identity checksum mismatch")
        fold_universe = set(train) | set(valid)
        if universe is None:
            universe = fold_universe
        elif fold_universe != universe:
            raise ValueError("folds do not share one exact FIT identity universe")
        all_keys.update(fold_universe)
        task_counts: dict[str, int] = {}
        for key in valid:
            task = "/".join(key.split("/")[:2])
            task_counts[task] = task_counts.get(task, 0) + 1
        if any(count != 5 for count in task_counts.values()) or len(task_counts) != 40:
            raise ValueError("fold validation task quota mismatch")
    if len(value.get("folds", [])) != 4 or len(all_keys) != 800:
        raise ValueError("fold manifest does not cover the exact FIT universe")
    return value


def write_normalization_bundle(
    output_root: Path,
    normalization: B3Normalization,
    *,
    fold_id: int,
    variant: str,
    train_identity_sha256: str,
    registry_sha256: str,
    s1_corpus_sha256: str,
    runner_binding: Mapping[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    if fold_id not in FOLD_VALIDATION_RANGES or variant not in ("B3_25D", "B3_25D9D"):
        raise ValueError("invalid normalization fold or variant")
    for name, value in (("train_identity_sha256", train_identity_sha256), ("registry_sha256", registry_sha256), ("s1_corpus_sha256", s1_corpus_sha256)):
        if not _is_sha(value):
            raise ValueError(f"invalid normalization binding: {name}")
    if not isinstance(runner_binding, Mapping) or runner_binding.get("status") != "PASS":
        raise ValueError("normalization requires a PASS runner binding")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "normalization.json", {
            "schema": "B3_OFFICIAL_V3_NORMALIZATION_BUNDLE_V1",
            "normalization": normalization.to_dict(),
            "normalization_sha256": normalization.sha256,
        })
        _write_json(staging / "source_manifest.json", {
            "schema": "B3_OFFICIAL_V3_NORMALIZATION_SOURCE_MANIFEST_V1",
            "fold_id": fold_id,
            "variant": variant,
            "train_identity_sha256": train_identity_sha256,
            "registry_sha256": registry_sha256,
            "s1_corpus_sha256": s1_corpus_sha256,
            "feature_order_sha256": json_sha(list(B3_FEATURES_25D)),
            "runner_binding": dict(runner_binding),
            "normalization_sha256": normalization.sha256,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        })
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_normalization_bundle(root: Path, *, fold_id: int | None = None, variant: str | None = None) -> tuple[B3Normalization, dict[str, Any]]:
    verify_sealed_directory(root)
    value = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    source = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    normalization = B3Normalization.from_dict(value["normalization"])
    if value.get("schema") != "B3_OFFICIAL_V3_NORMALIZATION_BUNDLE_V1" or value.get("normalization_sha256") != normalization.sha256:
        raise ValueError("normalization bundle hash/schema mismatch")
    if source.get("normalization_sha256") != normalization.sha256 or source.get("feature_order_sha256") != json_sha(list(B3_FEATURES_25D)):
        raise ValueError("normalization source binding mismatch")
    if fold_id is not None and source.get("fold_id") != fold_id:
        raise ValueError("normalization fold mismatch")
    if variant is not None and source.get("variant") != variant:
        raise ValueError("normalization variant mismatch")
    return normalization, source


def build_training_authorization(
    output_root: Path,
    *,
    variant: str,
    fold_id: int,
    seed: int,
    input_snapshots: Mapping[str, str],
    runner_binding: Mapping[str, Any],
    generator_script_sha256: str,
) -> dict[str, Any]:
    if fold_id not in FOLD_VALIDATION_RANGES or variant not in ("B3_25D", "B3_25D9D") or seed not in VIABILITY_SEEDS:
        raise ValueError("invalid authorization matrix coordinates")
    if not _is_sha(generator_script_sha256) or not all(_is_sha(value) for value in input_snapshots.values()):
        raise ValueError("authorization input snapshots must be SHA-256 digests")
    required_snapshots = {
        "formal_fit_registry_sha256", "formal_registry_summary_sha256", "formal_registry_root_sha256",
        "s1_corpus_sha256", "s1_root_audit_sha256", "teacher_aggregate_sha256",
        "training_protocol_sha256", "source_contract_sha256", "protocol_sha256", "feature_rebuilder_sha256",
        "normalization_bundle_sha256", "normalization_sha256", "fold_manifest_sha256",
    }
    if set(input_snapshots) != required_snapshots:
        raise ValueError(f"authorization input snapshot set mismatch: {sorted(set(input_snapshots) ^ required_snapshots)}")
    if runner_binding.get("status") != "PASS" or runner_binding.get("runner_worktree_clean") is not True:
        raise ValueError("authorization requires a PASS runner binding")
    if runner_binding.get("runner_binding_sha256") != json_sha({key: value for key, value in runner_binding.items() if key != "runner_binding_sha256"}):
        raise ValueError("authorization runner binding SHA is invalid")
    if not isinstance(runner_binding.get("runner_head"), str) or len(runner_binding["runner_head"]) != 40:
        raise ValueError("authorization runner head is invalid")
    if output_root.exists():
        raise FileExistsError(output_root)
    generated = {
        "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_GENERATOR_V1",
        "generator_script_sha256": generator_script_sha256,
        "generator_worktree_clean": True,
        "generator_entrypoint": "build_b3_v3_training_authorization.py",
    }
    payload: dict[str, Any] = {
        "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_V1",
        "authorization_status": "PASS",
        "formal_fit_ready": True,
        "s1_materialization_status": "PASS",
        "teacher_aggregate_status": "PASS",
        "formal_training_authorized": True,
        "formal_attack_authorized": False,
        "variant": variant,
        "fold_id": fold_id,
        "seed": seed,
        "runner_head": runner_binding.get("runner_head"),
        "input_snapshots": dict(sorted(input_snapshots.items())),
        "runner_binding": dict(runner_binding),
        "authorization_generation": generated,
        "formal_fit_registry_sha256": input_snapshots["formal_fit_registry_sha256"],
        "formal_registry_summary_sha256": input_snapshots["formal_registry_summary_sha256"],
        "formal_registry_root_sha256": input_snapshots["formal_registry_root_sha256"],
        "s1_corpus_sha256": input_snapshots["s1_corpus_sha256"],
        "s1_root_audit_sha256": input_snapshots["s1_root_audit_sha256"],
        "teacher_aggregate_sha256": input_snapshots["teacher_aggregate_sha256"],
        "training_protocol_sha256": input_snapshots["training_protocol_sha256"],
        "source_contract_sha256": input_snapshots["source_contract_sha256"],
        "protocol_sha256": input_snapshots["protocol_sha256"],
        "feature_rebuilder_sha256": input_snapshots["feature_rebuilder_sha256"],
        "normalization_bundle_sha256": input_snapshots["normalization_bundle_sha256"],
        "normalization_sha256": input_snapshots["normalization_sha256"],
        "fold_manifest_sha256": input_snapshots["fold_manifest_sha256"],
    }
    payload["authorization_payload_sha256"] = json_sha(payload)
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "authorization.json", payload)
        _write_json(staging / "input_snapshots.json", {"schema": "B3_OFFICIAL_V3_AUTHORIZATION_INPUT_SNAPSHOTS_V1", **payload["input_snapshots"]})
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def load_training_authorization_bundle(root: Path) -> dict[str, Any]:
    verify_sealed_directory(root)
    payload = json.loads((root / "authorization.json").read_text(encoding="utf-8"))
    expected = payload.get("authorization_payload_sha256")
    body = dict(payload)
    body.pop("authorization_payload_sha256", None)
    if not _is_sha(expected) or expected != json_sha(body):
        raise ValueError("authorization payload hash mismatch")
    from .b3_formal import validate_training_authorization
    validate_training_authorization(payload)
    return payload


__all__ = [
    "FOLD_VALIDATION_RANGES", "VIABILITY_SEEDS", "CHECKPOINT_STATUSES", "sha256_file",
    "seal_directory", "verify_sealed_directory", "build_fit_fold_manifest", "write_fit_fold_bundle",
    "load_fit_fold_bundle", "write_normalization_bundle", "load_normalization_bundle",
    "build_training_authorization", "load_training_authorization_bundle",
]
