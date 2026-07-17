"""Sealed, preparation-only contracts for Official V3 training.

This module builds the immutable inputs used by the later trainer.  It does
not read live CLEAN data by itself and it never authorizes an attack.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .b3_formal import AUTHORIZATION_INPUT_NAMES, B3Normalization, B3_FEATURES_25D, json_sha


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
    fold_id: int | str,
    variant: str,
    train_identity_sha256: str,
    registry_sha256: str,
    s1_corpus_sha256: str,
    runner_binding: Mapping[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    if (fold_id not in FOLD_VALIDATION_RANGES and fold_id != "FULL_FIT") or variant not in ("B3_25D", "B3_25D9D"):
        raise ValueError("invalid normalization fold or variant")
    for name, value in (("train_identity_sha256", train_identity_sha256), ("registry_sha256", registry_sha256), ("s1_corpus_sha256", s1_corpus_sha256)):
        if not _is_sha(value):
            raise ValueError(f"invalid normalization binding: {name}")
    _validate_runner_binding_record(runner_binding)
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
            "fit_scope": "FULL_FIT" if fold_id == "FULL_FIT" else "FIT_FOLD",
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


def load_normalization_bundle(root: Path, *, fold_id: int | str | None = None, variant: str | None = None) -> tuple[B3Normalization, dict[str, Any]]:
    verify_sealed_directory(root)
    value = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    source = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    normalization = B3Normalization.from_dict(value["normalization"])
    if value.get("schema") != "B3_OFFICIAL_V3_NORMALIZATION_BUNDLE_V1" or value.get("normalization_sha256") != normalization.sha256:
        raise ValueError("normalization bundle hash/schema mismatch")
    if source.get("normalization_sha256") != normalization.sha256 or source.get("feature_order_sha256") != json_sha(list(B3_FEATURES_25D)):
        raise ValueError("normalization source binding mismatch")
    _validate_runner_binding_record(source.get("runner_binding", {}))
    if source.get("formal_training_authorized") is not False or source.get("formal_attack_authorized") is not False:
        raise ValueError("normalization bundle cannot authorize training or attack")
    if fold_id is not None and source.get("fold_id") != fold_id:
        raise ValueError("normalization fold mismatch")
    if fold_id is not None and source.get("fit_scope") != ("FULL_FIT" if fold_id == "FULL_FIT" else "FIT_FOLD"):
        raise ValueError("normalization fit scope mismatch")
    if variant is not None and source.get("variant") != variant:
        raise ValueError("normalization variant mismatch")
    return normalization, source


def _snapshot_root(path: Path, name: str) -> Path:
    """Resolve a root input without allowing a directory to be hashed loosely."""

    path = path.resolve()
    if path.is_dir():
        root = path
    elif path.name == "SHA256SUMS" and path.is_file():
        root = path.parent
    else:
        raise ValueError(f"{name} must be a sealed root or its SHA256SUMS file: {path}")
    if not (root / "SHA256SUMS").is_file():
        raise ValueError(f"{name} is not a sealed root: {root}")
    return root


def _snapshot_file(path: Path, name: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name} must be a file: {path}")
    return path


def _snapshot_sha(name: str, path: Path) -> str:
    if name in {"formal_registry_root_sha256", "s1_corpus_sha256", "normalization_bundle_sha256"}:
        root = _snapshot_root(path, name)
        verify_sealed_directory(root)
        return sha256_file(root / "SHA256SUMS")
    if name == "fold_manifest_sha256":
        root = _snapshot_root(path, name) if path.is_dir() or path.name == "SHA256SUMS" else None
        if root is not None:
            verify_sealed_directory(root)
            return sha256_file(root / "SHA256SUMS")
    if name == "normalization_sha256" and path.is_dir():
        path = path / "normalization.json"
    return sha256_file(_snapshot_file(path, name))


def _validate_runner_binding_record(runner_binding: Mapping[str, Any]) -> None:
    if not isinstance(runner_binding, Mapping) or runner_binding.get("status") != "PASS" or runner_binding.get("runner_worktree_clean") is not True:
        raise ValueError("runner binding must be a measured clean PASS record")
    expected = json_sha({key: value for key, value in runner_binding.items() if key != "runner_binding_sha256"})
    if runner_binding.get("runner_binding_sha256") != expected:
        raise ValueError("runner binding SHA is invalid")
    for name in ("runner_head", "runner_script_git_blob_sha1", "config_git_blob_sha1"):
        value = runner_binding.get(name)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"runner binding field is missing: {name}")


def _measure_generator_provenance(*, runner_repo: Path, generator_script: Path, expected_head: str) -> dict[str, Any]:
    repo = runner_repo.resolve()
    script = generator_script.resolve()
    if len(expected_head) != 40 or any(c not in "0123456789abcdefABCDEF" for c in expected_head):
        raise ValueError("expected runner HEAD must be a full Git SHA")
    actual_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual_head != expected_head:
        raise ValueError("authorization generator HEAD does not match expected runner HEAD")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty.strip():
        raise ValueError("authorization generator worktree is dirty")
    try:
        relative = script.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError("authorization generator must be inside runner repository") from exc
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative], text=True, capture_output=True)
    if tracked.returncode != 0:
        raise ValueError("authorization generator is not tracked by the expected HEAD")
    committed = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"], stderr=subprocess.STDOUT)
    committed_sha = hashlib.sha256(committed).hexdigest()
    actual_sha = sha256_file(script)
    if actual_sha != committed_sha:
        raise ValueError("authorization generator bytes differ from the expected HEAD")
    blob = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{relative}"], text=True).strip()
    return {
        "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_GENERATOR_V1",
        "generator_script_sha256": actual_sha,
        "generator_script_git_blob_sha1": blob,
        "generator_head": actual_head,
        "generator_worktree_clean": True,
        "generator_script_tracked": True,
        "generator_entrypoint": Path(relative).name,
        "generator_script_git_path": relative,
        "semantic_inputs_verified": True,
    }


def _build_training_authorization_from_verified(
    output_root: Path,
    *,
    variant: str,
    fold_id: int | str,
    seed: int,
    fit_scope: str,
    input_snapshots: Mapping[str, str],
    runner_binding: Mapping[str, Any],
    generator_provenance: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    if fit_scope not in ("FIT_FOLD", "FULL_FIT") or (fit_scope == "FIT_FOLD" and fold_id not in FOLD_VALIDATION_RANGES) or (fit_scope == "FULL_FIT" and fold_id != "FULL_FIT") or variant not in ("B3_25D", "B3_25D9D") or seed not in VIABILITY_SEEDS:
        raise ValueError("invalid authorization matrix coordinates")
    if set(input_snapshots) != set(AUTHORIZATION_INPUT_NAMES) or not all(_is_sha(value) for value in input_snapshots.values()):
        raise ValueError("authorization input snapshot set or digest is invalid")
    if verification.get("status") != "PASS" or verification.get("semantic_inputs_verified") is not True:
        raise ValueError("authorization requires a PASS semantic evidence audit")
    if runner_binding.get("status") != "PASS" or runner_binding.get("runner_worktree_clean") is not True:
        raise ValueError("authorization requires a measured PASS runner binding")
    if runner_binding.get("runner_binding_sha256") != json_sha({key: value for key, value in runner_binding.items() if key != "runner_binding_sha256"}):
        raise ValueError("authorization runner binding SHA is invalid")
    if generator_provenance.get("generator_worktree_clean") is not True or generator_provenance.get("generator_script_tracked") is not True:
        raise ValueError("authorization generator provenance is incomplete")
    if output_root.exists():
        raise FileExistsError(output_root)
    payload: dict[str, Any] = {
        "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_V1",
        "authorization_status": "PASS",
        "formal_fit_ready": True,
        "s1_materialization_status": "PASS",
        "teacher_aggregate_status": "PASS",
        "formal_training_authorized": True,
        "formal_attack_authorized": False,
        "variant": variant,
        "fit_scope": fit_scope,
        "fold_id": fold_id,
        "seed": seed,
        "runner_head": runner_binding.get("runner_head"),
        "input_snapshots": dict(input_snapshots),
        "runner_binding": dict(runner_binding),
        "authorization_generation": dict(generator_provenance),
        "verification": dict(verification),
    }
    payload.update(input_snapshots)
    payload["authorization_payload_sha256"] = json_sha(payload)
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


def build_training_authorization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Disable the former raw-SHA constructor at the formal trust boundary."""

    raise ValueError("raw authorization construction is disabled; use build_training_authorization_from_paths")


def build_training_authorization_from_paths(
    output_root: Path,
    *,
    variant: str,
    fold_id: int | str,
    seed: int,
    fit_scope: str = "FIT_FOLD",
    input_paths: Mapping[str, Path],
    runner_repo: Path,
    expected_runner_head: str,
    runner_config: Path,
    runner_script: Path,
    generator_script: Path,
    policy_intent_root: Path | None = None,
) -> dict[str, Any]:
    """Measure and semantically audit every input before writing authorization."""

    if set(input_paths) != set(AUTHORIZATION_INPUT_NAMES):
        missing = sorted(set(AUTHORIZATION_INPUT_NAMES) - set(input_paths))
        extra = sorted(set(input_paths) - set(AUTHORIZATION_INPUT_NAMES))
        raise ValueError(f"authorization input set mismatch: missing={missing} extra={extra}")
    paths = {name: Path(path).resolve() for name, path in input_paths.items()}
    snapshots = {name: _snapshot_sha(name, paths[name]) for name in AUTHORIZATION_INPUT_NAMES}
    from .b3_official_v3_s1 import (
        audit_materialized_root, build_s1_runner_binding, load_formal_fit_registry, load_s1_protocol,
        verify_checksum_manifest,
    )
    from .b3_v3_dataset import compute_fit_normalization, load_episode, select_fit_fold_episodes
    from .official_v3_contract import load_contract

    registry_root = _snapshot_root(paths["formal_registry_root_sha256"], "formal_registry_root_sha256")
    s1_root = _snapshot_root(paths["s1_corpus_sha256"], "s1_corpus_sha256")
    normalization_root = _snapshot_root(paths["normalization_bundle_sha256"], "normalization_bundle_sha256")
    fold_root = _snapshot_root(paths["fold_manifest_sha256"], "fold_manifest_sha256")
    verify_checksum_manifest(registry_root)
    verify_checksum_manifest(s1_root)
    registry_csv = _snapshot_file(paths["formal_fit_registry_sha256"], "formal_fit_registry_sha256")
    registry_summary = _snapshot_file(paths["formal_registry_summary_sha256"], "formal_registry_summary_sha256")
    rows = load_formal_fit_registry(registry_csv, registry_summary)
    if len(rows) != 800 or any(row.get("formal_selected") not in (True, "True", "true", 1, "1") for row in rows):
        raise ValueError("formal FIT registry is not exactly 800 selected identities")
    runner_binding = build_s1_runner_binding(
        runner_repo=runner_repo, expected_runner_head=expected_runner_head,
        config_path=runner_config, runner_script_path=runner_script,
    )
    root_report = audit_materialized_root(
        s1_root, rows, require_runner_binding=True,
        feature_order_sha256=json_sha(list(B3_FEATURES_25D)),
        expected_runner_binding=runner_binding,
        expected_input_sha256={
            "registry_csv_sha256": snapshots["formal_fit_registry_sha256"],
            "registry_summary_sha256": snapshots["formal_registry_summary_sha256"],
            "source_contract_sha256": snapshots["source_contract_sha256"],
            "protocol_sha256": snapshots["protocol_sha256"],
            "feature_rebuilder_sha256": snapshots["feature_rebuilder_sha256"],
        },
    )
    if root_report.get("status") != "PASS":
        raise ValueError("independent S1 root audit is not PASS")
    s1_audit_path = _snapshot_file(paths["s1_root_audit_sha256"], "s1_root_audit_sha256")
    aggregate_path = _snapshot_file(paths["teacher_aggregate_sha256"], "teacher_aggregate_sha256")
    root_aggregate_path = s1_root / "B3_OFFICIAL_V3_TEACHER_AGGREGATE_AUDIT_V1.json"
    if aggregate_path.resolve() != root_aggregate_path.resolve() or sha256_file(aggregate_path) != snapshots["teacher_aggregate_sha256"]:
        raise ValueError("Teacher aggregate input is not the sealed S1 aggregate")
    s1_audit = json.loads(s1_audit_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if s1_audit.get("status") != "PASS" or int(s1_audit.get("identity_count", 0)) != 800 or aggregate.get("status") != "PASS":
        raise ValueError("S1 root/Teacher aggregate status is not PASS")
    contract = load_contract(_snapshot_file(paths["source_contract_sha256"], "source_contract_sha256"))
    if contract.get("feature_order_sha256") != json_sha(list(B3_FEATURES_25D)):
        raise ValueError("source contract feature order mismatch")
    load_s1_protocol(_snapshot_file(paths["protocol_sha256"], "protocol_sha256"))
    training_protocol = json.loads(_snapshot_file(paths["training_protocol_sha256"], "training_protocol_sha256").read_text(encoding="utf-8"))
    if training_protocol.get("schema") != "B3_OFFICIAL_V3_TRAINING_PROTOCOL_V1" or training_protocol.get("status") != "PREPARATION_ONLY" or training_protocol.get("formal_training_authorized") is not False or training_protocol.get("formal_attack_authorized") is not False:
        raise ValueError("training protocol is not the frozen preparation-only V3 protocol")
    fold_manifest = load_fit_fold_bundle(fold_root)
    if fold_manifest.get("registry_sha256") != snapshots["formal_fit_registry_sha256"]:
        raise ValueError("fold manifest is bound to a different formal FIT registry")
    normalization, norm_source = load_normalization_bundle(normalization_root, fold_id=fold_id, variant=variant)
    if norm_source.get("registry_sha256") != snapshots["formal_fit_registry_sha256"] or norm_source.get("s1_corpus_sha256") != snapshots["s1_corpus_sha256"]:
        raise ValueError("normalization source snapshot mismatch")
    if norm_source.get("runner_binding") != runner_binding:
        raise ValueError("normalization runner binding does not match measured runner")
    all_episodes = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None
        if variant == "B3_25D9D":
            if policy_intent_root is None:
                raise ValueError("B3_25D9D authorization requires policy_intent_root")
            nine_d = policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        all_episodes.append(load_episode(root, row, include_9d_root=nine_d))
    train_episodes = all_episodes if fit_scope == "FULL_FIT" else select_fit_fold_episodes(all_episodes, fold_manifest, fold_id=int(fold_id), partition="train")
    expected_train_identity_sha = json_sha(sorted(item.canonical_parent_key for item in train_episodes))
    if norm_source.get("train_identity_sha256") != expected_train_identity_sha:
        raise ValueError("normalization train identity binding does not match the measured training scope")
    recomputed = compute_fit_normalization(train_episodes, include_9d=variant == "B3_25D9D")
    if recomputed.sha256 != normalization.sha256:
        raise ValueError(f"normalization does not match the measured {len(train_episodes)}-episode {fit_scope} data")
    generator = _measure_generator_provenance(
        runner_repo=runner_repo, generator_script=generator_script, expected_head=expected_runner_head,
    )
    verification = {
        "schema": "B3_OFFICIAL_V3_AUTHORIZATION_VERIFICATION_V1",
        "status": "PASS",
        "semantic_inputs_verified": True,
        "registry_audit_status": "PASS",
        "s1_root_audit_status": root_report.get("status"),
        "teacher_aggregate_status": aggregate.get("status"),
        "normalization_recomputed_from_train_count": len(train_episodes),
        "fit_scope": fit_scope,
        "normalization_recomputed": True,
        "runner_binding_measured": True,
        "generator_provenance_measured": True,
        "policy_intent_root_sha256": sha256_file(policy_intent_root / "SHA256SUMS") if policy_intent_root is not None and policy_intent_root.is_dir() and (policy_intent_root / "SHA256SUMS").is_file() else None,
    }
    return _build_training_authorization_from_verified(
        output_root, variant=variant, fold_id=fold_id, seed=seed, fit_scope=fit_scope,
        input_snapshots=snapshots, runner_binding=runner_binding,
        generator_provenance=generator, verification=verification,
    )


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
    "AUTHORIZATION_INPUT_NAMES", "build_training_authorization", "build_training_authorization_from_paths", "load_training_authorization_bundle",
]
