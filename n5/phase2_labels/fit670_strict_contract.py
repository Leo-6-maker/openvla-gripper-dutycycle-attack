"""Strict FIT-670 V2 contracts.

This module is intentionally stdlib-only so transition, supervisor, worker and
finalizer can all run the same fail-closed validation before model import.
Legacy V1 receipts are development-only and are rejected here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

TRANSITION_GATE = "FIT670-INFERENCE_TRANSITION"
TRANSITION_SCHEMA = "FIT670_INFERENCE_TRANSITION_V2"
ALLOWLIST_GATE = "FIT670-INFERENCE_IDENTITY_ALLOWLIST"
ALLOWLIST_SCHEMA = "FIT670_IDENTITY_ALLOWLIST_V1"
SHARD_SCHEMA = "FIT670_GPU_SHARD_PLAN_V1"
EPISODE_SCHEMA = "FIT670_EPISODE_V2"
FOUR_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ContractViolation(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractViolation(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractViolation(f"JSON root is not an object: {path}")
    return value


def full_seal_check(root: Path) -> str:
    root = Path(root).resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise ContractViolation(f"unsealed root: {root}")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ContractViolation(f"symlink under sealed root: {root}")
    side_tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if len(side_tokens) != 2 or side_tokens[1] != "SHA256SUMS":
        raise ContractViolation(f"malformed SHA256SUMS sidecar: {root}")
    actual_sums_sha = sha256_file(sums)
    if side_tokens[0] != actual_sums_sha:
        raise ContractViolation(f"SHA256SUMS sidecar mismatch: {root}")
    declared: Dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, rel = line.partition("  ")
        if not sep or not HEX64.fullmatch(digest):
            raise ContractViolation(f"malformed SHA256SUMS line: {line!r}")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or rel in declared:
            raise ContractViolation(f"unsafe/duplicate sealed path: {rel!r}")
        target = (root / rel_path).resolve()
        if root not in target.parents or not target.is_file():
            raise ContractViolation(f"sealed path escapes or is missing: {rel}")
        if sha256_file(target) != digest:
            raise ContractViolation(f"sealed file mismatch: {rel}")
        rel_key = rel if not rel.startswith("./") else rel[2:]
        declared[rel_key] = digest
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if set(declared) != actual:
        raise ContractViolation(
            f"seal closure mismatch: missing={sorted(actual-set(declared))[:5]} "
            f"extra={sorted(set(declared)-actual)[:5]}"
        )
    return actual_sums_sha


def _required(record: Mapping[str, Any], names: Iterable[str], context: str) -> None:
    missing = [name for name in names if name not in record]
    if missing:
        raise ContractViolation(f"{context} missing fields: {missing}")


def identity_key(record: Mapping[str, Any]) -> str:
    _required(
        record,
        ("episode_id", "suite", "task_id", "state_id", "collection_seed",
         "initial_state_sha256"),
        "identity",
    )
    suite = record["suite"]
    task_id = record["task_id"]
    state_id = record["state_id"]
    if suite not in FOUR_SUITES or type(task_id) is not int or type(state_id) is not int:
        raise ContractViolation(f"invalid identity coordinates: {record}")
    expected = f"{suite}/task_{task_id:02d}/state_{state_id:02d}"
    if record["episode_id"] != expected:
        raise ContractViolation(f"episode_id mismatch: {record['episode_id']} != {expected}")
    if type(record["collection_seed"]) is not int:
        raise ContractViolation(f"invalid collection_seed: {expected}")
    if not HEX64.fullmatch(str(record["initial_state_sha256"])):
        raise ContractViolation(f"invalid initial_state_sha256: {expected}")
    return expected


def legacy_identity_set_digest(identities: List[Mapping[str, Any]]) -> str:
    """Reproduce the frozen V1 allowlist builder exactly."""
    slim = [{k: v for k, v in item.items() if k != "fold"} for item in identities]
    return hashlib.sha256(json.dumps(slim, sort_keys=True).encode()).hexdigest()


def validate_allowlist(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    data = load_json(path)
    if data.get("gate") != ALLOWLIST_GATE or data.get("schema") != ALLOWLIST_SCHEMA:
        raise ContractViolation("wrong FIT670 allowlist gate/schema")
    identities = data.get("identities")
    if not isinstance(identities, list) or len(identities) != 670:
        raise ContractViolation(f"allowlist identity count != 670: {len(identities or [])}")
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in identities:
        if not isinstance(raw, dict):
            raise ContractViolation("allowlist identity is not an object")
        key = identity_key(raw)
        if key in by_id:
            raise ContractViolation(f"duplicate allowlist identity: {key}")
        by_id[key] = raw
    if data.get("n_identities") != 670 or data.get("protected_overlap") != 0:
        raise ContractViolation("allowlist count/protected overlap contract failed")
    expected_digest = legacy_identity_set_digest(identities)
    if data.get("identity_set_digest") != expected_digest:
        raise ContractViolation("allowlist identity_set_digest is not recomputable")
    return data, by_id


def validate_shard_plan(
    path: Path,
    allowlist_path: Path,
    expected_n_shards: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    allowlist, allowed = validate_allowlist(allowlist_path)
    plan = load_json(path)
    if plan.get("schema") != SHARD_SCHEMA or plan.get("status") != "FROZEN":
        raise ContractViolation("wrong shard plan schema/status")
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContractViolation("shard plan has no shards")
    n_shards = plan.get("n_shards")
    if n_shards != len(shards) or n_shards not in (4, 6, 8):
        raise ContractViolation("invalid shard count")
    if expected_n_shards is not None and n_shards != expected_n_shards:
        raise ContractViolation(
            f"runtime shard count mismatch: plan={n_shards}, expected={expected_n_shards}"
        )
    if plan.get("n_identities") != 670:
        raise ContractViolation("shard plan n_identities != 670")
    if plan.get("input_allowlist_sha256") != sha256_file(allowlist_path):
        raise ContractViolation("shard plan is not bound to allowlist bytes")
    membership: Dict[str, int] = {}
    shard_ids = set()
    for shard in shards:
        sid = shard.get("shard_id")
        if type(sid) is not int or sid in shard_ids:
            raise ContractViolation(f"invalid/duplicate shard id: {sid}")
        shard_ids.add(sid)
        rows = shard.get("identities")
        if not isinstance(rows, list) or shard.get("n_identities") != len(rows):
            raise ContractViolation(f"shard {sid} count mismatch")
        for raw in rows:
            key = identity_key(raw)
            if key in membership:
                raise ContractViolation(f"identity assigned twice: {key}")
            if key not in allowed:
                raise ContractViolation(f"shard contains unallowlisted identity: {key}")
            if canonical_json_sha(raw) != canonical_json_sha(allowed[key]):
                raise ContractViolation(f"shard identity bytes differ from allowlist: {key}")
            membership[key] = sid
    if shard_ids != set(range(n_shards)):
        raise ContractViolation(f"shard ids are not 0..{n_shards-1}")
    if set(membership) != set(allowed):
        raise ContractViolation(
            f"shard closure failed: missing={len(set(allowed)-set(membership))} "
            f"extra={len(set(membership)-set(allowed))}"
        )
    return plan, membership


def git_identity(repo: Path) -> Tuple[str, str]:
    repo = Path(repo).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ContractViolation(f"git identity failed for {repo}: {exc}") from exc
    if not HEX40.fullmatch(commit) or not HEX40.fullmatch(tree) or dirty:
        raise ContractViolation(f"repo is invalid or dirty: {repo}")
    return commit, tree


def git_commit_time(repo: Path, commit: str) -> datetime:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(Path(repo).resolve()), "show", "-s", "--format=%cI", commit],
            text=True,
        ).strip()
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise ContractViolation(f"cannot resolve commit chronology: {repo}@{commit}") from exc
    if value.tzinfo is None:
        raise ContractViolation("git commit timestamp is timezone-naive")
    return value.astimezone(timezone.utc)


def assert_import_origin(module_name: str, expected_root: Path) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ContractViolation(f"cannot resolve import origin: {module_name}")
    origin = Path(spec.origin).resolve()
    root = Path(expected_root).resolve()
    if origin != root and root not in origin.parents:
        raise ContractViolation(
            f"{module_name} import escapes frozen root: {origin} not under {root}"
        )
    return str(origin)


def validate_transition_v2(
    transition_root: Path,
    *,
    allowlist_path: Path,
    shard_plan_path: Path,
    output_root: Path,
    physical_gpu: int,
    shard_id: int,
    model_path: Path,
    official_worker: Path,
    registry_summary: Path,
    alias_ledger: Path,
    repo_root: Path,
    upstream_root: Path,
    libero_root: Path,
    source_files: Mapping[str, Path],
    collection_mode: str,
) -> Dict[str, Any]:
    from fit_transition import compute_model_tree_fingerprint

    full_seal_check(transition_root)
    manifest = load_json(Path(transition_root) / "TRANSITION_MANIFEST.json")
    if manifest.get("gate") != TRANSITION_GATE or manifest.get("schema") != TRANSITION_SCHEMA:
        raise ContractViolation("legacy or wrong transition receipt")
    if collection_mode not in {"canary", "formal"}:
        raise ContractViolation(f"invalid runtime collection mode: {collection_mode}")
    if manifest.get("collection_mode") != collection_mode:
        raise ContractViolation("transition/runtime collection mode mismatch")
    if collection_mode == "formal":
        canary_root = Path(str(manifest.get("canary_review_root", "")))
        if full_seal_check(canary_root) != manifest.get(
            "canary_review_sha256sums_sha256"
        ):
            raise ContractViolation("formal transition has invalid canary review seal")
        canary = load_json(canary_root / "CANARY_REVIEW.json")
        if (
            canary.get("status") != "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE"
            or canary.get("identity_set_digest") != manifest.get("identity_set_digest")
            or canary.get("shard_plan_sha256") != manifest.get("shard_plan_sha256")
            or canary.get("collection_source_commit")
            != manifest.get("collection_source_commit")
        ):
            raise ContractViolation("formal transition canary binding failed")
    allowlist, _ = validate_allowlist(allowlist_path)
    plan, _ = validate_shard_plan(
        shard_plan_path, allowlist_path, manifest.get("n_shards")
    )
    exact = {
        "identity_allowlist_file_sha256": sha256_file(allowlist_path),
        "identity_set_digest": allowlist["identity_set_digest"],
        "shard_plan_sha256": sha256_file(shard_plan_path),
        "model_path": str(Path(model_path).resolve()),
        "official_worker_sha256": sha256_file(official_worker),
        "registry_summary_sha256": sha256_file(registry_summary),
        "alias_ledger_sha256": sha256_file(alias_ledger),
        "model_tree_sha256": compute_model_tree_fingerprint(model_path),
        "processor_sha256": sha256_file(Path(model_path) / "preprocessor_config.json"),
        "identity_allowlist_root_sha256sums_sha256": full_seal_check(
            Path(allowlist_path).parent
        ),
        "shard_plan_root_sha256sums_sha256": full_seal_check(
            Path(shard_plan_path).parent
        ),
        "c1_root_sha256sums_sha256": full_seal_check(
            Path(registry_summary).parent
        ),
    }
    for key, value in exact.items():
        if manifest.get(key) != value:
            raise ContractViolation(f"transition binding mismatch: {key}")
    if manifest.get("authorized_identities") != 670 or manifest.get("max_episodes") != 670:
        raise ContractViolation("transition is not frozen for exactly 670 identities")
    allowed_roots = [str(Path(p).resolve()) for p in manifest.get("allowed_output_roots", [])]
    if str(Path(output_root).resolve()) not in allowed_roots:
        raise ContractViolation("output root is not authorized")
    allowed_gpus = manifest.get("allowed_physical_gpus")
    mapping = manifest.get("shard_to_physical_gpu")
    if not isinstance(allowed_gpus, list) or physical_gpu not in allowed_gpus:
        raise ContractViolation("physical GPU not authorized")
    if not isinstance(mapping, dict) or mapping.get(str(shard_id)) != physical_gpu:
        raise ContractViolation("runtime shard/GPU mapping mismatch")
    permissions = {
        "openvla_inference_authorized": True,
        "clean_action_only": True,
        "forward_before_capture": True,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "detector_load_authorized": False,
        "attack_authorized": False,
        "protected_payload_read": False,
    }
    for key, value in permissions.items():
        if manifest.get(key) is not value:
            raise ContractViolation(f"permission mismatch: {key}")
    repo_commit, repo_tree = git_identity(repo_root)
    up_commit, up_tree = git_identity(upstream_root)
    lib_commit, lib_tree = git_identity(libero_root)
    runtime = {
        "collection_source_commit": repo_commit,
        "collection_source_tree": repo_tree,
        "upstream_commit": up_commit,
        "upstream_tree": up_tree,
        "libero_commit": lib_commit,
        "libero_tree": lib_tree,
    }
    for key, value in runtime.items():
        if manifest.get(key) != value:
            raise ContractViolation(f"runtime provenance mismatch: {key}")
    created_at = manifest.get("created_at")
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractViolation("invalid transition created_at") from exc
    if created.tzinfo is None or created.astimezone(timezone.utc) <= git_commit_time(
        repo_root, repo_commit
    ):
        raise ContractViolation("transition chronology does not follow source commit")
    frozen_source = manifest.get("collection_source_files")
    if not isinstance(frozen_source, dict) or set(frozen_source) != set(source_files):
        raise ContractViolation("transition source-file set mismatch")
    for key, path in source_files.items():
        if frozen_source.get(key) != sha256_file(path):
            raise ContractViolation(f"source file SHA mismatch: {key}")
    return manifest


def enrich_entity_record(
    record: Dict[str, Any], resolution: Mapping[str, Any]
) -> Dict[str, Any]:
    logical_name = next(
        (
            resolution.get(key)
            for key in ("logical_name", "bddl_name", "name", "entity_name")
            if resolution.get(key)
        ),
        None,
    )
    if not logical_name:
        raise ContractViolation("C1 entity lacks logical identity")
    alias_to = resolution.get("alias_to")
    kind = resolution.get("resolution") or resolution.get("resolution_kind")
    if kind == "APPROVED_STRUCTURAL_ALIAS" and not alias_to:
        raise ContractViolation(f"alias resolution lacks alias_to: {logical_name}")
    record = dict(record)
    record["logical_name"] = str(logical_name)
    record["alias_to"] = None if alias_to is None else str(alias_to)
    record["resolution_kind"] = str(kind)
    record["binding_identity"] = canonical_json_sha(
        {
            "logical_name": record["logical_name"],
            "alias_to": record["alias_to"],
            "role": record.get("role"),
            "entity_type": record.get("entity_type"),
            "entity_id": record.get("entity_id"),
        }
    )
    return record


def validate_episode_v2(
    episode_root: Path,
    expected_identity: Mapping[str, Any],
    transition_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    full_seal_check(episode_root)
    episode = load_json(Path(episode_root) / "episode.json")
    key = identity_key(episode)
    if key != identity_key(expected_identity):
        raise ContractViolation(f"episode identity mismatch: {key}")
    if episode.get("schema") != EPISODE_SCHEMA:
        raise ContractViolation(f"episode has legacy schema: {key}")
    bindings = episode.get("episode_bindings")
    if not isinstance(bindings, dict):
        raise ContractViolation(f"missing episode bindings: {key}")
    expected_bindings = {
        "identity_set_digest": transition_manifest["identity_set_digest"],
        "shard_plan_sha256": transition_manifest["shard_plan_sha256"],
        "collection_source_commit": transition_manifest["collection_source_commit"],
        "collection_source_tree": transition_manifest["collection_source_tree"],
        "transition_schema": TRANSITION_SCHEMA,
    }
    for field, value in expected_bindings.items():
        if bindings.get(field) != value:
            raise ContractViolation(f"episode binding mismatch {field}: {key}")
    telemetry = episode.get("telemetry")
    if not isinstance(telemetry, list):
        raise ContractViolation(f"missing embedded telemetry: {key}")
    n_steps = 0
    for row in telemetry:
        if not isinstance(row, dict):
            raise ContractViolation(f"invalid telemetry row: {key} step {n_steps}")
        entities = row.get("entities")
        if not isinstance(entities, list):
            raise ContractViolation(f"entities missing: {key} step {n_steps}")
        seen_bindings = set()
        for entity in entities:
            _required(
                entity,
                (
                    "logical_name", "alias_to", "resolution_kind", "binding_identity",
                    "role", "entity_type", "entity_id", "position", "rotation_wxyz",
                ),
                f"{key} entity",
            )
            if entity["binding_identity"] in seen_bindings:
                raise ContractViolation(f"duplicate entity binding: {key} step {n_steps}")
            seen_bindings.add(entity["binding_identity"])
            values = list(entity["position"]) + list(entity["rotation_wxyz"])
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                raise ContractViolation(f"nonfinite entity pose: {key} step {n_steps}")
        contacts = row.get("contact_pairs")
        if not isinstance(contacts, list):
            raise ContractViolation(f"contact pairs missing: {key} step {n_steps}")
        if row.get("contact_ncon_total") != len(contacts) or row.get("contact_truncated") is not False:
            raise ContractViolation(f"contact closure failed: {key} step {n_steps}")
        n_steps += 1
    if episode.get("n_steps") != n_steps or n_steps <= 0:
        raise ContractViolation(f"step count mismatch: {key}")
    return episode
