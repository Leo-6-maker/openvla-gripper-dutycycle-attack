"""Build an independent, sealed T0-A audit for the frozen FIT670 formal root."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
EXPECTED_COUNT = 670
BINDING_FIELDS = (
    "suite", "task_id", "task_name", "state_id", "seed", "episode_id", "initial_state_sha256",
    "relative_path", "episode_sha256", "episode_sha256sums_sha256", "worker_id", "shard_id",
    "worker_result_target", "worker_result_steps", "worker_result_source_sha256",
    "worker_result_episode_sha256sums_sha256", "worker_result_initial_state_sha256",
    "worker_result_binding_mode", "worker_manifest_sha256", "worker_seal_sha256sums_sha256",
    "collection_source_commit", "collection_source_tree", "collector_script_sha256",
    "transition_manifest_sha256", "transition_sha256sums_sha256", "allowlist_sha256",
    "c1_canonical_digest", "schema",
)


def _sha64(value: Any, field: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _sha40(value: Any, field: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{field} must be a lowercase git SHA")
    return value


def _forbidden_path(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_PARTS for part in path.resolve().parts)


def _safe_episode_path(root: Path, identity: str) -> Path:
    parts = identity.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe identity: {identity}")
    relative = Path("episodes", *parts, "episode.json")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"episode path missing or escaped: {identity}")
    return relative


def _canonical_digest(entries: list[dict[str, Any]]) -> str:
    rows = [{key: entry[key] for key in BINDING_FIELDS} for entry in entries]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity_set_digest(entries: list[dict[str, Any]]) -> str:
    canonical = [{key: entry[key] for key in ("episode_id", "suite", "task_id", "state_id", "collection_seed", "initial_state_sha256")} for entry in entries]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite payload value: {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")


def _validate_episode_relations(episode: dict[str, Any], identity: str) -> tuple[int, str]:
    relations = episode.get("relations")
    geometry_status = episode.get("geometry_status")
    if not isinstance(relations, list):
        raise ValueError(f"T0-A relation records are not a list: {identity}")
    if not relations and geometry_status != "NOT_APPLICABLE":
        raise ValueError(f"T0-A empty relation records without NOT_APPLICABLE status: {identity}")
    return len(relations), str(geometry_status or "APPLICABLE")


def _verify_formal_finalization(finalization_root: Path, expected_episode_digest: str) -> dict[str, Any]:
    root = finalization_root.resolve()
    if _forbidden_path(root):
        raise ValueError("finalization path is forbidden-looking")
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError("formal finalization top-level seal missing")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError("formal finalization sidecar mismatch")
    rows = [line.split("  ", 1) for line in sums.read_text(encoding="utf-8").splitlines()]
    if not rows or any(len(row) != 2 for row in rows):
        raise ValueError("formal finalization seal is empty or malformed")
    if len(rows) != 1 or rows[0][1] != "GLOBAL_MANIFEST.json":
        raise ValueError("unexpected formal finalization seal scope")
    global_path = root / "GLOBAL_MANIFEST.json"
    if rows[0][0] != sha256_file(global_path):
        raise ValueError("formal GLOBAL_MANIFEST seal mismatch")
    global_manifest = _read_json(global_path)
    _sha40(global_manifest.get("collection_source_commit"), "collection_source_commit")
    _sha40(global_manifest.get("collection_source_tree"), "collection_source_tree")
    episode_seals = global_manifest.get("episode_seals")
    if not isinstance(episode_seals, dict) or len(episode_seals) != EXPECTED_COUNT:
        raise ValueError("formal episode seal map is incomplete")
    map_digest = hashlib.sha256(json.dumps(episode_seals, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if global_manifest.get("episode_seal_digest") != expected_episode_digest or map_digest != expected_episode_digest:
        raise ValueError("formal episode seal digest mismatch")
    auxiliary = {}
    for name in ("PROGRESS_RECONCILIATION.json", "IDENTITY_CLOSURE.json", "PER_SHARD_CLOSURE.json", "WORKER_RUNTIME_STATE.json", "STAGING_AUDIT.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"formal finalization auxiliary missing: {name}")
        auxiliary[name] = sha256_file(path)
    progress = _read_json(root / "PROGRESS_RECONCILIATION.json")
    required = {
        "verdict": "PASS", "allowlist_count": EXPECTED_COUNT, "published_count": EXPECTED_COUNT,
        "duplicates": 0, "bad_seals": 0, "unallowlisted": 0, "staging_residue": 0,
    }
    if any(progress.get(key) != value for key, value in required.items()) or progress.get("missing") != [] or progress.get("extra") != []:
        raise ValueError("formal progress reconciliation is not PASS")
    identity_closure = _read_json(root / "IDENTITY_CLOSURE.json")
    if set(identity_closure.get("allowlist_ids", [])) != set(episode_seals) or len(identity_closure.get("allowlist_ids", [])) != EXPECTED_COUNT:
        raise ValueError("formal identity closure mismatch")
    per_shard = json.loads((root / "PER_SHARD_CLOSURE.json").read_text(encoding="utf-8"))
    if not isinstance(per_shard, list) or len(per_shard) != 8 or not all(item.get("match") is True and item.get("missing") == [] and item.get("extra") == [] for item in per_shard):
        raise ValueError("formal shard closure is not PASS")
    worker_state = _read_json(root / "WORKER_RUNTIME_STATE.json")
    workers = worker_state.get("workers")
    if not isinstance(workers, dict) or len(workers) != 8 or not all(item.get("manifest_present") is True and item.get("n_fail") == 0 and item.get("n_skipped") == 0 for item in workers.values()):
        raise ValueError("formal worker runtime state is not PASS")
    staging = _read_json(root / "STAGING_AUDIT.json")
    if staging.get("verdict") != "PASS" or staging.get("staging_residue_count") != 0 or staging.get("staging_residue") != []:
        raise ValueError("formal staging audit is not PASS")
    return {
        "root": str(root), "global_manifest_sha256": sha256_file(global_path), "sha256sums_sha256": sha256_file(sums),
        "episode_seals": episode_seals, "episode_seal_digest": expected_episode_digest,
        "auxiliary_sha256": auxiliary, "seal_scope": "GLOBAL_MANIFEST_ONLY_WITH_AUXILIARY_HASHES_FROZEN_IN_THIS_AUDIT",
        "identity_set_digest": global_manifest.get("identity_set_digest"),
        "collection_source_commit": global_manifest.get("collection_source_commit"),
        "collection_source_tree": global_manifest.get("collection_source_tree"),
    }


def _verify_worker_result_counts(worker_root: Path, results: list[dict[str, Any]], plan_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    worker = _read_json(worker_root / "WORKER_MANIFEST.json")
    result_steps: dict[str, int] = {}
    for result in results:
        identity = str(result.get("episode_id", ""))
        if identity not in plan_by_id:
            raise ValueError(f"worker identity is not in shard plan: {identity}")
        raw_steps = result.get("steps", result.get("step_count"))
        if not isinstance(raw_steps, int) or isinstance(raw_steps, bool) or raw_steps <= 0:
            raise ValueError(f"worker step count missing/invalid: {identity}")
        result_steps[identity] = raw_steps
    if len(result_steps) != len(results):
        raise ValueError("worker result identities are duplicated")
    return {"manifest": worker, "result_steps": result_steps}


def audit(formal_root: Path, finalization_root: Path, transition_path: Path, allowlist_path: Path, shard_plan_path: Path, output_root: Path, *, expected_transition_seal: str, expected_episode_digest: str) -> dict[str, Any]:
    roots = [formal_root.resolve(), finalization_root.resolve(), transition_path.resolve(), allowlist_path.resolve(), shard_plan_path.resolve()]
    if any(_forbidden_path(path) for path in roots):
        raise ValueError("T0-A input path is forbidden-looking")
    if output_root.exists():
        raise FileExistsError(output_root)
    transition_root = transition_path.resolve().parent
    transition_seal = verify_seal(transition_root)
    if transition_seal["sha256sums_sha256"] != expected_transition_seal:
        raise ValueError("T0-A transition seal mismatch")
    transition = _read_json(transition_path)
    if transition.get("schema") != "FIT670_INFERENCE_TRANSITION_V2" or transition.get("collection_mode") != "formal":
        raise ValueError("T0-A transition schema/mode mismatch")
    if transition.get("teacher_labels_authorized") is not False or transition.get("student_training_authorized") is not False or transition.get("attack_authorized") is not False:
        raise ValueError("T0-A historical transition permissions are not closed")
    if transition.get("protected_payload_read") is not False or transition.get("protected_overlap_verified") != 0:
        raise ValueError("T0-A protected boundary is not closed")
    finalization = _verify_formal_finalization(finalization_root, expected_episode_digest)
    if finalization["collection_source_commit"] != transition.get("collection_source_commit") or finalization["collection_source_tree"] != transition.get("collection_source_tree"):
        raise ValueError("T0-A source chronology mismatch")
    allowlist = _read_json(allowlist_path)
    if sha256_file(allowlist_path) != transition.get("identity_allowlist_file_sha256"):
        raise ValueError("T0-A allowlist SHA mismatch")
    allowlist_seal = verify_seal(allowlist_path.parent)
    if allowlist_seal["sha256sums_sha256"] != transition.get("identity_allowlist_root_sha256sums_sha256"):
        raise ValueError("T0-A allowlist root seal mismatch")
    if allowlist.get("schema") != "FIT670_IDENTITY_ALLOWLIST_V1" or allowlist.get("protected_overlap") != 0 or allowlist.get("identity_set_digest") != transition.get("identity_set_digest"):
        raise ValueError("T0-A allowlist schema/protected overlap mismatch")
    entries = allowlist.get("identities")
    if not isinstance(entries, list) or not entries or len(entries) != EXPECTED_COUNT:
        raise ValueError("T0-A allowlist is empty")
    allowlist_by_id = {str(item["episode_id"]): item for item in entries}
    if len(allowlist_by_id) != EXPECTED_COUNT or set(allowlist_by_id) != set(finalization["episode_seals"]):
        raise ValueError("T0-A identity set is not exact 670")
    authorized = transition.get("authorized_identities")
    if isinstance(authorized, int) and not isinstance(authorized, bool):
        if authorized != EXPECTED_COUNT:
            raise ValueError("T0-A authorized identity count is not exact")
    elif isinstance(authorized, list):
        if not authorized or len(authorized) != EXPECTED_COUNT or set(map(str, authorized)) != set(allowlist_by_id):
            raise ValueError("T0-A authorized identity list is not exact")
    else:
        raise ValueError("T0-A authorized identity binding is missing")
    if _identity_set_digest(entries) != transition.get("identity_set_digest") or _identity_set_digest(entries) != allowlist.get("identity_set_digest"):
        raise ValueError("T0-A identity digest recomputation mismatch")
    if transition.get("identity_set_digest") != finalization.get("identity_set_digest"):
        raise ValueError("T0-A finalization identity digest mismatch")
    shard_plan = _read_json(shard_plan_path)
    if shard_plan.get("schema") != "FIT670_GPU_SHARD_PLAN_V1" or shard_plan.get("n_identities") != EXPECTED_COUNT or shard_plan.get("n_shards") != 8:
        raise ValueError("T0-A shard plan schema/cardinality mismatch")
    if sha256_file(shard_plan_path) != transition.get("shard_plan_sha256"):
        raise ValueError("T0-A shard plan SHA mismatch")
    if shard_plan.get("input_allowlist_sha256") != transition.get("identity_allowlist_file_sha256"):
        raise ValueError("T0-A shard plan allowlist binding mismatch")
    if transition.get("shard_plan_root_sha256sums_sha256"):
        shard_root_seal = verify_seal(shard_plan_path.parent)
        if shard_root_seal["sha256sums_sha256"] != transition.get("shard_plan_root_sha256sums_sha256"):
            raise ValueError("T0-A shard plan root seal mismatch")
    plan_by_id = {}
    shard_ids = set()
    for shard in shard_plan.get("shards", []):
        shard_id = int(shard["gpu"])
        shard_ids.add(shard_id)
        for item in shard.get("identities", []):
            identity = str(item["episode_id"])
            if identity in plan_by_id:
                raise ValueError(f"T0-A duplicate shard identity: {identity}")
            plan_by_id[identity] = {"worker_id": f"gpu_{shard_id}", "shard_id": shard_id, **item}
    if shard_ids != set(range(8)) or set(plan_by_id) != set(allowlist_by_id):
        raise ValueError("T0-A shard identity closure mismatch")
    worker_by_id = {}
    worker_result_steps: dict[str, int] = {}
    worker_audit = []
    for shard_id in range(8):
        worker_root = formal_root / f"gpu_{shard_id}"
        worker_seal = verify_seal(worker_root)
        worker = _read_json(worker_root / "WORKER_MANIFEST.json")
        results = worker.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"T0-A worker {shard_id} has empty results")
        if worker.get("n_success") != len(results) or worker.get("n_fail") != 0 or worker.get("n_skipped") != 0:
            raise ValueError(f"T0-A worker {shard_id} count mismatch")
        worker_info = _verify_worker_result_counts(worker_root, results, plan_by_id)
        if worker_info["manifest"].get("collection_source_commit") != transition.get("collection_source_commit"):
            raise ValueError(f"T0-A worker source commit mismatch: gpu_{shard_id}")
        if worker_info["manifest"].get("collection_source_tree") != transition.get("collection_source_tree"):
            raise ValueError(f"T0-A worker source tree mismatch: gpu_{shard_id}")
        if worker_info["manifest"].get("identity_set_digest") != transition.get("identity_set_digest"):
            raise ValueError(f"T0-A worker identity digest mismatch: gpu_{shard_id}")
        if worker_info["manifest"].get("shard_plan_sha256") != transition.get("shard_plan_sha256"):
            raise ValueError(f"T0-A worker shard plan binding mismatch: gpu_{shard_id}")
        for result in results:
            identity = str(result["episode_id"])
            if identity in worker_by_id or identity not in plan_by_id or plan_by_id[identity]["shard_id"] != shard_id or result.get("status") != "OK":
                raise ValueError(f"T0-A worker identity mismatch: {identity}")
            target = result.get("target")
            expected_target = str((formal_root / _safe_episode_path(formal_root, identity)).parent.resolve())
            if not isinstance(target, str) or Path(target).resolve() != Path(expected_target):
                raise ValueError(f"T0-A worker target binding mismatch: {identity}")
            worker_by_id[identity] = {"worker_id": f"gpu_{shard_id}", "shard_id": shard_id, "target": str(Path(target).resolve()), "steps": worker_info["result_steps"][identity], "worker_manifest_sha256": sha256_file(worker_root / "WORKER_MANIFEST.json"), "worker_seal_sha256sums_sha256": worker_seal["sha256sums_sha256"]}
            worker_result_steps[identity] = worker_info["result_steps"][identity]
        worker_audit.append({"worker_id": f"gpu_{shard_id}", "shard_id": shard_id, "worker_manifest_sha256": sha256_file(worker_root / "WORKER_MANIFEST.json"), "worker_seal_sha256sums_sha256": worker_seal["sha256sums_sha256"], "count": len(results)})
    if set(worker_by_id) != set(allowlist_by_id) or len(worker_by_id) != EXPECTED_COUNT:
        raise ValueError("T0-A worker union is not exact")
    if set(worker_result_steps) != set(allowlist_by_id):
        raise ValueError("T0-A worker step binding is not exact")
    source_staging = sorted(path.relative_to(formal_root).as_posix() for path in formal_root.rglob("*") if ".staging." in path.name)
    if source_staging:
        raise ValueError(f"T0-A formal source has staging residue: {source_staging[:3]}")
    episode_bindings = {}
    schema_rows = []
    non_applicable_geometry_episodes = 0
    for identity in sorted(allowlist_by_id):
        relative = _safe_episode_path(formal_root, identity)
        episode_path = formal_root / relative
        episode_root = episode_path.parent
        episode_seal = verify_seal(episode_root)
        if episode_seal["sha256sums_sha256"] != finalization["episode_seals"][identity]:
            raise ValueError(f"T0-A episode seal mismatch: {identity}")
        episode = _read_json(episode_path)
        _assert_finite(episode, identity)
        expected_identity = f"{episode.get('suite')}/task_{int(episode.get('task_id')):02d}/state_{int(episode.get('state_id')):02d}"
        if episode.get("episode_id") != identity or expected_identity != identity:
            raise ValueError(f"T0-A episode identity mismatch: {identity}")
        if episode.get("schema") != "FIT670_EPISODE_V2" or episode.get("schema_version") != "FIT670_FEATURE_SCHEMA_V1":
            raise ValueError(f"T0-A episode schema mismatch: {identity}")
        entry = allowlist_by_id[identity]
        for key in ("suite", "task_id", "state_id", "collection_seed", "initial_state_sha256"):
            if key not in entry:
                raise ValueError(f"T0-A allowlist identity field missing: {identity}.{key}")
        for key in ("collection_seed", "initial_state_sha256"):
            if episode.get(key) != entry.get(key):
                raise ValueError(f"T0-A episode {key} mismatch: {identity}")
        if entry.get("suite") != episode.get("suite") or entry.get("task_id") != episode.get("task_id") or entry.get("state_id") != episode.get("state_id"):
            raise ValueError(f"T0-A allowlist/episode identity mismatch: {identity}")
        if episode.get("attack_enabled") is not False or episode.get("teacher_labels_generated") is not False:
            raise ValueError(f"T0-A episode permission mismatch: {identity}")
        provenance = episode.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("collector_commit") != transition.get("collection_source_commit") or provenance.get("collector_tree") != transition.get("collection_source_tree"):
            raise ValueError(f"T0-A episode source binding mismatch: {identity}")
        relation_count, geometry_status = _validate_episode_relations(episode, identity)
        if relation_count == 0:
            non_applicable_geometry_episodes += 1
        telemetry = episode.get("telemetry")
        steps = episode.get("steps")
        if not isinstance(telemetry, list) or not telemetry or not isinstance(steps, list) or len(steps) != len(telemetry) or not steps or episode.get("n_steps") != len(steps) or episode.get("step_count") != len(steps):
            raise ValueError(f"T0-A empty/schema episode records: {identity}")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf") for value in [episode.get("collection_seed"), episode.get("task_id"), episode.get("state_id")]):
            raise ValueError(f"T0-A nonfinite identity metadata: {identity}")
        if worker_result_steps.get(identity) != len(steps) or (plan_by_id[identity].get("steps") is not None and plan_by_id[identity].get("steps") != len(steps)):
            raise ValueError(f"T0-A worker/shard step count mismatch: {identity}")
        if episode.get("task_language") in (None, ""):
            raise ValueError(f"T0-A task name is missing: {identity}")
        worker = worker_by_id[identity]
        episode_bindings[identity] = {
            "suite": episode["suite"], "task_id": episode["task_id"], "task_name": episode.get("task_language"), "state_id": episode["state_id"], "seed": episode["collection_seed"], "episode_id": identity,
            "initial_state_sha256": episode["initial_state_sha256"], "relative_path": relative.as_posix(), "episode_sha256": sha256_file(episode_path), "episode_sha256sums_sha256": episode_seal["sha256sums_sha256"],
            "worker_id": worker["worker_id"], "shard_id": worker["shard_id"], "worker_result_target": worker["target"], "worker_result_steps": worker["steps"], "worker_result_source_sha256": sha256_file(episode_path), "worker_result_episode_sha256sums_sha256": episode_seal["sha256sums_sha256"], "worker_result_initial_state_sha256": episode["initial_state_sha256"], "worker_result_binding_mode": "RESULT_TARGET_STEPS_JOINED_TO_SEALED_EPISODE", "worker_manifest_sha256": worker["worker_manifest_sha256"], "worker_seal_sha256sums_sha256": worker["worker_seal_sha256sums_sha256"],
            "collection_source_commit": transition["collection_source_commit"], "collection_source_tree": transition["collection_source_tree"], "collector_script_sha256": provenance["collector_script_sha256"],
            "transition_manifest_sha256": sha256_file(transition_path), "transition_sha256sums_sha256": transition_seal["sha256sums_sha256"], "allowlist_sha256": sha256_file(allowlist_path), "c1_canonical_digest": transition.get("c1_canonical_digest"), "schema": episode.get("schema"),
        }
        schema_rows.append({"episode_id": identity, "step_count": len(steps), "telemetry_count": len(telemetry), "relation_count": relation_count, "geometry_status": geometry_status, "schema": episode.get("schema"), "schema_version": episode.get("schema_version")})
    actual_episode_digest = hashlib.sha256(json.dumps({identity: row["episode_sha256sums_sha256"] for identity, row in sorted(episode_bindings.items())}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual_episode_digest != expected_episode_digest:
        raise ValueError("T0-A episode seal digest recomputation mismatch")
    report = {
        "schema": "V5_R3_FORMAL_INPUT_AUDIT_V1", "status": "PASS_FORMAL_INPUT_CONSUMABLE", "created_at": datetime.now(timezone.utc).isoformat(), "formal_root": str(formal_root.resolve()),
        "episode_count": len(episode_bindings), "episode_list_nonempty": bool(episode_bindings), "episode_identity_unique": len(episode_bindings) == EXPECTED_COUNT, "identity_set_digest": transition["identity_set_digest"],
        "collection_source_commit": transition["collection_source_commit"], "collection_source_tree": transition["collection_source_tree"], "transition_manifest_sha256": sha256_file(transition_path), "transition_sha256sums_sha256": transition_seal["sha256sums_sha256"],
        "allowlist_sha256": sha256_file(allowlist_path), "allowlist_root_sha256sums_sha256": allowlist_seal["sha256sums_sha256"], "shard_plan_sha256": sha256_file(shard_plan_path),
        "finalization": finalization, "worker_closure": worker_audit, "episode_bindings": episode_bindings, "episode_binding_digest": _canonical_digest([episode_bindings[key] for key in sorted(episode_bindings)]), "schema_rows": schema_rows, "non_applicable_geometry_episodes": non_applicable_geometry_episodes,
        "gate": {"duplicate": 0, "missing": 0, "extra": 0, "unallowlisted": 0, "bad_episode_seal": 0, "bad_worker_seal": 0, "schema_error": 0, "empty_entity_records": 0, "identity_binding_error": 0, "source_binding_error": 0, "nonfinite": 0, "staging_residue": 0, "protected_reads": 0},
        "payload_semantics_read": True, "protected_reads": 0, "teacher_labels_generated": False, "labels_generated": False, "student_started": False, "attack_authorized": False,
        "source_staging_residue": source_staging,
    }
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "FORMAL_INPUT_MANIFEST.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "EPISODE_IDENTITY_CLOSURE.json").write_text(json.dumps({"count": len(episode_bindings), "identity_set_digest": report["identity_set_digest"], "identities": sorted(episode_bindings)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "EPISODE_SEAL_AUDIT.json").write_text(json.dumps({identity: item["episode_sha256sums_sha256"] for identity, item in episode_bindings.items()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "WORKER_CLOSURE.json").write_text(json.dumps(worker_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "SHARD_CLOSURE.json").write_text(json.dumps({"shard_count": 8, "union_count": len(episode_bindings), "intersection": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "SCHEMA_AUDIT.json").write_text(json.dumps(schema_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "SOURCE_BINDING_AUDIT.json").write_text(json.dumps(episode_bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "TRANSITION_AUDIT.json").write_text(json.dumps({"manifest_sha256": report["transition_manifest_sha256"], "seal_sha256sums_sha256": report["transition_sha256sums_sha256"], "historical_teacher_labels_authorized": False, "protected_payload_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "GPU_IDENTITY_AUDIT.json").write_text(json.dumps(worker_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "STAGING_AUDIT.json").write_text(json.dumps({"source_staging_residue": source_staging, "protected_reads": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_FORMAL_INPUT_AUDIT_FAILURE_V1", "reason": "publish_failure"}, indent=2) + "\n", encoding="utf-8")
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--finalization-root", type=Path, required=True)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-sha256sums-sha256", required=True)
    parser.add_argument("--episode-seal-digest", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.formal_root.resolve(), args.finalization_root.resolve(), args.transition.resolve(), args.allowlist.resolve(), args.shard_plan.resolve(), args.output_root.resolve(), expected_transition_seal=args.transition_sha256sums_sha256, expected_episode_digest=args.episode_seal_digest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
