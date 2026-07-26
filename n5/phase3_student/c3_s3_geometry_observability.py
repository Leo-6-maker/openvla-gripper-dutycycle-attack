"""C3-S3 geometry observability seal.

This is a read-only audit.  It consumes the sealed H0.3-R6 registry, optionally
initializes LIBERO only to record model geometry metadata, and never steps an
environment or reads Teacher/outcome data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from c3_s3_input_contract import (
    GEOMETRY_SCHEMA,
    audit_episode_geometry,
    load_allowlist,
    load_jsonl_exact,
    require_allowed_path,
    sha256_file as contract_sha256_file,
    verify_manifest_binding,
)


FOUR_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
EXPECTED_TASKS = tuple(f"{suite}/task_{idx:02d}" for suite in FOUR_SUITES for idx in range(10))
PROTECTED_TOKENS = ("cal", "check", "g10", "t2r-d", "t2rd")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {path_value}")
    return path


def verify_sha256sums(root: Path) -> Dict[str, str]:
    sums_path = root / "SHA256SUMS"
    sidecar_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not sidecar_path.is_file():
        raise ValueError("sealed root is missing SHA256SUMS or sidecar")
    sidecar = sidecar_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 1 or sidecar[0] != sha256_file(sums_path):
        raise ValueError("SHA256SUMS sidecar mismatch")
    expected: Dict[str, str] = {}
    for raw in sums_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        digest, name = raw.split(None, 1)
        name = name.lstrip("*")
        rel = safe_relative(name)
        if rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        target = root / rel
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"sealed file mismatch: {name}")
        expected[str(rel).replace("\\", "/")] = digest
    return expected


def quat_normalize(q: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(q) != 4:
        raise ValueError("quaternion must have four components")
    norm = math.sqrt(sum(float(x) * float(x) for x in q))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid quaternion")
    return tuple(float(x) / norm for x in q)


def quat_mul(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float, float]:
    w1, x1, y1, z1 = quat_normalize(left)
    w2, x2, y2, z2 = quat_normalize(right)
    return quat_normalize((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def quat_inverse(q: Sequence[float]) -> Tuple[float, float, float, float]:
    w, x, y, z = quat_normalize(q)
    return (w, -x, -y, -z)


def rotate_vector(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    w, x, y, z = quat_normalize(q)
    vx, vy, vz = (float(x) for x in v)
    # q * [0,v] * q^-1, expanded to avoid a dependency on numpy.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def pose_compose(parent: Mapping[str, Sequence[float]], child: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    pquat = quat_normalize(parent["quat"])
    cpos = rotate_vector(pquat, child["pos"])
    return {
        "pos": [float(parent["pos"][i]) + cpos[i] for i in range(3)],
        "quat": list(quat_mul(pquat, child["quat"])),
    }


def pose_inverse(pose: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    inv = quat_inverse(pose["quat"])
    rotated = rotate_vector(inv, [-float(x) for x in pose["pos"]])
    return {"pos": list(rotated), "quat": list(inv)}


def _max_abs(left: Sequence[float], right: Sequence[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def transform_contract_tests() -> Dict[str, Any]:
    half = math.sqrt(0.5)
    cases = {
        "identity": ((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        "rot_x_90": ((half, half, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "rot_y_90": ((half, 0.0, half, 0.0), (0.0, 0.0, 1.0)),
        "rot_z_90": ((half, 0.0, 0.0, half), (1.0, 0.0, 0.0)),
    }
    expected = {
        "identity": (1.0, 2.0, 3.0),
        "rot_x_90": (0.0, 0.0, 1.0),
        "rot_y_90": (1.0, 0.0, 0.0),
        "rot_z_90": (0.0, 1.0, 0.0),
    }
    results = {}
    for name, (quat, vector) in cases.items():
        actual = rotate_vector(quat, vector)
        results[name] = {"max_abs_error": _max_abs(actual, expected[name]), "pass": _max_abs(actual, expected[name]) <= 1e-12}
    p = {"pos": [0.2, -0.1, 0.3], "quat": list(cases["rot_z_90"][0])}
    c = {"pos": [0.5, 0.0, 0.0], "quat": list(cases["rot_x_90"][0])}
    composed = pose_compose(p, c)
    recovered = pose_compose(composed, pose_inverse(c))
    results["composition_inverse"] = {
        "position_error": _max_abs(recovered["pos"], p["pos"]),
        "quaternion_error": min(_max_abs(recovered["quat"], p["quat"]), _max_abs(recovered["quat"], [-x for x in p["quat"]])),
        "pass": _max_abs(recovered["pos"], p["pos"]) <= 1e-12,
    }
    results["pass"] = all(item["pass"] for item in results.values() if isinstance(item, dict) and "pass" in item)
    return {
        "coordinate_frame": "right_handed_world_and_body_local",
        "quaternion_order": "wxyz",
        "position_unit": "meter",
        "quaternion_unit": "unit_norm",
        "cases": results,
        "pass": bool(results["pass"]),
    }


def load_task_files(root: Path) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for file in sorted((root / "per_task").glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        task_key = data.get("task_key")
        if not task_key or task_key in tasks:
            raise ValueError(f"invalid or duplicate task file: {file}")
        tasks[task_key] = data
    if tuple(sorted(tasks)) != tuple(sorted(EXPECTED_TASKS)):
        raise ValueError(f"task closure mismatch: {len(tasks)} files")
    return tasks


def canonical_relations(data: Mapping[str, Any]) -> Any:
    legacy = data.get("legacy", {})
    return {
        "task_key": data.get("task_key"),
        "status": legacy.get("status"),
        "task_disposition": legacy.get("task_disposition"),
        "goal_predicates": legacy.get("goal_predicates", []),
        "relations": legacy.get("relations", []),
    }


def compare_registry(r6_root: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest = json.loads((r6_root / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("commit") != "beb0721d36bd27412cde7d60623b8cb2f671a4bf":
        raise ValueError("C3-S3 input is not the sealed R6 source snapshot")
    verify_sha256sums(r6_root)
    run_a = load_task_files(r6_root / "run_A")
    run_b = load_task_files(r6_root / "run_B")
    rows: Dict[str, Dict[str, Any]] = {}
    equal = True
    for task_key in EXPECTED_TASKS:
        a = canonical_relations(run_a[task_key])
        b = canonical_relations(run_b[task_key])
        digest_a = sha256_bytes(canonical_json(a))
        digest_b = sha256_bytes(canonical_json(b))
        equal = equal and digest_a == digest_b
        rows[task_key] = {"task_key": task_key, "canonical_digest_a": digest_a, "canonical_digest_b": digest_b, "canonical_equal": digest_a == digest_b, "registry_status": a["status"], "task_disposition": a["task_disposition"], "relations": a["relations"]}
    if not equal:
        raise ValueError("C1-V2 run A/B canonical mapping mismatch")
    source = {
        "r6_root": str(r6_root),
        "r6_root_sha256s_sha256": sha256_file(r6_root / "SHA256SUMS"),
        "r6_manifest_sha256": sha256_file(r6_root / "ARTIFACT_MANIFEST.json"),
        "r6_source_commit": manifest["source"]["commit"],
        "r6_source_tree": manifest["source"]["tree"],
        "run_a_canonical_payload_sha256": manifest["runs"]["run_A"]["canonical_payload_sha256"],
        "run_b_canonical_payload_sha256": manifest["runs"]["run_B"]["canonical_payload_sha256"],
        "run_a_run_b_canonical_equal": equal,
    }
    return source, rows


def extract_model_metadata() -> Dict[str, Dict[str, Any]]:
    """Initialize each LIBERO model once; no action or rollout is executed."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    result: Dict[str, Dict[str, Any]] = {}
    for suite in FOUR_SUITES:
        benchmark = get_benchmark(suite)(0)
        for task_idx in range(10):
            task = benchmark.get_task(task_idx)
            bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=224, camera_widths=224, render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=500)
            try:
                env.reset()
                model = env.sim.model
                bodies = {}
                for body_id in range(model.nbody):
                    name = model.body(body_id).name
                    if not name or name == "world":
                        continue
                    bodies[name] = {
                        "id": int(body_id),
                        "parent_id": int(model.body_parentid[body_id]),
                        "joint_count": int(model.body_jntnum[body_id]),
                        "pos_local": [float(x) for x in model.body_pos[body_id]],
                        "quat_local_wxyz": [float(x) for x in model.body_quat[body_id]],
                    }
                sites = {}
                for site_id in range(model.nsite):
                    name = model.site(site_id).name
                    if not name:
                        continue
                    sites[name] = {
                        "id": int(site_id),
                        "body_id": int(model.site_bodyid[site_id]),
                        "pos_local": [float(x) for x in model.site_pos[site_id]],
                        "quat_local_wxyz": [float(x) for x in model.site_quat[site_id]],
                        "size_m": [float(x) for x in model.site_size[site_id]],
                    }
                result[f"{suite}/task_{task_idx:02d}"] = {
                    "bddl_path": str(bddl_path),
                    "bddl_sha256": sha256_file(bddl_path),
                    "nbody": int(model.nbody),
                    "nsite": int(model.nsite),
                    "bodies": bodies,
                    "sites": sites,
                    "environment_initialization_only": True,
                    "action_steps": 0,
                }
            finally:
                env.close()
    return result


def classify_rows(rows: Mapping[str, Dict[str, Any]], metadata: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for task_key in EXPECTED_TASKS:
        row = rows[task_key]
        task_meta = metadata.get(task_key, {})
        body_by_id = {int(v["id"]): (name, v) for name, v in task_meta.get("bodies", {}).items()}
        site_by_id = {int(v["id"]): v for v in task_meta.get("sites", {}).values()}
        relations = row.get("relations", [])
        if not relations:
            output.append({"task_key": task_key, "relation_index": None, "classification": "NON_PLACEMENT_EXCLUDED", "observability_status": "EXCLUDED", "reason": "C1 has no supported placement relation"})
            continue
        for index, relation in enumerate(relations):
            target = relation.get("target_resolution", {})
            # C1-V2's sealed relation schema retains target_is_region rather
            # than duplicating the internal semantic-role field.
            target_role = relation.get("target_semantic_role") or ("REGION_TARGET" if relation.get("target_is_region") else "OBJECT_TARGET")
            if target_role == "OBJECT_TARGET" and target.get("resolution") in {"EXACT_BODY", "EXACT_GEOM", "APPROVED_STRUCTURAL_ALIAS"}:
                classification = "DYNAMIC_RECONSTRUCTABLE"
                reason = "object target has a resolved body/geom; step-bound object_state evidence is still required"
                parent = None
                joint_count = None
            elif target.get("resolution") != "EXACT_SITE":
                classification = "ARTICULATED_UNKNOWN"
                reason = "target is not an exact region site"
                parent = None
                joint_count = None
            else:
                parent_id = int(target.get("parent_body_id", -1))
                parent = body_by_id.get(parent_id)
                joint_count = parent[1].get("joint_count") if parent else None
                if parent is None:
                    classification = "ARTICULATED_UNKNOWN"
                    reason = "site parent body missing from model metadata"
                elif joint_count == 0:
                    classification = "STATIC_FIXTURE"
                    reason = "region site parent has zero joints"
                elif any(token in parent[0].lower() for token in ("cabinet", "drawer")):
                    classification = "ARTICULATED_UNKNOWN"
                    reason = "articulated cabinet/drawer requires a sealed reconstruction contract"
                else:
                    classification = "DYNAMIC_RECONSTRUCTABLE"
                    reason = "region site parent has joints and local site transform"
            output.append({
                "task_key": task_key,
                "relation_index": index,
                "predicate": relation.get("predicate"),
                "object_bddl": relation.get("object_bddl"),
                "target_bddl": relation.get("target_bddl"),
                "target_resolution": target.get("resolution"),
                "parent_body": parent[0] if parent else None,
                "parent_joint_count": joint_count,
                "classification": classification,
                "observability_status": "MAPPING_ONLY_REPLAY_EVIDENCE_REQUIRED",
                "reason": reason,
                "unknown_is_negative": False,
                "silent_fallback": False,
            })
    return output


def load_episode_manifest(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"episode manifest does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        return data["episodes"]
    raise ValueError("episode manifest must be a list or {episodes: [...]} ")


def audit_episode_manifest(path: Path, allowlist: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and exactly replay an explicitly allowlisted geometry manifest."""
    manifest_path, root_entry = require_allowed_path(path, allowlist, episode=True)
    expected_manifest_sha = root_entry.get("manifest_sha256")
    if expected_manifest_sha and contract_sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("episode manifest SHA mismatch")
    entries = load_episode_manifest(manifest_path)
    seen: set[str] = set()
    audited: List[Dict[str, Any]] = []
    for entry in entries:
        episode_id = str(entry.get("episode_id", ""))
        if not episode_id or episode_id in seen:
            raise ValueError(f"duplicate or missing episode_id: {episode_id!r}")
        seen.add(episode_id)
        if entry.get("schema", GEOMETRY_SCHEMA) != GEOMETRY_SCHEMA:
            raise ValueError("episode geometry schema mismatch")
        if entry.get("task_key") not in EXPECTED_TASKS:
            raise ValueError(f"unknown episode task: {entry.get('task_key')}")
        step_count = entry.get("step_count")
        if type(step_count) is not int or step_count <= 0:
            raise ValueError(f"invalid step_count for {episode_id}")
        if entry.get("reference_chain") != "INDEPENDENT_WORLD_REFERENCE":
            raise ValueError(f"independent reference chain missing for {episode_id}")
        source_desc = entry.get("source_telemetry")
        reference_desc = entry.get("reference_telemetry")
        if not isinstance(source_desc, Mapping) or not isinstance(reference_desc, Mapping):
            raise ValueError(f"source/reference descriptors missing for {episode_id}")
        source_path, _ = require_allowed_path(Path(str(source_desc.get("path", ""))), allowlist, episode=True)
        reference_path, _ = require_allowed_path(Path(str(reference_desc.get("path", ""))), allowlist, episode=True)
        if source_path == reference_path:
            raise ValueError(f"reference chain is not independent for {episode_id}")
        for descriptor, resolved in ((source_desc, source_path), (reference_desc, reference_path)):
            declared = descriptor.get("sha256")
            if not isinstance(declared, str) or contract_sha256_file(resolved) != declared:
                raise ValueError(f"source SHA mismatch for {episode_id}: {resolved}")
        source_rows = load_jsonl_exact(source_path, episode_id=episode_id, step_count=step_count, role="source")
        reference_rows = load_jsonl_exact(reference_path, episode_id=episode_id, step_count=step_count, role="reference")
        audited.append(audit_episode_geometry(entry, source_rows, reference_rows))
    return audited, {
        "status": "PASS" if audited else "HOLD_INPUTS_MISSING",
        "manifest_path": str(manifest_path),
        "manifest_sha256": contract_sha256_file(manifest_path),
        "episode_count": len(audited),
        "source_reference_independence": "INDEPENDENT_WORLD_REFERENCE",
    }


def aggregate_replay_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    static_positions: List[float] = []
    static_rotations: List[float] = []
    dynamic_positions: List[float] = []
    dynamic_rotations: List[float] = []
    for row in rows:
        static_positions.extend(float(value) for value in row.get("static_position_errors_m", []))
        static_rotations.extend(float(value) for value in row.get("static_rotation_errors_rad", []))
        dynamic_positions.extend(float(value) for value in row.get("dynamic_position_errors_m", []))
        dynamic_rotations.extend(float(value) for value in row.get("dynamic_rotation_errors_rad", []))
    dynamic_position_p99 = p99(dynamic_positions)
    dynamic_rotation_p99 = p99(dynamic_rotations)
    return {
        "static_position_max_error_m": max(static_positions, default=None),
        "static_rotation_max_error_rad": max(static_rotations, default=None),
        "dynamic_position_p99_m": dynamic_position_p99["value"],
        "dynamic_rotation_p99_rad": dynamic_rotation_p99["value"],
        "dynamic_position_p99_denominator": dynamic_position_p99["count"],
        "dynamic_rotation_p99_denominator": dynamic_rotation_p99["count"],
        "episode_count": len(rows),
        "unknown_articulated_count": sum(int(r.get("unknown_articulated_count", 0)) for r in rows),
    }


def build_payload(source: Dict[str, Any], task_rows: Mapping[str, Dict[str, Any]], metadata: Mapping[str, Dict[str, Any]], episode_rows: Sequence[Dict[str, Any]], transform: Dict[str, Any]) -> Dict[str, Any]:
    geometry_rows = classify_rows(task_rows, metadata)
    canonical = {"source": source, "geometry_rows": geometry_rows, "episode_rows": list(episode_rows), "transform_contract": transform}
    return {
        "schema": "OFFICIAL_V3_C3_S3_GEOMETRY_OBSERVABILITY_V2",
        "source": source,
        "transform_contract": transform,
        "task_rows": geometry_rows,
        "episode_rows": list(episode_rows),
        "canonical_payload_sha256": sha256_bytes(canonical_json(canonical)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def seal_root(staging: Path, final: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if final.exists():
        raise FileExistsError(f"output root already exists: {final}")
    payload_files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    manifest["payload_files"] = payload_files
    write_json(staging / "MANIFEST.json", manifest)
    payload_files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    lines = [f"{sha256_file(staging / rel)}  {rel}" for rel in payload_files if rel not in {"SHA256SUMS", "SHA256SUMS.sha256"}]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.replace(staging, final)
    return {"sha256sums_sha256": sums_sha, "file_count": len(lines), "root": str(final)}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    r6_root = Path(args.r6_root).resolve()
    allowlist_path = Path(args.input_allowlist).resolve()
    allowlist = load_allowlist(allowlist_path)
    r6_resolved, r6_entry = require_allowed_path(r6_root, allowlist, episode=False, regular=False)
    r6_binding = verify_manifest_binding(r6_resolved, r6_entry)
    source, task_rows = compare_registry(r6_root)
    source_b, task_rows_b = compare_registry(r6_root)
    source["audit_code_commit"] = args.audit_code_commit
    source["audit_code_tree"] = args.audit_code_tree
    source_b["audit_code_commit"] = args.audit_code_commit
    source_b["audit_code_tree"] = args.audit_code_tree
    source["input_allowlist_path"] = str(allowlist_path)
    source["input_allowlist_sha256"] = contract_sha256_file(allowlist_path)
    source["r6_manifest_binding"] = r6_binding
    source_b["input_allowlist_path"] = str(allowlist_path)
    source_b["input_allowlist_sha256"] = contract_sha256_file(allowlist_path)
    source_b["r6_manifest_binding"] = r6_binding
    metadata = extract_model_metadata() if args.extract_model_metadata else {}
    metadata_b = extract_model_metadata() if args.extract_model_metadata else {}
    episode_rows: List[Dict[str, Any]] = []
    episode_inventory: Dict[str, Any] = {
        "status": "HOLD_INPUTS_MISSING",
        "manifest_path": None,
        "manifest_sha256": None,
        "episode_count": 0,
        "source_reference_independence": None,
    }
    if args.episode_manifest:
        episode_rows, episode_inventory = audit_episode_manifest(Path(args.episode_manifest).resolve(), allowlist)
    transform = transform_contract_tests()
    payload_a = build_payload(source, task_rows, metadata, episode_rows, transform)
    payload_b = build_payload(source_b, task_rows_b, metadata_b, episode_rows, transform)
    independent_equal = payload_a["canonical_payload_sha256"] == payload_b["canonical_payload_sha256"]
    task_rows_out = payload_a["task_rows"]
    supported_rows = [r for r in task_rows_out if r.get("classification") != "NON_PLACEMENT_EXCLUDED"]
    replay_metrics = aggregate_replay_metrics(episode_rows)
    replay_evidence = episode_inventory.get("status") == "PASS" and bool(episode_rows)
    hold_reasons: List[str] = []
    if not replay_evidence:
        hold_reasons.append("allowlisted per-episode geometry telemetry and independent world-reference root are unavailable")
    if any(r.get("classification") == "ARTICULATED_UNKNOWN" for r in supported_rows):
        hold_reasons.append("articulated geometry remains explicitly unknown and is not treated as negative")
    if not independent_equal:
        hold_reasons.append("independent canonical rebuild mismatch")
    if not transform["pass"]:
        hold_reasons.append("coordinate transform contract failed")
    summary = {
        "gate": "C3-S3",
        "schema": "OFFICIAL_V3_C3_S3_GEOMETRY_OBSERVABILITY_V2",
        "status": "PASS" if (independent_equal and transform["pass"] and replay_evidence and not any(r.get("classification") == "ARTICULATED_UNKNOWN" for r in supported_rows)) else "HOLD",
        "source_mutation": 0,
        "protected_reads": 0,
        "model_inference": False,
        "rollout_steps": 0,
        "attack_steps": 0,
        "mapping_rows": len(task_rows_out),
        "mapping_completeness": len(task_rows_out) > 0 and all(r.get("classification") in {"STATIC_FIXTURE", "DYNAMIC_RECONSTRUCTABLE", "ARTICULATED_UNKNOWN", "NON_PLACEMENT_EXCLUDED"} for r in task_rows_out),
        "supported_mapping_rows": len(supported_rows),
        "articulated_unknown_rows": sum(r.get("classification") == "ARTICULATED_UNKNOWN" for r in supported_rows),
        "episode_rows": len(episode_rows),
        "episode_manifest_present": bool(episode_rows),
        "input_inventory_status": episode_inventory.get("status"),
        "input_allowlist_path": str(allowlist_path),
        "input_allowlist_sha256": contract_sha256_file(allowlist_path),
        "static_replay_evidence": "PASS" if replay_evidence else "HOLD_INPUTS_MISSING",
        "dynamic_replay_evidence": "PASS" if replay_evidence else "HOLD_INPUTS_MISSING",
        "static_position_max_error_m": replay_metrics["static_position_max_error_m"],
        "static_rotation_max_error_rad": replay_metrics["static_rotation_max_error_rad"],
        "dynamic_position_p99_error_m": replay_metrics["dynamic_position_p99_m"],
        "dynamic_rotation_p99_error_rad": replay_metrics["dynamic_rotation_p99_rad"],
        "dynamic_position_p99_denominator": replay_metrics["dynamic_position_p99_denominator"],
        "dynamic_rotation_p99_denominator": replay_metrics["dynamic_rotation_p99_denominator"],
        "unknown_articulated_pose_count": replay_metrics["unknown_articulated_count"],
        "silent_fallback_count": sum(bool(r.get("silent_fallback")) for r in task_rows_out),
        "unknown_to_negative_count": sum(bool(r.get("unknown_is_negative")) for r in task_rows_out),
        "independent_canonical_digest_equal": independent_equal,
        "canonical_digest_a": payload_a["canonical_payload_sha256"],
        "canonical_digest_b": payload_b["canonical_payload_sha256"],
        "hold_reasons": hold_reasons,
    }
    parent = Path(args.out_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (args.output_name or f"c3_s3_geometry_observability_{args.source_commit[:8]}_{uuid.uuid4().hex[:8]}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        write_json(staging / "source_binding.json", source)
        write_json(staging / "transform_contract.json", transform)
        write_json(staging / "build_a.json", payload_a)
        write_json(staging / "build_b.json", payload_b)
        write_jsonl(staging / "task_geometry_rows.jsonl", task_rows_out)
        write_jsonl(staging / "episode_observability_rows.jsonl", episode_rows)
        write_json(staging / "summary.json", summary)
        seal = seal_root(staging, final, {
            "gate": "C3-S3",
            "schema": "OFFICIAL_V3_C3_S3_GEOMETRY_OBSERVABILITY_V2",
            "source_commit": args.source_commit,
            "audit_code_commit": args.audit_code_commit,
            "audit_code_tree": args.audit_code_tree,
            "source_tree": source["r6_source_tree"],
            "input_r6_root_sha256s_sha256": source["r6_root_sha256s_sha256"],
            "protected_reads": 0,
            "model_inference": False,
            "rollout_steps": 0,
            "attack_steps": 0,
            "input_allowlist_path": str(allowlist_path),
            "input_allowlist_sha256": contract_sha256_file(allowlist_path),
            "episode_inventory": episode_inventory,
        })
        print(json.dumps({"root": str(final), "status": summary["status"], "sha256sums_sha256": seal["sha256sums_sha256"], "canonical_digest": summary["canonical_digest_a"]}, sort_keys=True))
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-root", required=True)
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--audit-code-commit", required=True)
    parser.add_argument("--audit-code-tree", required=True)
    parser.add_argument("--input-allowlist", default=str(Path(__file__).resolve().parents[2] / "configs" / "C3_S3_ALLOWED_INPUTS_V1.json"))
    parser.add_argument("--extract-model-metadata", action="store_true")
    parser.add_argument("--episode-manifest")
    args = parser.parse_args()
    if any(token in str(args.r6_root).lower() for token in PROTECTED_TOKENS):
        raise SystemExit("protected input path rejected")
    try:
        run(args)
    except Exception as exc:
        print(f"C3-S3 HOLD: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
