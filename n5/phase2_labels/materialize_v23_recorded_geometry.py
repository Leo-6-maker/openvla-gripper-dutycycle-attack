"""Materialize geometry from recorded telemetry plus the frozen LIBERO model.

Unlike replay_v23_real_geometry.py, this module never calls env.step().
Object poses come from the sealed per-step object_state stream.  Static site
geometry and local bounds come from the frozen task model.  Unsupported
articulated targets remain UNKNOWN.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from replay_v23_real_geometry import (
    GeometryHold,
    body_local_bounds,
    body_path,
    compose,
    geom_local_bounds,
    mat_to_quat,
    pose,
    read_jsonl,
    sha256_file,
    verify_sealed_root,
    verify_source_files,
)


SCHEMA = "V23_RECORDED_TELEMETRY_GEOMETRY_V1"
OBJECT_STATE_WIDTH = 14
LIBERO_EXPECTED_HEAD = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_EXPECTED_TREE = "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60"
ALIAS_LEDGER_SCHEMA = "V23_RECORDED_GEOMETRY_ALIAS_LEDGER_V1"


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def model_inventory(model: Any) -> dict[str, Any]:
    return {
        "nbody": int(model.nbody),
        "nsite": int(model.nsite),
        "ngeom": int(model.ngeom),
        "bodies": [
            {
                "id": i,
                "name": str(model.body(i).name or ""),
                "parent_id": int(model.body_parentid[i]),
                "jnt_adr": int(model.body_jntadr[i]),
                "jnt_num": int(model.body_jntnum[i]),
            }
            for i in range(int(model.nbody))
        ],
        "sites": [
            {"id": i, "name": str(model.site(i).name or ""), "body_id": int(model.site_bodyid[i])}
            for i in range(int(model.nsite))
        ],
        "geoms": [
            {
                "id": i,
                "name": str(model.geom(i).name or ""),
                "body_id": int(model.geom_bodyid[i]),
                "type": int(model.geom_type[i]),
            }
            for i in range(int(model.ngeom))
        ],
    }


def model_inventory_sha(model: Any) -> str:
    return canonical_sha(model_inventory(model))


def git_snapshot(root: Path) -> tuple[str, str]:
    root = root.resolve()
    if not (root / ".git").exists():
        raise GeometryHold(f"LIBERO git metadata missing: {root}")
    def run(fmt: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), "show", "-s", f"--format={fmt}", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GeometryHold(f"LIBERO git snapshot unavailable: {root}") from exc
    return run("%H"), run("%T")


def verify_libero_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    bddl_root = (root / "libero" / "libero" / "bddl_files").resolve()
    if not root.is_dir() or not bddl_root.is_dir() or not bddl_root.is_relative_to(root):
        raise GeometryHold(f"invalid LIBERO root: {root}")
    head, tree = git_snapshot(root)
    if head != LIBERO_EXPECTED_HEAD or tree != LIBERO_EXPECTED_TREE:
        raise GeometryHold(f"LIBERO snapshot mismatch: {head}/{tree}")
    return {"root": str(root), "bddl_root": str(bddl_root), "head": head, "tree": tree}


def load_alias_ledger(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != ALIAS_LEDGER_SCHEMA or data.get("status") != "FROZEN_TASK_SPECIFIC_EXCEPTION":
        raise GeometryHold("alias ledger schema/status mismatch")
    entries = data.get("entries")
    if data.get("unique_alias_mappings") != 1 or not isinstance(entries, list) or len(entries) != 1:
        raise GeometryHold("alias ledger cardinality mismatch")
    result = {}
    for entry in entries:
        key = f"{entry.get('suite')}/task_{int(entry.get('task_id', -1)):02d}/{entry.get('bddl_object')}"
        if key in result:
            raise GeometryHold(f"duplicate alias ledger key: {key}")
        result[key] = entry
    return result


def xyzw_to_wxyz(value: list[float]) -> list[float]:
    if len(value) != 4 or not all(math.isfinite(float(x)) for x in value):
        raise GeometryHold("invalid recorded object quaternion")
    return [float(value[3]), float(value[0]), float(value[1]), float(value[2])]


def object_map(index_payload: Mapping[str, Any], suite: str, task_id: int) -> dict[str, dict[str, Any]]:
    for task in index_payload.get("tasks", []):
        if task.get("suite") == suite and int(task.get("task_id", -1)) == task_id:
            return {str(row["object_name"]): row for row in task.get("objects", [])}
    raise GeometryHold(f"index map task missing: {suite}/task_{task_id:02d}")


def recorded_body_pose(state: list[float], entry: Mapping[str, Any]) -> dict[str, list[float]]:
    start = int(entry["slice_start"])
    end = int(entry["slice_end_exclusive"])
    if end > len(state) or end - start != OBJECT_STATE_WIDTH:
        raise GeometryHold("recorded object-state slice mismatch")
    return pose(state[start:start + 3], xyzw_to_wxyz(state[start + 3:start + 7]))


def model_body_pose(sim: Any, model: Any, body_id: int) -> dict[str, list[float]]:
    return pose(sim.data.body_xpos[body_id].tolist(), sim.data.body_xquat[body_id].tolist())


def subtree_has_articulated_joint(model: Any, root_id: int) -> bool:
    body_ids = {
        body_id
        for body_id in range(int(model.nbody))
        if body_id == root_id or body_path(model, root_id, body_id) is not None
    }
    # MuJoCo type 0 is a free joint: it makes an object dynamic, but it is not
    # an articulated target. Hinge/slide/ball joints anywhere below the root
    # remain fail-closed.
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_bodyid[joint_id]) in body_ids and int(model.jnt_type[joint_id]) != 0:
            return True
    return False


def dynamic_root_for_site(model: Any, site_id: int, object_roots: Mapping[str, int]) -> tuple[str, int] | None:
    current = int(model.site_bodyid[site_id])
    chain = []
    while current > 0:
        chain.append(current)
        for name, root_id in object_roots.items():
            if current == root_id:
                return name, root_id
        current = int(model.body_parentid[current])
    return None


def site_local_pose(model: Any, site_id: int, root_id: int) -> dict[str, list[float]]:
    child_id = int(model.site_bodyid[site_id])
    local = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    for body_id in body_path(model, root_id, child_id) or []:
        local = compose(local, pose(model.body_pos[body_id].tolist(), model.body_quat[body_id].tolist()))
    local = compose(local, pose(model.site_pos[site_id].tolist(), model.site_quat[site_id].tolist()))
    return local


def geom_local_pose(model: Any, geom_id: int, root_id: int) -> dict[str, list[float]]:
    body_id = int(model.geom_bodyid[geom_id])
    path = body_path(model, root_id, body_id)
    if path is None:
        raise GeometryHold("geometry is outside recorded object body")
    local = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    for child_id in path:
        local = compose(local, pose(model.body_pos[child_id].tolist(), model.body_quat[child_id].tolist()))
    return compose(local, pose(model.geom_pos[geom_id].tolist(), model.geom_quat[geom_id].tolist()))


def static_site_pose(sim: Any, model: Any, site_id: int) -> dict[str, list[float]]:
    return pose(sim.data.site_xpos[site_id].tolist(), mat_to_quat(sim.data.site_xmat[site_id].tolist()))


def target_case(
    sim: Any,
    model: Any,
    resolution: Mapping[str, Any],
    state: list[float],
    objects: Mapping[str, Mapping[str, Any]],
    object_roots: Mapping[str, int],
    bounds_cache: dict[tuple[str, int], tuple[list[float], list[float]]],
) -> dict[str, Any]:
    name = str(resolution.get("alias_to") or resolution.get("name") or "")
    kind = resolution.get("entity_type")
    if kind == "site":
        site_id = int(model.site(name).id)
        if site_id != int(resolution.get("entity_id", -1)):
            raise GeometryHold(f"site identity mismatch: {name}")
        parent_id = int(model.site_bodyid[site_id])
        if parent_id != int(resolution.get("parent_body_id", parent_id)):
            raise GeometryHold(f"site parent identity mismatch: {name}")
        if str(model.body(parent_id).name) != str(resolution.get("parent_body_name", model.body(parent_id).name)):
            raise GeometryHold(f"site parent name mismatch: {name}")
        dynamic = dynamic_root_for_site(model, site_id, object_roots)
        if dynamic is not None:
            object_name, root_id = dynamic
            if subtree_has_articulated_joint(model, root_id):
                return {"id": name, "role": "REGION_TARGET", "source": "UNKNOWN_ARTICULATED", "known": False}
            body_origin = recorded_body_pose(state, objects[object_name])
            site_local = site_local_pose(model, site_id, root_id)
            site_pose = compose(body_origin, site_local)
            source = "RECORDED_OBJECT_STATE_FROZEN_SITE_LOCAL"
        else:
            site_pose = static_site_pose(sim, model, site_id)
            source = "FROZEN_MODEL_SITE"
        return {
            "id": name,
            "role": "REGION_TARGET",
            "pose": site_pose,
            "parent_body_origin_pose": body_origin if dynamic is not None else None,
            "local_geometry_pose": site_local if dynamic is not None else None,
            "half_extents": [float(x) for x in model.site_size[site_id].tolist()],
            "source": source,
            "known": True,
        }
    if kind == "body":
        alias_name = name
        body_id = int(model.body(alias_name).id)
        if body_id != int(resolution.get("entity_id", -1)):
            raise GeometryHold(f"body identity mismatch: {alias_name}")
        bddl_name = str(resolution.get("alias_from") or alias_name.removesuffix("_main"))
        has_joint = subtree_has_articulated_joint(model, body_id)
        if has_joint:
            return {"id": bddl_name, "role": "OBJECT_TARGET", "source": "UNKNOWN_ARTICULATED", "known": False}
        center, half = body_local_bounds(model, body_id, bounds_cache)
        if bddl_name in objects:
            body_pose = recorded_body_pose(state, objects[bddl_name])
            source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL"
        else:
            body_pose = model_body_pose(sim, model, body_id)
            source = "FROZEN_MODEL_BODY"
        return {
            "id": bddl_name,
            "role": "OBJECT_TARGET",
            "pose": compose(body_pose, pose(center, [1.0, 0.0, 0.0, 0.0])),
            "body_origin_pose": body_pose,
            "local_geometry_center": center,
            "half_extents": half,
            "source": source,
            "known": True,
        }
    return {"id": name, "role": "UNKNOWN", "source": "UNKNOWN_TARGET_RESOLUTION", "known": False}


def ancestor_chain(model: Any, body_id: int) -> list[dict[str, Any]]:
    chain = []
    current = int(body_id)
    while current >= 0:
        chain.append({"body_id": current, "body_name": str(model.body(current).name or "")})
        if current == 0:
            break
        current = int(model.body_parentid[current])
    if not chain or chain[-1]["body_id"] != 0:
        raise GeometryHold(f"invalid body ancestor chain: {body_id}")
    return list(reversed(chain))


def validate_alias_exception(
    entry: Mapping[str, Any],
    suite: str,
    task_id: int,
    bddl_sha256: str,
    registry_task_sha256: str,
    inventory_sha256: str,
    resolution: Mapping[str, Any],
    index_entry: Mapping[str, Any],
    model: Any,
) -> None:
    geom_id = int(resolution.get("entity_id", -1))
    geom_name = str(model.geom(geom_id).name or "") if 0 <= geom_id < int(model.ngeom) else ""
    geom_body_id = int(model.geom_bodyid[geom_id]) if geom_name else -1
    actual = {
        "suite": suite,
        "task_id": task_id,
        "bddl_object": str(resolution.get("name") or ""),
        "registry_resolution": str(resolution.get("resolution") or ""),
        "registry_geom_id": geom_id,
        "registry_geom_name": geom_name,
        "registry_geom_parent_body_id": geom_body_id,
        "registry_geom_parent_body_name": str(model.body(geom_body_id).name or "") if geom_body_id >= 0 else "",
        "index_map_object_index": int(index_entry.get("object_index", -1)),
        "index_map_body_id": int(index_entry.get("body_id", -1)),
        "index_map_body_name": str(index_entry.get("body_name") or ""),
        "object_state_slice_start": int(index_entry.get("slice_start", -1)),
        "object_state_slice_end_exclusive": int(index_entry.get("slice_end_exclusive", -1)),
        "object_state_slice_width": int(index_entry.get("slice_end_exclusive", -1)) - int(index_entry.get("slice_start", -1)),
        "bddl_sha256": bddl_sha256,
        "registry_task_sha256": registry_task_sha256,
        "model_inventory_sha256": inventory_sha256,
    }
    for key, value in actual.items():
        if entry.get(key) != value:
            raise GeometryHold(f"alias ledger mismatch: {key}")
    if entry.get("registry_geom_ancestor_chain") != ancestor_chain(model, geom_body_id):
        raise GeometryHold("alias ledger registry ancestor mismatch")
    if entry.get("index_map_body_ancestor_chain") != ancestor_chain(model, int(index_entry["body_id"])):
        raise GeometryHold("alias ledger object ancestor mismatch")


def object_case(
    model: Any,
    resolution: Mapping[str, Any],
    state: list[float],
    objects: Mapping[str, Mapping[str, Any]],
    bounds_cache: dict[tuple[str, int], tuple[list[float], list[float]]],
    alias_entry: Mapping[str, Any] | None,
    suite: str,
    task_id: int,
    bddl_sha256: str,
    registry_task_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    bddl_name = str(resolution.get("alias_from") or resolution.get("name") or "")
    if bddl_name not in objects:
        return {"id": bddl_name, "role": "MANIPULATED_OBJECT", "source": "UNKNOWN_OBJECT_MAPPING", "known": False}
    entry = objects[bddl_name]
    body_name = str(entry.get("body_name", ""))
    if not body_name or int(model.body(body_name).id) != int(entry.get("body_id", -1)):
        raise GeometryHold(f"object-state/model body mismatch: {bddl_name}")
    body_id = int(entry["body_id"])
    kind = resolution.get("entity_type")
    identity_status = "INDEX_BODY_IDENTITY"
    if kind == "body":
        if body_id != int(resolution.get("entity_id", -1)):
            raise GeometryHold(f"object-state/body identity mismatch: {bddl_name}")
        local_pose = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        center, half = body_local_bounds(model, body_id, bounds_cache)
        source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL"
    elif kind == "geom":
        geom_id = int(resolution.get("entity_id", -1))
        if not 0 <= geom_id < int(model.ngeom):
            raise GeometryHold(f"object-state/geom identity mismatch: {bddl_name}")
        geom_body_id = int(model.geom_bodyid[geom_id])
        geom_name = str(model.geom(geom_id).name or "")
        if geom_body_id == body_id:
            local_pose = geom_local_pose(model, geom_id, body_id)
            center, half = geom_local_bounds(model, geom_id, bounds_cache)
            source = "RECORDED_OBJECT_STATE_FROZEN_GEOM_LOCAL"
            identity_status = "EXACT_GEOM_TO_INDEX_BODY"
        elif geom_body_id == 0 and geom_name == bddl_name:
            if alias_entry is None:
                raise GeometryHold(f"unledgered init geom alias: {bddl_name}")
            validate_alias_exception(alias_entry, suite, task_id, bddl_sha256, registry_task_sha256, inventory_sha256, resolution, entry, model)
            # Some C1 rows name an init marker geom after the BDDL object;
            # the physical body is the sealed object-state index-map binding.
            local_pose = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
            center, half = body_local_bounds(model, body_id, bounds_cache)
            source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL_INIT_GEOM_ALIAS"
            identity_status = "INIT_GEOM_ALIAS_TO_INDEX_BODY"
        else:
            raise GeometryHold(f"object-state/geom identity mismatch: {bddl_name}")
    else:
        return {"id": bddl_name, "role": "MANIPULATED_OBJECT", "source": "UNKNOWN_OBJECT_MAPPING", "known": False}
    body_origin = recorded_body_pose(state, entry)
    return {
        "id": bddl_name,
        "role": "MANIPULATED_OBJECT",
        "pose": compose(body_origin, compose(local_pose, pose(center, [1.0, 0.0, 0.0, 0.0]))),
        "body_origin_pose": body_origin,
        "local_geometry_pose": local_pose,
        "local_geometry_center": center,
        "half_extents": half,
        "source": source,
        "geometry_entity_type": kind,
        "registry_identity_status": identity_status,
        "known": True,
    }


def build_episode(
    record: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    registry_root: Path,
    libero_info: Mapping[str, Any],
    alias_ledger: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = str(record["suite"])
    task_id = int(record["task_id"])
    episode_id = str(record["episode_id"])
    metadata_path, step_path, sidecar_path = verify_source_files(record)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    steps = read_jsonl(step_path)
    sidecars = read_jsonl(sidecar_path)
    if len(steps) != len(sidecars) or [int(row.get("step", -1)) for row in sidecars] != list(range(len(sidecars))):
        raise GeometryHold(f"source step closure failed: {episode_id}")
    task_file = registry_root / "run_A" / "per_task" / f"{suite}_task_{task_id:02d}.json"
    if not task_file.is_file():
        raise GeometryHold(f"registry task missing: {task_file}")
    registry = json.loads(task_file.read_text(encoding="utf-8"))["legacy"]
    if registry.get("status") != "OK":
        raise GeometryHold(f"registry task not OK: {episode_id}")
    b = get_benchmark(suite)(0)
    task = b.get_task(task_id)
    bddl_root = Path(str(libero_info["bddl_root"]))
    if Path(get_libero_path("bddl_files")).resolve() != bddl_root.resolve():
        raise GeometryHold("installed LIBERO BDDL root differs from --libero-root")
    bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
    if not bddl.is_file() or not bddl.is_relative_to(bddl_root):
        raise GeometryHold(f"BDDL escapes --libero-root: {bddl}")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64,
                             render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=2)
    try:
        env.reset()
        inner = env.env
        index_objects = object_map(index_payload, suite, task_id)
        bddl_sha256 = sha256_file(bddl)
        registry_task_sha256 = sha256_file(task_file)
        inventory_sha256 = model_inventory_sha(inner.sim.model)
        object_roots = {}
        for name, entry in index_objects.items():
            body_name = str(entry.get("body_name", ""))
            if not body_name:
                raise GeometryHold(f"index map body name missing: {episode_id}:{name}")
            body_id = int(inner.sim.model.body(body_name).id)
            if body_id != int(entry.get("body_id", -1)):
                raise GeometryHold(f"index map body id mismatch: {episode_id}:{name}")
            object_roots[name] = body_id
        bounds_cache: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
        rows = []
        alias_occurrence_rows = 0
        for sidecar in sidecars:
            state = [float(x) for x in sidecar.get("object_state", [])]
            if len(state) != sum(1 for _ in index_objects) * OBJECT_STATE_WIDTH:
                # Per-task object count is the index-map cardinality; no fallback.
                raise GeometryHold(f"object-state width mismatch: {episode_id}:{sidecar.get('step')}")
            relations = []
            for relation_index, relation in enumerate(registry.get("relations", [])):
                object_resolution = relation["object_resolution"]
                object_name = str(object_resolution.get("alias_from") or object_resolution.get("name") or "")
                alias_key = f"{suite}/task_{task_id:02d}/{object_name}"
                object_row = object_case(
                    inner.sim.model,
                    object_resolution,
                    state,
                    index_objects,
                    bounds_cache,
                    alias_ledger.get(alias_key),
                    suite,
                    task_id,
                    bddl_sha256,
                    registry_task_sha256,
                    inventory_sha256,
                )
                alias_occurrence_rows += int(object_row.get("registry_identity_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY")
                target_row = target_case(inner.sim, inner.sim.model, relation["target_resolution"], state, index_objects, object_roots, bounds_cache)
                relations.append({
                    "episode_id": episode_id,
                    "step": int(sidecar["step"]),
                    "predicate": relation["predicate"],
                    "relation_index": relation_index,
                    "object": object_row,
                    "target": target_row,
                    "geometry_source": "RECORDED_TELEMETRY_PLUS_FROZEN_MODEL",
                    "known": bool(object_row.get("known") and target_row.get("known")),
                })
            rows.append({"episode_id": episode_id, "step": int(sidecar["step"]), "relations": relations})
        return {
            "episode_id": episode_id,
            "suite": suite,
            "task_id": task_id,
            "state_id": int(record["state_id"]),
            "step_count": len(rows),
            "source_mode": "RECORDED_TELEMETRY_PLUS_FROZEN_MODEL",
            "pilot_metadata_sha256": sha256_file(metadata_path),
            "pilot_steps_sha256": sha256_file(step_path),
            "pilot_sidecar_sha256": sha256_file(sidecar_path),
            "registry_task_sha256": registry_task_sha256,
            "bddl_sha256": bddl_sha256,
            "model_inventory_sha256": inventory_sha256,
            "alias_occurrence_rows": alias_occurrence_rows,
            "libero_head": libero_info["head"],
            "libero_tree": libero_info["tree"],
            "collector_source_sha256": metadata.get("collector_source_sha256"),
            "protected_payload_read": False,
            "action_replay": False,
            "model_inference": False,
        }, rows
    finally:
        env.close()


def write_root(output_root: Path, root_payload: dict[str, Any], episode_payloads: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise GeometryHold(f"output exists: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise GeometryHold(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        episode_root = staging / "episodes"
        episode_root.mkdir()
        for manifest, rows in episode_payloads:
            directory = episode_root / manifest["episode_id"].replace("/", "__")
            directory.mkdir()
            (directory / "episode_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with (directory / "geometry_cases.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        (staging / "MANIFEST.json").write_text(json.dumps(root_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payloads = sorted(p for p in staging.rglob("*") if p.is_file())
        sums = "\n".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}" for p in payloads) + "\n"
        (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink(): path.unlink()
            elif path.is_dir(): path.rmdir()
        staging.rmdir()
        raise


def verify_pilot_inputs(pilot: Mapping[str, Any], pilot_manifest: Path) -> dict[str, Any]:
    if pilot.get("schema") != "V23_DEV_PILOT_V1" or pilot.get("protected_payload_read") is not False:
        raise GeometryHold("pilot input contract does not prove FIT-only exclusion")
    if pilot.get("episode_count") != 40 or len(pilot.get("records", [])) != 40:
        raise GeometryHold("pilot episode closure is not 40")
    pilot_seal = verify_sealed_root(pilot_manifest.parent)
    d0_spec = pilot.get("d0_receipt", {})
    d0_path = Path(str(d0_spec.get("path", ""))).resolve()
    if not d0_path.is_file() or sha256_file(d0_path) != d0_spec.get("sha256"):
        raise GeometryHold("D0 receipt binding failed")
    d0_root = verify_sealed_root(d0_path.parent)
    d0 = json.loads(d0_path.read_text(encoding="utf-8"))
    aggregate = d0.get("aggregate", {})
    if (
        d0.get("decision") != "PASS"
        or d0.get("dev_pool_closure_670") is not True
        or aggregate.get("dev_pool_unique") != 670
        or aggregate.get("clean_protected_overlap_count") != 1330
        or aggregate.get("protected_cross_manifest_overlap_count") != 0
        or aggregate.get("protected_union_unique") != 1330
    ):
        raise GeometryHold("D0 receipt decision/closure mismatch")
    dev_spec = pilot.get("dev_pool_manifest", {})
    dev_path = Path(str(dev_spec.get("path", ""))).resolve()
    if not dev_path.is_file() or sha256_file(dev_path) != dev_spec.get("sha256"):
        raise GeometryHold("DEV_POOL manifest binding failed")
    if d0.get("dev_pool_identity_manifest_sha256") != dev_spec.get("sha256"):
        raise GeometryHold("D0 DEV_POOL manifest binding failed")
    with dev_path.open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    dev_ids = {(str(row["suite"]), int(row["task_id"]), int(row["state_id"])) for row in dev_rows}
    if len(dev_rows) != 670 or len(dev_ids) != 670:
        raise GeometryHold("DEV_POOL identity closure failed")
    pilot_ids = set()
    for record in pilot.get("records", []):
        key = (str(record["suite"]), int(record["task_id"]), int(record["state_id"]))
        if key in pilot_ids or key not in dev_ids:
            raise GeometryHold(f"pilot identity is duplicate or outside DEV_POOL: {key}")
        pilot_ids.add(key)
    return {
        "pilot_root_sha256s_sha256": pilot_seal["sha256sums_sha256"],
        "d0_root_sha256s_sha256": d0_root["sha256sums_sha256"],
        "d0_receipt_sha256": sha256_file(d0_path),
        "dev_pool_manifest_sha256": sha256_file(dev_path),
        "dev_pool_unique": len(dev_ids),
        "pilot_in_dev_pool": len(pilot_ids),
        "protected_overlap_count": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", type=Path, required=True)
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--registry-root", type=Path, required=True)
    ap.add_argument("--libero-root", type=Path, required=True)
    ap.add_argument("--alias-ledger", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--code-snapshot-commit", required=True)
    ap.add_argument("--canary", action="store_true")
    args = ap.parse_args()
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    records = pilot.get("records", [])
    input_binding = verify_pilot_inputs(pilot, args.pilot_manifest)
    libero_info = verify_libero_root(args.libero_root)
    alias_ledger = load_alias_ledger(args.alias_ledger)
    index_seal = verify_sealed_root(args.index_root)
    registry_seal = verify_sealed_root(args.registry_root)
    index_payload = json.loads((args.index_root / "OBJECT_STATE_INDEX_MAP_V1.json").read_text(encoding="utf-8"))
    if index_payload.get("status") != "DERIVED_STATIC_SCHEMA_SEALED":
        raise GeometryHold("index map is not sealed derived schema")
    chosen = []
    seen_suites = set()
    for record in records:
        if args.canary and record["suite"] in seen_suites:
            continue
        chosen.append(record)
        seen_suites.add(record["suite"])
    episodes = [build_episode(record, index_payload, args.registry_root, libero_info, alias_ledger) for record in chosen]
    root_payload = {
        "schema": SCHEMA,
        "status": "DERIVED_RECORDED_GEOMETRY_NONCONSUMABLE",
        "code_snapshot_commit": args.code_snapshot_commit,
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "alias_ledger_sha256": sha256_file(args.alias_ledger),
        "pilot_manifest_sha256": sha256_file(args.pilot_manifest),
        **input_binding,
        "libero_head": libero_info["head"],
        "libero_tree": libero_info["tree"],
        "index_root_sha256s_sha256": index_seal["sha256sums_sha256"],
        "registry_root_sha256s_sha256": registry_seal["sha256sums_sha256"],
        "episode_count": len(episodes),
        "source_mode": "RECORDED_TELEMETRY_PLUS_FROZEN_MODEL",
        "protected_payload_read": False,
        "action_replay": False,
        "model_inference": False,
        "teacher_labeling": False,
        "consumer_eligible": False,
    }
    write_root(args.output_root, root_payload, episodes)
    print(json.dumps({"status": root_payload["status"], "episode_count": len(episodes), "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
