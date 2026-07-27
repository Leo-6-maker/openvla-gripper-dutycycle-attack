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


def chain_has_articulated_joint(model: Any, root_id: int, child_id: int) -> bool:
    path = body_path(model, root_id, child_id) or []
    body_ids = {root_id, *path}
    # MuJoCo type 0 is a free joint: it makes an object dynamic, but it is not
    # an articulated target. Hinge/slide/ball joints remain fail-closed.
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
        dynamic = dynamic_root_for_site(model, site_id, object_roots)
        if dynamic is not None:
            object_name, root_id = dynamic
            if chain_has_articulated_joint(model, root_id, int(model.site_bodyid[site_id])):
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
        bddl_name = str(resolution.get("alias_from") or alias_name.removesuffix("_main"))
        has_joint = chain_has_articulated_joint(model, body_id, body_id)
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


def object_case(
    model: Any,
    resolution: Mapping[str, Any],
    state: list[float],
    objects: Mapping[str, Mapping[str, Any]],
    bounds_cache: dict[tuple[str, int], tuple[list[float], list[float]]],
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
    if kind == "body":
        if body_id != int(resolution.get("entity_id", -1)):
            raise GeometryHold(f"object-state/body identity mismatch: {bddl_name}")
        local_pose = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        center, half = body_local_bounds(model, body_id, bounds_cache)
        source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL"
    elif kind == "geom":
        geom_id = int(resolution.get("entity_id", -1))
        if not 0 <= geom_id < int(model.ngeom) or int(model.geom_bodyid[geom_id]) != body_id:
            raise GeometryHold(f"object-state/geom identity mismatch: {bddl_name}")
        local_pose = geom_local_pose(model, geom_id, body_id)
        center, half = geom_local_bounds(model, geom_id, bounds_cache)
        source = "RECORDED_OBJECT_STATE_FROZEN_GEOM_LOCAL"
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
        "known": True,
    }


def build_episode(record: Mapping[str, Any], index_payload: Mapping[str, Any], registry_root: Path, libero_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64,
                             render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=2)
    try:
        env.reset()
        inner = env.env
        index_objects = object_map(index_payload, suite, task_id)
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
        for sidecar in sidecars:
            state = [float(x) for x in sidecar.get("object_state", [])]
            if len(state) != sum(1 for _ in index_objects) * OBJECT_STATE_WIDTH:
                # Per-task object count is the index-map cardinality; no fallback.
                raise GeometryHold(f"object-state width mismatch: {episode_id}:{sidecar.get('step')}")
            relations = []
            for relation_index, relation in enumerate(registry.get("relations", [])):
                object_row = object_case(inner.sim.model, relation["object_resolution"], state, index_objects, bounds_cache)
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
            "registry_task_sha256": sha256_file(task_file),
            "bddl_sha256": sha256_file(bddl),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", type=Path, required=True)
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--registry-root", type=Path, required=True)
    ap.add_argument("--libero-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--code-snapshot-commit", required=True)
    ap.add_argument("--canary", action="store_true")
    args = ap.parse_args()
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    records = pilot.get("records", [])
    if pilot.get("protected_payload_read") is not False:
        raise GeometryHold("pilot manifest does not prove protected exclusion")
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
    episodes = [build_episode(record, index_payload, args.registry_root, args.libero_root) for record in chosen]
    root_payload = {
        "schema": SCHEMA,
        "status": "DERIVED_RECORDED_GEOMETRY_NONCONSUMABLE",
        "code_snapshot_commit": args.code_snapshot_commit,
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "pilot_manifest_sha256": sha256_file(args.pilot_manifest),
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
