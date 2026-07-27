"""Independent G-REC verifier.

This file deliberately does not import the production materializer or replay
geometry helpers. It recomputes pose composition and bounds from the recorded
object-state stream and the frozen LIBERO model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ReviewHold(RuntimeError):
    pass


LIBERO_HEAD = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_TREE = "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60"
POS_TOL = 1e-6
ROT_TOL = 1e-6
EXTENT_TOL = 1e-6
BODY_ROT_TOL = 1e-7


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


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


def verify_seal(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise ReviewHold(f"seal missing: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if side != [sha256_file(sums), "SHA256SUMS"]:
        raise ReviewHold(f"seal sidecar mismatch: {root}")
    expected = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(None, 1)
        name = name.lstrip("*").strip()
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ReviewHold(f"unsafe sealed path: {name}")
        target = root / rel
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise ReviewHold(f"sealed payload mismatch: {target}")
        expected[rel.as_posix()] = digest
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if set(expected) != actual:
        raise ReviewHold(f"sealed closure mismatch: {root}")
    return sha256_file(sums)


def qnorm(q: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in q)
    norm = math.sqrt(sum(x * x for x in values))
    if len(values) != 4 or not math.isfinite(norm) or norm <= 0:
        raise ReviewHold("invalid quaternion")
    return tuple(x / norm for x in values)  # type: ignore[return-value]


def qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = qnorm(a)
    w2, x2, y2, z2 = qnorm(b)
    return qnorm((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                  w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2))


def qrotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = qnorm(q)
    vx, vy, vz = (float(x) for x in v)
    tx = 2 * (y*vz-z*vy); ty = 2 * (z*vx-x*vz); tz = 2 * (x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def mat_to_quat(m: Sequence[float]) -> tuple[float, float, float, float]:
    if len(m) != 9 or not all(math.isfinite(float(x)) for x in m):
        raise ReviewHold("invalid rotation matrix")
    a00,a01,a02,a10,a11,a12,a20,a21,a22 = (float(x) for x in m)
    trace = a00 + a11 + a22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2; q = (0.25*s, (a21-a12)/s, (a02-a20)/s, (a10-a01)/s)
    elif a00 > a11 and a00 > a22:
        s = math.sqrt(1+a00-a11-a22)*2; q = ((a21-a12)/s, 0.25*s, (a01+a10)/s, (a02+a20)/s)
    elif a11 > a22:
        s = math.sqrt(1+a11-a00-a22)*2; q = ((a02-a20)/s, (a01+a10)/s, 0.25*s, (a12+a21)/s)
    else:
        s = math.sqrt(1+a22-a00-a11)*2; q = ((a10-a01)/s, (a02+a20)/s, (a12+a21)/s, 0.25*s)
    return qnorm(q)


def pose(pos: Sequence[float], quat: Sequence[float]) -> dict[str, list[float]]:
    if len(pos) != 3 or not all(math.isfinite(float(x)) for x in pos):
        raise ReviewHold("invalid position")
    q = qnorm(quat)
    return {"pos": [float(x) for x in pos], "quat": list(q)}


def compose(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    moved = qrotate(a["quat"], b["pos"])
    return {"pos": [float(a["pos"][i]) + moved[i] for i in range(3)], "quat": list(qmul(a["quat"], b["quat"]))}


def quat_distance(a: Sequence[float], b: Sequence[float]) -> float:
    qa, qb = qnorm(a), qnorm(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    dot = min(1.0, max(0.0, dot))
    return 2 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot)


def body_path(model: Any, root_id: int, child_id: int) -> list[int] | None:
    path = []
    current = int(child_id)
    while current != int(root_id):
        path.append(current)
        current = int(model.body_parentid[current])
        if current <= 0 or len(path) > int(model.nbody):
            return None
    return list(reversed(path))


def world_body_path(model: Any, child_id: int) -> list[int]:
    """Return the complete model-local chain from world body to child."""
    path = []
    current = int(child_id)
    for _ in range(int(model.nbody)):
        if current == 0:
            return list(reversed(path))
        if current < 0 or current >= int(model.nbody):
            break
        path.append(current)
        current = int(model.body_parentid[current])
    raise ReviewHold(f"invalid world ancestor chain: {child_id}")


def body_descendant_or_self(model: Any, root_id: int, body_id: int) -> bool:
    current = int(body_id)
    for _ in range(int(model.nbody)):
        if current == int(root_id):
            return True
        if current <= 0 or current >= int(model.nbody):
            return False
        current = int(model.body_parentid[current])
    return False


def body_chain_has_joint(model: Any, body_id: int, include_free: bool = True) -> bool:
    chain = set(world_body_path(model, body_id))
    return any(
        int(model.jnt_bodyid[joint_id]) in chain
        and (include_free or int(model.jnt_type[joint_id]) != 0)
        for joint_id in range(int(model.njnt))
    )


def fixed_body_pose(model: Any, body_id: int) -> dict[str, list[float]]:
    if body_chain_has_joint(model, body_id, include_free=True):
        raise ReviewHold(f"fixed body has a joint: {body_id}")
    result = pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    for ancestor_id in world_body_path(model, body_id):
        result = compose(result, pose(model.body_pos[ancestor_id].tolist(), model.body_quat[ancestor_id].tolist()))
    return result


def fixed_site_pose(model: Any, site_id: int) -> dict[str, list[float]]:
    parent_id = int(model.site_bodyid[site_id])
    result = fixed_body_pose(model, parent_id)
    return compose(result, pose(model.site_pos[site_id].tolist(), model.site_quat[site_id].tolist()))


def shape_vertices(model: Any, geom_id: int) -> list[tuple[float, float, float]]:
    kind = int(model.geom_type[geom_id]); size = [abs(float(x)) for x in model.geom_size[geom_id].tolist()]
    if kind == 2: half = [size[0]] * 3
    elif kind == 3: half = [size[0], size[0], size[0] + size[1]]
    elif kind == 4: half = size[:3]
    elif kind == 5: half = [size[0], size[0], size[1]]
    elif kind == 6: half = size[:3]
    elif kind == 7:
        mesh_id = int(model.geom_dataid[geom_id]); start = int(model.mesh_vertadr[mesh_id]); count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start+count], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
            raise ReviewHold("mesh geometry unavailable")
        return [tuple(float(vertices[i, j]) * size[j] for j in range(3)) for i in range(len(vertices))]
    else:
        raise ReviewHold(f"unsupported geom type: {kind}")
    if len(half) != 3 or any(x <= 0 or not math.isfinite(x) for x in half):
        raise ReviewHold("degenerate geometry")
    return [(sx*half[0], sy*half[1], sz*half[2]) for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]


def local_bounds(model: Any, body_id: int, cache: dict[tuple[str, int], tuple[list[float], list[float]]]) -> tuple[list[float], list[float]]:
    key = ("body", int(body_id))
    if key in cache: return cache[key]
    ids = [i for i in range(int(model.nbody)) if i == body_id or body_path(model, body_id, i) is not None]
    corners = []
    for gid in range(int(model.ngeom)):
        geom_body = int(model.geom_bodyid[gid])
        if geom_body not in ids: continue
        path = body_path(model, body_id, geom_body)
        if path is None: continue
        local = pose([0,0,0], [1,0,0,0])
        for bid in path: local = compose(local, pose(model.body_pos[bid].tolist(), model.body_quat[bid].tolist()))
        local = compose(local, pose(model.geom_pos[gid].tolist(), model.geom_quat[gid].tolist()))
        for corner in shape_vertices(model, gid):
            shifted = qrotate(local["quat"], corner)
            corners.append(tuple(local["pos"][i] + shifted[i] for i in range(3)))
    if not corners: raise ReviewHold(f"body has no supported geoms: {body_id}")
    lo = [min(x[i] for x in corners) for i in range(3)]; hi = [max(x[i] for x in corners) for i in range(3)]
    center = [(lo[i]+hi[i])/2 for i in range(3)]; half = [(hi[i]-lo[i])/2 for i in range(3)]
    if any(x <= 0 or not math.isfinite(x) for x in half): raise ReviewHold("invalid body bounds")
    cache[key] = (center, half); return center, half


def geom_bounds(model: Any, geom_id: int, cache: dict[tuple[str, int], tuple[list[float], list[float]]]) -> tuple[list[float], list[float]]:
    key = ("geom", int(geom_id))
    if key in cache: return cache[key]
    vertices = shape_vertices(model, geom_id)
    lo = [min(x[i] for x in vertices) for i in range(3)]; hi = [max(x[i] for x in vertices) for i in range(3)]
    center = [(lo[i]+hi[i])/2 for i in range(3)]; half = [(hi[i]-lo[i])/2 for i in range(3)]
    if any(x <= 0 or not math.isfinite(x) for x in half): raise ReviewHold("invalid geom bounds")
    cache[key] = (center, half); return center, half


def local_body_child_pose(model: Any, root_id: int, child_id: int) -> dict[str, list[float]]:
    path = body_path(model, root_id, child_id)
    if path is None: raise ReviewHold("body hierarchy mismatch")
    local = pose([0,0,0], [1,0,0,0])
    for bid in path: local = compose(local, pose(model.body_pos[bid].tolist(), model.body_quat[bid].tolist()))
    return local


def local_site_pose(model: Any, site_id: int, root_id: int) -> dict[str, list[float]]:
    return compose(local_body_child_pose(model, root_id, int(model.site_bodyid[site_id])), pose(model.site_pos[site_id].tolist(), model.site_quat[site_id].tolist()))


def local_geom_pose(model: Any, geom_id: int, root_id: int) -> dict[str, list[float]]:
    return compose(local_body_child_pose(model, root_id, int(model.geom_bodyid[geom_id])), pose(model.geom_pos[geom_id].tolist(), model.geom_quat[geom_id].tolist()))


def subtree_articulated(model: Any, root_id: int) -> bool:
    bodies = {i for i in range(int(model.nbody)) if body_descendant_or_self(model, root_id, i)}
    return any(int(model.jnt_bodyid[j]) in bodies and int(model.jnt_type[j]) != 0 for j in range(int(model.njnt)))


def subtree_has_any_joint(model: Any, root_id: int) -> bool:
    bodies = {i for i in range(int(model.nbody)) if body_descendant_or_self(model, root_id, i)}
    return any(int(model.jnt_bodyid[j]) in bodies for j in range(int(model.njnt)))


def recorded_pose(state: Sequence[float], entry: Mapping[str, Any]) -> dict[str, list[float]]:
    start, end = int(entry["slice_start"]), int(entry["slice_end_exclusive"])
    if end - start != 14 or end > len(state): raise ReviewHold("object-state slice mismatch")
    xyzw = state[start+3:start+7]
    return pose(state[start:start+3], [xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def ancestor_chain(model: Any, body_id: int) -> list[dict[str, Any]]:
    out = []; current = int(body_id)
    while current >= 0:
        out.append({"body_id": current, "body_name": str(model.body(current).name or "")})
        if current == 0: break
        current = int(model.body_parentid[current])
    if not out or out[-1]["body_id"] != 0: raise ReviewHold("invalid ancestor chain")
    return list(reversed(out))


def validate_alias(entry: Mapping[str, Any], suite: str, task_id: int, resolution: Mapping[str, Any], index_entry: Mapping[str, Any], model: Any, bddl_sha: str, registry_sha: str, inventory_sha: str) -> None:
    gid = int(resolution.get("entity_id", -1)); geom_name = str(model.geom(gid).name or "")
    parent = int(model.geom_bodyid[gid])
    actual = {
        "suite": suite, "task_id": task_id, "bddl_object": str(resolution.get("name") or ""),
        "registry_resolution": str(resolution.get("resolution") or ""), "registry_geom_id": gid,
        "registry_geom_name": geom_name, "registry_geom_parent_body_id": parent,
        "registry_geom_parent_body_name": str(model.body(parent).name or ""),
        "index_map_object_index": int(index_entry["object_index"]), "index_map_body_id": int(index_entry["body_id"]),
        "index_map_body_name": str(index_entry["body_name"]), "object_state_slice_start": int(index_entry["slice_start"]),
        "object_state_slice_end_exclusive": int(index_entry["slice_end_exclusive"]), "object_state_slice_width": int(index_entry["slice_end_exclusive"])-int(index_entry["slice_start"]),
        "bddl_sha256": bddl_sha, "registry_task_sha256": registry_sha, "model_inventory_sha256": inventory_sha,
    }
    for key, value in actual.items():
        if entry.get(key) != value: raise ReviewHold(f"alias ledger mismatch: {key}")
    if entry.get("registry_geom_ancestor_chain") != ancestor_chain(model, parent): raise ReviewHold("registry alias ancestry mismatch")
    if entry.get("index_map_body_ancestor_chain") != ancestor_chain(model, int(index_entry["body_id"])): raise ReviewHold("index alias ancestry mismatch")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip(): rows.append(json.loads(raw))
    return rows


def source_files(record: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(record["source_episode_root"])).resolve()
    files = {str(x["name"]): x for x in record.get("source_files", [])}
    required = {"episode_metadata.json", "step_records.jsonl", "privileged_teacher_sidecar.jsonl"}
    if set(files) != required: raise ReviewHold("source file closure mismatch")
    paths = {name: root/name for name in required}
    for name, path in paths.items():
        spec = files[name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["size_bytes"] or sha256_file(path) != spec["sha256"]:
            raise ReviewHold(f"source file mismatch: {path}")
    return paths["episode_metadata.json"], paths["step_records.jsonl"], paths["privileged_teacher_sidecar.jsonl"]


def expected_object(model: Any, resolution: Mapping[str, Any], state: Sequence[float], objects: Mapping[str, Mapping[str, Any]], cache: dict, alias: Mapping[str, Any] | None, suite: str, task_id: int, bddl_sha: str, registry_sha: str, inventory_sha: str) -> dict[str, Any]:
    name = str(resolution.get("alias_from") or resolution.get("name") or "")
    entry = objects.get(name)
    if entry is None: return {"known": False, "source": "UNKNOWN_OBJECT_MAPPING"}
    body_id = int(entry["body_id"]); body_name = str(entry["body_name"])
    if int(model.body(body_name).id) != body_id: raise ReviewHold("index/model body mismatch")
    origin = recorded_pose(state, entry); kind = resolution.get("entity_type")
    if kind == "body":
        if int(resolution.get("entity_id", -1)) != body_id: raise ReviewHold("object body identity mismatch")
        local = pose([0,0,0], [1,0,0,0]); center, half = local_bounds(model, body_id, cache); source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL"; status = "INDEX_BODY_IDENTITY"
    elif kind == "geom":
        gid = int(resolution.get("entity_id", -1)); geom_body = int(model.geom_bodyid[gid]); geom_name = str(model.geom(gid).name or "")
        if geom_body == body_id:
            local = local_geom_pose(model, gid, body_id); center, half = geom_bounds(model, gid, cache); source = "RECORDED_OBJECT_STATE_FROZEN_GEOM_LOCAL"; status = "EXACT_GEOM_TO_INDEX_BODY"
        elif geom_body == 0 and geom_name == name:
            if alias is None: raise ReviewHold("unledgered alias in independent review")
            validate_alias(alias, suite, task_id, resolution, entry, model, bddl_sha, registry_sha, inventory_sha)
            local = pose([0,0,0], [1,0,0,0]); center, half = local_bounds(model, body_id, cache); source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL_INIT_GEOM_ALIAS"; status = "INIT_GEOM_ALIAS_TO_INDEX_BODY"
        else: raise ReviewHold("object geom identity mismatch")
    else: return {"known": False, "source": "UNKNOWN_OBJECT_MAPPING"}
    return {"known": True, "source": source, "pose": compose(origin, compose(local, pose(center, [1,0,0,0]))), "body_origin_pose": origin, "local_geometry_pose": local, "local_geometry_center": center, "half_extents": half, "registry_identity_status": status}


def expected_target(sim: Any, model: Any, resolution: Mapping[str, Any], state: Sequence[float], objects: Mapping[str, Mapping[str, Any]], roots: Mapping[str, int], cache: dict) -> dict[str, Any]:
    name = str(resolution.get("alias_to") or resolution.get("name") or ""); kind = resolution.get("entity_type")
    if kind == "site":
        sid = int(model.site(name).id)
        if sid != int(resolution.get("entity_id", -1)): raise ReviewHold("target site identity mismatch")
        parent = int(model.site_bodyid[sid])
        if parent != int(resolution.get("parent_body_id", -1)) or str(model.body(parent).name) != str(resolution.get("parent_body_name")): raise ReviewHold("target site parent mismatch")
        dynamic = None; current = parent
        while current > 0:
            for object_name, root_id in roots.items():
                if current == root_id: dynamic = (object_name, root_id); break
            if dynamic: break
            current = int(model.body_parentid[current])
        if dynamic and subtree_articulated(model, dynamic[1]): return {"known": False, "source": "ARTICULATED_UNKNOWN"}
        if dynamic:
            object_name, root_id = dynamic; local = local_site_pose(model, sid, root_id); target_pose = compose(recorded_pose(state, objects[object_name]), local); source = "RECORDED_OBJECT_STATE_FROZEN_SITE_LOCAL"; parent_origin = recorded_pose(state, objects[object_name])
        else:
            if body_chain_has_joint(model, parent, include_free=True):
                return {"known": False, "source": "UNKNOWN_UNOBSERVED_JOINTED"}
            target_pose = fixed_site_pose(model, sid); source = "MODEL_FIXED_CHAIN"; local = None; parent_origin = None
        return {"known": True, "source": source, "pose": target_pose, "half_extents": [float(x) for x in model.site_size[sid].tolist()], "local_geometry_pose": local, "parent_body_origin_pose": parent_origin}
    if kind == "body":
        bid = int(model.body(name).id)
        if bid != int(resolution.get("entity_id", -1)): raise ReviewHold("target body identity mismatch")
        if subtree_articulated(model, bid) or body_chain_has_joint(model, bid, include_free=False): return {"known": False, "source": "ARTICULATED_UNKNOWN"}
        center, half = local_bounds(model, bid, cache); bddl_name = str(resolution.get("alias_from") or name.removesuffix("_main"))
        if bddl_name in objects:
            origin = recorded_pose(state, objects[bddl_name]); source = "RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL"
        else:
            if subtree_has_any_joint(model, bid) or body_chain_has_joint(model, bid, include_free=True):
                return {"known": False, "source": "UNKNOWN_UNOBSERVED_JOINTED"}
            origin = fixed_body_pose(model, bid); source = "MODEL_FIXED_CHAIN"
        return {"known": True, "source": source, "pose": compose(origin, pose(center, [1,0,0,0])), "half_extents": half, "body_origin_pose": origin, "local_geometry_center": center}
    return {"known": False, "source": "UNKNOWN_TARGET_RESOLUTION"}


def joint_chain(model: Any, ancestor_ids: Sequence[int]) -> list[dict[str, Any]]:
    allowed = {int(x) for x in ancestor_ids}
    return [
        {
            "joint_id": int(joint_id),
            "joint_name": str(model.joint(joint_id).name or ""),
            "body_id": int(model.jnt_bodyid[joint_id]),
            "joint_type": int(model.jnt_type[joint_id]),
        }
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) in allowed
    ]


def entity_hierarchy(model: Any, resolution: Mapping[str, Any], index_entry: Mapping[str, Any] | None) -> dict[str, Any]:
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    if kind == "site":
        body_id = int(model.site_bodyid[entity_id])
    elif kind == "geom":
        body_id = int(model.geom_bodyid[entity_id])
    elif kind == "body":
        body_id = entity_id
    elif index_entry is not None:
        body_id = int(index_entry.get("body_id", -1))
    else:
        return {"parent_root_body": None, "ancestor_chain": [], "joint_chain": []}
    ancestors = ancestor_chain(model, body_id)
    ids = [int(x["body_id"]) for x in ancestors]
    return {
        "parent_root_body": {
            "parent_body_id": int(model.body_parentid[body_id]),
            "parent_body_name": str(model.body(int(model.body_parentid[body_id])).name or "") if body_id else None,
            "root_body_id": ids[0],
            "root_body_name": ancestors[0]["body_name"],
            "entity_body_id": body_id,
            "entity_body_name": str(model.body(body_id).name or ""),
        },
        "ancestor_chain": ancestors,
        "joint_chain": joint_chain(model, ids),
    }


def diagnostic_row(
    context: Mapping[str, Any],
    comparison_kind: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    actual_pose: Mapping[str, Any],
    expected_pose: Mapping[str, Any],
) -> dict[str, Any]:
    position_error = [
        float(actual_pose["pos"][i]) - float(expected_pose["pos"][i])
        for i in range(3)
    ]
    position_linf = max(abs(x) for x in position_error)
    position_l2 = math.sqrt(sum(x * x for x in position_error))
    return {
        **dict(context),
        "comparison_kind": comparison_kind,
        "source": actual.get("source"),
        "expected_source": expected.get("source"),
        "actual_pose": actual_pose,
        "expected_pose": expected_pose,
        "position_error_vector": position_error,
        "position_error_l_inf_m": position_linf,
        "position_error_l2_m": position_l2,
        "rotation_error_rad": quat_distance(actual_pose["quat"], expected_pose["quat"]),
        "local_center": {
            "actual": actual.get("local_geometry_center"),
            "expected": expected.get("local_geometry_center"),
        },
        "extents": {
            "actual": actual.get("half_extents"),
            "expected": expected.get("half_extents"),
        },
        "actual_geometry_entity_type": actual.get("geometry_entity_type"),
        "alias_status": actual.get("registry_identity_status"),
    }


def pose_diff(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    metrics: dict[str, float],
    context: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    if actual.get("known") != expected.get("known") or actual.get("source") != expected.get("source"):
        raise ReviewHold("known/source mismatch")
    if not expected.get("known"): return
    ap, ep = actual["pose"], expected["pose"]
    metrics["geometry_position_max_error_m"] = max(metrics["geometry_position_max_error_m"], max(abs(float(a)-float(b)) for a,b in zip(ap["pos"], ep["pos"])))
    metrics["geometry_rotation_max_error_rad"] = max(metrics["geometry_rotation_max_error_rad"], quat_distance(ap["quat"], ep["quat"]))
    diagnostics.append(diagnostic_row(context, "geometry", actual, expected, ap, ep))
    for key in ("body_origin_pose", "parent_body_origin_pose"):
        if key in expected and expected[key] is not None:
            if key not in actual: raise ReviewHold(f"missing {key}")
            body_actual, body_expected = actual[key], expected[key]
            metrics["body_origin_position_max_error_m"] = max(metrics["body_origin_position_max_error_m"], max(abs(float(a)-float(b)) for a,b in zip(body_actual["pos"], body_expected["pos"])))
            metrics["body_origin_rotation_max_error_rad"] = max(metrics["body_origin_rotation_max_error_rad"], quat_distance(body_actual["quat"], body_expected["quat"]))
            diagnostics.append(diagnostic_row(context, key, actual, expected, body_actual, body_expected))
    if "half_extents" in expected:
        if "half_extents" not in actual: raise ReviewHold("missing extents")
        metrics["extent_max_error_m"] = max(metrics["extent_max_error_m"], max(abs(float(a)-float(b)) for a,b in zip(actual["half_extents"], expected["half_extents"])))
    if "local_geometry_center" in expected and expected["local_geometry_center"] is not None:
        if actual.get("local_geometry_center") is None: raise ReviewHold("missing local geometry center")
        metrics["local_center_max_error_m"] = max(metrics["local_center_max_error_m"], max(abs(float(a)-float(b)) for a,b in zip(actual["local_geometry_center"], expected["local_geometry_center"])))


def review_run(run_root: Path, pilot: Mapping[str, Any], index_payload: Mapping[str, Any], registry_root: Path, libero_root: Path, alias_ledger: Mapping[str, Mapping[str, Any]], expected_commit: str) -> dict[str, Any]:
    seal_sha = verify_seal(run_root); manifest = json.loads((run_root/"MANIFEST.json").read_text())
    if manifest.get("code_snapshot_commit") != expected_commit or manifest.get("consumer_eligible") is not False: raise ReviewHold("run manifest binding mismatch")
    import libero.libero.benchmark as benchmark_module
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    if subprocess.check_output(["git", "-C", str(libero_root), "show", "-s", "--format=%H", "HEAD"], text=True).strip() != LIBERO_HEAD or subprocess.check_output(["git", "-C", str(libero_root), "show", "-s", "--format=%T", "HEAD"], text=True).strip() != LIBERO_TREE: raise ReviewHold("LIBERO snapshot mismatch")
    if Path(get_libero_path("bddl_files")).resolve() != (libero_root/"libero/libero/bddl_files").resolve(): raise ReviewHold("LIBERO BDDL root mismatch")
    metrics = {"body_origin_position_max_error_m": 0.0, "body_origin_rotation_max_error_rad": 0.0, "geometry_position_max_error_m": 0.0, "geometry_rotation_max_error_rad": 0.0, "extent_max_error_m": 0.0, "local_center_max_error_m": 0.0}
    diagnostics: list[dict[str, Any]] = []
    episode_count=step_count=relation_rows=alias_rows=unknown_rows=0; errors=[]; task_cache={}
    records = {str(r["episode_id"]): r for r in pilot["records"]}
    for episode_dir in sorted((run_root/"episodes").iterdir()):
        em = json.loads((episode_dir/"episode_manifest.json").read_text()); eid=em["episode_id"]
        if eid not in records: raise ReviewHold(f"episode not in pilot: {eid}")
        record=records[eid]; _, _, sidecar_path=source_files(record); sidecars=read_jsonl(sidecar_path)
        suite=str(record["suite"]); task_id=int(record["task_id"]); key=f"{suite}/task_{task_id:02d}"
        if key not in task_cache:
            task=benchmark_module.get_benchmark(suite)(0).get_task(task_id); bddl=(libero_root/"libero/libero/bddl_files"/task.problem_folder/task.bddl_file).resolve(); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=64,camera_widths=64,render_gpu_device_id=-1,has_renderer=False,has_offscreen_renderer=False,horizon=2); env.reset(); task_cache[key]=(env,bddl)
        env,bddl=task_cache[key]; model=env.env.sim.model; sim=env.env.sim; registry_path=registry_root/"run_A"/"per_task"/(key.replace("/","_")+".json"); registry=json.loads(registry_path.read_text())["legacy"]; registry_sha=sha256_file(registry_path); bddl_sha=sha256_file(bddl); inventory_sha=model_inventory_sha(model)
        objects={str(x["object_name"]):x for x in next(t for t in index_payload["tasks"] if t["suite"]==suite and int(t["task_id"])==task_id)["objects"]}; roots={str(x["object_name"]):int(x["body_id"]) for x in objects.values()}; cache={}
        output_rows=read_jsonl(episode_dir/"geometry_cases.jsonl")
        if [int(x["step"]) for x in output_rows] != list(range(len(output_rows))) or len(output_rows)!=len(sidecars): raise ReviewHold(f"step closure mismatch: {eid}")
        for output_row, sidecar in zip(output_rows, sidecars):
            state=[float(x) for x in sidecar.get("object_state", [])]
            expected_relations=registry.get("relations",[])
            if len(output_row.get("relations",[])) != len(expected_relations): raise ReviewHold(f"relation closure mismatch: {eid}:{output_row.get('step')}")
            for relation_index, (out_rel, relation) in enumerate(zip(output_row["relations"], expected_relations)):
                obj_resolution = relation["object_resolution"]
                target_resolution = relation["target_resolution"]
                obj_name=str(obj_resolution.get("alias_from") or obj_resolution.get("name") or ""); alias=alias_ledger.get(f"{suite}/task_{task_id:02d}/{obj_name}")
                obj=expected_object(model, relation["object_resolution"], state, objects, cache, alias, suite, task_id, bddl_sha, registry_sha, inventory_sha)
                target=expected_target(sim, model, relation["target_resolution"], state, objects, roots, cache)
                for side, actual, expected, resolution, index_entry in (
                    ("object", out_rel["object"], obj, obj_resolution, objects.get(obj_name)),
                    ("target", out_rel["target"], target, target_resolution, None),
                ):
                    hierarchy = entity_hierarchy(model, resolution, index_entry)
                    context = {
                        "episode_id": eid,
                        "suite": suite,
                        "task_id": task_id,
                        "state_id": int(record["state_id"]),
                        "step": int(output_row["step"]),
                        "relation_index": relation_index,
                        "predicate": relation.get("predicate"),
                        "side": side,
                        "entity_id": resolution.get("entity_id"),
                        "entity_name": resolution.get("alias_from") or resolution.get("alias_to") or resolution.get("name"),
                        "entity_type": resolution.get("entity_type"),
                        "parent_root_body": hierarchy["parent_root_body"],
                        "ancestor_chain": hierarchy["ancestor_chain"],
                        "joint_chain": hierarchy["joint_chain"],
                    }
                    pose_diff(actual, expected, metrics, context, diagnostics)
                relation_rows+=1
                alias_rows += int(out_rel["object"].get("registry_identity_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY"); unknown_rows += int(not out_rel.get("known", False))
            step_count += 1
        episode_count += 1
    for env, _ in task_cache.values(): env.close()
    passed = not errors and episode_count == 40 and step_count == 9422 and relation_rows == 11880 and alias_rows == 217 and unknown_rows == 0 and all(value <= limit for value, limit in ((metrics["body_origin_position_max_error_m"],1e-8),(metrics["body_origin_rotation_max_error_rad"],BODY_ROT_TOL),(metrics["geometry_position_max_error_m"],POS_TOL),(metrics["geometry_rotation_max_error_rad"],ROT_TOL),(metrics["extent_max_error_m"],EXTENT_TOL)))
    ordered = sorted(diagnostics, key=lambda row: (row["position_error_l2_m"], row["position_error_l_inf_m"], row["rotation_error_rad"]), reverse=True)
    source_counts: dict[str, int] = {}
    high_source_counts: dict[str, int] = {}
    high_entity_counts: dict[str, int] = {}
    high_side_counts: dict[str, int] = {}
    for row in diagnostics:
        source = str(row.get("source") or "<missing>")
        source_counts[source] = source_counts.get(source, 0) + 1
        if row["position_error_l_inf_m"] > 1e-6:
            high_source_counts[source] = high_source_counts.get(source, 0) + 1
            entity_key = "|".join(str(row.get(key) or "<missing>") for key in ("side", "predicate", "entity_name", "entity_type"))
            high_entity_counts[entity_key] = high_entity_counts.get(entity_key, 0) + 1
            side = str(row.get("side") or "<missing>")
            high_side_counts[side] = high_side_counts.get(side, 0) + 1
    difference_summary = {
        "difference_rows_total": len(diagnostics),
        "position_l_inf_gt_1e-6_m": sum(row["position_error_l_inf_m"] > 1e-6 for row in diagnostics),
        "position_l_inf_gt_1e-4_m": sum(row["position_error_l_inf_m"] > 1e-4 for row in diagnostics),
        "rotation_gt_1e-8_rad": sum(row["rotation_error_rad"] > 1e-8 for row in diagnostics),
        "source_counts": source_counts,
        "high_position_error_source_counts": dict(sorted(high_source_counts.items())),
        "high_position_error_side_counts": dict(sorted(high_side_counts.items())),
        "high_position_error_entity_counts": dict(sorted(high_entity_counts.items())),
        "high_position_error_alias_rows": sum(row.get("alias_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY" and row["position_error_l_inf_m"] > 1e-6 for row in diagnostics),
        "alias_rows_total": sum(row.get("alias_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY" for row in diagnostics),
        "alias_rows_in_top_100": sum(row.get("alias_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY" for row in ordered[:100]),
        "top_100_difference_rows": ordered[:100],
    }
    return {"status": "PASS" if passed else "HOLD", "run_root": str(run_root), "run_sha256s_sha256": seal_sha, "episodes": episode_count, "steps": step_count, "relation_rows": relation_rows, "alias_rows": alias_rows, "supported_unknown_rows": unknown_rows, "metrics": metrics, "thresholds": {"body_origin_position_m":1e-8,"body_origin_rotation_rad":BODY_ROT_TOL,"geometry_position_m":POS_TOL,"geometry_rotation_rad":ROT_TOL,"extent_m":EXTENT_TOL}, "difference_summary": difference_summary, "errors": errors}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--run-root", type=Path, required=True); ap.add_argument("--pilot-manifest", type=Path, required=True); ap.add_argument("--index-root", type=Path, required=True); ap.add_argument("--registry-root", type=Path, required=True); ap.add_argument("--libero-root", type=Path, required=True); ap.add_argument("--alias-ledger", type=Path, required=True); ap.add_argument("--expected-commit", required=True); ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    try:
        pilot=json.loads(args.pilot_manifest.read_text()); index_payload=json.loads((args.index_root/"OBJECT_STATE_INDEX_MAP_V1.json").read_text()); ledger=json.loads(args.alias_ledger.read_text()); aliases={f"{x['suite']}/task_{int(x['task_id']):02d}/{x['bddl_object']}":x for x in ledger["entries"]}; result=review_run(args.run_root,pilot,index_payload,args.registry_root,args.libero_root,aliases,args.expected_commit)
    except Exception as exc:
        result={"status":"HOLD","error_type":type(exc).__name__,"error":str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True)); return 0 if result.get("status")=="PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
