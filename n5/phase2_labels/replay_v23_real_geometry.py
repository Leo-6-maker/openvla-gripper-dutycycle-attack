"""Deterministic no-model replay for the frozen 40-episode V23 pilot.

The only runtime producer here is LIBERO/MuJoCo.  OpenVLA, Teacher labels and
task outcome fields are deliberately absent from the output.  This is the
smallest real-input bridge needed by the V23 runner; it is not a relabeler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
NUM_STEPS_WAIT = 10
FORBIDDEN_TOKENS = ("cal", "check", "g10", "t2r")


class GeometryHold(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reject_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        raise GeometryHold(f"forbidden/protected path: {path}")


def verify_sealed_root(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise GeometryHold(f"sealed root incomplete: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if len(side) != 2 or side[1] != "SHA256SUMS" or side[0] != sha256_file(sums):
        raise GeometryHold(f"root sidecar mismatch: {root}")
    expected: dict[str, str] = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(None, 1)
        name = name.lstrip("*").strip()
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise GeometryHold(f"unsafe checksum path: {name}")
        target = root / rel
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise GeometryHold(f"sealed file mismatch: {target}")
        expected[rel.as_posix()] = digest
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if set(expected) != actual:
        raise GeometryHold(f"sealed file closure mismatch: {root}")
    return {"root": str(root.resolve()), "sha256sums_sha256": sha256_file(sums), "files": expected}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise GeometryHold(f"non-object JSONL row: {path}:{line_no}")
            rows.append(row)
    return rows


def verify_source_files(record: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(record["source_episode_root"])).resolve()
    reject_path(root)
    files = {str(item["name"]): item for item in record.get("source_files", []) if isinstance(item, Mapping)}
    required = {"episode_metadata.json", "step_records.jsonl", "privileged_teacher_sidecar.jsonl"}
    if set(files) != required:
        raise GeometryHold(f"pilot source file closure mismatch: {record.get('episode_id')}")
    paths = {name: root / name for name in required}
    for name, path in paths.items():
        spec = files[name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != spec.get("size_bytes") or sha256_file(path) != spec.get("sha256"):
            raise GeometryHold(f"pilot source SHA mismatch: {path}")
    return paths["episode_metadata.json"], paths["step_records.jsonl"], paths["privileged_teacher_sidecar.jsonl"]


def qnorm(q: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in q)
    norm = math.sqrt(sum(x * x for x in values))
    if len(values) != 4 or not math.isfinite(norm) or norm <= 0:
        raise GeometryHold("non-finite quaternion")
    return tuple(x / norm for x in values)  # type: ignore[return-value]


def qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = qnorm(a); w2, x2, y2, z2 = qnorm(b)
    return qnorm((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                  w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2))


def qrotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = qnorm(q); vx, vy, vz = (float(x) for x in v)
    tx = 2 * (y*vz-z*vy); ty = 2 * (z*vx-x*vz); tz = 2 * (x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def mat_to_quat(m: Sequence[float]) -> tuple[float, float, float, float]:
    if len(m) != 9 or not all(math.isfinite(float(x)) for x in m):
        raise GeometryHold("non-finite rotation matrix")
    a00,a01,a02,a10,a11,a12,a20,a21,a22 = (float(x) for x in m)
    trace = a00 + a11 + a22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25*s, (a21-a12)/s, (a02-a20)/s, (a10-a01)/s)
    elif a00 > a11 and a00 > a22:
        s = math.sqrt(1.0+a00-a11-a22)*2
        q = ((a21-a12)/s, 0.25*s, (a01+a10)/s, (a02+a20)/s)
    elif a11 > a22:
        s = math.sqrt(1.0+a11-a00-a22)*2
        q = ((a02-a20)/s, (a01+a10)/s, 0.25*s, (a12+a21)/s)
    else:
        s = math.sqrt(1.0+a22-a00-a11)*2
        q = ((a10-a01)/s, (a02+a20)/s, (a12+a21)/s, 0.25*s)
    return qnorm(q)


def pose(pos: Sequence[float], quat: Sequence[float]) -> dict[str, list[float]]:
    if len(pos) != 3 or not all(math.isfinite(float(x)) for x in pos):
        raise GeometryHold("non-finite position")
    q = qnorm(quat)
    return {"pos": [float(x) for x in pos], "quat": list(q)}


def compose(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    rotated = qrotate(a["quat"], b["pos"])
    return {"pos": [float(a["pos"][i]) + rotated[i] for i in range(3)], "quat": list(qmul(a["quat"], b["quat"]))}


def body_path(model: Any, root_id: int, child_id: int) -> list[int] | None:
    path = []
    current = child_id
    while current != root_id:
        path.append(current)
        current = int(model.body_parentid[current])
        if current <= 0 or len(path) > int(model.nbody):
            return None
    return list(reversed(path))


def shape_corners(kind: int, size: Sequence[float]) -> list[tuple[float, float, float]]:
    s = [abs(float(x)) for x in size]
    if kind == 2: half = [s[0], s[0], s[0]]
    elif kind == 3: half = [s[0], s[0], s[0] + s[1]]
    elif kind == 4: half = s[:3]
    elif kind == 5: half = [s[0], s[0], s[1]]
    elif kind == 6: half = s[:3]
    else: raise GeometryHold(f"unsupported object geom type: {kind}")
    if len(half) != 3 or any(not math.isfinite(x) or x <= 0 for x in half):
        raise GeometryHold("degenerate object geometry")
    return [(sx*half[0], sy*half[1], sz*half[2]) for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]


def body_local_bounds(model: Any, body_id: int) -> tuple[list[float], list[float]]:
    ids = [i for i in range(int(model.nbody)) if i == body_id or body_path(model, body_id, i) is not None]
    corners: list[tuple[float, float, float]] = []
    for gid in range(int(model.ngeom)):
        geom_body = int(model.geom_bodyid[gid])
        if geom_body not in ids:
            continue
        local = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
        path = body_path(model, body_id, geom_body)
        if path is None:
            continue
        for bid in path:
            local = compose(local, {"pos": model.body_pos[bid].tolist(), "quat": model.body_quat[bid].tolist()})
        local = compose(local, {"pos": model.geom_pos[gid].tolist(), "quat": model.geom_quat[gid].tolist()})
        for corner in shape_corners(int(model.geom_type[gid]), model.geom_size[gid].tolist()):
            corners.append(tuple(local["pos"][i] + qrotate(local["quat"], corner)[i] for i in range(3)))
    if not corners:
        raise GeometryHold(f"body has no supported geoms: {body_id}")
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    half = [(hi[i]-lo[i])/2 for i in range(3)]
    center = [(hi[i]+lo[i])/2 for i in range(3)]
    if any(x <= 0 or not math.isfinite(x) for x in half):
        raise GeometryHold("invalid body bounds")
    return center, half


def calibrated_qpos_threshold(model: Any) -> dict[str, Any]:
    ranges = []
    joints = []
    for jid in range(int(model.njnt)):
        name = str(model.joint(jid).name or "")
        if "finger" not in name:
            continue
        span = float(model.jnt_range[jid][1] - model.jnt_range[jid][0])
        if span <= 0 or not math.isfinite(span):
            raise GeometryHold(f"invalid gripper joint range: {name}")
        ranges.append(span); joints.append({"name": name, "id": jid, "range": model.jnt_range[jid].tolist()})
    if len(ranges) != 2:
        raise GeometryHold(f"expected two calibrated gripper joints, found {len(ranges)}")
    return {"formula": "0.5 * mean(abs(joint_range_span))", "value": 0.5 * sum(ranges) / len(ranges), "joints": joints}


def load_relations(registry_root: Path, task_key: str) -> tuple[dict[str, Any], str]:
    file = registry_root / "run_A" / "per_task" / (task_key.replace("/", "_") + ".json")
    if not file.is_file():
        raise GeometryHold(f"C1 per-task registry missing: {task_key}")
    data = strict_json(file)
    if data.get("task_key") != task_key or data.get("legacy", {}).get("status") != "OK":
        raise GeometryHold(f"C1 task binding failed: {task_key}")
    return data["legacy"], sha256_file(file)


def target_geometry(sim: Any, model: Any, resolution: Mapping[str, Any]) -> tuple[dict[str, list[float]], list[float], str]:
    name = resolution.get("alias_to") or resolution.get("name")
    kind = resolution.get("entity_type")
    expected_id = int(resolution.get("entity_id", -1))
    if kind == "site":
        sid = model.site(name).id
        if sid != expected_id:
            raise GeometryHold(f"site identity mismatch: {name}")
        return pose(sim.data.site_xpos[sid].tolist(), mat_to_quat(sim.data.site_xmat[sid].tolist())), [float(x) for x in resolution.get("size", [])], "site"
    if kind == "body":
        bid = model.body(name).id
        if bid != expected_id:
            raise GeometryHold(f"body identity mismatch: {name}")
        center, half = body_local_bounds(model, bid)
        body_pose = pose(sim.data.body_xpos[bid].tolist(), sim.data.body_xquat[bid].tolist())
        return compose(body_pose, {"pos": center, "quat": [1,0,0,0]}), half, "body"
    raise GeometryHold(f"unsupported target entity: {kind}")


def build_case(sim: Any, model: Any, relation: Mapping[str, Any], episode_id: str, step: int) -> dict[str, Any]:
    obj = relation["object_resolution"]
    name = obj.get("alias_to") or obj.get("name")
    bid = model.body(name).id
    if bid != int(obj.get("entity_id", -1)):
        raise GeometryHold(f"object identity mismatch: {name}")
    center, half = body_local_bounds(model, bid)
    obj_pose = compose(pose(sim.data.body_xpos[bid].tolist(), sim.data.body_xquat[bid].tolist()), {"pos": center, "quat": [1,0,0,0]})
    target_pose, target_half, target_kind = target_geometry(sim, model, relation["target_resolution"])
    if len(target_half) != 3 or any(float(x) <= 0 for x in target_half):
        raise GeometryHold(f"target extents invalid: {episode_id}:{step}")
    object_id = f"{episode_id}#relation_{relation['relation_index']}#object"
    target_id = f"{episode_id}#relation_{relation['relation_index']}#target"
    return {
        "episode_id": episode_id, "step": step, "predicate": relation["predicate"],
        "relation_index": int(relation["relation_index"]),
        "expected_identity": {"episode_id": episode_id, "step": step, "object_id": object_id, "target_id": target_id},
        "object": {"id": object_id, "role": "MANIPULATED_OBJECT", "pose": obj_pose, "half_extents": half, "source": "DIRECT_SIM_STATE"},
        "target": {"id": target_id, "role": "REGION_TARGET" if relation.get("target_is_region") else "OBJECT_TARGET", "pose": target_pose, "half_extents": target_half, "source": "DIRECT_SIM_STATE"},
        "geometry_source": "DIRECT_SIM_STATE",
        "target_entity_kind": target_kind,
    }


def replay_episode(record: Mapping[str, Any], registry_root: Path, benchmark_cache: dict[str, Any]) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = str(record["suite"]); task_idx = int(record["task_id"]); state_id = int(record["state_id"]); episode_id = str(record["episode_id"])
    if suite not in HORIZONS or not (0 < int(record.get("observed_step_count", 0)) <= HORIZONS[suite]):
        raise GeometryHold(f"pilot horizon mismatch: {episode_id}")
    metadata_path, steps_path, sidecar_path = verify_source_files(record)
    meta = strict_json(metadata_path); steps = read_jsonl(steps_path); sidecar = read_jsonl(sidecar_path)
    if len(steps) != len(sidecar) or [r.get("step") for r in steps] != list(range(len(steps))):
        raise GeometryHold(f"source step closure failed: {episode_id}")
    if meta.get("official_horizon") != HORIZONS[suite] or not isinstance(meta.get("official_seed"), int):
        raise GeometryHold(f"official runtime binding missing: {episode_id}")
    task = benchmark_cache.setdefault(suite, __import__("libero.libero.benchmark", fromlist=["get_benchmark"]).get_benchmark(suite)(0)).get_task(task_idx)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    legacy, registry_sha = load_relations(registry_root, f"{suite}/task_{task_idx:02d}")
    if sha256_file(bddl) != legacy.get("bddl_sha256"):
        raise GeometryHold(f"BDDL SHA mismatch: {episode_id}")
    from libero.libero.benchmark import get_benchmark
    state = get_benchmark(suite)(0).get_task_init_states(task_idx)[state_id]
    if sha256_bytes(pickle.dumps(state, protocol=4)) != meta.get("initial_state_sha256"):
        raise GeometryHold(f"initial state SHA mismatch: {episode_id}")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
                             render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=1000)
    env.seed(int(meta["official_seed"])); env.reset(); obs = env.set_init_state(pickle.loads(pickle.dumps(state, protocol=4)))
    for _ in range(NUM_STEPS_WAIT): obs = env.step([0,0,0,0,0,0,-1])[0]
    model = env.sim.model; site_id = model.site_name2id("gripper0_grip_site")
    qpos_binding = calibrated_qpos_threshold(model)
    relation_rows = []
    qpos_errors = []; eef_errors = []; all_cases = []
    try:
        relations = []
        for idx, rel in enumerate(legacy.get("relations", [])):
            item = dict(rel); item["relation_index"] = idx; relations.append(item)
        if not relations:
            raise GeometryHold(f"no C1 relation for {episode_id}")
        for step_row, side_row in zip(steps, sidecar):
            step = int(step_row["step"])
            qpos_errors.append(float(np.max(np.abs(np.asarray(side_row["robot0_gripper_qpos"], float)-np.asarray(obs["robot0_gripper_qpos"], float)))))
            eef_errors.append(float(np.max(np.abs(np.asarray(side_row["eef_feature_pos"], float)-np.asarray(env.sim.data.site_xpos[site_id], float)))))
            all_cases.append({"episode_id": episode_id, "step": step, "relations": [build_case(env.sim, model, rel, episode_id, step) for rel in relations]})
            obs, _reward, done, _info = env.step([float(x) for x in step_row["applied_action_7d"]])
            if done and step + 1 < len(steps):
                raise GeometryHold(f"replay terminated before source horizon: {episode_id}:{step}")
    finally:
        env.close()
    return {
        "episode_id": episode_id, "suite": suite, "task_idx": task_idx, "state_id": state_id,
        "step_count": len(all_cases), "geometry_cases": all_cases, "relation_count": len(relations),
        "qpos_close_threshold": qpos_binding["value"], "qpos_calibration": qpos_binding,
        "qpos_sidecar_max_abs_error": max(qpos_errors, default=None), "eef_sidecar_max_abs_error": max(eef_errors, default=None),
        "registry_task_sha256": registry_sha, "source_mode": "DETERMINISTIC_LIBERO_MUJOCO_REPLAY",
        "model_inference": False, "teacher_labeling": False, "attack": False,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def seal_staging(staging: Path) -> dict[str, Any]:
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    lines = [f"{sha256_file(staging / rel)}  {rel}" for rel in files]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": len(files)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pilot_root = Path(args.pilot_root).resolve(); registry_root = Path(args.registry_root).resolve(); d0_root = Path(args.dev_pool_root).resolve()
    for root in (pilot_root, registry_root, d0_root): reject_path(root)
    pilot_seal = verify_sealed_root(pilot_root); registry_seal = verify_sealed_root(registry_root); d0_seal = verify_sealed_root(d0_root)
    pilot_manifest_path = pilot_root / "PILOT_INPUT_MANIFEST.json"; pilot_manifest = strict_json(pilot_manifest_path)
    if pilot_manifest.get("schema") != "V23_DEV_PILOT_V1" or pilot_manifest.get("episode_count") != 40: raise GeometryHold("pilot manifest closure failed")
    pool = {row["episode_id"] for row in csv.DictReader((d0_root / "DEV_POOL_IDENTITY_MANIFEST.csv").open(encoding="utf-8", newline=""))}
    records = pilot_manifest.get("records", [])
    ids = {str(row.get("episode_id")) for row in records}
    if len(ids) != 40 or not ids <= pool: raise GeometryHold("pilot identities are not proven inside DEV_POOL")
    benchmark_cache: dict[str, Any] = {}
    episodes = [replay_episode(row, registry_root, benchmark_cache) for row in sorted(records, key=lambda x: x["episode_id"])]
    if len(episodes) != 40 or {x["episode_id"] for x in episodes} != ids: raise GeometryHold("40-episode replay closure failed")
    source_binding = {"pilot_manifest_sha256": sha256_file(pilot_manifest_path), "pilot_root_sha256s_sha256": pilot_seal["sha256sums_sha256"], "registry_root_sha256s_sha256": registry_seal["sha256sums_sha256"], "dev_pool_root_sha256s_sha256": d0_seal["sha256sums_sha256"], "source_mode": "DETERMINISTIC_LIBERO_MUJOCO_REPLAY", "model_inference": False, "teacher_labeling": False, "attack": False}
    parent = Path(args.output_parent).resolve(); parent.mkdir(parents=True, exist_ok=True); final = parent / args.output_name
    if final.exists(): raise GeometryHold(f"output exists: {final}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"; staging.mkdir()
    try:
        for ep in episodes:
            safe = ep["episode_id"].replace("/", "__")
            directory = staging / "episodes" / safe; directory.mkdir(parents=True)
            write_json(directory / "episode_manifest.json", {k: ep[k] for k in ("episode_id","suite","task_idx","state_id","step_count","relation_count","qpos_close_threshold","qpos_calibration","registry_task_sha256","source_mode")})
            with (directory / "geometry_cases.jsonl").open("w", encoding="utf-8") as f:
                for row in ep["geometry_cases"]: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        manifest = {"schema": "C3_T1_PILOT_GEOMETRY_ROOT_V1", "status": "FROZEN_DETERMINISTIC_REPLAY", "episode_count": 40, "episodes": [{k: ep[k] for k in ("episode_id","suite","task_idx","state_id","step_count","relation_count","qpos_close_threshold","qpos_calibration","source_mode")} for ep in episodes], "source_binding": source_binding, "protected_payload_read": False, "model_inference": False, "teacher_labeling": False, "attack": False}
        write_json(staging / "dataset_manifest.json", manifest); write_json(staging / "source_binding.json", source_binding); write_json(staging / "runtime_audit.json", {"status":"PASS", "episodes":40, "qpos_sidecar_max_abs_error": max(ep["qpos_sidecar_max_abs_error"] for ep in episodes), "eef_sidecar_max_abs_error": max(ep["eef_sidecar_max_abs_error"] for ep in episodes), "protected_payload_read":False, "model_inference":False, "teacher_labeling":False, "attack":False})
        seal = seal_staging(staging); os.rename(staging, final)
        return {"root": str(final), "status":"PASS", **seal, "episodes":40}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--pilot-root", required=True); parser.add_argument("--registry-root", required=True); parser.add_argument("--dev-pool-root", required=True); parser.add_argument("--output-parent", required=True); parser.add_argument("--output-name", required=True)
    try: print(json.dumps(run(parser.parse_args()), sort_keys=True))
    except Exception as exc: print(json.dumps({"status":"HOLD_GEOMETRY_SOURCE_INSUFFICIENT", "reason":f"{type(exc).__name__}:{exc}", "model_inference":False, "teacher_labeling":False, "attack":False}, sort_keys=True)); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
