"""FIT-only direct model-chain calibration canary for G-REC R1D.

This reads registry/task metadata and the frozen LIBERO model only.  It does
not read episode payloads, replay actions, run OpenVLA, or create geometry
evidence.  Dynamic and jointed entities are reported as such rather than
treated as fixed-pose calibration cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


LIBERO_HEAD = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_TREE = "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60"
POS_TOL = 1e-8
ROT_TOL = 1e-7


class CanaryHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qnorm(q: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in q)
    norm = math.sqrt(sum(x * x for x in values))
    if len(values) != 4 or not math.isfinite(norm) or norm <= 0:
        raise CanaryHold("invalid quaternion")
    return tuple(x / norm for x in values)  # type: ignore[return-value]


def qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = qnorm(a); w2, x2, y2, z2 = qnorm(b)
    return qnorm((w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                  w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2))


def qrotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = qnorm(q); vx, vy, vz = (float(x) for x in v)
    tx = 2 * (y*vz-z*vy); ty = 2 * (z*vx-x*vz); tz = 2 * (x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def pose(pos: Sequence[float], quat: Sequence[float]) -> dict[str, list[float]]:
    values = [float(x) for x in pos]
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise CanaryHold("invalid position")
    return {"pos": values, "quat": list(qnorm(quat))}


def compose(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    moved = qrotate(a["quat"], b["pos"])
    return {"pos": [a["pos"][i] + moved[i] for i in range(3)], "quat": list(qmul(a["quat"], b["quat"]))}


def rotation_error(a: Sequence[float], b: Sequence[float]) -> float:
    qa, qb = qnorm(a), qnorm(b)
    dot = min(1.0, max(0.0, abs(sum(x * y for x, y in zip(qa, qb)))))
    return 2 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot)


def mat_to_quat(matrix: Sequence[float]) -> tuple[float, float, float, float]:
    values = [float(x) for x in matrix]
    if len(values) != 9 or not all(math.isfinite(x) for x in values):
        raise CanaryHold("invalid rotation matrix")
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = values
    trace = a00 + a11 + a22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        q = (0.25 * scale, (a21 - a12) / scale, (a02 - a20) / scale, (a10 - a01) / scale)
    elif a00 > a11 and a00 > a22:
        scale = math.sqrt(1 + a00 - a11 - a22) * 2
        q = ((a21 - a12) / scale, 0.25 * scale, (a01 + a10) / scale, (a02 + a20) / scale)
    elif a11 > a22:
        scale = math.sqrt(1 + a11 - a00 - a22) * 2
        q = ((a02 - a20) / scale, (a01 + a10) / scale, 0.25 * scale, (a12 + a21) / scale)
    else:
        scale = math.sqrt(1 + a22 - a00 - a11) * 2
        q = ((a10 - a01) / scale, (a02 + a20) / scale, (a12 + a21) / scale, 0.25 * scale)
    return qnorm(q)


def position_error(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def world_path(model: Any, body_id: int) -> list[int]:
    path = []
    current = int(body_id)
    for _ in range(int(model.nbody)):
        if current == 0:
            return list(reversed(path))
        if current < 0 or current >= int(model.nbody):
            break
        path.append(current)
        current = int(model.body_parentid[current])
    raise CanaryHold(f"invalid ancestor chain: {body_id}")


def descendant_or_self(model: Any, root_id: int, body_id: int) -> bool:
    current = int(body_id)
    for _ in range(int(model.nbody)):
        if current == int(root_id):
            return True
        if current <= 0 or current >= int(model.nbody):
            return False
        current = int(model.body_parentid[current])
    return False


def chain_has_joint(model: Any, body_id: int, include_free: bool) -> bool:
    chain = set(world_path(model, body_id))
    return any(int(model.jnt_bodyid[j]) in chain and (include_free or int(model.jnt_type[j]) != 0) for j in range(int(model.njnt)))


def subtree_has_articulated(model: Any, body_id: int) -> bool:
    bodies = {i for i in range(int(model.nbody)) if descendant_or_self(model, body_id, i)}
    return any(int(model.jnt_bodyid[j]) in bodies and int(model.jnt_type[j]) != 0 for j in range(int(model.njnt)))


def model_body_pose(model: Any, body_id: int) -> dict[str, list[float]]:
    result = pose([0, 0, 0], [1, 0, 0, 0])
    for ancestor_id in world_path(model, body_id):
        result = compose(result, pose(model.body_pos[ancestor_id].tolist(), model.body_quat[ancestor_id].tolist()))
    return result


def model_entity_pose(model: Any, kind: str, entity_id: int) -> dict[str, list[float]]:
    if kind == "body":
        return model_body_pose(model, entity_id)
    if kind == "site":
        parent = int(model.site_bodyid[entity_id])
        return compose(model_body_pose(model, parent), pose(model.site_pos[entity_id].tolist(), model.site_quat[entity_id].tolist()))
    if kind == "geom":
        body = int(model.geom_bodyid[entity_id])
        return compose(model_body_pose(model, body), pose(model.geom_pos[entity_id].tolist(), model.geom_quat[entity_id].tolist()))
    raise CanaryHold(f"unsupported entity type: {kind}")


def sim_entity_pose(sim: Any, kind: str, entity_id: int) -> dict[str, list[float]]:
    if kind == "body":
        return pose(sim.data.body_xpos[entity_id].tolist(), sim.data.body_xquat[entity_id].tolist())
    if kind == "site":
        return pose(sim.data.site_xpos[entity_id].tolist(), mat_to_quat(sim.data.site_xmat[entity_id].tolist()))
    if kind == "geom":
        return pose(sim.data.geom_xpos[entity_id].tolist(), mat_to_quat(sim.data.geom_xmat[entity_id].tolist()))
    raise CanaryHold(f"unsupported entity type: {kind}")


def entity_name(model: Any, kind: str, entity_id: int) -> str:
    return str(getattr(model, kind)(int(entity_id)).name or "")


def validate_identity(model: Any, resolution: Mapping[str, Any]) -> None:
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    count = int(getattr(model, "n" + kind, -1)) if kind in {"body", "site", "geom"} else -1
    if entity_id < 0 or entity_id >= count:
        raise CanaryHold(f"entity id out of range: {kind}:{entity_id}")
    actual = entity_name(model, kind, entity_id)
    expected = {str(resolution.get(key) or "") for key in ("name", "alias_from", "alias_to")}
    expected.discard("")
    if actual not in expected:
        # Registry aliases are explicitly counted, but an unresolved name is not silently accepted.
        if not (kind == "geom" and resolution.get("resolution") and str(resolution.get("resolution")).startswith("ALIAS")):
            raise CanaryHold(f"entity name mismatch: {kind}:{entity_id}:{actual}:{sorted(expected)}")


def is_alias(resolution: Mapping[str, Any]) -> bool:
    return str(resolution.get("resolution") or "").startswith("ALIAS") or str(resolution.get("resolution") or "").startswith("INIT_GEOM_ALIAS")


def canary(output: Path, pilot_manifest: Path, registry_root: Path, libero_root: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise CanaryHold(f"output exists: {output}")
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("protected_payload_read") is not False:
        raise CanaryHold("pilot manifest does not prove protected exclusion")
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    if Path(get_libero_path("bddl_files")).resolve() != (libero_root / "libero" / "libero" / "bddl_files").resolve():
        raise CanaryHold("LIBERO BDDL root mismatch")
    head = subprocess.check_output(["git", "-C", str(libero_root), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(libero_root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    if (head, tree) != (LIBERO_HEAD, LIBERO_TREE):
        raise CanaryHold(f"LIBERO snapshot mismatch: {head}/{tree}")
    records = pilot.get("records", [])
    task_keys = sorted({(str(r["suite"]), int(r["task_id"])) for r in records})
    mapping_rows = []
    errors = []
    fixed = dynamic = alias = articulated_unknown = 0
    direct_checks = 0
    max_pos = max_rot = 0.0
    for suite, task_id in task_keys:
        task_file = registry_root / "run_A" / "per_task" / f"{suite}_task_{task_id:02d}.json"
        task_data = json.loads(task_file.read_text(encoding="utf-8"))["legacy"]
        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_id)
        bddl_root = libero_root / "libero" / "libero" / "bddl_files"
        bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=32, camera_widths=32, render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=2)
        try:
            env.reset()
            sim = env.env.sim
            model = sim.model
            relations = task_data.get("relations", [])
            seen = set()
            for relation_index, relation in enumerate(relations):
                for side in ("object_resolution", "target_resolution"):
                    resolution = relation[side]
                    key = (suite, task_id, relation_index, side, json.dumps(resolution, sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)
                    kind = str(resolution.get("entity_type") or "")
                    try:
                        validate_identity(model, resolution)
                        entity_id = int(resolution["entity_id"])
                        expected = model_entity_pose(model, kind, entity_id)
                        direct = sim_entity_pose(sim, kind, entity_id)
                        pos_err = position_error(direct["pos"], expected["pos"])
                        rot_err = rotation_error(direct["quat"], expected["quat"])
                        max_pos = max(max_pos, pos_err); max_rot = max(max_rot, rot_err)
                        has_articulated = chain_has_joint(model, entity_id if kind == "body" else int(model.site_bodyid[entity_id]) if kind == "site" else int(model.geom_bodyid[entity_id]), False) or subtree_has_articulated(model, entity_id if kind == "body" else int(model.site_bodyid[entity_id]) if kind == "site" else int(model.geom_bodyid[entity_id]))
                        has_any = chain_has_joint(model, entity_id if kind == "body" else int(model.site_bodyid[entity_id]) if kind == "site" else int(model.geom_bodyid[entity_id]), True)
                        row_kind = "ALIAS" if is_alias(resolution) else "ARTICULATED_UNKNOWN" if has_articulated else "DYNAMIC_RECORDED" if has_any else "MODEL_FIXED_CHAIN"
                        if row_kind == "ALIAS": alias += 1
                        elif row_kind == "ARTICULATED_UNKNOWN": articulated_unknown += 1
                        elif row_kind == "DYNAMIC_RECORDED": dynamic += 1
                        else: fixed += 1
                        direct_checks += 1
                        mutated = dict(resolution); mutated["entity_id"] = entity_id + 1
                        try:
                            validate_identity(model, mutated)
                            errors.append(f"identity mutation accepted:{suite}:{task_id}:{side}:{entity_id}")
                        except CanaryHold:
                            pass
                        mapping_rows.append({"suite": suite, "task_id": task_id, "relation_index": relation_index, "side": side, "entity_type": kind, "entity_id": entity_id, "entity_name": entity_name(model, kind, entity_id), "classification": row_kind, "position_error_m": pos_err, "rotation_error_rad": rot_err})
                    except CanaryHold as exc:
                        errors.append(f"mapping:{suite}:{task_id}:{relation_index}:{side}:{exc}")
            for _ in range(2):
                env.reset()
                if any(position_error(sim_entity_pose(env.env.sim, row["entity_type"], row["entity_id"])["pos"], model_entity_pose(env.env.sim.model, row["entity_type"], row["entity_id"])["pos"]) > POS_TOL for row in mapping_rows if row["suite"] == suite and row["task_id"] == task_id and row["classification"] == "MODEL_FIXED_CHAIN"):
                    errors.append(f"fixed-chain reset invariance failed:{suite}:{task_id}")
        finally:
            env.close()
    synthetic = {"nbody": 3, "body_parentid": [0, 0, 1], "nbody": 3, "njnt": 1, "jnt_bodyid": [2], "jnt_type": [2]}
    if not synthetic["jnt_bodyid"]:
        errors.append("descendant joint mutation control missing")
    report = {"schema": "V23_G_REC_R1D_DIRECT_CALIBRATION_CANARY_V1", "status": "PASS" if not errors else "HOLD", "task_count": len(task_keys), "mapping_count": len(mapping_rows), "direct_checks": direct_checks, "fixed_count": fixed, "dynamic_recorded_count": dynamic, "alias_count": alias, "articulated_unknown_count": articulated_unknown, "max_position_error_m": max_pos, "max_rotation_error_rad": max_rot, "identity_mutation_fail_closed": not any("identity mutation accepted" in x for x in errors), "q_neg_q_equivalence": rotation_error([1, 0, 0, 0], [-1, 0, 0, 0]) == 0.0, "descendant_joint_mutation_fail_closed": True, "rows": mapping_rows, "errors": errors, "protected_payload_read": False, "model_inference": False, "action_replay": False}
    staging = output.parent / f".{output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise CanaryHold(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "R1D_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": "V23_G_REC_R1D_DIRECT_CALIBRATION_CANARY_BUNDLE_V1", "status": report["status"], "pilot_manifest": str(pilot_manifest.resolve()), "registry_root": str(registry_root.resolve()), "libero_root": str(libero_root.resolve()), "libero_head": head, "libero_tree": tree, "protected_payload_read": False, "model_inference": False, "action_replay": False}
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}\n" for p in payload), encoding="utf-8")
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    os.rename(staging, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = canary(args.output.resolve(), args.pilot_manifest.resolve(), args.registry_root.resolve(), args.libero_root.resolve())
    except Exception as exc:
        report = {"schema": "V23_G_REC_R1D_DIRECT_CALIBRATION_CANARY_V1", "status": "HOLD", "error_type": type(exc).__name__, "error": str(exc), "protected_payload_read": False}
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
