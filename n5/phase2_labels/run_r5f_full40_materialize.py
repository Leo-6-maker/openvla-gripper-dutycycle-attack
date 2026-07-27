"""[DeepSeek] R5-F: Corrected FIT Full40 Materialization (v2 — static audit fixes).

Reads the frozen V23 pilot manifest to obtain exact 40 episode identities
(suite, task_id, state_id, seed, initial_state_sha). Validates closure before
collection. Runs the corrected forward-before-capture protocol for each episode.

Protocol: PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE
Prerequisite: R5-E SAME_LIVE_GATE_PASS receipt

Usage (server):
  python n5/phase2_labels/run_r5f_full40_materialize.py \
    --model-path /path/to/openvla-checkpoint \
    --upstream-root /path/to/openvla-upstream \
    --official-worker /path/to/official_clean_worker.py \
    --pilot-manifest /path/to/frozen_pilot_manifest.json \
    --r5e-receipt /path/to/r5e_sealed_root \
    --registry-root /path/to/new_c1_v2_registry/run_A/per_task \
    --alias-ledger /path/to/new_c1_v2_registry/ALIAS_LEDGER.json \
    --output-root /path/to/output \
    --run-label A \
    --gpu 0
"""
import argparse, copy, hashlib, importlib, json, math, os, pickle
import platform, random, socket, subprocess, sys, time, uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FOUR_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FORBIDDEN_PATH_TOKENS = {"cal", "check", "g10", "t2r", "attack", "teacher", "student"}


class CollectionHold(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def reject_path(path):
    parts = {p.lower() for p in Path(path).resolve().parts}
    if parts & FORBIDDEN_PATH_TOKENS:
        raise CollectionHold(f"forbidden path: {path}")


def mat_to_quat(m):
    values = [float(x) for x in m]
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = values
    trace = a00 + a11 + a22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (a21 - a12) / s, (a02 - a20) / s, (a10 - a01) / s)
    elif a00 > a11 and a00 > a22:
        s = math.sqrt(1 + a00 - a11 - a22) * 2
        q = ((a21 - a12) / s, 0.25 * s, (a01 + a10) / s, (a02 + a20) / s)
    elif a11 > a22:
        s = math.sqrt(1 + a11 - a00 - a22) * 2
        q = ((a02 - a20) / s, (a01 + a10) / s, 0.25 * s, (a12 + a21) / s)
    else:
        s = math.sqrt(1 + a22 - a00 - a11) * 2
        q = ((a10 - a01) / s, (a02 + a20) / s, (a12 + a21) / s, 0.25 * s)
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def _verify_source_stability(qpos_before, qvel_before, act_before, time_before, data, step, label):
    qpos_after = data.qpos.copy()
    qvel_after = data.qvel.copy()
    act_after = data.act.copy() if hasattr(data, 'act') and data.act is not None else None
    time_after = float(data.time)
    pos_drift = float(np.max(np.abs(qpos_before - qpos_after)))
    vel_drift = float(np.max(np.abs(qvel_before - qvel_after)))
    time_drift = abs(float(time_before) - time_after)
    act_none_transition = (act_before is None) != (act_after is None)
    act_len_change = False
    act_drift = 0.0
    if not act_none_transition and act_before is not None and act_after is not None:
        if len(act_before) != len(act_after):
            act_len_change = True
        elif len(act_before) > 0:
            act_drift = float(np.max(np.abs(act_before - act_after)))
    if pos_drift > 0 or vel_drift > 0 or time_drift > 0 or act_drift > 0 or act_none_transition or act_len_change:
        raise CollectionHold(
            f"source state mutated by {label} at step {step}: "
            f"qpos_drift={pos_drift:.2e} qvel_drift={vel_drift:.2e} "
            f"time_drift={time_drift:.2e} act_drift={act_drift:.2e}"
            f"{' act_none_transition' if act_none_transition else ''}"
            f"{' act_len_change' if act_len_change else ''}")
    return True


def collect_entity(model, data, resolution):
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    if kind == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            raise CollectionHold(f"body id out of range: {entity_id}")
        actual_name = str(model.body(entity_id).name or "")
        pos = data.body_xpos[entity_id].tolist()
        quat = [float(x) for x in data.body_xquat[entity_id]]
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite body position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite body quaternion: {actual_name}")
        parent = int(model.body_parentid[entity_id])
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": parent, "world_pose": {"position": pos, "quaternion": quat}}
    if kind == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            raise CollectionHold(f"site id out of range: {entity_id}")
        actual_name = str(model.site(entity_id).name or "")
        body_id = int(model.site_bodyid[entity_id])
        pos = data.site_xpos[entity_id].tolist()
        quat = mat_to_quat(data.site_xmat[entity_id])
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite site position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite site quaternion: {actual_name}")
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": {"position": pos, "quaternion": quat}}
    if kind == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            raise CollectionHold(f"geom id out of range: {entity_id}")
        actual_name = str(model.geom(entity_id).name or "")
        body_id = int(model.geom_bodyid[entity_id])
        pos = data.geom_xpos[entity_id].tolist()
        quat = mat_to_quat(data.geom_xmat[entity_id])
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite geom position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite geom quaternion: {actual_name}")
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": {"position": pos, "quaternion": quat}}
    raise CollectionHold(f"unsupported entity kind: {kind}")


def load_pilot_identities(pilot_path):
    """Parse frozen pilot manifest. Returns list of validated episode identity dicts.
    Requires exactly 40 records: 10 per suite, 1 per task, no duplicates."""
    with open(pilot_path) as f:
        pilot = json.load(f)
    if pilot.get("protected_payload_read") is not False:
        raise CollectionHold("pilot manifest must have protected_payload_read=false")
    if pilot.get("no_attack") is not True:
        raise CollectionHold("pilot manifest must have no_attack=true")

    records = pilot.get("records", [])
    if len(records) != 40:
        raise CollectionHold(f"pilot must have exactly 40 records, got {len(records)}")

    identities = []
    seen = set()
    suite_counts = {s: 0 for s in FOUR_SUITES}
    task_per_suite = {s: set() for s in FOUR_SUITES}

    for rec in records:
        suite = str(rec["suite"])
        task_id = int(rec["task_id"])
        state_id = int(rec["state_id"])
        ep_id = str(rec["episode_id"])
        if "collection_seed" not in rec:
            raise CollectionHold(
                f"pilot record {ep_id} missing collection_seed")
        seed_val = int(rec["collection_seed"])
        init_sha = rec.get("initial_state_sha256", "")
        if not init_sha or not isinstance(init_sha, str) or len(init_sha) != 64:
            raise CollectionHold(
                f"pilot record {ep_id} missing or invalid initial_state_sha256: {init_sha[:20] if init_sha else 'MISSING'}")

        if suite not in FOUR_SUITES:
            raise CollectionHold(f"unknown suite: {suite}")
        if task_id < 0 or task_id >= 10:
            raise CollectionHold(f"task_id out of range: {task_id}")
        if ep_id in seen:
            raise CollectionHold(f"duplicate episode_id: {ep_id}")
        seen.add(ep_id)

        identities.append({
            "suite": suite, "task_id": task_id, "state_id": state_id,
            "episode_id": ep_id, "collection_seed": seed_val,
            "initial_state_sha256": init_sha,
        })
        suite_counts[suite] += 1
        task_per_suite[suite].add(task_id)

    for suite in FOUR_SUITES:
        if suite_counts[suite] != 10:
            raise CollectionHold(f"{suite}: expected 10 records, got {suite_counts[suite]}")
        if task_per_suite[suite] != set(range(10)):
            raise CollectionHold(f"{suite}: missing task ids: {set(range(10)) - task_per_suite[suite]}")

    return identities


def load_resolutions(registry_path, allow_articulated=False):
    """Load relation-bound entity resolutions from C1-V2 per-task registry.
    If allow_articulated and task is ARTICULATED_UNSUPPORTED with 0 relations,
    returns empty dict (geometry NOT_APPLICABLE)."""
    with open(registry_path) as f:
        data = json.load(f)
    legacy = data.get("legacy", data)
    relations = legacy.get("relations", [])
    disposition = legacy.get("task_disposition", "")
    if not relations:
        if allow_articulated and disposition == "ARTICULATED_UNSUPPORTED":
            return {}, relations  # NOT_APPLICABLE — no geometry entities
        raise CollectionHold(f"registry has no relations: {registry_path}")

    VALID = frozenset({"EXACT_BODY", "EXACT_SITE", "EXACT_GEOM", "APPROVED_STRUCTURAL_ALIAS"})
    unique = {}
    for rel in relations:
        for side in ("object_resolution", "target_resolution"):
            res = rel[side]
            resolution = res.get("resolution", "?")
            if resolution in VALID:
                unique[(res["entity_type"], res["entity_id"])] = res
            elif resolution in ("UNRESOLVED", "AMBIGUOUS") or resolution.startswith("BLOCKED_"):
                raise CollectionHold(
                    f"registry has {resolution} {side}: {res.get('name', '?')}")
    if not unique:
        raise CollectionHold(f"no relation-bound entities in registry: {registry_path}")
    return unique, relations


def verify_r5e_receipt(r5e_path):
    """Verify R5-E sealed root — full seal check + status verification."""
    r5e = Path(r5e_path).resolve()
    if not r5e.is_dir():
        raise CollectionHold(f"R5-E receipt not found: {r5e}")
    sums_path = r5e / "SHA256SUMS"
    side_path = r5e / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        raise CollectionHold(f"R5-E receipt not sealed: {r5e}")
    # Verify sidecar
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        raise CollectionHold(f"R5-E seal sidecar mismatch: {r5e}")
    # Verify every file in SHA256SUMS
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            raise CollectionHold(f"R5-E SHA256SUMS malformed line: {line}")
        digest, name = parts
        name = name.lstrip("*")
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        target = r5e / name
        if not target.is_file():
            raise CollectionHold(f"R5-E sealed file missing: {name}")
        if sha256_file(target) != digest:
            raise CollectionHold(f"R5-E seal file mismatch: {name}")
    # Verify manifest status
    manifest_path = r5e / "MANIFEST.json"
    if not manifest_path.is_file():
        raise CollectionHold(f"R5-E manifest missing: {r5e}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in ("SAME_LIVE_GATE_PASS", "PASS"):
        raise CollectionHold(
            f"R5-E status is not PASS: {manifest.get('status', '?')}")
    if not manifest.get("consumer_eligible", False):
        raise CollectionHold("R5-E receipt is not consumer_eligible")
    return {
        "path": str(r5e), "sha256sums_sha256": sidecar[0],
        "manifest_sha256": sha256_file(manifest_path),
        "status": manifest["status"],
    }


def verify_entity_identity(model, entity_type, entity_id, expected_name):
    """Verify MuJoCo entity name matches registry expectation. Raises on mismatch."""
    if entity_type == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            raise CollectionHold(f"body id {entity_id} out of range")
        actual = str(model.body(entity_id).name or "")
    elif entity_type == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            raise CollectionHold(f"site id {entity_id} out of range")
        actual = str(model.site(entity_id).name or "")
    elif entity_type == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            raise CollectionHold(f"geom id {entity_id} out of range")
        actual = str(model.geom(entity_id).name or "")
    else:
        raise CollectionHold(f"unknown entity type: {entity_type}")
    if actual != expected_name:
        raise CollectionHold(
            f"entity identity mismatch: expected '{expected_name}', "
            f"got '{actual}' for {entity_type}#{entity_id}")
    return True


def _validate_episode_shapes(episode):
    """Validate shape + finiteness of all actions, states, EEF, gripper, entities."""
    import math
    for step_data in episode.get("steps", []):
        for key in ["action_raw_7d", "score_action_7d", "action_env_7d"]:
            arr = step_data.get(key, [])
            if len(arr) != 7 or not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"invalid {key} at step {step_data.get('step')}")
    for tel in episode.get("telemetry", []):
        ss = tel.get("sim_state", {})
        for key in ["qpos", "qvel"]:
            arr = ss.get(key, [])
            if not arr or not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"invalid {key} at step {tel.get('step')}")
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
            arr = tel.get(key, [])
            if key == "robot0_eef_pos" and len(arr) != 3:
                raise CollectionHold(f"invalid eef_pos shape at step {tel.get('step')}")
            if not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"non-finite {key} at step {tel.get('step')}")
        for ent in tel.get("entities", []):
            pos = ent["world_pose"]["position"]
            quat = ent["world_pose"]["quaternion"]
            if len(pos) != 3 or len(quat) != 4:
                raise CollectionHold(f"invalid entity pose shape at step {tel.get('step')}")
            if not all(math.isfinite(float(x)) for x in pos):
                raise CollectionHold(f"non-finite entity position at step {tel.get('step')}")
            if not all(math.isfinite(float(x)) for x in quat):
                raise CollectionHold(f"non-finite entity quaternion at step {tel.get('step')}")


def capture_one_episode(module, suite, task_idx, state_id, collection_seed,
                        registry_dir, canonical_state, task, adapter):
    """Collect a single episode using corrected forward-before-capture protocol."""
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    reg_path = Path(registry_dir) / f"{suite}_task_{task_idx:02d}.json"
    registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
    legacy = registry_data.get("legacy", registry_data)
    is_articulated = legacy.get("task_disposition") == "ARTICULATED_UNSUPPORTED"
    resolutions, relations = load_resolutions(str(reg_path), allow_articulated=True)

    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    task_bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
    task_bddl_sha = sha256_file(task_bddl)

    module.set_official_seed(collection_seed)
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(collection_seed)
        env.reset()
        obs = env.set_init_state(copy.deepcopy(canonical_state))
        for _ in range(int(module.NUM_STEPS_WAIT)):
            obs = env.step([0, 0, 0, 0, 0, 0, -1])[0]

        # Verify all registry entities match the live MuJoCo model
        model = env.sim.model
        for (etype, eid), res in resolutions.items():
            expected_name = res.get("alias_to", res.get("name", "?"))
            verify_entity_identity(model, etype, eid, expected_name)

        rows = []
        privileged = []
        generation_counts = []

        for step in range(HORIZONS[suite]):
            # R5-C1: forward-before-capture protocol
            qpos_pre = env.sim.data.qpos.copy()
            qvel_pre = env.sim.data.qvel.copy()
            act_pre = env.sim.data.act.copy() if (hasattr(env.sim.data, 'act') and
                        env.sim.data.act is not None) else None
            time_pre = float(env.sim.data.time)

            # Verify source state is finite
            if not all(math.isfinite(float(x)) for x in qpos_pre):
                raise CollectionHold(f"non-finite qpos at step {step}")
            if not all(math.isfinite(float(x)) for x in qvel_pre):
                raise CollectionHold(f"non-finite qvel at step {step}")

            env.sim.forward()
            _verify_source_stability(qpos_pre, qvel_pre, act_pre, time_pre,
                                     env.sim.data, step, "capture_forward")
            model = env.sim.model; data = env.sim.data
            sim_state = env.sim.get_state()
            entities = [collect_entity(model, data, res) for res in resolutions.values()]

            privileged.append({
                "step": step, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "sim_state": {
                    "time": float(data.time),
                    "qpos": sim_state.qpos.tolist(),
                    "qvel": sim_state.qvel.tolist(),
                    "act": getattr(sim_state, "act", None).tolist() if getattr(sim_state, "act", None) is not None else None,
                },
                "robot0_eef_pos": jsonable(obs.get("robot0_eef_pos", [])),
                "robot0_eef_quat": jsonable(obs.get("robot0_eef_quat", [])),
                "robot0_gripper_qpos": jsonable(obs.get("robot0_gripper_qpos", [])),
                "object_state": jsonable(obs.get("object-state", [])),
                "entities": entities,
                "forward_before_capture": True,
                "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
                "contact_count": int(data.ncon),
            })

            image = get_libero_image(obs, 224)
            clean_action, generation, score_meta = adapter.predict_action_with_scores(
                image, str(task.language))
            count = score_meta.get("generation_passes_per_step")
            if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                raise CollectionHold(f"generation pass count: {count}")
            generation_counts.append(count)
            score_action = [float(x) for x in jsonable(score_meta["score_action"])]
            raw_action = [float(x) for x in jsonable(clean_action)]
            if len(raw_action) != 7 or len(score_action) != 7:
                raise CollectionHold(f"action shape failed at step {step}")
            if max(abs(a - b) for a, b in zip(raw_action, score_action)) > 1e-6:
                raise CollectionHold(f"action parity failed at step {step}")
            executed = [float(x) for x in jsonable(adapter.postprocess(clean_action))]
            if len(executed) != 7:
                raise CollectionHold(f"executed action shape failed at step {step}")
            rows.append({
                "step": step, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "action_raw_7d": raw_action, "score_action_7d": score_action,
                "action_env_7d": executed, "generation_passes_per_step": count,
                "single_generation_parity_pass": True, "action_mutation_by_detector": False,
            })
            obs, _reward, done, _info = env.step(executed)
            if done:
                break
    finally:
        env.close()

    if not generation_counts or any(x != 1 for x in generation_counts):
        raise CollectionHold("generation closure failed")

    return {
        "episode_id": f"{suite}/task_{task_idx:02d}/state_{state_id}",
        "suite": suite, "task_id": task_idx, "state_id": state_id,
        "collection_seed": collection_seed,
        "pilot_identity_bound": True,
        "task_bddl_sha256": task_bddl_sha,
        "registry_task_sha256": sha256_file(str(reg_path)),
        "step_count": len(rows), "official_horizon": HORIZONS[suite],
        "generation_passes_per_step": generation_counts,
        "steps": rows, "telemetry": privileged,
        "relations": relations,
        "source_mode": "NEW_FIT_ONLY_CORRECTED_COLLECTOR",
        "forward_before_capture": True,
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "geometry_status": "NOT_APPLICABLE" if (is_articulated and not resolutions) else "OK",
        "placement_state": "UNKNOWN" if (is_articulated and not resolutions) else "OK",
        "placement_mask": 0 if (is_articulated and not resolutions) else 1,
        "original_payload_target_pose_available": False,
        "model_inference": True, "attack_enabled": False,
        "detector_loaded": False, "teacher_labels_generated": False,
    }


def seal_root(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": len(payload)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True,
                        help="Sealed FIT-INFERENCE transition receipt root")
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True, choices=["A", "B"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate all inputs but do NOT load model or collect")
    args = parser.parse_args()

    # Path safety audit
    for path in [args.model_path, args.upstream_root, args.official_worker,
                 args.pilot_manifest, args.transition_receipt, args.registry_root,
                 args.alias_ledger]:
        reject_path(path)

    out_root = Path(args.output_root).resolve() / f"run_{args.run_label}"
    if out_root.exists() or out_root.is_symlink():
        raise SystemExit(f"output exists: {out_root}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Validate GPU mapping: CUDA_VISIBLE_DEVICES must map requested gpu to device 0
    visible = str(args.gpu)
    visible_gpus = [int(x.strip()) for x in visible.split(",") if x.strip()]
    if len(visible_gpus) != 1 or visible_gpus[0] != int(args.gpu):
        raise SystemExit(f"GPU mapping mismatch: CUDA_VISIBLE_DEVICES={visible} gpu={args.gpu}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    random.seed(args.seed)

    # Source provenance
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    repo_root = script_path.parent.parent.parent
    source_commit = git_value(repo_root, "rev-parse", "HEAD")
    source_tree = git_value(repo_root, "rev-parse", "HEAD^{tree}")
    protocol_path = repo_root / "reports" / "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE.json"
    protocol_sha = sha256_file(protocol_path) if protocol_path.is_file() else "MISSING"

    # ── Input binding ──
    pilot_path = Path(args.pilot_manifest).resolve()
    pilot_sha = sha256_file(pilot_path)

    # Load and validate pilot identities
    identities = load_pilot_identities(str(pilot_path))
    print(f"Pilot manifest: {pilot_sha}")
    print(f"  Identities: {len(identities)} (10 per suite)")

    # Registry manifest binding
    registry_summary_path = Path(args.registry_root).parent / "ENTITY_REGISTRY_V2_SUMMARY.json"
    registry_manifest = {}
    if registry_summary_path.is_file():
        registry_manifest = {
            "path": str(registry_summary_path.resolve()),
            "sha256": sha256_file(registry_summary_path),
        }

    alias_ledger_path = Path(args.alias_ledger).resolve()
    alias_ledger_sha = sha256_file(alias_ledger_path)

    # ── FIT-INFERENCE Transition Verification (BEFORE any model load) ──
    from fit_transition import verify_transition, TransitionRejected
    from libero.libero import get_libero_path
    try:
        transition_manifest = verify_transition(
            transition_root=args.transition_receipt,
            execution_source_commit=source_commit,
            script_sha=script_sha,
            model_path=args.model_path,
            official_worker_path=args.official_worker,
            pilot_manifest_path=pilot_path,
            registry_root=args.registry_root,
            alias_ledger_path=alias_ledger_path,
            upstream_root=args.upstream_root,
            libero_root=get_libero_path("bddl_files"),
            output_root=str(args.output_root.resolve()),
            gpu=0,
            physical_gpu=args.gpu,
        )
    except TransitionRejected as e:
        raise SystemExit(f"TRANSITION_REJECTED: {e}")

    print(f"Transition receipt: VERIFIED")
    print(f"  source commit: {source_commit}")

    # ── Preflight-only: stop before model load ──
    if args.preflight_only:
        print("\nPREFLIGHT_ONLY: All validations passed. No model loaded.")
        print(f"  openvla_import = 0")
        print(f"  load_policy = 0")
        print(f"  cuda_allocation = 0")
        print(f"  rollout = 0")
        return 0

    # ── Load module ──
    worker = Path(args.official_worker).resolve()
    if not worker.is_file():
        raise SystemExit(f"worker missing: {worker}")

    old_argv = sys.argv
    sys.argv = [str(worker), "--suite", "libero_10", "--gpu", str(args.gpu)]
    try:
        spec = importlib.util.spec_from_file_location("official_clean_worker", str(worker))
        if spec is None or spec.loader is None:
            raise SystemExit("cannot load worker")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv

    module.set_official_seed(args.seed)
    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
        base_vla_name=str(args.model_path))

    from libero.libero import get_libero_path
    from libero.libero import benchmark

    staging = out_root.parent / f".{out_root.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if staging.exists():
        raise SystemExit(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "episodes").mkdir()

    try:

    print("=" * 70)
    print(f"[DeepSeek] R5-F: Corrected FIT Full40 Materialization — Run {args.run_label}")
    print(f"  model={args.model_path}  gpu={args.gpu}  seed={args.seed}")
    print(f"  source_commit={source_commit}")
    print(f"  protocol_amendment_sha={protocol_sha}")
    print(f"  transition_receipt={args.transition_receipt}")
    print("=" * 70)

    suite_dict = benchmark.get_benchmark_dict()
    collections = []
    failures = []
    total_start = time.time()
    input_bindings = []

    for ident in identities:
        suite = ident["suite"]; task_idx = ident["task_id"]
        state_id = ident["state_id"]; ep_id = ident["episode_id"]
        coll_seed = ident["collection_seed"]
        task_key = f"{suite}/task_{task_idx:02d}"

        print(f"\n  {ep_id}...", end=" ", flush=True)
        try:
            suite_obj = suite_dict[suite]()
            task = suite_obj.get_task(task_idx)
            states = suite_obj.get_task_init_states(task_idx)
            if state_id >= len(states):
                raise CollectionHold(f"state_id {state_id} >= {len(states)}")
            canonical_state = copy.deepcopy(states[state_id])

            # Verify initial-state SHA
            init_state_sha = sha256_bytes(pickle.dumps(canonical_state, protocol=4))
            declared_sha = ident["initial_state_sha256"]
            if init_state_sha != declared_sha:
                raise CollectionHold(
                    f"initial_state_sha mismatch: computed={init_state_sha[:16]} "
                    f"declared={declared_sha[:16]}")

            # Check for articulated NOT_APPLICABLE tasks
            reg_path = Path(str(args.registry_root)) / f"{suite}_task_{task_idx:02d}.json"
            registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
            legacy = registry_data.get("legacy", registry_data)
            is_articulated = legacy.get("task_disposition") == "ARTICULATED_UNSUPPORTED"

            episode = capture_one_episode(
                module, suite, task_idx, state_id, coll_seed,
                str(args.registry_root), canonical_state, task, adapter)

            # Enforce output episode_id matches pilot
            if episode["episode_id"] != ep_id:
                raise CollectionHold(
                    f"episode_id mismatch: output={episode['episode_id']} pilot={ep_id}")

            # Shape + finite validation
            _validate_episode_shapes(episode)

            ep_dir = staging / "episodes" / f"{suite}_task_{task_idx:02d}"
            ep_dir.mkdir()
            (ep_dir / "episode.json").write_text(
                json.dumps(episode, indent=2, sort_keys=True, default=str), encoding="utf-8")
            ep_sha = sha256_file(ep_dir / "episode.json")

            binding = {
                "pilot_episode_id": ep_id,
                "output_episode_id": episode["episode_id"],
                "suite": suite, "task_id": task_idx, "state_id": state_id,
                "collection_seed": coll_seed,
                "initial_state_sha_verified": True,
                "initial_state_sha": init_state_sha,
                "steps": episode["step_count"],
                "entities": len(episode["telemetry"][0]["entities"]) if episode["telemetry"] else 0,
                "episode_sha256": ep_sha,
                "status": "OK",
            }
            input_bindings.append(binding)
            collections.append({"task_key": task_key, "episode_id": ep_id,
                               "steps": episode["step_count"], "sha256": ep_sha})
            print(f"steps={episode['step_count']} "
                  f"entities={len(episode['telemetry'][0]['entities']) if episode['telemetry'] else 0} "
                  f"init_sha=VERIFIED OK")
        except CollectionHold as e:
            print(f"HOLD: {e}")
            failures.append({"task_key": task_key, "episode_id": ep_id, "error": str(e)})
            input_bindings.append({
                "pilot_episode_id": ep_id, "suite": suite,
                "task_id": task_idx, "state_id": state_id,
                "status": "FAIL", "error": str(e),
            })

    elapsed = time.time() - total_start

    # ── Manifest ──
    n_collected = len(collections)
    n_failed = len(failures)
    all_ok = n_collected == 40 and n_failed == 0
    collection_status = "COMPLETE" if all_ok else "PARTIAL_NONCONSUMABLE"

    manifest = {
        "gate": "R5-F_CORRECTED_FULL40_MATERIALIZATION",
        "schema": "G_REC_CORRECTED_FULL40_V2",
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "protocol_amendment_sha256": protocol_sha,
        "run_label": args.run_label,
        "status": collection_status,
        "consumer_eligible": all_ok,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "source_commit": source_commit, "source_tree": source_tree,
        "script_sha256": script_sha,
        "transition_receipt_sha256sums": sha256_file(Path(args.transition_receipt) / "SHA256SUMS"),
        "c1_canonical_digest": transition_manifest.get("c1_canonical_digest"),
        "identity_set_digest": transition_manifest.get("identity_set_digest"),
        "pilot_manifest_sha256": pilot_sha,
        "registry_manifest": registry_manifest,
        "alias_ledger_sha256": alias_ledger_sha,
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": module.checkpoint_tree_fingerprint(args.model_path)[0],
        "processor_sha256": sha256_file(args.model_path / "preprocessor_config.json"),
        "upstream_root": str(args.upstream_root.resolve()),
        "upstream_commit": git_value(args.upstream_root, "rev-parse", "HEAD"),
        "libero_root": str(Path(get_libero_path("bddl_files")).resolve().parents[2]),
        "environment": {
            "python": sys.executable, "python_version": platform.python_version(),
            "torch": module.torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "hostname": socket.gethostname(),
        },
        "n_pilot_identities": len(identities),
        "n_collected": n_collected, "n_failed": n_failed,
        "total_steps": sum(c["steps"] for c in collections),
        "collections": collections,
        "failures": failures,
        "input_bindings": input_bindings,
        "forward_before_capture": True,
        "no_detector": True, "attack_enabled": False,
        "teacher_labels_generated": False,
        "protected_payload_read": False,
        "forbidden_path_audit": "PASS",
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "SEAL_RECEIPT.json").write_text(json.dumps({
        "schema": "V23_G_REC_CORRECTED_FULL40_SEAL_V2",
        "status": f"SEALED_{collection_status}",
        "run_label": args.run_label,
        "consumer_eligible": all_ok,
    }, indent=2, sort_keys=True), encoding="utf-8")

    seal = seal_root(staging)

    # Atomic no-replace publication
    try:
        staging.rename(out_root)
    except OSError:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit(f"rename failed — output may have appeared: {out_root}")

    finally:
        if staging.exists() and staging != out_root:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)

    print(f"\n{'=' * 70}")
    print(f"Run {args.run_label}: {n_collected}/{len(identities)} episodes collected")
    print(f"  Failures: {n_failed}")
    print(f"  Total steps: {sum(c['steps'] for c in collections)}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Sealed: {out_root}")
    print(f"  SHA256SUMS: {seal['sha256sums_sha256']}")
    print(f"  Status: {collection_status}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f['episode_id']}: {f['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
