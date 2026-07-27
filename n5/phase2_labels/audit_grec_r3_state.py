"""[DeepSeek] G-REC R3-A+B: Recorded State Audit + State-Forward Reference.

Audit fresh40 telemetry for state completeness (R3-A), then build
independent state-forward world poses from qpos/qvel via sim.forward()
and compare against collector's direct recorded poses (R3-B).

DEVELOPMENT_ONLY — consumer_eligible = false.
"""
import json, os, sys, hashlib, time, math, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

# ── Frozen thresholds ──
BODY_ORIGIN_POS_LIMIT = 1e-8
BODY_ORIGIN_ROT_LIMIT = 1e-7
GEOM_POS_LIMIT = 1e-6
GEOM_ROT_LIMIT = 1e-6
EXTENT_LIMIT = 1e-6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quat_to_matrix(w, x, y, z):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def geodesic_distance(q1, q2):
    """Sign-invariant geodesic distance (radians)."""
    q1n = np.array(q1) / np.linalg.norm(q1)
    q2n = np.array(q2) / np.linalg.norm(q2)
    dot = abs(np.dot(q1n, q2n))
    dot = min(dot, 1.0)
    return 2.0 * math.atan2(math.sqrt(max(0, 1 - dot*dot)), dot)


def R3A_audit_task(episode_path):
    """R3-A: Audit recorded state completeness for one episode."""
    with open(episode_path) as f:
        ep = json.load(f)

    ident = ep.get("episode_id", ep.get("collection_episode_id", "unknown"))
    telemetry = ep.get("telemetry", [])
    if not telemetry:
        return {"episode_id": ident, "status": "NO_TELEMETRY", "state_forward_ready": False}

    t0 = telemetry[0]
    sim_state = t0.get("sim_state", {})
    qpos = sim_state.get("qpos", [])
    qvel = sim_state.get("qvel", [])
    act = sim_state.get("act")
    time_val = sim_state.get("time", 0)

    result = {
        "episode_id": ident,
        "suite": t0.get("suite", ""),
        "task_idx": t0.get("task_idx", -1),
        "steps": len(telemetry),
        "recorded_qpos_width": len(qpos),
        "recorded_qvel_width": len(qvel),
        "has_act": act is not None and len(act) > 0 if isinstance(act, list) else False,
        "has_time": time_val is not None,
        "has_sim_state": bool(sim_state),
        "has_entities": "entities" in t0,
        "has_contacts": "mujoco_contact_pairs" in t0,
        "state_forward_ready": False,
        "issues": [],
    }

    if not sim_state:
        result["issues"].append("MISSING_SIM_STATE")
    if len(qpos) == 0:
        result["issues"].append("MISSING_QPOS")
    if len(qvel) == 0:
        result["issues"].append("MISSING_QVEL")

    return result


def R3B_forward_one_step(env, sim_state, entity_keys):
    """R3-B: Forward sim from qpos/qvel and compute independent world poses."""
    sim = env.sim
    sim.data.qpos[:] = np.array(sim_state["qpos"], dtype=np.float64)
    sim.data.qvel[:] = np.array(sim_state["qvel"], dtype=np.float64)
    if sim_state.get("act") and len(sim_state["act"]) > 0:
        sim.data.act[:] = np.array(sim_state["act"], dtype=np.float64)
    sim.data.time = float(sim_state.get("time", 0))
    sim.forward()

    result = {}
    for etype, eid in entity_keys:
        if etype == "body":
            bid = int(eid)
            xpos = sim.data.body_xpos[bid].copy()
            xmat = sim.data.body_xmat[bid].copy()
            result[(etype, eid)] = {"pos": xpos.tolist(), "xmat": xmat.flatten().tolist()}
        elif etype == "site":
            sid = int(eid)
            xpos = sim.data.site_xpos[sid].copy()
            xmat = sim.data.site_xmat[sid].copy()
            result[(etype, eid)] = {"pos": xpos.tolist(), "xmat": xmat.flatten().tolist()}
        elif etype == "geom":
            gid = int(eid)
            xpos = sim.data.geom_xpos[gid].copy()
            xmat = sim.data.geom_xmat[gid].copy()
            result[(etype, eid)] = {"pos": xpos.tolist(), "xmat": xmat.flatten().tolist()}
    return result


def compare_poses(direct_pose, forward_pose):
    """Compare direct recorded pose against state-forward reference pose."""
    # Direct pose uses {"position": [x,y,z], "quaternion": [x,y,z,w]}
    # Forward pose uses {"pos": [x,y,z], "xmat": [9 floats]}
    dp = np.array(direct_pose.get("position", direct_pose.get("pos", [0, 0, 0])))
    fp = np.array(forward_pose.get("pos", [0, 0, 0]))
    pos_l1 = float(np.sum(np.abs(dp - fp)))
    pos_linf = float(np.max(np.abs(dp - fp)))
    pos_l2 = float(np.linalg.norm(dp - fp))

    # For rotation: compare forward xmat vs direct quat-derived xmat
    dquat = direct_pose.get("quaternion", [0, 0, 0, 1])
    if len(dquat) == 4:
        dmat = quat_to_matrix(dquat[0], dquat[1], dquat[2], dquat[3]).flatten()
    else:
        dmat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    fxmat = np.array(forward_pose.get("xmat", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
    rot_diff = float(np.max(np.abs(dmat - fxmat)))

    return {
        "pos_L1": pos_l1, "pos_Linf": pos_linf, "pos_L2": pos_l2,
        "rot_max_abs": rot_diff,
    }


def main():
    print("=" * 60)
    print("[DeepSeek] G-REC R3-A+B: State Audit + Forward Reference")
    print("=" * 60)

    fresh40_dirs = [
        "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_l10_b5c7853_20260727",
        "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_goal_b5c7853_20260727",
        "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_object_b5c7853_20260727",
        "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_spatial_b5c7853_20260727",
    ]

    # ── R3-A: State completeness audit ──
    print("\n--- R3-A: State Completeness Audit ---")
    all_audits = []
    per_suite = defaultdict(list)
    all_qpos_ok = True

    for batch_dir in fresh40_dirs:
        ep_dirs = sorted(glob.glob(os.path.join(batch_dir, "episodes", "*/")))
        for ep_dir in ep_dirs:
            ep_file = os.path.join(ep_dir, "episode.json")
            if not os.path.isfile(ep_file):
                continue
            audit = R3A_audit_task(ep_file)
            all_audits.append(audit)
            suite = audit.get("suite", "unknown")
            per_suite[suite].append(audit)
            issues_str = ",".join(audit["issues"]) if audit["issues"] else "OK"
            print(f"  {audit['episode_id']}: qpos={audit['recorded_qpos_width']} "
                  f"qvel={audit['recorded_qvel_width']} steps={audit['steps']} "
                  f"{issues_str}")
            if audit["issues"]:
                all_qpos_ok = False

    n_total = len(all_audits)
    n_qpos = sum(1 for a in all_audits if a["recorded_qpos_width"] > 0)
    print(f"\nR3-A Summary: {n_qpos}/{n_total} have qpos, "
          f"all_ok={'YES' if all_qpos_ok else 'NO'}")

    # R3-A verdict
    if all_qpos_ok:
        print("R3-A: STATE_FORWARD_READY — all episodes have qpos/qvel")
    else:
        print("R3-A: SCHEMA_MISMATCH — some episodes missing sim_state")
        return

    # ── R3-B: State-forward reference (canary: first episode) ──
    print("\n--- R3-B: State-Forward Reference (canary) ---")
    canary_ep = all_audits[0]
    canary_dir = None
    for batch_dir in fresh40_dirs:
        pattern = os.path.join(batch_dir, "episodes", "*", "episode.json")
        for ep_file in sorted(glob.glob(pattern)):
            with open(ep_file) as f:
                ep = json.load(f)
            if ep.get("episode_id") == canary_ep["episode_id"]:
                canary_dir = os.path.dirname(ep_file)
                break
        if canary_dir:
            break

    if not canary_dir:
        print("R3-B: HOLD — cannot locate canary episode file")
        return

    print(f"Canary: {canary_ep['episode_id']}")

    # Load BDDL and model
    suite = canary_ep["suite"]  # e.g., "libero_10"
    task_idx = canary_ep["task_idx"]
    print(f"  Suite: {suite}, Task: {task_idx}")

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco

    benchmark = get_benchmark(suite)(0)
    task = benchmark.get_task(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
    bddl_sha = hashlib.sha256(open(bddl_path, "rb").read()).hexdigest()
    print(f"  BDDL SHA: {bddl_sha[:16]}...")

    # Create env to get model
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=224, camera_widths=224,
        render_gpu_device_id=-1,
        has_renderer=False, has_offscreen_renderer=False,
        horizon=500,
    )
    env.reset()
    model = env.sim.model
    print(f"  Model: nq={model.nq}, nv={model.nv}, na={model.na}, "
          f"nmocap={model.nmocap}")

    # Check qpos dimension match
    recorded_qpos_w = canary_ep["recorded_qpos_width"]
    if recorded_qpos_w != model.nq:
        print(f"  WARNING: recorded qpos width {recorded_qpos_w} != model.nq {model.nq}")
        if model.nmocap > 0:
            print(f"  MOCAP required: nmocap={model.nmocap} — may need mocap state")
    else:
        print(f"  qpos dimension match: {recorded_qpos_w} == {model.nq} OK")

    # Run state-forward on first step
    with open(os.path.join(canary_dir, "episode.json")) as f:
        ep = json.load(f)

    t0 = ep["telemetry"][0]
    ss = t0["sim_state"]
    entities_0 = t0.get("entities", [])

    # Collect entity keys
    entity_keys = set()
    for e in entities_0:
        entity_keys.add((e["entity_type"], e["entity_id"]))

    forward_poses = R3B_forward_one_step(env, ss, entity_keys)

    # Compare
    errors = []
    for e in entities_0:
        key = (e["entity_type"], e["entity_id"])
        if key not in forward_poses:
            continue
        err = compare_poses(e["world_pose"], forward_poses[key])
        err["entity_name"] = e.get("entity_name", "?")
        err["entity_type"] = e["entity_type"]
        errors.append(err)

    pos_errors = [e["pos_Linf"] for e in errors]
    rot_errors = [e["rot_max_abs"] for e in errors]

    if pos_errors:
        print(f"\n  Position L_inf: max={max(pos_errors):.2e}, "
              f"median={np.median(pos_errors):.2e}")
    if rot_errors:
        print(f"  Rotation max_abs: max={max(rot_errors):.2e}, "
              f"median={np.median(rot_errors):.2e}")

    body_limit_ok = all(e <= BODY_ORIGIN_POS_LIMIT for e in pos_errors)
    rot_limit_ok = all(e <= BODY_ORIGIN_ROT_LIMIT for e in rot_errors)

    print(f"\n  Body origin pos ≤ {BODY_ORIGIN_POS_LIMIT}: "
          f"{'PASS' if body_limit_ok else 'FAIL'}")
    print(f"  Body origin rot ≤ {BODY_ORIGIN_ROT_LIMIT}: "
          f"{'PASS' if rot_limit_ok else 'FAIL'}")

    env.close()
    print("\n[DeepSeek] R3-A+B canary complete")


if __name__ == "__main__":
    main()
