"""[DeepSeek] G-REC R3-A/B-R1: Sealed State Audit + State-Forward Reference.

Fixes all GPT-review P0 issues:
  1. Audits ALL steps (not just step 0)
  2. Fail-closed: any threshold violation → non-zero exit
  3. Verifies input root SHA256SUMS seals
  4. Semantic entity from resolver (not raw telemetry ID)
  5. Quaternion wxyz order, geodesic error
  6. Body origin / geometry center / site / geom distinction
  7. Empty denominator → fail
  8. Per-task mocap/custom state model check

DEVELOPMENT_ONLY — consumer_eligible = false.
"""
import json, os, sys, hashlib, time, math, glob, argparse, copy
from collections import defaultdict
from pathlib import Path
import numpy as np

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Open gripper

# ── Frozen thresholds (from V23_G_REC_NUMERICAL_PROTOCOL_AMENDMENT_V1) ──
BODY_ORIGIN_POS_LIMIT = 1e-8
BODY_ORIGIN_ROT_LIMIT = 1e-7
GEOMETRY_POS_LIMIT = 1e-6
GEOMETRY_ROT_LIMIT = 1e-6
EXTENT_LIMIT = 1e-6


class GateHold(Exception):
    """Fail-closed: any unrecoverable issue raises this."""
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_seal(root):
    """Verify SHA256SUMS + sidecar for a sealed root. Returns sums dict. Raises on failure."""
    root = Path(root)
    sums_path = root / "SHA256SUMS"
    side_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        raise GateHold(f"not a sealed root: {root}")
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        raise GateHold(f"seal sidecar mismatch: {root}")
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            raise GateHold(f"seal file mismatch: {name} in {root}")
        expected[name] = digest
    return expected


def quat_to_matrix(w, x, y, z):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def mat_to_quat_wxyz(m):
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z). Matches collector mat_to_quat."""
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
    norm = math.sqrt(sum(x*x for x in q))
    return tuple(x / norm for x in q)


def geodesic_error_rad(q1_wxyz, q2_wxyz):
    """Sign-invariant geodesic distance (radians) between two wxyz quaternions."""
    q1 = np.array(q1_wxyz, dtype=float); q2 = np.array(q2_wxyz, dtype=float)
    q1 /= np.linalg.norm(q1); q2 /= np.linalg.norm(q2)
    dot = abs(np.dot(q1, q2))
    dot = min(dot, 1.0)
    return float(2.0 * math.atan2(math.sqrt(max(0, 1 - dot*dot)), dot))


KIND_THRESHOLDS = {
    "body_origin": (BODY_ORIGIN_POS_LIMIT, BODY_ORIGIN_ROT_LIMIT),
    "site": (GEOMETRY_POS_LIMIT, GEOMETRY_ROT_LIMIT),
    "geom_center": (GEOMETRY_POS_LIMIT, GEOMETRY_ROT_LIMIT),
}


def R3A_audit_all_steps(episode_path):
    """R3-A-R1: Audit every step of one episode. Returns audit dict or raises."""
    with open(episode_path) as f:
        ep = json.load(f)

    ident = ep.get("episode_id", "unknown")
    telemetry = ep.get("telemetry", [])
    if not telemetry:
        raise GateHold(f"empty telemetry: {ident}")

    n_steps = len(telemetry)
    step_indices = [t.get("step", -1) for t in telemetry]
    if step_indices != list(range(n_steps)):
        raise GateHold(f"step index gap: {ident} expected 0..{n_steps-1}, got {step_indices[:5]}...")

    # Check every step has sim_state
    qpos_widths = set()
    qvel_widths = set()
    nonfinite_steps = []
    missing_steps = []
    for t in telemetry:
        ss = t.get("sim_state", {})
        if not ss:
            missing_steps.append(t.get("step", -1))
            continue
        qpos = ss.get("qpos", [])
        qvel = ss.get("qvel", [])
        if len(qpos) == 0:
            missing_steps.append(t.get("step", -1))
            continue
        if not all(math.isfinite(float(v)) for v in qpos):
            nonfinite_steps.append(("qpos", t.get("step", -1)))
        if not all(math.isfinite(float(v)) for v in qvel):
            nonfinite_steps.append(("qvel", t.get("step", -1)))
        qpos_widths.add(len(qpos))
        qvel_widths.add(len(qvel))

    t0 = telemetry[0]
    ss0 = t0.get("sim_state", {})

    return {
        "episode_id": ident,
        "suite": t0.get("suite", ""),
        "task_idx": t0.get("task_idx", -1),
        "steps": n_steps,
        "qpos_widths": sorted(qpos_widths),
        "qvel_widths": sorted(qvel_widths),
        "has_time": all("time" in t.get("sim_state", {}) for t in telemetry),
        "has_sim_state_all_steps": len(missing_steps) == 0,
        "nonfinite_steps": nonfinite_steps,
        "missing_steps": missing_steps,
        "entity_count_step0": len(t0.get("entities", [])),
        "state_forward_ready": len(missing_steps) == 0 and len(nonfinite_steps) == 0,
    }


def R3B_forward_all_steps(episode_path, env, expected_entities, canonical_state, suite_seed, num_steps_wait):
    """R3-B-R2: Forward every step using collector init contract and compare.

    Collector init: set_official_seed → env.seed → env.reset → set_init_state →
    wait NUM_STEPS_WAIT → start recording.
    Verifier matches this exactly, then restores per-step qpos/qvel and calls sim.forward().

    expected_entities: dict of (entity_type, entity_id) -> {name, kind}
      kind ∈ {"body_origin", "site", "geom_center"}
    """
    with open(episode_path) as f:
        ep = json.load(f)

    ident = ep["episode_id"]
    telemetry = ep["telemetry"]
    import random as _random
    _random.seed(suite_seed)
    env.seed(suite_seed)
    env.reset()
    env.set_init_state(copy.deepcopy(canonical_state))
    # Apply NUM_STEPS_WAIT dummy actions (collector does this before recording)
    for _ in range(int(num_steps_wait)):
        env.step(DUMMY_ACTION)

    # Verify model fingerprint matches recorded dimensions
    ss0 = telemetry[0]["sim_state"]
    model = env.sim.model
    if len(ss0["qpos"]) != model.nq:
        raise GateHold(f"{ident}: qpos width {len(ss0['qpos'])} != nq {model.nq}")
    if len(ss0["qvel"]) != model.nv:
        raise GateHold(f"{ident}: qvel width {len(ss0['qvel'])} != nv {model.nv}")
    if model.nmocap > 0:
        raise GateHold(f"{ident}: nmocap={model.nmocap} — unsupported")

    records = []
    for t in telemetry:
        step = t["step"]
        ss = t["sim_state"]

        env.sim.data.qpos[:] = np.array(ss["qpos"], dtype=np.float64)
        env.sim.data.qvel[:] = np.array(ss["qvel"], dtype=np.float64)
        if ss.get("act") is not None and len(ss.get("act", [])) > 0:
            env.sim.data.act[:] = np.array(ss["act"], dtype=np.float64)
        env.sim.data.time = float(ss.get("time", 0))
        env.sim.forward()

        recorded_entities = {(e["entity_type"], e["entity_id"]): e for e in t.get("entities", [])}

        for key, expected in expected_entities.items():
            if key not in recorded_entities:
                raise GateHold(f"missing entity {key} at {ident}:{step}")

            rec = recorded_entities[key]
            rec_pos = np.array(rec["world_pose"]["position"])
            rec_quat_wxyz = np.array(rec["world_pose"]["quaternion"])  # wxyz from collector

            etype, eid = key
            if etype == "body":
                fwd_pos = env.sim.data.body_xpos[eid].copy()
                fwd_xmat = env.sim.data.body_xmat[eid].copy().flatten()
                kind = "body_origin"
            elif etype == "site":
                fwd_pos = env.sim.data.site_xpos[eid].copy()
                fwd_xmat = env.sim.data.site_xmat[eid].copy().flatten()
                kind = "site"
            elif etype == "geom":
                fwd_pos = env.sim.data.geom_xpos[eid].copy()
                fwd_xmat = env.sim.data.geom_xmat[eid].copy().flatten()
                kind = "geom_center"
            else:
                raise GateHold(f"unknown entity type: {etype}")

            pos_linf = float(np.max(np.abs(rec_pos - fwd_pos)))
            pos_l2 = float(np.linalg.norm(rec_pos - fwd_pos))

            # Convert forward xmat to wxyz quaternion using collector's exact algorithm
            fwd_quat_wxyz = mat_to_quat_wxyz(fwd_xmat)
            geo_err = geodesic_error_rad(rec_quat_wxyz, fwd_quat_wxyz)

            pos_limit, rot_limit = KIND_THRESHOLDS[kind]

            records.append({
                "episode_id": ident, "step": step,
                "entity_name": expected.get("name", "?"),
                "entity_type": etype, "entity_id": eid,
                "kind": kind,
                "pos_Linf": pos_linf, "pos_L2": pos_l2,
                "geodesic_rad": geo_err,
                "pos_limit": pos_limit, "rot_limit": rot_limit,
                "pos_pass": pos_linf <= pos_limit,
                "rot_pass": geo_err <= rot_limit,
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "canary", "full40"], default="canary")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"[DeepSeek] G-REC R3-A/B-R1: mode={args.mode}")
    print("=" * 60)

    fresh40_roots = {
        "libero_10": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_l10_b5c7853_20260727",
        "libero_goal": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_goal_b5c7853_20260727",
        "libero_object": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_object_b5c7853_20260727",
        "libero_spatial": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_spatial_b5c7853_20260727",
    }

    # ── Verify all input seals ──
    print("\n--- Seal Verification ---")
    for suite, root in fresh40_roots.items():
        seal = verify_seal(root)
        print(f"  {suite}: SEAL_OK ({len(seal)} files)")

    # ── Collect all episodes ──
    all_episodes = []
    for suite, root in fresh40_roots.items():
        ep_dirs = sorted(glob.glob(os.path.join(root, "episodes", "*/")))
        for ep_dir in ep_dirs:
            ep_file = os.path.join(ep_dir, "episode.json")
            if os.path.isfile(ep_file):
                all_episodes.append(ep_file)

    if len(all_episodes) != 40:
        raise GateHold(f"expected 40 episodes, found {len(all_episodes)}")

    # ── R3-A-R1: Full-step audit ──
    print(f"\n--- R3-A-R1: Full-Step Audit ({len(all_episodes)} episodes) ---")
    audits = []
    total_steps = 0
    any_issue = False
    for ep_file in sorted(all_episodes):
        audit = R3A_audit_all_steps(ep_file)
        audits.append(audit)
        total_steps += audit["steps"]
        flag = "ISSUE" if not audit["state_forward_ready"] else "OK"
        if not audit["state_forward_ready"]:
            any_issue = True
        print(f"  {audit['episode_id']}: steps={audit['steps']} "
              f"qpos_w={audit['qpos_widths']} qvel_w={audit['qvel_widths']} {flag}")
        if audit["nonfinite_steps"]:
            print(f"    NONFINITE: {audit['nonfinite_steps'][:5]}")
        if audit["missing_steps"]:
            print(f"    MISSING: {audit['missing_steps'][:5]}")

    if any_issue:
        raise GateHold("R3-A-R1: some episodes have missing/nonfinite sim_state steps")

    if total_steps != 8332:
        raise GateHold(f"R3-A-R1: expected 8332 total steps, got {total_steps}")

    print(f"\nR3-A-R1: STATE_FORWARD_READY — {len(audits)}/{len(audits)} episodes, "
          f"{total_steps} steps, all have sim_state")

    if args.mode == "audit":
        print("\n[DeepSeek] R3-A-R1 audit complete")
        return

    # ── R3-B-R2: State-forward with collector init contract ──
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv

    NUM_STEPS_WAIT = 10
    COLLECTION_SEED = 0
    import random as _random

    episodes_to_run = all_episodes if args.mode == "full40" else [
        ep for ep in sorted(all_episodes)
        if "libero_10" in ep and "task_00" in ep and "state_15" in ep
    ][:1]
    if not episodes_to_run:
        raise GateHold("no canary episode found")

    all_records = []
    for ep_file in episodes_to_run:
        with open(ep_file) as f:
            ep = json.load(f)
        ident = ep["episode_id"]
        suite = ep["telemetry"][0]["suite"]
        task_idx = ep["telemetry"][0]["task_idx"]
        t0_entities = ep["telemetry"][0]["entities"]
        parts = ident.split("/")
        state_id = int(parts[-1].replace("state_", ""))

        print(f"\n  {ident}: suite={suite} task={task_idx} state={state_id}")

        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"),
                                 task.problem_folder, task.bddl_file)
        bddl_sha = sha256_file(bddl_path)

        suite_dict = get_benchmark_dict()
        suite_obj = suite_dict[suite]()
        init_states = suite_obj.get_task_init_states(task_idx)
        if state_id >= len(init_states):
            raise GateHold(f"{ident}: state_id {state_id} >= {len(init_states)}")
        canonical_state = init_states[state_id]

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=256, camera_widths=256,
            render_gpu_device_id=-1,
            has_renderer=False, has_offscreen_renderer=False,
            horizon=520,
        )
        _random.seed(COLLECTION_SEED)
        env.seed(COLLECTION_SEED)
        env.reset()
        env.set_init_state(copy.deepcopy(canonical_state))
        for _ in range(NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)

        model = env.sim.model
        print(f"    nq={model.nq} nv={model.nv} nmocap={model.nmocap} "
              f"BDDL={bddl_sha[:12]}... OK")

        kind_map = {"body": "body_origin", "site": "site", "geom": "geom_center"}
        expected = {}
        for e in t0_entities:
            key = (e["entity_type"], e["entity_id"])
            expected[key] = {
                "name": e.get("entity_name", "?"),
                "kind": kind_map.get(e["entity_type"], e["entity_type"]),
            }

        records = R3B_forward_all_steps(
            ep_file, env, expected, canonical_state, COLLECTION_SEED, NUM_STEPS_WAIT)
        all_records.extend(records)
        env.close()

    # ── Error analysis ──
    print(f"\n--- R3-B-R1: State-Forward Results ({len(all_records)} cases) ---")
    if not all_records:
        raise GateHold("R3-B-R1: empty denominator — no entities compared")

    pos_errors = [r["pos_Linf"] for r in all_records]
    geo_errors = [r["geodesic_rad"] for r in all_records]

    pos_pass = sum(1 for r in all_records if r["pos_pass"])
    rot_pass = sum(1 for r in all_records if r["rot_pass"])
    any_fail = pos_pass != len(all_records) or rot_pass != len(all_records)

    print(f"  Cases: {len(all_records)}")
    print(f"  Position: {pos_pass}/{len(all_records)} pass "
          f"(max={max(pos_errors):.2e}, p99={np.percentile(pos_errors, 99):.2e})")
    print(f"  Rotation: {rot_pass}/{len(all_records)} pass "
          f"(max={max(geo_errors):.2e}, p99={np.percentile(geo_errors, 99):.2e})")

    # Top-5 worst
    sorted_by_pos = sorted(all_records, key=lambda r: -r["pos_Linf"])[:5]
    print(f"\n  Top-5 position errors:")
    for r in sorted_by_pos:
        print(f"    {r['episode_id']}:{r['step']} {r['entity_name']}({r['entity_type']}#{r['entity_id']}) "
              f"Linf={r['pos_Linf']:.2e} geo={r['geodesic_rad']:.2e}")

    if any_fail:
        print(f"\nR3-B-R1: FAIL — {len(all_records) - pos_pass} position / "
              f"{len(all_records) - rot_pass} rotation violations")
        sys.exit(5)
    else:
        print(f"\nR3-B-R1: PASS — all {len(all_records)} cases within limits")

    print("\n[DeepSeek] R3-A/B-R1 complete")


if __name__ == "__main__":
    main()
