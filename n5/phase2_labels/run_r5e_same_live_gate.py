"""[DeepSeek] R5-E: Formal Corrected Same-Live Gate (v2 — static audit fixes).

Relation-bound A/B/C triple-read across all 40 task types.
Entities resolved via C1-V2 registry (body for objects, site for regions).
Tests: A=pre-forward, B=after capture forward, C=after verification forward.

PASS criteria:
  - exact 40/40 tasks tested, skip=0
  - every expected relation side present exactly once
  - unresolved/ambiguous/blocked/forbidden = 0
  - actual MuJoCo entity name/type/id matches registry
  - B→C position failures = 0
  - B→C rotation failures = 0
  - source-state mutation across both forwards = 0
  - nonfinite source state or A/B/C poses = 0
  - all task identities exact once

A→B may be nonzero (diagnostic only — stale read reporting).
The consumable pose is B.
"""
import json, os, sys, math, copy, hashlib, time, argparse, uuid, subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_STEPS_WAIT = 10

# Thresholds from PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE
BODY_POS_LIMIT = 1e-8
BODY_ROT_LIMIT = 1e-7
SITE_POS_LIMIT = 1e-8
SITE_ROT_LIMIT = 1e-7
GEOM_POS_LIMIT = 1e-6
GEOM_ROT_LIMIT = 1e-6
VERIFICATION_LIMIT = 1e-15

FOUR_SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']
EXPECTED_TASKS = [(s, tid) for s in FOUR_SUITES for tid in range(10)]
VALID_RESOLUTIONS = frozenset({
    'EXACT_SITE', 'EXACT_BODY', 'EXACT_GEOM', 'APPROVED_STRUCTURAL_ALIAS',
})


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def geodesic_wxyz(q1, q2):
    q1n = np.array(q1, float); q2n = np.array(q2, float)
    q1n /= np.linalg.norm(q1n); q2n /= np.linalg.norm(q2n)
    d = abs(np.dot(q1n, q2n)); d = min(d, 1.0)
    return float(2.0 * math.atan2(math.sqrt(max(0, 1 - d*d)), d))


def mat_to_quat_wxyz(m):
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


def load_relation_entities(registry_path):
    """Extract relation-bound entity info from C1-V2 per-task registry.
    Returns (entities_dict, closure_report).
    Raises on unresolved/ambiguous/blocked or missing relations."""
    with open(registry_path) as f:
        data = json.load(f)
    legacy = data.get("legacy", data)
    relations = legacy.get("relations", [])
    if not relations:
        raise ValueError(f"registry has no relations: {registry_path}")

    entities = {}
    closure = {"n_relations": len(relations), "object_ok": 0, "object_issues": [],
               "target_ok": 0, "target_issues": []}

    for irel, rel in enumerate(relations):
        for side, role_key in [("object_resolution", "object"), ("target_resolution", "target")]:
            res = rel.get(side, {})
            resolution = res.get("resolution", "?")
            if resolution in VALID_RESOLUTIONS:
                key = (res["entity_type"], res["entity_id"])
                bddl_name = rel.get(f"{role_key}_bddl", "?")
                entities[key] = {
                    "name": res.get("name", bddl_name),
                    "entity_type": res["entity_type"],
                    "entity_id": res["entity_id"],
                    "resolution": resolution,
                    "semantic_role": res.get("semantic_role", "?"),
                    "bddl_name": bddl_name,
                    "relation_index": irel,
                    "side": role_key,
                }
                closure[f"{role_key}_ok"] += 1
            elif resolution == "UNRESOLVED":
                closure[f"{role_key}_issues"].append({
                    "relation_index": irel, "side": role_key,
                    "issue": "UNRESOLVED",
                    "detail": res.get("error_detail", {}),
                })
            elif resolution == "AMBIGUOUS":
                closure[f"{role_key}_issues"].append({
                    "relation_index": irel, "side": role_key,
                    "issue": "AMBIGUOUS",
                    "detail": res.get("error_detail", {}),
                })
            elif resolution.startswith("BLOCKED_"):
                closure[f"{role_key}_issues"].append({
                    "relation_index": irel, "side": role_key,
                    "issue": f"BLOCKED: {resolution}",
                    "detail": res.get("error_detail", {}),
                })
            else:
                closure[f"{role_key}_issues"].append({
                    "relation_index": irel, "side": role_key,
                    "issue": f"UNKNOWN_RESOLUTION: {resolution}",
                })

    return entities, closure


def verify_entity_identity(model, entity_type, entity_id, expected_name):
    """Verify that the MuJoCo entity matches the registry expectation."""
    if entity_type == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            return False, f"body id {entity_id} out of range [0, {model.nbody})"
        actual = str(model.body(entity_id).name or "")
    elif entity_type == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            return False, f"site id {entity_id} out of range [0, {model.nsite})"
        actual = str(model.site(entity_id).name or "")
    elif entity_type == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            return False, f"geom id {entity_id} out of range [0, {model.ngeom})"
        actual = str(model.geom(entity_id).name or "")
    else:
        return False, f"unknown entity type: {entity_type}"
    if actual != expected_name:
        return False, f"name mismatch: expected '{expected_name}', got '{actual}'"
    return True, "OK"


def test_one_task(suite, task_idx, state_id, seed, test_steps, registry_dir, apply_actions):
    """Run A/B/C same-live parity test on one task. Returns (records, summary) or (None, error_dict)."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    import random as _random

    # Load registry first to fail fast
    reg_path = os.path.join(registry_dir, f"{suite}_task_{task_idx:02d}.json")
    if not os.path.isfile(reg_path):
        return None, {"status": "SKIP", "error": f"registry missing: {reg_path}"}
    try:
        expected_entities, closure = load_relation_entities(reg_path)
    except Exception as e:
        return None, {"status": "SKIP", "error": f"registry load failed: {e}"}

    # Reject unresolved/ambiguous/blocked
    issues = closure["object_issues"] + closure["target_issues"]
    if issues:
        return None, {"status": "SKIP",
                      "error": f"registry has {len(issues)} resolution issues",
                      "closure": closure}

    _random.seed(seed)
    bm = get_benchmark(suite)(0); t = bm.get_task(task_idx)
    bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    sd = get_benchmark_dict(); so = sd[suite]()
    init_states = so.get_task_init_states(task_idx)
    if state_id >= len(init_states):
        return None, {"status": "SKIP",
                      "error": f"state_id {state_id} >= {len(init_states)}"}
    cs = init_states[state_id]

    env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256,
                             render_gpu_device_id=-1, has_renderer=False,
                             has_offscreen_renderer=False, horizon=520)
    try:
        env.seed(seed); env.reset(); env.set_init_state(copy.deepcopy(cs))
        for _ in range(NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)

        model = env.sim.model

        # Verify every registry entity matches actual MuJoCo model
        identity_issues = []
        for (etype, eid), info in expected_entities.items():
            ok, msg = verify_entity_identity(model, etype, eid, info["name"])
            if not ok:
                identity_issues.append({"entity": info, "error": msg})
        if identity_issues:
            return None, {"status": "SKIP",
                          "error": f"{len(identity_issues)} entity identity mismatches",
                          "identity_issues": identity_issues[:10]}

        if not expected_entities:
            return None, {"status": "SKIP", "error": "no relation-bound entities in registry"}

        records = []
        for step in range(test_steps):
            # Save source state BEFORE forward
            qpos_before = env.sim.data.qpos.copy()
            qvel_before = env.sim.data.qvel.copy()
            act_before = env.sim.data.act.copy() if (hasattr(env.sim.data, 'act') and
                          env.sim.data.act is not None) else None
            time_before = float(env.sim.data.time)

            # A: read BEFORE forward
            A_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    pos = env.sim.data.body_xpos[eid].copy()
                    rot = env.sim.data.body_xquat[eid].copy()
                elif etype == "site":
                    pos = env.sim.data.site_xpos[eid].copy()
                    rot = env.sim.data.site_xmat[eid].copy().flatten()
                elif etype == "geom":
                    pos = env.sim.data.geom_xpos[eid].copy()
                    rot = env.sim.data.geom_xmat[eid].copy().flatten()
                else:
                    return None, {"status": "SKIP",
                                  "error": f"unsupported entity type: {etype}"}
                if not all(math.isfinite(float(x)) for x in pos):
                    return None, {"status": "SKIP",
                                  "error": f"non-finite A position: {info['name']} step {step}"}
                A_poses[(etype, eid)] = (pos, rot, info)

            # First forward → B (capture forward)
            env.sim.forward()

            # Verify source state stability after first forward
            qpos_drift1 = float(np.max(np.abs(qpos_before - env.sim.data.qpos.copy())))
            qvel_drift1 = float(np.max(np.abs(qvel_before - env.sim.data.qvel.copy())))
            time_drift1 = abs(time_before - float(env.sim.data.time))
            act_drift1 = 0.0
            if act_before is not None and hasattr(env.sim.data, 'act') and env.sim.data.act is not None:
                act_drift1 = float(np.max(np.abs(act_before - env.sim.data.act.copy())))
            source_mutated_1 = (qpos_drift1 > 0 or qvel_drift1 > 0 or
                                time_drift1 > 0 or act_drift1 > 0)

            B_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    pos = env.sim.data.body_xpos[eid].copy()
                    rot = env.sim.data.body_xquat[eid].copy()
                elif etype == "site":
                    pos = env.sim.data.site_xpos[eid].copy()
                    rot = env.sim.data.site_xmat[eid].copy().flatten()
                elif etype == "geom":
                    pos = env.sim.data.geom_xpos[eid].copy()
                    rot = env.sim.data.geom_xmat[eid].copy().flatten()
                if not all(math.isfinite(float(x)) for x in pos):
                    return None, {"status": "SKIP",
                                  "error": f"non-finite B position: {info['name']} step {step}"}
                B_poses[(etype, eid)] = (pos, rot)

            # Save state before second forward
            qpos_mid = env.sim.data.qpos.copy()
            qvel_mid = env.sim.data.qvel.copy()
            act_mid = env.sim.data.act.copy() if (hasattr(env.sim.data, 'act') and
                       env.sim.data.act is not None) else None
            time_mid = float(env.sim.data.time)

            # Second forward → C (verification forward)
            env.sim.forward()

            # Verify source state stability after second forward
            qpos_drift2 = float(np.max(np.abs(qpos_mid - env.sim.data.qpos.copy())))
            qvel_drift2 = float(np.max(np.abs(qvel_mid - env.sim.data.qvel.copy())))
            time_drift2 = abs(time_mid - float(env.sim.data.time))
            act_drift2 = 0.0
            if act_mid is not None and hasattr(env.sim.data, 'act') and env.sim.data.act is not None:
                act_drift2 = float(np.max(np.abs(act_mid - env.sim.data.act.copy())))
            source_mutated_2 = (qpos_drift2 > 0 or qvel_drift2 > 0 or
                                time_drift2 > 0 or act_drift2 > 0)

            C_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    pos = env.sim.data.body_xpos[eid].copy()
                    rot = env.sim.data.body_xquat[eid].copy()
                elif etype == "site":
                    pos = env.sim.data.site_xpos[eid].copy()
                    rot = env.sim.data.site_xmat[eid].copy().flatten()
                elif etype == "geom":
                    pos = env.sim.data.geom_xpos[eid].copy()
                    rot = env.sim.data.geom_xmat[eid].copy().flatten()
                if not all(math.isfinite(float(x)) for x in pos):
                    return None, {"status": "SKIP",
                                  "error": f"non-finite C position: {info['name']} step {step}"}
                C_poses[(etype, eid)] = (pos, rot)

            # Compare
            for (etype, eid), info in expected_entities.items():
                a_pos, a_rot, _ = A_poses[(etype, eid)]
                b_pos, b_rot = B_poses[(etype, eid)]
                c_pos, c_rot = C_poses[(etype, eid)]

                ab_pos_err = float(np.max(np.abs(a_pos - b_pos)))
                bc_pos_err = float(np.max(np.abs(b_pos - c_pos)))

                if etype == "site":
                    ab_rot_err = float(np.max(np.abs(a_rot - b_rot)))
                    bc_rot_err = float(np.max(np.abs(b_rot - c_rot)))
                    ab_geo = ab_rot_err
                    bc_geo = bc_rot_err
                    b_rot_is_finite = all(math.isfinite(float(x)) for x in b_rot)
                else:
                    if etype == "body":
                        a_quat = a_rot; b_quat = b_rot; c_quat = c_rot
                    else:
                        a_quat = mat_to_quat_wxyz(a_rot)
                        b_quat = mat_to_quat_wxyz(b_rot)
                        c_quat = mat_to_quat_wxyz(c_rot)
                    ab_geo = geodesic_wxyz(a_quat, b_quat)
                    bc_geo = geodesic_wxyz(b_quat, c_quat)
                    b_rot_is_finite = all(math.isfinite(float(x)) for x in b_quat)

                pos_limit = BODY_POS_LIMIT if etype == "body" else (
                    SITE_POS_LIMIT if etype == "site" else GEOM_POS_LIMIT)
                rot_limit = BODY_ROT_LIMIT if etype == "body" else (
                    SITE_ROT_LIMIT if etype == "site" else GEOM_ROT_LIMIT)

                b_pos_finite = all(math.isfinite(float(x)) for x in b_pos)
                source_finite = all(math.isfinite(float(x)) for x in qpos_before)

                records.append({
                    "suite": suite, "task_idx": task_idx, "state_id": state_id,
                    "step": step,
                    "entity_type": etype, "entity_id": int(eid),
                    "entity_name": info["name"],
                    "semantic_role": info["semantic_role"],
                    "resolution": info["resolution"],
                    "AB_pos_Linf": ab_pos_err, "AB_rot_err": ab_geo,
                    "BC_pos_Linf": bc_pos_err, "BC_rot_err": bc_geo,
                    "BC_pos_pass": bc_pos_err <= VERIFICATION_LIMIT,
                    "BC_rot_pass": bc_geo <= VERIFICATION_LIMIT,
                    "AB_stale": ab_pos_err > VERIFICATION_LIMIT,
                    "pos_limit": pos_limit, "rot_limit": rot_limit,
                    "source_mutated_fwd1": source_mutated_1,
                    "source_mutated_fwd2": source_mutated_2,
                    "nonfinite_pose": not (b_pos_finite and b_rot_is_finite),
                    "nonfinite_source": not source_finite,
                    "fwd1_qpos_drift": qpos_drift1, "fwd1_qvel_drift": qvel_drift1,
                    "fwd1_act_drift": act_drift1, "fwd1_time_drift": time_drift1,
                    "fwd2_qpos_drift": qpos_drift2, "fwd2_qvel_drift": qvel_drift2,
                    "fwd2_act_drift": act_drift2, "fwd2_time_drift": time_drift2,
                })

            # Advance
            action = apply_actions[step % len(apply_actions)]
            env.step(action)
    finally:
        env.close()

    n = len(records)
    bc_pos_fail = sum(1 for r in records if not r["BC_pos_pass"])
    bc_rot_fail = sum(1 for r in records if not r["BC_rot_pass"])
    ab_stale = sum(1 for r in records if r["AB_stale"])
    src_mut = sum(1 for r in records if (r["source_mutated_fwd1"] or r["source_mutated_fwd2"]))
    nonfin = sum(1 for r in records if (r["nonfinite_pose"] or r["nonfinite_source"]))

    # Entity identity closure
    recorded_keys = set((r["entity_type"], r["entity_id"]) for r in records)
    expected_keys = set(expected_entities.keys())
    closure_ok = recorded_keys == expected_keys

    summary = {
        "suite": suite, "task_idx": task_idx, "state_id": state_id,
        "seed": seed, "test_steps": test_steps,
        "n_entities": len(expected_entities), "n_records": n,
        "BC_pos_fail": bc_pos_fail, "BC_rot_fail": bc_rot_fail,
        "AB_stale_count": ab_stale,
        "source_mutations": src_mut,
        "nonfinite": nonfin,
        "entity_closure_ok": closure_ok,
        "registry_closure": closure,
        "status": "PASS" if (bc_pos_fail == 0 and bc_rot_fail == 0 and
                             src_mut == 0 and nonfin == 0 and closure_ok) else "FAIL",
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output seal root")
    parser.add_argument("--registry", required=True, help="C1-V2 run_A/per_task directory")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--state", type=int, default=0, help="State index (0-based)")
    parser.add_argument("--mode", choices=["formal", "smoke"], default="formal",
                        help="formal = full 40/40 (consumable). smoke = subset only (NONCONSUMABLE).")
    parser.add_argument("--suites", nargs="*", default=None,
                        help="Subset suites (only valid with --mode smoke)")
    args = parser.parse_args()

    is_smoke = args.mode == "smoke"
    suites_to_run = args.suites if args.suites else FOUR_SUITES
    if not is_smoke:
        if args.suites is not None:
            raise SystemExit("--suites is only valid with --mode smoke")
        suites_to_run = FOUR_SUITES
        expected_task_count = 40
    else:
        expected_task_count = len(suites_to_run) * 10

    out = Path(args.out).resolve()
    if out.exists() or out.is_symlink():
        raise SystemExit(f"output exists: {out}")
    staging = out.parent / f".{out.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if staging.exists():
        raise SystemExit(f"staging exists: {staging}")
    staging.mkdir(parents=True)

    # Source provenance
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    repo_root = script_path.parent.parent.parent
    source_commit = git_value(repo_root, "rev-parse", "HEAD")
    source_tree = git_value(repo_root, "rev-parse", "HEAD^{tree}")
    protocol_path = repo_root / "reports" / "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE.json"
    protocol_sha = sha256_file(protocol_path) if protocol_path.is_file() else "MISSING"

    # Action sequences: zero actions then varied actions for physics interaction
    zero_actions = [[0.0]*7] * 3
    varied_actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    ]
    apply_actions = zero_actions + varied_actions
    while len(apply_actions) < args.steps:
        apply_actions.append([0.0]*7)
    apply_actions = apply_actions[:args.steps]

    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_epoch = time.time()

    print("=" * 70)
    print(f"[DeepSeek] R5-E: Formal Corrected Same-Live Gate")
    print(f"  mode={args.mode} suites={suites_to_run} steps={args.steps} seed={args.seed}")
    print(f"  state_id={args.state}")
    print(f"  registry={args.registry}")
    print(f"  source_commit={source_commit}")
    print("=" * 70)

    all_records = []
    all_summaries = []
    tested_tasks = set()
    expected_task_set = set(f"{s}/task_{tid:02d}" for s in suites_to_run for tid in range(10))
    skipped = 0

    for suite in suites_to_run:
        for task_idx in range(10):
            task_key = f"{suite}/task_{task_idx:02d}"
            print(f"\n  {task_key}/state_{args.state}...", end=" ", flush=True)
            records, summary = test_one_task(
                suite, task_idx, args.state, args.seed, args.steps,
                args.registry, apply_actions)
            if records is None:
                print(f"SKIP: {summary.get('error', '?')}")
                skipped += 1
                all_summaries.append({"task_key": task_key, "status": "SKIP",
                                      "error": summary.get("error", ""),
                                      "closure": summary.get("closure")})
                continue
            all_records.extend(records)
            all_summaries.append(summary)
            tested_tasks.add(task_key)
            ok = summary["status"] == "PASS"
            print(f"entities={summary['n_entities']} records={summary['n_records']} "
                  f"BC_fail={summary['BC_pos_fail']}/{summary['BC_rot_fail']} "
                  f"stale={summary['AB_stale_count']} mut={summary['source_mutations']} "
                  f"{'PASS' if ok else 'FAIL'}")

    n_tested = len(tested_tasks)
    total_bc_pos_fail = sum(s.get("BC_pos_fail", 0) for s in all_summaries)
    total_bc_rot_fail = sum(s.get("BC_rot_fail", 0) for s in all_summaries)
    total_stale = sum(s.get("AB_stale_count", 0) for s in all_summaries)
    total_mutations = sum(s.get("source_mutations", 0) for s in all_summaries)
    total_nonfinite = sum(s.get("nonfinite", 0) for s in all_summaries)
    entity_closure_ok = all(s.get("entity_closure_ok", False) for s in all_summaries
                            if s.get("status") != "SKIP")
    all_finite = total_nonfinite == 0

    # Strict gate: exactly 40 tested, skip=0, all PASS
    identity_ok = (tested_tasks == expected_task_set and skipped == 0)
    gate_pass = (
        identity_ok
        and n_tested == expected_task_count
        and total_bc_pos_fail == 0
        and total_bc_rot_fail == 0
        and total_mutations == 0
        and total_nonfinite == 0
        and entity_closure_ok
    )

    print(f"\n{'=' * 70}")
    print(f"Tasks tested: {n_tested}/{expected_task_count}  Skipped: {skipped}")
    print(f"Total records: {len(all_records)}")
    print(f"Identity closure: {'PASS' if identity_ok else 'FAIL'}")
    print(f"Entity closure: {'PASS' if entity_closure_ok else 'FAIL'}")
    print(f"B→C position failures: {total_bc_pos_fail}")
    print(f"B→C rotation failures: {total_bc_rot_fail}")
    print(f"Source state mutations: {total_mutations}")
    print(f"Nonfinite: {total_nonfinite}")
    print(f"A→B stale reads (diagnostic): {total_stale}")
    status_label = "SAME_LIVE_GATE_PASS" if gate_pass else (
        "SMOKE_NONCONSUMABLE_PASS" if (is_smoke and identity_ok and
            total_bc_pos_fail == 0 and total_bc_rot_fail == 0 and
            total_mutations == 0 and total_nonfinite == 0)
        else "SAME_LIVE_GATE_FAIL")
    print(f"VERDICT: {status_label}")

    end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed = time.time() - start_epoch

    # Registry manifest binding
    registry_manifest = {}
    registry_manifest_path = Path(args.registry).parent / "ENTITY_REGISTRY_V2_SUMMARY.json"
    if registry_manifest_path.is_file():
        registry_manifest = {
            "path": str(registry_manifest_path.resolve()),
            "sha256": sha256_file(registry_manifest_path),
        }

    # Write evidence
    manifest = {
        "gate": "R5-E_FORMAL_SAME_LIVE_GATE",
        "schema": "G_REC_SAME_LIVE_GATE_V2",
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "protocol_amendment_sha256": protocol_sha,
        "status": status_label,
        "mode": args.mode,
        "consumer_eligible": gate_pass and not is_smoke,
        "start_time": start_time, "end_time": end_time, "elapsed_s": elapsed,
        "source_commit": source_commit, "source_tree": source_tree,
        "script_sha256": script_sha,
        "registry_manifest": registry_manifest,
        "n_tasks_tested": n_tested, "n_tasks_expected": expected_task_count,
        "n_tasks_skipped": skipped,
        "tested_task_identities": sorted(tested_tasks),
        "expected_task_identities": sorted(expected_task_set),
        "identity_closure_ok": identity_ok,
        "total_records": len(all_records),
        "BC_pos_fail": total_bc_pos_fail, "BC_rot_fail": total_bc_rot_fail,
        "source_mutations": total_mutations,
        "nonfinite": total_nonfinite,
        "entity_closure_ok": entity_closure_ok,
        "AB_stale_diagnostic": total_stale,
        "thresholds": {
            "verification_limit": VERIFICATION_LIMIT,
            "body_pos": BODY_POS_LIMIT, "body_rot": BODY_ROT_LIMIT,
            "site_pos": SITE_POS_LIMIT, "site_rot": SITE_ROT_LIMIT,
            "geom_pos": GEOM_POS_LIMIT, "geom_rot": GEOM_ROT_LIMIT,
        },
        "suites_run": suites_to_run,
        "state_id": args.state,
        "steps_per_task": args.steps,
        "seed": args.seed,
        "python_version": sys.version,
        "executable": sys.executable,
        "command": sys.argv,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with open(staging / "case_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(staging / "per_task_summary.jsonl", "w") as f:
        for s in all_summaries:
            f.write(json.dumps(s, sort_keys=True, default=str) + "\n")

    # Seal
    sums = {}
    for fn in ["MANIFEST.json", "case_records.jsonl", "per_task_summary.jsonl"]:
        sums[fn] = sha256_file(staging / fn)
    with open(staging / "SHA256SUMS", "w") as f:
        for fn, s in sorted(sums.items()):
            f.write(f"{s}  {fn}\n")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    with open(staging / "SHA256SUMS.sha256", "w") as f:
        f.write(f"{sums_sha}  SHA256SUMS\n")

    # Atomic no-replace publication
    try:
        staging.rename(out)
    except OSError:
        raise SystemExit(f"rename failed — output may have appeared: {out}")

    print(f"\nSealed: {out}")
    print(f"  SHA256SUMS: {sums_sha}")
    return 0 if gate_pass else (0 if (is_smoke and identity_ok) else 5)


if __name__ == "__main__":
    raise SystemExit(main())
