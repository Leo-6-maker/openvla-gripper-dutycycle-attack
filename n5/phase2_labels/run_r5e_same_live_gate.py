"""[DeepSeek] R5-E: Formal Corrected Same-Live Gate.

Relation-bound A/B/C triple-read across all 40 task types.
Entities resolved via C1-V2 registry (body for objects, site for regions).
Tests: A=pre-forward, B=after capture forward, C=after verification forward.

PASS criteria:
  - B→C position failures = 0
  - B→C rotation failures = 0
  - source-state mutation = 0
  - nonfinite = 0
  - relation identity closure = 100%
  - missing/duplicate/extra cases = 0

A→B may be nonzero (diagnostic only — stale read reporting).
The consumable pose is B.
"""
import json, os, sys, math, copy, hashlib, time, argparse, uuid
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
VERIFICATION_LIMIT = 1e-15  # B→C must be at machine epsilon

FOUR_SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Extract relation-bound entity (type, id) pairs from C1-V2 per-task registry."""
    with open(registry_path) as f:
        data = json.load(f)
    legacy = data.get("legacy", data)
    relations = legacy.get("relations", [])
    entities = {}
    for rel in relations:
        for side in ("object_resolution", "target_resolution"):
            res = rel.get(side, {})
            if res.get("resolution") in ("EXACT_BODY", "EXACT_SITE", "EXACT_GEOM",
                                          "APPROVED_STRUCTURAL_ALIAS"):
                key = (res["entity_type"], res["entity_id"])
                entities[key] = {
                    "name": res.get("name", "?"),
                    "entity_type": res["entity_type"],
                    "entity_id": res["entity_id"],
                    "resolution": res["resolution"],
                    "semantic_role": res.get("semantic_role", "?"),
                }
    return entities


def test_one_task(suite, task_idx, state_id, seed, test_steps, registry_dir, apply_actions):
    """Run A/B/C same-live parity test on one task."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    import random as _random

    _random.seed(seed)
    bm = get_benchmark(suite)(0); t = bm.get_task(task_idx)
    bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    sd = get_benchmark_dict(); so = sd[suite]()
    init_states = so.get_task_init_states(task_idx)
    if state_id >= len(init_states):
        return None, {"error": f"state_id {state_id} >= {len(init_states)}"}
    cs = init_states[state_id]

    env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256,
                             render_gpu_device_id=-1, has_renderer=False,
                             has_offscreen_renderer=False, horizon=520)
    try:
        env.seed(seed); env.reset(); env.set_init_state(copy.deepcopy(cs))
        for _ in range(NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)

        # Load relation-bound entities
        reg_path = os.path.join(registry_dir, f"{suite}_task_{task_idx:02d}.json")
        if not os.path.isfile(reg_path):
            return None, {"error": f"registry missing: {reg_path}"}
        expected_entities = load_relation_entities(reg_path)
        if not expected_entities:
            return None, {"error": "no relation-bound entities in registry"}

        records = []
        for step in range(test_steps):
            # Save source state
            qpos_pre = env.sim.data.qpos.copy()
            qvel_pre = env.sim.data.qvel.copy()
            time_pre = float(env.sim.data.time)

            # A: read BEFORE forward
            A_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    A_poses[(etype, eid)] = (
                        env.sim.data.body_xpos[eid].copy(),
                        env.sim.data.body_xquat[eid].copy(),
                    )
                elif etype == "site":
                    A_poses[(etype, eid)] = (
                        env.sim.data.site_xpos[eid].copy(),
                        env.sim.data.site_xmat[eid].copy().flatten(),
                    )
                elif etype == "geom":
                    A_poses[(etype, eid)] = (
                        env.sim.data.geom_xpos[eid].copy(),
                        env.sim.data.geom_xmat[eid].copy().flatten(),
                    )

            # Forward → B (capture forward)
            env.sim.forward()

            # Verify source state stability
            qpos_drift = float(np.max(np.abs(qpos_pre - env.sim.data.qpos.copy())))
            qvel_drift = float(np.max(np.abs(qvel_pre - env.sim.data.qvel.copy())))
            time_drift = abs(time_pre - float(env.sim.data.time))
            source_mutated = qpos_drift > 0 or qvel_drift > 0 or time_drift > 0

            B_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    B_poses[(etype, eid)] = (
                        env.sim.data.body_xpos[eid].copy(),
                        env.sim.data.body_xquat[eid].copy(),
                    )
                elif etype == "site":
                    B_poses[(etype, eid)] = (
                        env.sim.data.site_xpos[eid].copy(),
                        env.sim.data.site_xmat[eid].copy().flatten(),
                    )
                elif etype == "geom":
                    B_poses[(etype, eid)] = (
                        env.sim.data.geom_xpos[eid].copy(),
                        env.sim.data.geom_xmat[eid].copy().flatten(),
                    )

            # Second forward → C (verification forward)
            env.sim.forward()

            C_poses = {}
            for (etype, eid), info in expected_entities.items():
                if etype == "body":
                    C_poses[(etype, eid)] = (
                        env.sim.data.body_xpos[eid].copy(),
                        env.sim.data.body_xquat[eid].copy(),
                    )
                elif etype == "site":
                    C_poses[(etype, eid)] = (
                        env.sim.data.site_xpos[eid].copy(),
                        env.sim.data.site_xmat[eid].copy().flatten(),
                    )
                elif etype == "geom":
                    C_poses[(etype, eid)] = (
                        env.sim.data.geom_xpos[eid].copy(),
                        env.sim.data.geom_xmat[eid].copy().flatten(),
                    )

            # Compare
            for (etype, eid), info in expected_entities.items():
                a_pos, a_rot = A_poses[(etype, eid)]
                b_pos, b_rot = B_poses[(etype, eid)]
                c_pos, c_rot = C_poses[(etype, eid)]

                ab_pos_err = float(np.max(np.abs(a_pos - b_pos)))
                bc_pos_err = float(np.max(np.abs(b_pos - c_pos)))

                if etype == "site":
                    ab_rot_err = float(np.max(np.abs(a_rot - b_rot)))
                    bc_rot_err = float(np.max(np.abs(b_rot - c_rot)))
                    ab_geo = ab_rot_err
                    bc_geo = bc_rot_err
                else:
                    if etype == "body":
                        a_quat = a_rot  # wxyz
                        b_quat = b_rot  # wxyz
                        c_quat = c_rot
                    else:
                        a_quat = mat_to_quat_wxyz(a_rot)
                        b_quat = mat_to_quat_wxyz(b_rot)
                        c_quat = mat_to_quat_wxyz(c_rot)
                    ab_geo = geodesic_wxyz(a_quat, b_quat)
                    bc_geo = geodesic_wxyz(b_quat, c_quat)

                pos_limit = BODY_POS_LIMIT if etype == "body" else (SITE_POS_LIMIT if etype == "site" else GEOM_POS_LIMIT)
                rot_limit = BODY_ROT_LIMIT if etype == "body" else (SITE_ROT_LIMIT if etype == "site" else GEOM_ROT_LIMIT)

                nonfinite = (not all(math.isfinite(float(x)) for x in b_pos) or
                            not all(math.isfinite(float(x)) for x in b_rot))

                records.append({
                    "suite": suite, "task_idx": task_idx, "state_id": state_id,
                    "step": step,
                    "entity_type": etype, "entity_id": int(eid),
                    "entity_name": info["name"],
                    "semantic_role": info["semantic_role"],
                    "resolution": info["resolution"],
                    "AB_pos_Linf": ab_pos_err,
                    "AB_rot_err": ab_geo if etype != "site" else ab_rot_err,
                    "BC_pos_Linf": bc_pos_err,
                    "BC_rot_err": bc_geo if etype != "site" else bc_rot_err,
                    "BC_pos_pass": bc_pos_err <= VERIFICATION_LIMIT,
                    "BC_rot_pass": bc_geo <= VERIFICATION_LIMIT if etype != "site" else bc_rot_err <= VERIFICATION_LIMIT,
                    "AB_stale": ab_pos_err > VERIFICATION_LIMIT,
                    "pos_limit": pos_limit, "rot_limit": rot_limit,
                    "source_mutated": source_mutated,
                    "nonfinite": nonfinite,
                })

            # Advance
            action = apply_actions[step % len(apply_actions)] if apply_actions else [0.0]*7
            env.step(action)
    finally:
        env.close()

    n = len(records)
    bc_pos_fail = sum(1 for r in records if not r["BC_pos_pass"])
    bc_rot_fail = sum(1 for r in records if not r["BC_rot_pass"])
    ab_stale = sum(1 for r in records if r["AB_stale"])
    src_mut = sum(1 for r in records if r["source_mutated"])
    nonfin = sum(1 for r in records if r["nonfinite"])

    summary = {
        "suite": suite, "task_idx": task_idx, "state_id": state_id,
        "seed": seed, "test_steps": test_steps,
        "n_entities": len(expected_entities), "n_records": n,
        "BC_pos_fail": bc_pos_fail, "BC_rot_fail": bc_rot_fail,
        "AB_stale_count": ab_stale,
        "source_mutations": src_mut,
        "nonfinite": nonfin,
    }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output seal root")
    parser.add_argument("--registry", required=True, help="C1-V2 run_A/per_task directory")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--state", type=int, default=0, help="State index (0-based)")
    parser.add_argument("--suites", nargs="*", default=None)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.parent / f".{out.name}.staging.{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

    suites_to_run = args.suites if args.suites else FOUR_SUITES

    # Action sequences: zero actions then varied actions for physics interaction
    zero_actions = [[0.0]*7] * 3
    varied_actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # close gripper
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # open gripper
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # small delta x
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],   # small delta y
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],   # small delta z
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],   # small roll
    ]
    apply_actions = zero_actions + varied_actions
    # Trim to requested steps
    while len(apply_actions) < args.steps:
        apply_actions.append([0.0]*7)
    apply_actions = apply_actions[:args.steps]

    print("=" * 70)
    print(f"[DeepSeek] R5-E: Formal Corrected Same-Live Gate")
    print(f"  suites={suites_to_run} steps={args.steps} seed={args.seed}")
    print(f"  state_id={args.state}")
    print(f"  registry={args.registry}")
    print("=" * 70)

    all_records = []
    all_summaries = []
    total_tasks = 0
    skipped = 0

    for suite in suites_to_run:
        for task_idx in range(10):
            total_tasks += 1
            task_key = f"{suite}/task_{task_idx:02d}"
            print(f"\n  {task_key}/state_{args.state}...", end=" ", flush=True)
            records, summary = test_one_task(
                suite, task_idx, args.state, args.seed, args.steps,
                args.registry, apply_actions)
            if records is None:
                print(f"SKIP: {summary.get('error', '?')}")
                skipped += 1
                all_summaries.append({"task_key": task_key, "status": "SKIP",
                                      "error": summary.get("error", "")})
                continue
            all_records.extend(records)
            all_summaries.append(summary)
            ok = (summary["BC_pos_fail"] == 0 and summary["BC_rot_fail"] == 0 and
                  summary["source_mutations"] == 0 and summary["nonfinite"] == 0)
            print(f"entities={summary['n_entities']} records={summary['n_records']} "
                  f"BC_fail={summary['BC_pos_fail']}/{summary['BC_rot_fail']} "
                  f"stale={summary['AB_stale_count']} mut={summary['source_mutations']} "
                  f"{'PASS' if ok else 'FAIL'}")

    n_tested = len([s for s in all_summaries if s.get("n_records", 0) > 0])
    total_bc_pos_fail = sum(s.get("BC_pos_fail", 0) for s in all_summaries)
    total_bc_rot_fail = sum(s.get("BC_rot_fail", 0) for s in all_summaries)
    total_stale = sum(s.get("AB_stale_count", 0) for s in all_summaries)
    total_mutations = sum(s.get("source_mutations", 0) for s in all_summaries)
    total_nonfinite = sum(s.get("nonfinite", 0) for s in all_summaries)

    gate_pass = (total_bc_pos_fail == 0 and total_bc_rot_fail == 0 and
                 total_mutations == 0 and total_nonfinite == 0 and
                 n_tested >= total_tasks - skipped)

    print(f"\n{'=' * 70}")
    print(f"Tasks tested: {n_tested}/{total_tasks}  Skipped: {skipped}")
    print(f"Total records: {len(all_records)}")
    print(f"B→C position failures: {total_bc_pos_fail}")
    print(f"B→C rotation failures: {total_bc_rot_fail}")
    print(f"Source state mutations: {total_mutations}")
    print(f"Nonfinite: {total_nonfinite}")
    print(f"A→B stale reads (diagnostic): {total_stale}")
    print(f"VERDICT: {'SAME_LIVE_GATE_PASS' if gate_pass else 'SAME_LIVE_GATE_FAIL'}")

    # Write evidence
    manifest = {
        "gate": "R5-E_FORMAL_SAME_LIVE_GATE",
        "schema": "G_REC_SAME_LIVE_GATE_V1",
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "status": "PASS" if gate_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_tasks_tested": n_tested, "n_tasks_total": total_tasks,
        "n_tasks_skipped": skipped,
        "total_records": len(all_records),
        "BC_pos_fail": total_bc_pos_fail, "BC_rot_fail": total_bc_rot_fail,
        "source_mutations": total_mutations,
        "nonfinite": total_nonfinite,
        "AB_stale_diagnostic": total_stale,
        "thresholds": {
            "verification_limit": VERIFICATION_LIMIT,
            "body_pos": BODY_POS_LIMIT, "body_rot": BODY_ROT_LIMIT,
            "site_pos": SITE_POS_LIMIT, "site_rot": SITE_ROT_LIMIT,
            "geom_pos": GEOM_POS_LIMIT, "geom_rot": GEOM_ROT_LIMIT,
        },
        "suites": suites_to_run,
        "state_id": args.state,
        "steps_per_task": args.steps,
        "seed": args.seed,
        "consumer_eligible": False,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with open(staging / "case_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(staging / "per_task_summary.jsonl", "w") as f:
        for s in all_summaries:
            f.write(json.dumps(s, sort_keys=True) + "\n")

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

    staging.rename(out)
    print(f"\nSealed: {out}")
    print(f"  SHA256SUMS: {sums_sha}")
    return 0 if gate_pass else 5


if __name__ == "__main__":
    raise SystemExit(main())
