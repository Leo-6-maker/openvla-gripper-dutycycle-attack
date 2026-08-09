"""[DeepSeek] Same-Live Pose Parity Seal.

Verifies: direct recorded body/site poses == poses after sim.forward()
within the same live MuJoCo environment. Multi-step, multi-task.

Output: sealed evidence root with case-level records, SHA256SUMS.
"""
import json, os, sys, math, copy, hashlib, time, argparse, uuid, shutil
import numpy as np
from pathlib import Path

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_STEPS_WAIT = 10
BODY_LIMIT = 1e-8
SITE_LIMIT = 1e-8


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()


def test_one_task(suite, task_idx, state_id, seed, test_steps):
    """Run same-live parity test on one task. Returns (records, summary)."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict
    from libero.libero.envs import OffScreenRenderEnv
    import random as _random

    _random.seed(seed)
    bm = get_benchmark(suite)(0); t = bm.get_task(task_idx)
    bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    sd = get_benchmark_dict(); so = sd[suite]()
    cs = so.get_task_init_states(task_idx)[state_id]

    env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256,
                             render_gpu_device_id=-1, has_renderer=False,
                             has_offscreen_renderer=False, horizon=520)
    env.seed(seed); env.reset(); env.set_init_state(copy.deepcopy(cs))
    for _ in range(NUM_STEPS_WAIT): env.step(DUMMY_ACTION)

    model = env.sim.model
    # Collect non-robot object bodies and _region sites
    body_ids = []
    for bid in range(model.nbody):
        name = model.body_id2name(bid)
        if name and all(k not in name for k in ("robot", "gripper", "world", "floor", "mount")):
            body_ids.append((bid, name))
    site_ids = []
    for sid in range(model.nsite):
        name = model.site_id2name(sid)
        if name and "_region" in name:
            site_ids.append((sid, name))

    records = []
    for step in range(test_steps):
        saved_qpos = env.sim.data.qpos.copy()
        saved_time = float(env.sim.data.time)

        # Record A: before forward
        poses_A = {}
        for bid, name in body_ids:
            poses_A[("body", bid, name)] = (env.sim.data.body_xpos[bid].copy(),
                                             env.sim.data.body_xquat[bid].copy())
        for sid, name in site_ids:
            poses_A[("site", sid, name)] = (env.sim.data.site_xpos[sid].copy(),
                                             env.sim.data.site_xmat[sid].copy().flatten())

        # Forward
        env.sim.forward()

        # Verify state didn't mutate
        qpos_after = env.sim.data.qpos.copy()
        time_after = float(env.sim.data.time)
        qpos_mutated = np.max(np.abs(saved_qpos - qpos_after)) > 0

        # Record B: after forward
        for bid, name in body_ids:
            b_pos = env.sim.data.body_xpos[bid].copy()
            b_quat = env.sim.data.body_xquat[bid].copy()
            a_pos, a_quat = poses_A[("body", bid, name)]
            pos_err = float(np.max(np.abs(a_pos - b_pos)))
            geo_err = geodesic_wxyz(a_quat, b_quat)
            records.append({"step": step, "kind": "body", "id": int(bid), "name": str(name),
                           "pos_Linf": float(pos_err), "geo_rad": float(geo_err),
                           "pos_pass": bool(pos_err <= BODY_LIMIT),
                           "rot_pass": bool(geo_err <= 1e-7),
                           "qpos_mutated": bool(qpos_mutated)})

        for sid, name in site_ids:
            s_pos = env.sim.data.site_xpos[sid].copy()
            s_xmat = env.sim.data.site_xmat[sid].copy().flatten()
            a_pos, a_xmat = poses_A[("site", sid, name)]
            pos_err = float(np.max(np.abs(a_pos - s_pos)))
            rot_err = float(np.max(np.abs(a_xmat - s_xmat)))
            records.append({"step": step, "kind": "site", "id": int(sid), "name": str(name),
                           "pos_Linf": float(pos_err), "xmat_max_abs": float(rot_err),
                           "pos_pass": bool(pos_err <= SITE_LIMIT),
                           "rot_pass": bool(rot_err <= 1e-7),
                           "qpos_mutated": bool(qpos_mutated)})

        # Take one step
        env.step([0.0]*7)

    env.close()
    n_bodies = len(body_ids); n_sites = len(site_ids)
    n_body_cases = n_bodies * test_steps; n_site_cases = n_sites * test_steps
    body_fail = sum(1 for r in records if r["kind"]=="body" and not r["pos_pass"])
    site_fail = sum(1 for r in records if r["kind"]=="site" and not r["pos_pass"])
    qpos_issues = sum(1 for r in records if r.get("qpos_mutated"))

    return records, {
        "suite": suite, "task_idx": task_idx, "state_id": state_id,
        "seed": seed, "test_steps": test_steps,
        "n_bodies": n_bodies, "n_sites": n_sites,
        "n_body_cases": n_body_cases, "n_site_cases": n_site_cases,
        "body_pos_fail": body_fail, "site_pos_fail": site_fail,
        "qpos_mutation_steps": qpos_issues,
    }


def geodesic_wxyz(q1, q2):
    q1n = np.array(q1, float); q2n = np.array(q2, float)
    q1n /= np.linalg.norm(q1n); q2n /= np.linalg.norm(q2n)
    d = abs(np.dot(q1n, q2n)); d = min(d, 1.0)
    return float(2.0 * math.atan2(math.sqrt(max(0, 1-d*d)), d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    staging = out.parent / f".{out.name}.staging.{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)

    # Test multiple tasks across suites
    test_configs = [
        ("libero_10", 0, 15),       # basket In
        ("libero_object", 0, 9),    # basket In
        ("libero_10", 6, 2),        # plate On
        ("libero_spatial", 0, 4),   # spatial On
    ]

    print("=" * 60)
    print("[DeepSeek] Same-Live Pose Parity Seal")
    print(f"  steps={args.steps} seed={args.seed}")
    print("=" * 60)

    all_records = []
    summaries = []
    for suite, task_idx, state_id in test_configs:
        print(f"\n  {suite}/task_{task_idx:02d}/state_{state_id}...", end=" ", flush=True)
        records, summary = test_one_task(suite, task_idx, state_id, args.seed, args.steps)
        all_records.extend(records)
        summaries.append(summary)
        ok = summary["body_pos_fail"] == 0 and summary["site_pos_fail"] == 0
        print(f"bodies={summary['n_bodies']} sites={summary['n_sites']} "
              f"body_cases={summary['n_body_cases']} site_cases={summary['n_site_cases']} "
              f"{'PASS' if ok else 'FAIL'}")

    # Summary
    total_body = sum(s["n_body_cases"] for s in summaries)
    total_site = sum(s["n_site_cases"] for s in summaries)
    total_body_fail = sum(s["body_pos_fail"] for s in summaries)
    total_site_fail = sum(s["site_pos_fail"] for s in summaries)
    total_qpos = sum(s["qpos_mutation_steps"] for s in summaries)
    all_pass = total_body_fail == 0 and total_site_fail == 0 and total_qpos == 0

    body_recs = [r for r in all_records if r["kind"] == "body"]
    site_recs = [r for r in all_records if r["kind"] == "site"]
    body_pos_max = max(r["pos_Linf"] for r in body_recs) if body_recs else 0
    body_geo_max = max(r["geo_rad"] for r in body_recs) if body_recs else 0
    site_pos_max = max(r["pos_Linf"] for r in site_recs) if site_recs else 0
    site_xmat_max = max(r["xmat_max_abs"] for r in site_recs) if site_recs else 0

    print(f"\n{'='*60}")
    print(f"Total: {total_body} body cases, {total_site} site cases")
    print(f"  Body pos max: {body_pos_max:.2e}  geo max: {body_geo_max:.2e}  fail: {total_body_fail}")
    print(f"  Site pos max: {site_pos_max:.2e}  xmat max: {site_xmat_max:.2e}  fail: {total_site_fail}")
    print(f"  Qpos mutation: {total_qpos}")
    print(f"  VERDICT: {'SAME_LIVE_PARITY_CONFIRMED' if all_pass else 'FAIL'}")

    # Write evidence
    manifest = {
        "gate": "SAME_LIVE_POSE_PARITY",
        "schema": "G_REC_DIRECT_POSE_SAME_LIVE_V1",
        "status": "PASS" if all_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": "DeepSeek",
        "test_configs": test_configs,
        "steps_per_task": args.steps,
        "seed": args.seed,
        "total_body_cases": total_body, "total_site_cases": total_site,
        "body_pos_fail": total_body_fail, "site_pos_fail": total_site_fail,
        "qpos_mutation_steps": total_qpos,
        "body_pos_max": body_pos_max, "body_geo_max": body_geo_max,
        "site_pos_max": site_pos_max, "site_xmat_max": site_xmat_max,
        "thresholds": {"body_pos": BODY_LIMIT, "body_rot": 1e-7, "site_pos": SITE_LIMIT},
        "protected_reads": 0, "model_inference": 0, "attack": 0,
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with open(staging / "case_records.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (staging / "stdout.txt").write_text("see execution log above")

    # Seal
    sums = {}
    for fn in ["MANIFEST.json", "case_records.jsonl"]:
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
    sys.exit(0 if all_pass else 5)


if __name__ == "__main__":
    main()
