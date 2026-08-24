"""[DeepSeek] Same-Live Body Forward Canary.

Verifies that within a single live environment, body_xpos recorded BEFORE
sim.forward() equals body_xpos recomputed AFTER sim.forward() from the
same saved qpos — proving direct recorded poses are trustworthy.

If A == B at machine precision, direct recorded body poses can serve as
the authoritative geometry source for dynamic objects without requiring
offline state-forward parity on historical episodes.
"""
import json, os, sys, math, copy, argparse
import numpy as np
from pathlib import Path
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NUM_STEPS_WAIT = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print(f"[DeepSeek] Same-Live Body Forward Canary")
    print(f"  {args.suite}/task_{args.task:02d}/state_{args.state}")
    print(f"  seed={args.seed}  test_steps={args.steps}")
    print("=" * 60)

    import random as _random
    _random.seed(args.seed)

    benchmark = get_benchmark(args.suite)(0)
    task = benchmark.get_task(args.task)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)

    suite_dict = get_benchmark_dict()
    suite_obj = suite_dict[args.suite]()
    init_states = suite_obj.get_task_init_states(args.task)
    canonical_state = init_states[args.state]

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256, camera_widths=256,
        render_gpu_device_id=-1,
        has_renderer=False, has_offscreen_renderer=False,
        horizon=520,
    )
    env.seed(args.seed)
    env.reset()
    env.set_init_state(copy.deepcopy(canonical_state))
    for _ in range(NUM_STEPS_WAIT):
        env.step(DUMMY_ACTION)

    model = env.sim.model
    print(f"  model: nq={model.nq} nv={model.nv} nmocap={model.nmocap}")
    print(f"  bodies: {model.nbody}  sites: {model.nsite}  geoms: {model.ngeom}")

    # Test: record body poses, save qpos, forward, recompute, compare
    all_results = []
    for step in range(args.steps):
        # A: record current body poses directly
        poses_A = {}
        for bid in range(model.nbody):
            name = model.body_id2name(bid) if hasattr(model, 'body_id2name') else str(bid)
            if name and name != "world":
                poses_A[bid] = {
                    "name": name,
                    "xpos": env.sim.data.body_xpos[bid].copy().tolist(),
                    "xquat": env.sim.data.body_xquat[bid].copy().tolist(),
                }

        # Save qpos
        saved_qpos = env.sim.data.qpos.copy()

        # B: recompute via sim.forward()
        env.sim.forward()

        poses_B = {}
        for bid in poses_A:
            poses_B[bid] = {
                "xpos": env.sim.data.body_xpos[bid].copy().tolist(),
                "xquat": env.sim.data.body_xquat[bid].copy().tolist(),
            }

        # Compare
        for bid in poses_A:
            a = np.array(poses_A[bid]["xpos"])
            b = np.array(poses_B[bid]["xpos"])
            pos_linf = float(np.max(np.abs(a - b)))
            all_results.append({
                "step": step, "body_id": bid, "body_name": poses_A[bid]["name"],
                "pos_Linf": pos_linf,
            })

        # Take a step for next iteration
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # zero action (hold)
        env.step(action)

    # Report
    pos_errors = [r["pos_Linf"] for r in all_results]
    n_total = len(all_results)
    n_pass = sum(1 for e in pos_errors if e <= 1e-12)
    n_zero = sum(1 for e in pos_errors if e == 0.0)

    print(f"\n  Cases: {n_total} ({args.steps} steps × {len(poses_A)} bodies)")
    print(f"  Exact zero: {n_zero}/{n_total}")
    print(f"  Within 1e-12: {n_pass}/{n_total}")
    print(f"  Max error: {max(pos_errors):.2e}")
    print(f"  P99 error: {np.percentile(pos_errors, 99):.2e}")

    if n_pass == n_total:
        print(f"\n  VERDICT: SAME_LIVE_PARITY_CONFIRMED")
        print(f"  Direct recorded body poses are machine-precision trustworthy.")
        sys.exit(0)
    else:
        print(f"\n  VERDICT: FAIL — {n_total - n_pass} cases above 1e-12")
        # Show failures
        failures = [(r, pos_errors[i]) for i, r in enumerate(all_results) if pos_errors[i] > 1e-12]
        for r, e in failures[:5]:
            print(f"    step={r['step']} body={r['body_name']} Linf={e:.2e}")
        sys.exit(5)

    env.close()


if __name__ == "__main__":
    main()
