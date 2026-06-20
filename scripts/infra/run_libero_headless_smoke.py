#!/usr/bin/env python3
"""MIG3A: LIBERO headless render smoke — no OpenVLA model loaded."""
import os, json, hashlib, time, numpy as np, argparse
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def run_smoke(gpu=6, seed=42, output=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    bench_dict = benchmark.get_benchmark_dict()
    task_suite = bench_dict["libero_spatial"]()
    print("Tasks in suite:", task_suite.n_tasks)

    task = task_suite.get_task(0)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print("Task:", task.name)
    print("BDDL:", bddl_path)
    with open(bddl_path) as f:
        bddl_sha = hashlib.sha256(f.read().encode()).hexdigest()
    print("BDDL SHA:", bddl_sha)

    results = []
    for run in range(2):
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=256, camera_widths=256,
            has_renderer=True, has_offscreen_renderer=True,
            render_gpu_device_id=0, use_camera_obs=True,
        )
        env.seed(seed)
        obs = env.reset()
        av = obs["agentview_image"]
        sha = hashlib.sha256(np.array(Image.fromarray(av)).tobytes()).hexdigest()
        rs = obs["robot0_joint_pos"]
        print("Run %d: SHA=%s shape=%s" % (run + 1, sha, av.shape))

        for i in range(11):
            obs, rew, done, info = env.step(np.zeros(7))
        env.close()
        results.append({"sha": sha, "robot_state": [float(x) for x in rs]})

    r0, r1 = results
    img_match = r0["sha"] == r1["sha"]
    state_match = np.allclose(r0["robot_state"], r1["robot_state"])
    print("Image match:", img_match)
    print("State match:", state_match)

    fp = {
        "task": task.name, "task_idx": 0, "bddl_sha": bddl_sha,
        "seed": seed, "render_gpu": gpu,
        "libero_source": "pip 0.1.1", "mujoco_version": "3.9.0",
        "agentview_shape": list(av.shape), "agentview_dtype": str(av.dtype),
        "two_run_image_match": img_match, "two_run_state_match": state_match,
    }
    if output:
        with open(output, "w") as f:
            json.dump(fp, f, indent=2)
    print("SMOKE", "PASS" if img_match else "FAIL")
    return img_match


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    run_smoke(args.gpu, args.seed, args.output)
