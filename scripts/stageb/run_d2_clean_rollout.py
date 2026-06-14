#!/usr/bin/env python3
"""D2: Fresh clean rollout — PERSISTENT multi-state launcher.
Loads OpenVLA model ONCE, then loops through all missing states.
Eliminates CUDA context conflicts between subprocess calls.
"""

import argparse, csv, os, sys, time, json, glob
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch

# Import runner internals directly
REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from run_s20d_v4_fixed_window_l3_runner import (
    load_model_s20d, decode_with_scores, postprocess_openvla_action_for_libero,
    physical_gripper_state, prompt, resolve_target_object,
)
from libero.libero.envs import OffScreenRenderEnv

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
SUITE = "libero_object"
TASK_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_GUESS = {
    'alphabet_soup': 'alphabet_soup_1', 'bbq_sauce': 'bbq_sauce_1',
    'butter': 'butter_1', 'chocolate_pudding': 'chocolate_pudding_1',
    'cream_cheese': 'cream_cheese_1', 'ketchup': 'ketchup_1',
    'milk': 'milk_1', 'orange_juice': 'orange_juice_1',
    'salad_dressing': 'salad_dressing_1', 'tomato_sauce': 'tomato_sauce_1',
}

GPU_VISIBLE = "0,1,2,4,5,6,7"
RENDER_GPU = 0
MAX_STEPS = 280
NUM_WAIT = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE

    with open(args.manifest, newline="") as f:
        all_jobs = list(csv.DictReader(f))

    by_task = defaultdict(list)
    for j in all_jobs:
        by_task[j["task_key"]].append(j["state_id"])

    total = len(all_jobs)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] D2 persistent rollout: {total} states, {len(by_task)} tasks")
    print(f"Loading model...")
    t0 = time.time()
    model, processor, device, unnorm_key, K_trigger, _ = load_model_s20d(MODEL_PATH, -1)
    print(f"Model loaded in {time.time()-t0:.0f}s")

    completed = 0; failed = 0; job_id = 500000

    for task in sorted(by_task):
        states = sorted(by_task[task], key=int)
        # Build environment
        bddl_file = os.path.join(
            os.path.dirname(OffScreenRenderEnv.__init__.__code__.co_filename),
            "..", "..", "bddl_files", SUITE, f"{task}.bddl_file")
        env_init = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=256, camera_widths=256,
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=['agentview'],
            control_freq=20, render_gpu_device_id=RENDER_GPU,
            horizon=MAX_STEPS + NUM_WAIT)
        instruction = prompt(task, TASK_ORDER)
        target_name = TARGET_GUESS.get(task, 'akita_black_bowl_1')

        for sid_str in states:
            sid = int(sid_str)
            tag = f"{task}_s{sid}_w0_10_s20d_clean_seed0"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}")

            try:
                env_init.seed(0)
                obs = env_init.reset()
                init_states = getattr(env_init, 'init_states', None)
                if init_states is None:
                    init_states = [env_init.sim.get_state().flatten().tolist()]
                if sid >= len(init_states):
                    print(f"  SKIP: sid {sid} out of range (max {len(init_states)-1})")
                    failed += 1; continue
                obs = env_init.set_init_state(init_states[sid])

                dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                for _ in range(NUM_WAIT):
                    obs, _, _, _ = env_init.step(dummy_action)

                trace_rows = []; success_check = False; success_done = False
                done_step = -1; success_step = -1

                for step in range(MAX_STEPS):
                    if 'agentview_image' not in obs:
                        break
                    img_uint8 = obs['agentview_image']
                    clean_action, _, _, _ = decode_with_scores(
                        model, processor, device, img_uint8, instruction,
                        unnorm_key, K_trigger, libero_official_preprocess=False,
                        libero_preprocess_backend='official_pil_lanczos',
                        center_crop=True, resize_size=224)
                    env_action = postprocess_openvla_action_for_libero(clean_action)
                    obs, reward, terminated, truncated, info = env_init.step(env_action)
                    done = terminated or truncated
                    phys = physical_gripper_state(env_action)
                    raw_gripper = 1.0 - info.get('gripper_qpos', phys)

                    trace_rows.append({
                        "step": step, "obj_x": obs.get('object-state', [0,0,0,0,0,0,0,0])[0] if obs.get('object-state') else 0,
                        "obj_y": obs.get('object-state', [0,0,0,0,0,0,0,0])[1] if obs.get('object-state') else 0,
                        "obj_z": obs.get('object-state', [0,0,0,0,0,0,0,0])[2] if obs.get('object-state') else 0,
                        "eef_x": obs['robot0_eef_pos'][0], "eef_y": obs['robot0_eef_pos'][1],
                        "eef_z": obs['robot0_eef_pos'][2],
                        "clean_gripper_env": info.get('gripper_env', phys),
                        "decoded_open_bool": int(phys > 0.5),
                        "gripper_qpos_before": info.get('gripper_qpos', 0),
                    })

                    if not success_check:
                        try:
                            if getattr(env_init, 'check_success', lambda: False)():
                                success_check = True; success_step = step
                        except: pass
                    if done and not success_done:
                        success_done = True; done_step = step
                    if done:
                        break

                # Write trace CSV
                trace_path = os.path.join(args.output_dir, f"trace_{tag}_job{job_id}.csv")
                import csv as csv_mod
                with open(trace_path, "w", newline="") as f:
                    w = csv_mod.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                    w.writeheader(); w.writerows(trace_rows)

                status = f"steps={len(trace_rows)} succ_check={success_check} succ_done={success_done}"
                print(f"  OK: {status}")
                completed += 1
            except Exception as e:
                print(f"  FAIL: {e}")
                failed += 1

            job_id += 1

    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: {completed}/{total} ({failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
