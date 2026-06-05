#!/usr/bin/env python3
"""Phase E low-budget VIS canary.

CPU-only dry-run is supported and exits before importing or loading OpenVLA,
LIBERO, torch, or any GPU-dependent runtime.

Scientific boundary:
    Phase E canary v0 is INVALID_ACTION_SPACE_CONFOUNDED because it passed raw
    OpenVLA actions directly to env.step(). This version applies the official
    normalize_gripper_action + invert_gripper_action transform before every
    env.step().
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
UNNORM = "libero_object"
OUTPUT_DIR = "/data/liuyu/outputs/fast_vis_phaseE_low_budget_20260605"
ACTION_TRANSFORM_VERSION = "v1_official_normalize_gripper_then_invert_before_env_step_20260605"
MEASUREMENT_VERSION = "v2_mujoco_gripper_qpos_primary_obs_audit_fallback_20260605"
INVALID_PREVIOUS_PHASE_E_V0 = "INVALID_ACTION_SPACE_CONFOUNDED"

TASK_ID = {
    "alphabet_soup": 0,
    "cream_cheese": 1,
    "salad_dressing": 2,
    "bbq_sauce": 3,
    "ketchup": 4,
    "tomato_sauce": 5,
    "butter": 6,
    "milk": 7,
    "chocolate_pudding": 8,
    "orange_juice": 9,
}

TASK_INSTRUCTION = {
    "cream_cheese": "pick up the cream cheese and place it in the basket",
    "bbq_sauce": "pick up the bbq sauce and place it in the basket",
}

DEFAULT_CANARY = [
    ("cream_cheese", 4, 28, 45, 1),
    ("bbq_sauce", 5, 27, 44, 0),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--output-csv", default="tables/fast_vis_low_budget_sweep_v0.csv")
    ap.add_argument("--gpu-pair", default="4,5")
    ap.add_argument("--eps-raw-pixels", type=int, default=4)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--pgd-restarts", type=int, default=1)
    ap.add_argument("--compressed-len", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def validate_gpu_pair(gpu_pair: str):
    ids = [x.strip() for x in gpu_pair.split(",") if x.strip()]
    if any(x in {"3", "7"} for x in ids):
        raise SystemExit(f"INFRA_FAILED: GPU3/GPU7 are blacklisted; requested --gpu-pair={gpu_pair}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible == "2,6" and gpu_pair.replace(" ", "") == "2,6":
        raise SystemExit(
            "INFRA_FAILED: do not combine CUDA_VISIBLE_DEVICES=2,6 with --gpu-pair 2,6; "
            "inside a remapped visible set, --gpu-pair would need logical 0,1, but this is not recommended"
        )
    return [int(x) for x in ids]


def from_pretrained_local(cls, path: str, **kwargs):
    try:
        return cls.from_pretrained(path, local_files_only=True, **kwargs)
    except TypeError:
        return cls.from_pretrained(path, **kwargs)


def normalize_gripper_action(action, binarize=True):
    import numpy as np

    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action


def invert_gripper_action(action):
    import numpy as np

    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


def transform_for_env_step(raw_action):
    return invert_gripper_action(normalize_gripper_action(raw_action, binarize=True))


def read_mujoco_gripper_qpos(env):
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is not None and data is not None and hasattr(data, "qpos"):
        joint_names = list(getattr(model, "joint_names", []) or [])
        preferred = []
        fallback = []
        for name in joint_names:
            lname = str(name).lower()
            if "gripper" in lname and "finger" in lname:
                preferred.append(name)
            elif "gripper" in lname:
                fallback.append(name)
        for name in preferred + fallback:
            try:
                jid = model.joint_name2id(name)
                adr = int(model.jnt_qposadr[jid])
                return float(data.qpos[adr]), f"mujoco_joint:{name}", "ok"
            except Exception:
                continue
    return None, "", "missing_mujoco_gripper_qpos"


def read_obs_gripper_qpos(obs):
    import numpy as np

    if isinstance(obs, dict) and "robot0_gripper_qpos" in obs:
        arr = np.asarray(obs.get("robot0_gripper_qpos"), dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return float(arr[0]), "obs.robot0_gripper_qpos", "ok"
    return None, "", "missing_obs_robot0_gripper_qpos"


def read_gripper_qpos(obs, env):
    mujoco_qpos, mujoco_source, mujoco_status = read_mujoco_gripper_qpos(env)
    obs_qpos, obs_source, obs_status = read_obs_gripper_qpos(obs)
    warning = ""
    if mujoco_qpos is not None and obs_qpos is not None and abs(float(mujoco_qpos) - float(obs_qpos)) > 1e-3:
        warning = "mujoco_obs_qpos_mismatch"
    if mujoco_qpos is not None:
        used = float(mujoco_qpos)
        source = mujoco_source
        status = "ok"
    elif obs_qpos is not None:
        used = float(obs_qpos)
        source = obs_source
        status = "ok"
    else:
        used = None
        source = "unavailable"
        status = "missing_gripper_qpos"
    return {
        "used": used,
        "source": source,
        "status": status,
        "mujoco": mujoco_qpos,
        "obs": obs_qpos,
        "warning": warning,
        "source_priority": "mujoco_primary_obs_fallback",
    }


def decode_tokens_to_action(tokens, mask, q01, q99, bins, vocab_size):
    import numpy as np

    action = np.zeros(7, np.float32)
    nb = len(bins)
    for dim in range(7):
        if mask[dim]:
            bid = min(int(vocab_size - tokens[dim] - 1), nb - 1)
            action[dim] = 0.5 * (bins[bid] + 1.0) * (q99[dim] - q01[dim]) + q01[dim]
    return action


def centered_window(parent_start: int, parent_end: int, length: int):
    center = (int(parent_start) + int(parent_end)) // 2
    start = max(0, center - int(length) // 2)
    return start, start + int(length)


def run_canary(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gripper_attack.attack_adapter import (
        TokenPrefixPGDAttacker,
        prepare_openvla_image_for_attack,
        _prompt,
        get_adv_inputs_from_attack_result,
    )

    gpu_ids = validate_gpu_pair(args.gpu_pair)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(str(REPO / "tables"), exist_ok=True)

    print("=== Phase E Canary: Low-Budget VIS Smoke ===")
    print(f"Previous Phase E v0 status: {INVALID_PREVIOUS_PHASE_E_V0}")
    print(f"Budget: eps={args.eps_raw_pixels} steps={args.pgd_steps} restarts={args.pgd_restarts} L={args.compressed_len}")
    print(f"GPU pair argument: {args.gpu_pair} (physical IDs; this script does not set CUDA_VISIBLE_DEVICES)")

    t0 = time.time()
    max_memory = {gpu_ids[0]: "10500MiB", gpu_ids[1]: "10500MiB", "cpu": "64GiB"} if len(gpu_ids) >= 2 else None
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if max_memory is not None:
        model_kwargs.update(device_map="auto", max_memory=max_memory)
    model = from_pretrained_local(AutoModelForVision2Seq, MODEL_PATH, **model_kwargs)
    processor = from_pretrained_local(AutoProcessor, MODEL_PATH, trust_remote_code=True)
    device = str(next(model.parameters()).device)
    print(f"Model: {device} in {time.time() - t0:.1f}s")

    stats = model.get_action_stats(UNNORM)
    mask = np.asarray(stats["mask"], dtype=bool)
    q01 = np.asarray(stats["q01"], np.float32)
    q99 = np.asarray(stats["q99"], np.float32)
    bins = np.asarray(model.bin_centers, np.float32)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    eps_proc = args.eps_raw_pixels / 255.0

    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    results = []

    for i, (task_key, state_id, parent_start, parent_end, label) in enumerate(DEFAULT_CANARY):
        task_id = TASK_ID[task_key]
        window_start, window_end = centered_window(parent_start, parent_end, args.compressed_len)
        instruction = TASK_INSTRUCTION[task_key]
        print(f"\n[{i + 1}/{len(DEFAULT_CANARY)}] {task_key}_s{state_id} parent=[{parent_start},{parent_end}] L={args.compressed_len} [{window_start},{window_end}] label={label}")
        t_ep = time.time()
        env = None
        try:
            task = task_suite.get_task(task_id)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl,
                camera_heights=224,
                camera_widths=224,
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["agentview"],
                control_freq=20,
                render_gpu_device_id=gpu_ids[0],
            )
            env.seed(0)
            init_states = task_suite.get_task_init_states(task_id)
            obs = env.reset()
            env.sim.data.qvel[:] = 0
            env.sim.forward()
            env.set_init_state(init_states[state_id])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            step = 0
            done = False
            vis_open_count = 0
            qpos_start = None
            qpos_min = None
            qpos_source = ""
            qpos_mujoco = ""
            qpos_obs = ""
            qpos_used = ""
            qpos_warning = ""
            raw_clean_action_gripper = ""
            raw_adv_action_gripper = ""
            env_clean_action_gripper_after_transform = ""
            env_adv_action_gripper_after_transform = ""
            post_transform_gripper_action = ""

            while step < args.max_steps and not done:
                img_np = np.array(Image.fromarray(obs["agentview_image"][::-1]))
                img = prepare_openvla_image_for_attack(img_np)
                inputs = processor(_prompt(instruction), img, return_tensors="pt")
                inputs.pop("attention_mask", None)
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

                with torch.no_grad():
                    clean_out = model.generate(
                        input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=7,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                clean_tokens = clean_out.sequences[0, -7:].cpu().numpy()
                clean_action = decode_tokens_to_action(clean_tokens, mask, q01, q99, bins, vocab_size)
                env_clean_action = transform_for_env_step(clean_action)

                if step == window_start:
                    raw_clean_action_gripper = float(clean_action[-1])
                    env_clean_action_gripper_after_transform = float(env_clean_action[-1])

                qpos_audit = read_gripper_qpos(obs, env)
                if qpos_audit["mujoco"] is not None:
                    qpos_mujoco = float(qpos_audit["mujoco"])
                if qpos_audit["obs"] is not None:
                    qpos_obs = float(qpos_audit["obs"])
                if qpos_audit["used"] is not None:
                    qpos_used = float(qpos_audit["used"])
                if qpos_audit["warning"]:
                    qpos_warning = qpos_audit["warning"]
                if not qpos_source:
                    qpos_source = qpos_audit["source"]
                if qpos_audit["used"] is not None and step == window_start and qpos_start is None:
                    qpos_start = float(qpos_audit["used"])
                if qpos_audit["used"] is not None and window_start <= step < window_end:
                    qpos_min = float(qpos_audit["used"]) if qpos_min is None else min(qpos_min, float(qpos_audit["used"]))

                raw_action = clean_action
                env_action = env_clean_action

                if window_start <= step < window_end:
                    target_action = clean_action.copy()
                    target_action[-1] = 0.0
                    config = {
                        "attack_optimizer": {
                            "epsilon": float(eps_proc),
                            "step_size": float(max(eps_proc / args.pgd_steps, 1e-4)),
                            "num_steps": args.pgd_steps,
                            "random_start": False,
                            "temporal_init": "none",
                            "objective": "prefix_locked_gripper_open_margin",
                            "gripper_margin": 5.0,
                            "arm_preserve_weight": 0.1,
                        }
                    }
                    attacker = TokenPrefixPGDAttacker(model, processor, config, seed=0, device=device)
                    result = attacker.attack(
                        img_np,
                        instruction=instruction,
                        clean_action=clean_action,
                        target_action=target_action,
                        unnorm_key=UNNORM,
                    )
                    adv_inputs = get_adv_inputs_from_attack_result(result)
                    with torch.no_grad():
                        adv_out = model.generate(
                            adv_inputs["input_ids"].to(device),
                            pixel_values=adv_inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
                            max_new_tokens=7,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    adv_tokens = adv_out.sequences[0, -7:].cpu().numpy()
                    adv_action = decode_tokens_to_action(adv_tokens, mask, q01, q99, bins, vocab_size)
                    env_adv_action = transform_for_env_step(adv_action)
                    raw_action = adv_action
                    env_action = env_adv_action
                    raw_adv_action_gripper = float(adv_action[-1])
                    env_adv_action_gripper_after_transform = float(env_adv_action[-1])
                    post_transform_gripper_action = float(env_action[-1])
                    if float(adv_action[-1]) < 0.5:
                        vis_open_count += 1
                else:
                    post_transform_gripper_action = float(env_action[-1])

                obs, reward, done, info = env.step(env_action)
                step += 1

            qpos_opening_delta = ""
            provenance_status = "ok"
            if qpos_start is None or qpos_min is None:
                provenance_status = "MEASUREMENT_FAILED:missing_gripper_qpos"
            else:
                qpos_opening_delta = round(float(qpos_start) - float(qpos_min), 6)

            runtime = time.time() - t_ep
            print(f"  done={done} steps={step} vis_open={vis_open_count} qpos_delta={qpos_opening_delta} rt={runtime:.1f}s")
            results.append({
                "task_key": task_key,
                "state_id": state_id,
                "parent_window_start": parent_start,
                "parent_window_end": parent_end,
                "window_start": window_start,
                "window_end": window_end,
                "compressed_len": args.compressed_len,
                "eps_raw_pixels": args.eps_raw_pixels,
                "pgd_steps": args.pgd_steps,
                "pgd_restarts": args.pgd_restarts,
                "objective": "prefix_locked_gripper_open_margin",
                "gpu_pair": args.gpu_pair,
                "runtime_sec": round(runtime, 2),
                "VIS_OPEN_count": vis_open_count,
                "qpos_opening_delta": qpos_opening_delta,
                "done": int(done),
                "steps": step,
                "action_transform_version": ACTION_TRANSFORM_VERSION,
                "raw_clean_action_gripper": raw_clean_action_gripper,
                "raw_adv_action_gripper": raw_adv_action_gripper,
                "env_clean_action_gripper_after_transform": env_clean_action_gripper_after_transform,
                "env_adv_action_gripper_after_transform": env_adv_action_gripper_after_transform,
                "post_transform_gripper_action": post_transform_gripper_action,
                "measurement_version": MEASUREMENT_VERSION,
                "gripper_qpos_source": qpos_source or "unavailable",
                "gripper_qpos_mujoco": qpos_mujoco,
                "gripper_qpos_obs": qpos_obs,
                "gripper_qpos_used": qpos_used,
                "gripper_qpos_source_priority": "mujoco_primary_obs_fallback",
                "gripper_qpos_warning": qpos_warning,
                "previous_phase_e_v0_status": INVALID_PREVIOUS_PHASE_E_V0,
                "provenance_status": provenance_status,
                "denominator_status": "low_budget_no_precheck",
                "label_source": "low_budget_vis",
                "label_confidence": "silver_candidate",
                "label": label,
                "full_vis_label": label,
            })
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append({
                "task_key": task_key,
                "state_id": state_id,
                "parent_window_start": parent_start,
                "parent_window_end": parent_end,
                "window_start": window_start,
                "window_end": window_end,
                "compressed_len": args.compressed_len,
                "full_vis_label": label,
                "action_transform_version": ACTION_TRANSFORM_VERSION,
                "measurement_version": MEASUREMENT_VERSION,
                "previous_phase_e_v0_status": INVALID_PREVIOUS_PHASE_E_V0,
                "provenance_status": f"ERROR:{str(exc)[:100]}",
                "denominator_status": "low_budget_no_precheck",
                "gpu_pair": args.gpu_pair,
                "runtime_sec": "",
                "label_source": "low_budget_vis",
                "label_confidence": "silver_candidate",
                "label": label,
            })
        finally:
            if env is not None:
                env.close()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = REPO / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row.keys()})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCanary done: {len(results)} rows; wrote {output_csv}")


def main():
    args = parse_args()
    validate_gpu_pair(args.gpu_pair)
    if args.dry_run:
        print("DRY RUN: Phase E low-budget canary")
        print(f"Previous Phase E v0 status: {INVALID_PREVIOUS_PHASE_E_V0}")
        for task_key, state_id, parent_start, parent_end, label in DEFAULT_CANARY:
            window_start, window_end = centered_window(parent_start, parent_end, args.compressed_len)
            print(f"  {task_key}_s{state_id} parent=[{parent_start},{parent_end}] window=[{window_start},{window_end}] label={label}")
        return
    run_canary(args)


if __name__ == "__main__":
    main()
