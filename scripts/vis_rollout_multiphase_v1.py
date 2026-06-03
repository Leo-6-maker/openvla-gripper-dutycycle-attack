# -*- coding: utf-8 -*-
"""Fixed-budget multi-phase VIS rollout scheduler for LIBERO-Object."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# sys.path is set inside functions that need gripper_attack imports;
# the raw_gripper_is_open reference used in summary helpers needs this:
try:
    from gripper_attack.gripper_semantics import raw_gripper_is_open
except ImportError:
    REPO = os.environ.get('VLA_REPO', '/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524')
    sys.path.insert(0, f'{REPO}/src')
    from gripper_attack.gripper_semantics import raw_gripper_is_open


REPO = "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
DEFAULT_OUT_BASE = "/data/liuyu/outputs/vis_multiphase_attack_20260602"
UNNORM_KEY = "libero_object"
DEFAULT_EPSILON_PIXELS = 8.0
DEFAULT_ATTACK_STEPS = 40
DEFAULT_STEP_SIZE_PIXELS = 1.0
DEFAULT_ATTACK_OBJECTIVE = "gripper_open_region_ce"


TASK_CONFIGS = {
    "cream_cheese": {
        "task_id": 1,
        "instruction": "pick up the cream cheese and place it in the basket",
        "phase_starts": {
            "contact_grasp_onset": 65,
            "early_lift_carry": 90,
            "preplace_release": 120,
        },
    },
    "ketchup": {
        "task_id": 4,
        "instruction": "pick up the ketchup and place it in the basket",
        "phase_starts": {
            "contact_grasp_onset": 93,
            "early_lift_carry": 118,
            "preplace_release": 148,
        },
    },
    "tomato_sauce": {
        "task_id": 5,
        "instruction": "pick up the tomato sauce and place it in the basket",
        "phase_starts": {
            "contact_grasp_onset": 128,
            "early_lift_carry": 153,
            "preplace_release": 183,
        },
    },
    "salad_dressing": {
        "task_id": 2,
        "instruction": "put the salad dressing in the basket",
        "phase_starts": {
            "contact_grasp_onset": 88,
            "early_lift_carry": 113,
            "preplace_release": 143,
        },
    },
}


SCHEDULES = {
    "clean": [],
    "single_best_phase_d16": [("contact_grasp_onset", 16)],
    "two_phase_equal_d8_d8": [("contact_grasp_onset", 8), ("early_lift_carry", 8)],
    "three_phase_equal_d6_d6_d6": [
        ("contact_grasp_onset", 6),
        ("early_lift_carry", 6),
        ("preplace_release", 6),
    ],
    "two_phase_stronger_d12_d12": [("contact_grasp_onset", 12), ("early_lift_carry", 12)],
    "single_best_phase_d20": [("contact_grasp_onset", 20)],
    "two_phase_strong_carry_preplace_d20_d20": [("early_lift_carry", 20), ("preplace_release", 20)],
    "three_phase_strong_d16_d16_d16": [
        ("contact_grasp_onset", 16),
        ("early_lift_carry", 16),
        ("preplace_release", 16),
    ],
    "ultra_three_phase_d20_d20_d20": [
        ("contact_grasp_onset", 20),
        ("early_lift_carry", 20),
        ("preplace_release", 20),
    ],
}


def prompt(instruction):
    return f"In: What action should the robot take to {instruction}?\nOut:"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), required=True)
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), required=True)
    parser.add_argument("--attack_type", choices=["clean", "vis_pgd", "random_linf"], required=True)
    parser.add_argument("--gpu_pair", default="4,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state_id", type=int, default=0)
    parser.add_argument("--output_root", default=DEFAULT_OUT_BASE)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--epsilon_pixels", type=float, default=DEFAULT_EPSILON_PIXELS)
    parser.add_argument("--attack_steps", type=int, default=DEFAULT_ATTACK_STEPS)
    parser.add_argument("--step_size_pixels", type=float, default=DEFAULT_STEP_SIZE_PIXELS)
    parser.add_argument("--attack_objective", default=DEFAULT_ATTACK_OBJECTIVE)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_phase_windows(task_name, schedule):
    cfg = TASK_CONFIGS[task_name]
    windows = []
    for phase_name, budget in SCHEDULES[schedule]:
        start = int(cfg["phase_starts"][phase_name])
        windows.append({
            "phase_name": phase_name,
            "start": start,
            "end": start + int(budget) - 1,
            "budget": int(budget),
        })
    return windows


def phase_for_step(policy_step, windows):
    for window in windows:
        if window["start"] <= policy_step <= window["end"]:
            return window
    return None


def get_libero_image(obs, resize_size=224):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    return np.array(img)


def decode_image(model_ctx, img_np, instruction):
    model, processor, device, mdtype, action_dim, vs, bc, mask, low, high = model_ctx
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    inputs = processor(prompt(str(instruction).lower()), pil_img, return_tensors="pt")
    inputs.pop("attention_mask", None)
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.to(device=device, dtype=mdtype)
        else:
            inputs[key] = value.to(device)
    if not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            (inputs["input_ids"], torch.tensor([[29871]], dtype=torch.long, device=device)),
            dim=1,
        )
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    disc = np.clip(vs - token_ids - 1, 0, len(bc) - 1)
    norm_actions = bc[disc].astype(np.float32)
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)
    return action, token_ids, gen


def run_pgd_attack(model_ctx, img_np, instruction, clean_action, clean_gen, seed, args):
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result
    from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs
    from gripper_attack.gripper_semantics import raw_gripper_is_open as _raw_gripper_is_open

    model, processor, device, *_ = model_ctx
    target_action = np.asarray(clean_action, dtype=np.float32).copy()
    target_action[-1] = 1.0
    attack_cfg = {
        "attack_optimizer": {
            "method": "token_prefix_pgd",
            "objective": args.attack_objective,
            "epsilon": float(args.epsilon_pixels) / 255.0,
            "step_size": float(args.step_size_pixels) / 255.0,
            "num_steps": int(args.attack_steps),
            "random_start": False,
        }
    }
    attacker = TokenPrefixPGDAttacker(
        model,
        processor,
        attack_cfg,
        seed=seed,
        preprocess_kwargs={"libero_official_preprocess": False, "center_crop": False, "resize_size": 224},
        device=device,
    )
    result = attacker.attack(
        observation=img_np,
        instruction=instruction,
        clean_action=clean_action,
        target_action=target_action,
        clean_model_output=clean_gen,
        unnorm_key=UNNORM_KEY,
    )
    adv_inputs = get_adv_inputs_from_attack_result(result)
    adv_decoded = redecode_openvla_action_from_adv_inputs(
        model=model,
        processor=processor,
        adv_inputs=adv_inputs,
        instruction=instruction,
        unnorm_key=UNNORM_KEY,
    )
    return np.asarray(adv_decoded.action, dtype=np.float32), adv_decoded.token_ids, result


def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action


def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


def qpos_width(obs):
    qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
    if qpos.size == 0:
        return 0.0, 0.0
    return float(qpos[0]), float(np.sum(np.abs(qpos)))


def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("[1] Loading model...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: "9000MiB", 1: "9000MiB", "cpu": "64GiB"},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    vs = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    bc = np.array(model.bin_centers)
    action_dim = int(model.get_action_dim(UNNORM_KEY))
    stats = model.get_action_stats(UNNORM_KEY)
    mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    low = np.array(stats["q01"])
    high = np.array(stats["q99"])
    print(f"    device={device}, mdtype={mdtype}, action_dim={action_dim}", flush=True)
    return model, processor, device, mdtype, action_dim, vs, bc, mask, low, high


def init_env(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = TASK_CONFIGS[args.task]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()
    task = task_suite.get_task(cfg["task_id"])
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(cfg["task_id"])
    env_args = {
        "bddl_file_name": bddl,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": ["agentview"],
        "control_freq": 20,
        "render_gpu_device_id": int(args.gpu_pair.split(",")[0]),
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)
    obs = env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    env.set_init_state(initial_states[int(args.state_id)])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))
    return env, obs


def aggregate_phase_rows(trace_rows, windows):
    rows = []
    for window in windows:
        phase_rows = [r for r in trace_rows if r["phase_name"] == window["phase_name"]]
        attack_rows = [r for r in phase_rows if r["pgd_applied"] or r["random_applied"]]
        qpos_values = [r["qpos_post_step"] for r in phase_rows]
        width_values = [r["width_post_step"] for r in phase_rows]
        rows.append({
            "phase_name": window["phase_name"],
            "phase_start": window["start"],
            "phase_end": window["end"],
            "phase_budget": window["budget"],
            "phase_steps_observed": len(phase_rows),
            "phase_attack_steps": len(attack_rows),
            "phase_open_count": sum(1 for r in attack_rows if raw_gripper_is_open(float(r["adv_grip"]))),
            "phase_longest_open_streak": longest_open_streak(attack_rows),
            "phase_token_flips": sum(1 for r in attack_rows if r["token_flip"]),
            "phase_qpos_delta": max(qpos_values) - min(qpos_values) if qpos_values else 0.0,
            "phase_width_delta": max(width_values) - min(width_values) if width_values else 0.0,
            "phase_arm_l2_mean": float(np.mean([r["arm_l2"] for r in attack_rows])) if attack_rows else 0.0,
        })
    return rows


def longest_open_streak(rows):
    cur = 0
    best = 0
    for row in rows:
        if raw_gripper_is_open(float(row["adv_grip"])):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def infer_failure_phase(success, phase_rows):
    if success:
        return "success"
    if not phase_rows:
        return "unknown_failure"
    strongest = max(phase_rows, key=lambda r: (r["phase_qpos_delta"], r["phase_open_count"]))
    if strongest["phase_qpos_delta"] > 0 or strongest["phase_open_count"] > 0:
        return strongest["phase_name"]
    return "unknown_failure"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(args):
    out = Path(args.output_root)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    windows = build_phase_windows(args.task, args.schedule)
    total_budget = sum(w["budget"] for w in windows)
    run_condition = args.schedule if args.attack_type != "random_linf" else f"random_matched_{args.schedule}"
    if args.attack_type == "clean":
        run_condition = "clean"
    print(f"[0] VIS multiphase: task={args.task} condition={run_condition} seed={args.seed} state={args.state_id}", flush=True)
    print(f"    GPU pair: {args.gpu_pair}; windows={windows}", flush=True)
    print(f"    payload: eps={args.epsilon_pixels}/255 steps={args.attack_steps} step_size={args.step_size_pixels}/255 objective={args.attack_objective}", flush=True)
    if args.dry_run:
        print(json.dumps({"task": args.task, "condition": run_condition, "windows": windows, "total_budget": total_budget}))
        return

    model_ctx = load_model()
    print("[2] Initializing LIBERO environment...", flush=True)
    env, obs = init_env(args)
    rng = np.random.RandomState(args.seed + 250000)
    cfg = TASK_CONFIGS[args.task]
    trace_rows = []
    success = False
    t_start = time.time()
    t = 5
    policy_step = 0
    attack_counter = 0

    while t < args.max_steps + 5:
        img_np = get_libero_image(obs, 224)
        eef_pos = obs["robot0_eef_pos"].copy()
        qpos_pre, width_pre = qpos_width(obs)
        phase = phase_for_step(policy_step, windows)
        phase_name = phase["phase_name"] if phase else "none"
        attack_attempted = args.attack_type in {"vis_pgd", "random_linf"} and phase is not None
        pgd_applied = False
        random_applied = False
        attack_dt = 0.0
        arm_l2 = 0.0
        linf = 0.0
        token_flip = False
        clean_action, clean_token_ids, clean_gen = decode_image(model_ctx, img_np, cfg["instruction"])
        clean_grip = float(clean_action[-1])
        raw_action = clean_action
        adv_grip = clean_grip
        effective_attack_step_idx = ""

        if attack_attempted and args.attack_type == "vis_pgd":
            try:
                attack_t0 = time.time()
                adv_action, adv_token_ids, atk_result = run_pgd_attack(
                    model_ctx,
                    img_np,
                    cfg["instruction"],
                    clean_action,
                    clean_gen,
                    args.seed + policy_step,
                    args,
                )
                attack_dt = time.time() - attack_t0
                raw_action = adv_action
                adv_grip = float(raw_action[-1])
                arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
                linf = float(atk_result.observation_perturb_linf)
                token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])
                pgd_applied = True
                effective_attack_step_idx = attack_counter
                attack_counter += 1
            except Exception as exc:
                print(f"    PGD ERROR at step {policy_step}: {str(exc)[:120]}", flush=True)
                raw_action = clean_action
                adv_grip = clean_grip

        elif attack_attempted and args.attack_type == "random_linf":
            img_f = img_np.astype(np.float32) / 255.0
            eps = float(args.epsilon_pixels) / 255.0
            noise = rng.uniform(-eps, eps, img_f.shape).astype(np.float32)
            adv_img_np = (np.clip(img_f + noise, 0.0, 1.0) * 255).astype(np.uint8)
            adv_action, adv_token_ids, _ = decode_image(model_ctx, adv_img_np, cfg["instruction"])
            raw_action = adv_action
            adv_grip = float(raw_action[-1])
            arm_l2 = float(np.linalg.norm(adv_action[:6] - clean_action[:6]))
            linf = float(np.abs(noise).max())
            token_flip = int(clean_token_ids[-1]) != int(adv_token_ids[-1])
            random_applied = True
            effective_attack_step_idx = attack_counter
            attack_counter += 1

        env_action = invert_gripper_action(normalize_gripper_action(raw_action, binarize=True))
        obs, reward, done, info = env.step(env_action)
        qpos_post, width_post = qpos_width(obs)
        trace_rows.append({
            "task": args.task,
            "condition": run_condition,
            "attack_type": args.attack_type,
            "schedule": args.schedule,
            "seed": args.seed,
            "state_id": args.state_id,
            "step": t,
            "policy_step": policy_step,
            "phase_name": phase_name,
            "in_phase_window": phase is not None,
            "attack_attempted": attack_attempted,
            "pgd_applied": pgd_applied,
            "random_applied": random_applied,
            "effective_attack_step_idx": effective_attack_step_idx,
            "raw_gripper": float(raw_action[-1]),
            "env_gripper": float(env_action[-1]),
            "clean_grip": clean_grip,
            "adv_grip": adv_grip,
            "arm_l2": arm_l2,
            "linf": linf,
            "token_flip": token_flip,
            "attack_dt": attack_dt,
            "qpos_pre_step": qpos_pre,
            "qpos_post_step": qpos_post,
            "width_pre_step": width_pre,
            "width_post_step": width_post,
            "eef_x": float(eef_pos[0]),
            "eef_y": float(eef_pos[1]),
            "eef_z": float(eef_pos[2]),
            "done": bool(done),
            "reward": float(reward),
        })
        if done:
            success = True
            break
        t += 1
        policy_step += 1

    phase_rows = aggregate_phase_rows(trace_rows, windows)
    attack_rows = [r for r in trace_rows if r["pgd_applied"] or r["random_applied"]]
    total_dt = time.time() - t_start
    failure_phase = infer_failure_phase(success, phase_rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"vismp_{args.task}_state{args.state_id}_seed{args.seed}_{run_condition}_{ts}"
    trace_path = out / "runs" / f"{run_id}_trace.csv"
    phase_path = out / "runs" / f"{run_id}_phasewise.csv"
    write_csv(trace_path, trace_rows)
    for row in phase_rows:
        row.update({"run_id": run_id, "task": args.task, "condition": run_condition, "seed": args.seed, "state_id": args.state_id})
    write_csv(phase_path, phase_rows)
    summary = {
        "run_id": run_id,
        "task": args.task,
        "condition": run_condition,
        "attack_type": args.attack_type,
        "schedule": args.schedule,
        "seed": args.seed,
        "state_id": args.state_id,
        "official_success": success,
        "cq_success": None,
        "cq_failure": None,
        "manual_audit_needed": True,
        "failure_phase": failure_phase,
        "epsilon_pixels": args.epsilon_pixels,
        "attack_steps": args.attack_steps,
        "step_size_pixels": args.step_size_pixels,
        "objective": args.attack_objective,
        "gpu_pair": args.gpu_pair,
        "total_budget": total_budget,
        "total_attack_steps": len(attack_rows),
        "token_flips_attacked_steps": sum(1 for r in attack_rows if r["token_flip"]),
        "open_count_attacked_steps": sum(1 for r in attack_rows if raw_gripper_is_open(float(r["adv_grip"]))),
        "longest_open_streak_attacked": longest_open_streak(attack_rows),
        "qpos_delta_total": max([r["qpos_post_step"] for r in trace_rows]) - min([r["qpos_post_step"] for r in trace_rows]) if trace_rows else 0.0,
        "width_delta_total": max([r["width_post_step"] for r in trace_rows]) - min([r["width_post_step"] for r in trace_rows]) if trace_rows else 0.0,
        "arm_l2_mean_attacked": float(np.mean([r["arm_l2"] for r in attack_rows])) if attack_rows else 0.0,
        "total_dt_s": round(total_dt, 1),
        "trace_path": str(trace_path),
        "phasewise_path": str(phase_path),
    }
    summary_path = out / "runs" / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[4] Episode finished: success={success}, steps={policy_step}, attacks={len(attack_rows)}, time={total_dt:.0f}s", flush=True)
    print(json.dumps(summary), flush=True)
    print(f"Saved: {trace_path}", flush=True)


def main():
    args = parse_args()
    run_episode(args)


if __name__ == "__main__":
    main()
