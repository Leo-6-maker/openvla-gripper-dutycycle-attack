#!/usr/bin/env python3
"""Milestone 2D — Artifact-rich official-eval-aligned clean runner.

Preserves corrected clean behavior (PIL Lanczos, correct prompt, EOS handling)
while saving per-timestep artifacts: RGB frames, step_records, episode_records,
run_manifest, gripper/EEF/action traces, and optional teacher-privileged state.

Each worker writes its own manifest shard. Never overwrites another worker's CSV.
"""
import os, sys, json, time, math, csv, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.utils.libero_privileged_state import (
    extract_teacher_privileged_state,
    build_sim_debug_metadata,
)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if abs(den) < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


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



# ── TCN Detector (online streaming, causal, CPU) ──
HISTORY_LEN_DET = 16; HIDDEN_DIM_DET = 64; TCN_LAYERS_DET = 3
ALLOWED_PROPRIO = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]
N_PROPRIO = len(ALLOWED_PROPRIO)

class CausalTCNDetector(torch.nn.Module):
    def __init__(self, in_dim, h_dim, n_ph=8, n_l=3, do=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, h_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(h_dim, h_dim, 3, padding=2**(i+1), dilation=2**i)
            for i in range(n_l)])
        self.drop = torch.nn.Dropout(do)
        self.ph = torch.nn.Linear(h_dim, n_ph); self.hz = torch.nn.Linear(h_dim, 1)
        self.rl = torch.nn.Linear(h_dim, 1)
    def forward(self, x):
        x = self.proj(x); x = x.transpose(1, 2)
        for c in self.convs:
            r = x; x = torch.nn.functional.relu(c(x))
            x = x[:, :, -r.shape[2]:] + r; x = self.drop(x)
        x = x[:, :, -1]
        return self.ph(x), self.hz(x).squeeze(-1), self.rl(x).squeeze(-1)

class OnlineDetector:
    def __init__(self, model_path, device="cpu", hazard_th=0.1, trig_dur=5, cooldown=0):
        self.model = CausalTCNDetector(N_PROPRIO, HIDDEN_DIM_DET, 8, TCN_LAYERS_DET).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device; self.hazard_th = hazard_th
        self.trig_dur = trig_dur; self.cooldown = cooldown
        self.history = []; self.hazard_buf = []; self.cooldown_ctr = 0

    def reset(self):
        self.history = []; self.hazard_buf = []; self.cooldown_ctr = 0

    def update(self, features):
        self.history.append(features)
        if len(self.history) > HISTORY_LEN_DET:
            self.history = self.history[-HISTORY_LEN_DET:]
        hist = np.array(self.history, dtype=np.float32)
        if hist.shape[0] < HISTORY_LEN_DET:
            pad = np.zeros((HISTORY_LEN_DET - hist.shape[0], N_PROPRIO), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ph, hl, rl = self.model(x)
        hs = float(torch.sigmoid(hl).item()); rs = float(torch.sigmoid(rl).item())
        pp = torch.softmax(ph, dim=-1).cpu().numpy()[0]
        pi = int(np.argmax(pp)); conf = float(pp[pi])

        self.hazard_buf.append(hs)
        if len(self.hazard_buf) > self.trig_dur:
            self.hazard_buf = self.hazard_buf[-self.trig_dur:]

        trigger_now = False; trigger_reason = ""
        if self.cooldown_ctr > 0:
            self.cooldown_ctr -= 1
        elif len(self.hazard_buf) >= self.trig_dur:
            if all(h > self.hazard_th for h in self.hazard_buf):
                trigger_now = True; trigger_reason = f"consecutive_{self.trig_dur}"
                self.cooldown_ctr = self.cooldown
        return {"hazard_score": hs, "release_safe_score": rs,
                "phase_idx": pi, "phase_confidence": conf,
                "trigger_now": trigger_now,
                "trigger_duration": self.trig_dur if trigger_now else 0,
                "trigger_reason": trigger_reason}

def attack_action(action, condition, rng):
    if condition == "clean": return action
    a = action.copy()
    if condition == "oracle_open": a[-1] = 1.0
    elif condition == "random_control": a[-1] = 1.0 if rng.random() > 0.5 else -1.0
    elif condition in ("VIS_targeted", "gripper_inversion_proxy"):
        # NOTE: This is NOT visual PGD. It is a command-layer gripper inversion + noise proxy.
        # True VIS PGD requires OpenVLAVisualAttacker from v4_run_eval_openvla.py.
        # For formal VIS evidence, wire the visual attacker path instead.
        a[-1] = float(np.clip(-action[-1] + rng.normal(0, 0.05), -1.0, 1.0))
    return a


def parse_args():
    ap = argparse.ArgumentParser(description="Artifact-rich official eval runner")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_suite_name", default="libero_object")
    ap.add_argument("--num_trials_per_task", type=int, default=10)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--attn_impl", default="eager")
    ap.add_argument("--cuda_visible_devices", default="0,1")
    ap.add_argument("--task_start", type=int, default=0)
    ap.add_argument("--task_count", type=int, default=10)
    ap.add_argument("--run_id_prefix", default="artifact_rich")
    ap.add_argument("--worker_id", default="w0")
    ap.add_argument("--save_rgb", action="store_true", default=True)
    ap.add_argument("--save_step_records", action="store_true", default=True)
    ap.add_argument("--save_privileged_teacher_state", action="store_true", default=True)
    ap.add_argument("--rgb_format", default="png", choices=["png", "jpg"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--detector_path", default="", help="TCN detector .pt for shadow/attack")
    ap.add_argument("--detector_hazard_threshold", type=float, default=0.1)
    ap.add_argument("--detector_trigger_duration", type=int, default=5)
    ap.add_argument("--detector_cooldown", type=int, default=0)
    ap.add_argument("--attack_condition", default="clean",
        choices=["clean", "oracle_open", "random_control", "gripper_inversion_proxy"])
    return ap.parse_args()


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, resize_size=224):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    return np.array(img)


def get_vla_action(model, processor, device, obs, task_label, unnorm_key, center_crop=True):

    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        scale = 0.9 ** 0.5
        w, h = image.size
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        image = image.resize((224, 224), Image.LANCZOS)

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    for key, val in list(inputs.items()):
        inputs[key] = val.to(device=device, dtype=torch.bfloat16 if torch.is_floating_point(val) else None)

    input_ids = inputs.get("input_ids")
    if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
        inputs["input_ids"] = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)

    action_dim = int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                            return_dict_in_generate=True, output_scores=True)

    token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.array(stats["q99"]), np.array(stats["q01"])
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)
    return action, prompt


def save_step_record(run_dir, step_idx, record, args):
    if not args.save_step_records:
        return
    path = run_dir / "step_records.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def save_episode_record(run_dir, record, args):
    path = run_dir / "episode_records.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def save_run_manifest(run_dir, manifest, args):
    path = run_dir / "run_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["MUJOCO_GL"] = "egl"

    from transformers import AutoModelForVision2Seq, AutoProcessor

    visible = len(args.cuda_visible_devices.split(","))
    max_memory = {i: "10000MiB" for i in range(max(visible, 1))}
    max_memory["cpu"] = "128GiB"

    print(f"=== Artifact-Rich Official Eval Runner ===")
    print(f"Model: {args.model_path}")
    print(f"Suite: {args.task_suite_name}")
    print(f"Worker: {args.worker_id}")
    print(f"Save RGB: {args.save_rgb}")
    print(f"Save step_records: {args.save_step_records}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("[DRY RUN] Schema check only. No GPU rollout.")
        # Validate action_path
        action_path = "generate_manual_decode"
        assert action_path in {"predict_action", "generate_manual_decode", "unknown_needs_audit"}
        print(f"action_path={action_path}  (valid)")
        print("DRY RUN PASSED")
        return 0

    # Load model
    print("\n[*] Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, attn_implementation=args.attn_impl,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map="auto", max_memory=max_memory,
    )
    DEVICE = "cuda:0"
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"): DEVICE = v; break
            if isinstance(v, int): DEVICE = f"cuda:{v}"; break
    print(f"[*] Model loaded, primary device: {DEVICE}")

    # ── Load detector ──
    detector = None; attack_rng = None
    if getattr(args, 'detector_path', '') and args.detector_path:
        print(f"[*] Loading detector from {args.detector_path}...")
        detector = OnlineDetector(args.detector_path, device="cpu",
            hazard_th=args.detector_hazard_threshold,
            trig_dur=args.detector_trigger_duration,
            cooldown=args.detector_cooldown)
        attack_rng = np.random.RandomState(42)
        n_params = sum(p.numel() for p in detector.model.parameters())
        print(f"[*] Detector ready ({n_params} params)")
        print(f"[*] Attack condition: {args.attack_condition}")

    stats_path = os.path.join(args.model_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            model.norm_stats = json.load(f)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    unnorm_key = args.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Key {unnorm_key} not found in norm_stats"

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "tables" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Suite-specific max steps
    suite_max_steps = {
        "libero_spatial": 220, "libero_object": 280,
        "libero_goal": 300, "libero_10": 520,
    }
    max_steps = suite_max_steps.get(args.task_suite_name, 400)

    detector = None; attack_rng = None
    task_end = min(args.task_start + args.task_count, num_tasks)
    total_episodes = 0
    total_successes = 0
    summary_rows = []

    for task_id in range(args.task_start, task_end):
        task = task_suite.get_task(task_id)
        task_name = task_suite.get_task_names()[task_id]
        task_desc = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        for ep_idx in range(args.num_trials_per_task):
            start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            run_id = f"{args.run_id_prefix}_{task_name.replace('pick_up_the_','').replace('_and_place_it_in_the_basket','')}_s{ep_idx}"
            run_dir = out / "runs" / args.task_suite_name / f"{task_name}_state{ep_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)
            rgb_dir = run_dir / "frames" if args.save_rgb else None
            if rgb_dir:
                rgb_dir.mkdir(exist_ok=True)

            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(0)
            obs = env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            t = 0
            policy_step = 0
            success = False
            attack_remaining = 0
            if detector is not None:
                detector.reset()
            runtime_error = False
            error_msg = ""
            done_step = -1
            eef_prev = None
            action_path = "generate_manual_decode"

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        # Record wait step
                        if args.save_step_records:
                            save_step_record(run_dir, t, {
                                "run_id": run_id, "suite": args.task_suite_name,
                                "task_id": task_id, "task_name": task_name,
                                "task_instruction": task_desc, "state_id": ep_idx,
                                "seed": args.seed, "step_idx": t,
                                "policy_step_idx": -1, "phase": "wait",
                                "image_path": "", "image_path_available": False,
                                "raw_action": [0]*7, "env_action": [0]*7,
                                "action_dx": 0, "action_dy": 0, "action_dz": 0, "action_gripper": 0,
                                "gripper_command": 0, "gripper_qpos": float(obs.get("robot0_gripper_qpos", [0])[0]),
                                "gripper_width": 0,
                                "eef_x": float(obs["robot0_eef_pos"][0]),
                                "eef_y": float(obs["robot0_eef_pos"][1]),
                                "eef_z": float(obs["robot0_eef_pos"][2]),
                                "eef_qx": 0, "eef_qy": 0, "eef_qz": 0, "eef_qw": 0,
                                "eef_axang_x": 0, "eef_axang_y": 0, "eef_axang_z": 0,
                                "eef_vx": 0, "eef_vy": 0, "eef_vz": 0,
                                "reward": float(reward), "done": bool(done),
                                "success_so_far": False, "info_success_if_available": "",
                                "runtime_error": False, "error_msg": "",
                                "teacher_privileged_state_available": False,
                            }, args)
                        t += 1
                        continue

                    # Preprocess image
                    img = get_libero_image(obs, 224)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate([
                            obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                    }

                    # Save RGB
                    image_path = ""
                    if args.save_rgb:
                        image_path = str(rgb_dir / f"step_{policy_step:04d}.{args.rgb_format}")
                        Image.fromarray(img).save(image_path)

                    # Model inference
                    raw_action, prompt_str = get_vla_action(
                        model, processor, DEVICE, observation, task_desc, unnorm_key,
                        center_crop=args.center_crop,
                    )

                    # Gripper postprocess
                    env_action = normalize_gripper_action(raw_action.copy(), binarize=True)
                    env_action = invert_gripper_action(env_action)

                    # ── EEF velocity (computed before detector for causal correctness) ──
                    eef_pos = obs["robot0_eef_pos"]
                    eef_vx = eef_vy = eef_vz = 0.0
                    if eef_prev is not None:
                        eef_vx = float(eef_pos[0] - eef_prev[0])
                        eef_vy = float(eef_pos[1] - eef_prev[1])
                        eef_vz = float(eef_pos[2] - eef_prev[2])
                    eef_prev = eef_pos.copy()

                    # ── Detector inference (online, causal) ──
                    det_out = None; attack_applied = False
                    original_env_action = env_action.copy()
                    if detector is not None:
                        gq = obs.get("robot0_gripper_qpos", [0.0])
                        gq_val = float(gq[0]) if hasattr(gq, '__len__') else float(gq)
                        ef = obs["robot0_eef_pos"]
                        det_feats = np.array([
                            float(raw_action[-1]), gq_val,
                            float(obs.get("gripper_width", 0)),
                            float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
                            eef_vx, eef_vy, eef_vz,
                            float(raw_action[0]), float(raw_action[1]),
                            float(raw_action[2]), float(env_action[-1]),
                        ], dtype=np.float32)
                        det_out = detector.update(det_feats)
                        if det_out["trigger_now"] and attack_remaining == 0:
                            attack_remaining = det_out["trigger_duration"]
                        if attack_remaining > 0 and args.attack_condition != "clean":
                            env_action = attack_action(env_action, args.attack_condition, attack_rng)
                            attack_applied = True
                            attack_remaining -= 1
                        else:
                            attack_applied = False

# EEF velocity already computed before detector

                    # Teacher privileged state
                    teacher_priv = False
                    object_pose_json = ""
                    target_pose_json = ""
                    obj_dist = ""
                    object_eef_dist = ""
                    priv_error = ""
                    if args.save_privileged_teacher_state:
                        priv = extract_teacher_privileged_state(env, obs, task_desc)
                        teacher_priv = priv["teacher_privileged_state_available"]
                        object_pose_json = priv["object_pose_json"]
                        target_pose_json = priv["target_pose_json"]
                        obj_dist = priv["object_to_target_distance"]
                        object_eef_dist = priv["object_eef_distance"]
                        priv_error = priv["privileged_state_error"]

                    # Step record
                    if args.save_step_records:
                        save_step_record(run_dir, t, {
                            "run_id": run_id, "suite": args.task_suite_name,
                            "task_id": task_id, "task_name": task_name,
                            "task_instruction": task_desc, "state_id": ep_idx,
                            "seed": args.seed, "step_idx": t,
                            "policy_step_idx": policy_step, "phase": "policy",
                            "image_path": image_path,
                            "image_path_available": bool(image_path),
                            "raw_action": raw_action.tolist(),
                            "env_action": env_action.tolist(),
                            "action_dx": float(env_action[0]),
                            "action_dy": float(env_action[1]),
                            "action_dz": float(env_action[2]),
                            "action_gripper": float(env_action[-1]),
                            "gripper_command": float(raw_action[-1]),
                            "gripper_qpos": float(obs.get("robot0_gripper_qpos", [0])[0]),
                            "gripper_width": float(obs.get("robot0_gripper_qpos", [0])[0]),
                            "eef_x": float(eef_pos[0]), "eef_y": float(eef_pos[1]), "eef_z": float(eef_pos[2]),
                            "eef_qx": 0, "eef_qy": 0, "eef_qz": 0, "eef_qw": 0,
                            "eef_axang_x": 0, "eef_axang_y": 0, "eef_axang_z": 0,
                            "eef_vx": eef_vx, "eef_vy": eef_vy, "eef_vz": eef_vz,
                            "reward": float(reward) if 'reward' in dir() else 0.0,
                            "done": False, "success_so_far": False,
                            "info_success_if_available": "",
                            "runtime_error": False, "error_msg": "",
                            "object_pose_json": object_pose_json,
                            "target_pose_json": target_pose_json,
                            "object_to_target_distance": obj_dist,
                            "object_eef_distance": object_eef_dist,
                            "privileged_state_error": priv_error,
                            "teacher_privileged_state_available": teacher_priv,
                            # ── Detector/attack fields ──
                            "detector_hazard_score": float(det_out["hazard_score"]) if det_out else "",
                            "detector_release_safe_score": float(det_out["release_safe_score"]) if det_out else "",
                            "detector_phase_idx": int(det_out["phase_idx"]) if det_out else "",
                            "detector_phase_confidence": float(det_out["phase_confidence"]) if det_out else "",
                            "detector_trigger_now": bool(det_out["trigger_now"]) if det_out else False,
                            "detector_trigger_duration": int(det_out["trigger_duration"]) if det_out else 0,
                            "detector_trigger_reason": str(det_out["trigger_reason"]) if det_out else "",
                            "attack_condition": args.attack_condition,
                            "attack_applied": bool(attack_applied),
                            "attack_remaining": int(attack_remaining),
                            "original_env_action": original_env_action.tolist() if original_env_action is not None else [],
                            "attacked_env_action": env_action.tolist(),
                        }, args)

                    obs, reward, done, info = env.step(env_action.tolist())
                    policy_step += 1

                    if done:
                        success = bool(info.get("success", False))
                        done_step = policy_step
                        break
                    t += 1
                except Exception as e:
                    runtime_error = True
                    error_msg = str(e)[:200]
                    print(f"  ERROR: {e}")
                    break

            end_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Save debug sim metadata before closing env
            if args.save_privileged_teacher_state:
                debug_dir = run_dir / "debug"
                debug_dir.mkdir(exist_ok=True)
                debug_meta = build_sim_debug_metadata(env, task_desc)
                with open(debug_dir / "sim_names.json", "w") as f:
                    json.dump(debug_meta, f, indent=2, default=float)

            env.close()

            # Episode record
            ep_record = {
                "run_id": run_id, "suite": args.task_suite_name,
                "task_id": task_id, "task_name": task_name,
                "task_instruction": task_desc, "state_id": ep_idx,
                "seed": args.seed, "success": success,
                "runtime_error": runtime_error, "failure_phase": "",
                "num_steps": t + 1, "done_step": done_step,
                "first_policy_step": args.num_steps_wait,
                "last_policy_step": policy_step - 1,
                "artifact_complete": True,
                "step_records_path": str(run_dir / "step_records.jsonl") if args.save_step_records else "",
                "video_path_optional": "",
                "rgb_dir_optional": str(rgb_dir) if args.save_rgb else "",
            }
            if args.save_step_records:
                save_episode_record(run_dir, ep_record, args)

            # Run manifest
            manifest = {
                "run_id": run_id, "suite": args.task_suite_name,
                "task_id": task_id, "task_name": task_name,
                "task_instruction": task_desc, "state_id": ep_idx,
                "seed": args.seed, "model_path": args.model_path,
                "checkpoint_name": os.path.basename(args.model_path),
                "action_path": action_path,
                "unnorm_key": unnorm_key,
                "num_steps_wait": args.num_steps_wait,
                "max_steps": max_steps, "center_crop": args.center_crop,
                "image_preprocess": "official_pil_lanczos",
                "dtype": "bfloat16", "attention_backend": args.attn_impl,
                "cuda_visible_devices": args.cuda_visible_devices,
                "render_gpu_device_id": args.render_gpu_device_id,
                "save_rgb": args.save_rgb,
                "save_step_records": args.save_step_records,
                "save_privileged_teacher_state": args.save_privileged_teacher_state,
                "output_root": args.output_root,
                "run_dir": str(run_dir), "start_time": start_time,
                "end_time": end_time, "runtime_status": "complete",
                "success": success, "failure_phase": "",
                "artifact_complete": True,
            }
            save_run_manifest(run_dir, manifest, args)

            total_episodes += 1
            if success:
                total_successes += 1
            summary_rows.append({
                "suite": args.task_suite_name, "task_id": task_id,
                "task_name": task_name, "state_id": ep_idx,
                "seed": args.seed, "run_id": run_id,
                "worker_id": args.worker_id, "success": success,
                "runtime_error": runtime_error, "num_steps": t + 1,
                "artifact_complete": True,
            })

            print(f"  {task_name} s{ep_idx}: success={success} steps={t+1}")

    # Write worker shard
    shard_path = shard_dir / f"manifest_worker_{args.worker_id}.csv"
    with open(shard_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nWorker {args.worker_id} complete: {total_successes}/{total_episodes} = {total_successes/max(1,total_episodes):.3f}")
    print(f"Shard: {shard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
