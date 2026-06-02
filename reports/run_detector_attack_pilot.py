#!/usr/bin/env python3
"""Matched Attack Pilot: detector-triggered attack on GPU0+7.

Attack conditions use SAME detector trigger. Attack module only consumes:
  trigger_now, trigger_duration, trigger_reason, trigger_confidence
"""
import os, sys, json, time, csv, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO))

# ── Detector model ──
HISTORY_LEN = 16; HIDDEN_DIM = 64; TCN_LAYERS = 3
ALLOWED_PROPRIO = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]
N_PROPRIO = len(ALLOWED_PROPRIO)

class CausalTCN(torch.nn.Module):
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
            r = x; x = F.relu(c(x)); x = x[:, :, -r.shape[2]:] + r; x = self.drop(x)
        x = x[:, :, -1]
        return self.ph(x), self.hz(x).squeeze(-1), self.rl(x).squeeze(-1)


class OnlineDetector:
    def __init__(self, model_path, device="cpu", hazard_th=0.1, trig_dur=5, cooldown=0):
        self.model = CausalTCN(N_PROPRIO, HIDDEN_DIM, 8, TCN_LAYERS).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
        self.hazard_th = hazard_th
        self.trig_dur = trig_dur
        self.cooldown = cooldown
        self.history = []
        self.hazard_buf = []
        self.cooldown_ctr = 0

    def update(self, features):
        self.history.append(features)
        if len(self.history) > HISTORY_LEN:
            self.history = self.history[-HISTORY_LEN:]
        hist = np.array(self.history, dtype=np.float32)
        if hist.shape[0] < HISTORY_LEN:
            pad = np.zeros((HISTORY_LEN - hist.shape[0], hist.shape[1]), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ph, hl, rl = self.model(x)
        hs = float(torch.sigmoid(hl).item())
        rs = float(torch.sigmoid(rl).item())
        phase_p = torch.softmax(ph, dim=-1).cpu().numpy()[0]
        pi = int(np.argmax(phase_p))
        conf = float(phase_p[pi])

        self.hazard_buf.append(hs)
        if len(self.hazard_buf) > self.trig_dur:
            self.hazard_buf = self.hazard_buf[-self.trig_dur:]

        trigger_now = False; trigger_reason = ""
        if self.cooldown_ctr > 0:
            self.cooldown_ctr -= 1
        elif len(self.hazard_buf) >= self.trig_dur and all(h > self.hazard_th for h in self.hazard_buf):
            trigger_now = True
            trigger_reason = f"consecutive_{self.trig_dur}"
            self.cooldown_ctr = self.cooldown
        return {"hazard_score": hs, "release_safe_score": rs, "phase_idx": pi,
                "phase_confidence": conf, "trigger_now": trigger_now,
                "trigger_duration": self.trig_dur if trigger_now else 0,
                "trigger_reason": trigger_reason}


def attack_action(clean_action, condition, rng):
    """Modify action based on attack condition. Only gripper channel is touched."""
    action = clean_action.copy()
    if condition == "clean_rerun":
        return action
    elif condition == "oracle_open":
        action[-1] = 1.0  # Force gripper fully open
    elif condition == "random_control":
        action[-1] = 1.0 if rng.random() > 0.5 else -1.0  # Random open/close
    elif condition == "VIS_targeted":
        # Simplified VIS: invert gripper with small noise
        noise = rng.normal(0, 0.05)
        action[-1] = -clean_action[-1] + noise
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    ap.add_argument("--detector_path", default="/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt")
    ap.add_argument("--condition", required=True, choices=["clean_rerun", "oracle_open", "random_control", "VIS_targeted"])
    ap.add_argument("--cuda_visible_devices", default="0,7")
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--task_start", type=int, default=0)
    ap.add_argument("--state_start", type=int, default=0)
    ap.add_argument("--num_states", type=int, default=5)
    ap.add_argument("--task_name_filter", default="cream_cheese")  # or milk
    ap.add_argument("--output_root", default="/data/liuyu/outputs/milestone_2g_object100_attack_pilot_20260529")
    ap.add_argument("--hazard_threshold", type=float, default=0.1)
    ap.add_argument("--trigger_duration", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["MUJOCO_GL"] = "egl"

    # Only import LIBERO after CUDA_VISIBLE_DEVICES is set
    import libero
    from src.utils.libero_privileged_state import extract_teacher_privileged_state

    print(f"Attack Pilot: {args.condition}")
    print(f"  GPUs: {args.cuda_visible_devices}")
    print(f"  Detector: ht={args.hazard_threshold}, dur={args.trigger_duration}")

    if args.dry_run:
        print("[DRY RUN] Config only.")
        return

    # ── Load detector ──
    print("Loading detector...")
    detector = OnlineDetector(args.detector_path, device="cpu",
                              hazard_th=args.hazard_threshold, trig_dur=args.trigger_duration)
    print(f"  Detector ready ({sum(p.numel() for p in detector.model.parameters())} params)")

    # ── Load OpenVLA ──
    print("Loading OpenVLA...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print("  OpenVLA loaded")

    # ── Setup LIBERO ──
    print("Setting up LIBERO...")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()
    task_names = [task.name for task in task_suite.tasks]
    print(f"  {len(task_names)} tasks available")

    # Filter tasks
    if args.task_name_filter:
        task_names = [t for t in task_names if args.task_name_filter in t.lower()]
        print(f"  Filtered to {len(task_names)} tasks: {task_names}")

    task_names = task_names[args.task_start:args.task_start + 1]  # One task at a time
    if not task_names:
        print("No tasks match filter!")
        return

    os.makedirs(args.output_root + "/tables", exist_ok=True)
    os.makedirs(args.output_root + "/logs", exist_ok=True)

    rng = np.random.RandomState(42)
    all_results = []

    for task_name in task_names:
        task_id = [i for i, t in enumerate(task_suite.tasks) if t.name == task_name][0]
        task = task_suite.tasks[task_id]
        task_desc = task.language
        print(f"\nTask: {task_name} ({task_desc})")

        # Get initial states
        env_args = {"task_name": task_name, "problem_name": "LIBERO_OBJECT", "bddl_file_name": task.problem}
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        initial_states = task_suite.get_task_init_states(task_id)

        for state_id in range(args.state_start, min(args.state_start + args.num_states, len(initial_states))):
            obs = env.reset()
            obs = env.set_init_state(initial_states[state_id])
            detector.history = []; detector.hazard_buf = []; detector.cooldown_ctr = 0
            step_logs = []
            success = False

            for t in range(410):  # max 400 + 10 wait
                if t < 10:
                    obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                    continue

                # Get VLA action (simplified — uses official preprocessing)
                img = obs["agentview_image"][::-1, ::-1]
                w, h = img.shape[1], img.shape[0]
                s = 0.9 ** 0.5
                cw, ch = max(1, int(w * s)), max(1, int(h * s))
                l, t_crop = (w - cw) // 2, (h - ch) // 2
                img_crop = img[t_crop:t_crop + ch, l:l + cw]
                img_pil = Image.fromarray(img_crop).resize((224, 224), Image.LANCZOS)

                prompt = f"In: {task_desc}\nOut:"
                inputs = processor(prompt, img_pil).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                            pad_token_id=processor.tokenizer.eos_token_id)
                action_text = processor.tokenizer.decode(outputs[0]).split("Out:")[-1].strip()
                try:
                    parts = action_text.replace(",", " ").split()
                    raw_action = np.array([float(p) for p in parts[:7]], dtype=np.float32)
                except:
                    raw_action = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

                # Normalize gripper
                env_action = raw_action.copy()
                env_action[-1] = 2.0 * env_action[-1] - 1.0
                env_action[-1] = np.sign(env_action[-1])
                if env_action[-1] == 0: env_action[-1] = 1.0
                env_action[-1] = -env_action[-1]  # invert

                # ── Detector update ──
                eef = obs["robot0_eef_pos"]
                gripper_qpos = float(obs["robot0_gripper_qpos"][0])
                det_feats = np.array([
                    float(obs.get("gripper_command", 0)),
                    gripper_qpos,
                    float(obs.get("gripper_width", 0)),
                    float(eef[0]), float(eef[1]), float(eef[2]),
                    0.0, 0.0, 0.0,  # eef vel (simplified)
                    float(raw_action[0]), float(raw_action[1]), float(raw_action[2]), float(raw_action[3]),
                ], dtype=np.float32)
                det_out = detector.update(det_feats)

                # ── Attack injection ──
                attack_applied = False
                original_action = env_action.copy()
                if det_out["trigger_now"]:
                    env_action = attack_action(env_action, args.condition, rng)
                    attack_applied = True

                # ── Step ──
                obs, reward, done, info = env.step(env_action.tolist())
                success = bool(done and info.get("success", False))

                step_logs.append({
                    "step": t, "trigger": det_out["trigger_now"],
                    "hazard": det_out["hazard_score"], "release": det_out["release_safe_score"],
                    "phase": det_out["phase_idx"], "attack_applied": attack_applied,
                    "action_orig": original_action.tolist(), "action_attacked": env_action.tolist(),
                    "success": success,
                })

                if done:
                    break

            env.close()

            # Save per-episode log
            ep_key = f"{task_name}::{state_id}::{args.condition}"
            with open(args.output_root + f"/logs/ep_{ep_key.replace('::','_').replace(' ','_')[:80]}.json", "w") as f:
                json.dump({"task": task_name, "state": state_id, "condition": args.condition,
                          "success": success, "steps": len(step_logs), "logs": step_logs}, f, indent=2)

            all_results.append({"task": task_name, "state": state_id, "condition": args.condition,
                               "success": success, "n_steps": len(step_logs),
                               "n_triggers": sum(1 for s in step_logs if s["trigger"]),
                               "n_attacks": sum(1 for s in step_logs if s["attack_applied"])})
            print(f"  s{state_id}: success={success} steps={len(step_logs)} "
                  f"trig={sum(1 for s in step_logs if s['trigger'])} "
                  f"attk={sum(1 for s in step_logs if s['attack_applied'])}")

    # ── Write summary ──
    with open(args.output_root + "/tables/attack_pilot_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    # Aggregate
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r["success"])
    print("\n=== ATTACK PILOT SUMMARY ===")
    for cond, succs in sorted(by_cond.items()):
        print(f"  {cond}: SR={sum(succs)}/{len(succs)} ({sum(succs)/len(succs):.2f})")

    print(f"\nOutput: {args.output_root}")
    with open(args.output_root + "/reports/ATTACK_PILOT_STATUS.md", "w") as f:
        f.write(f"# Attack Pilot Status\n\nCondition: {args.condition}\n\n")
        for cond, succs in sorted(by_cond.items()):
            f.write(f"- {cond}: SR={sum(succs)}/{len(succs)}\n")


if __name__ == "__main__":
    main()
