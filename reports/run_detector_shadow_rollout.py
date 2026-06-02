#!/usr/bin/env python3
"""Shadow-mode detector rollout: run clean LIBERO eval while detector logs triggers.
Detector outputs are recorded but NOT executed — actions remain clean.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
            torch.nn.Conv1d(h_dim, h_dim, 3, padding=2 ** (i + 1), dilation=2 ** i)
            for i in range(n_l)
        ])
        self.drop = torch.nn.Dropout(do)
        self.ph = torch.nn.Linear(h_dim, n_ph)
        self.hz = torch.nn.Linear(h_dim, 1)
        self.rl = torch.nn.Linear(h_dim, 1)

    def forward(self, x):
        x = self.proj(x); x = x.transpose(1, 2)
        for c in self.convs:
            r = x; x = F.relu(c(x)); x = x[:, :, -r.shape[2]:] + r; x = self.drop(x)
        x = x[:, :, -1]
        return self.ph(x), self.hz(x).squeeze(-1), self.rl(x).squeeze(-1)


class ShadowDetector:
    """Wraps a trained TCN model for online streaming inference."""

    def __init__(self, model_path: str, in_dim: int, device: str = "cpu",
                 hazard_threshold: float = 0.5, trigger_duration: int = 5,
                 cooldown: int = 10):
        self.model = CausalTCN(in_dim, HIDDEN_DIM, 8, TCN_LAYERS)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.device = device
        self.hazard_threshold = hazard_threshold
        self.trigger_duration = trigger_duration
        self.cooldown = cooldown

        # Streaming state
        self.history: list[np.ndarray] = []
        self.hazard_buffer: list[float] = []
        self.trigger_active = False
        self.cooldown_counter = 0

    def update(self, features: np.ndarray) -> dict:
        """Process one timestep. Returns detector output dict."""
        self.history.append(features)
        if len(self.history) > HISTORY_LEN:
            self.history = self.history[-HISTORY_LEN:]

        # Build history tensor
        hist = np.array(self.history, dtype=np.float32)
        if hist.shape[0] < HISTORY_LEN:
            pad = np.zeros((HISTORY_LEN - hist.shape[0], hist.shape[1]), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)

        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            phase_logits, hazard_logit, release_logit = self.model(x)

        hazard_score = float(torch.sigmoid(hazard_logit).item())
        release_score = float(torch.sigmoid(release_logit).item())
        phase_probs = torch.softmax(phase_logits, dim=-1).cpu().numpy()[0]
        phase_idx = int(np.argmax(phase_probs))
        confidence = float(phase_probs[phase_idx])

        self.hazard_buffer.append(hazard_score)
        if len(self.hazard_buffer) > self.trigger_duration:
            self.hazard_buffer = self.hazard_buffer[-self.trigger_duration:]

        # Trigger logic
        trigger_now = False
        trigger_reason = ""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if len(self.hazard_buffer) >= self.trigger_duration:
                if all(h > self.hazard_threshold for h in self.hazard_buffer):
                    trigger_now = True
                    trigger_reason = f"consecutive_{self.trigger_duration}"
                    self.cooldown_counter = self.cooldown

        return {
            "hazard_score": hazard_score,
            "release_safe_score": release_score,
            "phase_idx": phase_idx,
            "phase_confidence": confidence,
            "trigger_now": trigger_now,
            "trigger_duration": self.trigger_duration if trigger_now else 0,
            "trigger_reason": trigger_reason,
            "cooldown_active": self.cooldown_counter > 0,
        }


def extract_proprio_features(obs: dict, action: np.ndarray | None) -> np.ndarray:
    """Extract 13 deployment-allowed proprio features from obs + action."""
    gripper_qpos = float(np.mean(obs.get("robot0_gripper_qpos", [0.0])))
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    feats = [
        float(obs.get("gripper_command", 0.0)),
        gripper_qpos,
        float(obs.get("gripper_width", 0.0)),
        float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
        float(obs.get("robot0_eef_vel_lin", np.zeros(3))[0]),
        float(obs.get("robot0_eef_vel_lin", np.zeros(3))[1]),
        float(obs.get("robot0_eef_vel_lin", np.zeros(3))[2]),
        0.0, 0.0, 0.0, 0.0,  # action placeholders (filled from previous action)
    ]
    if action is not None:
        feats[9] = float(action[0])  # action_dx
        feats[10] = float(action[1])  # action_dy
        feats[11] = float(action[2])  # action_dz
        feats[12] = float(action[3])  # action_gripper
    return np.array(feats, dtype=np.float32)


def build_shadow_config(base_yaml: str, output_root: str) -> dict:
    """Build shadow validation config dict."""
    return {
        "mode": "shadow",
        "detector": {
            "model_type": "ProprioNoStep_TCN",
            "model_path": "models/ProprioNoStep_baseline.pt",
            "input_dim": 13,
            "history_len": 16,
            "hazard_threshold": 0.1,
            "trigger_duration": 5,
            "cooldown": 10,
        },
        "rollout": {
            "suite": "libero_object",
            "tasks": [
                "LIVING_ROOM_SCENE4_pick_up_the_cream_cheese_and_place_it_in_the_basket",
                "LIVING_ROOM_SCENE4_pick_up_the_ketchup_and_place_it_in_the_basket",
                "LIVING_ROOM_SCENE4_pick_up_the_milk_and_place_it_in_the_basket",
                "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_place_it_in_the_basket",
            ],
            "states_per_task": 5,
            "num_trials_per_task": 1,
            "max_steps": 400,
        },
        "output": {
            "root": output_root,
            "save_rgb": True,
            "save_step_records": True,
            "save_shadow_log": True,
        },
        "boundary": {
            "no_attack": True,
            "no_privileged_input": True,
            "no_visual_perturbation": True,
            "no_oracle_open": True,
            "no_random_control": True,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_type", default="ProprioNoStep",
                    choices=["ProprioNoStep", "VisualNoStep", "VisualProprioNoStep"])
    ap.add_argument("--task_suite_name", default="libero_object")
    ap.add_argument("--task_start", type=int, default=0)
    ap.add_argument("--task_count", type=int, default=10)
    ap.add_argument("--num_trials_per_task", type=int, default=10)
    ap.add_argument("--hazard_threshold", type=float, default=0.1)
    ap.add_argument("--trigger_duration", type=int, default=5)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--output_root", default="/data/liuyu/outputs/milestone_2f_object100_online_shadow_validation_20260527")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print(f"Shadow Detector Validation")
    print(f"  Model: {args.model_type} ({args.model_path})")
    print(f"  Suite: {args.task_suite_name}")
    print(f"  Threshold: {args.hazard_threshold}, Duration: {args.trigger_duration}, Cooldown: {args.cooldown}")
    print(f"  Output: {args.output_root}")

    if args.dry_run:
        print("[DRY RUN] Config only — no episodes executed.")
        cfg = build_shadow_config(args.model_path, args.output_root)
        os.makedirs(args.output_root + "/configs", exist_ok=True)
        with open(args.output_root + "/configs/shadow_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Config written to {args.output_root}/configs/shadow_config.json")
        return

    print("Shadow mode not yet implemented — use dry_run to generate config.")
    print("Full implementation requires LIBERO env integration.")


if __name__ == "__main__":
    main()
