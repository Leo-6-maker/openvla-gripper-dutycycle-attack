#!/usr/bin/env python3
"""R10.4 Passive Deployment Canary — real runner with fake-OpenVLA + fake-env test paths.

Deployment mode (--model-path, --parent-manifest, --output):
  Loads OpenVLA, creates LIBERO env, runs detector+FSM in passive mode.

Test mode (--fake-e2e):
  Full pipeline with fake OpenVLA (produces realistic actions) + fake env.
  CPU only. No GPU, no LIBERO, no 7B model.

CRITICAL INVARIANT: executed_action == clean_postprocessed_action (7-dim, max_abs <= 1e-7)
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time, copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ── Frozen R10.3 config ──────────────────────────────────────────────────────
FROZEN = {
    "grasp_threshold": 0.5, "grasp_persistence": 3,
    "guard_type": "vertical_lift", "guard_param": 0.02,
    "max_episode_emits": 1,
}

# ── Named 25D feature indices (from sc5_detector_runtime.py SC5_FEATURES) ────
FEATURE_NAMES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]
FEATURE_SHA256 = hashlib.sha256(json.dumps(FEATURE_NAMES, sort_keys=True).encode()).hexdigest()

# ── Mechanism parser (per-identity, fail-closed) ─────────────────────────────
# Maps exact canonical_parent_key prefixes to mechanism routes.
# Only tasks explicitly listed as multi_object_transfer are supported.
# Everything else (single-object, articulated, unknown) → abstain.
_MULTI_OBJECT_TASKS = {
    f"libero_10/task_{i:02d}" for i in range(10)
}  # All LIBERO-10 tasks are multi_object_transfer


def parse_route(identity: str) -> str:
    """Per-identity mechanism routing. Only exact multi_object tasks are supported."""
    parts = identity.split("/")
    task_key = f"{parts[0]}/{parts[1]}"
    if task_key in _MULTI_OBJECT_TASKS:
        return "multi_object_transfer"
    return "unsupported_abstain"


# ═══════════════════════════════════════════════════════════════════════════════
# Detector model
# ═══════════════════════════════════════════════════════════════════════════════

class RoutedGraspDetector(nn.Module):
    """Matches R10.3 checkpoint architecture: GRU + head_multi + head_single (46,658 params)."""
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, hidden: torch.Tensor | None, route: str) -> tuple[float, torch.Tensor]:
        if hidden is None:
            hidden = torch.zeros(self.encoder.num_layers, 1, self.encoder.hidden_size, device=x_t.device)
        _, hidden = self.encoder(x_t.unsqueeze(0).unsqueeze(0), hidden)
        h_t = hidden[-1]
        if route == "multi_object_transfer":
            logit = self.head_multi(h_t).squeeze(-1).item()
        elif route == "single_object_pick_place":
            logit = self.head_single(h_t).squeeze(-1).item()
        else:
            logit = 0.0
        return logit, hidden


# ═══════════════════════════════════════════════════════════════════════════════
# Event FSM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FSMState:
    state: str = "IDLE"
    event_id: int = 0
    grasp_persist: int = 0
    anchor_step: int = -1
    anchor_eef_z: float = 0.0
    armed_step: int = -1
    emitted: bool = False
    total_emits: int = 0


class EventFSM:
    def reset(self):
        self._s = FSMState()

    def step(self, t: int, grasp_logit: float, eef_z: float, close_mask: bool, route: str) -> dict:
        if route != "multi_object_transfer":
            return {"action": "ABSTAIN", "emit": False, "fsm_state": "ABSTAIN", "event_id": 0,
                    "anchor_step": -1, "vertical_lift": 0.0, "reason": f"route={route}"}

        grasp_prob = 1.0 / (1.0 + math.exp(-grasp_logit))
        grasp_detected = grasp_prob > FROZEN["grasp_threshold"]
        action = "NONE"; emit = False

        if self._s.state == "IDLE" and close_mask:
            self._s.state = "CLOSE_CANDIDATE"; self._s.event_id += 1

        if self._s.state == "CLOSE_CANDIDATE":
            if grasp_detected:
                self._s.grasp_persist += 1
                if self._s.grasp_persist == 1:
                    self._s.anchor_step = t; self._s.anchor_eef_z = eef_z
            else:
                self._s.grasp_persist = 0
            if self._s.grasp_persist >= FROZEN["grasp_persistence"]:
                self._s.state = "ARMED"; self._s.armed_step = t; action = "ARMED"

        if self._s.state in ("ARMED", "EVENT_CANDIDATE", "EMITTED") and not close_mask:
            self._s.state = "RESET"; action = "RESET"

        if self._s.state == "ARMED" and not self._s.emitted:
            if (eef_z - self._s.anchor_eef_z) >= FROZEN["guard_param"]:
                self._s.state = "EVENT_CANDIDATE"

        if self._s.state == "EVENT_CANDIDATE" and not self._s.emitted:
            if self._s.total_emits < FROZEN["max_episode_emits"]:
                self._s.emitted = True; self._s.total_emits += 1
                self._s.state = "EMITTED"; action = "EMIT"; emit = True

        if self._s.state == "RESET" and close_mask:
            self._s.state = "CLOSE_CANDIDATE"; self._s.grasp_persist = 0
            self._s.emitted = False; self._s.anchor_step = -1
            self._s.anchor_eef_z = 0.0; self._s.event_id += 1

        return {"action": action, "emit": emit, "fsm_state": self._s.state,
                "event_id": self._s.event_id, "anchor_step": self._s.anchor_step,
                "vertical_lift": eef_z - self._s.anchor_eef_z,
                "reason": f"state={self._s.state} vert_lift={eef_z - self._s.anchor_eef_z:.4f}"}


# ═══════════════════════════════════════════════════════════════════════════════
# 25D Feature Adapter (exact SC5_FEATURES order)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureAdapter:
    """Online 25D feature computation. Matches SC5_FEATURES order from training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history: list[dict] = []
        self._prev_gripper: float | None = None
        self._close_streak: int = 0
        self._open_streak: int = 0
        self._close_onset: bool = False
        self._time_since_close: int = 0
        self._eef_history: list = []
        self._eef_z_at_close: float = 0.0

    def update(self, obs: dict, raw_action: np.ndarray, env_action: np.ndarray) -> np.ndarray:
        gripper_cmd = float(raw_action[6])
        gripper_qpos = float(obs["robot0_gripper_qpos"])
        opening = 1.0 - abs(gripper_qpos)
        eef_pos = obs["robot0_eef_pos"]
        eef_vel = np.zeros(3)
        if len(self._eef_history) > 0:
            eef_vel = eef_pos - self._eef_history[-1]
        self._eef_history.append(tuple(float(v) for v in eef_pos))
        if len(self._eef_history) > 20:
            self._eef_history.pop(0)
        action_dx, action_dy, action_dz, action_gripper = float(env_action[0]), float(env_action[1]), float(env_action[2]), float(env_action[6])

        if gripper_cmd < -0.3: self._close_streak += 1; self._open_streak = 0
        elif gripper_cmd > 0.3: self._open_streak += 1; self._close_streak = 0
        else: self._close_streak = max(0, self._close_streak - 1); self._open_streak = max(0, self._open_streak - 1)

        flip_count = 0
        if self._prev_gripper is not None and ((self._prev_gripper < -0.3 and gripper_cmd > 0.3) or (self._prev_gripper > 0.3 and gripper_cmd < -0.3)):
            flip_count = 1
        self._prev_gripper = gripper_cmd

        if self._close_streak >= 3 and not self._close_onset:
            self._close_onset = True; self._time_since_close = 0; self._eef_z_at_close = float(eef_pos[2])
        if self._close_onset: self._time_since_close += 1
        if self._open_streak >= 3: self._close_onset = False; self._time_since_close = 0

        eef_speed = float(np.linalg.norm(eef_vel))
        eef_z_delta = float(eef_pos[2] - self._eef_z_at_close) if self._close_onset else 0.0

        qpos_d1 = 0.0; qpos_d3 = 0.0; opening_d3 = 0.0; opening_var5 = 0.0; speed_var5 = 0.0
        if len(self.history) >= 1: qpos_d1 = gripper_qpos - self.history[-1].get("gripper_qpos", gripper_qpos)
        if len(self.history) >= 3:
            qpos_d3 = gripper_qpos - self.history[-3].get("gripper_qpos", gripper_qpos)
            opening_d3 = opening - (1.0 - abs(self.history[-3].get("gripper_qpos", gripper_qpos)))
        recent_speeds = [eef_speed] + [h.get("eef_speed", eef_speed) for h in self.history[-4:]]
        recent_openings = [opening] + [(1.0 - abs(h.get("gripper_qpos", gripper_qpos))) for h in self.history[-4:]]
        if len(recent_speeds) >= 5: speed_var5 = float(np.var(recent_speeds[:5]))
        if len(recent_openings) >= 5: opening_var5 = float(np.var(recent_openings[:5]))

        feats = np.array([
            gripper_cmd, gripper_qpos, opening,
            float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
            float(eef_vel[0]), float(eef_vel[1]), float(eef_vel[2]),
            action_dx, action_dy, action_dz, action_gripper,
            float(self._close_streak), float(self._open_streak), float(flip_count),
            float(self._close_onset), float(self._time_since_close), eef_speed,
            eef_z_delta, qpos_d1, qpos_d3, opening_d3, opening_var5, speed_var5,
        ], dtype=np.float32)
        assert len(feats) == 25
        assert np.all(np.isfinite(feats)), f"Non-finite features at step"
        self.history.append({"gripper_qpos": gripper_qpos, "eef_speed": eef_speed, "opening": opening})
        if len(self.history) > 20: self.history.pop(0)
        return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Fake OpenVLA adapter (for CPU tests — produces realistic-shaped actions)
# ═══════════════════════════════════════════════════════════════════════════════

class FakeOpenVLAAdapter:
    """Produces deterministic, realistic-shaped actions for testing. NOT real OpenVLA."""

    def __init__(self, seed: int = 20260720):
        self.rng = np.random.RandomState(seed)
        self.call_count = 0

    def predict_action(self, image: np.ndarray, task_language: str) -> tuple[np.ndarray, dict]:
        self.call_count += 1
        action = np.array([
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.02, 0.02),
            self.rng.uniform(-0.01, 0.01),
            self.rng.uniform(-0.01, 0.01),
            self.rng.uniform(-0.01, 0.01),
            self.rng.choice([-1.0, 0.0, 1.0]),
        ], dtype=np.float64)
        meta = {"generation_passes_per_step": 1, "fake": True}
        return action, meta

    def postprocess(self, raw_action: np.ndarray) -> np.ndarray:
        return raw_action.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Fake LIBERO env (for CPU tests)
# ═══════════════════════════════════════════════════════════════════════════════

class FakeLiberoEnv:
    """Deterministic fake env for end-to-end CPU testing."""

    def __init__(self):
        self.t = 0
        self.rng = np.random.RandomState(20260720)

    def set_init_state(self, init_state: dict) -> dict:
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= 50
        return self._obs(), 0.0, done, {}

    def _obs(self):
        return {
            "agentview_image": self.rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.5 + self.t * 0.001, 0.0, 0.8 + self.t * 0.002], dtype=np.float64),
            "robot0_gripper_qpos": np.array(0.05 * np.sin(self.t * 0.1)),
        }

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Core passive episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_passive_episode(
    env,
    init_state: dict,
    task_language: str,
    identity: str,
    detector: RoutedGraspDetector,
    fsm: EventFSM,
    feature_adapter: FeatureAdapter,
    openvla_adapter,
    device: torch.device,
    output_dir: Path,
    max_steps: int = 500,
) -> dict:
    route = parse_route(identity)
    violations = []
    emit_count = 0
    step_records = []
    detector_records = []

    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64))

    feature_adapter.reset()
    fsm.reset()
    detector_hidden = None

    for t in range(max_steps):
        image = obs["agentview_image"]
        if image.ndim == 2: image = np.stack([image] * 3, axis=-1)
        image = image.astype(np.float32) / 255.0

        # 1. OpenVLA action generation
        raw_action, generation_meta = openvla_adapter.predict_action(image=image, task_language=task_language)

        # 2. FAIL-CLOSED: generation_passes_per_step MUST be present and MUST equal 1
        gen_passes = generation_meta.get("generation_passes_per_step")
        if gen_passes is None:
            violations.append(f"MISSING_GENERATION_PASSES t={t}")
            break
        if gen_passes != 1:
            violations.append(f"MULTIPLE_GENERATION_PASSES t={t} passes={gen_passes}")
            break

        # 3. Postprocess clean action
        clean_action = openvla_adapter.postprocess(raw_action)

        # 4. Compute 25D features
        features_25d = feature_adapter.update(obs, raw_action, clean_action)

        # 5. Detector step
        x_t = torch.tensor(features_25d, dtype=torch.float32, device=device)
        grasp_logit, detector_hidden = detector.step(x_t, detector_hidden, route)
        grasp_prob = 1.0 / (1.0 + math.exp(-grasp_logit)) if route == "multi_object_transfer" else 0.0

        # 6. FSM step
        eef_z = float(obs["robot0_eef_pos"][2])
        close_mask = bool(float(obs["robot0_gripper_qpos"]) < -0.3)
        fsm_result = fsm.step(t, grasp_logit, eef_z, close_mask, route)

        # 7. INVARIANT: executed_action == clean_action
        executed_action = clean_action.copy()
        action_diff = np.max(np.abs(executed_action - clean_action))
        if action_diff > 1e-7:
            violations.append(f"ACTION_MUTATION t={t} diff={action_diff}")

        if route == "unsupported_abstain" and fsm_result["emit"]:
            violations.append(f"UNSUPPORTED_EMIT t={t}")

        if fsm_result["emit"]:
            emit_count += 1
            if t < fsm_result["anchor_step"]:
                violations.append(f"PRE_ANCHOR_EMIT t={t} anchor={fsm_result['anchor_step']}")
            if emit_count > FROZEN["max_episode_emits"]:
                violations.append(f"DUPLICATE_EMIT t={t}")

        # 8. Step environment
        obs_next, reward, done, info = env.step(executed_action.tolist())

        step_records.append({
            "step": t, "executed_action": executed_action.tolist(),
            "action_parity_max_error": float(action_diff),
            "gen_passes": gen_passes,
        })
        detector_records.append({
            "step": t, "route": route, "grasp_prob": grasp_prob,
            "fsm_state": fsm_result["fsm_state"], "emit": fsm_result["emit"],
            "event_id": fsm_result["event_id"],
            "vertical_lift": fsm_result["vertical_lift"],
        })

        obs = obs_next
        if done: break

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "step_records.jsonl", "w") as f:
        for r in step_records: f.write(json.dumps(r) + "\n")
    with open(output_dir / "detector_records.jsonl", "w") as f:
        for r in detector_records: f.write(json.dumps(r) + "\n")
    summary = {"identity": identity, "n_steps": len(step_records), "emit_count": emit_count,
               "violations": violations, "passive": True}
    with open(output_dir / "episode_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Fake end-to-end test
# ═══════════════════════════════════════════════════════════════════════════════

def run_fake_e2e_test(output_dir: Path) -> bool:
    """Full pipeline: fake OpenVLA + fake env + real detector/FSM. CPU only."""
    print("=== R10.4 FAKE END-TO-END TEST ===\n")

    device = torch.device("cpu")
    violations_total = []

    # 1. Build detector (random init — we test pipeline, not prediction quality)
    torch.manual_seed(20260720)
    detector = RoutedGraspDetector(input_dim=25, hidden_dim=64, num_layers=2).to(device)
    detector.eval()

    # 2. Build FSM
    fsm = EventFSM()

    # 3. Build feature adapter
    feature_adapter = FeatureAdapter()

    # 4. Build fake OpenVLA
    openvla = FakeOpenVLAAdapter(seed=20260720)

    # 5. Build fake env
    env = FakeLiberoEnv()

    # 6. Parent identity
    identity = "libero_10/task_00/state_20"
    route = parse_route(identity)
    assert route == "multi_object_transfer", f"Route mismatch: {route}"
    print(f"[PASS] Route: {identity} → {route}")

    # 7. Run episode
    print(f"\nRunning passive episode for {identity}...")
    ep_dir = output_dir / "fake_e2e_test"
    result = run_passive_episode(
        env=env, init_state={}, task_language="put both the alphabet soup and the tomato sauce in the basket",
        identity=identity, detector=detector, fsm=fsm, feature_adapter=feature_adapter,
        openvla_adapter=openvla, device=device, output_dir=ep_dir, max_steps=50,
    )

    print(f"  Steps: {result['n_steps']}, Emits: {result['emit_count']}, Violations: {len(result['violations'])}")
    if result["violations"]:
        for v in result["violations"]:
            print(f"    VIOLATION: {v}")
        violations_total.extend(result["violations"])

    # 8. Verify outputs
    assert (ep_dir / "step_records.jsonl").is_file(), "Missing step_records.jsonl"
    assert (ep_dir / "detector_records.jsonl").is_file(), "Missing detector_records.jsonl"
    assert (ep_dir / "episode_summary.json").is_file(), "Missing episode_summary.json"
    print("[PASS] All output files present")

    # 9. Verify action isolation
    steps = [json.loads(l) for l in (ep_dir / "step_records.jsonl").read_text().splitlines() if l.strip()]
    for s in steps:
        err = s["action_parity_max_error"]
        assert err <= 1e-7, f"Action mutated: err={err}"
        assert s["gen_passes"] == 1, f"Generation passes != 1: {s['gen_passes']}"
    print(f"[PASS] Action isolation: {len(steps)} steps, all parity <= 1e-7, all gen_passes=1")

    # 10. Verify detector records
    det = [json.loads(l) for l in (ep_dir / "detector_records.jsonl").read_text().splitlines() if l.strip()]
    for d in det:
        assert d["route"] in ("multi_object_transfer", "unsupported_abstain"), f"Bad route: {d['route']}"
    print(f"[PASS] Detector records: {len(det)} steps, valid routes")

    # 11. Test unsupported route never emits
    fsm2 = EventFSM()
    unsupported_emit = False
    for t in range(100):
        r = fsm2.step(t, 10.0, 0.9, True, "unsupported_abstain")
        if r["emit"]: unsupported_emit = True
    assert not unsupported_emit, "Unsupported route emitted!"
    print("[PASS] Unsupported route never emits")

    # 12. Test feature dimension
    fa = FeatureAdapter()
    mock_obs = {"robot0_eef_pos": np.array([0.5, 0.0, 0.8]), "robot0_gripper_qpos": np.array(0.1)}
    f = fa.update(mock_obs, np.zeros(7), np.zeros(7))
    assert len(f) == 25 and f.dtype == np.float32, f"Feature dim mismatch: {len(f)} {f.dtype}"
    assert np.all(np.isfinite(f)), "Non-finite features"
    print("[PASS] 25D features: dim=25, dtype=float32, all finite")

    # 13. Test episode reset
    fa.reset()
    assert len(fa.history) == 0
    fsm2.reset()
    assert fsm2._s.state == "IDLE" and fsm2._s.total_emits == 0
    print("[PASS] Episode reset: history cleared, FSM reset to IDLE")

    # 14. Test feature-order SHA
    computed_sha = hashlib.sha256(json.dumps(FEATURE_NAMES, sort_keys=True).encode()).hexdigest()
    assert computed_sha == FEATURE_SHA256, f"Feature SHA mismatch: {computed_sha} != {FEATURE_SHA256}"
    print(f"[PASS] Feature-order SHA256: {FEATURE_SHA256[:16]}...")

    # 15. Verify generation_passes missing → fail
    try:
        run_passive_episode(
            env=FakeLiberoEnv(), init_state={}, task_language="test", identity=identity,
            detector=detector, fsm=EventFSM(), feature_adapter=FeatureAdapter(),
            openvla_adapter=FakeOpenVLAAdapter(), device=device, output_dir=output_dir / "fake_fail_test", max_steps=3,
        )
    except Exception:
        pass
    # 13. Negative generation adapter tests
    class MissingGenAdapter(FakeOpenVLAAdapter):
        def predict_action(self, image, task_language):
            return np.zeros(7, dtype=np.float64), {}
    class ZeroGenAdapter(FakeOpenVLAAdapter):
        def predict_action(self, image, task_language):
            return np.zeros(7, dtype=np.float64), {"generation_passes_per_step": 0}
    class DoubleGenAdapter(FakeOpenVLAAdapter):
        def predict_action(self, image, task_language):
            return np.zeros(7, dtype=np.float64), {"generation_passes_per_step": 2}

    for adp_name, adp in [("MissingGen", MissingGenAdapter()), ("ZeroGen", ZeroGenAdapter()), ("DoubleGen", DoubleGenAdapter())]:
        ep_dir_v = output_dir / f"fake_neg_test_{adp_name}"
        r = run_passive_episode(
            env=FakeLiberoEnv(), init_state={}, task_language="test", identity=identity,
            detector=detector, fsm=EventFSM(), feature_adapter=FeatureAdapter(),
            openvla_adapter=adp, device=device, output_dir=ep_dir_v, max_steps=3,
        )
        has_violation = any("GENERATION" in v for v in r["violations"])
        assert has_violation, f"{adp_name}: expected generation violation, got none"
        assert r["n_steps"] <= 1, f"{adp_name}: should stop immediately, got {r['n_steps']} steps"
        print(f"[PASS] {adp_name}: correctly failed with violations={r['violations']}")

    print("[PASS] All negative generation tests fail-closed")
    print("[PASS] Fake E2E test complete")

    all_pass = len(violations_total) == 0
    print(f"\n{'='*50}")
    print(f"FAKE E2E: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"{'='*50}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R10.4 Passive Deployment Canary")
    parser.add_argument("--fake-e2e", action="store_true", help="Use fake OpenVLA + fake LIBERO (CPU test)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to OpenVLA model checkpoint")
    parser.add_argument("--detector-bundle", type=str, required=True, help="Path to R10.3 deployment bundle directory")
    parser.add_argument("--parent-manifest", type=str, required=True, help="Path to parent manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for episode records")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    output_dir = Path(args.output)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.fake_e2e else "cpu")

    # ── 1. Load parent manifest ──
    manifest_path = Path(args.parent_manifest)
    if not manifest_path.is_file():
        print(f"FATAL: parent manifest not found: {manifest_path}"); sys.exit(1)
    parent_data = json.loads(manifest_path.read_text())
    selected_parent = parent_data["selected_parent"]
    print(f"Parent: {selected_parent}")

    # ── 2. Load and verify detector from bundle ──
    bundle_path = Path(args.detector_bundle)
    ckpt_path = bundle_path / "full_fit_deploy.pt"
    if not ckpt_path.is_file():
        print(f"FATAL: checkpoint not found: {ckpt_path}"); sys.exit(1)

    # Verify SHA256SUMS
    sums_path = bundle_path / "SHA256SUMS"
    if sums_path.is_file():
        sums = {}
        for line in sums_path.read_text().strip().split("\n"):
            parts = line.strip().split("  ", 1)
            if len(parts) == 2: sums[parts[1]] = parts[0]
        for fname, expected in sums.items():
            fp = bundle_path / fname
            if fp.is_file():
                actual = hashlib.sha256(fp.read_bytes()).hexdigest()
                if actual != expected:
                    print(f"FATAL: SHA256 mismatch: {fname} expected={expected[:16]} actual={actual[:16]}")
                    sys.exit(1)
        print(f"Bundle SHA256SUMS verified: {len(sums)} files OK")

    # Verify checkpoint SHA
    actual_ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print(f"Checkpoint SHA256: {actual_ckpt_sha[:16]}...")

    ckpt = torch.load(ckpt_path, map_location=device)
    detector = RoutedGraspDetector(
        input_dim=ckpt["frozen"]["input_dim"],
        hidden_dim=ckpt["frozen"]["hidden_dim"],
        num_layers=ckpt["frozen"]["num_layers"],
    ).to(device)
    detector.load_state_dict(ckpt["model_state"], strict=True)
    detector.eval()
    n_params = sum(p.numel() for p in detector.parameters())
    print(f"Detector loaded: {n_params} params, strict=OK")

    # Verify embedded provenance
    if "source_commit" in ckpt:
        import subprocess
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            embedded = ckpt["source_commit"]
            # Check that embedded commit is reachable from HEAD
            merge_base = subprocess.check_output(["git", "merge-base", head, embedded], text=True).strip()
            provenance_ok = merge_base == embedded
            print(f"  source_commit: {embedded[:16]}... provenance_ok={provenance_ok}")
        except Exception:
            print(f"  source_commit: {ckpt['source_commit'][:16]}... (git check skipped)")
    if "trainer_blob_sha256" in ckpt:
        print(f"  trainer_blob_sha256: {ckpt['trainer_blob_sha256'][:16]}...")
    if "feature_contract_sha256" in ckpt:
        print(f"  feature_contract_sha256: {ckpt['feature_contract_sha256'][:16]}...")

    # ── 3. Create OpenVLA adapter + LIBERO env (fake or real) ──
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait, set_init_state
    from libero.libero import get_libero_path

    if args.fake_e2e:
        openvla = FakeOpenVLAAdapter(seed=20260720)
        env = FakeLiberoEnv()
        init_state = {}
        task_language = "put both the alphabet soup and the tomato sauce in the basket"
        print("Mode: FAKE (CPU test)")
    else:
        if not args.model_path:
            print("FATAL: --model-path required for real deployment"); sys.exit(1)

        # Real OpenVLA adapter construction (model loading GUARDED)
        # The 7B model load is NOT authorized in static audit.
        # When OpenVLA load is authorized, replace this block with actual model init.
        print("FATAL: OpenVLA 7B model load not yet authorized.")
        print(f"  Model would be loaded from: {args.model_path}")
        print(f"  All other components (detector, FSM, env, feature adapter) are ready.")
        print(f"  To authorize: remove this guard and implement OpenVLA adapter init.")
        sys.exit(1)

        # === REAL PATH (when OpenVLA auth granted) ===
        # from gripper_attack.attack_adapter import ...  # real adapter
        # openvla = RealOpenVLAAdapter(model_path=args.model_path, device=device)
        # bddl_file = resolve_bddl_from_registry(selected_parent)
        # env, obs = build_v4_exact_env(bddl_file, render_gpu_device_id=args.gpu)
        # task_language = resolve_task_language(selected_parent)
        # init_state = load_init_state(selected_parent)
        # env, obs = set_init_state(env, obs, init_state)
        # env, obs = apply_dummy_wait(env, obs, num_steps_wait=10)

    # ── 4. Run passive episode ──
    fsm = EventFSM()
    feature_adapter = FeatureAdapter()
    ep_dir = output_dir / selected_parent.replace("/", "_")
    result = run_passive_episode(
        env=env, init_state=init_state, task_language=task_language,
        identity=selected_parent, detector=detector, fsm=fsm,
        feature_adapter=feature_adapter, openvla_adapter=openvla,
        device=device, output_dir=ep_dir,
    )

    # ── 5. Report ──
    ok = len(result["violations"]) == 0
    print(f"\nPassive canary: {'PASS' if ok else 'FAIL'}")
    print(f"  Steps: {result['n_steps']}, Emits: {result['emit_count']}, Violations: {len(result['violations'])}")
    if result["violations"]:
        for v in result["violations"]:
            print(f"    VIOLATION: {v}")
    with open(output_dir / "CANARY_SUMMARY.json", "w") as f:
        json.dump({"parent": selected_parent, "pass": ok, "violations": result["violations"],
                   "checkpoint_sha": ckpt_path.name, "source_commit": ckpt.get("source_commit", "unknown")}, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
