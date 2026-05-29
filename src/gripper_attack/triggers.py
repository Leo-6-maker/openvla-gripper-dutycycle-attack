from __future__ import annotations
from collections import deque
import os
import numpy as np
from .types import TriggerContext, TriggerDecision
from .uncertainty import arm_entropy, gripper_entropy, motion_weighted_arm_entropy, motion_weighted_xyz_entropy, prefix_entropy, prefix_top2_margin, xyz_entropy


class BaseTrigger:
    name = "base"
    def reset(self, episode_id: int, max_steps: int) -> None:
        self.episode_id = int(episode_id); self.max_steps = int(max_steps)
    def evaluate(self, context: TriggerContext) -> TriggerDecision:
        raise NotImplementedError


class CleanTrigger(BaseTrigger):
    name = "clean"
    def evaluate(self, context): return TriggerDecision(self.name, False, score=0.0)


class DenseTrigger(BaseTrigger):
    name = "dense"
    def evaluate(self, context): return TriggerDecision(self.name, True, score=1.0)


class RandomBernoulliBudgetedTrigger(BaseTrigger):
    name = "random_bernoulli_budgeted"
    def __init__(self, seed: int = 0): self.seed = int(seed)
    def evaluate(self, context):
        rng = np.random.RandomState(self.seed + int(context.episode_id) * 1000003 + int(context.step_idx))
        active = bool(rng.rand() < float(context.rho))
        return TriggerDecision(self.name, active, score=float(active))


class _GateRandomBudgetedTrigger(RandomBernoulliBudgetedTrigger):
    gate_key = "grasp_gate_active"
    privileged = False
    name = "gate_random_budgeted"
    def __init__(self, seed: int = 0, thresholds: dict | None = None):
        super().__init__(seed)
        self.thresholds = thresholds or {}
    def _metadata_gate_key(self) -> str:
        if self.gate_key == "proxy_grasp_gate_active":
            return str(self.thresholds.get("proxy_gate_metadata_key") or self.gate_key)
        return self.gate_key
    def evaluate(self, context):
        if not bool(context.metadata.get(self._metadata_gate_key(), False)):
            return TriggerDecision(self.name, False, score=0.0, privileged=self.privileged)
        decision = super().evaluate(context)
        decision.trigger_name = self.name
        decision.privileged = self.privileged
        return decision


class PrivGraspWindowRandomBudgetedTrigger(_GateRandomBudgetedTrigger):
    name = "priv_grasp_window_random_budgeted"
    gate_key = "grasp_gate_active"
    privileged = True


class ProxyGraspWindowRandomBudgetedTrigger(_GateRandomBudgetedTrigger):
    name = "proxy_grasp_window_random_budgeted"
    gate_key = "proxy_grasp_gate_active"
    privileged = False


class ProxyLiftCarryEefRiseRandomBudgetedTrigger(_GateRandomBudgetedTrigger):
    name = "proxy_lift_carry_eefrise_random_budgeted"
    gate_key = "proxy_lift_carry_eefrise_gate_active"
    privileged = False


class _GateBurstBudgetedTrigger(BaseTrigger):
    gate_key = "grasp_gate_active"
    privileged = False
    name = "gate_burst_budgeted"

    def reset(self, episode_id: int, max_steps: int) -> None:
        super().reset(episode_id, max_steps)
        self.burst_remaining = 0

    def _burst_steps(self) -> int:
        return max(1, int(os.environ.get(
            "V4_GATE_BURST_STEPS",
            os.environ.get("V4_SPIKE_BURST_STEPS", "10"),
        )))

    def evaluate(self, context):
        gate_active = bool(context.metadata.get(self.gate_key, False))
        if int(getattr(self, "burst_remaining", 0)) > 0:
            self.burst_remaining = int(self.burst_remaining) - 1
            return TriggerDecision(
                self.name,
                True,
                score=1.0,
                privileged=self.privileged,
                threshold_active=gate_active,
                reason=f"gate_burst_remaining={int(self.burst_remaining)};{self.gate_key}={str(gate_active).lower()}",
            )

        active = gate_active
        if active:
            steps = self._burst_steps()
            self.burst_remaining = max(0, int(steps) - 1)
            reason = f"gate_burst_start={int(steps)};{self.gate_key}=true"
        else:
            reason = f"{self.gate_key}=false"
        return TriggerDecision(
            self.name,
            active,
            score=float(active),
            privileged=self.privileged,
            threshold_active=gate_active,
            reason=reason,
        )


class PrivLiftCarryGateBurstBudgetedTrigger(_GateBurstBudgetedTrigger):
    name = "priv_lift_carry_gate_burst_budgeted"
    gate_key = "priv_lift_carry_gate_active"
    privileged = True


class ProxyLiftCarryGateBurstBudgetedTrigger(_GateBurstBudgetedTrigger):
    name = "proxy_lift_carry_gate_burst_budgeted"
    gate_key = "proxy_lift_carry_gate_active"
    privileged = False


class ProxyLiftCarryEefRiseGateBurstBudgetedTrigger(_GateBurstBudgetedTrigger):
    name = "proxy_lift_carry_eefrise_gate_burst_budgeted"
    gate_key = "proxy_lift_carry_eefrise_gate_active"
    privileged = False


class PeriodicBudgetedTrigger(BaseTrigger):
    name = "periodic_budgeted"
    def evaluate(self, context):
        if context.rho <= 0: return TriggerDecision(self.name, False, score=0.0)
        interval = max(1, int(round(1.0 / float(context.rho))))
        active = (int(context.step_idx) % interval) == 0
        return TriggerDecision(self.name, active, score=float(active))


class FixedStepWindowBudgetedTrigger(BaseTrigger):
    name = "fixed_step_window_budgeted"
    privileged = False

    def _start_end(self) -> tuple[int, int]:
        start = int(os.environ.get("V4_FIXED_ATTACK_START", os.environ.get("V4_ABSOLUTE_ATTACK_START", "70")))
        end = int(os.environ.get("V4_FIXED_ATTACK_END", os.environ.get("V4_ABSOLUTE_ATTACK_END", "80")))
        if end < start:
            start, end = end, start
        return start, end

    def evaluate(self, context):
        start, end = self._start_end()
        active = start <= int(context.step_idx) <= end
        return TriggerDecision(
            self.name,
            bool(active),
            score=float(active),
            privileged=self.privileged,
            threshold_active=bool(active),
            reason=f"fixed_step_window={start}-{end};step={int(context.step_idx)}",
        )


class MokaSecondPotRelativeWindowBudgetedTrigger(BaseTrigger):
    name = "moka_second_pot_relative_window_budgeted"
    privileged = False

    def _start_end(self) -> tuple[int, int]:
        start = int(os.environ.get("V4_MOKA_SECOND_WINDOW_START", "0"))
        end = int(os.environ.get("V4_MOKA_SECOND_WINDOW_END", "30"))
        if end < start:
            start, end = end, start
        return start, end

    def evaluate(self, context):
        if context.phase_attack_enabled is False:
            return TriggerDecision(self.name, False, score=0.0, threshold_active=False, reason="phase_attack_disabled")
        stage = str((context.metadata or {}).get("moka_stage_id", ""))
        anchor = (context.metadata or {}).get("moka_stage_anchor_step", None)
        if stage != "second_pot_phase":
            return TriggerDecision(self.name, False, score=0.0, threshold_active=False, reason=f"stage={stage}")
        if anchor is None:
            return TriggerDecision(self.name, False, score=0.0, threshold_active=False, reason="missing_anchor")
        rel = int(context.step_idx) - int(anchor)
        start, end = self._start_end()
        active = bool(start <= rel <= end)
        return TriggerDecision(
            self.name,
            active,
            score=float(active),
            privileged=self.privileged,
            threshold_active=active,
            reason=f"moka_relative_window={start}-{end};relative_step={rel}",
        )


class _ThresholdBase(BaseTrigger):
    score_name = ""; mode = "ge"; name = "threshold"
    def __init__(self, thresholds: dict): self.thresholds = thresholds
    def _threshold(self, task_id, rho):
        key = f"rho_{float(rho):.2f}"
        task = self.thresholds.get("tasks", {}).get(task_id, {})
        if key not in task or self.score_name not in task[key]:
            raise KeyError(f"missing threshold for task={task_id} rho={rho} score={self.score_name}; run calibration first")
        return float(task[key][self.score_name])
    def _score(self, context): raise NotImplementedError
    def evaluate(self, context):
        if context.prefix_logits is None:
            return TriggerDecision(self.name, False, None, False, True, "missing_prefix_logits", threshold_active=False)
        score = self._score(context)
        thr = self._threshold(context.task_id, context.rho)
        active = score >= thr if self.mode == "ge" else score <= thr
        return TriggerDecision(self.name, bool(active), score=score, threshold_active=bool(active))


class _GraspGateThresholdMixin:
    gate_key = "grasp_gate_active"
    privileged = False
    def _metadata_gate_key(self) -> str:
        if self.gate_key == "proxy_grasp_gate_active":
            thresholds = getattr(self, "thresholds", {}) or {}
            return str(thresholds.get("proxy_gate_metadata_key") or self.gate_key)
        return self.gate_key
    def evaluate(self, context):
        if not bool(context.metadata.get(self._metadata_gate_key(), False)):
            return TriggerDecision(self.name, False, score=0.0, privileged=self.privileged, threshold_active=False)
        decision = super().evaluate(context)
        decision.privileged = self.privileged
        return decision


class _CooldownMixin:
    cooldown_steps = 10
    def reset(self, episode_id: int, max_steps: int) -> None:
        super().reset(episode_id, max_steps)
        self.last_active_step = -10**9

    def _cooldown_active(self, step_idx: int) -> bool:
        return int(step_idx) - int(getattr(self, "last_active_step", -10**9)) <= int(self.cooldown_steps)

    def _mark_active(self, step_idx: int) -> None:
        self.last_active_step = int(step_idx)


class EntropyThresholdTrigger(_ThresholdBase):
    name = "entropy_threshold"; score_name = "entropy"; mode = "ge"
    def _score(self, context): return prefix_entropy(context.prefix_logits)


class EntropyThresholdCooldownTrigger(_CooldownMixin, EntropyThresholdTrigger):
    name = "entropy_threshold_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class XyzEntropyThresholdTrigger(_ThresholdBase):
    name = "xyz_entropy_threshold"; score_name = "xyz_entropy"; mode = "ge"
    def _score(self, context): return xyz_entropy(context.prefix_logits)


class GripperEntropyThresholdTrigger(_ThresholdBase):
    name = "gripper_entropy_threshold"; score_name = "gripper_entropy"; mode = "ge"
    def _score(self, context): return gripper_entropy(context.prefix_logits)


class GraspCompositeEntropyThresholdTrigger(_ThresholdBase):
    name = "grasp_composite_entropy_threshold"; score_name = "grasp_composite_entropy"; mode = "ge"
    def _score(self, context):
        x = xyz_entropy(context.prefix_logits)
        g = gripper_entropy(context.prefix_logits)
        stats = {}
        try:
            key = f"rho_{float(context.rho):.2f}"
            stats = self.thresholds.get("tasks", {}).get(context.task_id, {}).get(key, {})
        except Exception:
            stats = {}
        xm = float(stats.get("xyz_entropy_mean", 0.0)); xs = max(float(stats.get("xyz_entropy_std", 1.0)), 1e-6)
        gm = float(stats.get("gripper_entropy_mean", 0.0)); gs = max(float(stats.get("gripper_entropy_std", 1.0)), 1e-6)
        return float(max((x - xm) / xs, (g - gm) / gs))


class XyzEntropyCooldownTrigger(_CooldownMixin, XyzEntropyThresholdTrigger):
    name = "xyz_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class GripperEntropyCooldownTrigger(_CooldownMixin, GripperEntropyThresholdTrigger):
    name = "gripper_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class GraspCompositeEntropyCooldownTrigger(_CooldownMixin, GraspCompositeEntropyThresholdTrigger):
    name = "grasp_composite_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class PrivGraspXyzEntropyCooldownTrigger(_GraspGateThresholdMixin, XyzEntropyCooldownTrigger):
    name = "priv_grasp_xyz_entropy_cooldown"
    gate_key = "grasp_gate_active"
    privileged = True


class PrivGraspGripperEntropyCooldownTrigger(_GraspGateThresholdMixin, GripperEntropyCooldownTrigger):
    name = "priv_grasp_gripper_entropy_cooldown"
    gate_key = "grasp_gate_active"
    privileged = True


class PrivGraspCompositeEntropyCooldownTrigger(_GraspGateThresholdMixin, GraspCompositeEntropyCooldownTrigger):
    name = "priv_grasp_composite_entropy_cooldown"
    gate_key = "grasp_gate_active"
    privileged = True


class ProxyGraspXyzEntropyCooldownTrigger(_GraspGateThresholdMixin, XyzEntropyCooldownTrigger):
    name = "proxy_grasp_xyz_entropy_cooldown"
    gate_key = "proxy_grasp_gate_active"
    privileged = False


class ProxyGraspGripperEntropyCooldownTrigger(_GraspGateThresholdMixin, GripperEntropyCooldownTrigger):
    name = "proxy_grasp_gripper_entropy_cooldown"
    gate_key = "proxy_grasp_gate_active"
    privileged = False


class ProxyGraspCompositeEntropyCooldownTrigger(_GraspGateThresholdMixin, GraspCompositeEntropyCooldownTrigger):
    name = "proxy_grasp_composite_entropy_cooldown"
    gate_key = "proxy_grasp_gate_active"
    privileged = False


class ProxyGraspCompositeEntropySpikeCooldownTrigger(_CooldownMixin, _GraspGateThresholdMixin, GraspCompositeEntropyThresholdTrigger):
    name = "proxy_grasp_composite_entropy_spike_cooldown"
    gate_key = "proxy_grasp_gate_active"
    privileged = False
    cooldown_steps = 10

    def __init__(self, thresholds: dict | None = None, window: int = 10, z_threshold: float = 2.0):
        super().__init__(thresholds or {})
        self.window = int(os.environ.get("V4_SPIKE_WINDOW", window))
        self.z_threshold = float(os.environ.get("V4_SPIKE_Z_THRESHOLD", z_threshold))
        self.cooldown_steps = int(os.environ.get("V4_SPIKE_COOLDOWN_STEPS", self.cooldown_steps))
        self.burst_steps = max(1, int(os.environ.get("V4_SPIKE_BURST_STEPS", "1")))
        self.burst_remaining = 0
        self.history = deque(maxlen=max(self.window, 2))

    def reset(self, episode_id: int, max_steps: int) -> None:
        super().reset(episode_id, max_steps)
        self.history.clear()
        self.burst_remaining = 0

    def evaluate(self, context):
        if context.prefix_logits is None:
            return TriggerDecision(self.name, False, None, False, True, "missing_prefix_logits", threshold_active=False, privileged=self.privileged)
        score = self._score(context)
        hist = list(self.history)
        self.history.append(float(score))
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            self._mark_active(context.step_idx)
            return TriggerDecision(self.name, True, score=score, privileged=self.privileged, threshold_active=True, reason=f"spike_burst_remaining={self.burst_remaining}")
        gate_on = bool(context.metadata.get(self._metadata_gate_key(), False))
        if not gate_on:
            return TriggerDecision(self.name, False, score=score, privileged=self.privileged, threshold_active=False)
        if len(hist) < 3:
            return TriggerDecision(self.name, False, score=score, privileged=self.privileged, threshold_active=False, reason="insufficient_spike_history")
        mu = float(np.mean(hist))
        sigma = max(float(np.std(hist)), 1e-6)
        z = (float(score) - mu) / sigma
        active = bool(z >= self.z_threshold and score > mu)
        if active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, score=score, privileged=self.privileged, threshold_active=active, reason="cooldown_blocked")
        if active:
            self._mark_active(context.step_idx)
            self.burst_remaining = max(0, int(self.burst_steps) - 1)
        reason = f"spike_z={z:.3f}"
        if active and self.burst_remaining > 0:
            reason += f";burst_start={self.burst_steps}"
        return TriggerDecision(self.name, active, score=score, privileged=self.privileged, threshold_active=active, reason=reason)


class ArmEntropyThresholdTrigger(_ThresholdBase):
    name = "arm_entropy_threshold"; score_name = "arm_entropy"; mode = "ge"
    def _score(self, context): return arm_entropy(context.prefix_logits)


class ArmEntropyCooldownTrigger(_CooldownMixin, ArmEntropyThresholdTrigger):
    name = "arm_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class MotionWeightedXyzEntropyThresholdTrigger(_ThresholdBase):
    name = "motion_weighted_xyz_entropy_threshold"; score_name = "motion_weighted_xyz_entropy"; mode = "ge"
    def _score(self, context): return motion_weighted_xyz_entropy(context.prefix_logits, context.clean_action)


class MotionWeightedXyzEntropyCooldownTrigger(_CooldownMixin, MotionWeightedXyzEntropyThresholdTrigger):
    name = "motion_weighted_xyz_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class MotionWeightedArmEntropyThresholdTrigger(_ThresholdBase):
    name = "motion_weighted_arm_entropy_threshold"; score_name = "motion_weighted_arm_entropy"; mode = "ge"
    def _score(self, context): return motion_weighted_arm_entropy(context.prefix_logits, context.clean_action)


class MotionWeightedArmEntropyCooldownTrigger(_CooldownMixin, MotionWeightedArmEntropyThresholdTrigger):
    name = "motion_weighted_arm_entropy_cooldown"
    cooldown_steps = 10
    def evaluate(self, context):
        decision = super().evaluate(context)
        if decision.raw_active and self._cooldown_active(context.step_idx):
            return TriggerDecision(self.name, False, decision.score, decision.signal_available, decision.fallback, "cooldown_blocked", decision.oracle, decision.privileged, threshold_active=decision.threshold_active)
        if decision.raw_active:
            self._mark_active(context.step_idx)
        decision.trigger_name = self.name
        return decision


class MarginThresholdTrigger(_ThresholdBase):
    name = "margin_threshold"; score_name = "margin"; mode = "le"
    def _score(self, context): return prefix_top2_margin(context.prefix_logits)


class OracleOfflineEntropyTopKTrigger(BaseTrigger):
    name = "oracle_offline_entropy_topk"
    def __init__(self, active_steps: set[int] | None = None): self.active_steps = active_steps or set()
    def evaluate(self, context): return TriggerDecision(self.name, int(context.step_idx) in self.active_steps, oracle=True)


class OracleOfflineMarginTopKTrigger(OracleOfflineEntropyTopKTrigger):
    name = "oracle_offline_margin_topk"


class TCNDetectorBudgetedTrigger(BaseTrigger):
    name = "tcn_proprio_budgeted"
    HISTORY_LEN = 16
    HIDDEN_DIM = 64
    TCN_LAYERS = 3
    ALLOWED = [
        "gripper_command", "gripper_qpos", "gripper_width",
        "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
    ]
    N_PROPRIO = len(ALLOWED)

    def __init__(self, model_path=None, hazard_threshold=0.1, trigger_duration=5):
        self.model_path = model_path or "/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt"
        self.hazard_th = float(hazard_threshold)
        self.trig_dur = int(trigger_duration)
        self._history = []
        self._hazard_buf = []
        self._model = None  # lazy load

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
        class _TCN(nn.Module):
            def __init__(self, in_dim, h_dim, n_ph=8, n_l=3, do=0.1):
                super().__init__()
                self.proj = nn.Linear(in_dim, h_dim)
                self.convs = nn.ModuleList([nn.Conv1d(h_dim, h_dim, 3, padding=2**(i+1), dilation=2**i) for i in range(n_l)])
                self.drop = nn.Dropout(do)
                self.ph = nn.Linear(h_dim, n_ph); self.hz = nn.Linear(h_dim, 1); self.rl = nn.Linear(h_dim, 1)
            def forward(self, x):
                x = self.proj(x); x = x.transpose(1, 2)
                for c in self.convs:
                    r = x; x = F.relu(c(x)); x = x[:, :, -r.shape[2]:] + r; x = self.drop(x)
                x = x[:, :, -1]; return self.ph(x), self.hz(x).squeeze(-1), self.rl(x).squeeze(-1)
        self._model = _TCN(self.N_PROPRIO, self.HIDDEN_DIM, 8, self.TCN_LAYERS)
        self._model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self._model.eval()
        self._torch = torch; self._np = np

    def reset(self, episode_id, max_steps):
        super().reset(episode_id, max_steps)
        self._history = []; self._hazard_buf = []

    def evaluate(self, context):
        self._ensure_model()
        torch = self._torch; np = self._np

        # Extract features from context
        meta = context.metadata or {}
        action = context.clean_action
        ax, ay, az, ag = (0.0, 0.0, 0.0, 0.0)
        if action is not None:
            if hasattr(action, 'tolist'):
                a = action.tolist()
            else:
                a = list(action)
            if len(a) >= 4:
                ax, ay, az, ag = float(a[0]), float(a[1]), float(a[2]), float(a[3])

        feats = np.array([
            float(meta.get('gripper_command', 0)), float(meta.get('gripper_qpos', 0)),
            float(meta.get('gripper_width', 0)),
            float(meta.get('eef_x', 0)), float(meta.get('eef_y', 0)), float(meta.get('eef_z', 0)),
            float(meta.get('eef_vx', 0)), float(meta.get('eef_vy', 0)), float(meta.get('eef_vz', 0)),
            ax, ay, az, ag,
        ], dtype=np.float32)

        self._history.append(feats)
        if len(self._history) > self.HISTORY_LEN:
            self._history = self._history[-self.HISTORY_LEN:]

        hist = np.array(self._history, dtype=np.float32)
        if hist.shape[0] < self.HISTORY_LEN:
            pad = np.zeros((self.HISTORY_LEN - hist.shape[0], self.N_PROPRIO), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)

        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, hl, _ = self._model(x)
        hs = float(torch.sigmoid(hl).item())

        self._hazard_buf.append(hs)
        if len(self._hazard_buf) > self.trig_dur:
            self._hazard_buf = self._hazard_buf[-self.trig_dur:]

        trigger_now = False
        if len(self._hazard_buf) >= self.trig_dur and all(h > self.hazard_th for h in self._hazard_buf):
            trigger_now = True

        return TriggerDecision(self.name, trigger_now, score=hs, reason=f"hazard={hs:.3f}")


def make_trigger(name: str, seed: int = 0, thresholds: dict | None = None):
    if name in ("tcn_proprio_budgeted", "tcn_proprio"): return TCNDetectorBudgetedTrigger(thresholds.get("model_path") if thresholds else None, float(thresholds.get("hazard_th", 0.1)) if thresholds else 0.1, int(thresholds.get("dur", 5)) if thresholds else 5)
    if name in ("clean", "none"): return CleanTrigger()
    if name == "dense": return DenseTrigger()
    if name in ("random_bernoulli_budgeted", "random"): return RandomBernoulliBudgetedTrigger(seed)
    if name == "priv_grasp_window_random_budgeted": return PrivGraspWindowRandomBudgetedTrigger(seed)
    if name == "proxy_grasp_window_random_budgeted": return ProxyGraspWindowRandomBudgetedTrigger(seed, thresholds or {})
    if name == "proxy_lift_carry_eefrise_random_budgeted": return ProxyLiftCarryEefRiseRandomBudgetedTrigger(seed, thresholds or {})
    if name == "priv_lift_carry_gate_burst_budgeted": return PrivLiftCarryGateBurstBudgetedTrigger()
    if name == "proxy_lift_carry_gate_burst_budgeted": return ProxyLiftCarryGateBurstBudgetedTrigger()
    if name == "proxy_lift_carry_eefrise_gate_burst_budgeted": return ProxyLiftCarryEefRiseGateBurstBudgetedTrigger()
    if name in ("fixed_step_window_budgeted", "constant_delta_window_budgeted", "absolute_step_window_budgeted"): return FixedStepWindowBudgetedTrigger()
    if name == "moka_second_pot_relative_window_budgeted": return MokaSecondPotRelativeWindowBudgetedTrigger()
    if name in ("periodic_budgeted", "periodic_sparse"): return PeriodicBudgetedTrigger()
    if name == "entropy_threshold": return EntropyThresholdTrigger(thresholds or {})
    if name == "entropy_threshold_cooldown": return EntropyThresholdCooldownTrigger(thresholds or {})
    if name == "xyz_entropy_threshold": return XyzEntropyThresholdTrigger(thresholds or {})
    if name == "xyz_entropy_cooldown": return XyzEntropyCooldownTrigger(thresholds or {})
    if name == "gripper_entropy_threshold": return GripperEntropyThresholdTrigger(thresholds or {})
    if name == "gripper_entropy_cooldown": return GripperEntropyCooldownTrigger(thresholds or {})
    if name == "grasp_composite_entropy_threshold": return GraspCompositeEntropyThresholdTrigger(thresholds or {})
    if name == "grasp_composite_entropy_cooldown": return GraspCompositeEntropyCooldownTrigger(thresholds or {})
    if name == "priv_grasp_xyz_entropy_cooldown": return PrivGraspXyzEntropyCooldownTrigger(thresholds or {})
    if name == "priv_grasp_gripper_entropy_cooldown": return PrivGraspGripperEntropyCooldownTrigger(thresholds or {})
    if name == "priv_grasp_composite_entropy_cooldown": return PrivGraspCompositeEntropyCooldownTrigger(thresholds or {})
    if name == "proxy_grasp_xyz_entropy_cooldown": return ProxyGraspXyzEntropyCooldownTrigger(thresholds or {})
    if name == "proxy_grasp_gripper_entropy_cooldown": return ProxyGraspGripperEntropyCooldownTrigger(thresholds or {})
    if name == "proxy_grasp_composite_entropy_cooldown": return ProxyGraspCompositeEntropyCooldownTrigger(thresholds or {})
    if name == "proxy_grasp_composite_entropy_spike_cooldown": return ProxyGraspCompositeEntropySpikeCooldownTrigger(thresholds or {})
    if name == "arm_entropy_threshold": return ArmEntropyThresholdTrigger(thresholds or {})
    if name == "arm_entropy_cooldown": return ArmEntropyCooldownTrigger(thresholds or {})
    if name == "motion_weighted_xyz_entropy_threshold": return MotionWeightedXyzEntropyThresholdTrigger(thresholds or {})
    if name == "motion_weighted_xyz_entropy_cooldown": return MotionWeightedXyzEntropyCooldownTrigger(thresholds or {})
    if name == "motion_weighted_arm_entropy_threshold": return MotionWeightedArmEntropyThresholdTrigger(thresholds or {})
    if name == "motion_weighted_arm_entropy_cooldown": return MotionWeightedArmEntropyCooldownTrigger(thresholds or {})
    if name == "margin_threshold": return MarginThresholdTrigger(thresholds or {})
    if name == "oracle_offline_entropy_topk": return OracleOfflineEntropyTopKTrigger()
    if name == "oracle_offline_margin_topk": return OracleOfflineMarginTopKTrigger()
    raise ValueError(f"unknown V4 trigger: {name}")
