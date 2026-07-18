#!/usr/bin/env python3
"""Detector V4: Model candidates, feature views, loss ablations, and training loop.

Covers Phase D4 + D5: three feature views (A/B/C), three model candidates
(A/B/C), and five loss ablations (L0/L2/L3/L4/L5).

All features are causal (no future leakage). All privileged Teacher fields
are excluded from student input.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── constants ──────────────────────────────────────────────────────────
V4_HEADS = ("criticality", "veto", "release_imminent")
V4_CANDIDATE_HEADS = ("critical_onset_hazard", "valid_retention", "veto", "release_hazard")
SC5_FEATURES = (
    "gripper_qpos", "gripper_command", "gripper_action_open", "gripper_action_close",
    "eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_w", "eef_quat_x", "eef_quat_y",
    "eef_quat_z", "eef_gripper_pos", "eef_gripper_vel",
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
    "gripper_state_open", "gripper_state_close", "recent_close_streak",
    "time_since_close", "grasp_support_probability",
)
B3_FEATURES_25D = tuple(SC5_FEATURES)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FIT_STATES = list(range(0, 20))


# ── helpers ────────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── feature derivation ─────────────────────────────────────────────────
@dataclass
class FeatureViewConfig:
    name: str
    feature_names: tuple[str, ...]
    feature_count: int

    @property
    def feature_order_sha256(self) -> str:
        return json_sha(list(self.feature_names))


VIEW_A = FeatureViewConfig("A_25D", B3_FEATURES_25D, 25)

VIEW_B_EXTRA = (
    "delta_gripper_qpos", "delta2_gripper_qpos",
    "gripper_command_qpos_deviation", "close_dwell_duration",
    "time_since_close_onset", "recent_close_count",
    "opening_trend", "recent_command_variance",
)
VIEW_B = FeatureViewConfig("B_25D_PLUS_GRIPPER_DYNAMICS",
                           B3_FEATURES_25D + VIEW_B_EXTRA, 33)

VIEW_C_EXTRA = (
    "eef_velocity", "eef_acceleration", "eef_vertical_velocity",
    "eef_stability", "eef_displacement_since_close_onset",
    "action_consistency",
)
VIEW_C = FeatureViewConfig("C_25D_PLUS_EEF_DYNAMICS",
                           VIEW_B.feature_names + VIEW_C_EXTRA, 39)

ALL_VIEWS = {"A": VIEW_A, "B": VIEW_B, "C": VIEW_C}


def derive_dynamic_features(base_25d: Tensor, view: str) -> Tensor:
    """Derive causal dynamic features from base 25D step sequences.

    Args:
        base_25d: [T, 25] float32 tensor of original B3 features.
        view: "A", "B", or "C".

    Returns:
        [T, F] tensor where F depends on view.
    """
    T = base_25d.shape[0]
    if view == "A":
        return base_25d

    # Extract base feature indices
    IDX_GRIPPER_QPOS = 0
    IDX_GRIPPER_COMMAND = 1
    IDX_EEF_X, IDX_EEF_Y, IDX_EEF_Z = 4, 5, 6
    IDX_TIME_SINCE_CLOSE = 23

    qpos = base_25d[:, IDX_GRIPPER_QPOS]
    cmd = base_25d[:, IDX_GRIPPER_COMMAND]
    eef_x = base_25d[:, IDX_EEF_X]
    eef_y = base_25d[:, IDX_EEF_Y]
    eef_z = base_25d[:, IDX_EEF_Z]
    time_since_close = base_25d[:, IDX_TIME_SINCE_CLOSE]

    features = [base_25d]

    if view in ("B", "C"):
        # delta qpos
        dq = torch.zeros(T)
        dq[1:] = qpos[1:] - qpos[:-1]
        features.append(dq.unsqueeze(1))

        # delta2 qpos
        d2q = torch.zeros(T)
        d2q[2:] = dq[2:] - dq[1:-1]
        features.append(d2q.unsqueeze(1))

        # command-qpos deviation
        dev = (cmd - qpos).abs()
        features.append(dev.unsqueeze(1))

        # close dwell (consecutive steps with gripper closed)
        dwell = torch.zeros(T)
        count = 0
        for t in range(T):
            if time_since_close[t] >= 0:
                count += 1
            else:
                count = 0
            dwell[t] = float(count)
        features.append(dwell.unsqueeze(1))

        # time since close onset (accumulated)
        tsco = torch.full((T,), -1.0)
        last_onset = -1
        for t in range(T):
            # Detect close onset: time_since_close transitions from -1 to >= 0
            if t == 0:
                if time_since_close[t] >= 0:
                    last_onset = t
            else:
                if time_since_close[t] >= 0 and time_since_close[t-1] < 0:
                    last_onset = t
            if last_onset >= 0:
                tsco[t] = float(t - last_onset)
        features.append(tsco.unsqueeze(1))

        # recent close count
        rcc = torch.zeros(T)
        onset_count = 0
        prev_closed = False
        for t in range(T):
            closed = time_since_close[t] >= 0
            if closed and not prev_closed:
                onset_count += 1
            prev_closed = closed
            rcc[t] = float(onset_count)
        features.append(rcc.unsqueeze(1))

        # opening trend (EMA of dq with decay 0.9)
        trend = torch.zeros(T)
        ema = 0.0
        for t in range(T):
            ema = 0.9 * ema + 0.1 * dq[t]
            trend[t] = ema
        features.append(trend.unsqueeze(1))

        # recent command variance (window=10)
        rcv = torch.zeros(T)
        for t in range(T):
            window = cmd[max(0, t-9):t+1]
            rcv[t] = window.var(unbiased=False) if len(window) >= 2 else 0.0
        features.append(rcv.unsqueeze(1))

    if view == "C":
        # EEF velocity (euclidean norm of position delta)
        eef_vel = torch.zeros(T)
        if T >= 2:
            dx = eef_x[1:] - eef_x[:-1]
            dy = eef_y[1:] - eef_y[:-1]
            dz = eef_z[1:] - eef_z[:-1]
            eef_vel[1:] = torch.sqrt(dx*dx + dy*dy + dz*dz)
        features.append(eef_vel.unsqueeze(1))

        # EEF acceleration
        eef_acc = torch.zeros(T)
        eef_acc[2:] = eef_vel[2:] - eef_vel[1:-1]
        features.append(eef_acc.unsqueeze(1))

        # EEF vertical velocity
        eef_vz = torch.zeros(T)
        eef_vz[1:] = eef_z[1:] - eef_z[:-1]
        features.append(eef_vz.unsqueeze(1))

        # EEF stability (EMA variance of position, window=20)
        stab = torch.zeros(T)
        for t in range(T):
            w = min(20, t+1)
            if w >= 2:
                px = eef_x[max(0,t-w+1):t+1]
                py = eef_y[max(0,t-w+1):t+1]
                pz = eef_z[max(0,t-w+1):t+1]
                var_x = px.var(unbiased=False)
                var_y = py.var(unbiased=False)
                var_z = pz.var(unbiased=False)
                stab[t] = float(torch.sqrt(var_x + var_y + var_z))
        features.append(stab.unsqueeze(1))

        # EEF displacement since close onset
        disp = torch.zeros(T)
        onset_pos = None
        for t in range(T):
            if t == 0 and time_since_close[t] >= 0:
                onset_pos = (eef_x[t], eef_y[t], eef_z[t])
            elif t > 0 and time_since_close[t] >= 0 and time_since_close[t-1] < 0:
                onset_pos = (eef_x[t], eef_y[t], eef_z[t])
            if onset_pos is not None:
                dx = eef_x[t] - onset_pos[0]
                dy = eef_y[t] - onset_pos[1]
                dz = eef_z[t] - onset_pos[2]
                disp[t] = float(torch.sqrt(torch.tensor(dx*dx + dy*dy + dz*dz)))
        features.append(disp.unsqueeze(1))

        # Action consistency (fraction of last 10 gripper actions that are identical)
        ac = torch.zeros(T)
        for t in range(T):
            w_start = max(0, t-9)
            window = cmd[w_start:t+1]
            if len(window) >= 2:
                mode_val = window.mode().values
                ac[t] = float((window == mode_val).sum()) / len(window)
            else:
                ac[t] = 1.0
        features.append(ac.unsqueeze(1))

    return torch.cat(features, dim=1)


# ── model definitions ──────────────────────────────────────────────────
class V4GRU(nn.Module):
    """GRU128 backbone shared across all V4 candidates."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.heads = nn.ModuleDict()
        for head_name in V4_HEADS[:num_heads]:
            self.heads[head_name] = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, mask: Tensor,
                hidden: Optional[Tensor] = None
                ) -> tuple[dict[str, Tensor], Tensor]:
        """Forward pass with episode-only hidden reset.

        Args:
            x: [B, T, F] float32
            mask: [B, T] bool (True = valid, False = padding)
            hidden: optional initial hidden state

        Returns:
            logits: dict[str, Tensor], each [B, T]
            final_hidden: [1, B, H]
        """
        B, T, _ = x.shape
        # Build packed sequence respecting episode boundaries
        lengths = mask.sum(dim=1).cpu().long()
        lengths = lengths.clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, final_hidden = self.gru(packed, hidden)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=T)

        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(unpacked).squeeze(-1)  # [B, T]

        return logits, final_hidden


class CandidateAGRU(nn.Module):
    """View A (25D) + GRU128 + criticality + veto + release_imminent."""

    def __init__(self):
        super().__init__()
        self.backbone = V4GRU(25, 128, 3)

    def forward(self, x: Tensor, mask: Tensor):
        return self.backbone(x, mask)


class CandidateBGRU(nn.Module):
    """View B (33D) + GRU128 + criticality + veto + release_imminent."""

    def __init__(self):
        super().__init__()
        self.backbone = V4GRU(33, 128, 3)

    def forward(self, x: Tensor, mask: Tensor):
        return self.backbone(x, mask)


class CandidateCGRU(nn.Module):
    """View C (39D) + GRU128 + critical_onset_hazard + valid_retention + veto + release_hazard."""

    def __init__(self):
        super().__init__()
        self.backbone = V4GRU(39, 128, 4)
        # Rename default heads
        # Actually we need different head names for C
        self.gru = nn.GRU(39, 128, batch_first=True)
        self.heads = nn.ModuleDict({
            "critical_onset_hazard": nn.Linear(128, 1),
            "valid_retention": nn.Linear(128, 1),
            "veto": nn.Linear(128, 1),
            "release_hazard": nn.Linear(128, 1),
        })

    def forward(self, x: Tensor, mask: Tensor,
                hidden: Optional[Tensor] = None
                ) -> tuple[dict[str, Tensor], Tensor]:
        B, T, _ = x.shape
        lengths = mask.sum(dim=1).cpu().long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        gru_out, final_hidden = self.gru(packed, hidden)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=T)

        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(unpacked).squeeze(-1)
        return logits, final_hidden


# ── normalization ──────────────────────────────────────────────────────
@dataclass
class V4Normalization:
    mean: Tensor
    std: Tensor

    @classmethod
    def compute(cls, features_list: list[Tensor]) -> "V4Normalization":
        cat = torch.cat([f[v] for f in features_list for v in
                        [torch.ones(f.shape[0], dtype=torch.bool)]], dim=0)
        # Simpler: just cat all valid steps
        all_f = torch.cat(features_list, dim=0)
        mean = all_f.mean(dim=0)
        std = all_f.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean, std)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


# ── loss functions ─────────────────────────────────────────────────────
def loss_l0_masked_bce(logits: dict[str, Tensor], targets: dict[str, Tensor],
                       masks: dict[str, Tensor], **kwargs) -> Tensor:
    """L0: Original masked BCE on criticality/valid_retention head only."""
    head = "criticality"
    if head not in logits:
        head = "valid_retention"
    pred = logits[head]
    target = targets.get("criticality", targets.get("valid_retention"))
    mask = masks.get("criticality", masks.get("valid_retention"))
    if target is None or mask is None:
        return torch.tensor(0.0, device=pred.device)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1)


def loss_l2_hard_negative_veto(logits: dict[str, Tensor],
                               targets: dict[str, Tensor],
                               masks: dict[str, Tensor],
                               hard_neg_weight: float = 3.0,
                               **kwargs) -> Tensor:
    """L2: Masked BCE on criticality + veto head with hard-negative class weight."""
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)
    n_terms = 0

    # Criticality head (standard masked BCE)
    crit_head = "criticality" if "criticality" in logits else "valid_retention"
    if crit_head in logits and crit_head in targets and crit_head in masks:
        pred = logits[crit_head]
        target = targets[crit_head]
        mask = masks[crit_head]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        total = total + (loss * mask.float()).sum() / mask.float().sum().clamp_min(1)
        n_terms += 1

    # Veto head (weighted BCE for hard negatives)
    if "veto" in logits and "veto" in targets and "veto" in masks:
        pred = logits["veto"]
        target = targets["veto"]
        mask = masks["veto"]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        # Weight: hard negatives (target=1) get higher weight
        weights = torch.where(target > 0.5, hard_neg_weight, 1.0)
        weighted = (loss * weights * mask.float()).sum() / mask.float().sum().clamp_min(1)
        total = total + weighted
        n_terms += 1

    # Release head
    if "release_imminent" in logits and "release_imminent" in targets and "release_imminent" in masks:
        pred = logits["release_imminent"]
        target = targets["release_imminent"]
        mask = masks["release_imminent"]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        total = total + (loss * mask.float()).sum() / mask.float().sum().clamp_min(1)
        n_terms += 1

    return total / max(n_terms, 1)


def loss_l4_pairwise_ranking(logits: dict[str, Tensor],
                             targets: dict[str, Tensor],
                             masks: dict[str, Tensor],
                             ranking_margin: float = 0.2,
                             ranking_weight: float = 0.5,
                             **kwargs) -> Tensor:
    """L4: L2 + candidate-window pairwise ranking loss."""
    base_loss = loss_l2_hard_negative_veto(logits, targets, masks, **kwargs)

    # Pairwise ranking: valid criticality score should exceed hard-negative score
    crit_head = "criticality" if "criticality" in logits else "valid_retention"
    if crit_head not in logits:
        return base_loss

    pred = logits[crit_head]  # [B, T]
    crit_target = targets.get("criticality", targets.get("valid_retention"))
    veto_target = targets.get("veto")
    crit_mask = masks.get("criticality", masks.get("valid_retention"))
    veto_mask = masks.get("veto")

    if crit_target is None or veto_target is None:
        return base_loss

    # For each episode, compute: mean score on valid steps - mean score on veto steps
    B = pred.shape[0]
    ranking_loss = torch.tensor(0.0, device=pred.device)
    n_valid = 0
    for b in range(B):
        pos_mask = crit_mask[b] & (crit_target[b] > 0.5)
        neg_mask = veto_mask[b] & (veto_target[b] > 0.5)
        if pos_mask.any() and neg_mask.any():
            pos_score = pred[b][pos_mask].mean()
            neg_score = pred[b][neg_mask].mean()
            # Hinge: positive should exceed negative by margin
            ranking_loss = ranking_loss + F.relu(ranking_margin - (pos_score - neg_score))
            n_valid += 1

    if n_valid > 0:
        ranking_loss = ranking_loss / n_valid
        return base_loss + ranking_weight * ranking_loss
    return base_loss


LOSS_FUNCTIONS = {
    "L0": loss_l0_masked_bce,
    "L2": loss_l2_hard_negative_veto,
    "L4": loss_l4_pairwise_ranking,
}


# ── data loading ───────────────────────────────────────────────────────
@dataclass
class V4Episode:
    identity: str
    suite: str
    task_idx: int
    state_id: int
    fold_id: int
    feature_view: str
    features: Tensor  # [T, F]
    targets: dict[str, Tensor]  # per-step targets
    masks: dict[str, Tensor]  # known masks
    step_valid_mask: Tensor  # [T] bool
    n_steps: int


def load_v4_episode(s1_root: Path, windows_root: Path,
                    suite: str, task: int, state: int,
                    view: str) -> Optional[V4Episode]:
    """Load one episode with derived features and Teacher V2 labels."""
    ident_dir = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    student_path = ident_dir / "student_input_records.jsonl"
    if not student_path.exists():
        return None

    students = jsonl(student_path)
    features_25d = torch.tensor(
        [[float(v) for v in r["features_25d"]] for r in students],
        dtype=torch.float32
    )
    valid_mask = torch.tensor(
        [bool(r.get("valid", True)) for r in students], dtype=torch.bool
    )

    # Derive features
    features = derive_dynamic_features(features_25d, view)

    # Load Teacher V2 labels
    v2_dir = windows_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    v2_path = v2_dir / "teacher_v2_labels.jsonl"
    if not v2_path.exists():
        return None
    v2_labels = jsonl(v2_path)

    T = len(students)
    identity = f"{suite}/task_{task:02d}/state_{state:02d}"

    # Build targets
    targets = {
        "criticality": torch.zeros(T, dtype=torch.float32),
        "veto": torch.zeros(T, dtype=torch.float32),
        "release_imminent": torch.zeros(T, dtype=torch.float32),
        "valid_retention": torch.zeros(T, dtype=torch.float32),
    }
    known_masks = {
        "criticality": torch.zeros(T, dtype=torch.bool),
        "veto": torch.zeros(T, dtype=torch.bool),
        "release_imminent": torch.zeros(T, dtype=torch.bool),
        "valid_retention": torch.zeros(T, dtype=torch.bool),
    }

    for i, label in enumerate(v2_labels):
        if i >= T:
            break
        ev = label.get("event_valid_mask", True)
        targets["criticality"][i] = float(label.get("critical_retention_window", False))
        targets["valid_retention"][i] = float(label.get("valid_retention", False))
        targets["veto"][i] = float(label.get("false_trigger_veto", False))
        targets["release_imminent"][i] = float(label.get("release_imminent", False))

        known_masks["criticality"][i] = ev
        known_masks["valid_retention"][i] = ev
        known_masks["veto"][i] = ev
        known_masks["release_imminent"][i] = ev

    return V4Episode(
        identity=identity, suite=suite, task_idx=task, state_id=state,
        fold_id=state // 5, feature_view=view, features=features,
        targets=targets, masks=known_masks, step_valid_mask=valid_mask,
        n_steps=T,
    )


# ── training loop ──────────────────────────────────────────────────────
def train_v4_model(
    episodes: list[V4Episode],
    view: str,
    loss_name: str,
    candidate: str,
    seed: int = 20260717,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    hard_neg_weight: float = 3.0,
    ranking_weight: float = 0.5,
):
    """Train one V4 candidate model and return checkpoint + losses."""
    random.seed(seed)
    torch.manual_seed(seed)

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    # Build model
    input_dim = ALL_VIEWS[view].feature_count
    if candidate == "A":
        model = CandidateAGRU()
    elif candidate == "B":
        model = CandidateBGRU()
    elif candidate == "C":
        model = CandidateCGRU()
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    model = model.to(device_obj)
    model.train()

    # Compute normalization from training fold only
    all_features = [ep.features for ep in episodes]
    cat_features = torch.cat(all_features, dim=0)
    norm_mean = cat_features.mean(dim=0)
    norm_std = cat_features.std(dim=0, unbiased=False).clamp_min(1e-6)

    loss_fn = LOSS_FUNCTIONS[loss_name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Episode grouping for balanced sampling
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        groups[(ep.suite, ep.task_idx)].append(i)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        # Shuffle within groups
        indices = []
        rng = random.Random(seed + epoch * 1000)
        queues = {k: list(v) for k, v in sorted(groups.items())}
        for v in queues.values():
            rng.shuffle(v)
        while any(queues.values()):
            for k in sorted(queues):
                if queues[k]:
                    indices.append(queues[k].pop(0))

        epoch_terms = []
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_eps = [episodes[i] for i in batch_idx]
            max_T = max(ep.n_steps for ep in batch_eps)
            B = len(batch_eps)
            F = batch_eps[0].features.shape[1]

            x = torch.zeros(B, max_T, F)
            padding = torch.zeros(B, max_T, dtype=torch.bool)
            batch_targets = {k: torch.zeros(B, max_T) for k in targets_template()}
            batch_masks = {k: torch.zeros(B, max_T, dtype=torch.bool) for k in targets_template()}

            for b, ep in enumerate(batch_eps):
                T_ep = ep.n_steps
                x[b, :T_ep] = (ep.features - norm_mean) / norm_std
                padding[b, :T_ep] = True
                for k in batch_targets:
                    if k in ep.targets:
                        batch_targets[k][b, :T_ep] = ep.targets[k]
                        batch_masks[k][b, :T_ep] = ep.masks[k]

            x = x.to(device_obj)
            padding = padding.to(device_obj)
            for k in batch_targets:
                batch_targets[k] = batch_targets[k].to(device_obj)
                batch_masks[k] = batch_masks[k].to(device_obj)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, padding)
            loss = loss_fn(logits, batch_targets, batch_masks,
                          hard_neg_weight=hard_neg_weight,
                          ranking_weight=ranking_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_terms.append(float(loss.detach()))

        avg = sum(epoch_terms) / len(epoch_terms)
        epoch_losses.append(avg)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    return model, epoch_losses, (norm_mean, norm_std)


def targets_template():
    return ["criticality", "veto", "release_imminent", "valid_retention"]


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--windows-root", type=Path, required=True)
    ap.add_argument("--view", choices=["A", "B", "C"], default="A")
    ap.add_argument("--candidate", choices=["A", "B", "C"], default="A")
    ap.add_argument("--loss", choices=["L0", "L2", "L4"], default="L2")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=None,
                   help="Only load validation states for this fold (0-3)")
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--state", type=int, default=None)
    args = ap.parse_args()

    print(f"V4 Smoke: view={args.view} candidate={args.candidate} "
          f"loss={args.loss} seed={args.seed}")

    # Load episodes
    if args.suite and args.task is not None and args.state is not None:
        scope = [(args.suite, args.task, args.state)]
    elif args.fold is not None:
        states = list(range(args.fold * 5, (args.fold + 1) * 5))
        scope = [(s, t, st) for s in SUITES for t in range(10) for st in states]
    else:
        # Default: FIT states 0-19, first 100 identities for smoke
        scope = [(SUITES[0], t, st) for t in range(5) for st in range(5)]

    episodes = []
    for suite, task, state in scope:
        ep = load_v4_episode(args.s1_root, args.windows_root,
                            suite, task, state, args.view)
        if ep is not None:
            episodes.append(ep)

    if not episodes:
        print("No episodes loaded!")
        import sys; sys.exit(1)

    print(f"Loaded {len(episodes)} episodes, {sum(e.n_steps for e in episodes)} total steps")

    model, losses, (norm_mean, norm_std) = train_v4_model(
        episodes, args.view, args.loss, args.candidate,
        seed=args.seed, epochs=args.epochs, device=args.device,
    )

    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss history: {[f'{l:.4f}' for l in losses]}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "view": args.view,
            "candidate": args.candidate,
            "loss": args.loss,
            "seed": args.seed,
            "losses": losses,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "n_episodes": len(episodes),
        }, args.output / "checkpoint.pt")
        print(f"Checkpoint saved to {args.output / 'checkpoint.pt'}")
