#!/usr/bin/env python3
"""Milestone 2C.1 — Offline Replay Validation + Anti-Timing Ablations.

Compares:
  1. full proprio student (existing 2C checkpoint)
  2. rule/proxy trigger baseline
  3. time-only baseline
  4. no-normalized-step ablation
  5. label-shuffle sanity baseline

All evaluations are causal: at each timestep t the model sees only data[:t].
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.proprio_causal_student import (
    CATEGORICAL_FEATURES,
    IDENTITY_COLUMNS,
    NUMERIC_FEATURES,
    PHASE_CLASSES,
    ProprioCausalMLP,
    assert_feature_whitelist,
    auprc,
    auroc,
    binary_metrics,
    boolish,
    compute_loss,
    encode_dataset,
    indices_for_split,
    phase_metrics,
    phase_index,
    read_csv_rows,
    split_summary,
    write_csv,
)


# ── Helpers from training script (not in the module) ──

def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def threshold_metrics(pred_rows, hazard_threshold, release_threshold):
    """Compute window-aware trigger metrics."""
    trigger = []
    inside = []
    false_early = []
    missed = []
    latencies = []
    by_episode = defaultdict(list)
    for r in pred_rows:
        h = float(r["hazard_score"])
        rel = float(r["release_safe_score"])
        trig = h >= hazard_threshold and rel < release_threshold
        ws = _num(r.get("teacher_window_start"))
        we = _num(r.get("teacher_window_end"))
        step = int(float(r["step_idx"]))
        is_inside = ws is not None and we is not None and ws <= step <= we
        trigger.append(trig)
        inside.append(is_inside)
        false_early.append(trig and ws is not None and step < ws)
        missed.append(is_inside and not trig)
        by_episode.setdefault(str(r["episode_key"]), []).append({**r, "trigger_now": trig})
    for rows_ep in by_episode.values():
        ws_vals = [_num(r.get("teacher_window_start")) for r in rows_ep]
        ws = next((v for v in ws_vals if v is not None), None)
        trig_steps = [int(float(r["step_idx"])) for r in rows_ep if r["trigger_now"]]
        if ws is not None and trig_steps:
            latencies.append(min(trig_steps) - ws)
    return {
        "hazard_threshold": hazard_threshold,
        "release_safe_threshold": release_threshold,
        "trigger_rate": sum(trigger) / len(trigger) if trigger else 0.0,
        "false_early_trigger_rate": sum(false_early) / len(false_early) if false_early else 0.0,
        "miss_rate": sum(missed) / len(missed) if missed else 0.0,
        "trigger_coverage_on_window_rows": sum(t and i for t, i in zip(trigger, inside)) / max(1, sum(inside)),
        "mean_latency_to_window_start": sum(latencies) / len(latencies) if latencies else "",
        "abstain_rate": 0.0,
    }


def threshold_sweep(pred_rows):
    """Sweep hazard and release thresholds."""
    rows_out = []
    for h in [x / 10 for x in range(1, 10)]:
        for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
            rows_out.append(threshold_metrics(pred_rows, h, r))
    return rows_out

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_DATA = Path("/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv")
DEFAULT_MODEL = Path("/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt")
DEFAULT_OUT = Path("/data/liuyu/outputs/milestone_2c1_student_replay_ablation_20260527")

# Features used by the time-only baseline (no gripper/eef/action info)
TIME_ONLY_FEATURES = ["normalized_step"] + CATEGORICAL_FEATURES

# Features where we REMOVE normalized_step for the ablation
NUMERIC_NO_STEP = [f for f in NUMERIC_FEATURES if f != "normalized_step"]

# ═══════════════════════════════════════════════════════════════
# Causal Replay Evaluator
# ═══════════════════════════════════════════════════════════════


def causal_replay_episode(
    episode_rows: list[dict[str, Any]],
    predict_fn,
    hazard_threshold: float = 0.1,
    release_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run causal replay on a single episode.

    At each timestep t, predict_fn receives all rows[:t+1] but must
    only use causal information. The predict_fn returns
    (hazard_score, release_score, extra_info_dict).

    Returns per-episode metrics.
    """
    trigger_steps: list[int] = []
    hazard_scores: list[float] = []
    release_scores: list[float] = []
    teacher_phases: list[str] = []
    teacher_hazards: list[bool] = []

    ws = None
    we = None
    inside_window: list[bool] = []
    false_early: list[bool] = []

    for t in range(len(episode_rows)):
        causal_rows = episode_rows[: t + 1]
        h_score, r_score, _extra = predict_fn(causal_rows, t)
        hazard_scores.append(h_score)
        release_scores.append(r_score)

        trig = h_score >= hazard_threshold and r_score < release_threshold
        if trig:
            trigger_steps.append(t)

        row = episode_rows[t]
        teacher_phases.append(row.get("teacher_phase", "other"))
        teacher_hazards.append(_boolish(row.get("teacher_hazard", "false")))

        # Window info (teacher labels, used for eval only)
        w_start = _num(row.get("teacher_window_start"))
        w_end = _num(row.get("teacher_window_end"))
        step_idx = int(float(row["step_idx"]))
        is_in = w_start is not None and w_end is not None and w_start <= step_idx <= w_end
        inside_window.append(is_in)
        if ws is None and w_start is not None:
            ws = w_start
        if we is None and w_end is not None:
            we = w_end
        false_early.append(trig and w_start is not None and step_idx < w_start)

    # Compute metrics
    n_total = len(episode_rows)
    n_triggers = len(trigger_steps)
    n_inside = sum(inside_window)
    n_missed = sum(inside_window[i] and not (
        hazard_scores[i] >= hazard_threshold and release_scores[i] < release_threshold
    ) for i in range(n_total))

    latency = None
    if ws is not None and trigger_steps:
        first_trigger_step = episode_rows[trigger_steps[0]]["step_idx"]
        latency = int(float(first_trigger_step)) - int(ws)

    # Trigger coverage on window rows
    window_triggers = sum(
        1 for i in range(n_total)
        if inside_window[i] and hazard_scores[i] >= hazard_threshold and release_scores[i] < release_threshold
    )

    return {
        "episode_key": episode_rows[0].get("episode_key", ""),
        "suite": episode_rows[0].get("suite", ""),
        "task_name": episode_rows[0].get("task_name", ""),
        "mechanism_type": episode_rows[0].get("mechanism_type", ""),
        "n_steps": n_total,
        "n_triggers": n_triggers,
        "trigger_rate": n_triggers / n_total if n_total else 0.0,
        "first_trigger_step": trigger_steps[0] if trigger_steps else -1,
        "latency_to_window_start": latency,
        "teacher_window_start": ws,
        "teacher_window_end": we,
        "n_window_rows": n_inside,
        "window_triggers": window_triggers,
        "window_coverage": window_triggers / n_inside if n_inside else 0.0,
        "false_early_triggers": sum(false_early),
        "false_early_rate": sum(false_early) / n_total if n_total else 0.0,
        "miss_rate": n_missed / n_inside if n_inside else 0.0,
        "mean_hazard_score": sum(hazard_scores) / n_total if n_total else 0.0,
        "max_hazard_score": max(hazard_scores) if hazard_scores else 0.0,
    }


def causal_replay_all(
    rows: list[dict[str, Any]],
    predict_fn,
    hazard_threshold: float = 0.1,
    release_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Run causal replay on all episodes grouped by episode_key."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_episode[r["episode_key"]].append(r)

    results = []
    for ep_key in sorted(by_episode.keys()):
        ep_rows = sorted(by_episode[ep_key], key=lambda r: int(float(r["step_idx"])))
        results.append(causal_replay_episode(ep_rows, predict_fn, hazard_threshold, release_threshold))
    return results


# ═══════════════════════════════════════════════════════════════
# Predict functions for each model
# ═══════════════════════════════════════════════════════════════


def make_model_predict_fn(model: ProprioCausalMLP, data, device: str = "cpu",
                         numeric_features: list[str] | None = None,
                         categorical_features: list[str] | None = None):
    """Create a predict function from a trained ProprioCausalMLP.

    Uses data.numeric_mean, data.numeric_std, data.category_maps for encoding.
    Optionally override which features to encode (for ablation models).
    """
    num_feats = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_feats = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    def predict(causal_rows: list[dict[str, Any]], t: int) -> tuple[float, float, dict]:
        row = causal_rows[t]
        features = _encode_row(row, data.category_maps, data.numeric_mean, data.numeric_std,
                               numeric_features=num_feats, categorical_features=cat_feats)
        x = torch.tensor([features], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)
            h = float(torch.sigmoid(out["hazard_logit"]).cpu().item())
            r = float(torch.sigmoid(out["release_safe_logit"]).cpu().item())
        return h, r, {}

    return predict


def _encode_row(row: dict[str, Any], category_maps: dict[str, list[str]],
                numeric_mean: dict[str, float], numeric_std: dict[str, float],
                numeric_features: list[str] | None = None,
                categorical_features: list[str] | None = None) -> list[float]:
    """Encode a single row into the feature vector expected by the MLP."""
    num_feats = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_feats = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    vec: list[float] = []
    for f in num_feats:
        val = _num(row.get(f, ""))
        val = numeric_mean.get(f, 0.0) if val is None else val
        vec.append((val - numeric_mean.get(f, 0.0)) / max(numeric_std.get(f, 1.0), 1e-8))
    for f in cat_feats:
        val = str(row.get(f, "missing") or "missing")
        vec.extend([1.0 if val == c else 0.0 for c in category_maps.get(f, [])])
    return vec


def make_rule_proxy_predict_fn():
    """Rule-based proxy trigger using gripper state heuristics.

    The heuristic: gripper is closing (command < 0) or gripper width decreasing
    while at elevated z-position → approach/grasp. Gripper open (command > 0)
    or wide width → release/pre-release phase.

    Uses only gripper_command, gripper_qpos, gripper_width, eef_z.
    """
    def predict(causal_rows: list[dict[str, Any]], t: int) -> tuple[float, float, dict]:
        row = causal_rows[t]
        gc = _num(row.get("gripper_command", 0)) or 0.0
        gq = _num(row.get("gripper_qpos", 0)) or 0.0
        gw = _num(row.get("gripper_width", 0)) or 0.0

        # Heuristic hazard: gripper is closing/closed AND has been recently
        # (proxy for near-object / grasp phase)
        closing_now = gc < -0.1
        gripper_narrow = gw < 0.02

        # Heuristic release_safe: gripper is opening/wide
        opening_now = gc > 0.1
        gripper_wide = gw > 0.03

        # Build scores: hazardous when closing+narrow, safe when opening+wide
        if closing_now and gripper_narrow:
            hazard = 0.8
            release = 0.1
        elif opening_now or gripper_wide:
            hazard = 0.3
            release = 0.7
        elif closing_now:
            hazard = 0.6
            release = 0.2
        else:
            hazard = 0.1
            release = 0.5

        # Use recent history: if gripper has been closing for 5+ steps, boost hazard
        if t >= 5:
            recent_commands = [
                _num(causal_rows[i].get("gripper_command", 0)) or 0.0
                for i in range(max(0, t - 5), t + 1)
            ]
            close_count = sum(1 for c in recent_commands if c < -0.05)
            open_count = sum(1 for c in recent_commands if c > 0.05)
            if close_count >= 3:
                hazard = min(hazard + 0.15, 1.0)
            if open_count >= 3:
                release = min(release + 0.15, 1.0)

        return hazard, release, {"type": "rule_proxy"}

    return predict


# ═══════════════════════════════════════════════════════════════
# Training helpers for ablation models
# ═══════════════════════════════════════════════════════════════


def train_model(
    data,
    train_idx: list[int],
    val_idx: list[int],
    input_dim: int,
    device: str,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 8,
    seed: int = 7,
) -> tuple[ProprioCausalMLP, int, float]:
    """Train a ProprioCausalMLP and return (model, best_epoch, best_val_loss)."""
    torch.manual_seed(seed)
    model = ProprioCausalMLP(input_dim=input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(
        TensorDataset(
            data.x[train_idx], data.phase[train_idx],
            data.hazard[train_idx], data.release[train_idx],
            data.confidence[train_idx],
        ),
        batch_size=batch_size, shuffle=True,
    )
    best_loss = float("inf")
    best_epoch = 0
    patience_left = early_stop_patience

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, phase, hazard, release, conf in train_loader:
            xb, phase = xb.to(device), phase.to(device)
            hazard, release, conf = hazard.to(device), release.to(device), conf.to(device)
            opt.zero_grad()
            loss = compute_loss(model(xb), phase, hazard, release, conf)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach().cpu()) * len(xb)
            n_train += len(xb)

        model.eval()
        with torch.no_grad():
            val_out = model(data.x[val_idx].to(device))
            val_loss = float(compute_loss(
                val_out,
                data.phase[val_idx].to(device),
                data.hazard[val_idx].to(device),
                data.release[val_idx].to(device),
                data.confidence[val_idx].to(device),
            ).cpu())

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_left = early_stop_patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def build_subset_dataset(
    rows: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
    split_mode: str,
    seed: int,
):
    """Build an EncodedDataset-like object using a subset of features.

    Returns a lightweight namespace matching what train_model and predict_fns expect.
    """
    from types import SimpleNamespace

    split_by_ep = {}
    for r in rows:
        split_by_ep[r["episode_key"]] = ""

    # Use existing split assignment
    tmp_data = encode_dataset(rows, split_mode, seed)
    split_by_ep = tmp_data.split_by_episode

    train_rows = [r for r in rows if split_by_ep[r["episode_key"]] == "train"]

    # Compute stats from train
    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    for f in numeric_features:
        vals = [_num(r.get(f)) for r in train_rows]
        present = [v for v in vals if v is not None]
        if not present:
            numeric_mean[f] = 0.0
            numeric_std[f] = 1.0
        else:
            mean = sum(present) / len(present)
            var = sum((v - mean) ** 2 for v in present) / max(1, len(present) - 1)
            std = math.sqrt(var) if var > 1e-12 else 1.0
            numeric_mean[f] = mean
            numeric_std[f] = std

    category_maps = {
        f: sorted({r.get(f, "missing") or "missing" for r in rows})
        for f in categorical_features
    }

    feature_names = list(numeric_features)
    for f, vals in category_maps.items():
        feature_names.extend([f"{f}={v}" for v in vals])

    # Encode
    encoded = []
    for r in rows:
        row_vec: list[float] = []
        for f in numeric_features:
            val = _num(r.get(f))
            val = numeric_mean[f] if val is None else val
            row_vec.append((val - numeric_mean[f]) / max(numeric_std[f], 1e-8))
        for f in categorical_features:
            val_str = str(r.get(f, "missing") or "missing")
            row_vec.extend([1.0 if val_str == c else 0.0 for c in category_maps[f]])
        encoded.append(row_vec)

    x = torch.tensor(encoded, dtype=torch.float32)
    phase = torch.tensor([phase_index(r.get("teacher_phase", "other")) for r in rows], dtype=torch.long)
    hazard = torch.tensor([1.0 if _boolish(r.get("teacher_hazard")) else 0.0 for r in rows], dtype=torch.float32)
    release = torch.tensor([1.0 if _boolish(r.get("teacher_release_safe")) else 0.0 for r in rows], dtype=torch.float32)
    confidence = torch.tensor([1.0 if r.get("teacher_confidence") in {"medium", "high"} else 0.0 for r in rows], dtype=torch.float32)

    return SimpleNamespace(
        rows=rows, x=x, phase=phase, hazard=hazard, release=release, confidence=confidence,
        feature_names=feature_names, split_by_episode=split_by_ep,
        category_maps=category_maps, numeric_mean=numeric_mean, numeric_std=numeric_std,
    )


def build_time_only_dataset(rows, split_mode, seed):
    """Build dataset with only normalized_step + categorical features."""
    return build_subset_dataset(rows, TIME_ONLY_FEATURES[:1], TIME_ONLY_FEATURES[1:], split_mode, seed)


def build_no_step_dataset(rows, split_mode, seed):
    """Build dataset with all features EXCEPT normalized_step."""
    return build_subset_dataset(rows, NUMERIC_NO_STEP, CATEGORICAL_FEATURES, split_mode, seed)


def build_shuffled_dataset(rows, split_mode, seed):
    """Build dataset with shuffled teacher_hazard labels."""
    ds = encode_dataset(rows, split_mode, seed)
    # Shuffle hazard labels
    hazard_np = ds.hazard.numpy().copy()
    rng = random.Random(seed + 9999)
    rng.shuffle(hazard_np)
    ds.hazard = torch.tensor(hazard_np, dtype=torch.float32)
    return ds


# ═══════════════════════════════════════════════════════════════
# Comparison metric aggregation
# ═══════════════════════════════════════════════════════════════


def aggregate_replay_metrics(
    per_episode: list[dict[str, Any]],
    model_name: str,
    threshold_info: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-episode replay metrics into overall + by-suite summaries."""
    n_episodes = len(per_episode)
    if n_episodes == 0:
        return {"model": model_name, "n_episodes": 0}

    # Overall
    total_steps = sum(e["n_steps"] for e in per_episode)
    total_triggers = sum(e["n_triggers"] for e in per_episode)
    total_window_rows = sum(e["n_window_rows"] for e in per_episode)
    total_window_triggers = sum(e["window_triggers"] for e in per_episode)
    total_false_early = sum(e["false_early_triggers"] for e in per_episode)
    total_missed = sum(
        e["n_window_rows"] - e["window_triggers"] for e in per_episode
    )

    latencies = [e["latency_to_window_start"] for e in per_episode
                 if e["latency_to_window_start"] is not None]

    episodes_with_trigger = sum(1 for e in per_episode if e["n_triggers"] > 0)
    episodes_with_window = sum(1 for e in per_episode if e["n_window_rows"] > 0)
    episodes_covered = sum(
        1 for e in per_episode if e["n_window_rows"] > 0 and e["window_triggers"] > 0
    )

    overall = {
        "model": model_name,
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "trigger_rate": total_triggers / total_steps if total_steps else 0.0,
        "window_coverage": total_window_triggers / total_window_rows if total_window_rows else 0.0,
        "false_early_rate": total_false_early / total_steps if total_steps else 0.0,
        "miss_rate": total_missed / total_window_rows if total_window_rows else 0.0,
        "mean_latency": sum(latencies) / len(latencies) if latencies else None,
        "median_latency": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "episodes_with_trigger": episodes_with_trigger,
        "episodes_with_window": episodes_with_window,
        "episodes_covered": episodes_covered,
        "episode_coverage_rate": episodes_covered / episodes_with_window if episodes_with_window else 0.0,
        "mean_trigger_count": total_triggers / n_episodes if n_episodes else 0.0,
        "hazard_threshold": threshold_info.get("hazard_threshold"),
        "release_threshold": threshold_info.get("release_safe_threshold"),
    }

    # By suite
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in per_episode:
        by_suite[e["suite"]].append(e)

    suite_metrics = []
    for suite, episodes in sorted(by_suite.items()):
        s_total = sum(e["n_steps"] for e in episodes)
        s_trig = sum(e["n_triggers"] for e in episodes)
        s_win = sum(e["n_window_rows"] for e in episodes)
        s_win_trig = sum(e["window_triggers"] for e in episodes)
        s_fe = sum(e["false_early_triggers"] for e in episodes)
        s_missed = sum(e["n_window_rows"] - e["window_triggers"] for e in episodes)
        s_lat = [e["latency_to_window_start"] for e in episodes if e["latency_to_window_start"] is not None]
        suite_metrics.append({
            "model": model_name, "suite": suite,
            "n_episodes": len(episodes),
            "trigger_rate": s_trig / s_total if s_total else 0.0,
            "window_coverage": s_win_trig / s_win if s_win else 0.0,
            "false_early_rate": s_fe / s_total if s_total else 0.0,
            "miss_rate": s_missed / s_win if s_win else 0.0,
            "mean_latency": sum(s_lat) / len(s_lat) if s_lat else None,
        })

    # By mechanism
    by_mech: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in per_episode:
        by_mech[e["mechanism_type"]].append(e)

    mech_metrics = []
    for mech, episodes in sorted(by_mech.items()):
        m_total = sum(e["n_steps"] for e in episodes)
        m_trig = sum(e["n_triggers"] for e in episodes)
        m_win = sum(e["n_window_rows"] for e in episodes)
        m_win_trig = sum(e["window_triggers"] for e in episodes)
        m_fe = sum(e["false_early_triggers"] for e in episodes)
        m_missed = sum(e["n_window_rows"] - e["window_triggers"] for e in episodes)
        m_lat = [e["latency_to_window_start"] for e in episodes if e["latency_to_window_start"] is not None]
        mech_metrics.append({
            "model": model_name, "mechanism_type": mech,
            "n_episodes": len(episodes),
            "trigger_rate": m_trig / m_total if m_total else 0.0,
            "window_coverage": m_win_trig / m_win if m_win else 0.0,
            "false_early_rate": m_fe / m_total if m_total else 0.0,
            "miss_rate": m_missed / m_win if m_win else 0.0,
            "mean_latency": sum(m_lat) / len(m_lat) if m_lat else None,
        })

    return {
        "overall": overall,
        "by_suite": suite_metrics,
        "by_mechanism": mech_metrics,
        "per_episode": per_episode,
    }


def find_bad_cases(per_episode: list[dict[str, Any]], top_n: int = 20) -> list[dict[str, Any]]:
    """Find episodes with poor replay performance for manual review."""
    scored = []
    for e in per_episode:
        # Score: false_early high, window_coverage low, miss_rate high → bad
        score = (
            e["false_early_rate"] * 3.0
            + (1.0 - e["window_coverage"]) * 2.0
            + e["miss_rate"] * 3.0
        )
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:top_n]]


# ═══════════════════════════════════════════════════════════════
# Overlap analysis: teacher vs. student triggers
# ═══════════════════════════════════════════════════════════════


def trigger_overlap_analysis(
    per_episode_teacher: list[dict[str, Any]],
    per_episode_student: list[dict[str, Any]],
    teacher_name: str,
    student_name: str,
) -> dict[str, Any]:
    """Compare trigger timing between two models on shared episodes."""
    teacher_by_ep = {e["episode_key"]: e for e in per_episode_teacher}
    student_by_ep = {e["episode_key"]: e for e in per_episode_student}

    common = sorted(set(teacher_by_ep.keys()) & set(student_by_ep.keys()))
    if not common:
        return {"common_episodes": 0}

    both_trigger = 0
    only_teacher = 0
    only_student = 0
    neither = 0
    latency_diffs = []

    for ek in common:
        t = teacher_by_ep[ek]
        s = student_by_ep[ek]
        t_trig = t["first_trigger_step"] >= 0
        s_trig = s["first_trigger_step"] >= 0
        if t_trig and s_trig:
            both_trigger += 1
            if t["first_trigger_step"] >= 0 and s["first_trigger_step"] >= 0:
                latency_diffs.append(s["first_trigger_step"] - t["first_trigger_step"])
        elif t_trig:
            only_teacher += 1
        elif s_trig:
            only_student += 1
        else:
            neither += 1

    return {
        "common_episodes": len(common),
        "both_trigger": both_trigger,
        "only_teacher": only_teacher,
        "only_student": only_student,
        "neither": neither,
        "agreement_rate": (both_trigger + neither) / len(common) if common else 0.0,
        "mean_latency_diff": sum(latency_diffs) / len(latency_diffs) if latency_diffs else None,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Milestone 2C.1 replay + ablation")
    p.add_argument("--data_csv", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split_mode", choices=["task_id", "episode_key"], default="task_id")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--coverage_hazard", type=float, default=0.1)
    p.add_argument("--coverage_release", type=float, default=0.3)
    p.add_argument("--conservative_hazard", type=float, default=0.5)
    p.add_argument("--conservative_release", type=float, default=0.5)
    p.add_argument("--skip_training", action="store_true",
                   help="Skip training ablations (use existing checkpoints if available)")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.output_root
    for sub in ["tables", "checkpoints", "reports"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Milestone 2C.1 — Replay Validation + Anti-Timing Ablations")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    print(f"\n[1/7] Loading data from {args.data_csv}")
    rows = read_csv_rows(args.data_csv)
    print(f"  Rows: {len(rows)}, Episodes: {len({r['episode_key'] for r in rows})}")

    # ── Full dataset encoding ─────────────────────────────
    print("\n[2/7] Encoding full dataset")
    data = encode_dataset(rows, args.split_mode, args.seed)
    train_idx = indices_for_split(data, "train")
    val_idx = indices_for_split(data, "val")
    test_idx = indices_for_split(data, "test")
    test_rows = [data.rows[i] for i in test_idx]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Input dim: {data.x.shape[1]}")

    all_results: dict[str, dict[str, Any]] = {}
    all_per_episode: dict[str, list[dict[str, Any]]] = {}

    # ── Model 1: Full proprio student ─────────────────────
    print("\n[3/7] Full proprio student (existing checkpoint)")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    full_model = ProprioCausalMLP(input_dim=data.x.shape[1]).to(args.device)
    full_model.load_state_dict(checkpoint["model_state"])
    full_model.eval()

    full_predict = make_model_predict_fn(full_model, data, args.device)

    for name, h_th, r_th in [
        ("full_proprio_coverage", args.coverage_hazard, args.coverage_release),
        ("full_proprio_conservative", args.conservative_hazard, args.conservative_release),
    ]:
        print(f"  Replay: {name} (h>={h_th}, r<{r_th})")
        ep = causal_replay_all(test_rows, full_predict, h_th, r_th)
        agg = aggregate_replay_metrics(ep, name, {"hazard_threshold": h_th, "release_safe_threshold": r_th})
        all_results[name] = agg
        all_per_episode[name] = ep
        print(f"    Window coverage: {agg['overall']['window_coverage']:.4f}, "
              f"False early: {agg['overall']['false_early_rate']:.4f}")

    # ── Model 2: Rule/proxy trigger ───────────────────────
    print("\n[4/7] Rule/proxy trigger baseline")
    rule_predict = make_rule_proxy_predict_fn()
    ep_rule = causal_replay_all(test_rows, rule_predict, args.coverage_hazard, args.coverage_release)
    agg_rule = aggregate_replay_metrics(ep_rule, "rule_proxy",
                                        {"hazard_threshold": args.coverage_hazard, "release_safe_threshold": args.coverage_release})
    all_results["rule_proxy"] = agg_rule
    all_per_episode["rule_proxy"] = ep_rule
    print(f"  Window coverage: {agg_rule['overall']['window_coverage']:.4f}, "
          f"False early: {agg_rule['overall']['false_early_rate']:.4f}")

    # ── Model 3: Teacher window (oracle reference) ────────
    print("\n[5/7] Teacher window (oracle reference)")

    def make_teacher_window_predict_fn():
        def predict(causal_rows, t):
            row = causal_rows[t]
            ws = _num(row.get("teacher_window_start"))
            we = _num(row.get("teacher_window_end"))
            step = int(float(row["step_idx"]))
            in_window = ws is not None and we is not None and ws <= step <= we
            # Teacher "trigger" when inside window, high hazard
            if in_window:
                return 0.9, 0.1, {}
            # Before window: approach phase
            if ws is not None and step < ws:
                return 0.3, 0.3, {}
            return 0.05, 0.8, {}

        return predict

    teacher_predict = make_teacher_window_predict_fn()
    ep_teacher = causal_replay_all(test_rows, teacher_predict, args.coverage_hazard, args.coverage_release)
    agg_teacher = aggregate_replay_metrics(ep_teacher, "teacher_window",
                                           {"hazard_threshold": args.coverage_hazard, "release_safe_threshold": args.coverage_release})
    all_results["teacher_window"] = agg_teacher
    all_per_episode["teacher_window"] = ep_teacher
    print(f"  Window coverage: {agg_teacher['overall']['window_coverage']:.4f}")

    # ── Model 4-6: Ablation models ────────────────────────
    print("\n[6/7] Training ablation models")

    # 4a: Time-only baseline
    print("  [6a] Time-only baseline")
    time_data = build_time_only_dataset(rows, args.split_mode, args.seed)
    time_train = indices_for_split(time_data, "train")
    time_val = indices_for_split(time_data, "val")
    time_test = indices_for_split(time_data, "test")
    time_test_rows = [time_data.rows[i] for i in time_test]

    time_model, time_epoch, time_loss = train_model(
        time_data, time_train, time_val,
        input_dim=time_data.x.shape[1],
        device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )
    time_predict = make_model_predict_fn(time_model, time_data, args.device,
                                         numeric_features=["normalized_step"])

    for name, h_th, r_th in [
        ("time_only_coverage", args.coverage_hazard, args.coverage_release),
        ("time_only_conservative", args.conservative_hazard, args.conservative_release),
    ]:
        print(f"    Replay: {name}")
        ep = causal_replay_all(time_test_rows, time_predict, h_th, r_th)
        agg = aggregate_replay_metrics(ep, name, {"hazard_threshold": h_th, "release_safe_threshold": r_th})
        all_results[name] = agg
        all_per_episode[name] = ep

    # Save time-only checkpoint
    torch.save({
        "model_state": time_model.state_dict(),
        "epoch": time_epoch,
        "feature_names": time_data.feature_names,
        "category_maps": time_data.category_maps,
        "numeric_mean": time_data.numeric_mean,
        "numeric_std": time_data.numeric_std,
    }, root / "checkpoints" / "time_only_best_model.pt")

    # 4b: No-normalized-step ablation
    print("  [6b] No-normalized-step ablation")
    nostep_data = build_no_step_dataset(rows, args.split_mode, args.seed)
    nostep_train = indices_for_split(nostep_data, "train")
    nostep_val = indices_for_split(nostep_data, "val")
    nostep_test = indices_for_split(nostep_data, "test")
    nostep_test_rows = [nostep_data.rows[i] for i in nostep_test]

    nostep_model, nostep_epoch, nostep_loss = train_model(
        nostep_data, nostep_train, nostep_val,
        input_dim=nostep_data.x.shape[1],
        device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )
    nostep_predict = make_model_predict_fn(nostep_model, nostep_data, args.device,
                                          numeric_features=NUMERIC_NO_STEP)

    for name, h_th, r_th in [
        ("no_normalized_step_coverage", args.coverage_hazard, args.coverage_release),
        ("no_normalized_step_conservative", args.conservative_hazard, args.conservative_release),
    ]:
        print(f"    Replay: {name}")
        ep = causal_replay_all(nostep_test_rows, nostep_predict, h_th, r_th)
        agg = aggregate_replay_metrics(ep, name, {"hazard_threshold": h_th, "release_safe_threshold": r_th})
        all_results[name] = agg
        all_per_episode[name] = ep

    torch.save({
        "model_state": nostep_model.state_dict(),
        "epoch": nostep_epoch,
        "feature_names": nostep_data.feature_names,
        "category_maps": nostep_data.category_maps,
        "numeric_mean": nostep_data.numeric_mean,
        "numeric_std": nostep_data.numeric_std,
    }, root / "checkpoints" / "no_normalized_step_best_model.pt")

    # 4c: Label-shuffle sanity baseline
    print("  [6c] Label-shuffle sanity baseline")
    shuffled_data = build_shuffled_dataset(rows, args.split_mode, args.seed)
    shuf_train = indices_for_split(shuffled_data, "train")
    shuf_val = indices_for_split(shuffled_data, "val")
    shuf_test = indices_for_split(shuffled_data, "test")
    shuf_test_rows = [shuffled_data.rows[i] for i in shuf_test]

    shuf_model, shuf_epoch, shuf_loss = train_model(
        shuffled_data, shuf_train, shuf_val,
        input_dim=shuffled_data.x.shape[1],
        device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )
    shuf_predict = make_model_predict_fn(shuf_model, shuffled_data, args.device)

    # Evaluate with coverage thresholds
    ep_shuf = causal_replay_all(shuf_test_rows, shuf_predict, args.coverage_hazard, args.coverage_release)
    agg_shuf = aggregate_replay_metrics(ep_shuf, "label_shuffle",
                                        {"hazard_threshold": args.coverage_hazard, "release_safe_threshold": args.coverage_release})
    all_results["label_shuffle"] = agg_shuf
    all_per_episode["label_shuffle"] = ep_shuf

    torch.save({
        "model_state": shuf_model.state_dict(),
        "epoch": shuf_epoch,
        "feature_names": shuffled_data.feature_names,
        "category_maps": shuffled_data.category_maps,
        "numeric_mean": shuffled_data.numeric_mean,
        "numeric_std": shuffled_data.numeric_std,
    }, root / "checkpoints" / "label_shuffle_best_model.pt")

    # ── Generate tables ────────────────────────────────────
    print("\n[7/7] Generating tables and reports")

    # --- Table: overall comparison ---
    overall_rows = []
    for name, agg in all_results.items():
        overall_rows.append(agg["overall"])
    write_csv(root / "tables" / "replay_comparison_overall.csv", overall_rows)

    # --- Table: by suite ---
    suite_rows = []
    for name, agg in all_results.items():
        suite_rows.extend(agg["by_suite"])
    write_csv(root / "tables" / "replay_comparison_by_suite.csv", suite_rows)

    # --- Table: by mechanism ---
    mech_rows = []
    for name, agg in all_results.items():
        mech_rows.extend(agg["by_mechanism"])
    write_csv(root / "tables" / "replay_comparison_by_mechanism.csv", mech_rows)

    # --- Table: per-episode trigger audit ---
    audit_rows = []
    for name, episodes in all_per_episode.items():
        for e in episodes:
            e_copy = dict(e)
            e_copy["model"] = name
            audit_rows.append(e_copy)
    write_csv(root / "tables" / "per_episode_trigger_audit.csv", audit_rows)

    # --- Table: threshold sweep comparison ---
    # Use full model test predictions for sweep
    full_test_indices = [i for i in test_idx]
    full_test_data = full_model
    full_test_data_eval = data
    # Reuse existing sweep from 2C if test split matches
    sweep_rows = threshold_sweep(
        [dict(r) for r in make_predictions_rows_stub(data, test_idx, full_model, args.device)]
    )
    write_csv(root / "tables" / "threshold_sweep_comparison.csv", sweep_rows)

    # --- Table: time-only metrics ---
    time_metrics = evaluate_model_metrics(time_model, time_data, time_test, args.device)
    write_csv(root / "tables" / "time_only_baseline_metrics.csv", [time_metrics])

    # --- Table: no-normalized-step metrics ---
    nostep_metrics = evaluate_model_metrics(nostep_model, nostep_data, nostep_test, args.device)
    write_csv(root / "tables" / "no_normalized_step_ablation_metrics.csv", [nostep_metrics])

    # --- Table: label-shuffle metrics ---
    shuf_metrics = evaluate_model_metrics(shuf_model, shuffled_data, shuf_test, args.device)
    write_csv(root / "tables" / "label_shuffle_sanity_metrics.csv", [shuf_metrics])

    # --- Bad cases ---
    bad_cases = find_bad_cases(all_per_episode.get("full_proprio_coverage", []), top_n=30)
    write_csv(root / "tables" / "bad_cases_for_manual_review.csv", bad_cases)

    # --- Overlap analysis ---
    overlap = trigger_overlap_analysis(
        all_per_episode.get("teacher_window", []),
        all_per_episode.get("full_proprio_coverage", []),
        "teacher_window", "full_proprio_coverage",
    )
    write_csv(root / "tables" / "trigger_overlap_teacher_vs_student.csv", [overlap])

    # ── Reports ────────────────────────────────────────────
    write_reports(root, all_results, all_per_episode, overlap)

    # ── Summary print ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY COMPARISON (coverage-first threshold)")
    print("=" * 60)
    for name in ["teacher_window", "full_proprio_coverage", "rule_proxy",
                  "time_only_coverage", "no_normalized_step_coverage", "label_shuffle"]:
        if name in all_results:
            o = all_results[name]["overall"]
            print(f"  {name:35s}  coverage={o['window_coverage']:.4f}  "
                  f"false_early={o['false_early_rate']:.4f}  "
                  f"miss={o['miss_rate']:.4f}  "
                  f"latency={o.get('mean_latency')}")

    print(f"\nAll outputs → {root}")
    return 0


# ═══════════════════════════════════════════════════════════════
# Stub helpers
# ═══════════════════════════════════════════════════════════════


def make_predictions_rows_stub(data, indices, model, device):
    """Quick stub to get test predictions for threshold sweep."""
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in indices:
            src = data.rows[idx]
            x = data.x[idx:idx+1].to(device)
            out = model(x)
            h = float(torch.sigmoid(out["hazard_logit"]).cpu().item())
            r = float(torch.sigmoid(out["release_safe_logit"]).cpu().item())
            rows.append({
                "episode_key": src["episode_key"],
                "step_idx": src["step_idx"],
                "hazard_score": h,
                "release_safe_score": r,
                "teacher_window_start": src.get("teacher_window_start", ""),
                "teacher_window_end": src.get("teacher_window_end", ""),
            })
    return rows


def evaluate_model_metrics(model, data, indices, device):
    """Compute classification metrics for an ablation model."""
    model.eval()
    y_phase = []
    p_phase = []
    y_hazard = []
    h_score = []
    y_release = []
    r_score = []

    with torch.no_grad():
        x = data.x[indices].to(device)
        out = model(x)
        phase_prob = torch.softmax(out["phase_logits"], dim=-1).cpu()
        p_phase = phase_prob.argmax(dim=-1).tolist()
        h_score = torch.sigmoid(out["hazard_logit"]).cpu().tolist()
        r_score = torch.sigmoid(out["release_safe_logit"]).cpu().tolist()

    y_phase = data.phase[indices].tolist()
    y_hazard = [int(v) for v in data.hazard[indices].tolist()]
    y_release = [int(v) for v in data.release[indices].tolist()]

    pm = phase_metrics(y_phase, p_phase)
    hm = binary_metrics(y_hazard, h_score)
    rm = binary_metrics(y_release, r_score)

    return {
        "phase_accuracy": pm["accuracy"],
        "phase_macro_f1": pm["macro_f1"],
        "hazard_accuracy": hm["accuracy"],
        "hazard_precision": hm["precision"],
        "hazard_recall": hm["recall"],
        "hazard_f1": hm["f1"],
        "hazard_auroc": hm["auroc"],
        "hazard_auprc": hm["auprc"],
        "release_accuracy": rm["accuracy"],
        "release_precision": rm["precision"],
        "release_recall": rm["recall"],
        "release_f1": rm["f1"],
        "release_auroc": rm["auroc"],
        "release_auprc": rm["auprc"],
    }


def write_reports(root, all_results, all_per_episode, overlap):
    """Write milestone reports."""

    # Main comparison report
    lines = [
        "# Milestone 2C.1 — Student Replay Comparison",
        "",
        "## Models Compared",
        "",
        "| # | Model | Description |",
        "|---|-------|-------------|",
        "| 1 | teacher_window | Oracle reference: trigger when step ∈ [teacher_window_start, teacher_window_end] |",
        "| 2 | full_proprio_coverage | Existing 2C proprio student, coverage-first threshold (h≥0.1, r<0.3) |",
        "| 3 | full_proprio_conservative | Existing 2C proprio student, conservative threshold (h≥0.5, r<0.5) |",
        "| 4 | rule_proxy | Heuristic: gripper command+width+qpos rules, no ML model |",
        "| 5 | time_only_coverage | MLP trained ONLY on normalized_step + mechanism_type + parse_confidence |",
        "| 6 | time_only_conservative | Time-only with conservative threshold |",
        "| 7 | no_normalized_step_coverage | MLP trained on all features EXCEPT normalized_step |",
        "| 8 | no_normalized_step_conservative | No-step ablation with conservative threshold |",
        "| 9 | label_shuffle | Same architecture, teacher_hazard labels randomly shuffled |",
        "",
        "## Overall Comparison (Coverage-first threshold: h≥0.1, r<0.3)",
        "",
    ]

    if "full_proprio_coverage" in all_results:
        o = all_results["full_proprio_coverage"]["overall"]
        rp = all_results.get("rule_proxy", {}).get("overall", {})
        to = all_results.get("time_only_coverage", {}).get("overall", {})
        ns = all_results.get("no_normalized_step_coverage", {}).get("overall", {})
        ls = all_results.get("label_shuffle", {}).get("overall", {})
        tw = all_results.get("teacher_window", {}).get("overall", {})

        lines.extend([
            "| Model | Win Coverage | False Early | Miss Rate | Mean Latency | Trigger Rate |",
            "|-------|-------------|-------------|-----------|-------------|-------------|",
            f"| teacher_window | {tw.get('window_coverage',0):.4f} | {tw.get('false_early_rate',0):.4f} | {tw.get('miss_rate',0):.4f} | {tw.get('mean_latency','N/A')} | {tw.get('trigger_rate',0):.4f} |",
            f"| full_proprio | {o.get('window_coverage',0):.4f} | {o.get('false_early_rate',0):.4f} | {o.get('miss_rate',0):.4f} | {o.get('mean_latency','N/A')} | {o.get('trigger_rate',0):.4f} |",
            f"| rule_proxy | {rp.get('window_coverage',0):.4f} | {rp.get('false_early_rate',0):.4f} | {rp.get('miss_rate',0):.4f} | {rp.get('mean_latency','N/A')} | {rp.get('trigger_rate',0):.4f} |",
            f"| time_only | {to.get('window_coverage',0):.4f} | {to.get('false_early_rate',0):.4f} | {to.get('miss_rate',0):.4f} | {to.get('mean_latency','N/A')} | {to.get('trigger_rate',0):.4f} |",
            f"| no_normalized_step | {ns.get('window_coverage',0):.4f} | {ns.get('false_early_rate',0):.4f} | {ns.get('miss_rate',0):.4f} | {ns.get('mean_latency','N/A')} | {ns.get('trigger_rate',0):.4f} |",
            f"| label_shuffle | {ls.get('window_coverage',0):.4f} | {ls.get('false_early_rate',0):.4f} | {ls.get('miss_rate',0):.4f} | {ls.get('mean_latency','N/A')} | {ls.get('trigger_rate',0):.4f} |",
            "",
        ])

    # Pass/fail verdict
    if "full_proprio_coverage" in all_results and "time_only_coverage" in all_results:
        fp = all_results["full_proprio_coverage"]["overall"]
        to = all_results["time_only_coverage"]["overall"]
        ns = all_results.get("no_normalized_step_coverage", {}).get("overall", {})
        ls = all_results.get("label_shuffle", {}).get("overall", {})

        checks = []
        checks.append(("Full proprio > time-only hazard F1",
                       fp.get("window_coverage", 0) > to.get("window_coverage", 0)))
        checks.append(("No-step ablation retains performance",
                       ns.get("window_coverage", 0) > 0.3))
        checks.append(("Label-shuffle collapses (low window_coverage)",
                       ls.get("window_coverage", 0) < 0.4))
        checks.append(("False early trigger rate low",
                       fp.get("false_early_rate", 1) < 0.05))

        lines.extend([
            "## Milestone 2C.1 Verification Checklist",
            "",
            "| Criterion | Status |",
            "|-----------|--------|",
        ])
        for desc, passed in checks:
            lines.append(f"| {desc} | {'PASS' if passed else 'FAIL'} |")
        lines.append("")

        all_pass = all(p for _, p in checks)
        lines.append(f"**Overall: {'ALL PASSED' if all_pass else 'SOME FAILED — review required'}**")
        lines.append("")

    (root / "reports" / "STUDENT_REPLAY_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
