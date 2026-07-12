"""Calibrate R9P preview detector thresholds on PREVIEW_CAL and evaluate on PREVIEW_CHECK.

Uses the full 6-head model with per-head thresholds. Computes real T10 window metrics:
feasible_hit, full_T10_containment, start_delay, early_trigger, late_trigger,
negative_episode_any_trigger, release_safe_emit.

The scheduler uses: critical_window, release_safe, grounding_confidence for gating.
Evaluation measures whether the resulting trigger aligns with the teacher's window_start
and whether the full 10-step burst window has critical support.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import Tensor

from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9PEpisodeDataset,
    _hash_language_embedding,
    load_normalization,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    FixedBurstTriggerScheduler,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

SCHEMA = "c2g.r9p.preview_calibration.2026-07-12.v1"


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[C2gGripperCriticalWindowDetector, dict]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg = raw["model_config"]
    config = C2gDetectorConfig(**cfg)
    model = C2gGripperCriticalWindowDetector(config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model, raw


def evaluate_episode_t10(
    model: C2gGripperCriticalWindowDetector,
    episode_data: dict,
    device: torch.device,
    use_policy_intent: bool,
    scheduler_kwargs: dict,
    norm: dict | None = None,
) -> dict[str, Any]:
    """Run FixedBurstTriggerScheduler and compute T10-aligned metrics."""
    proprio_raw = episode_data["features_25d"].unsqueeze(0).to(device)
    policy_raw = episode_data["features_9d"].unsqueeze(0).to(device) if use_policy_intent else None
    lang_text = episode_data.get("task_language", "")
    lang_emb = _hash_language_embedding(lang_text)
    language = torch.from_numpy(lang_emb).unsqueeze(0).to(device)

    # Apply normalization if available
    if norm is not None:
        p_mean = torch.from_numpy(norm["proprio_mean"]).to(device)
        p_std = torch.from_numpy(norm["proprio_std"]).to(device).clamp_min(1e-8)
        proprio = (proprio_raw - p_mean) / p_std
        if policy_raw is not None:
            pi_mean = torch.from_numpy(norm["policy_intent_mean"]).to(device)
            pi_std = torch.from_numpy(norm["policy_intent_std"]).to(device).clamp_min(1e-8)
            policy = (policy_raw - pi_mean) / pi_std
        else:
            policy = None
    else:
        proprio = proprio_raw
        policy = policy_raw

    T = proprio.shape[1]
    scheduler = FixedBurstTriggerScheduler(**scheduler_kwargs)

    with torch.no_grad():
        outputs = model(proprio, language, policy_intent=policy, return_sequence=True)

    critical_probs = torch.sigmoid(outputs["critical_window"]).squeeze(0).cpu().numpy()
    release_probs = torch.sigmoid(outputs["release_safe"]).squeeze(0).cpu().numpy()
    grounding_probs = torch.sigmoid(outputs["grounding_confidence"]).squeeze(0).cpu().numpy()
    start_probs = torch.sigmoid(outputs["window_start"]).squeeze(0).cpu().numpy()

    triggered = False
    trigger_step = -1
    for t in range(T):
        decision = scheduler.update(
            critical_probability=float(critical_probs[t]),
            release_safe_probability=float(release_probs[t]),
            grounding_confidence_probability=float(grounding_probs[t]),
            valid=True,
        )
        if decision.trigger_started:
            triggered = True
            trigger_step = t
            break

    # Teacher labels
    has_start = bool(episode_data["targets"]["window_start"].any().item())
    teacher_start_step = -1
    if has_start:
        start_tgt = episode_data["targets"]["window_start"]
        start_mask = episode_data["masks"]["window_start"]
        valid_starts = (start_tgt > 0.5) & start_mask
        if valid_starts.any():
            teacher_start_step = int(torch.nonzero(valid_starts, as_tuple=False)[0, 0])

    # T10 metrics
    feasible_hit = False
    full_T10_containment = False
    start_delay = -1
    early_trigger = False
    late_trigger = False
    negative_any_trigger = False
    release_safe_at_trigger = False

    if triggered:
        # Feasible hit: y_burst_feasible at trigger_step is True and known
        burst_tgt = episode_data["targets"]["burst_feasible"]
        burst_mask = episode_data["masks"]["burst_feasible"]
        if trigger_step < T and burst_mask[trigger_step]:
            feasible_hit = bool(burst_tgt[trigger_step] > 0.5)

        # Full T10 containment: contiguous critical[t : t+10] all known and True
        critical_tgt = episode_data["targets"]["critical_window"]
        critical_mask = episode_data["masks"]["critical_window"]
        end = min(trigger_step + 10, T)
        window_critical = critical_tgt[trigger_step:end]
        window_mask = critical_mask[trigger_step:end]
        full_T10_containment = bool(
            len(window_critical) == 10
            and window_mask.all()
            and (window_critical > 0.5).all()
        )

        # Start delay from teacher window_start
        start_tgt = episode_data["targets"]["window_start"]
        start_mask = episode_data["masks"]["window_start"]
        valid_starts = (start_tgt > 0.5) & start_mask
        if valid_starts.any():
            teacher_start_step = int(torch.nonzero(valid_starts, as_tuple=False)[0, 0])
            start_delay = trigger_step - teacher_start_step
            early_trigger = start_delay < 0
            late_trigger = start_delay > 3 and not feasible_hit
        else:
            negative_any_trigger = True

        # Release-safe at trigger (check for all episodes including negatives)
        release_tgt = episode_data["targets"]["release_safe"]
        release_mask = episode_data["masks"]["release_safe"]
        if trigger_step < T and release_mask[trigger_step]:
            release_safe_at_trigger = bool(release_tgt[trigger_step] > 0.5)
    elif not triggered and not has_start:
        pass  # correctly no trigger on negative episode

    return {
        "triggered": triggered,
        "trigger_step": trigger_step,
        "has_start": has_start,
        "teacher_start_step": teacher_start_step,
        "feasible_hit": feasible_hit,
        "full_T10_containment": full_T10_containment,
        "start_delay": start_delay,
        "early_trigger": early_trigger,
        "late_trigger": late_trigger,
        "negative_any_trigger": negative_any_trigger,
        "release_safe_at_trigger": release_safe_at_trigger,
    }


def run_calibration(
    materialization_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    *,
    device_str: str = "cuda",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, raw = load_model(checkpoint_path, device)
    use_policy_intent = model.config.use_policy_intent

    norm = load_normalization(materialization_root)

    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    cal_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CAL")
    check_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CHECK")

    # Grid: tau_critical, tau_release, tau_ground + burst_feasible as secondary
    tau_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    persistence_configs = [
        {"persistence_window": 3, "persistence_required": 2},
        {"persistence_window": 5, "persistence_required": 3},
        {"persistence_window": 1, "persistence_required": 1},
    ]

    # Phase 1: filter by safety constraints
    MAX_FALSE_RATE = 0.10
    MAX_RELEASE_RATE = 0.02
    feasible_configs = []

    for tau_critical in tau_values:
        for tau_release in [0.3, 0.4, 0.5, 0.6]:
            for tau_ground in [0.3, 0.5, 0.7]:
                for p_cfg in persistence_configs:
                    sk = {
                        "burst_length": 10,
                        "tau_critical": tau_critical,
                        "tau_release": tau_release,
                        "tau_ground": tau_ground,
                        **p_cfg,
                    }
                    results = []
                    for i in range(len(cal_ds)):
                        ep = cal_ds[i]
                        r = evaluate_episode_t10(model, ep, device, use_policy_intent, sk, norm)
                        results.append(r)

                    n_pos = sum(1 for r in results if r["has_start"])
                    n_feasible = sum(1 for r in results if r["feasible_hit"])
                    n_negative = sum(1 for r in results if not r["has_start"])
                    n_false = sum(1 for r in results if r["negative_any_trigger"])
                    n_release = sum(1 for r in results if r["release_safe_at_trigger"])
                    n_full_T10 = sum(1 for r in results if r["full_T10_containment"])

                    false_rate = n_false / max(n_negative, 1)
                    release_rate = n_release / max(len(results), 1)

                    # Safety-constraint filter first
                    if false_rate > MAX_FALSE_RATE or release_rate > MAX_RELEASE_RATE:
                        continue

                    feasible_configs.append({
                        "config": sk,
                        "feasible_rate": n_feasible / max(n_pos, 1),
                        "full_T10_count": n_full_T10,
                        "false_rate": false_rate,
                        "release_rate": release_rate,
                        "n_pos": n_pos,
                        "n_feasible": n_feasible,
                        "n_false": n_false,
                        "n_release": n_release,
                        "n_full_T10": n_full_T10,
                        "n_episodes": len(results),
                        "n_negative": n_negative,
                    })

    if not feasible_configs:
        return {
            "schema": SCHEMA,
            "status": "HOLD_C2G_R9P_NO_FEASIBLE_THRESHOLD",
            "error": f"No config satisfies false<={MAX_FALSE_RATE} and release<={MAX_RELEASE_RATE}",
        }

    # Phase 2: within feasible, maximize feasible_hit, then T10 containment, then minimize false
    best = max(feasible_configs, key=lambda c: (
        c["feasible_rate"],
        c["full_T10_count"],
        -c["false_rate"],
        -c["release_rate"],
        c["config"]["tau_critical"],  # prefer higher thresholds
    ))

    best_config = best["config"]
    best_metrics = {
        "feasible_hit_rate": best["feasible_rate"],
        "feasible_hit_count": best["n_feasible"],
        "full_T10_containment_count": best["n_full_T10"],
        "negative_episode_any_trigger_rate": best["false_rate"],
        "false_trigger_count": best["n_false"],
        "release_safe_emit_rate": best["release_rate"],
        "release_safe_emit_count": best["n_release"],
        "n_cal_episodes": best["n_episodes"],
        "n_positive": best["n_pos"],
        "n_negative": best["n_negative"],
        "feasible_configs_tested": len(feasible_configs),
    }

    if output_root.exists():
        raise FileExistsError(f"calibration output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    # PREVIEW_CHECK evaluation with best config
    sk = {
        "burst_length": 10,
        "tau_critical": best_config["tau_critical"],
        "tau_release": best_config["tau_release"],
        "tau_ground": best_config["tau_ground"],
        "persistence_window": best_config["persistence_window"],
        "persistence_required": best_config["persistence_required"],
    }
    check_results = []
    for i in range(len(check_ds)):
        ep = check_ds[i]
        r = evaluate_episode_t10(model, ep, device, use_policy_intent, sk, norm)
        check_results.append(r)

    n_pos = sum(1 for r in check_results if r["has_start"])
    n_feasible = sum(1 for r in check_results if r["feasible_hit"])
    n_full_T10 = sum(1 for r in check_results if r["full_T10_containment"])
    n_negative = sum(1 for r in check_results if not r["has_start"])
    n_false = sum(1 for r in check_results if r["negative_any_trigger"])
    n_release = sum(1 for r in check_results if r["release_safe_at_trigger"])
    n_early = sum(1 for r in check_results if r["early_trigger"])
    n_late = sum(1 for r in check_results if r["late_trigger"])

    delays = [r["start_delay"] for r in check_results if r["feasible_hit"]]
    median_delay = float(np.median(delays)) if delays else -1.0

    per_suite = {}
    for suite in TARGET_SUITES:
        suite_results = [r for i, r in enumerate(check_results)
                         if check_ds[i]["suite"] == suite]
        suite_pos = sum(1 for r in suite_results if r["has_start"])
        suite_feas = sum(1 for r in suite_results if r["feasible_hit"])
        per_suite[suite] = {
            "n": len(suite_results),
            "positive": suite_pos,
            "feasible_hit": suite_feas,
            "feasible_hit_rate": suite_feas / max(suite_pos, 1),
        }

    check_report = {
        "schema": SCHEMA,
        "total": len(check_results),
        "positive_episodes": n_pos,
        "negative_episodes": n_negative,
        "feasible_hit": n_feasible,
        "feasible_hit_rate": n_feasible / max(n_pos, 1),
        "full_T10_containment": n_full_T10,
        "full_T10_containment_rate": n_full_T10 / max(n_pos, 1),
        "negative_any_trigger": n_false,
        "negative_any_trigger_rate": n_false / max(n_negative, 1),
        "release_safe_emit": n_release,
        "release_safe_emit_rate": n_release / max(len(check_results), 1),
        "early_trigger_count": n_early,
        "late_trigger_count": n_late,
        "median_start_delay": median_delay,
        "per_suite": per_suite,
    }
    write_json(output_root / "preview_check_report.json", check_report)

    # Gate logic
    fr = check_report["feasible_hit_rate"]
    fp = check_report["negative_any_trigger_rate"]
    rs = check_report["release_safe_emit_rate"]
    suite_rates = [per_suite[s]["feasible_hit_rate"] for s in TARGET_SUITES]

    status = "PASS_C2G_R9P_PREVIEW_CHECK"
    if fr < 0.55:
        status = "HOLD_C2G_R9P_LOW_FEASIBLE_HIT"
    elif fp > 0.15:
        status = "HOLD_C2G_R9P_HIGH_FALSE_TRIGGER"
    elif rs > 0.03:
        status = "HOLD_C2G_R9P_HIGH_RELEASE_SAFE_EMIT"
    elif sum(1 for s in suite_rates if s >= 0.50) < 2:
        status = "HOLD_C2G_R9P_INSUFFICIENT_SUITE_COVERAGE"

    config_out = {
        "schema": SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization_sha256": norm["sha256"] if norm else None,
        "thresholds": best_config,
        "calibration_metrics": best_metrics,
        "preview_check": check_report,
    }
    write_json(output_root / "preview_detector_config.json", config_out)

    report = {
        "schema": SCHEMA,
        "status": status,
        "calibration": best_metrics,
        "preview_check": check_report,
        "config": best_config,
    }
    write_json(output_root / "preview_threshold_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate R9P preview detector thresholds")
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint.pt")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_calibration(
        materialization_root=args.materialization_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device_str=args.device,
    )
    print(f"Calibration: {report['status']}")
    check = report["preview_check"]
    print(f"  Feasible hit: {check['feasible_hit_rate']:.3f}  "
          f"T10 containment: {check['full_T10_containment_rate']:.3f}")
    print(f"  False trigger: {check['negative_any_trigger_rate']:.3f}  "
          f"Release-safe emit: {check['release_safe_emit_rate']:.3f}")
    return 0 if "PASS" in report["status"] else 1


if __name__ == "__main__":
    sys.exit(main())
