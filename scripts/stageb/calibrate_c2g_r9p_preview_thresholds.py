"""Calibrate R9P preview detector thresholds on PREVIEW_CAL and evaluate on PREVIEW_CHECK.

Grid search over tau_start, tau_critical, tau_release, tau_ground with persistence
options (1-of-1, 2-of-3, 3-of-5). Lexicographic selection per the R9P spec.
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
from torch.utils.data import DataLoader

from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9PEpisodeDataset,
    _hash_language_embedding,
    collate_episodes,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    FixedBurstTriggerScheduler,
    SchedulerState,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_jsonl,
    write_json,
)

SCHEMA = "c2g.r9p.preview_calibration.2026-07-12.v1"


def load_model(checkpoint_path: Path, device: torch.device) -> C2gGripperCriticalWindowDetector:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg = raw["model_config"]
    config = C2gDetectorConfig(**cfg)
    model = C2gGripperCriticalWindowDetector(config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model


def evaluate_episode_with_thresholds(
    model: C2gGripperCriticalWindowDetector,
    episode_data: dict,
    device: torch.device,
    use_policy_intent: bool,
    scheduler_kwargs: dict,
) -> dict[str, Any]:
    """Run FixedBurstTriggerScheduler over one episode, return trigger metrics."""
    proprio = episode_data["features_25d"].unsqueeze(0).to(device)
    policy = episode_data["features_9d"].unsqueeze(0).to(device) if use_policy_intent else None
    lang_text = episode_data.get("task_language", "")
    lang_emb = _hash_language_embedding(lang_text)
    language = torch.from_numpy(lang_emb).unsqueeze(0).to(device)

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

    # Check if episode is positive (has window_start label)
    has_start = bool(episode_data["targets"]["window_start"].any().item())
    # Check if release was safe at trigger
    release_safe_at_trigger = False
    if triggered and trigger_step < T:
        release_safe_at_trigger = bool(
            episode_data["targets"]["release_safe"][trigger_step].item() > 0.5
        )

    return {
        "triggered": triggered,
        "trigger_step": trigger_step,
        "has_start": has_start,
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
    model = load_model(checkpoint_path, device)
    use_policy_intent = model.config.use_policy_intent

    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    cal_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CAL")
    check_ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CHECK")

    # Grid
    tau_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    persistence_configs = [
        {"persistence_window": 1, "persistence_required": 1},
        {"persistence_window": 3, "persistence_required": 2},
        {"persistence_window": 5, "persistence_required": 3},
    ]

    best_config = None
    best_metrics = None
    # Lexicographic: recall, then -false_trigger, then -release_safe_trigger
    best_key = (-999, 999, 999)

    for tau_critical in tau_values:
        for tau_release in [0.3, 0.4, 0.5, 0.6]:
            for tau_ground in [0.3, 0.5, 0.7]:
                for p_cfg in persistence_configs:
                    scheduler_kwargs = {
                        "burst_length": 10,
                        "tau_critical": tau_critical,
                        "tau_release": tau_release,
                        "tau_ground": tau_ground,
                        **p_cfg,
                    }
                    results = []
                    for i in range(len(cal_ds)):
                        ep = cal_ds[i]
                        r = evaluate_episode_with_thresholds(
                            model, ep, device, use_policy_intent, scheduler_kwargs)
                        results.append(r)

                    n_pos = sum(1 for r in results if r["has_start"])
                    n_triggered_pos = sum(1 for r in results if r["has_start"] and r["triggered"])
                    n_false_trigger = sum(1 for r in results if not r["has_start"] and r["triggered"])
                    n_release_trigger = sum(1 for r in results if r["release_safe_at_trigger"])
                    recall = n_triggered_pos / max(n_pos, 1)

                    key = (round(recall, 3), -n_false_trigger, -n_release_trigger)
                    if key > best_key:
                        best_key = key
                        best_config = {
                            "tau_start": tau_critical,  # same threshold grid
                            "tau_critical": tau_critical,
                            "tau_release": tau_release,
                            "tau_ground": tau_ground,
                            **p_cfg,
                        }
                        best_metrics = {
                            "positive_recall": recall,
                            "false_triggers": n_false_trigger,
                            "release_safe_triggers": n_release_trigger,
                            "n_cal_episodes": len(results),
                            "n_positive": n_pos,
                        }

    output_root.mkdir(parents=True, exist_ok=True)
    config_out = {
        "schema": SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "thresholds": best_config,
        "calibration_metrics": best_metrics,
    }
    write_json(output_root / "preview_detector_config.json", config_out)

    # PREVIEW_CHECK evaluation
    check_results = []
    scheduler_kwargs = {
        "burst_length": 10,
        "tau_critical": best_config["tau_critical"],
        "tau_release": best_config["tau_release"],
        "tau_ground": best_config["tau_ground"],
        "persistence_window": best_config["persistence_window"],
        "persistence_required": best_config["persistence_required"],
    }
    for i in range(len(check_ds)):
        ep = check_ds[i]
        r = evaluate_episode_with_thresholds(
            model, ep, device, use_policy_intent, scheduler_kwargs)
        check_results.append(r)

    n_pos_check = sum(1 for r in check_results if r["has_start"])
    n_trig_pos_check = sum(1 for r in check_results if r["has_start"] and r["triggered"])
    n_false_check = sum(1 for r in check_results if not r["has_start"] and r["triggered"])
    n_release_check = sum(1 for r in check_results if r["release_safe_at_trigger"])

    check_report = {
        "schema": SCHEMA,
        "total": len(check_results),
        "positive_episodes": n_pos_check,
        "triggered_positive": n_trig_pos_check,
        "positive_recall": n_trig_pos_check / max(n_pos_check, 1),
        "false_triggers": n_false_check,
        "false_trigger_rate": n_false_check / max(len(check_results) - n_pos_check, 1),
        "release_safe_triggers": n_release_check,
        "release_safe_trigger_rate": n_release_check / max(len(check_results), 1),
    }
    write_json(output_root / "preview_check_report.json", check_report)

    status = "PASS_C2G_R9P_PREVIEW_CHECK"
    if check_report["positive_recall"] < 0.55:
        status = "HOLD_C2G_R9P_LOW_RECALL"
    elif check_report["false_trigger_rate"] > 0.15:
        status = "HOLD_C2G_R9P_HIGH_FALSE_TRIGGER"

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
    print(f"  CHECK recall: {report['preview_check']['positive_recall']:.3f}")
    print(f"  CHECK false trigger rate: {report['preview_check']['false_trigger_rate']:.3f}")
    return 0 if "PASS" in report["status"] else 1


if __name__ == "__main__":
    sys.exit(main())
