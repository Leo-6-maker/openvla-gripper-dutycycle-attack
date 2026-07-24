"""Select one R9Q detector checkpoint from CAL only.

Each checkpoint is run over CAL once. Threshold and persistence search then
operates on cached causal scores, avoiding repeated model forwards while
preserving the calibration evaluator's unknown-safe semantics.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from scripts.stageb.calibrate_c2g_r9p_preview_thresholds import load_model
from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9PEpisodeDataset,
    _hash_language_embedding,
    load_normalization,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    FixedBurstTriggerScheduler,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_jsonl,
    sha256_file,
    write_json,
)


SCHEMA = "c2g.r9q.calibration_selection.2026-07-13.v1"
CHECKPOINT_RE = re.compile(r"b2_seed(?P<seed>\d+)[/\\]epoch_(?P<epoch>\d+)\.pt$")


def _label_state(episode: dict[str, Any]) -> dict[str, Any]:
    length = int(episode["features_25d"].shape[0])
    valid = episode.get("valid_mask")
    valid = torch.ones(length, dtype=torch.bool) if valid is None else valid.bool().cpu()
    if valid.numel() != length:
        raise ValueError("valid_mask length does not match episode length")

    targets = {h: episode["targets"][h].bool().cpu().numpy() for h in R9P_HEAD_NAMES}
    masks = {h: episode["masks"][h].bool().cpu().numpy() for h in R9P_HEAD_NAMES}
    valid_np = valid.numpy()
    known = {h: masks[h] & valid_np for h in R9P_HEAD_NAMES}
    valid_start = targets["window_start"] & known["window_start"]
    valid_burst = targets["burst_feasible"] & known["burst_feasible"]
    has_start = bool(valid_start.any() or valid_burst.any())
    fully_known = bool(
        valid_np.any()
        and all(bool(known[h][valid_np].all()) for h in R9P_HEAD_NAMES)
    )
    return {
        "valid": valid_np,
        "known": known,
        "targets": targets,
        "has_start": has_start,
        "trigger_negative": bool(fully_known and not has_start),
        "fully_known": fully_known,
        "teacher_start_step": int(np.flatnonzero(valid_start)[0]) if valid_start.any() else -1,
    }


def _score_episode(model, episode: dict[str, Any], device: torch.device, norm: dict) -> dict[str, np.ndarray]:
    proprio_raw = episode["features_25d"].unsqueeze(0).to(device)
    policy_raw = episode["features_9d"].unsqueeze(0).to(device) if model.config.use_policy_intent else None
    language = torch.from_numpy(_hash_language_embedding(episode.get("task_language", "")))
    language = language.unsqueeze(0).to(device)

    p_mean = torch.from_numpy(norm["proprio_mean"]).to(device)
    p_std = torch.from_numpy(norm["proprio_std"]).to(device).clamp_min(1e-8)
    proprio = (proprio_raw - p_mean) / p_std
    if policy_raw is not None:
        pi_mean = torch.from_numpy(norm["policy_intent_mean"]).to(device)
        pi_std = torch.from_numpy(norm["policy_intent_std"]).to(device).clamp_min(1e-8)
        policy = (policy_raw - pi_mean) / pi_std
    else:
        policy = None

    with torch.no_grad():
        outputs = model(proprio, language, policy_intent=policy, return_sequence=True)
    return {
        head: torch.sigmoid(outputs[head]).squeeze(0).detach().cpu().numpy()
        for head in R9P_HEAD_NAMES
    }


def _evaluate_cached(scores: dict[str, np.ndarray], labels: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    scheduler = FixedBurstTriggerScheduler(**config)
    valid = labels["valid"]
    known = labels["known"]
    targets = labels["targets"]
    length = len(valid)
    triggered = False
    trigger_step = -1
    for t in range(length):
        decision = scheduler.update(
            critical_probability=float(scores["critical_window"][t]),
            release_safe_probability=float(scores["release_safe"][t]),
            grounding_confidence_probability=float(scores["grounding_confidence"][t]),
            valid=bool(valid[t] and known["critical_window"][t]
                       and known["release_safe"][t]
                       and known["grounding_confidence"][t]),
        )
        if decision.trigger_started:
            triggered = True
            trigger_step = t
            break

    feasible_hit = bool(
        triggered
        and trigger_step < length
        and known["burst_feasible"][trigger_step]
        and targets["burst_feasible"][trigger_step]
    )
    end = min(trigger_step + 10, length) if triggered else 0
    full_t10 = bool(
        triggered
        and end - trigger_step == 10
        and known["critical_window"][trigger_step:end].all()
        and targets["critical_window"][trigger_step:end].all()
    )
    start_step = labels["teacher_start_step"]
    start_delay = trigger_step - start_step if triggered and start_step >= 0 else -1
    release_safe = bool(
        triggered
        and known["release_safe"][trigger_step]
        and targets["release_safe"][trigger_step]
    )
    negative_any = bool(triggered and labels["trigger_negative"])
    return {
        "triggered": triggered,
        "trigger_step": trigger_step,
        "has_start": labels["has_start"],
        "fully_known": labels["fully_known"],
        "trigger_negative": labels["trigger_negative"],
        "feasible_hit": feasible_hit,
        "full_T10_containment": full_t10,
        "start_delay": start_delay,
        "early_trigger": bool(start_step >= 0 and start_delay < 0),
        "late_trigger": bool(start_step >= 0 and start_delay > 3 and not feasible_hit),
        "negative_any_trigger": negative_any,
        "release_safe_at_trigger": release_safe,
    }


def _metrics(results: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in results if r["has_start"]]
    negatives = [r for r in results if r["trigger_negative"]]
    feasible = [r for r in results if r["feasible_hit"]]
    delays = [r["start_delay"] for r in feasible]
    per_suite = {}
    for suite in TARGET_SUITES:
        rows = [r for r, ep in zip(results, episodes) if ep["suite"] == suite]
        suite_pos = sum(r["has_start"] for r in rows)
        suite_neg = sum(r["trigger_negative"] for r in rows)
        suite_feas = sum(r["feasible_hit"] for r in rows)
        suite_false = sum(r["negative_any_trigger"] for r in rows)
        per_suite[suite] = {
            "n": len(rows),
            "positive": suite_pos,
            "fully_known_negative": suite_neg,
            "feasible_hit": suite_feas,
            "feasible_hit_rate": suite_feas / max(suite_pos, 1),
            "negative_any_trigger": suite_false,
            "negative_any_trigger_rate": suite_false / max(suite_neg, 1),
        }
    return {
        "n_episodes": len(results),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_feasible": len(feasible),
        "feasible_hit_rate": len(feasible) / max(len(positives), 1),
        "full_T10_count": sum(r["full_T10_containment"] for r in results),
        "full_T10_rate": sum(r["full_T10_containment"] for r in results) / max(len(positives), 1),
        "false_trigger_count": sum(r["negative_any_trigger"] for r in results),
        "false_trigger_rate": sum(r["negative_any_trigger"] for r in results) / max(len(negatives), 1),
        "release_safe_count": sum(r["release_safe_at_trigger"] for r in results),
        "release_safe_rate": sum(r["release_safe_at_trigger"] for r in results) / max(len(results), 1),
        "median_start_delay": float(np.median(delays)) if delays else -1.0,
        "per_suite": per_suite,
    }


def _checkpoint_info(path: Path) -> tuple[int, int]:
    match = CHECKPOINT_RE.search(path.as_posix())
    if not match:
        raise ValueError(f"unexpected checkpoint path: {path}")
    return int(match.group("seed")), int(match.group("epoch"))


def select(materialization_root: Path, models_root: Path, output_root: Path, device_str: str) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    norm = load_normalization(materialization_root)
    if norm is None:
        raise FileNotFoundError("normalization.json")
    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    episodes = [R9PEpisodeDataset(index_rows, materialization_root, split_filter="CAL")[i]
                for i in range(sum(r["preview_split"] == "CAL" for r in index_rows))]
    if len(episodes) != 116:
        raise ValueError(f"expected 116 CAL episodes, got {len(episodes)}")

    checkpoints = sorted(models_root.glob("b2_seed*/epoch_*.pt"), key=lambda p: _checkpoint_info(p))
    if len(checkpoints) != 90:
        raise ValueError(f"expected 90 B2 checkpoints, got {len(checkpoints)}")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    grid = {
        "tau_critical": [0.3, 0.4, 0.5, 0.6, 0.7],
        "tau_release": [0.3, 0.4, 0.5, 0.6],
        "tau_ground": [0.3, 0.5, 0.7],
        "persistence": [
            {"persistence_window": 1, "persistence_required": 1},
            {"persistence_window": 3, "persistence_required": 2},
            {"persistence_window": 5, "persistence_required": 3},
        ],
    }
    candidates = []
    for checkpoint in checkpoints:
        seed, epoch = _checkpoint_info(checkpoint)
        model, raw = load_model(checkpoint, device)
        scores = [_score_episode(model, episode, device, norm) for episode in episodes]
        labels = [_label_state(episode) for episode in episodes]
        del model, raw
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for tau_critical in grid["tau_critical"]:
            for tau_release in grid["tau_release"]:
                for tau_ground in grid["tau_ground"]:
                    for persistence in grid["persistence"]:
                        config = {
                            "burst_length": 10,
                            "tau_critical": tau_critical,
                            "tau_release": tau_release,
                            "tau_ground": tau_ground,
                            **persistence,
                        }
                        results = [_evaluate_cached(score, label, config)
                                   for score, label in zip(scores, labels)]
                        metrics = _metrics(results, episodes)
                        candidate = {
                            "seed": seed,
                            "epoch": epoch,
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": sha256_file(checkpoint),
                            "config": config,
                            **metrics,
                            "safety_pass": metrics["false_trigger_rate"] <= 0.10
                            and metrics["release_safe_rate"] <= 0.02,
                        }
                        if candidate["safety_pass"]:
                            candidates.append(candidate)

    if not candidates:
        report = {"schema": SCHEMA, "status": "HOLD_C2G_R9Q_NO_SAFE_CAL_CONFIG", "candidate_count": 0}
        output_root.mkdir(parents=True)
        write_json(output_root / "cal_selection_report.json", report)
        return report

    best = max(candidates, key=lambda c: (
        c["feasible_hit_rate"],
        c["full_T10_rate"],
        -c["median_start_delay"] if c["median_start_delay"] >= 0 else float("-inf"),
        -c["false_trigger_rate"],
        -c["release_safe_rate"],
        -c["epoch"],
        -c["seed"],
        c["checkpoint_sha256"],
    ))
    output_root.mkdir(parents=True)
    with (output_root / "cal_selection_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    selected_config = {
        "schema": SCHEMA,
        "mode": "CALIBRATE_ONLY",
        "checkpoint_path": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "normalization_sha256": norm["sha256"],
        "thresholds": best["config"],
        "calibration_metrics": best,
        "check_consumption_count": 0,
    }
    write_json(output_root / "selected_preview_detector_config.json", selected_config)
    write_json(output_root / "preview_detector_config.json", selected_config)
    report = {
        "schema": SCHEMA,
        "status": "PASS_C2G_R9Q_CALIBRATION_SELECTED",
        "calibration_only": True,
        "check_reads": 0,
        "checkpoint_count": len(checkpoints),
        "cal_rows": len(episodes),
        "candidate_count": len(candidates),
        "selected": best,
        "grid": grid,
        "normalization_sha256": norm["sha256"],
    }
    write_json(output_root / "cal_selection_report.json", report)
    write_json(output_root / "preview_threshold_report.json", {
        "schema": SCHEMA,
        "status": report["status"],
        "calibration": best,
        "config": best["config"],
        "preview_check": None,
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    report = select(args.materialization_root, args.models_root, args.output_root, args.device)
    print(json.dumps({"status": report["status"], "selected": report.get("selected")}, sort_keys=True))
    return 0 if report["status"] == "PASS_C2G_R9Q_CALIBRATION_SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
