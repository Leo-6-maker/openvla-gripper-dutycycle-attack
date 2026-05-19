#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml


DEFAULT_CONFIG = "configs/generic_autowindow_detector.yaml"


@dataclass(frozen=True)
class DetectorConfig:
    window_len: int = 10
    min_close_streak: int = 3
    lift_dz: float = 0.02
    eef_lift_dz: float = 0.035
    descent_dz: float = 0.006
    release_open_threshold: float = -0.5
    close_threshold: float = 0.5
    late_offset_ratio: float = 0.55
    min_lift_to_cue_steps: int = 8
    min_carry_steps: int = 4
    near_target_quantile: float = 0.20
    near_target_margin: float = 0.02
    medium_confidence_requires: int = 2
    min_lift_to_done_steps: int = 12
    min_preplace_progress: float = 0.65
    require_late_preplace: bool = True
    allow_near_target_alone: bool = False
    near_target_alone_confidence: str = "low"
    descent_min_delta: float = 0.006
    slowdown_window: int = 5
    fallback_confidence: str = "medium"


def load_config(path: Path) -> tuple[DetectorConfig, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    cfg = DetectorConfig(
        window_len=int(raw.get("window_len", DetectorConfig.window_len)),
        min_close_streak=int(raw.get("min_close_streak", DetectorConfig.min_close_streak)),
        lift_dz=float(raw.get("lift_dz", DetectorConfig.lift_dz)),
        eef_lift_dz=float(raw.get("eef_lift_dz", DetectorConfig.eef_lift_dz)),
        descent_dz=float(raw.get("descent_dz", DetectorConfig.descent_dz)),
        release_open_threshold=float(raw.get("release_open_threshold", DetectorConfig.release_open_threshold)),
        close_threshold=float(raw.get("close_threshold", DetectorConfig.close_threshold)),
        late_offset_ratio=float(raw.get("late_offset_ratio", DetectorConfig.late_offset_ratio)),
        min_lift_to_cue_steps=int(raw.get("min_lift_to_cue_steps", DetectorConfig.min_lift_to_cue_steps)),
        min_carry_steps=int(raw.get("min_carry_steps", DetectorConfig.min_carry_steps)),
        near_target_quantile=float(raw.get("near_target_quantile", DetectorConfig.near_target_quantile)),
        near_target_margin=float(raw.get("near_target_margin", DetectorConfig.near_target_margin)),
        medium_confidence_requires=int(raw.get("medium_confidence_requires", DetectorConfig.medium_confidence_requires)),
        min_lift_to_done_steps=int(raw.get("min_lift_to_done_steps", DetectorConfig.min_lift_to_done_steps)),
        min_preplace_progress=float(raw.get("min_preplace_progress", DetectorConfig.min_preplace_progress)),
        require_late_preplace=bool(raw.get("require_late_preplace", DetectorConfig.require_late_preplace)),
        allow_near_target_alone=bool(raw.get("allow_near_target_alone", DetectorConfig.allow_near_target_alone)),
        near_target_alone_confidence=str(raw.get("near_target_alone_confidence", DetectorConfig.near_target_alone_confidence)),
        descent_min_delta=float(raw.get("descent_min_delta", raw.get("descent_dz", DetectorConfig.descent_min_delta))),
        slowdown_window=int(raw.get("slowdown_window", DetectorConfig.slowdown_window)),
        fallback_confidence=str(raw.get("fallback_confidence", DetectorConfig.fallback_confidence)),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return cfg, digest


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
    return bool(value)


def step_idx(row: dict[str, Any], default: int) -> int:
    value = row.get("step_idx", row.get("step", default))
    try:
        return int(value)
    except Exception:
        return int(default)


def infer_state(run_id: str, manifest: dict[str, Any]) -> str:
    if manifest.get("state") not in (None, ""):
        return str(manifest.get("state"))
    command = str(manifest.get("command", ""))
    for pattern in (r"state(\d+)", r"--state_ids\s+(\d+)"):
        match = re.search(pattern, f"{run_id} {command}")
        if match:
            return match.group(1)
    return ""


def infer_seed(run_id: str, manifest: dict[str, Any]) -> str:
    if manifest.get("seed") not in (None, ""):
        return str(manifest.get("seed"))
    match = re.search(r"seed(\d+)", run_id)
    return match.group(1) if match else ""


def infer_clean_run(run_id: str, manifest: dict[str, Any]) -> bool:
    trigger = str(manifest.get("trigger_name", manifest.get("trigger", ""))).strip().lower()
    if trigger:
        return trigger == "clean"
    condition = str(manifest.get("condition", "")).strip().lower()
    if condition:
        return condition == "clean"
    return False


def episode_success(run_dir: Path) -> bool:
    episodes = read_jsonl(run_dir / "episode_records.jsonl")
    if not episodes:
        return False
    last = episodes[-1]
    return bool(last.get("success", last.get("official_success", False)))


def read_status(run_dir: Path) -> str:
    return str(read_json(run_dir / "progress.json").get("status", ""))


def numeric_value(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def has_signal(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> bool:
    return any(row.get(key) not in (None, "") for row in rows for key in keys)


def eef_z(row: dict[str, Any]) -> float | None:
    return numeric_value(row, ("eef_z_after", "eef_z", "robot0_eef_pos_z", "proxy_lift_carry_eef_z"))


def object_z(row: dict[str, Any]) -> float | None:
    direct = numeric_value(row, ("object_z_after", "target_object_z_after", "object_z"))
    if direct is not None:
        return direct
    candidates: list[float] = []
    for key, value in row.items():
        lowered = key.lower()
        if not (lowered.endswith("_z_after") or lowered.endswith("_z")):
            continue
        if any(token in lowered for token in ("eef", "robot", "gripper", "qpos", "action", "episode_min")):
            continue
        number = to_float(value)
        if number is not None:
            candidates.append(number)
    return candidates[0] if candidates else None


def distance_signal(row: dict[str, Any]) -> float | None:
    direct = numeric_value(row, ("object_target_dist", "target_dist", "receptacle_dist", "receptacle_xy_dist", "target_receptacle_xy_dist"))
    if direct is not None:
        return direct
    candidates: list[float] = []
    for key, value in row.items():
        lowered = key.lower()
        if not (lowered.endswith("_dist") or lowered.endswith("_dxy") or lowered.endswith("_xy_dist")):
            continue
        if any(token in lowered for token in ("eef", "qpos", "action")):
            continue
        number = to_float(value)
        if number is not None:
            candidates.append(number)
    return min(candidates) if candidates else None


def gripper_value(row: dict[str, Any]) -> float | None:
    return numeric_value(row, ("clean_gripper_env", "env_action_gripper", "executed_gripper_env"))


def gripper_closed(row: dict[str, Any], cfg: DetectorConfig) -> bool:
    value = gripper_value(row)
    if value is not None:
        return value > cfg.close_threshold
    return bool(to_bool(row.get("grasp_close_intent")))


def gripper_open(row: dict[str, Any], cfg: DetectorConfig) -> bool:
    value = gripper_value(row)
    return value is not None and value < cfg.release_open_threshold


def first_streak(rows: list[dict[str, Any]], predicate, streak_len: int) -> int | None:
    streak = 0
    first_idx = 0
    for idx, row in enumerate(rows):
        if predicate(row, idx):
            if streak == 0:
                first_idx = idx
            streak += 1
            if streak >= streak_len:
                return step_idx(rows[first_idx], first_idx)
        else:
            streak = 0
    return None


def first_true(rows: list[dict[str, Any]], predicate) -> int | None:
    for idx, row in enumerate(rows):
        if predicate(row, idx):
            return step_idx(row, idx)
    return None


def first_done_step(rows: list[dict[str, Any]], clean_success: bool) -> int | None:
    done = first_true(
        rows,
        lambda row, _: to_bool(row.get("success_done")) is True or to_bool(row.get("success_check")) is True,
    )
    if done is not None:
        return done
    if clean_success and rows:
        return step_idx(rows[-1], len(rows) - 1)
    return None


def base_level(rows: list[dict[str, Any]], fn) -> float | None:
    values = [fn(row) for row in rows[:5]]
    values = [value for value in values if value is not None]
    return median(values) if values else None


def available_signals(rows: list[dict[str, Any]]) -> list[str]:
    signals: list[str] = []
    if any(object_z(row) is not None for row in rows):
        signals.append("object_pose")
    if any(distance_signal(row) is not None for row in rows):
        signals.append("target_distance")
    if has_signal(rows, ("eef_z_after", "eef_z", "robot0_eef_pos_z", "proxy_lift_carry_eef_z")):
        signals.append("eef_pose")
    if has_signal(rows, ("clean_gripper_env", "env_action_gripper", "executed_gripper_env", "grasp_close_intent")):
        signals.append("gripper_signal")
    return signals


def confidence_for(signals: list[str], mode: str, cfg: DetectorConfig) -> str:
    if mode in {"preplace_cue", "release_intent", "near_target", "eef_descent"} and len(signals) >= cfg.medium_confidence_requires:
        return "high" if len(signals) >= 3 else "medium"
    if mode == "near_target_late":
        return cfg.near_target_alone_confidence
    if mode == "late_carry_fallback":
        return cfg.fallback_confidence if len(signals) >= cfg.medium_confidence_requires else "low"
    return "low"


def clamp_window(start: int, rows: list[dict[str, Any]], cfg: DetectorConfig) -> tuple[int, int]:
    max_start = max(0, len(rows) - cfg.window_len)
    start = max(0, min(int(start), max_start))
    return start, start + cfg.window_len - 1


def blank_detection(signal_text: str, reason: str) -> dict[str, Any]:
    return {
        "mechanism_type": "unknown_low_signal",
        "window_detected": False,
        "auto_window_start": "",
        "auto_window_end": "",
        "auto_window_len": "",
        "detector_mode": "failed_no_signal",
        "confidence": "low",
        "grasp_step": "",
        "lift_step": "",
        "carry_start_step": "",
        "near_target_step": "",
        "eef_descent_step": "",
        "preplace_step": "",
        "release_intent_step": "",
        "done_step": "",
        "signals_available": signal_text,
        "failure_reason": reason,
    }


def phase_cues(rows: list[dict[str, Any]], clean_success: bool, cfg: DetectorConfig) -> dict[str, Any]:
    signals = available_signals(rows)
    signal_text = ";".join(signals)
    if not rows:
        return blank_detection(signal_text, "empty_step_records")

    close_step = first_streak(rows, lambda row, _: gripper_closed(row, cfg), cfg.min_close_streak)
    base_obj_z = base_level(rows, object_z)
    base_eef_z = base_level(rows, eef_z)

    def lifted(row: dict[str, Any], idx: int) -> bool:
        cur = step_idx(row, idx)
        if close_step is not None and cur <= close_step:
            return False
        z = object_z(row)
        if z is not None and base_obj_z is not None and z - base_obj_z >= cfg.lift_dz:
            return True
        ez = eef_z(row)
        return bool(ez is not None and base_eef_z is not None and ez - base_eef_z >= cfg.eef_lift_dz)

    lift_step = first_true(rows, lifted)
    carry_start_step = None
    if close_step is not None and lift_step is not None:
        carry_start_step = first_streak(
            rows,
            lambda row, idx: step_idx(row, idx) >= lift_step and gripper_closed(row, cfg),
            cfg.min_carry_steps,
        )

    release_intent_step = None
    if carry_start_step is not None:
        release_intent_step = first_true(
            rows,
            lambda row, idx: step_idx(row, idx) > carry_start_step + cfg.min_lift_to_cue_steps and gripper_open(row, cfg),
        )

    done_step = first_done_step(rows, clean_success)

    def progress_at(step: int | None) -> float | None:
        if step is None or lift_step is None or done_step is None:
            return None
        return (float(step) - float(lift_step)) / max(1.0, float(done_step) - float(lift_step))

    def is_late_enough(step: int | None) -> bool:
        progress = progress_at(step)
        if progress is None:
            return False
        return (not cfg.require_late_preplace) or progress >= cfg.min_preplace_progress

    near_target_first_step = None
    near_target_late_step = None
    dists = [distance_signal(row) for row in rows if distance_signal(row) is not None]
    if dists and lift_step is not None:
        early = dists[: max(1, min(5, len(dists)))]
        approach_delta = median(early) - min(dists)
        if approach_delta > cfg.near_target_margin:
            ordered = sorted(dists)
            quantile_idx = max(0, min(len(ordered) - 1, int(len(ordered) * cfg.near_target_quantile)))
            threshold = ordered[quantile_idx] + cfg.near_target_margin
            near_target_first_step = first_true(
                rows,
                lambda row, idx: step_idx(row, idx) > lift_step + cfg.min_lift_to_cue_steps
                and distance_signal(row) is not None
                and distance_signal(row) <= threshold,
            )
            near_target_late_step = first_true(
                rows,
                lambda row, idx: step_idx(row, idx) > lift_step + cfg.min_lift_to_cue_steps
                and is_late_enough(step_idx(row, idx))
                and distance_signal(row) is not None
                and distance_signal(row) <= threshold,
            )

    eef_descent_step = None
    if lift_step is not None:
        peak_step = None
        peak_z = None
        for idx, row in enumerate(rows):
            cur = step_idx(row, idx)
            if cur <= lift_step:
                continue
            z = eef_z(row)
            if z is None:
                continue
            if peak_z is None or z > peak_z:
                peak_z = z
                peak_step = cur
            if peak_step is not None and cur > peak_step and peak_z is not None and peak_z - z >= cfg.descent_min_delta:
                eef_descent_step = cur
                break

    slowdown_step = None
    if lift_step is not None and cfg.slowdown_window > 1:
        for idx in range(cfg.slowdown_window * 2, len(rows)):
            cur = step_idx(rows[idx], idx)
            if cur <= lift_step + cfg.min_lift_to_cue_steps or not is_late_enough(cur):
                continue
            current_dist = distance_signal(rows[idx])
            prior_dist = distance_signal(rows[idx - cfg.slowdown_window])
            pre_prior_dist = distance_signal(rows[idx - cfg.slowdown_window * 2])
            current_eef = eef_z(rows[idx])
            prior_eef = eef_z(rows[idx - cfg.slowdown_window])
            pre_prior_eef = eef_z(rows[idx - cfg.slowdown_window * 2])
            dist_slow = (
                current_dist is not None
                and prior_dist is not None
                and pre_prior_dist is not None
                and abs(pre_prior_dist - prior_dist) > cfg.near_target_margin
                and abs(prior_dist - current_dist) <= cfg.near_target_margin
            )
            eef_slow = (
                current_eef is not None
                and prior_eef is not None
                and pre_prior_eef is not None
                and abs(pre_prior_eef - prior_eef) > cfg.descent_min_delta
                and abs(prior_eef - current_eef) <= cfg.descent_min_delta
            )
            if dist_slow or eef_slow:
                slowdown_step = cur
                break

    release_late = release_intent_step if is_late_enough(release_intent_step) else None
    descent_late = eef_descent_step if is_late_enough(eef_descent_step) else None
    near_with_support = None
    if near_target_late_step is not None and (descent_late is not None or slowdown_step is not None):
        supported_steps = [step for step in (descent_late, slowdown_step) if step is not None]
        near_with_support = max(near_target_late_step, min(supported_steps))
    near_late_alone = near_target_late_step if cfg.allow_near_target_alone else None

    selected_preplace_step = None
    selected_preplace_reason = ""
    for step, reason in (
        (release_late, "release_intent"),
        (descent_late, "eef_descent"),
        (near_with_support, "near_target_supported"),
        (near_late_alone, "near_target_late"),
    ):
        if step is not None:
            selected_preplace_step = step
            selected_preplace_reason = reason
            break

    rejected_early_near_target = bool(
        near_target_first_step is not None
        and near_target_late_step != near_target_first_step
        and not is_late_enough(near_target_first_step)
    )

    return {
        "signals_available": signal_text,
        "signal_count": len(signals),
        "grasp_step": close_step,
        "lift_step": lift_step,
        "carry_start_step": carry_start_step,
        "near_target_first_step": near_target_first_step,
        "near_target_late_step": near_target_late_step,
        "near_target_step": near_target_late_step,
        "eef_descent_step": eef_descent_step,
        "slowdown_step": slowdown_step,
        "preplace_step": selected_preplace_step,
        "selected_preplace_step": selected_preplace_step,
        "selected_preplace_reason": selected_preplace_reason,
        "rejected_early_near_target": rejected_early_near_target,
        "progress_at_near_target_first": progress_at(near_target_first_step),
        "progress_at_selected_preplace": progress_at(selected_preplace_step),
        "release_intent_step": release_intent_step,
        "done_step": done_step,
    }


def missing_cues(cues: dict[str, Any]) -> str:
    missing = [
        name
        for name in ("grasp_step", "lift_step", "carry_start_step", "preplace_step", "release_intent_step", "done_step")
        if cues.get(name) is None
    ]
    return ";".join(missing)


def legacy_select_window(rows: list[dict[str, Any]], clean_success: bool, cfg: DetectorConfig, cues: dict[str, Any]) -> dict[str, Any]:
    lift_step = cues.get("lift_step")
    carry_start_step = cues.get("carry_start_step")
    release_intent_step = cues.get("release_intent_step")
    near_target_step = cues.get("near_target_step")
    eef_descent_step = cues.get("eef_descent_step")
    signals = cues.get("signals_available", "").split(";") if cues.get("signals_available") else []
    valid_cues = [
        (step, mode)
        for step, mode in (
            (release_intent_step, "release_intent"),
            (near_target_step, "near_target"),
            (eef_descent_step, "eef_descent"),
        )
        if step is not None and lift_step is not None and step - lift_step >= cfg.min_lift_to_cue_steps
    ]

    start = end = ""
    mode = "failed_no_signal"
    failure_reason = ""
    if cues.get("grasp_step") is None:
        failure_reason = "no_stable_grasp"
    elif lift_step is None:
        failure_reason = "no_lift_detected"
    elif carry_start_step is None:
        failure_reason = "no_stable_carry"
    elif valid_cues:
        cue_step, cue_mode = min(valid_cues, key=lambda item: item[0])
        start, end = clamp_window(int(cue_step) - cfg.window_len, rows, cfg)
        mode = "preplace_cue" if cue_mode in {"near_target", "eef_descent"} else cue_mode
    elif clean_success:
        lift_idx = next((idx for idx, row in enumerate(rows) if step_idx(row, idx) >= int(lift_step)), 0)
        tail_len = max(cfg.window_len, len(rows) - lift_idx)
        start_idx = lift_idx + int(tail_len * cfg.late_offset_ratio)
        start, end = clamp_window(start_idx, rows, cfg)
        mode = "late_carry_fallback"
        failure_reason = "no_reliable_preplace_cue"
    else:
        failure_reason = "clean_not_successful_no_preplace_cue"

    detected = start != "" and end != ""
    confidence = confidence_for(signals, mode, cfg) if detected else "low"
    return {
        "selected_window_start_v2": start,
        "selected_window_end_v2": end,
        "selected_mode_v2": mode,
        "confidence_v2": confidence,
        "failure_reason_v2": failure_reason,
    }


def detect_window(rows: list[dict[str, Any]], clean_success: bool, cfg: DetectorConfig) -> dict[str, Any]:
    cues = phase_cues(rows, clean_success, cfg)
    signal_text = str(cues.get("signals_available", ""))
    if not rows:
        return blank_detection(signal_text, "empty_step_records")
    lift_step = cues.get("lift_step")
    carry_start_step = cues.get("carry_start_step")
    release_intent_step = cues.get("release_intent_step")
    preplace_step = cues.get("selected_preplace_step")
    preplace_reason = str(cues.get("selected_preplace_reason", ""))
    done_step = cues.get("done_step")
    signals = signal_text.split(";") if signal_text else []

    start = end = ""
    mode = "failed_no_signal"
    failure_reason = ""
    if cues.get("grasp_step") is None:
        failure_reason = "no_stable_grasp"
    elif lift_step is None:
        failure_reason = "no_lift_detected"
    elif carry_start_step is None:
        failure_reason = "no_stable_carry"
    elif preplace_step is not None and preplace_step - lift_step >= cfg.min_lift_to_cue_steps:
        start, end = clamp_window(int(preplace_step) - cfg.window_len, rows, cfg)
        mode = "release_intent" if preplace_reason == "release_intent" else "near_target_late" if preplace_reason == "near_target_late" else "preplace_cue"
    elif clean_success and done_step is not None and done_step - lift_step >= cfg.min_lift_to_done_steps:
        fallback_ratio = max(cfg.late_offset_ratio, cfg.min_preplace_progress)
        center = int(round(lift_step + fallback_ratio * (done_step - lift_step)))
        start, end = clamp_window(center - cfg.window_len // 2, rows, cfg)
        mode = "late_carry_fallback"
        failure_reason = "no_reliable_preplace_cue"
    else:
        failure_reason = "clean_not_successful_or_no_reliable_lift_to_done_interval"

    detected = start != "" and end != ""
    confidence = confidence_for(signals, mode, cfg) if detected else "low"
    mechanism_type = "trajectory_transfer_candidate" if detected and confidence != "low" else "unknown_low_signal"
    if not clean_success:
        mechanism_type = "clean_unstable"

    return {
        "mechanism_type": mechanism_type,
        "window_detected": bool(detected),
        "auto_window_start": start,
        "auto_window_end": end,
        "auto_window_len": "" if not detected else int(end) - int(start) + 1,
        "detector_mode": mode,
        "confidence": confidence,
        "grasp_step": "" if cues.get("grasp_step") is None else cues.get("grasp_step"),
        "lift_step": "" if lift_step is None else lift_step,
        "carry_start_step": "" if carry_start_step is None else carry_start_step,
        "near_target_step": "" if cues.get("near_target_step") is None else cues.get("near_target_step"),
        "near_target_first_step": "" if cues.get("near_target_first_step") is None else cues.get("near_target_first_step"),
        "near_target_late_step": "" if cues.get("near_target_late_step") is None else cues.get("near_target_late_step"),
        "eef_descent_step": "" if cues.get("eef_descent_step") is None else cues.get("eef_descent_step"),
        "slowdown_step": "" if cues.get("slowdown_step") is None else cues.get("slowdown_step"),
        "preplace_step": "" if preplace_step is None else preplace_step,
        "selected_preplace_step": "" if preplace_step is None else preplace_step,
        "selected_preplace_reason": preplace_reason,
        "rejected_early_near_target": bool(cues.get("rejected_early_near_target")),
        "progress_at_near_target_first": "" if cues.get("progress_at_near_target_first") is None else cues.get("progress_at_near_target_first"),
        "progress_at_selected_preplace": "" if cues.get("progress_at_selected_preplace") is None else cues.get("progress_at_selected_preplace"),
        "release_intent_step": "" if release_intent_step is None else release_intent_step,
        "done_step": "" if done_step is None else done_step,
        "signals_available": signal_text,
        "failure_reason": failure_reason,
    }


def candidate_rows(input_root: Path, cfg: DetectorConfig, config_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        run_id = run_dir.name
        clean_input = infer_clean_run(run_id, manifest)
        if not clean_input:
            continue
        steps = read_jsonl(run_dir / "step_records.jsonl")
        clean_success = episode_success(run_dir)
        detected = detect_window(steps, clean_success, cfg)
        rows.append(
            {
                "run_id": run_id,
                "task_id": str(manifest.get("task_id", "")),
                "state": infer_state(run_id, manifest),
                "seed": infer_seed(run_id, manifest),
                "clean_success": clean_success,
                "status": read_status(run_dir),
                "detector_config_hash": config_hash,
                **detected,
            }
        )
    return rows


def phase_cue_rows(input_root: Path, cfg: DetectorConfig, config_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        run_id = run_dir.name
        if not infer_clean_run(run_id, manifest):
            continue
        steps = read_jsonl(run_dir / "step_records.jsonl")
        clean_success = episode_success(run_dir)
        cues = phase_cues(steps, clean_success, cfg)
        legacy = legacy_select_window(steps, clean_success, cfg, cues)
        selected = detect_window(steps, clean_success, cfg)
        rows.append(
            {
                "run_id": run_id,
                "state": infer_state(run_id, manifest),
                "seed": infer_seed(run_id, manifest),
                "clean_success": clean_success,
                "grasp_step": "" if cues.get("grasp_step") is None else cues.get("grasp_step"),
                "lift_step": "" if cues.get("lift_step") is None else cues.get("lift_step"),
                "carry_start_step": "" if cues.get("carry_start_step") is None else cues.get("carry_start_step"),
                "near_target_step": "" if cues.get("near_target_step") is None else cues.get("near_target_step"),
                "near_target_first_step": "" if cues.get("near_target_first_step") is None else cues.get("near_target_first_step"),
                "near_target_late_step": "" if cues.get("near_target_late_step") is None else cues.get("near_target_late_step"),
                "eef_descent_step": "" if cues.get("eef_descent_step") is None else cues.get("eef_descent_step"),
                "slowdown_step": "" if cues.get("slowdown_step") is None else cues.get("slowdown_step"),
                "release_intent_step": "" if cues.get("release_intent_step") is None else cues.get("release_intent_step"),
                "done_step": "" if cues.get("done_step") is None else cues.get("done_step"),
                "selected_preplace_step": "" if cues.get("selected_preplace_step") is None else cues.get("selected_preplace_step"),
                "selected_preplace_reason": cues.get("selected_preplace_reason", ""),
                "rejected_early_near_target": bool(cues.get("rejected_early_near_target")),
                "progress_at_near_target_first": "" if cues.get("progress_at_near_target_first") is None else cues.get("progress_at_near_target_first"),
                "progress_at_selected_preplace": "" if cues.get("progress_at_selected_preplace") is None else cues.get("progress_at_selected_preplace"),
                "selected_window_start": selected.get("auto_window_start", ""),
                "selected_window_end": selected.get("auto_window_end", ""),
                "detector_mode": selected.get("detector_mode", ""),
                "confidence": selected.get("confidence", ""),
                **legacy,
                "signals_available": cues.get("signals_available", ""),
                "missing_cues": missing_cues(cues),
                "detector_config_hash": config_hash,
                "notes": legacy.get("failure_reason_v2", ""),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]], config_path: Path, config_hash: str) -> None:
    detected = sum(1 for row in rows if row.get("window_detected") is True)
    lines = [
        "# Generic Auto-Window Detector Summary",
        "",
        "No rollout, attack, benchmark, or raw artifact edit was performed.",
        "",
        f"- detector config: `{config_path}`",
        f"- detector config hash: `{config_hash}`",
        f"- clean runs parsed: {len(rows)}",
        f"- windows detected: {detected}",
        "",
        "## Candidates",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('run_id')}`: detected={row.get('window_detected')} "
            f"window={row.get('auto_window_start')}-{row.get('auto_window_end')} "
            f"mode={row.get('detector_mode')} confidence={row.get('confidence')} "
            f"reason={row.get('failure_reason')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic clean-trajectory contact-window detector.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--summary_md", default="")
    parser.add_argument("--phase_cues_csv", default="")
    args = parser.parse_args()

    cfg, config_hash = load_config(Path(args.config))
    rows = candidate_rows(Path(args.input_root), cfg, config_hash)
    if not rows:
        raise SystemExit("generic detector could not parse any clean artifacts")
    write_csv(Path(args.output_csv), rows)
    if args.summary_md:
        write_summary(Path(args.summary_md), rows, Path(args.config), config_hash)
    if args.phase_cues_csv:
        write_csv(Path(args.phase_cues_csv), phase_cue_rows(Path(args.input_root), cfg, config_hash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
