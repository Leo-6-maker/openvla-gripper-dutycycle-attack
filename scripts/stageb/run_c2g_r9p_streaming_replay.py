"""Streaming replay verification for R9P preview detector.

Runs step-by-step inference and verifies batch-offline logits match streaming logits.
Also verifies FSM state machine correctness.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9PEpisodeDataset,
    _hash_language_embedding,
    load_normalization,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    FixedBurstTriggerScheduler,
    SchedulerState,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

SCHEMA = "c2g.r9p.streaming_replay.2026-07-12.v1"
GATE_PASS = "PASS_C2G_R9P_STREAMING_REPLAY"


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[C2gGripperCriticalWindowDetector, dict]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg = raw["model_config"]
    config = C2gDetectorConfig(**cfg)
    model = C2gGripperCriticalWindowDetector(config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model, raw


def streaming_replay_episode(
    model: C2gGripperCriticalWindowDetector,
    episode_data: dict,
    device: torch.device,
    use_policy_intent: bool,
    thresholds: dict,
    norm: dict | None = None,
    atol: float = 1e-5,
) -> dict[str, Any]:
    proprio_raw = episode_data["features_25d"].unsqueeze(0).to(device)
    policy_raw = episode_data["features_9d"].unsqueeze(0).to(device) if use_policy_intent else None
    lang_text = episode_data.get("task_language", "")
    lang_emb = _hash_language_embedding(lang_text)
    language = torch.from_numpy(lang_emb).unsqueeze(0).to(device)

    # Apply normalization
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

    # Batch offline: run whole sequence at once
    with torch.no_grad():
        batch_outputs = model(
            proprio, language,
            policy_intent=policy,
            return_sequence=True,
        )
    batch_logits = {h: batch_outputs[h].squeeze(0).cpu().numpy() for h in R9P_HEAD_NAMES}

    # Streaming step-by-step
    stream_logits = {h: np.zeros(T, dtype=np.float32) for h in R9P_HEAD_NAMES}
    max_errors = {h: 0.0 for h in R9P_HEAD_NAMES}

    scheduler_kwargs = {
        "burst_length": thresholds.get("burst_length", 10),
        "tau_critical": thresholds.get("tau_critical", 0.5),
        "tau_release": thresholds.get("tau_release", 0.5),
        "tau_ground": thresholds.get("tau_ground", 0.5),
        "persistence_window": thresholds.get("persistence_window", 3),
        "persistence_required": thresholds.get("persistence_required", 2),
    }
    scheduler = FixedBurstTriggerScheduler(**scheduler_kwargs)
    fsm_states = []
    triggers = 0

    for t in range(T):
        step_proprio = proprio[:, :t+1, :]
        step_policy = policy[:, :t+1, :] if use_policy_intent and policy is not None else None
        with torch.no_grad():
            step_outputs = model(
                step_proprio, language,
                policy_intent=step_policy,
                return_sequence=True,
            )
        for h in R9P_HEAD_NAMES:
            val = step_outputs[h][0, -1].item()
            stream_logits[h][t] = val
            err = abs(val - batch_logits[h][t])
            max_errors[h] = max(max_errors[h], err)

        decision = scheduler.update(
            critical_probability=float(torch.sigmoid(step_outputs["critical_window"][0, -1]).item()),
            release_safe_probability=float(torch.sigmoid(step_outputs["release_safe"][0, -1]).item()),
            grounding_confidence_probability=float(torch.sigmoid(step_outputs["grounding_confidence"][0, -1]).item()),
            valid=True,
        )
        fsm_states.append(decision.state.value)
        if decision.trigger_started:
            triggers += 1

    # Check equivalence
    equivalence_ok = True
    for h in R9P_HEAD_NAMES:
        if max_errors[h] > atol:
            equivalence_ok = False

    # FSM verification
    fsm_ok = True
    fsm_issues = []
    if triggers > 1:
        fsm_ok = False
        fsm_issues.append(f"multiple_triggers: {triggers}")

    # Check reset
    scheduler.reset()
    assert scheduler.state == SchedulerState.IDLE, "reset failed"

    return {
        "equivalence_ok": equivalence_ok,
        "max_errors": {h: float(e) for h, e in max_errors.items()},
        "fsm_ok": fsm_ok,
        "fsm_issues": fsm_issues,
        "triggers": triggers,
        "T": T,
    }


def run_streaming_replay(
    materialization_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    *,
    detector_config_path: Path | None = None,
    max_episodes: int = 0,
    device_str: str = "cuda",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, raw = _load_model(checkpoint_path, device)
    use_policy_intent = model.config.use_policy_intent

    # Load frozen thresholds — fail-closed if config missing or invalid
    if not detector_config_path.exists():
        raise FileNotFoundError(f"detector config not found: {detector_config_path}")
    config = read_json(detector_config_path)

    # Verify checkpoint SHA binding
    config_ckpt_sha = config.get("checkpoint_sha256", "")
    actual_ckpt_sha = sha256_file(checkpoint_path)
    if not config_ckpt_sha:
        raise ValueError("detector config missing checkpoint_sha256 field")
    if config_ckpt_sha != actual_ckpt_sha:
        raise ValueError(
            f"detector config checkpoint_sha256 mismatch: "
            f"config={config_ckpt_sha}, actual={actual_ckpt_sha}"
        )

    # Verify normalization SHA binding
    norm = load_normalization(materialization_root)
    if norm is None:
        raise FileNotFoundError("normalization.json not found in materialization root")
    config_norm_sha = config.get("normalization_sha256", "")
    if config_norm_sha and config_norm_sha != norm["sha256"]:
        raise ValueError(
            f"detector config normalization_sha256 mismatch: "
            f"config={config_norm_sha}, actual={norm['sha256']}"
        )

    t = config.get("thresholds", {})
    if not t:
        raise ValueError("detector config has no thresholds")
    required_fields = ["burst_length", "tau_critical", "tau_release", "tau_ground",
                       "persistence_window", "persistence_required"]
    for f in required_fields:
        if f not in t:
            raise ValueError(f"detector config thresholds missing '{f}'")
        val = float(t[f])
        if not np.isfinite(val):
            raise ValueError(f"threshold '{f}' is non-finite: {val}")
    thresholds = {
        "burst_length": int(t["burst_length"]),
        "tau_critical": float(t["tau_critical"]),
        "tau_release": float(t["tau_release"]),
        "tau_ground": float(t["tau_ground"]),
        "persistence_window": int(t["persistence_window"]),
        "persistence_required": int(t["persistence_required"]),
    }

    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    ds = R9PEpisodeDataset(index_rows, materialization_root)
    n_episodes = len(ds) if max_episodes <= 0 else min(max_episodes, len(ds))

    results = []
    all_equiv = True
    all_fsm = True

    for i in range(n_episodes):
        ep = ds[i]
        r = streaming_replay_episode(model, ep, device, use_policy_intent, thresholds, norm)
        results.append(r)
        if not r["equivalence_ok"]:
            all_equiv = False
        if not r["fsm_ok"]:
            all_fsm = False

    output_root.mkdir(parents=True, exist_ok=True)
    status = GATE_PASS if (all_equiv and all_fsm) else f"HOLD_{GATE_PASS}"

    report = {
        "schema": SCHEMA,
        "status": status,
        "episodes_tested": n_episodes,
        "batch_stream_equivalence": all_equiv,
        "fsm_verification": all_fsm,
        "max_error_summary": {
            h: float(max(r["max_errors"][h] for r in results))
            for h in R9P_HEAD_NAMES
        },
        "total_triggers": sum(r["triggers"] for r in results),
        "multi_trigger_count": sum(1 for r in results if r["triggers"] > 1),
        "issues": [
            {"episode": i, "fsm_issues": r["fsm_issues"]}
            for i, r in enumerate(results) if r["fsm_issues"]
        ][:20],
    }
    write_json(output_root / "streaming_replay_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R9P streaming replay verification")
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--detector-config", type=Path, required=True,
                        help="Path to preview_detector_config.json with frozen thresholds")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_streaming_replay(
        materialization_root=args.materialization_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        detector_config_path=args.detector_config,
        max_episodes=args.max_episodes,
        device_str=args.device,
    )
    print(f"Streaming replay: {report['status']}")
    print(f"  Batch==Stream: {report['batch_stream_equivalence']}")
    print(f"  FSM verified: {report['fsm_verification']}")
    return 0 if report["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
