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
    read_jsonl,
    sha256_file,
    write_json,
)

SCHEMA = "c2g.r9p.streaming_replay.2026-07-12.v1"
GATE_PASS = "PASS_C2G_R9P_STREAMING_REPLAY"


def _load_model(checkpoint_path: Path, device: torch.device) -> C2gGripperCriticalWindowDetector:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg = raw["model_config"]
    config = C2gDetectorConfig(**cfg)
    model = C2gGripperCriticalWindowDetector(config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model


def streaming_replay_episode(
    model: C2gGripperCriticalWindowDetector,
    episode_data: dict,
    device: torch.device,
    use_policy_intent: bool,
    thresholds: dict,
    atol: float = 1e-5,
) -> dict[str, Any]:
    proprio = episode_data["features_25d"].unsqueeze(0).to(device)
    policy = episode_data["features_9d"].unsqueeze(0).to(device) if use_policy_intent else None
    lang_text = episode_data.get("task_language", "")
    lang_emb = _hash_language_embedding(lang_text)
    language = torch.from_numpy(lang_emb).unsqueeze(0).to(device)
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
    max_episodes: int = 0,
    device_str: str = "cuda",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    use_policy_intent = model.config.use_policy_intent

    thresholds = {
        "burst_length": 10,
        "tau_critical": 0.5,
        "tau_release": 0.5,
        "tau_ground": 0.5,
        "persistence_window": 3,
        "persistence_required": 2,
    }

    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    ds = R9PEpisodeDataset(index_rows, materialization_root)
    n_episodes = len(ds) if max_episodes <= 0 else min(max_episodes, len(ds))

    results = []
    all_equiv = True
    all_fsm = True

    for i in range(n_episodes):
        ep = ds[i]
        r = streaming_replay_episode(model, ep, device, use_policy_intent, thresholds)
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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_streaming_replay(
        materialization_root=args.materialization_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        max_episodes=args.max_episodes,
        device_str=args.device,
    )
    print(f"Streaming replay: {report['status']}")
    print(f"  Batch==Stream: {report['batch_stream_equivalence']}")
    print(f"  FSM verified: {report['fsm_verification']}")
    return 0 if report["status"] == GATE_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
