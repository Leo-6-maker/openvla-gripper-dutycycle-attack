"""Offline latency smoke for R9P preview detector — no runtime integration.

Measures forward-pass latency on full episode sequences using PREVIEW_CHECK NPZ files.
This is a pure model benchmarking step; it does NOT connect to OpenVLA, does NOT
run step-by-step FSM, does NOT modify actions, and does NOT verify runtime behavior.
"""
from __future__ import annotations

import argparse
import sys
import time
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
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_jsonl,
    write_json,
)

SCHEMA = "c2g.r9p.offline_latency_smoke.2026-07-12.v1"
GATE_PASS = "PASS_C2G_R9P_OFFLINE_LATENCY_SMOKE"


def _load_model(checkpoint_path: Path, device: torch.device) -> C2gGripperCriticalWindowDetector:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg = raw["model_config"]
    config = C2gDetectorConfig(**cfg)
    model = C2gGripperCriticalWindowDetector(config).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model


def measure_latency(
    model: C2gGripperCriticalWindowDetector,
    proprio: torch.Tensor,
    language: torch.Tensor,
    policy_intent: torch.Tensor | None,
    n_warmup: int = 5,
    n_measure: int = 50,
) -> dict[str, float]:
    use_cuda = proprio.device.type == "cuda"

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(proprio, language, policy_intent=policy_intent, return_sequence=False)

    times = []
    for _ in range(n_measure):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = model(proprio, language, policy_intent=policy_intent, return_sequence=False)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(proprio, language, policy_intent=policy_intent, return_sequence=False)
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "n_samples": n_measure,
    }


def run_latency_smoke(
    materialization_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    *,
    device_str: str = "cuda",
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    use_policy_intent = model.config.use_policy_intent

    norm = load_normalization(materialization_root)

    index_rows = read_jsonl(materialization_root / "dataset_index.jsonl")
    ds = R9PEpisodeDataset(index_rows, materialization_root, split_filter="CHECK")

    latencies = []
    n_test = min(len(ds), 10)
    for i in range(n_test):
        ep = ds[i]
        proprio_raw = ep["features_25d"].unsqueeze(0).to(device)
        policy_raw = ep["features_9d"].unsqueeze(0).to(device) if use_policy_intent else None

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

        lang_text = ep.get("task_language", "")
        lang_emb = _hash_language_embedding(lang_text)
        language = torch.from_numpy(lang_emb).unsqueeze(0).to(device)

        latency = measure_latency(model, proprio, language, policy)
        latencies.append(latency)

    avg_latency = float(np.mean([l["mean_ms"] for l in latencies]))
    max_latency = float(np.max([l["p99_ms"] for l in latencies]))

    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": SCHEMA,
        "status": GATE_PASS,
        "mode": "OFFLINE_LATENCY_SMOKE",
        "episodes_tested": n_test,
        "device": str(device),
        "latency": {
            "average_ms": avg_latency,
            "max_p99_ms": max_latency,
            "per_episode": latencies,
        },
        "boundaries": {
            "openvla_loaded": False,
            "runtime_connected": False,
            "fsm_verified": False,
            "actions_modified": False,
            "attack_delivered": False,
        },
    }
    write_json(output_root / "offline_latency_smoke_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R9P offline latency smoke")
    parser.add_argument("--materialization-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_latency_smoke(
        materialization_root=args.materialization_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device_str=args.device,
    )
    print(f"Latency smoke: {report['status']}")
    print(f"  Avg: {report['latency']['average_ms']:.2f}ms  "
          f"P99: {report['latency']['max_p99_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
