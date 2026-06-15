#!/usr/bin/env python3
"""GPU45 fixed-frame qualification for M3 Layer3 infrastructure.

This runner is deliberately not an attack runner.  It performs only fixed-frame
model-load, clean official generation repeatability, and a single clean-frame
surrogate-gradient repeatability check.  It does not run PGD, RAND21,
SHUFFLED_GRAD21, panel capture, LIBERO rollout, or any final frozen panel frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker  # noqa: E402
from gripper_attack.m3_controls import tensor_sha256  # noqa: E402
from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (  # noqa: E402
    load_config,
    load_frozen_input,
    load_model,
    model_fingerprint,
    official_decode,
    preprocess_raw_image,
    sha256_file,
    write_artifact_hash_manifest,
    write_csv,
    write_json,
)


DEFAULT_GPU_UUIDS = [
    "GPU-d0a54f5d-938c-a148-fff9-c135201e3f61",
    "GPU-9794d733-042f-46a2-fc86-5a3fe32a158a",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(args: list[str], *, cwd: Path | None = None, check: bool = False) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    out = proc.stdout.strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{out}")
    return out


def git_value(args: list[str]) -> str:
    return run_command(["git", *args], cwd=REPO_ROOT)


def git_dirty_status() -> str:
    status = git_value(["status", "--porcelain"])
    return "CLEAN" if not status else "DIRTY:" + status.replace("\n", "\\n")


def git_branch_name() -> str:
    branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    return branch or "UNKNOWN"


def nvidia_smi_query() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ]
    )


def nvidia_smi_compute_apps() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )


def physical_gpu_uuid_map(indices: list[int]) -> dict[int, str]:
    if not indices:
        raise ValueError("expected at least one physical GPU index")
    raw = run_command(
        [
            "nvidia-smi",
            f"--id={','.join(str(i) for i in indices)}",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        check=True,
    )
    mapping: dict[int, str] = {}
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            mapping[int(parts[0])] = parts[1]
    missing = [idx for idx in indices if idx not in mapping]
    if missing:
        raise RuntimeError(f"nvidia-smi did not return UUIDs for physical GPUs: {missing}")
    return mapping


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def tensor_l1_l2_linf(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().float().flatten()
    return {
        "l1": float(flat.abs().sum().cpu()),
        "l2": float(torch.linalg.vector_norm(flat, ord=2).cpu()),
        "linf": float(flat.abs().max().cpu()),
    }


def target_margin(stats: Mapping[str, Any]) -> float:
    for key in (
        "target_minus_competitor_logsumexp_margin",
        "target_minus_best_competitor_margin",
        "target_objective_margin",
    ):
        if key in stats:
            return float(stats[key])
    raise KeyError(f"target margin field missing from stats keys: {sorted(stats.keys())}")


def gradient_repeat(
    *,
    adapter: TokenPrefixPGDAttacker,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    action_dim: int,
    target_token_id: int,
    margin: float,
) -> dict[str, Any]:
    pixel = pixel_values.detach().clone().requires_grad_(True)
    prefix = adapter._generate_action_prefix_tokens(input_ids, pixel, prefix_len=int(action_dim) - 1)
    loss, stats = adapter._generated_prefix_target_token_loss_and_stats(
        input_ids,
        prefix,
        pixel,
        target_token_id=int(target_token_id),
        margin=float(margin),
    )
    loss.backward()
    grad = pixel.grad
    if grad is None:
        raise RuntimeError("gradient repeatability check produced no pixel gradient")
    finite = bool(torch.isfinite(grad).all().detach().cpu())
    norms = tensor_l1_l2_linf(grad)
    return {
        "loss": float(loss.detach().cpu()),
        "target_margin": target_margin(stats),
        "target_token_score": float(stats["target_token_score"]),
        "best_competitor_token_id": int(stats["best_competitor_token_id"]),
        "best_competitor_score": float(stats["best_competitor_score"]),
        "generated_arm_prefix": [int(x) for x in prefix.detach().cpu().tolist()],
        "gradient_finite": finite,
        "gradient_sha256": tensor_sha256(grad),
        "gradient_l1": norms["l1"],
        "gradient_l2": norms["l2"],
        "gradient_linf": norms["linf"],
    }


def write_phase_marker(path: Path, rows: list[dict[str, Any]], phase: str) -> None:
    rows.append(
        {
            "phase_index": len(rows),
            "phase": phase,
            "timestamp_utc": utc_now(),
            "monotonic_ns": time.monotonic_ns(),
        }
    )
    write_csv(path, rows, ["phase_index", "phase", "timestamp_utc", "monotonic_ns"])


def stable(values: list[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected_cuda_visible_devices", default="4,5")
    parser.add_argument("--expected_gpu_uuids", default=",".join(DEFAULT_GPU_UUIDS))
    parser.add_argument("--model_gpu_device_id", type=int, default=-1)
    args = parser.parse_args()

    if int(args.repeats) < 2:
        raise RuntimeError("qualification requires at least two repeats")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    phase_rows: list[dict[str, Any]] = []
    phase_path = output_dir / "m3_gpu45_phase_markers.csv"
    write_phase_marker(phase_path, phase_rows, "START")

    expected_visible = str(args.expected_cuda_visible_devices)
    actual_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if actual_visible != expected_visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be {expected_visible!r}; got {actual_visible!r}")
    physical_indices = parse_int_csv(expected_visible)
    expected_uuids = parse_str_csv(args.expected_gpu_uuids)
    uuid_map = physical_gpu_uuid_map(physical_indices)
    actual_uuids = [uuid_map[idx] for idx in physical_indices]
    if actual_uuids != expected_uuids:
        raise RuntimeError(f"GPU UUID binding mismatch: expected {expected_uuids}, got {actual_uuids}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    before_gpu = nvidia_smi_query()
    before_apps = nvidia_smi_compute_apps()
    (output_dir / "nvidia_smi_before.txt").write_text(before_gpu + "\n", encoding="utf-8")
    (output_dir / "nvidia_smi_compute_apps_before.txt").write_text(before_apps + "\n", encoding="utf-8")

    cfg = load_config(args.config)
    raw_image, clean_json = load_frozen_input(args.input_dir)
    write_phase_marker(phase_path, phase_rows, "LOAD_MODEL_BEGIN")
    model, processor, device = load_model(cfg["model"]["path"], int(args.model_gpu_device_id))
    model.eval()
    write_phase_marker(phase_path, phase_rows, "LOAD_MODEL_END")

    model_dtype = next(model.parameters()).dtype
    action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))
    instruction = str(clean_json["instruction"])
    target_token_id = int(cfg["attack_optimizer"]["target_token_id"])
    margin = float(cfg["attack_optimizer"]["gripper_margin"])
    tolerance = float(cfg["gates"]["score_tie_tolerance"])

    base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)
    input_ids_sha = tensor_sha256(base_inputs["input_ids"])
    pixel_sha = tensor_sha256(base_inputs["pixel_values"])
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=0,
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    adapter._freeze_model()

    write_phase_marker(phase_path, phase_rows, "REPEATS_BEGIN")
    repeat_rows: list[dict[str, Any]] = []
    for repeat_idx in range(int(args.repeats)):
        write_phase_marker(phase_path, phase_rows, f"REPEAT_{repeat_idx}_BEGIN")
        official = official_decode(
            model,
            base_inputs,
            action_dim=action_dim,
            unnorm_key=cfg["model"]["unnorm_key"],
            target_token_id=target_token_id,
            margin=margin,
            tolerance=tolerance,
            objective=str(cfg["attack_optimizer"]["objective"]),
        )
        grad = gradient_repeat(
            adapter=adapter,
            input_ids=base_inputs["input_ids"],
            pixel_values=base_inputs["pixel_values"],
            action_dim=action_dim,
            target_token_id=target_token_id,
            margin=margin,
        )
        repeat_rows.append(
            {
                "repeat_idx": repeat_idx,
                "official_tokens": json.dumps(official["tokens"]),
                "official_arm_prefix": json.dumps(official["arm_prefix"]),
                "official_gripper_token": official["gripper_token"],
                "official_score_row_sha256": official["score_row_sha256"],
                "official_target_margin": target_margin(official["target_stats"]),
                "official_target_token_score": official["target_stats"]["target_token_score"],
                "official_best_competitor_token_id": official["target_stats"]["best_competitor_token_id"],
                "official_best_competitor_score": official["target_stats"]["best_competitor_score"],
                "score_invariant_status": official["score_invariant"].get("status", ""),
                "gradient_loss": grad["loss"],
                "gradient_target_margin": grad["target_margin"],
                "gradient_target_token_score": grad["target_token_score"],
                "gradient_best_competitor_token_id": grad["best_competitor_token_id"],
                "gradient_best_competitor_score": grad["best_competitor_score"],
                "gradient_generated_arm_prefix": json.dumps(grad["generated_arm_prefix"]),
                "gradient_finite": grad["gradient_finite"],
                "gradient_sha256": grad["gradient_sha256"],
                "gradient_l1": grad["gradient_l1"],
                "gradient_l2": grad["gradient_l2"],
                "gradient_linf": grad["gradient_linf"],
            }
        )
        write_phase_marker(phase_path, phase_rows, f"REPEAT_{repeat_idx}_END")
    write_phase_marker(phase_path, phase_rows, "REPEATS_END")

    after_gpu = nvidia_smi_query()
    after_apps = nvidia_smi_compute_apps()
    (output_dir / "nvidia_smi_after.txt").write_text(after_gpu + "\n", encoding="utf-8")
    (output_dir / "nvidia_smi_compute_apps_after.txt").write_text(after_apps + "\n", encoding="utf-8")

    tokens_stable = stable([row["official_tokens"] for row in repeat_rows])
    gripper_stable = stable([row["official_gripper_token"] for row in repeat_rows])
    score_hash_stable = stable([row["official_score_row_sha256"] for row in repeat_rows])
    gradient_hash_stable = stable([row["gradient_sha256"] for row in repeat_rows])
    gradient_all_finite = all(str(row["gradient_finite"]) == "True" or row["gradient_finite"] is True for row in repeat_rows)
    no_optimization = True
    result_class = (
        "GPU45_FIXED_FRAME_QUALIFICATION_PASS"
        if tokens_stable
        and gripper_stable
        and score_hash_stable
        and gradient_hash_stable
        and gradient_all_finite
        else "GPU45_FIXED_FRAME_QUALIFICATION_FAIL"
    )

    repeat_fields = list(repeat_rows[0].keys())
    write_csv(output_dir / "m3_gpu45_repeatability_rows.csv", repeat_rows, repeat_fields)

    manifest_row = {
        "stage": "M3_GPU45_FIXED_FRAME_INFRA_QUALIFICATION",
        "result_class": result_class,
        "repo_commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_branch_name(),
        "dirty_status": git_dirty_status(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "transformers_version": __import__("transformers").__version__,
        "cuda_visible_devices": actual_visible,
        "physical_gpu_indices": ",".join(str(x) for x in physical_indices),
        "physical_gpu_uuids": ",".join(actual_uuids),
        "expected_gpu_uuids": ",".join(expected_uuids),
        "device_selected_by_model_loader": device,
        "model_path": cfg["model"]["path"],
        "model_fingerprint": json.dumps(model_fingerprint(model), sort_keys=True),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "input_dir": str(args.input_dir),
        "input_ids_sha256": input_ids_sha,
        "pixel_values_sha256": pixel_sha,
        "raw_image_sha256": hashlib.sha256(np.ascontiguousarray(raw_image).tobytes()).hexdigest(),
        "clean_generation_json_sha256": sha256_file(args.input_dir / "clean_generation_step78.json"),
        "repeats": int(args.repeats),
        "target_token_id": target_token_id,
        "objective": cfg["attack_optimizer"]["objective"],
        "no_pgd": no_optimization,
        "no_rand21": True,
        "no_shuffled_grad21": True,
        "no_libero_rollout": True,
        "no_panel_capture": True,
        "model_parameters_frozen_for_pixel_gradient": True,
        "torch_allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    write_csv(output_dir / "m3_gpu45_qualification_manifest.csv", [manifest_row], list(manifest_row.keys()))
    summary = {
        "result_class": result_class,
        "tokens_stable": tokens_stable,
        "gripper_stable": gripper_stable,
        "score_hash_stable": score_hash_stable,
        "gradient_hash_stable": gradient_hash_stable,
        "gradient_all_finite": gradient_all_finite,
        "no_optimization_steps": no_optimization,
        "authorized_scope": {
            "gpu45_fixed_frame_clean_forward_repeatability": True,
            "gpu45_fixed_frame_gradient_repeatability": True,
            "pgd_rand_shuffled_panel_or_rollout": False,
        },
        "repeat_rows_file": "m3_gpu45_repeatability_rows.csv",
        "manifest_file": "m3_gpu45_qualification_manifest.csv",
    }
    write_json(output_dir / "m3_gpu45_qualification_summary.json", summary)
    write_phase_marker(phase_path, phase_rows, "WRITE_ARTIFACT_HASH_MANIFEST")
    write_artifact_hash_manifest(output_dir)
    write_phase_marker(phase_path, phase_rows, "END")

    print(json.dumps({"status": result_class, "output_dir": str(output_dir)}, indent=2))
    return 0 if result_class == "GPU45_FIXED_FRAME_QUALIFICATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
