#!/usr/bin/env python3
"""GPU45 long-run fixed-frame diagnostics for M3 Layer3 infrastructure.

This script is a diagnostic runner only.  It does not run PGD21, RAND21,
SHUFFLED_GRAD21, final panel aggregation, LIBERO env.step, or closed-loop
rollouts.  It operates on development-only fixed inputs and records root-cause
evidence for direct-forward, generation, and one-backward pixel-gradient
repeatability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import random
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
from gripper_attack.execution_target import target_token_logratio_loss_and_stats  # noqa: E402
from gripper_attack.v3_generation_parity import extract_exact_new_tokens  # noqa: E402
from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (  # noqa: E402
    load_config,
    load_frozen_input,
    load_model,
    model_fingerprint,
    preprocess_raw_image,
    sha256_file,
    write_artifact_hash_manifest,
    write_csv,
    write_json,
)


AUTHORIZED_STEP78_INPUT = "/data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2/step78"
EXPECTED_VISIBLE = "4,5"
EXPECTED_UUIDS = [
    "GPU-d0a54f5d-938c-a148-fff9-c135201e3f61",
    "GPU-9794d733-042f-46a2-fc86-5a3fe32a158a",
]
FORBIDDEN_SEEDS = {85, 86, 428198}
TARGET_TOKEN = 31744
CLOSE_TOKEN = 31872


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


def canonical_json_sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def git_value(args: list[str]) -> str:
    return run_command(["git", *args], cwd=REPO_ROOT)


def git_dirty_status() -> str:
    status = git_value(["status", "--porcelain"])
    return "CLEAN" if not status else "DIRTY:" + status.replace("\n", "\\n")


def set_strict_determinism() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def physical_gpu_uuid_map(indices: list[int]) -> dict[int, str]:
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
    return mapping


def compute_apps_by_uuid() -> list[dict[str, str]]:
    raw = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu_uuid": parts[0], "pid": parts[1], "process_name": parts[2], "used_memory": parts[3]})
    return rows


def validate_gpu_binding(expected_visible: str, expected_uuids: list[str], *, require_idle: bool) -> dict[str, Any]:
    actual_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if actual_visible != expected_visible:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES mismatch: expected {expected_visible!r}, got {actual_visible!r}")
    physical_indices = parse_csv_ints(expected_visible)
    uuid_map = physical_gpu_uuid_map(physical_indices)
    ordered = [uuid_map[idx] for idx in physical_indices]
    if ordered != expected_uuids:
        raise RuntimeError(f"GPU UUID order mismatch: expected {expected_uuids}, got {ordered}")
    apps = compute_apps_by_uuid()
    if require_idle:
        busy = [app for app in apps if app["gpu_uuid"] in set(expected_uuids)]
        if busy:
            raise RuntimeError(f"target GPU has existing compute process before model load: {busy}")
    return {"physical_indices": physical_indices, "ordered_uuids": ordered, "compute_apps": apps}


def validate_forbidden_seed(seed: int | None) -> None:
    if seed is not None and int(seed) in FORBIDDEN_SEEDS:
        raise RuntimeError(f"forbidden seed requested: {seed}")


def validate_development_input(input_dir: Path, authorized_input_dir: str) -> None:
    resolved = str(input_dir.resolve())
    allowed = str(Path(authorized_input_dir).resolve())
    if resolved != allowed:
        raise RuntimeError(f"unauthorized fixed-frame input: {resolved}; expected {allowed}")
    lowered = resolved.lower()
    if "v5_final" in lowered or "final_8" in lowered or "seed85" in lowered or "seed86" in lowered:
        raise RuntimeError(f"forbidden final-panel marker in input path: {resolved}")


def clone_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in inputs.items()}


def freeze_model_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable_after = 0
    model.eval()
    for p in model.parameters():
        total += int(p.numel())
        p.requires_grad_(False)
        if p.requires_grad:
            trainable_after += int(p.numel())
    return {"parameter_count": total, "trainable_parameter_count_after_freeze": trainable_after}


def assert_no_model_parameter_grads(model: Any) -> None:
    bad = [name for name, p in model.named_parameters() if p.requires_grad or p.grad is not None]
    if bad:
        raise RuntimeError(f"model parameter gradients are not fully disabled: {bad[:5]}")


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def canonical_device_map(model: Any) -> dict[str, str]:
    device_map = getattr(model, "hf_device_map", {}) or {}
    return {str(k): str(v) for k, v in sorted(device_map.items(), key=lambda item: str(item[0]))}


def model_shard_summary(model: Any) -> list[dict[str, Any]]:
    rows = []
    for name, param in itertools.islice(model.named_parameters(), 200000):
        rows.append({"name": name, "dtype": str(param.dtype), "device": str(param.device), "shape": list(param.shape)})
    return rows


def row_summary(row: torch.Tensor, *, top_k: int = 10) -> dict[str, Any]:
    row_cpu = row.detach().float().cpu()
    top = torch.topk(row_cpu, k=int(top_k))
    finite = bool(torch.isfinite(row_cpu).all())
    target_score = float(row_cpu[TARGET_TOKEN])
    close_score = float(row_cpu[CLOSE_TOKEN])
    return {
        "score_row_sha256": tensor_sha256(row_cpu),
        "finite": finite,
        "target_31744_score": target_score,
        "close_31872_score": close_score,
        "target_minus_close": target_score - close_score,
        "top_tokens": [int(x) for x in top.indices.tolist()],
        "top_scores": [float(x) for x in top.values.tolist()],
        "top_token": int(top.indices[0]),
        "top_score": float(top.values[0]),
    }


def direct_forward_once(model: Any, inputs: Mapping[str, torch.Tensor], arm_prefix: list[int]) -> dict[str, Any]:
    cloned = clone_inputs(inputs)
    prefix = torch.tensor([arm_prefix], dtype=torch.long, device=cloned["input_ids"].device)
    context_ids = torch.cat([cloned["input_ids"], prefix], dim=1)
    cuda_sync()
    with torch.inference_mode():
        out = model(input_ids=context_ids, pixel_values=cloned["pixel_values"], use_cache=False, return_dict=True)
    cuda_sync()
    row = out.logits[0, -1, :].detach().float().cpu()
    summary = row_summary(row)
    summary.update({"context_len": int(context_ids.shape[1]), "prompt_len": int(cloned["input_ids"].shape[1])})
    return summary


def generation_once(model: Any, inputs: Mapping[str, torch.Tensor], *, action_dim: int) -> dict[str, Any]:
    cloned = clone_inputs(inputs)
    cuda_sync()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=cloned["input_ids"],
            pixel_values=cloned["pixel_values"],
            max_new_tokens=int(action_dim),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    cuda_sync()
    tokens = extract_exact_new_tokens(gen.sequences, prompt_len=int(cloned["input_ids"].shape[1]), expected_new_tokens=int(action_dim))
    score_hashes = [tensor_sha256(score[0].detach().float().cpu()) for score in gen.scores]
    argmaxes = [int(torch.argmax(score[0].detach().float().cpu())) for score in gen.scores]
    final_stats = row_summary(gen.scores[-1][0].detach().float().cpu())
    return {
        "tokens": tokens,
        "tokens_json": json.dumps(tokens),
        "arm_prefix": tokens[:6],
        "gripper_token": int(tokens[-1]),
        "sequence_sha256": tensor_sha256(gen.sequences.detach().cpu()),
        "per_token_score_sha256": json.dumps(score_hashes),
        "per_token_argmax": json.dumps(argmaxes),
        "prompt_len": int(cloned["input_ids"].shape[1]),
        **{f"final_{k}": v for k, v in final_stats.items()},
    }


def gradient_once(
    *,
    model: Any,
    adapter: TokenPrefixPGDAttacker,
    inputs: Mapping[str, torch.Tensor],
    action_dim: int,
    target_token_id: int,
    margin: float,
) -> dict[str, Any]:
    assert_no_model_parameter_grads(model)
    cloned = clone_inputs(inputs)
    pixel = cloned["pixel_values"].detach().clone().requires_grad_(True)
    if pixel.grad is not None:
        pixel.grad = None
    cuda_sync()
    prefix = adapter._generate_action_prefix_tokens(cloned["input_ids"], pixel, prefix_len=int(action_dim) - 1)
    loss, stats = adapter._generated_prefix_target_token_loss_and_stats(
        cloned["input_ids"],
        prefix,
        pixel,
        target_token_id=int(target_token_id),
        margin=float(margin),
    )
    loss.backward()
    cuda_sync()
    grad = pixel.grad
    if grad is None:
        raise RuntimeError("missing pixel gradient")
    grad_cpu = grad.detach().float().cpu()
    flat = grad_cpu.flatten()
    pos = int((flat > 0).sum())
    neg = int((flat < 0).sum())
    zero = int((flat == 0).sum())
    finite = torch.isfinite(flat)
    return {
        "loss": float(loss.detach().cpu()),
        "target_margin": float(stats.get("target_objective_margin", stats.get("target_minus_competitor_logsumexp_margin"))),
        "target_token_score": float(stats["target_token_score"]),
        "best_competitor_token_id": int(stats["best_competitor_token_id"]),
        "best_competitor_score": float(stats["best_competitor_score"]),
        "generated_arm_prefix": json.dumps([int(x) for x in prefix.detach().cpu().tolist()]),
        "gradient_sha256": tensor_sha256(grad_cpu),
        "gradient_l1": float(flat.abs().sum()),
        "gradient_l2": float(torch.linalg.vector_norm(flat, ord=2)),
        "gradient_linf": float(flat.abs().max()),
        "gradient_finite_ratio": float(finite.float().mean()),
        "gradient_positive_count": pos,
        "gradient_negative_count": neg,
        "gradient_zero_count": zero,
        "gradient_numel": int(flat.numel()),
    }


def pairwise_gradient_rows(rows: list[dict[str, Any]], grads: list[torch.Tensor]) -> list[dict[str, Any]]:
    out = []
    for i, j in itertools.combinations(range(len(grads)), 2):
        a = grads[i].detach().float().flatten()
        b = grads[j].detach().float().flatten()
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).cpu())
        sign_agreement = float((torch.sign(a) == torch.sign(b)).float().mean().cpu())
        out.append({"i": i, "j": j, "cosine_similarity": cosine, "sign_agreement": sign_agreement})
    return out


def first_generation_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"first_divergent_token_index": None, "reason": "NO_ROWS"}
    base_hashes = json.loads(rows[0]["per_token_score_sha256"])
    base_argmaxes = json.loads(rows[0]["per_token_argmax"])
    for idx in range(len(base_hashes)):
        for row_idx, row in enumerate(rows[1:], start=1):
            hashes = json.loads(row["per_token_score_sha256"])
            argmaxes = json.loads(row["per_token_argmax"])
            if hashes[idx] != base_hashes[idx] or argmaxes[idx] != base_argmaxes[idx]:
                return {
                    "first_divergent_token_index": idx,
                    "repeat_idx": row_idx,
                    "base_score_sha256": base_hashes[idx],
                    "other_score_sha256": hashes[idx],
                    "base_argmax": base_argmaxes[idx],
                    "other_argmax": argmaxes[idx],
                }
    return {"first_divergent_token_index": None, "reason": "NO_DIVERGENCE"}


def load_bundle(cfg: Mapping[str, Any], input_dir: Path, *, model_gpu_device_id: int) -> tuple[Any, Any, str, dict[str, torch.Tensor], dict[str, Any], int]:
    raw_image, clean_json = load_frozen_input(input_dir)
    model, processor, device = load_model(cfg["model"]["path"], int(model_gpu_device_id))
    freeze_model_parameters(model)
    model_dtype = next(model.parameters()).dtype
    action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))
    inputs = preprocess_raw_image(raw_image, processor, str(clean_json["instruction"]), cfg, device, model_dtype)
    return model, processor, device, inputs, clean_json, action_dim


def run_same_process(args: argparse.Namespace, cfg: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    model, processor, device, inputs, clean_json, action_dim = load_bundle(
        cfg, Path(args.input_dir), model_gpu_device_id=int(args.model_gpu_device_id)
    )
    arm_prefix = list(clean_json["official"]["arm_prefix"])
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=0,
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    adapter._freeze_model()
    assert_no_model_parameter_grads(model)

    device_map = canonical_device_map(model)
    write_json(output_dir / "device_map_manifest.json", {"hf_device_map": device_map, "sha256": canonical_json_sha(device_map)})
    write_json(output_dir / "model_shard_summary.json", model_shard_summary(model))

    direct_rows = []
    for idx in range(int(args.direct_repeats)):
        row = direct_forward_once(model, inputs, arm_prefix)
        row.update({"repeat_idx": idx, "profile": args.profile_name})
        direct_rows.append(row)
    write_csv(output_dir / "direct_forward_same_process.csv", direct_rows, list(direct_rows[0].keys()))

    generation_rows = []
    for idx in range(int(args.generation_repeats)):
        row = generation_once(model, inputs, action_dim=action_dim)
        row.update({"repeat_idx": idx, "profile": args.profile_name})
        generation_rows.append(row)
    write_csv(output_dir / "generation_same_process.csv", generation_rows, list(generation_rows[0].keys()))
    write_json(output_dir / "generation_divergence.json", first_generation_divergence(generation_rows))

    gradient_rows = []
    gradient_tensors = []
    for idx in range(int(args.gradient_repeats)):
        row = gradient_once(
            model=model,
            adapter=adapter,
            inputs=inputs,
            action_dim=action_dim,
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        )
        # Recompute the tensor hash row only; do not retain full tensors on disk.
        gradient_rows.append({"repeat_idx": idx, "profile": args.profile_name, **row})
    write_csv(output_dir / "gradient_same_process.csv", gradient_rows, list(gradient_rows[0].keys()))

    return {
        "direct_score_stable": len({r["score_row_sha256"] for r in direct_rows}) == 1,
        "direct_top_stable": len({json.dumps(r["top_tokens"]) for r in direct_rows}) == 1,
        "generation_tokens_stable": len({r["tokens_json"] for r in generation_rows}) == 1,
        "generation_gripper_stable": len({r["gripper_token"] for r in generation_rows}) == 1,
        "generation_score_stable": len({r["per_token_score_sha256"] for r in generation_rows}) == 1,
        "gradient_hash_stable": len({r["gradient_sha256"] for r in gradient_rows}) == 1,
        "gradient_all_finite": all(float(r["gradient_finite_ratio"]) == 1.0 for r in gradient_rows),
        "device_map_sha256": canonical_json_sha(device_map),
    }


def run_child_once(args: argparse.Namespace, cfg: Mapping[str, Any], output_dir: Path) -> None:
    model, processor, device, inputs, clean_json, action_dim = load_bundle(
        cfg, Path(args.input_dir), model_gpu_device_id=int(args.model_gpu_device_id)
    )
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=0,
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    adapter._freeze_model()
    arm_prefix = list(clean_json["official"]["arm_prefix"])
    result = {
        "child_index": int(args.child_index),
        "direct": direct_forward_once(model, inputs, arm_prefix),
        "generation": generation_once(model, inputs, action_dim=action_dim),
        "gradient": gradient_once(
            model=model,
            adapter=adapter,
            inputs=inputs,
            action_dim=action_dim,
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        ),
        "device_map_sha256": canonical_json_sha(canonical_device_map(model)),
    }
    write_json(output_dir / "child_once_summary.json", result)


def launch_fresh_processes(args: argparse.Namespace, output_dir: Path) -> None:
    rows = []
    for idx in range(int(args.fresh_process_count)):
        child_dir = output_dir / f"fresh_process_{idx:02d}"
        child_dir.mkdir(parents=True, exist_ok=False)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "child_once",
            "--config",
            str(args.config),
            "--input_dir",
            str(args.input_dir),
            "--output_dir",
            str(child_dir),
            "--child_index",
            str(idx),
            "--expected_cuda_visible_devices",
            str(args.expected_cuda_visible_devices),
            "--expected_gpu_uuids",
            str(args.expected_gpu_uuids),
            "--authorized_input_dir",
            str(args.authorized_input_dir),
            "--model_gpu_device_id",
            str(args.model_gpu_device_id),
            "--skip_idle_check",
        ]
        env = os.environ.copy()
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        (child_dir / "stdout_stderr.txt").write_text(proc.stdout, encoding="utf-8")
        row: dict[str, Any] = {"child_index": idx, "returncode": proc.returncode, "child_dir": str(child_dir)}
        if proc.returncode == 0:
            summary = json.loads((child_dir / "child_once_summary.json").read_text(encoding="utf-8"))
            row.update(
                {
                    "direct_score_sha256": summary["direct"]["score_row_sha256"],
                    "generation_tokens": json.dumps(summary["generation"]["tokens"]),
                    "generation_gripper": summary["generation"]["gripper_token"],
                    "generation_score_sha256": summary["generation"]["final_score_row_sha256"],
                    "gradient_sha256": summary["gradient"]["gradient_sha256"],
                    "gradient_finite_ratio": summary["gradient"]["gradient_finite_ratio"],
                    "device_map_sha256": summary["device_map_sha256"],
                }
            )
        rows.append(row)
    write_csv(output_dir / "fresh_process_repeatability.csv", rows, sorted({k for row in rows for k in row.keys()}))


def environment_manifest(args: argparse.Namespace, output_dir: Path, gpu_info: Mapping[str, Any]) -> None:
    manifest = {
        "timestamp_utc": utc_now(),
        "repo_commit": git_value(["rev-parse", "HEAD"]),
        "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_status": git_dirty_status(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "transformers": __import__("transformers").__version__,
        "accelerate": __import__("accelerate").__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_binding": gpu_info,
        "config": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "input_dir": str(args.input_dir),
        "input_files": {
            path.name: sha256_file(path)
            for path in sorted(Path(args.input_dir).glob("*"))
            if path.is_file()
        },
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", ""),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
    }
    write_json(output_dir / "environment_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["longrun", "child_once"], default="longrun")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--authorized_input_dir", default=AUTHORIZED_STEP78_INPUT)
    parser.add_argument("--expected_cuda_visible_devices", default=EXPECTED_VISIBLE)
    parser.add_argument("--expected_gpu_uuids", default=",".join(EXPECTED_UUIDS))
    parser.add_argument("--profile_name", default="C0_AUTO_DEVICE_MAP")
    parser.add_argument("--model_gpu_device_id", type=int, default=-1)
    parser.add_argument("--direct_repeats", type=int, default=10)
    parser.add_argument("--generation_repeats", type=int, default=10)
    parser.add_argument("--gradient_repeats", type=int, default=5)
    parser.add_argument("--fresh_process_count", type=int, default=0)
    parser.add_argument("--child_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_idle_check", action="store_true")
    args = parser.parse_args()

    validate_forbidden_seed(args.seed)
    validate_development_input(Path(args.input_dir), str(args.authorized_input_dir))
    set_strict_determinism()
    gpu_info = validate_gpu_binding(
        str(args.expected_cuda_visible_devices),
        parse_csv_strings(str(args.expected_gpu_uuids)),
        require_idle=not bool(args.skip_idle_check),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "nvidia_smi_before.txt").write_text(
        run_command(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "xid_before.txt").write_text(run_command(["dmesg", "-T"]) + "\n", encoding="utf-8")
    cfg = load_config(Path(args.config))
    environment_manifest(args, output_dir, gpu_info)

    if args.mode == "child_once":
        run_child_once(args, cfg, output_dir)
    else:
        summary = run_same_process(args, cfg, output_dir)
        if int(args.fresh_process_count) > 0:
            launch_fresh_processes(args, output_dir)
        write_json(output_dir / "qualification_summary.json", summary)

    (output_dir / "nvidia_smi_after.txt").write_text(
        run_command(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "xid_after.txt").write_text(run_command(["dmesg", "-T"]) + "\n", encoding="utf-8")
    write_artifact_hash_manifest(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

