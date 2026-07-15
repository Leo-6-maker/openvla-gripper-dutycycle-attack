#!/usr/bin/env python3
"""Run the B3 stateful engineering smoke; never consumes Official Teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from gripper_attack.b3_stateful import (
    B3_25D,
    B3_25D9D,
    B3_HEADS,
    compute_b3_loss,
    load_b3_checkpoint,
    save_b3_checkpoint,
)


MODEL_CLASSES = {"B3_25D": B3_25D, "B3_25D9D": B3_25D9D}
LENGTHS = (1, 2, 3, 16, 31, 220, 280, 300, 520)
BATCHES = (1, 4, 8)
CHUNKS = (16, 32, 64)
SEED = 20260715


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tracked_tree_sha256() -> str:
    digest = hashlib.sha256()
    names = git("ls-files", "-z").split("\0")
    for name in filter(None, names):
        path = REPO_ROOT / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hidden_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, tuple):
        return max(hidden_max_abs(a, b) for a, b in zip(left, right))
    return float((left - right.to(left.device)).abs().max().item())


def logits_max_abs(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((left[name] - right[name]).abs().max().item()) for name in left)


def stepwise(model, x25: torch.Tensor, x9: torch.Tensor | None, mask: torch.Tensor | None = None):
    rows = {f"{name}_logit": [] for name in B3_HEADS}
    hidden = None
    for step in range(x25.shape[1]):
        output, hidden = model.step(
            x25[:, step],
            None if x9 is None else x9[:, step],
            hidden,
            None if mask is None else mask[:, step],
        )
        for name, value in output.items():
            rows[name].append(value)
    return {name: torch.stack(values, dim=1) for name, values in rows.items()}, hidden


def make_inputs(model_name: str, batch: int, length: int, device: torch.device):
    x25 = torch.randn(batch, length, 25, device=device)
    x9 = torch.randn(batch, length, 9, device=device) if model_name == "B3_25D9D" else None
    return x25, x9


def same_device_parity(device: torch.device) -> dict[str, Any]:
    rows = []
    max_logits = 0.0
    max_hidden = 0.0
    for model_name, model_cls in MODEL_CLASSES.items():
        model = model_cls().to(device).eval()
        for length in LENGTHS:
            for batch in BATCHES:
                x25, x9 = make_inputs(model_name, batch, length, device)
                with torch.no_grad():
                    sequence, sequence_hidden = model.forward_sequence(x25, x9)
                    step, step_hidden = stepwise(model, x25, x9)
                logits_error = logits_max_abs(sequence, step)
                hidden_error = hidden_max_abs(sequence_hidden, step_hidden)
                max_logits = max(max_logits, logits_error)
                max_hidden = max(max_hidden, hidden_error)
                rows.append({"model": model_name, "length": length, "batch": batch,
                             "logits_max_abs": logits_error, "hidden_max_abs": hidden_error})
    return {
        "pass": max_logits <= 1e-6 and max_hidden <= 1e-6,
        "max_logits_abs": max_logits,
        "max_hidden_abs": max_hidden,
        "cases": rows,
    }


def cpu_gpu_parity(device: torch.device) -> dict[str, Any]:
    rows = []
    max_logits = 0.0
    max_hidden = 0.0
    for model_name, model_cls in MODEL_CLASSES.items():
        torch.manual_seed(SEED)
        gpu_model = model_cls().to(device).eval()
        cpu_model = model_cls().cpu().eval()
        cpu_model.load_state_dict({name: value.detach().cpu() for name, value in gpu_model.state_dict().items()})
        x25_gpu, x9_gpu = make_inputs(model_name, 1, 520, device)
        x25_cpu = x25_gpu.cpu()
        x9_cpu = None if x9_gpu is None else x9_gpu.cpu()
        with torch.no_grad():
            gpu_logits, gpu_hidden = gpu_model.forward_sequence(x25_gpu, x9_gpu)
            cpu_logits, cpu_hidden = cpu_model.forward_sequence(x25_cpu, x9_cpu)
        logits_error = logits_max_abs(gpu_logits, {name: value.to(device) for name, value in cpu_logits.items()})
        hidden_error = hidden_max_abs(gpu_hidden, cpu_hidden)
        max_logits = max(max_logits, logits_error)
        max_hidden = max(max_hidden, hidden_error)
        rows.append({"model": model_name, "length": 520, "batch": 1,
                     "logits_max_abs": logits_error, "hidden_max_abs": hidden_error})
    return {"pass": max_logits <= 1e-5 and max_hidden <= 1e-5,
            "max_logits_abs": max_logits, "max_hidden_abs": max_hidden, "cases": rows}


def padding_parity(device: torch.device) -> dict[str, Any]:
    rows = []
    maximum = 0.0
    for model_name, model_cls in MODEL_CLASSES.items():
        model = model_cls().to(device).eval()
        x25, x9 = make_inputs(model_name, 1, 31, device)
        mask = torch.tensor([[True] * 16 + [False] * 15], device=device)
        with torch.no_grad():
            prefix, prefix_hidden = model.forward_sequence(x25[:, :16], None if x9 is None else x9[:, :16])
            padded, padded_hidden = model.forward_sequence(x25, x9, mask=mask)
        error = max(logits_max_abs(prefix, {name: value[:, :16] for name, value in padded.items()}),
                    hidden_max_abs(prefix_hidden, padded_hidden))
        maximum = max(maximum, error)
        rows.append({"model": model_name, "valid_steps": 16, "padded_steps": 15, "max_abs": error})
    return {"pass": maximum <= 1e-6, "max_abs": maximum, "cases": rows}


def tbptt_parity(device: torch.device) -> dict[str, Any]:
    rows = []
    maximum = 0.0
    for model_name, model_cls in MODEL_CLASSES.items():
        model = model_cls().to(device).eval()
        x25, x9 = make_inputs(model_name, 1, 101, device)
        with torch.no_grad():
            full, full_hidden = model.forward_sequence(x25, x9)
            for chunk in CHUNKS:
                parts = {f"{name}_logit": [] for name in B3_HEADS}
                hidden = None
                for start in range(0, 101, chunk):
                    end = min(start + chunk, 101)
                    part, hidden = model.forward_sequence(x25[:, start:end],
                                                          None if x9 is None else x9[:, start:end],
                                                          hidden=hidden)
                    for name, value in part.items():
                        parts[name].append(value)
                    if isinstance(hidden, tuple):
                        hidden = tuple(value.detach() for value in hidden)
                    else:
                        hidden = hidden.detach()
                joined = {name: torch.cat(values, dim=1) for name, values in parts.items()}
                error = max(logits_max_abs(full, joined), hidden_max_abs(full_hidden, hidden))
                maximum = max(maximum, error)
                rows.append({"model": model_name, "chunk": chunk, "max_abs": error})
    return {"pass": maximum <= 1e-6, "max_abs": maximum, "cases": rows}


def masked_loss_and_grad(device: torch.device) -> dict[str, Any]:
    model = B3_25D9D().to(device).train()
    x25, x9 = make_inputs("B3_25D9D", 2, 6, device)
    outputs, _ = model.forward_sequence(x25, x9)
    targets = {name: torch.full_like(value, float("nan")) for name, value in outputs.items()}
    targets = {name.removesuffix("_logit"): value for name, value in targets.items()}
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    loss = compute_b3_loss(outputs, targets, masks)
    loss.backward()
    finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    return {"pass": bool(torch.isfinite(loss)) and loss.item() == 0.0 and finite,
            "loss": float(loss.item()), "gradients_finite": finite}


def checkpoint_roundtrip(device: torch.device, output_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    model = B3_25D9D().to(device).eval()
    checkpoint = output_root / "b3_stateful_smoke_checkpoint.pt"
    save_b3_checkpoint(checkpoint, model, extra={"seed": SEED, "device": str(device), **provenance})
    restored, config, normalization, payload = load_b3_checkpoint(checkpoint, map_location=device)
    restored = restored.to(device).eval()
    x25, x9 = make_inputs("B3_25D9D", 1, 16, device)
    with torch.no_grad():
        before, _ = model.forward_sequence(x25, x9)
        after, _ = restored.forward_sequence(x25, x9)
    error = logits_max_abs(before, after)
    flags = payload.get("status") == "ENGINEERING_SMOKE_ONLY" and payload.get("formal_model") is False
    return {"pass": error <= 1e-7 and flags, "max_abs": error,
            "checkpoint": str(checkpoint), "config_hash": config.sha256,
            "normalization_hash": normalization.sha256,
            "status": payload.get("status"), "formal_model": payload.get("formal_model")}


def synthetic_overfit(device: torch.device) -> dict[str, Any]:
    rows = []
    for model_name, model_cls in MODEL_CLASSES.items():
        torch.manual_seed(SEED)
        model = model_cls().to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        x25, x9 = make_inputs(model_name, 8, 8, device)
        labels = (x25[..., 0] > 0).float()
        targets = {name: labels for name in B3_HEADS}
        masks = {name: torch.ones_like(labels, dtype=torch.bool) for name in B3_HEADS}
        with torch.no_grad():
            initial = float(compute_b3_loss(model.forward_sequence(x25, x9)[0], targets, masks).item())
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.forward_sequence(x25, x9)
            loss = compute_b3_loss(logits, targets, masks)
            loss.backward()
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                return {"pass": False, "cases": rows, "failure": f"non-finite gradient in {model_name}"}
            optimizer.step()
        with torch.no_grad():
            final = float(compute_b3_loss(model.forward_sequence(x25, x9)[0], targets, masks).item())
        rows.append({"model": model_name, "initial_loss": initial, "final_loss": final})
    return {"pass": all(row["final_loss"] < row["initial_loss"] for row in rows), "cases": rows}


def throughput(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for model_name, model_cls in MODEL_CLASSES.items():
        for batch in BATCHES:
            model = model_cls().to(device).eval()
            x25, x9 = make_inputs(model_name, batch, 520, device)
            for _ in range(2):
                with torch.no_grad():
                    model.forward_sequence(x25, x9)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.no_grad():
                model.forward_sequence(x25, x9)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            labels = (x25[..., 0] > 0).float()
            targets = {name: labels for name in B3_HEADS}
            masks = {name: torch.ones_like(labels, dtype=torch.bool) for name in B3_HEADS}
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            train_start = time.perf_counter()
            logits, _ = model.forward_sequence(x25, x9)
            loss = compute_b3_loss(logits, targets, masks)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            training_ms = (time.perf_counter() - train_start) * 1000.0
            rows.append({"model": model_name, "batch": batch, "length": 520,
                         "forward_ms": elapsed_ms,
                         "forward_ms_per_step": elapsed_ms / 520.0,
                         "training_ms": training_ms,
                         "training_ms_per_batch": training_ms,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    return rows


def nvidia_smi() -> str:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return result.stdout + ("\n" + result.stderr if result.stderr else "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("GPU smoke requires an available CUDA device")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    config_path = REPO_ROOT / "configs" / "B3_STATEFUL_ENGINEERING_SMOKE_V1.json"
    script_path = Path(__file__).resolve()
    git_head = git("rev-parse", "HEAD")
    git_status = git("status", "--porcelain")
    pre_nvidia_smi = nvidia_smi()
    (output_root / "PRE_NVIDIA_SMI.txt").write_text(pre_nvidia_smi, encoding="utf-8")
    provenance = {
        "git_head": git_head,
        "protocol_config_sha256": sha256_file(REPO_ROOT / "configs" / "B3_RETENTION_PROTOCOL_V1.json"),
        "model_source_sha256": sha256_file(REPO_ROOT / "src" / "gripper_attack" / "b3_stateful.py"),
        "training_script_sha256": sha256_file(script_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    results = {
        "same_device_parity": same_device_parity(device),
        "cpu_gpu_parity": cpu_gpu_parity(device),
        "padding_parity": padding_parity(device),
        "tbptt_parity": tbptt_parity(device),
        "masked_loss_and_grad": masked_loss_and_grad(device),
        "checkpoint_roundtrip": checkpoint_roundtrip(device, output_root, provenance),
        "synthetic_overfit": synthetic_overfit(device),
        "throughput": throughput(device),
    }
    gates = [results[name]["pass"] for name in (
        "same_device_parity", "cpu_gpu_parity", "padding_parity", "tbptt_parity",
        "masked_loss_and_grad", "checkpoint_roundtrip", "synthetic_overfit",
    )]
    status = "PASS" if all(gates) else "FAIL"
    post_nvidia_smi = nvidia_smi()
    (output_root / "POST_NVIDIA_SMI.txt").write_text(post_nvidia_smi, encoding="utf-8")
    manifest = {
        "schema": "B3_STATEFUL_GPU_ENGINEERING_SMOKE_V1",
        "status": status,
        "engineering_only": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "eligible_for_model_selection": False,
        "uses_official_teacher_labels": False,
        "uses_openvla": False,
        "uses_mujoco": False,
        "attack_files_produced": False,
        "git_head": git_head,
        "worktree_clean": not bool(git_status),
        "git_status": git_status,
        "source_tree_sha256": tracked_tree_sha256(),
        "script_sha256": sha256_file(script_path),
        "config_sha256": sha256_file(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid_and_driver_before": pre_nvidia_smi,
        "gpu_uuid_and_driver_after": post_nvidia_smi,
        "dtype": "float32",
        "seed": SEED,
        "lengths": list(LENGTHS),
        "batches": list(BATCHES),
        "tbptt_chunks": list(CHUNKS),
        "parameter_count": {name: sum(parameter.numel() for parameter in MODEL_CLASSES[name]().parameters()) for name in MODEL_CLASSES},
        "results": results,
    }
    (output_root / "B3_STATEFUL_GPU_SMOKE_STATUS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = [
        "# B3 Stateful GPU Engineering Smoke",
        "",
        f"Status: `{status}`",
        "",
        "This output is engineering-only; it is not eligible for model selection, Official Teacher training, CAL/CHECK, or attack.",
        "",
        f"Git HEAD: `{git_head}`",
        f"Worktree clean: `{not bool(git_status)}`",
        f"GPU: `{torch.cuda.get_device_name(device)}`",
        "",
    ]
    for name in ("same_device_parity", "cpu_gpu_parity", "padding_parity", "tbptt_parity",
                 "masked_loss_and_grad", "checkpoint_roundtrip", "synthetic_overfit"):
        summary.append(f"- {name}: `{results[name]['pass']}`")
    (output_root / "B3_STATEFUL_GPU_SMOKE_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
