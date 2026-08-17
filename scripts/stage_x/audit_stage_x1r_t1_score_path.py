"""Clean-only Stage-X T1 score/gradient-path audit.

This script loads one immutable Q00 snapshot and one suite-matched victim.  It
does greedy generation, cached/no-cache score evaluation, and autograd only.
It never calls an attack loop, mutates pixels, or interacts with an
environment.  The output is diagnostic until an independent T1 authority seal
binds the route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.execution_target import target_token_logratio_loss_and_stats  # noqa: E402
from gripper_attack.stage_x_t1_native_token_authority import (  # noqa: E402
    NativeActionTokenAuthorityV2,
    SuiteActionTokenBinding,
)
from scripts.stage_x.audit_stage_x1r_pgd_alignment import action_token_logit_row_index  # noqa: E402
from scripts.stage_x.run_stage_x1r_victim_parity import (  # noqa: E402
    append_empty_action_token,
    load_model_and_processor,
)


CANONICAL_ACTION_TOKENIZER_SHA256 = "fdc98fcbf5b0926ef2181db71946d23ffbfa052cf8443dc933d52c42a191352c"
CANONICAL_MODEL_DECODER_SHA256 = "2e672e75958205b05f40f4cd2467d3763b8e36eb2728289cd055c54213338e85"
CANONICAL_ACTION_TOKENIZER_SOURCE = "/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4/prismatic/vla/action_tokenizer.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def gpu_uuid() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def top_summary(row: Any, target_token_id: int | None = None) -> dict[str, Any]:
    import torch

    values = row.detach().float().view(-1)
    top = torch.topk(values, k=2)
    result = {
        "top1_token": int(top.indices[0].item()),
        "top2_token": int(top.indices[1].item()),
        "top1_logit": float(top.values[0].item()),
        "top2_logit": float(top.values[1].item()),
    }
    if target_token_id is not None:
        target = int(target_token_id)
        result["target_token_id"] = target
        result["target_logit"] = float(values[target].item())
        result["target_rank"] = int((values > values[target]).sum().item()) + 1
    return result


def compare_rows(left: Any, right: Any, target_token_id: int | None = None) -> dict[str, Any]:
    left_f = left.detach().float().view(-1)
    right_f = right.detach().float().view(-1)
    delta = (left_f - right_f).abs()
    result = {
        "max_abs_logit_diff": float(delta.max().item()),
        "l2_logit_diff": float(torch_norm(delta)),
        "left": top_summary(left, target_token_id),
        "right": top_summary(right, target_token_id),
    }
    result["top1_exact"] = result["left"]["top1_token"] == result["right"]["top1_token"]
    result["top2_exact"] = result["left"]["top2_token"] == result["right"]["top2_token"]
    return result


def torch_norm(value: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(value.reshape(-1)).item())


def gradient_metrics(left: Any, right: Any) -> dict[str, Any]:
    import torch

    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    cosine = float((torch.dot(a, b) / (na * nb)).item()) if float(na) and float(nb) else None
    return {
        "left_l2": float(na.item()),
        "left_linf": float(a.abs().max().item()),
        "right_l2": float(nb.item()),
        "right_linf": float(b.abs().max().item()),
        "cosine": cosine,
        "sign_agreement_fraction": float((torch.sign(a) == torch.sign(b)).float().mean().item()),
    }


def cached_rows(model: Any, prompt_ids: Any, pixel_values: Any, generated: Any) -> list[Any]:
    import torch

    out = model(input_ids=prompt_ids, pixel_values=pixel_values, use_cache=True, return_dict=True)
    past = getattr(out, "past_key_values", None)
    if past is None:
        raise RuntimeError("CACHED_AR_PAST_KEY_VALUES_MISSING")
    rows = [out.logits[0, -1, :]]
    for token in generated[:-1].detach().to(device=prompt_ids.device, dtype=torch.long).view(-1):
        out = model(input_ids=token.view(1, 1), past_key_values=past, use_cache=True, return_dict=True)
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RuntimeError("CACHED_AR_STEP_PAST_KEY_VALUES_MISSING")
        rows.append(out.logits[0, -1, :])
    return rows


def build_authority(model: Any, processor: Any, suite: str, model_path: Path) -> NativeActionTokenAuthorityV2:
    from prismatic.vla.action_tokenizer import ActionTokenizer

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    native = ActionTokenizer(processor.tokenizer, bins=int(config["n_action_bins"]))
    stats = model.get_action_stats(suite)
    binding = SuiteActionTokenBinding(
        suite=suite,
        checkpoint_path=str(model_path),
        checkpoint_config_sha256=sha256_file(model_path / "config.json"),
        tokenizer_source=CANONICAL_ACTION_TOKENIZER_SOURCE,
        tokenizer_source_sha256=CANONICAL_ACTION_TOKENIZER_SHA256,
        model_decoder_source_sha256=CANONICAL_MODEL_DECODER_SHA256,
        tokenizer_files=tuple(
            (name, sha256_file(model_path / name))
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
            if (model_path / name).exists()
        ),
        tokenizer_vocab_size=int(processor.tokenizer.vocab_size),
        n_action_bins=int(native.n_bins),
        bins=tuple(float(x) for x in np.asarray(native.bins)),
        bin_centers=tuple(float(x) for x in np.asarray(native.bin_centers)),
        q01=tuple(float(x) for x in np.asarray(stats["q01"])),
        q99=tuple(float(x) for x in np.asarray(stats["q99"])),
        mask=tuple(bool(x) for x in np.asarray(stats.get("mask", np.ones(7, dtype=bool)))),
    )
    return NativeActionTokenAuthorityV2(binding)


def arm_cached_loss(rows: Iterable[Any], generated: Any) -> Any:
    import torch
    import torch.nn.functional as F

    losses = [F.cross_entropy(row.float().view(1, -1), token.view(1)) for row, token in zip(rows[:-1], generated[:-1])]
    return torch.stack(losses).mean()


def rehydrate_model_for_autograd(model: Any) -> int:
    """Clone model tensors for pixel autograd without changing their values."""
    import torch

    count = 0
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            # Some low_cpu_mem_usage/tied-weight paths expose inference storage
            # without a reliable is_inference() flag. Clone every parameter so
            # the clean gradient audit has ordinary autograd-readable storage.
            if parameter is not None:
                module._parameters[name] = torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
                count += 1
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.detach().clone()
    return count


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
    from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot

    device = str(args.device)
    model_path = Path(args.model_path)
    model, processor = load_model_and_processor(model_path, device)
    rehydrated_parameter_count = rehydrate_model_for_autograd(model)
    package = load_snapshot(Path(args.snapshot), materialize_torch=True)
    manifest, payload = package["manifest"], package["payload"]
    processed = processor(payload["prompt"], payload["processed_image"], return_tensors="pt")
    prepared = append_empty_action_token(processed)
    input_exact = bool(torch.equal(prepared["input_ids"], payload["input_ids"]))
    attention_exact = bool(torch.equal(prepared["attention_mask"], payload["attention_mask"]))
    pixel_cast = prepared["pixel_values"].to(dtype=payload["pixel_values"].dtype)
    pixel_exact = bool(torch.equal(pixel_cast, payload["pixel_values"]))
    if not (input_exact and attention_exact and pixel_exact):
        raise RuntimeError(f"SNAPSHOT_PROCESSOR_PARITY_FAIL:{input_exact}:{attention_exact}:{pixel_exact}")

    dtype = next(model.parameters()).dtype
    prompt_ids = prepared["input_ids"].to(device=device)
    pixel_base = prepared["pixel_values"].to(device=device, dtype=dtype)
    action_dim = int(model.get_action_dim(args.suite))
    with torch.inference_mode():
        generated_out = model.generate(
            input_ids=prompt_ids,
            pixel_values=pixel_base,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_len = int(prompt_ids.shape[1])
    generated = generated_out.sequences[0, prompt_len:].detach().to(dtype=torch.long)
    if int(generated.numel()) != action_dim:
        raise RuntimeError(f"CLEAN_GENERATION_LENGTH_FAIL:{generated.numel()}:{action_dim}")
    authority = build_authority(model, processor, args.suite, model_path)
    target_token_id = authority.open_token_id()
    cfg = {
        "attack_optimizer": {
            "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            "surrogate_score_path": "cached_autoregressive_generate_v1",
            "target_token_id": int(target_token_id),
        }
    }
    cached_adapter = TokenPrefixPGDAttacker(model, processor, cfg, device=device)
    nocache_cfg = json.loads(json.dumps(cfg))
    nocache_cfg["attack_optimizer"]["surrogate_score_path"] = "uncached_full_context_v1"
    nocache_adapter = TokenPrefixPGDAttacker(model, processor, nocache_cfg, device=device)
    arm_prefix = generated[:-1]

    official_rows = [
        (score[0, -1, :] if score.ndim == 3 else score[0, :]).detach()
        for score in generated_out.scores
    ]
    with torch.no_grad():
        manual_rows = cached_rows(model, prompt_ids, pixel_base, generated)
        full_ids = torch.cat([prompt_ids, generated.view(1, -1)], dim=1)
        full_out = model(input_ids=full_ids, pixel_values=pixel_base, use_cache=False, return_dict=True)
        nocache_rows = [full_out.logits[0, action_token_logit_row_index(dim, action_dim), :].detach() for dim in range(action_dim)]

    def grad_target(adapter: Any, pixels: Any):
        loss, stats = adapter._generated_prefix_target_token_loss_and_stats(
            prompt_ids,
            arm_prefix,
            pixels,
            target_token_id=int(target_token_id),
            margin=5.0,
        )
        grad = torch.autograd.grad(loss, pixels, retain_graph=False, create_graph=False)[0]
        return loss.detach(), stats, grad.detach()

    cached_pixels = pixel_base.detach().clone().requires_grad_(True)
    nocache_pixels = pixel_base.detach().clone().requires_grad_(True)
    cached_target_loss, cached_target_stats, cached_target_grad = grad_target(cached_adapter, cached_pixels)
    nocache_target_loss, nocache_target_stats, nocache_target_grad = grad_target(nocache_adapter, nocache_pixels)

    cached_arm_pixels = pixel_base.detach().clone().requires_grad_(True)
    nocache_arm_pixels = pixel_base.detach().clone().requires_grad_(True)
    cached_arm_rows = cached_rows(model, prompt_ids, cached_arm_pixels, generated)
    cached_arm_loss = arm_cached_loss(cached_arm_rows, generated)
    cached_arm_grad = torch.autograd.grad(cached_arm_loss, cached_arm_pixels, retain_graph=False, create_graph=False)[0].detach()
    nocache_arm_loss, nocache_arm_stats, nocache_arm_grad = nocache_adapter._clean_generated_arm_preservation_loss_and_stats(
        prompt_ids,
        generated.view(1, -1),
        nocache_arm_pixels,
        action_dim,
        arm_preserve_weight=1.0,
    )
    nocache_arm_grad = torch.autograd.grad(nocache_arm_loss, nocache_arm_pixels, retain_graph=False, create_graph=False)[0].detach()

    row_comparisons = []
    for dim in range(action_dim):
        row_comparisons.append({
            "dim": dim,
            "official_vs_cached": compare_rows(official_rows[dim], manual_rows[dim], int(target_token_id) if dim == action_dim - 1 else None),
            "official_vs_nocache": compare_rows(official_rows[dim], nocache_rows[dim], int(target_token_id) if dim == action_dim - 1 else None),
            "generated_token_id": int(generated[dim].item()),
            "row_index_no_cache": int(action_token_logit_row_index(dim, action_dim)),
        })
    return {
        "schema": "STAGE_X_X1R_T1_SCORE_PATH_AUDIT_V1",
        "status": "DIAGNOSTIC_CLEAN_ONLY",
        "suite": args.suite,
        "snapshot_root": str(args.snapshot),
        "snapshot_binding": manifest.get("binding", {}),
        "model_path": str(model_path),
        "source": {"commit": git_value("rev-parse", "HEAD"), "tree": git_value("rev-parse", "HEAD^{tree}")},
        "runtime": {
            "pid": os.getpid(),
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_uuid": gpu_uuid(),
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "use_cache": {"cached": True, "nocache": False},
            "model_parameter_rehydrated_for_autograd": int(rehydrated_parameter_count),
        },
        "processor_parity": {"input_ids_exact": input_exact, "attention_mask_exact": attention_exact, "pixel_values_exact": pixel_exact},
        "authority": authority.receipt(),
        "generated_token_ids": [int(x) for x in generated.cpu().tolist()],
        "row_comparisons": row_comparisons,
        "target_token": {
            "token_id": int(target_token_id),
            "cached_loss": float(cached_target_loss.item()),
            "nocache_loss": float(nocache_target_loss.item()),
            "cached_stats": cached_target_stats,
            "nocache_stats": nocache_target_stats,
            "gradient": gradient_metrics(cached_target_grad, nocache_target_grad),
        },
        "arm_preservation": {
            "cached_loss": float(cached_arm_loss.item()),
            "nocache_loss": float(nocache_arm_loss.item()),
            "nocache_stats": nocache_arm_stats,
            "gradient": gradient_metrics(cached_arm_grad, nocache_arm_grad),
            "cached_route": "manual_cached_autoregressive_past_key_values_v2",
            "nocache_route": "historical_full_teacher_forced_v1",
        },
        "counters": {
            "pgd_calls": 0,
            "env_step_calls": 0,
            "physical_interventions": 0,
            "vphys_reads": 0,
            "attack_outcome_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
        },
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "suite": report["suite"], "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
