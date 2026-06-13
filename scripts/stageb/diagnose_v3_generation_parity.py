#!/usr/bin/env python3
"""Standalone V3 generation-parity diagnostic.

Consumes validated replay bundles and compares the same prompt plus
adversarial tensor across four paths:

  A: official generate, default cache behavior, output_scores=True
  B: full-sequence forward, use_cache=False
  C: full-sequence forward, use_cache=True
  D: generate(use_cache=False), output_scores=True

This script does not create replay bundles and must not be used as a rollout or
GPU experiment launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gripper_attack.v3_generation_parity import (  # noqa: E402
    ACTION_DIM,
    classify_disc_and_raw,
    classify_path_diagnosis,
    extract_exact_new_tokens,
    generation_score_audit_from_row,
    path_result_schema,
    require_token_list,
    summarize_score_row,
    validate_generation_score_invariant,
    validate_replay_bundle,
)


def load_model_s20d(model_path: str, model_gpu_device_id: int = -1):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {
            "device_map": {"": int(model_gpu_device_id)},
            "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"},
        }
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw)
    return model, processor


def infer_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for value in model.hf_device_map.values():
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def infer_model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def score_context(model, bundle: dict[str, Any]):
    vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    n_bins = int(model.bin_centers.shape[0])
    stats = model.get_action_stats("libero_object")
    surrogate_top = int(bundle["surrogate_global_top_token"])
    return {
        "vocab_eff": vocab_eff,
        "n_bins": n_bins,
        "bin_centers": model.bin_centers,
        "action_stats": stats,
        "surrogate_top_token": surrogate_top,
    }


def token_execution(model, token_id: int) -> dict[str, Any]:
    vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    n_bins = int(model.bin_centers.shape[0])
    return classify_disc_and_raw(
        int(token_id), vocab_eff, n_bins, model.bin_centers,
        model.get_action_stats("libero_object"))


def run_generate_path(model, input_ids, pixel_values, bundle, *, path_name: str, use_cache_arg: bool | None):
    kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "max_new_tokens": ACTION_DIM,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if use_cache_arg is not None:
        kwargs["use_cache"] = bool(use_cache_arg)
    with torch.inference_mode():
        gen = model.generate(**kwargs)
    prompt_len = int(input_ids.shape[1])
    generated = extract_exact_new_tokens(
        gen.sequences, prompt_len=prompt_len, expected_new_tokens=ACTION_DIM)
    if len(generated) != ACTION_DIM:
        raise RuntimeError(f"{path_name}: generated {len(generated)} tokens, expected {ACTION_DIM}")
    if path_name == "A" and generated != [int(x) for x in bundle["official_generated_action_token_ids"]]:
        raise RuntimeError(
            f"Path A official reproduction mismatch: generated={generated} "
            f"bundle={bundle['official_generated_action_token_ids']}")
    if not getattr(gen, "scores", None) or len(gen.scores) != ACTION_DIM:
        raise RuntimeError(f"{path_name}: expected {ACTION_DIM} processed score rows")
    final_scores = gen.scores[-1][0].detach().float().cpu()
    emitted = int(generated[-1])
    processed = generation_score_audit_from_row(
        final_scores,
        emitted_token=emitted,
        **score_context(model, bundle),
    )
    ok, failure = validate_generation_score_invariant(processed, emitted)
    if not ok:
        raise RuntimeError(f"{path_name}: {failure}")
    return path_result_schema(
        path=path_name,
        cache_behavior="default" if use_cache_arg is None else f"use_cache={bool(use_cache_arg)}",
        prefix_tokens=generated[:6],
        generated_tokens=generated,
        emitted_gripper_token=emitted,
        processed_score_summary=processed,
        token_execution=token_execution(model, emitted),
    )


def run_forward_path(model, input_ids, pixel_values, prefix_ids, bundle, *, path_name: str, use_cache: bool):
    context = torch.cat([input_ids, prefix_ids.view(1, -1).to(input_ids.device)], dim=1)
    with torch.inference_mode():
        out = model(
            input_ids=context,
            pixel_values=pixel_values,
            use_cache=bool(use_cache),
            return_dict=True,
        )
    logits = out.logits.float()[0, -1, :].detach().cpu()
    raw = summarize_score_row(logits, **score_context(model, bundle))
    top = int(raw["top1_token"])
    return path_result_schema(
        path=path_name,
        cache_behavior=f"use_cache={bool(use_cache)}",
        prefix_tokens=[int(x) for x in prefix_ids.detach().cpu().tolist()],
        generated_tokens=None,
        emitted_gripper_token=None,
        raw_logit_summary=raw,
        token_execution=token_execution(model, top),
        unavailable_reason="forward_path_has_no_generated_tokens_or_processed_scores",
    )


def iter_replay_jsons(replay_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in replay_paths:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"replay path does not exist: {p}")
    if not paths:
        raise FileNotFoundError("no replay JSON files found")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Model path override. Defaults to each bundle model_path.")
    parser.add_argument("--model_gpu_device_id", type=int, default=-1)
    parser.add_argument("--replay", nargs="+", required=True, help="Replay JSON files or directories.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    replay_jsons = iter_replay_jsons(args.replay)
    bundles = []
    for path in replay_jsons:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        issues = validate_replay_bundle(bundle, bundle_dir=path.parent, verify_tensor=True)
        if issues:
            raise RuntimeError(f"invalid replay bundle {path}: {issues}")
        bundles.append((path, bundle))

    model_path = args.model or str(bundles[0][1]["model_path"])
    model, _processor = load_model_s20d(model_path, model_gpu_device_id=int(args.model_gpu_device_id))
    device = infer_input_device(model)
    dtype = infer_model_dtype(model)

    results = []
    for path, bundle in bundles:
        tensor_path = path.parent / str(bundle["adv_tensor_filename"])
        pixel_values = torch.load(tensor_path, map_location="cpu").to(device=device, dtype=dtype)
        input_ids = torch.tensor(bundle["prompt_input_ids"], device=device, dtype=torch.long)
        prefix_ids = torch.tensor(
            require_token_list(
                bundle["surrogate_generated_arm_prefix_token_ids"],
                expected_len=6,
                label="surrogate_generated_arm_prefix_token_ids",
            ),
            device=device,
            dtype=torch.long,
        )
        paths = {
            "A": run_generate_path(model, input_ids, pixel_values, bundle, path_name="A", use_cache_arg=None),
            "B": run_forward_path(model, input_ids, pixel_values, prefix_ids, bundle, path_name="B", use_cache=False),
            "C": run_forward_path(model, input_ids, pixel_values, prefix_ids, bundle, path_name="C", use_cache=True),
            "D": run_generate_path(model, input_ids, pixel_values, bundle, path_name="D", use_cache_arg=False),
        }
        diagnosis = classify_path_diagnosis(paths, bundle)
        results.append({
            "replay_json": str(path),
            "task": bundle["task"],
            "state_id": bundle["state_id"],
            "attack_seed": bundle["attack_seed"],
            "job_id": bundle["job_id"],
            "step": bundle["step"],
            "objective": bundle["objective"],
            "diagnosis": diagnosis,
            "paths": paths,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ok] wrote {len(results)} diagnostic entries -> {out_path}")
    for item in results:
        diagnosis_class = item["diagnosis"]["class"]
        print(
            f"[ok] {Path(item['replay_json']).name}: diagnosis={diagnosis_class} "
            f"A={item['paths']['A']['emitted_gripper_token']} "
            f"B={item['paths']['B']['raw_logit_top_token']} "
            f"C={item['paths']['C']['raw_logit_top_token']} "
            f"D={item['paths']['D']['emitted_gripper_token']}"
        )


if __name__ == "__main__":
    main()
