#!/usr/bin/env python3
"""M3 true-PGD fixed-frame runner for the Tomato step78 canary.

The script is intentionally gate-oriented:

* ``capture_input`` freezes the clean step78 observation.
* ``preflight_zero_step`` compares the surrogate score path with official
  generation before any 20-step optimization is allowed.
* ``canary`` runs the preregistered fixed-frame conditions.

It does not perform a LIBERO closed-loop attack rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from gripper_attack.attack_adapter import (  # noqa: E402
    OpenVLAVisualAttacker,
    TokenPrefixPGDAttacker,
    get_adv_inputs_from_attack_result,
    prepare_openvla_image_for_attack,
)
from gripper_attack.execution_target import (  # noqa: E402
    TARGET_31744,
    target_token_cw_loss_and_stats,
    target_token_logratio_loss_and_stats,
)
from gripper_attack.m3_controls import (  # noqa: E402
    project_and_cast_processor_values,
    rand_seed_schedule,
    sample_processor_delta,
    select_best_surrogate_only,
    tensor_sha256,
)
from gripper_attack.route_contract import (  # noqa: E402
    RouteContractError,
    route_config_from_attack_config,
    validate_true_pgd_attack_result,
)
from gripper_attack.v3_generation_parity import (  # noqa: E402
    extract_exact_new_tokens,
    generation_score_audit_from_row,
)
from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import (  # noqa: E402
    validate_processed_argmax_matches_emitted,
)
from v4_run_eval_openvla import (  # noqa: E402
    decode_with_scores,
    postprocess_openvla_action_for_libero,
    prompt,
)


TASK_IDX = {
    "alphabet_soup": 0,
    "cream_cheese": 1,
    "salad_dressing": 2,
    "bbq_sauce": 3,
    "ketchup": 4,
    "tomato_sauce": 5,
    "butter": 6,
    "milk": 7,
    "chocolate_pudding": 8,
    "orange_juice": 9,
}

MAIN_CONDITIONS = ["CLEAN", "PGD_DELTA0", "TRUE_PGD_FINAL", "RAND20", "SHUFFLED_GRAD_PGD20"]
V4_CONDITIONS = [
    "PGD_DELTA0",
    "TRUE_PGD_TRAJECTORY21_SELECTIVE",
    "RAND21_SELECTIVE",
    "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
]
PRELIGHT_CLASSES = {
    "SURROGATE_OFFICIAL_SCORE_PATH_MATCH",
    "SURROGATE_OFFICIAL_SCORE_PATH_TIE_EQUIVALENT",
    "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return {"tensor_sha256": tensor_sha256(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.tolist()
        return {"array_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(), "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items() if k not in {"adv_inputs", "delta0_adv_inputs"}}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def model_fingerprint(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_type": str(getattr(cfg, "model_type", "")),
        "vocab_size": int(getattr(getattr(cfg, "text_config", cfg), "vocab_size", 0) or 0),
        "pad_to_multiple_of": int(getattr(cfg, "pad_to_multiple_of", 0) or 0),
        "action_bins": int(getattr(getattr(model, "bin_centers", []), "shape", [0])[0] or 0),
        "norm_stats_keys": sorted(list(getattr(model, "norm_stats", {}).keys())),
    }


def load_model(model_path: str, model_gpu_device_id: int = -1):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        visible = torch.cuda.device_count()
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {"device_map": {"": int(model_gpu_device_id)}, "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"}}
    model = AutoModelCls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
        **extra_kw,
    )
    device = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                device = v
                break
            if isinstance(v, int):
                device = f"cuda:{v}"
                break
    return model, processor, device


def preprocess_raw_image(raw_image: np.ndarray, processor: Any, instruction: str, cfg: Mapping[str, Any], device: str, model_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    prep = dict(cfg.get("preprocess", {}))
    image = prepare_openvla_image_for_attack(raw_image, **prep)
    inputs = processor(prompt(instruction), image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    input_ids = inputs["input_ids"].to(device)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
    return {"input_ids": input_ids, "pixel_values": pixel_values}


def official_decode(
    model: Any,
    adv_inputs: Mapping[str, torch.Tensor],
    *,
    action_dim: int,
    unnorm_key: str,
    target_token_id: int,
    margin: float,
    tolerance: float,
    objective: str = "autoregressive_prefix_gripper_target_token_cw_v1",
) -> dict[str, Any]:
    input_ids = adv_inputs["input_ids"]
    pixel_values = adv_inputs["pixel_values"]
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=int(action_dim),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    tokens = extract_exact_new_tokens(gen.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=int(action_dim))
    score_row = gen.scores[-1][0].detach().float().cpu()
    invariant = validate_processed_argmax_matches_emitted(score_row, int(tokens[-1]), tolerance=float(tolerance))
    if str(objective) in {
        "autoregressive_prefix_gripper_target_token_logratio_v2",
        "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
    }:
        _, stats = target_token_logratio_loss_and_stats(score_row, target_token_id=int(target_token_id))
    else:
        _, stats = target_token_cw_loss_and_stats(score_row, target_token_id=int(target_token_id), margin=float(margin))
    vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    audit = generation_score_audit_from_row(
        score_row,
        emitted_token=int(tokens[-1]),
        vocab_eff=vocab_eff,
        n_bins=int(model.bin_centers.shape[0]),
        bin_centers=model.bin_centers,
        action_stats=model.get_action_stats(unnorm_key),
        surrogate_top_token=int(stats["best_competitor_token_id"]),
    )
    return {
        "tokens": tokens,
        "arm_prefix": tokens[:6],
        "gripper_token": int(tokens[-1]),
        "score_row_sha256": tensor_sha256(score_row),
        "score_invariant": invariant,
        "target_stats": stats,
        "score_audit": audit,
    }


def surrogate_stats_from_generated_prefix(
    attacker: TokenPrefixPGDAttacker,
    clean_input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    *,
    action_dim: int,
    target_token_id: int,
    margin: float,
) -> dict[str, Any]:
    prefix = attacker._generate_action_prefix_tokens(clean_input_ids, pixel_values, prefix_len=int(action_dim) - 1)
    with torch.no_grad():
        _loss, stats = attacker._generated_prefix_target_token_loss_and_stats(
            clean_input_ids,
            prefix,
            pixel_values,
            target_token_id=int(target_token_id),
            margin=float(margin),
        )
    stats = dict(stats)
    stats["generated_arm_prefix"] = [int(x) for x in prefix.detach().cpu().tolist()]
    return stats


def compare_surrogate_official(surrogate: Mapping[str, Any], official: Mapping[str, Any], *, tolerance: float) -> str:
    s_target = float(surrogate["target_token_score"])
    o_target = float(official["target_stats"]["target_token_score"])
    s_margin = float(surrogate["target_minus_best_competitor_margin"])
    o_margin = float(official["target_stats"]["target_minus_best_competitor_margin"])
    if abs(s_target - o_target) <= tolerance and abs(s_margin - o_margin) <= tolerance:
        return "SURROGATE_OFFICIAL_SCORE_PATH_MATCH"
    if abs(s_margin - o_margin) <= 1e-3:
        return "SURROGATE_OFFICIAL_SCORE_PATH_TIE_EQUIVALENT"
    return "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH"


def build_attacker(model: Any, processor: Any, cfg: Mapping[str, Any], *, seed: int, device: str, gradient_transform: str = "none") -> OpenVLAVisualAttacker:
    opt = dict(cfg["attack_optimizer"])
    opt["gradient_transform"] = str(gradient_transform)
    opt["gradient_transform_seed"] = int(seed) + 100000
    return OpenVLAVisualAttacker(
        model=model,
        processor=processor,
        config={"attack_optimizer": opt},
        seed=int(seed),
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )


def run_true_pgd_condition(
    *,
    name: str,
    model: Any,
    processor: Any,
    cfg: Mapping[str, Any],
    raw_image: np.ndarray,
    instruction: str,
    clean_action: np.ndarray,
    clean_gen: Any,
    device: str,
    seed: int,
    gradient_transform: str = "none",
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    attacker = build_attacker(model, processor, cfg, seed=seed, device=device, gradient_transform=gradient_transform)
    route = route_config_from_attack_config({"attack_optimizer": dict(cfg["attack_optimizer"], gradient_transform=gradient_transform)})
    result = attacker.attack(raw_image, instruction, clean_action, clean_action, clean_gen, unnorm_key=cfg["model"]["unnorm_key"])
    validate_true_pgd_attack_result(result, route)
    adv_inputs = get_adv_inputs_from_attack_result(result)
    return {"condition": name, "attack_result": result, "debug": result.debug}, adv_inputs


def make_delta0_inputs(
    *,
    adv_inputs: Mapping[str, torch.Tensor],
    seed: int,
    epsilon: float,
    random_start: bool,
) -> Mapping[str, torch.Tensor]:
    x = adv_inputs["pixel_values"]
    if random_start:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        delta = torch.empty_like(x.detach().float()).uniform_(-float(epsilon), float(epsilon), generator=gen)
    else:
        delta = torch.zeros_like(x.detach().float())
    projected, _ = project_and_cast_processor_values(x, delta, epsilon=float(epsilon), candidate_is_delta=True)
    return {"input_ids": adv_inputs["input_ids"], "pixel_values": projected.detach()}


def run_rand20(
    *,
    model: Any,
    processor: Any,
    cfg: Mapping[str, Any],
    base_inputs: Mapping[str, torch.Tensor],
    instruction: str,
    device: str,
    seed: int,
    action_dim: int,
    clean_action: np.ndarray,
    target_token_id: int,
    margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, torch.Tensor]]:
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=int(seed),
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    x = base_inputs["pixel_values"]
    seeds = rand_seed_schedule(int(seed), count=int(cfg["controls"]["rand20_count"]))
    candidate_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    inputs_by_id: dict[int, Mapping[str, torch.Tensor]] = {}
    for idx, cand_seed in enumerate(seeds):
        delta = sample_processor_delta(x.shape, epsilon=float(cfg["attack_optimizer"]["epsilon"]), seed=int(cand_seed), dtype=torch.float32, device=x.device)
        projected, corrections = project_and_cast_processor_values(x, delta, epsilon=float(cfg["attack_optimizer"]["epsilon"]), candidate_is_delta=True)
        cand_inputs = {"input_ids": base_inputs["input_ids"], "pixel_values": projected.detach()}
        stats = surrogate_stats_from_generated_prefix(
            adapter,
            cand_inputs["input_ids"],
            cand_inputs["pixel_values"],
            action_dim=int(action_dim),
            target_token_id=int(target_token_id),
            margin=float(margin),
        )
        score = float(stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"]))
        scores.append(score)
        inputs_by_id[idx] = cand_inputs
        candidate_rows.append(
            {
                "condition": "RAND20",
                "candidate_id": idx,
                "candidate_seed": int(cand_seed),
                "selection_metric": str(cfg["controls"].get("rand20_selection_metric", "surrogate_target31744_margin")),
                "surrogate_target31744_margin": score,
                "surrogate_target31744_best_competitor_margin": float(stats["target_minus_best_competitor_margin"]),
                "surrogate_target31744_logratio_margin": stats.get("target_minus_competitor_logsumexp_margin", ""),
                "delta_sha256": tensor_sha256((projected - x).detach().float()),
                "processor_input_sha256": tensor_sha256(projected.detach()),
                "budget_quantized_correction_count": int(corrections),
                "selected": 0,
            }
        )
    selected = select_best_surrogate_only(list(range(len(scores))), scores)
    candidate_rows[int(selected)]["selected"] = 1
    return {"condition": "RAND20", "selected_candidate": int(selected)}, candidate_rows, inputs_by_id[int(selected)]


def collect_condition_row(
    *,
    stage: str,
    commit: str,
    condition: str,
    attack_seed: int,
    output_dir: Path,
    official: Mapping[str, Any],
    surrogate: Mapping[str, Any] | None,
    route_status: str = "",
    result_class: str = "",
    arm_match: tuple[int, int] | None = None,
    processor_linf: float | None = None,
    stage_result: str = "",
    arm_selectivity_status: str = "",
) -> dict[str, Any]:
    stats = official["target_stats"]
    return {
        "stage": stage,
        "commit": commit,
        "condition": condition,
        "attack_seed": int(attack_seed),
        "route_status": route_status,
        "exact_7_tokens": True,
        "score_invariant_status": "PASS" if official["score_invariant"]["tie_aware_pass"] else "FAIL",
        "official_tokens": json.dumps(official["tokens"]),
        "official_gripper_token": official["gripper_token"],
        "official_target31744_score": stats["target_token_score"],
        "official_best_competitor_token": stats["best_competitor_token_id"],
        "official_best_competitor_score": stats["best_competitor_score"],
        "official_target31744_margin": stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"]),
        "official_target31744_best_competitor_margin": stats["target_minus_best_competitor_margin"],
        "official_target31744_logratio_margin": stats.get("target_minus_competitor_logsumexp_margin", ""),
        "surrogate_target31744_margin": "" if surrogate is None else surrogate.get("target_objective_margin", surrogate["target_minus_best_competitor_margin"]),
        "surrogate_target31744_best_competitor_margin": "" if surrogate is None else surrogate["target_minus_best_competitor_margin"],
        "surrogate_target31744_logratio_margin": "" if surrogate is None else surrogate.get("target_minus_competitor_logsumexp_margin", ""),
        "arm_prefix_match_count": "" if arm_match is None else int(arm_match[0]),
        "arm_prefix_match_denominator": "" if arm_match is None else int(arm_match[1]),
        "processor_linf": "" if processor_linf is None else float(processor_linf),
        "condition_result": result_class,
        "arm_selectivity_status": arm_selectivity_status,
        "stage_result": stage_result,
        "output_dir": str(output_dir),
    }


def arm_prefix_match_count(candidate_arm: list[int], clean_arm: list[int]) -> tuple[int, int]:
    n = min(len(candidate_arm), len(clean_arm))
    return sum(int(candidate_arm[i] == clean_arm[i]) for i in range(n)), n


def select_hard_feasible_candidate(
    rows: list[dict[str, Any]],
    *,
    arm_gate_min_match_count: int,
    target_token_id: int,
) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if int(row.get("arm_prefix_match_count", 0) or 0) >= int(arm_gate_min_match_count)
        and int(row.get("official_gripper_token", -1) or -1) == int(target_token_id)
        and str(row.get("score_invariant_status", "")) == "PASS"
    ]
    if not feasible:
        return None
    return sorted(
        feasible,
        key=lambda row: (
            -float(row.get("official_target31744_margin", "-inf")),
            float(row.get("processor_linf", "inf")),
            int(row.get("candidate_id", 10**9)),
        ),
    )[0]


def candidate_row_from_official(
    *,
    stage: str,
    commit: str,
    condition: str,
    attack_seed: int,
    candidate_id: int,
    candidate_source: str,
    official: Mapping[str, Any],
    clean_arm_prefix: list[int],
    processor_linf: float,
    delta_sha256: str,
    processor_input_sha256: str,
    budget_quantized_correction_count: int,
    candidate_seed: int | str = "",
    surrogate_target_margin: float | str = "",
) -> dict[str, Any]:
    stats = official["target_stats"]
    arm_match, arm_den = arm_prefix_match_count([int(x) for x in official["arm_prefix"]], clean_arm_prefix)
    return {
        "stage": stage,
        "commit": commit,
        "condition": condition,
        "attack_seed": int(attack_seed),
        "candidate_id": int(candidate_id),
        "candidate_source": candidate_source,
        "candidate_seed": candidate_seed,
        "official_tokens": json.dumps(official["tokens"]),
        "official_gripper_token": official["gripper_token"],
        "official_target31744_score": stats["target_token_score"],
        "official_best_competitor_token": stats["best_competitor_token_id"],
        "official_best_competitor_score": stats["best_competitor_score"],
        "official_target31744_margin": stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"]),
        "official_target31744_best_competitor_margin": stats["target_minus_best_competitor_margin"],
        "official_target31744_logratio_margin": stats.get("target_minus_competitor_logsumexp_margin", ""),
        "surrogate_target31744_margin": surrogate_target_margin,
        "arm_prefix_match_count": arm_match,
        "arm_prefix_match_denominator": arm_den,
        "score_invariant_status": "PASS" if official["score_invariant"]["tie_aware_pass"] else "FAIL",
        "processor_linf": float(processor_linf),
        "delta_sha256": delta_sha256,
        "processor_input_sha256": processor_input_sha256,
        "budget_quantized_correction_count": int(budget_quantized_correction_count),
        "feasible": 0,
        "selected": 0,
        "selection_reason": "",
    }


def trajectory_candidate_rows_from_debug(
    *,
    model: Any,
    debug: Mapping[str, Any],
    stage: str,
    commit: str,
    condition: str,
    attack_seed: int,
    action_dim: int,
    unnorm_key: str,
    target_token_id: int,
    margin: float,
    tolerance: float,
    objective: str,
    clean_arm_prefix: list[int],
) -> list[dict[str, Any]]:
    candidates = debug.get("trajectory_candidate_inputs")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{condition} missing trajectory_candidate_inputs")
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, Mapping):
            raise RuntimeError(f"{condition} has malformed trajectory candidate")
        official = official_decode(
            model,
            {"input_ids": cand["input_ids"], "pixel_values": cand["pixel_values"]},
            action_dim=action_dim,
            unnorm_key=unnorm_key,
            target_token_id=target_token_id,
            margin=margin,
            tolerance=tolerance,
            objective=objective,
        )
        rows.append(
            candidate_row_from_official(
                stage=stage,
                commit=commit,
                condition=condition,
                attack_seed=attack_seed,
                candidate_id=int(cand["candidate_index"]),
                candidate_source=str(cand.get("candidate_source", "")),
                official=official,
                clean_arm_prefix=clean_arm_prefix,
                processor_linf=float(cand.get("pixel_budget_adv_inputs_linf", 0.0) or 0.0),
                delta_sha256=str(cand.get("delta_sha256", "")),
                processor_input_sha256=str(cand.get("processor_input_sha256", "")),
                budget_quantized_correction_count=int(cand.get("pixel_budget_quantized_correction_count", 0) or 0),
            )
        )
    return rows


def run_rand21_official_candidates(
    *,
    model: Any,
    processor: Any,
    cfg: Mapping[str, Any],
    base_inputs: Mapping[str, torch.Tensor],
    device: str,
    seed: int,
    action_dim: int,
    clean_arm_prefix: list[int],
    target_token_id: int,
    margin: float,
    tolerance: float,
    stage: str,
    commit: str,
) -> list[dict[str, Any]]:
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=int(seed),
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    x = base_inputs["pixel_values"]
    count = int(cfg["controls"].get("rand21_count", 21))
    seeds = rand_seed_schedule(int(seed), count=count)
    rows: list[dict[str, Any]] = []
    for idx, cand_seed in enumerate(seeds):
        delta = sample_processor_delta(x.shape, epsilon=float(cfg["attack_optimizer"]["epsilon"]), seed=int(cand_seed), dtype=torch.float32, device=x.device)
        projected, corrections = project_and_cast_processor_values(x, delta, epsilon=float(cfg["attack_optimizer"]["epsilon"]), candidate_is_delta=True)
        cand_inputs = {"input_ids": base_inputs["input_ids"], "pixel_values": projected.detach()}
        stats = surrogate_stats_from_generated_prefix(
            adapter,
            cand_inputs["input_ids"],
            cand_inputs["pixel_values"],
            action_dim=int(action_dim),
            target_token_id=int(target_token_id),
            margin=float(margin),
        )
        official = official_decode(
            model,
            cand_inputs,
            action_dim=action_dim,
            unnorm_key=cfg["model"]["unnorm_key"],
            target_token_id=target_token_id,
            margin=margin,
            tolerance=tolerance,
            objective=str(cfg["attack_optimizer"]["objective"]),
        )
        rows.append(
            candidate_row_from_official(
                stage=stage,
                commit=commit,
                condition="RAND21_SELECTIVE",
                attack_seed=seed,
                candidate_id=idx,
                candidate_source="processor_random",
                candidate_seed=int(cand_seed),
                official=official,
                clean_arm_prefix=clean_arm_prefix,
                processor_linf=float((projected.float() - x.float()).abs().max().cpu()),
                delta_sha256=tensor_sha256((projected - x).detach().float()),
                processor_input_sha256=tensor_sha256(projected.detach()),
                budget_quantized_correction_count=int(corrections),
                surrogate_target_margin=float(stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"])),
            )
        )
    return rows


def run_capture_input(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model(cfg["model"]["path"], args.model_gpu_device_id)
    model_dtype = next(model.parameters()).dtype
    action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))

    bm = benchmark.get_benchmark_dict()
    task_suite = bm[cfg["input"]["suite"]]()
    task_obj = task_suite.get_task(TASK_IDX[cfg["input"]["task"]])
    init_states = task_suite.get_task_init_states(TASK_IDX[cfg["input"]["task"]])
    bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env, obs = build_v4_exact_env(bddl_file, int(args.render_gpu_device_id), int(args.max_steps), int(args.num_steps_wait))
    obs = env.set_init_state(init_states[int(cfg["input"]["state_id"])])
    env, obs = apply_dummy_wait(env, obs, int(args.num_steps_wait))
    instruction = task_obj.language

    clean_gen_at_step = None
    clean_action_at_step = None
    raw_at_step = None
    for step in range(int(cfg["input"]["absolute_step"]) + 1):
        raw = np.asarray(obs["agentview_image"]).copy()
        action, _scores, _dt, gen = decode_with_scores(
            model,
            processor,
            device,
            raw,
            instruction,
            cfg["model"]["unnorm_key"],
            8,
            libero_official_preprocess=bool(cfg["preprocess"]["libero_official_preprocess"]),
            center_crop=bool(cfg["preprocess"]["center_crop"]),
            resize_size=int(cfg["preprocess"]["resize_size"]),
            libero_preprocess_backend=str(cfg["preprocess"]["libero_preprocess_backend"]),
        )
        if step == int(cfg["input"]["absolute_step"]):
            raw_at_step = raw
            clean_gen_at_step = gen
            clean_action_at_step = np.asarray(action, dtype=np.float32)
            break
        obs, _reward, _done, _info = env.step(postprocess_openvla_action_for_libero(action, enabled=True))
    env.close()

    if raw_at_step is None:
        raise RuntimeError("failed to capture requested step")
    npy_path = output_dir / "raw_agentview_step78.npy"
    png_path = output_dir / "raw_agentview_step78.png"
    pt_path = output_dir / "processor_inputs_step78.pt"
    gen_path = output_dir / "clean_generation_step78.json"
    np.save(npy_path, raw_at_step)
    Image.fromarray(raw_at_step).save(png_path)
    proc_inputs = preprocess_raw_image(raw_at_step, processor, instruction, cfg, device, model_dtype)
    torch.save({k: v.detach().cpu() for k, v in proc_inputs.items()}, pt_path)
    official = official_decode(
        model,
        proc_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    write_json(
        gen_path,
        {
            "instruction": instruction,
            "clean_action": clean_action_at_step.tolist(),
            "clean_exact_7_tokens": official["tokens"],
            "official": official,
        },
    )
    manifest_row = {
        "source_commit": git_value(["rev-parse", "HEAD"]),
        "source_output_dir": str(output_dir),
        "task": cfg["input"]["task"],
        "state": cfg["input"]["state_id"],
        "step": cfg["input"]["absolute_step"],
        "instruction": instruction,
        "raw_image_shape": list(raw_at_step.shape),
        "raw_image_dtype": str(raw_at_step.dtype),
        "raw_image_sha256": hashlib.sha256(np.ascontiguousarray(raw_at_step).tobytes()).hexdigest(),
        "processed_tensor_shape": list(proc_inputs["pixel_values"].shape),
        "processed_tensor_dtype": str(proc_inputs["pixel_values"].dtype),
        "processed_tensor_sha256": tensor_sha256(proc_inputs["pixel_values"].detach().cpu()),
        "prompt": prompt(instruction),
        "prompt_token_ids_sha256": tensor_sha256(proc_inputs["input_ids"].detach().cpu()),
        "unnorm_key": cfg["model"]["unnorm_key"],
        "center_crop": cfg["preprocess"]["center_crop"],
        "resize_size": cfg["preprocess"]["resize_size"],
        "preprocess_backend": cfg["preprocess"]["libero_preprocess_backend"],
        "model_fingerprint": json.dumps(model_fingerprint(model), sort_keys=True),
        "clean_exact_7_tokens": json.dumps(official["tokens"]),
        "clean_arm_prefix": json.dumps(official["arm_prefix"]),
        "clean_gripper_token": official["gripper_token"],
        "clean_score_argmax": official["score_invariant"]["argmax_token"],
        "status": "CAPTURED",
    }
    write_csv(output_dir / "m3_step78_input_manifest.csv", [manifest_row], list(manifest_row.keys()))
    print(json.dumps({"status": "CAPTURED", "output_dir": str(output_dir), "clean_gripper_token": official["gripper_token"]}, indent=2))


def load_frozen_input(input_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path = input_dir / "raw_agentview_step78.npy"
    gen_path = input_dir / "clean_generation_step78.json"
    if not raw_path.exists() or not gen_path.exists():
        raise FileNotFoundError(f"frozen input missing raw npy or clean_generation json in {input_dir}")
    return np.load(raw_path), json.loads(gen_path.read_text(encoding="utf-8"))


def mark_and_select_candidates(
    rows: list[dict[str, Any]],
    *,
    arm_gate_min_match_count: int,
    target_token_id: int,
) -> dict[str, Any] | None:
    selected = select_hard_feasible_candidate(
        rows,
        arm_gate_min_match_count=arm_gate_min_match_count,
        target_token_id=target_token_id,
    )
    for row in rows:
        feasible = (
            int(row.get("arm_prefix_match_count", 0) or 0) >= int(arm_gate_min_match_count)
            and int(row.get("official_gripper_token", -1) or -1) == int(target_token_id)
            and str(row.get("score_invariant_status", "")) == "PASS"
        )
        row["feasible"] = int(feasible)
        if selected is not None and row is selected:
            row["selected"] = 1
            row["selection_reason"] = "max_official_margin_then_min_linf_then_earliest"
        elif feasible:
            row["selection_reason"] = "feasible_not_selected"
        else:
            row["selection_reason"] = "not_feasible"
    return selected


def selected_condition_row(
    *,
    selected: Mapping[str, Any] | None,
    stage: str,
    commit: str,
    condition: str,
    attack_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if selected is None:
        return {
            "stage": stage,
            "commit": commit,
            "condition": condition,
            "attack_seed": int(attack_seed),
            "selected_candidate_id": "",
            "condition_result": "NO_FEASIBLE_CANDIDATE",
            "official_tokens": "",
            "official_gripper_token": "",
            "official_target31744_margin": "",
            "arm_prefix_match_count": "",
            "arm_prefix_match_denominator": "",
            "processor_linf": "",
            "output_dir": str(output_dir),
        }
    return {
        "stage": stage,
        "commit": commit,
        "condition": condition,
        "attack_seed": int(attack_seed),
        "selected_candidate_id": selected["candidate_id"],
        "condition_result": "SELECTED_FEASIBLE_CANDIDATE",
        "official_tokens": selected["official_tokens"],
        "official_gripper_token": selected["official_gripper_token"],
        "official_target31744_margin": selected["official_target31744_margin"],
        "arm_prefix_match_count": selected["arm_prefix_match_count"],
        "arm_prefix_match_denominator": selected["arm_prefix_match_denominator"],
        "processor_linf": selected["processor_linf"],
        "output_dir": str(output_dir),
    }


def run_v4_hard_selection(
    *,
    output_dir: Path,
    cfg: Mapping[str, Any],
    model: Any,
    processor: Any,
    raw_image: np.ndarray,
    instruction: str,
    clean_action: np.ndarray,
    clean_gen: Any,
    base_inputs: Mapping[str, torch.Tensor],
    clean_official: Mapping[str, Any],
    device: str,
    action_dim: int,
    seed: int,
) -> None:
    commit = git_value(["rev-parse", "HEAD"])
    stage = str(cfg.get("stage", "M3_STEP78_TRUE_PGD_LOGRATIO_ARM_V4_HARD_FEASIBLE_SELECTION"))
    target_token_id = int(cfg["attack_optimizer"]["target_token_id"])
    margin = float(cfg["attack_optimizer"]["gripper_margin"])
    tolerance = float(cfg["gates"]["score_tie_tolerance"])
    arm_gate = int(cfg["gates"].get("arm_prefix_min_match_count", cfg["attack_optimizer"].get("arm_gate_min_match_count", 5)))
    clean_arm_prefix = [int(x) for x in clean_official["arm_prefix"]]

    true_info, _true_final_inputs = run_true_pgd_condition(
        name="TRUE_PGD_TRAJECTORY21_SELECTIVE",
        model=model,
        processor=processor,
        cfg=cfg,
        raw_image=raw_image,
        instruction=instruction,
        clean_action=clean_action,
        clean_gen=clean_gen,
        device=device,
        seed=seed,
        gradient_transform="none",
    )
    true_debug = true_info["debug"]
    true_rows = trajectory_candidate_rows_from_debug(
        model=model,
        debug=true_debug,
        stage=stage,
        commit=commit,
        condition="TRUE_PGD_TRAJECTORY21_SELECTIVE",
        attack_seed=seed,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=target_token_id,
        margin=margin,
        tolerance=tolerance,
        objective=str(cfg["attack_optimizer"]["objective"]),
        clean_arm_prefix=clean_arm_prefix,
    )
    true_selected = mark_and_select_candidates(true_rows, arm_gate_min_match_count=arm_gate, target_token_id=target_token_id)

    rand_rows = run_rand21_official_candidates(
        model=model,
        processor=processor,
        cfg=cfg,
        base_inputs=base_inputs,
        device=device,
        seed=seed,
        action_dim=action_dim,
        clean_arm_prefix=clean_arm_prefix,
        target_token_id=target_token_id,
        margin=margin,
        tolerance=tolerance,
        stage=stage,
        commit=commit,
    )
    rand_selected = mark_and_select_candidates(rand_rows, arm_gate_min_match_count=arm_gate, target_token_id=target_token_id)

    shuffled_info, _shuffled_final_inputs = run_true_pgd_condition(
        name="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
        model=model,
        processor=processor,
        cfg=cfg,
        raw_image=raw_image,
        instruction=instruction,
        clean_action=clean_action,
        clean_gen=clean_gen,
        device=device,
        seed=seed,
        gradient_transform=str(cfg["controls"]["shuffled_grad_mode"]),
    )
    shuffled_debug = shuffled_info["debug"]
    shuffled_rows = trajectory_candidate_rows_from_debug(
        model=model,
        debug=shuffled_debug,
        stage=stage,
        commit=commit,
        condition="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
        attack_seed=seed,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=target_token_id,
        margin=margin,
        tolerance=tolerance,
        objective=str(cfg["attack_optimizer"]["objective"]),
        clean_arm_prefix=clean_arm_prefix,
    )
    shuffled_selected = mark_and_select_candidates(shuffled_rows, arm_gate_min_match_count=arm_gate, target_token_id=target_token_id)

    condition_rows = [
        selected_condition_row(
            selected=true_selected,
            stage=stage,
            commit=commit,
            condition="TRUE_PGD_TRAJECTORY21_SELECTIVE",
            attack_seed=seed,
            output_dir=output_dir,
        ),
        selected_condition_row(
            selected=rand_selected,
            stage=stage,
            commit=commit,
            condition="RAND21_SELECTIVE",
            attack_seed=seed,
            output_dir=output_dir,
        ),
        selected_condition_row(
            selected=shuffled_selected,
            stage=stage,
            commit=commit,
            condition="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
            attack_seed=seed,
            output_dir=output_dir,
        ),
    ]
    all_candidates = true_rows + rand_rows + shuffled_rows

    result_class = "UNCLASSIFIED"
    if true_selected is None:
        result_class = "NO_FEASIBLE_PGD_CANDIDATE"
    elif rand_selected is not None and float(true_selected["official_target31744_margin"]) <= float(rand_selected["official_target31744_margin"]):
        result_class = "RANDOM_NOT_BEATEN"
    elif shuffled_selected is not None and float(true_selected["official_target31744_margin"]) <= float(shuffled_selected["official_target31744_margin"]):
        result_class = "SHUFFLED_NOT_BEATEN"
    else:
        result_class = "FULL_SELECTIVE_V4_SEED_PASS"
    for row in condition_rows:
        row["stage_result"] = result_class

    route_rows = []
    for condition, debug in [
        ("TRUE_PGD_TRAJECTORY21_SELECTIVE", true_debug),
        ("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", shuffled_debug),
    ]:
        route_rows.append({k: debug.get(k, "") for k in [
            "requested_method", "resolved_adapter_class", "strict_route", "allow_fallback", "fallback_used",
            "fallback_reason", "requested_objective", "resolved_objective", "target_token_id",
            "target_execution_class", "num_backwards", "num_loss_forwards", "num_generation_forwards",
            "adv_inputs_present", "x_adv_is_none", "action_adv_is_none", "pixel_budget_adv_inputs_linf",
            "trajectory_candidate_count",
        ]} | {"stage": stage, "commit": commit, "condition": condition, "attack_seed": seed, "status": "PASS"})

    write_csv(output_dir / "m3_v4_selected_results.csv", condition_rows, list(condition_rows[0].keys()))
    write_csv(output_dir / "m3_v4_candidate_audit.csv", all_candidates, list(all_candidates[0].keys()))
    write_csv(output_dir / "m3_v4_route_audit.csv", route_rows, list(route_rows[0].keys()))
    write_json(output_dir / "m3_v4_debug.json", {"true_pgd": true_debug, "shuffled_grad": shuffled_debug})
    print(json.dumps({"status": "V4_CANARY_COMPLETE", "result_class": result_class, "output_dir": str(output_dir)}, indent=2))


def run_preflight_or_canary(args: argparse.Namespace, cfg: Mapping[str, Any], *, canary: bool) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_image, clean_json = load_frozen_input(Path(args.input_dir))
    model, processor, device = load_model(cfg["model"]["path"], args.model_gpu_device_id)
    model_dtype = next(model.parameters()).dtype
    action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))
    instruction = str(clean_json["instruction"])
    clean_action = np.asarray(clean_json["clean_action"], dtype=np.float32)
    base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)
    clean_official = official_decode(
        model,
        base_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    clean_gen = type("CleanGen", (), {})()
    clean_gen.sequences = torch.tensor(
        [base_inputs["input_ids"][0].detach().cpu().tolist() + clean_official["tokens"]],
        dtype=torch.long,
        device=base_inputs["input_ids"].device,
    )
    clean_gen.scores = []

    seed = int(args.attack_seed)
    epsilon = float(cfg["attack_optimizer"]["epsilon"])
    delta0_inputs = make_delta0_inputs(
        adv_inputs=base_inputs,
        seed=seed,
        epsilon=epsilon,
        random_start=bool(cfg["attack_optimizer"]["random_start"]),
    )
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=seed,
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    clean_surrogate = surrogate_stats_from_generated_prefix(
        adapter,
        base_inputs["input_ids"],
        base_inputs["pixel_values"],
        action_dim=action_dim,
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
    )
    delta0_surrogate = surrogate_stats_from_generated_prefix(
        adapter,
        delta0_inputs["input_ids"],
        delta0_inputs["pixel_values"],
        action_dim=action_dim,
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
    )
    delta0_official = official_decode(
        model,
        delta0_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    clean_preflight = compare_surrogate_official(clean_surrogate, clean_official, tolerance=float(args.preflight_tolerance))
    delta0_preflight = compare_surrogate_official(delta0_surrogate, delta0_official, tolerance=float(args.preflight_tolerance))
    write_json(
        output_dir / "m3_step78_zero_step_preflight.json",
        {
            "clean_status": clean_preflight,
            "delta0_status": delta0_preflight,
            "clean_surrogate": clean_surrogate,
            "clean_official": clean_official,
            "delta0_surrogate": delta0_surrogate,
            "delta0_official": delta0_official,
        },
    )
    if clean_preflight == "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH" or delta0_preflight == "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH":
        print(json.dumps({"status": "PREFLIGHT_MISMATCH", "clean": clean_preflight, "delta0": delta0_preflight}, indent=2))
        return
    if not canary:
        print(json.dumps({"status": "PREFLIGHT_PASS", "clean": clean_preflight, "delta0": delta0_preflight}, indent=2))
        return
    if cfg.get("conditions") == V4_CONDITIONS:
        run_v4_hard_selection(
            output_dir=output_dir,
            cfg=cfg,
            model=model,
            processor=processor,
            raw_image=raw_image,
            instruction=instruction,
            clean_action=clean_action,
            clean_gen=clean_gen,
            base_inputs=base_inputs,
            clean_official=clean_official,
            device=device,
            action_dim=action_dim,
            seed=seed,
        )
        return

    condition_rows = []
    route_rows = []
    candidate_rows = []
    commit = git_value(["rev-parse", "HEAD"])
    stage = str(cfg.get("stage", "M3_STEP78_TRUE_PGD_CANARY"))
    condition_rows.append(
        collect_condition_row(
            stage=stage,
            commit=commit,
            condition="CLEAN",
            attack_seed=seed,
            output_dir=output_dir,
            official=clean_official,
            surrogate=clean_surrogate,
            result_class="CONTEXT",
        )
    )
    condition_rows.append(
        collect_condition_row(
            stage=stage,
            commit=commit,
            condition="PGD_DELTA0",
            attack_seed=seed,
            output_dir=output_dir,
            official=delta0_official,
            surrogate=delta0_surrogate,
            result_class="CONTROL",
            processor_linf=float((delta0_inputs["pixel_values"].float() - base_inputs["pixel_values"].float()).abs().max().cpu()),
        )
    )

    true_info, true_inputs = run_true_pgd_condition(
        name="TRUE_PGD_FINAL",
        model=model,
        processor=processor,
        cfg=cfg,
        raw_image=raw_image,
        instruction=instruction,
        clean_action=clean_action,
        clean_gen=clean_gen,
        device=device,
        seed=seed,
        gradient_transform="none",
    )
    true_debug = true_info["debug"]
    true_official = official_decode(
        model,
        true_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    true_surrogate = true_debug.get("generated_prefix_gripper_stats_final", {})
    arm_match = (int(true_debug.get("arm_prefix_match_count", 0) or 0), int(true_debug.get("arm_prefix_match_denominator", 0) or 0))
    condition_rows.append(
        collect_condition_row(
            stage=stage,
            commit=commit,
            condition="TRUE_PGD_FINAL",
            attack_seed=seed,
            output_dir=output_dir,
            official=true_official,
            surrogate=true_surrogate,
            route_status="PASS",
            result_class="TRUE_PGD_CONDITION_VALID",
            arm_match=arm_match,
            processor_linf=float(true_debug.get("pixel_budget_adv_inputs_linf", 0.0) or 0.0),
        )
    )
    route_rows.append({k: true_debug.get(k, "") for k in [
        "requested_method", "resolved_adapter_class", "strict_route", "allow_fallback", "fallback_used",
        "fallback_reason", "requested_objective", "resolved_objective", "target_token_id",
        "target_execution_class", "num_backwards", "num_loss_forwards", "num_generation_forwards",
        "adv_inputs_present", "x_adv_is_none", "action_adv_is_none", "pixel_budget_adv_inputs_linf",
    ]} | {"stage": stage, "commit": commit, "condition": "TRUE_PGD_FINAL", "attack_seed": seed, "status": "PASS"})

    rand_info, rand_candidates, rand_inputs = run_rand20(
        model=model,
        processor=processor,
        cfg=cfg,
        base_inputs=base_inputs,
        instruction=instruction,
        device=device,
        seed=seed,
        action_dim=action_dim,
        clean_action=clean_action,
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
    )
    rand_official = official_decode(
        model,
        rand_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    rand_selected = rand_candidates[int(rand_info["selected_candidate"])]
    condition_rows.append(
        collect_condition_row(
            stage=stage,
            commit=commit,
            condition="RAND20",
            attack_seed=seed,
            output_dir=output_dir,
            official=rand_official,
            surrogate={"target_minus_best_competitor_margin": rand_selected["surrogate_target31744_margin"]},
            result_class="CONTROL",
        )
    )
    for row in rand_candidates:
        candidate_rows.append({"stage": stage, "commit": commit, "attack_seed": seed, **row})

    shuffled_info, shuffled_inputs = run_true_pgd_condition(
        name="SHUFFLED_GRAD_PGD20",
        model=model,
        processor=processor,
        cfg=cfg,
        raw_image=raw_image,
        instruction=instruction,
        clean_action=clean_action,
        clean_gen=clean_gen,
        device=device,
        seed=seed,
        gradient_transform=str(cfg["controls"]["shuffled_grad_mode"]),
    )
    shuffled_debug = shuffled_info["debug"]
    shuffled_official = official_decode(
        model,
        shuffled_inputs,
        action_dim=action_dim,
        unnorm_key=cfg["model"]["unnorm_key"],
        target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
        margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    condition_rows.append(
        collect_condition_row(
            stage=stage,
            commit=commit,
            condition="SHUFFLED_GRAD_PGD20",
            attack_seed=seed,
            output_dir=output_dir,
            official=shuffled_official,
            surrogate=shuffled_debug.get("generated_prefix_gripper_stats_final", {}),
            route_status="PASS",
            result_class="CONTROL",
            arm_match=(int(shuffled_debug.get("arm_prefix_match_count", 0) or 0), int(shuffled_debug.get("arm_prefix_match_denominator", 0) or 0)),
            processor_linf=float(shuffled_debug.get("pixel_budget_adv_inputs_linf", 0.0) or 0.0),
        )
    )

    write_csv(output_dir / "m3_step78_condition_results.csv", condition_rows, list(condition_rows[0].keys()))
    route_fields = [
        "stage", "commit", "condition", "attack_seed", "requested_method", "resolved_adapter_class",
        "strict_route", "allow_fallback", "fallback_used", "fallback_reason", "requested_objective",
        "resolved_objective", "target_token_id", "target_execution_class", "num_backwards",
        "num_loss_forwards", "num_generation_forwards", "adv_inputs_present", "x_adv_is_none",
        "action_adv_is_none", "pixel_budget_adv_inputs_linf", "status",
    ]
    write_csv(output_dir / "m3_step78_route_audit.csv", route_rows, route_fields)
    if candidate_rows:
        write_csv(output_dir / "m3_step78_candidate_controls.csv", candidate_rows, list(candidate_rows[0].keys()))
    write_json(output_dir / "m3_step78_canary_debug.json", {"true_pgd": true_debug, "shuffled_grad": shuffled_debug})
    print(json.dumps({"status": "CANARY_COMPLETE_UNCLASSIFIED", "output_dir": str(output_dir)}, indent=2))


def write_manifest(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = ""
    row = {
        "stage": cfg.get("stage", "M3_STEP78_TRUE_PGD_CANARY"),
        "commit": git_value(["rev-parse", "HEAD"]),
        "dirty_status": git_value(["status", "--porcelain"]),
        "config_path": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "output_dir": str(output_dir),
        "model_path": cfg["model"]["path"],
        "model_fingerprint": "",
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_query": "",
        "command_line": " ".join(sys.argv),
        "status": args.mode,
    }
    write_csv(output_dir / "m3_step78_manifest.csv", [row], list(row.keys()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_step78_true_pgd_31744.yaml"))
    ap.add_argument("--mode", choices=["capture_input", "preflight_zero_step", "canary", "canary_v4"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--attack_seed", type=int, default=80)
    ap.add_argument("--model_gpu_device_id", type=int, default=-1)
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--preflight_tolerance", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if cfg.get("conditions") != MAIN_CONDITIONS and cfg.get("conditions") != V4_CONDITIONS:
        raise SystemExit(f"config conditions must be {MAIN_CONDITIONS} or {V4_CONDITIONS}")
    write_manifest(args, cfg)
    if args.mode == "capture_input":
        run_capture_input(args, cfg)
    elif args.mode == "preflight_zero_step":
        if not args.input_dir:
            raise SystemExit("--input_dir is required for preflight_zero_step")
        run_preflight_or_canary(args, cfg, canary=False)
    elif args.mode in {"canary", "canary_v4"}:
        if not args.input_dir:
            raise SystemExit("--input_dir is required for canary")
        run_preflight_or_canary(args, cfg, canary=True)


if __name__ == "__main__":
    main()
