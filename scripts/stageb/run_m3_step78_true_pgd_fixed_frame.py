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
from datetime import datetime, timezone
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
PANEL_MAIN_FRAMES = [70, 72, 74, 76, 80, 82, 84, 86]
PANEL_POSITIVE_CONTROL_FRAME = 78
PANEL_ALL_CAPTURE_FRAMES = [70, 72, 74, 76, 78, 80, 82, 84, 86]
PANEL_FROZEN_V4_CONFIG_SHA256 = "2dcef93bf2decf742e0c98f321267ae665b57890f3ab03088dfda3686ae8a2a8"
PANEL_FROZEN_OBJECTIVE = "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
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


def dirty_status_value() -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "GIT_STATUS_UNAVAILABLE"
    if not status:
        return "CLEAN"
    return "DIRTY:" + status.replace("\n", "\\n")


def gpu_query_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "NVIDIA_SMI_EMPTY"
    except Exception:
        return "NVIDIA_SMI_UNAVAILABLE"


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


def update_manifest_model_fingerprint(output_dir: Path, fingerprint: Mapping[str, Any]) -> None:
    update_manifest_model_fingerprint_text(output_dir, json.dumps(dict(fingerprint), sort_keys=True))


def update_manifest_model_fingerprint_text(output_dir: Path, fingerprint_text: str) -> None:
    path = output_dir / "m3_step78_manifest.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows or "model_fingerprint" not in fieldnames:
        return
    rows[0]["model_fingerprint"] = fingerprint_text
    write_csv(path, rows, fieldnames)


def write_artifact_hash_manifest(output_dir: Path, filename: str = "m3_artifact_hash_manifest.csv") -> None:
    rows = []
    for path in sorted(output_dir.glob("*")):
        if not path.is_file() or path.name == filename:
            continue
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if rows:
        write_csv(output_dir / filename, rows, ["file", "size_bytes", "sha256"])


def write_recursive_artifact_hash_manifest(
    output_dir: Path,
    filename: str = "m3_recursive_artifact_hash_manifest.csv",
) -> None:
    rows = []
    manifest_path = output_dir / filename
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.resolve() == manifest_path.resolve():
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(manifest_path, rows, ["relative_path", "size_bytes", "sha256"])


def verify_recursive_artifact_hash_manifest(
    output_dir: Path,
    filename: str = "m3_recursive_artifact_hash_manifest.csv",
) -> None:
    manifest_path = output_dir / filename
    if not manifest_path.exists():
        raise RuntimeError(f"recursive artifact manifest missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = output_dir / row["relative_path"]
        if not path.exists():
            raise RuntimeError(f"artifact manifest entry missing on disk: {path}")
        if str(path.stat().st_size) != str(row["size_bytes"]):
            raise RuntimeError(f"artifact size mismatch: {path}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"artifact sha256 mismatch: {path}")


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


def frame_filename(stem: str, step: int, suffix: str) -> str:
    return f"{stem}_step{int(step)}.{suffix}"


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


def clean_frame_eligibility(official: Mapping[str, Any]) -> dict[str, Any]:
    token = int(official.get("gripper_token", -1))
    tokens = official.get("tokens", [])
    invariant = official.get("score_invariant", {})
    if len(tokens) != 7:
        return {"status": "CLEAN_CONTEXT_INELIGIBLE", "reason": "clean_not_exact_7_tokens", "clean_gripper_token": token}
    if not bool(invariant.get("tie_aware_pass", False)):
        return {"status": "CLEAN_CONTEXT_INELIGIBLE", "reason": "clean_score_invariant_fail", "clean_gripper_token": token}
    if token == TARGET_31744:
        return {"status": "CLEAN_ALREADY_TARGET", "reason": "clean_already_31744", "clean_gripper_token": token}
    if token != 31872:
        return {"status": "CLEAN_NOT_CLOSE", "reason": "clean_not_31872", "clean_gripper_token": token}
    return {"status": "CLEAN_ELIGIBLE", "reason": "clean_close_31872", "clean_gripper_token": token}


def _selected_is_feasible(row: Mapping[str, Any] | None, *, target_token_id: int = TARGET_31744, arm_gate: int = 5) -> bool:
    if row is None:
        return False
    if str(row.get("condition_result", "")) == "NO_FEASIBLE_CANDIDATE":
        return False
    return (
        int(row.get("official_gripper_token", -1) or -1) == int(target_token_id)
        and int(row.get("arm_prefix_match_count", 0) or 0) >= int(arm_gate)
    )


def frame_full_selective_status(
    *,
    true_row: Mapping[str, Any] | None,
    rand_row: Mapping[str, Any] | None,
    shuffled_row: Mapping[str, Any] | None,
    clean_status: str = "CLEAN_ELIGIBLE",
    infra_valid: bool = True,
    target_token_id: int = TARGET_31744,
    arm_gate: int = 5,
) -> dict[str, Any]:
    if not infra_valid:
        return {"status": "INFRA_INVALID", "full_pass": False, "rand_win": False, "shuffled_win": False}
    if clean_status != "CLEAN_ELIGIBLE":
        return {"status": clean_status, "full_pass": False, "rand_win": False, "shuffled_win": False}
    true_ok = _selected_is_feasible(true_row, target_token_id=target_token_id, arm_gate=arm_gate)
    rand_ok = _selected_is_feasible(rand_row, target_token_id=target_token_id, arm_gate=arm_gate)
    shuffled_ok = _selected_is_feasible(shuffled_row, target_token_id=target_token_id, arm_gate=arm_gate)
    if not true_ok:
        return {"status": "FRAME_FAIL_TRUE_INFEASIBLE", "full_pass": False, "rand_win": False, "shuffled_win": False}
    true_margin = float(true_row["official_target31744_margin"])
    rand_win = True if not rand_ok else true_margin > float(rand_row["official_target31744_margin"])
    shuffled_win = True if not shuffled_ok else true_margin > float(shuffled_row["official_target31744_margin"])
    if rand_win and shuffled_win:
        return {
            "status": "FRAME_FULL_SELECTIVE_PASS",
            "full_pass": True,
            "rand_win": True,
            "shuffled_win": True,
            "rand_paired_margin": "" if not rand_ok else true_margin - float(rand_row["official_target31744_margin"]),
            "shuffled_paired_margin": "" if not shuffled_ok else true_margin - float(shuffled_row["official_target31744_margin"]),
            "rand_control_status": "CONTROL_INFEASIBLE_AUTO_WIN" if not rand_ok else "PAIRED_FINITE",
            "shuffled_control_status": "CONTROL_INFEASIBLE_AUTO_WIN" if not shuffled_ok else "PAIRED_FINITE",
        }
    return {
        "status": "FRAME_FAIL_CONTROL_NOT_BEATEN",
        "full_pass": False,
        "rand_win": rand_win,
        "shuffled_win": shuffled_win,
        "rand_paired_margin": "" if not rand_ok else true_margin - float(rand_row["official_target31744_margin"]),
        "shuffled_paired_margin": "" if not shuffled_ok else true_margin - float(shuffled_row["official_target31744_margin"]),
        "rand_control_status": "CONTROL_INFEASIBLE_AUTO_WIN" if not rand_ok else "PAIRED_FINITE",
        "shuffled_control_status": "CONTROL_INFEASIBLE_AUTO_WIN" if not shuffled_ok else "PAIRED_FINITE",
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    vals = sorted(float(v) for v in values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def panel_aggregate_status(frame_rows: list[Mapping[str, Any]], *, total_main_frames: int = 8) -> dict[str, Any]:
    main = [r for r in frame_rows if bool(r.get("main_denominator", True))]
    main_frames = [int(r.get("frame", -1)) for r in main]
    ineligible = [r for r in main if str(r.get("frame_status")) in {"CLEAN_CONTEXT_INELIGIBLE", "CLEAN_ALREADY_TARGET", "CLEAN_NOT_CLOSE"}]
    infra_invalid = [r for r in main if str(r.get("frame_status")) == "INFRA_INVALID"]
    full_pass = [r for r in main if str(r.get("frame_status")) == "FRAME_FULL_SELECTIVE_PASS"]
    rand_margins = [float(r["rand_paired_margin"]) for r in main if r.get("rand_paired_margin") not in {"", None}]
    shuffled_margins = [float(r["shuffled_paired_margin"]) for r in main if r.get("shuffled_paired_margin") not in {"", None}]
    reasons = []
    if len(main) != int(total_main_frames):
        reasons.append("wrong_main_denominator_count")
    if main_frames != PANEL_MAIN_FRAMES or len(set(main_frames)) != len(main_frames):
        reasons.append("wrong_main_frame_set")
    if infra_invalid:
        reasons.append("infra_invalid_frame")
    if len(ineligible) > 1:
        reasons.append("too_many_clean_ineligible_frames")
    if len(full_pass) < 6:
        reasons.append("fewer_than_6_full_selective_pass_frames")
    if len(rand_margins) < 4:
        reasons.append("rand_paired_frames_below_4")
    if len(shuffled_margins) < 4:
        reasons.append("shuffled_paired_frames_below_4")
    rand_median = "" if len(rand_margins) < 4 else _median(rand_margins)
    shuffled_median = "" if len(shuffled_margins) < 4 else _median(shuffled_margins)
    if rand_median != "" and rand_median <= 0:
        reasons.append("rand_median_not_positive")
    if shuffled_median != "" and shuffled_median <= 0:
        reasons.append("shuffled_median_not_positive")
    return {
        "panel_status": "PANEL_SINGLE_SEED_PASS" if not reasons else "PANEL_SINGLE_SEED_FAIL",
        "full_pass_count": len(full_pass),
        "clean_ineligible_count": len(ineligible),
        "infra_invalid_count": len(infra_invalid),
        "rand_paired_count": len(rand_margins),
        "shuffled_paired_count": len(shuffled_margins),
        "rand_median_margin": rand_median,
        "shuffled_median_margin": shuffled_median,
        "failure_reasons": ";".join(reasons),
    }


def validate_panel_seed(seed: int) -> None:
    if int(seed) != 85:
        raise SystemExit("panel prereg permits exactly one first panel run with attack_seed=85")


def step78_parity_status(new_manifest: Mapping[str, Any], frozen_manifest: Mapping[str, Any]) -> str:
    keys = [
        "raw_image_sha256",
        "processed_tensor_sha256",
        "prompt_token_ids_sha256",
        "clean_exact_7_tokens",
        "clean_arm_prefix",
        "clean_gripper_token",
    ]
    for key in keys:
        if str(new_manifest.get(key, "")) != str(frozen_manifest.get(key, "")):
            return "POSITIVE_CONTROL_INPUT_MISMATCH"
    return "POSITIVE_CONTROL_INPUT_MATCH"


def parse_panel_steps(panel_steps: str | None) -> list[int]:
    if panel_steps and str(panel_steps).strip():
        steps = [int(x) for x in str(panel_steps).split(",") if x.strip()]
    else:
        steps = list(PANEL_ALL_CAPTURE_FRAMES)
    validate_panel_frame_set(steps, require_positive_control=True)
    return steps


def validate_panel_frame_set(steps: list[int], *, require_positive_control: bool) -> None:
    expected = PANEL_ALL_CAPTURE_FRAMES if require_positive_control else PANEL_MAIN_FRAMES
    if list(steps) != expected or len(set(steps)) != len(steps):
        raise SystemExit(f"panel frame set must be exactly {expected}")


def read_single_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row in {path}, found {len(rows)}")
    return dict(rows[0])


def validate_manifest_provenance(output_dir: Path) -> None:
    row = read_single_csv(output_dir / "m3_step78_manifest.csv")
    dirty = str(row.get("dirty_status", ""))
    if dirty != "CLEAN":
        raise RuntimeError(f"provenance invalid: dirty_status={dirty!r}")
    gpu_query = str(row.get("gpu_query", ""))
    if not gpu_query or gpu_query in {"NVIDIA_SMI_UNAVAILABLE", "NVIDIA_SMI_EMPTY"}:
        raise RuntimeError(f"provenance invalid: gpu_query={gpu_query!r}")
    fingerprint = str(row.get("model_fingerprint", ""))
    if not fingerprint or fingerprint == "PENDING_MODEL_LOAD":
        raise RuntimeError("provenance invalid: model_fingerprint missing")
    if not (output_dir / "m3_artifact_hash_manifest.csv").exists():
        raise RuntimeError("provenance invalid: m3_artifact_hash_manifest.csv missing")


def validate_start_provenance() -> None:
    dirty = dirty_status_value()
    if dirty != "CLEAN":
        raise RuntimeError(f"provenance invalid before panel run: dirty_status={dirty!r}")
    gpu_query = gpu_query_snapshot()
    if not gpu_query or gpu_query in {"NVIDIA_SMI_UNAVAILABLE", "NVIDIA_SMI_EMPTY"}:
        raise RuntimeError(f"provenance invalid before panel run: gpu_query={gpu_query!r}")


def ensure_output_dir_absent_or_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")


def require_arg_value(args: argparse.Namespace, name: str) -> str:
    value = str(getattr(args, name, "") or "")
    if not value:
        raise SystemExit(f"--{name} is required for panel_seed85")
    return value


def validate_expected_panel_sources(args: argparse.Namespace) -> None:
    expected_commit = require_arg_value(args, "expected_commit")
    expected_config_sha = require_arg_value(args, "expected_config_sha256")
    expected_runner_sha = require_arg_value(args, "expected_runner_sha256")
    expected_attack_adapter_sha = require_arg_value(args, "expected_attack_adapter_sha256")
    expected_frozen_manifest_sha = require_arg_value(args, "expected_frozen_step78_manifest_sha256")
    frozen_manifest = Path(require_arg_value(args, "frozen_step78_manifest"))
    if not frozen_manifest.exists():
        raise FileNotFoundError(str(frozen_manifest))
    actual_commit = git_value(["rev-parse", "HEAD"])
    if actual_commit != expected_commit:
        raise RuntimeError(f"unexpected execution commit: {actual_commit} != {expected_commit}")
    actual_config_sha = sha256_file(Path(args.config))
    if actual_config_sha != expected_config_sha:
        raise RuntimeError(f"unexpected config sha256: {actual_config_sha} != {expected_config_sha}")
    if expected_config_sha != PANEL_FROZEN_V4_CONFIG_SHA256:
        raise RuntimeError(f"panel seed85 must use frozen arm-v4 config sha {PANEL_FROZEN_V4_CONFIG_SHA256}")
    actual_runner_sha = sha256_file(Path(__file__).resolve())
    if actual_runner_sha != expected_runner_sha:
        raise RuntimeError(f"unexpected runner sha256: {actual_runner_sha} != {expected_runner_sha}")
    adapter_path = REPO_ROOT / "src" / "gripper_attack" / "attack_adapter.py"
    actual_adapter_sha = sha256_file(adapter_path)
    if actual_adapter_sha != expected_attack_adapter_sha:
        raise RuntimeError(f"unexpected attack_adapter sha256: {actual_adapter_sha} != {expected_attack_adapter_sha}")
    actual_frozen_manifest_sha = sha256_file(frozen_manifest)
    if actual_frozen_manifest_sha != expected_frozen_manifest_sha:
        raise RuntimeError(
            f"unexpected frozen step78 manifest sha256: {actual_frozen_manifest_sha} != {expected_frozen_manifest_sha}"
        )


def claim_one_shot_sentinel(output_dir: Path, *, stage: str, seed: int) -> Path:
    sentinel = output_dir / f"{stage}_ONESHOT_SENTINEL.json"
    if sentinel.exists():
        raise RuntimeError(f"one-shot sentinel already exists: {sentinel}")
    write_json(
        sentinel,
        {
            "stage": stage,
            "seed": int(seed),
            "commit": git_value(["rev-parse", "HEAD"]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return sentinel


def input_manifest_path(frame_dir: Path, step: int) -> Path:
    return frame_dir / f"m3_step{int(step)}_input_manifest.csv"


def load_input_manifest(frame_dir: Path, step: int) -> dict[str, str]:
    return read_single_csv(input_manifest_path(frame_dir, step))


def step78_parity_from_manifest_paths(new_manifest_path: Path, frozen_manifest_path: Path) -> str:
    return step78_parity_status(read_single_csv(new_manifest_path), read_single_csv(frozen_manifest_path))


def clean_eligibility_from_frame_dir(frame_dir: Path, step: int) -> dict[str, Any]:
    gen_path = frame_dir / frame_filename("clean_generation", step, "json")
    if not gen_path.exists():
        return {"status": "CLEAN_CONTEXT_INELIGIBLE", "reason": "clean_generation_missing", "clean_gripper_token": ""}
    clean_json = json.loads(gen_path.read_text(encoding="utf-8"))
    official = clean_json.get("official", {})
    return clean_frame_eligibility(official)


def _rows_by_condition(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    return {str(r.get("condition", "")): r for r in rows}


def _candidate_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            condition = str(row.get("condition", ""))
            counts[condition] = counts.get(condition, 0) + 1
    return counts


def _parse_token_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x) for x in json.loads(str(value))]


def _float_equal(a: Any, b: Any, *, tolerance: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def audit_candidate_artifacts(
    *,
    candidate_rows: list[Mapping[str, Any]],
    selected_row: Mapping[str, Any] | None,
    condition: str,
    expected_seed: int,
    expected_commit: str,
    epsilon: float,
) -> tuple[bool, str]:
    rows = [dict(r) for r in candidate_rows if str(r.get("condition", "")) == condition]
    if len(rows) != 21:
        return False, f"{condition}:candidate_count_{len(rows)}"
    ids = [int(r.get("candidate_id", -1)) for r in rows]
    if sorted(ids) != list(range(21)) or len(set(ids)) != 21:
        return False, f"{condition}:candidate_ids_not_0_20_unique"
    for row in rows:
        if int(row.get("attack_seed", -1) or -1) != int(expected_seed):
            return False, f"{condition}:attack_seed_mismatch"
        if expected_commit and str(row.get("commit", "")) != expected_commit:
            return False, f"{condition}:commit_mismatch"
        try:
            if len(_parse_token_list(row.get("official_tokens", "[]"))) != 7:
                return False, f"{condition}:official_tokens_not_7"
        except Exception:
            return False, f"{condition}:official_tokens_unparseable"
        if float(row.get("processor_linf", 999.0) or 999.0) > float(epsilon) + 1e-9:
            return False, f"{condition}:processor_linf_over_budget"
    selected_flags = [r for r in rows if int(r.get("selected", 0) or 0) == 1]
    selected_feasible = selected_row is not None and str(selected_row.get("condition_result", "")) != "NO_FEASIBLE_CANDIDATE"
    if selected_feasible:
        if len(selected_flags) != 1:
            return False, f"{condition}:selected_flag_not_unique"
        selected_id = int(selected_row.get("selected_candidate_id", -1) or -1)
        matches = [r for r in rows if int(r.get("candidate_id", -1) or -1) == selected_id]
        if len(matches) != 1:
            return False, f"{condition}:selected_candidate_id_not_unique"
        candidate = matches[0]
        if candidate is not selected_flags[0] and int(candidate.get("selected", 0) or 0) != 1:
            return False, f"{condition}:selected_results_not_selected_candidate"
        if str(candidate.get("score_invariant_status", "")) != "PASS":
            return False, f"{condition}:selected_score_invariant_not_pass"
        if int(candidate.get("official_gripper_token", -1) or -1) != int(selected_row.get("official_gripper_token", -2) or -2):
            return False, f"{condition}:selected_token_mismatch"
        if int(candidate.get("arm_prefix_match_count", -1) or -1) != int(selected_row.get("arm_prefix_match_count", -2) or -2):
            return False, f"{condition}:selected_arm_match_mismatch"
        if not _float_equal(candidate.get("official_target31744_margin", 0.0), selected_row.get("official_target31744_margin", 999.0)):
            return False, f"{condition}:selected_margin_mismatch"
    elif selected_flags:
        return False, f"{condition}:selected_flag_present_without_selected_result"
    return True, "PASS"


def audit_route_artifacts(
    path: Path,
    *,
    expected_seed: int,
    expected_commit: str,
    expected_objective: str = PANEL_FROZEN_OBJECTIVE,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "route_audit_missing"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    required = {"TRUE_PGD_TRAJECTORY21_SELECTIVE", "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"}
    seen = {str(r.get("condition", "")) for r in rows}
    if not required.issubset(seen):
        return False, "route_required_conditions_missing"
    for row in rows:
        if row.get("condition") not in required:
            continue
        if int(row.get("attack_seed", -1) or -1) != int(expected_seed):
            return False, "route_attack_seed_mismatch"
        if expected_commit and str(row.get("commit", "")) != expected_commit:
            return False, "route_commit_mismatch"
        if str(row.get("strict_route", "")).lower() != "true":
            return False, "route_strict_route_not_true"
        if str(row.get("allow_fallback", "")).lower() != "false":
            return False, "route_allow_fallback_not_false"
        if str(row.get("fallback_used", "")).lower() != "false":
            return False, "route_fallback_used"
        if str(row.get("resolved_adapter_class", "")) != "TokenPrefixPGDAttacker":
            return False, "route_wrong_adapter"
        if str(row.get("requested_objective", "")) != expected_objective:
            return False, "route_wrong_requested_objective"
        if str(row.get("resolved_objective", "")) != expected_objective:
            return False, "route_wrong_resolved_objective"
        if int(row.get("target_token_id", -1) or -1) != TARGET_31744:
            return False, "route_wrong_target_token"
        if int(float(row.get("num_backwards", 0) or 0)) != 20:
            return False, "route_wrong_backward_count"
        if int(float(row.get("num_loss_forwards", 0) or 0)) != 21:
            return False, "route_wrong_loss_forward_count"
        if int(float(row.get("num_generation_forwards", 0) or 0)) != 21:
            return False, "route_wrong_generation_forward_count"
        if int(float(row.get("trajectory_candidate_count", 0) or 0)) != 21:
            return False, "route_wrong_trajectory_candidate_count"
    return True, "PASS"


def _route_artifacts_valid(path: Path) -> bool:
    return audit_route_artifacts(path, expected_seed=85, expected_commit="")[0]


def frame_status_from_v4_artifacts(
    frame_dir: Path,
    *,
    step: int,
    main_denominator: bool,
    clean_dir: Path | None = None,
    expected_seed: int = 85,
    expected_commit: str = "",
    expected_objective: str = PANEL_FROZEN_OBJECTIVE,
    epsilon: float = 6.0 / 255.0,
) -> dict[str, Any]:
    clean_status = clean_eligibility_from_frame_dir(clean_dir or frame_dir, step)
    base = {
        "frame": int(step),
        "main_denominator": bool(main_denominator),
        "clean_status": clean_status["status"],
        "clean_gripper_token": clean_status.get("clean_gripper_token", ""),
    }
    if clean_status["status"] != "CLEAN_ELIGIBLE":
        return {**base, "frame_status": clean_status["status"], "infra_status": "SKIPPED_CLEAN_INELIGIBLE"}
    selected_rows = _rows_by_condition(frame_dir / "m3_v4_selected_results.csv")
    candidate_path = frame_dir / "m3_v4_candidate_audit.csv"
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8", newline="") as f:
            candidate_rows = [dict(r) for r in csv.DictReader(f)]
    else:
        candidate_rows = []
    counts = _candidate_counts(candidate_path)
    infra_valid = True
    infra_reasons = []
    route_ok, route_reason = audit_route_artifacts(
        frame_dir / "m3_v4_route_audit.csv",
        expected_seed=expected_seed,
        expected_commit=expected_commit,
        expected_objective=expected_objective,
    )
    if not route_ok:
        infra_valid = False
        infra_reasons.append(route_reason)
    for condition in ["TRUE_PGD_TRAJECTORY21_SELECTIVE", "RAND21_SELECTIVE", "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"]:
        selected = selected_rows.get(condition)
        ok, reason = audit_candidate_artifacts(
            candidate_rows=candidate_rows,
            selected_row=selected,
            condition=condition,
            expected_seed=expected_seed,
            expected_commit=expected_commit,
            epsilon=epsilon,
        )
        if not ok:
            infra_valid = False
            infra_reasons.append(reason)
    status = frame_full_selective_status(
        true_row=selected_rows.get("TRUE_PGD_TRAJECTORY21_SELECTIVE"),
        rand_row=selected_rows.get("RAND21_SELECTIVE"),
        shuffled_row=selected_rows.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"),
        clean_status=clean_status["status"],
        infra_valid=infra_valid,
    )
    return {
        **base,
        "infra_status": "PASS" if infra_valid else "INFRA_INVALID",
        "infra_reasons": ";".join(infra_reasons),
        "frame_status": status["status"],
        "full_pass": status.get("full_pass", False),
        "rand_win": status.get("rand_win", False),
        "shuffled_win": status.get("shuffled_win", False),
        "rand_control_status": status.get("rand_control_status", ""),
        "shuffled_control_status": status.get("shuffled_control_status", ""),
        "rand_paired_margin": status.get("rand_paired_margin", ""),
        "shuffled_paired_margin": status.get("shuffled_paired_margin", ""),
        "true_candidate_count": counts.get("TRUE_PGD_TRAJECTORY21_SELECTIVE", 0),
        "rand_candidate_count": counts.get("RAND21_SELECTIVE", 0),
        "shuffled_candidate_count": counts.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", 0),
    }


def write_panel_joint_and_aggregate(output_dir: Path, frame_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    joint_path = output_dir / "m3_panel_frame_joint_results.csv"
    fields = [
        "frame",
        "main_denominator",
        "clean_status",
        "clean_gripper_token",
        "infra_status",
        "infra_reasons",
        "frame_status",
        "full_pass",
        "rand_win",
        "shuffled_win",
        "rand_control_status",
        "shuffled_control_status",
        "rand_paired_margin",
        "shuffled_paired_margin",
        "true_candidate_count",
        "rand_candidate_count",
        "shuffled_candidate_count",
    ]
    write_csv(joint_path, [dict(r) for r in frame_rows], fields)
    aggregate = panel_aggregate_status([r for r in frame_rows if bool(r.get("main_denominator", True))])
    write_csv(output_dir / "m3_panel_aggregate_result.csv", [aggregate], list(aggregate.keys()))
    return aggregate


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


def write_captured_clean_frame(
    *,
    output_dir: Path,
    cfg: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: str,
    model_dtype: torch.dtype,
    action_dim: int,
    instruction: str,
    raw_at_step: np.ndarray,
    clean_action_at_step: np.ndarray,
    step: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / frame_filename("raw_agentview", step, "npy")
    png_path = output_dir / frame_filename("raw_agentview", step, "png")
    pt_path = output_dir / frame_filename("processor_inputs", step, "pt")
    gen_path = output_dir / frame_filename("clean_generation", step, "json")
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
        "step": int(step),
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
    write_csv(output_dir / f"m3_step{step}_input_manifest.csv", [manifest_row], list(manifest_row.keys()))
    if step == PANEL_POSITIVE_CONTROL_FRAME:
        write_csv(output_dir / "m3_step78_input_manifest.csv", [manifest_row], list(manifest_row.keys()))
    write_artifact_hash_manifest(output_dir)
    return manifest_row


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
    step = int(cfg["input"]["absolute_step"])
    manifest_row = write_captured_clean_frame(
        output_dir=output_dir,
        cfg=cfg,
        model=model,
        processor=processor,
        device=device,
        model_dtype=model_dtype,
        action_dim=action_dim,
        instruction=instruction,
        raw_at_step=raw_at_step,
        clean_action_at_step=clean_action_at_step,
        step=step,
    )
    update_manifest_model_fingerprint(output_dir, model_fingerprint(model))
    write_artifact_hash_manifest(output_dir)
    print(json.dumps({"status": "CAPTURED", "output_dir": str(output_dir), "clean_gripper_token": manifest_row["clean_gripper_token"]}, indent=2))


def load_frozen_input(input_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path = input_dir / "raw_agentview_step78.npy"
    gen_path = input_dir / "clean_generation_step78.json"
    if not raw_path.exists():
        matches = sorted(input_dir.glob("raw_agentview_step*.npy"))
        if matches:
            raw_path = matches[0]
    if not gen_path.exists():
        matches = sorted(input_dir.glob("clean_generation_step*.json"))
        if matches:
            gen_path = matches[0]
    if not raw_path.exists() or not gen_path.exists():
        raise FileNotFoundError(f"frozen input missing raw npy or clean_generation json in {input_dir}")
    return np.load(raw_path), json.loads(gen_path.read_text(encoding="utf-8"))


def run_capture_panel_inputs(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path

    validate_start_provenance()
    steps = parse_panel_steps(args.panel_steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model(cfg["model"]["path"], args.model_gpu_device_id)
    update_manifest_model_fingerprint(output_dir, model_fingerprint(model))
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

    captured: list[dict[str, Any]] = []
    step_set = set(steps)
    for step in range(max(steps) + 1):
        raw = np.asarray(obs["agentview_image"]).copy()
        action, _scores, _dt, _gen = decode_with_scores(
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
        if step in step_set:
            frame_dir = output_dir / f"step{step}"
            captured.append(
                write_captured_clean_frame(
                    output_dir=frame_dir,
                    cfg={**cfg, "input": {**cfg["input"], "absolute_step": int(step)}},
                    model=model,
                    processor=processor,
                    device=device,
                    model_dtype=model_dtype,
                    action_dim=action_dim,
                    instruction=instruction,
                    raw_at_step=raw,
                    clean_action_at_step=np.asarray(action, dtype=np.float32),
                    step=step,
                )
            )
        if step < max(steps):
            obs, _reward, _done, _info = env.step(postprocess_openvla_action_for_libero(action, enabled=True))
    env.close()

    if sorted(int(r["step"]) for r in captured) != steps:
        raise RuntimeError(f"panel capture missed frames: expected {steps}, got {[r['step'] for r in captured]}")
    write_csv(output_dir / "m3_panel_capture_manifest.csv", captured, list(captured[0].keys()))
    write_artifact_hash_manifest(output_dir)
    print(json.dumps({"status": "PANEL_CAPTURED_SINGLE_CLEAN_REPLAY", "frames": steps, "output_dir": str(output_dir)}, indent=2))


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
    write_artifact_hash_manifest(output_dir)
    print(json.dumps({"status": "V4_CANARY_COMPLETE", "result_class": result_class, "output_dir": str(output_dir)}, indent=2))


def run_preflight_or_canary(args: argparse.Namespace, cfg: Mapping[str, Any], *, canary: bool) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_image, clean_json = load_frozen_input(Path(args.input_dir))
    model, processor, device = load_model(cfg["model"]["path"], args.model_gpu_device_id)
    update_manifest_model_fingerprint(output_dir, model_fingerprint(model))
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
        write_artifact_hash_manifest(output_dir)
        print(json.dumps({"status": "PREFLIGHT_MISMATCH", "clean": clean_preflight, "delta0": delta0_preflight}, indent=2))
        return
    if not canary:
        write_artifact_hash_manifest(output_dir)
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
    write_artifact_hash_manifest(output_dir)
    print(json.dumps({"status": "CANARY_COMPLETE_UNCLASSIFIED", "output_dir": str(output_dir)}, indent=2))


def run_panel_seed85(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    validate_panel_seed(int(args.attack_seed))
    parse_panel_steps(args.panel_steps)
    if not args.frozen_step78_manifest:
        raise SystemExit("--frozen_step78_manifest is required for panel_seed85")
    validate_expected_panel_sources(args)
    output_dir = Path(args.output_dir)
    ensure_output_dir_absent_or_empty(output_dir)
    validate_start_provenance()
    claim_one_shot_sentinel(output_dir, stage="m3_panel_seed85", seed=int(args.attack_seed))
    write_manifest(args, cfg)

    capture_args = argparse.Namespace(**vars(args))
    capture_args.output_dir = str(output_dir / "capture")
    run_capture_panel_inputs(capture_args, cfg)
    first_capture_manifest = load_input_manifest(output_dir / "capture" / f"step{PANEL_ALL_CAPTURE_FRAMES[0]}", PANEL_ALL_CAPTURE_FRAMES[0])
    update_manifest_model_fingerprint_text(output_dir, str(first_capture_manifest["model_fingerprint"]))
    write_artifact_hash_manifest(output_dir)

    step78_new = output_dir / "capture" / f"step{PANEL_POSITIVE_CONTROL_FRAME}" / "m3_step78_input_manifest.csv"
    parity = step78_parity_from_manifest_paths(step78_new, Path(args.frozen_step78_manifest))
    write_json(
        output_dir / "m3_panel_step78_parity.json",
        {
            "status": parity,
            "new_manifest": str(step78_new),
            "frozen_manifest": str(args.frozen_step78_manifest),
        },
    )
    if parity != "POSITIVE_CONTROL_INPUT_MATCH":
        write_artifact_hash_manifest(output_dir)
        raise RuntimeError("POSITIVE_CONTROL_INPUT_MISMATCH")

    frame_rows: list[dict[str, Any]] = []
    for step in PANEL_ALL_CAPTURE_FRAMES:
        clean_dir = output_dir / "capture" / f"step{step}"
        main_denominator = step in PANEL_MAIN_FRAMES
        clean_status = clean_eligibility_from_frame_dir(clean_dir, step)
        if step == PANEL_POSITIVE_CONTROL_FRAME:
            frame_rows.append(
                {
                    "frame": step,
                    "main_denominator": False,
                    "clean_status": clean_status["status"],
                    "clean_gripper_token": clean_status.get("clean_gripper_token", ""),
                    "infra_status": "POSITIVE_CONTROL_ONLY",
                    "infra_reasons": "",
                    "frame_status": "POSITIVE_CONTROL_INPUT_MATCH",
                    "full_pass": False,
                    "rand_win": "",
                    "shuffled_win": "",
                    "rand_control_status": "",
                    "shuffled_control_status": "",
                    "rand_paired_margin": "",
                    "shuffled_paired_margin": "",
                    "true_candidate_count": "",
                    "rand_candidate_count": "",
                    "shuffled_candidate_count": "",
                }
            )
            continue
        if clean_status["status"] != "CLEAN_ELIGIBLE":
            frame_rows.append(
                {
                    "frame": step,
                    "main_denominator": main_denominator,
                    "clean_status": clean_status["status"],
                    "clean_gripper_token": clean_status.get("clean_gripper_token", ""),
                    "infra_status": "SKIPPED_CLEAN_INELIGIBLE",
                    "infra_reasons": "",
                    "frame_status": clean_status["status"],
                    "full_pass": False,
                    "rand_win": "",
                    "shuffled_win": "",
                    "rand_control_status": "",
                    "shuffled_control_status": "",
                    "rand_paired_margin": "",
                    "shuffled_paired_margin": "",
                    "true_candidate_count": "",
                    "rand_candidate_count": "",
                    "shuffled_candidate_count": "",
                }
            )
            continue

        frame_output = output_dir / "frames" / f"step{step}"
        frame_args = argparse.Namespace(**vars(args))
        frame_args.input_dir = str(clean_dir)
        frame_args.output_dir = str(frame_output)
        frame_args.mode = "canary_v4"
        write_manifest(frame_args, cfg)
        run_preflight_or_canary(frame_args, cfg, canary=True)
        validate_manifest_provenance(frame_output)
        frame_rows.append(
            frame_status_from_v4_artifacts(
                frame_output,
                step=step,
                main_denominator=main_denominator,
                clean_dir=clean_dir,
                expected_seed=int(args.attack_seed),
                expected_commit=git_value(["rev-parse", "HEAD"]),
                expected_objective=str(cfg["attack_optimizer"]["objective"]),
                epsilon=float(cfg["attack_optimizer"]["epsilon"]),
            )
        )

    write_recursive_artifact_hash_manifest(output_dir)
    verify_recursive_artifact_hash_manifest(output_dir)
    aggregate = write_panel_joint_and_aggregate(output_dir, frame_rows)
    write_json(output_dir / "m3_panel_result.json", {"aggregate": aggregate, "frames": frame_rows})
    write_recursive_artifact_hash_manifest(output_dir)
    verify_recursive_artifact_hash_manifest(output_dir)
    write_artifact_hash_manifest(output_dir)
    validate_manifest_provenance(output_dir)
    print(json.dumps({"status": aggregate["panel_status"], "output_dir": str(output_dir), "aggregate": aggregate}, indent=2))


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
        "dirty_status": dirty_status_value(),
        "config_path": str(args.config),
        "config_sha256": sha256_file(Path(args.config)),
        "output_dir": str(output_dir),
        "model_path": cfg["model"]["path"],
        "model_fingerprint": "PENDING_MODEL_LOAD",
        "python": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_query": gpu_query_snapshot(),
        "command_line": " ".join(sys.argv),
        "status": args.mode,
    }
    write_csv(output_dir / "m3_step78_manifest.csv", [row], list(row.keys()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_step78_true_pgd_31744.yaml"))
    ap.add_argument(
        "--mode",
        choices=["capture_input", "capture_panel_inputs", "preflight_zero_step", "canary", "canary_v4", "panel_seed85"],
        required=True,
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--panel_steps", default="")
    ap.add_argument("--frozen_step78_manifest", default="")
    ap.add_argument("--expected_commit", default="")
    ap.add_argument("--expected_config_sha256", default="")
    ap.add_argument("--expected_runner_sha256", default="")
    ap.add_argument("--expected_attack_adapter_sha256", default="")
    ap.add_argument("--expected_frozen_step78_manifest_sha256", default="")
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
    if args.mode != "panel_seed85":
        write_manifest(args, cfg)
    if args.mode == "capture_input":
        run_capture_input(args, cfg)
    elif args.mode == "capture_panel_inputs":
        run_capture_panel_inputs(args, cfg)
    elif args.mode == "panel_seed85":
        run_panel_seed85(args, cfg)
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
