#!/usr/bin/env python3
"""Audit the M3 log-ratio v2 PGD trajectory for feasible intermediates.

This script has two modes:

* ``offline_telemetry`` extracts the per-step surrogate telemetry already
  recorded in ``m3_step78_canary_debug.json``.  This mode does not load the
  OpenVLA model.
* ``replay_candidates`` deterministically replays the true-PGD update loop and
  official-decodes delta0 plus every post-update iterate.  Scientific
  interpretation is allowed only if the replay reproduces the frozen final
  tensor hashes and final official output.

It does not run LIBERO and does not select a new objective, target token, frame,
or perturbation budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")


EXPECTED_DELTA0_SHA = "03676888a9627bc55088e7ce7d282f18105aa2b7e5d8e4b0521ed0da4b7d4ffb"
EXPECTED_FINAL_DELTA_SHA = "cca2ee4bad51c0faab933bc89733a58d05287f938c0f0b16bc02948f00e6229c"
EXPECTED_FINAL_PROCESSOR_SHA = "b25aafeb26b6032b1b094edd4586c6a0cd5c4520a619e3b822ff95ccbc2d1d76"
EXPECTED_FINAL_TOKENS = [31938, 31870, 31938, 31882, 31999, 31915, 31744]
EXPECTED_FINAL_MARGIN = 29.249469757080078
EXPECTED_FINAL_ARM_MATCH = 2


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def arm_match_count(prefix: list[int], clean_prefix: list[int]) -> int:
    n = min(len(prefix), len(clean_prefix))
    return sum(int(int(prefix[i]) == int(clean_prefix[i])) for i in range(n))


def _json_list(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def extract_offline_telemetry(debug: Mapping[str, Any], *, condition: str = "true_pgd") -> list[dict[str, Any]]:
    if condition not in debug:
        raise KeyError(f"debug JSON missing condition {condition!r}")
    node = debug[condition]
    clean_prefix = [int(x) for x in node.get("clean_generated_arm_prefix_token_ids", [])]
    logratio = node.get("target_token_logratio_margin_trajectory", [])
    best_margin = node.get("target_token_best_margin_trajectory")
    prefixes = node.get("generated_arm_prefix_trajectory", [])
    grads = node.get("gradient_norm_trajectory", [])
    n = max(len(logratio), len(prefixes), len(grads))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        prefix = [int(x) for x in prefixes[i]] if i < len(prefixes) else []
        grad = grads[i] if i < len(grads) and isinstance(grads[i], Mapping) else {}
        rows.append(
            {
                "iteration": i,
                "surrogate_logratio_margin": "" if i >= len(logratio) else float(logratio[i]),
                "surrogate_best_competitor_margin": (
                    "NOT_RECORDED"
                    if best_margin is None or i >= len(best_margin)
                    else float(best_margin[i])
                ),
                "arm_prefix": _json_list(prefix),
                "arm_match_count": arm_match_count(prefix, clean_prefix),
                "gradient_l1": grad.get("l1", ""),
                "gradient_l2": grad.get("l2", ""),
                "gradient_linf": grad.get("linf", ""),
            }
        )
    return rows


def classify_feasible_intermediate(
    rows: list[Mapping[str, Any]],
    *,
    target_token_id: int = 31744,
    arm_gate_min_match_count: int = 5,
    margin_threshold: float = 6.0,
) -> str:
    for row in rows:
        try:
            iteration = int(row.get("iteration", -1))
            token = int(row.get("official_gripper_token", -1))
            arm = int(row.get("arm_prefix_match_count", -1))
            margin = float(row.get("official_target31744_margin", "nan"))
        except (TypeError, ValueError):
            continue
        if iteration > 0 and token == int(target_token_id) and arm >= int(arm_gate_min_match_count) and margin > float(margin_threshold):
            return "FEASIBLE_INTERMEDIATE_EXISTS"
    return "NO_FEASIBLE_INTERMEDIATE"


def _float_close(a: Any, b: float, *, tol: float) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(tol))
    except (TypeError, ValueError):
        return False


def validate_reconstruction(rows: list[Mapping[str, Any]], *, margin_tolerance: float = 1e-5) -> tuple[str, list[str]]:
    issues: list[str] = []
    by_iter = {int(row.get("iteration", -999)): row for row in rows}
    delta0 = by_iter.get(0)
    final = by_iter.get(20)
    if delta0 is None:
        issues.append("missing iteration 0")
    elif delta0.get("delta_sha256") != EXPECTED_DELTA0_SHA:
        issues.append("delta0 hash mismatch")
    if final is None:
        issues.append("missing iteration 20")
    else:
        if final.get("delta_sha256") != EXPECTED_FINAL_DELTA_SHA:
            issues.append("final delta hash mismatch")
        if final.get("processor_input_sha256") != EXPECTED_FINAL_PROCESSOR_SHA:
            issues.append("final processor input hash mismatch")
        try:
            tokens = json.loads(str(final.get("official_tokens", "[]")))
        except json.JSONDecodeError:
            tokens = []
        if tokens != EXPECTED_FINAL_TOKENS:
            issues.append("final official tokens mismatch")
        if not _float_close(final.get("official_target31744_margin"), EXPECTED_FINAL_MARGIN, tol=margin_tolerance):
            issues.append("final official margin mismatch")
        try:
            arm = int(final.get("arm_prefix_match_count", -1))
        except (TypeError, ValueError):
            arm = -1
        if arm != EXPECTED_FINAL_ARM_MATCH:
            issues.append("final arm match mismatch")
    return ("RECONSTRUCTION_VALID" if not issues else "RECONSTRUCTION_INVALID", issues)


def _make_clean_generation(base_input_ids: Any, clean_tokens: list[int]) -> Any:
    import torch

    clean_gen = type("CleanGen", (), {})()
    clean_gen.sequences = torch.tensor(
        [base_input_ids[0].detach().cpu().tolist() + [int(x) for x in clean_tokens]],
        dtype=torch.long,
        device=base_input_ids.device,
    )
    clean_gen.scores = []
    return clean_gen


def replay_true_pgd_candidates(
    *,
    cfg: Mapping[str, Any],
    input_dir: Path,
    attack_seed: int,
    model_gpu_device_id: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
    from gripper_attack.m3_controls import shuffled_grad_direction, tensor_sha256
    from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
        build_attacker,
        load_frozen_input,
        load_model,
        official_decode,
        preprocess_raw_image,
    )

    raw_image, clean_json = load_frozen_input(input_dir)
    model, processor, device = load_model(cfg["model"]["path"], model_gpu_device_id)
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
        tolerance=float(tolerance),
        objective=str(cfg["attack_optimizer"]["objective"]),
    )
    clean_prefix = [int(x) for x in clean_official["arm_prefix"]]
    clean_gen = _make_clean_generation(base_inputs["input_ids"], clean_official["tokens"])

    wrapper = build_attacker(model, processor, cfg, seed=int(attack_seed), device=device, gradient_transform="none")
    adapter = wrapper.adapter
    if not isinstance(adapter, TokenPrefixPGDAttacker):
        raise RuntimeError(f"expected TokenPrefixPGDAttacker, got {type(adapter).__name__}")
    adapter._freeze_model()

    target_ids = adapter.action_to_token_ids(clean_action, cfg["model"]["unnorm_key"])
    clean_ids, _full_ids, _labels, x0 = adapter._build_inputs_and_labels(raw_image, instruction, target_ids)
    clean_generated_action_token_ids = adapter._exact_tokens_from_generation(
        clean_gen,
        prompt_len=int(clean_ids.shape[1]),
        action_dim=int(target_ids.numel()),
        context="m3 v2 trajectory replay",
    )
    if [int(x) for x in clean_generated_action_token_ids[:6].detach().cpu().tolist()] != clean_prefix:
        raise RuntimeError("clean generated prefix differs between replay setup and official decode")

    x_orig_model = x0.detach()
    x_orig = x_orig_model.detach().float()
    gen = torch.Generator(device=x_orig.device)
    gen.manual_seed(int(attack_seed))
    if bool(cfg["attack_optimizer"]["random_start"]):
        delta = torch.empty_like(x_orig).uniform_(-float(adapter.epsilon), float(adapter.epsilon), generator=gen)
    else:
        delta = torch.zeros_like(x_orig)
    adv = adapter._project_pixel_master(x_orig + delta, x_orig).detach()
    delta0_adv_model = adapter._cast_projected_pixel_values(adv.detach(), x_orig_model)
    candidate_states: list[dict[str, Any]] = []

    def record_candidate(
        *,
        iteration: int,
        update_step: int | str,
        pixel_values: torch.Tensor,
        surrogate_stats: Mapping[str, Any] | None = None,
        surrogate_prefix: list[int] | None = None,
        gradient_norm: Mapping[str, Any] | None = None,
    ) -> None:
        candidate_states.append(
            {
                "iteration": int(iteration),
                "update_step": update_step,
                "pixel_values": pixel_values.detach().clone(),
                "surrogate_stats": dict(surrogate_stats or {}),
                "surrogate_prefix": list(surrogate_prefix or []),
                "gradient_norm": dict(gradient_norm or {}),
            }
        )

    def official_row(state: Mapping[str, Any]) -> dict[str, Any]:
        pixel_values = state["pixel_values"]
        surrogate_stats = state["surrogate_stats"]
        surrogate_prefix = state["surrogate_prefix"]
        gradient_norm = state["gradient_norm"]
        official = official_decode(
            model,
            {"input_ids": clean_ids.detach(), "pixel_values": pixel_values.detach()},
            action_dim=action_dim,
            unnorm_key=cfg["model"]["unnorm_key"],
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
            tolerance=float(tolerance),
            objective=str(cfg["attack_optimizer"]["objective"]),
        )
        delta_model = (pixel_values.detach().float() - x_orig_model.detach().float()).detach()
        arm_match = arm_match_count([int(x) for x in official["arm_prefix"]], clean_prefix)
        stats = official["target_stats"]
        return {
            "iteration": int(state["iteration"]),
            "update_step": state["update_step"],
            "official_tokens": _json_list(official["tokens"]),
            "official_gripper_token": int(official["gripper_token"]),
            "official_target31744_score": float(stats["target_token_score"]),
            "official_best_competitor_token": int(stats["best_competitor_token_id"]),
            "official_best_competitor_score": float(stats["best_competitor_score"]),
            "official_target31744_margin": float(stats.get("target_objective_margin", stats["target_minus_best_competitor_margin"])),
            "official_target31744_best_competitor_margin": float(stats["target_minus_best_competitor_margin"]),
            "official_target31744_logratio_margin": stats.get("target_minus_competitor_logsumexp_margin", ""),
            "score_invariant_status": "PASS" if official["score_invariant"]["tie_aware_pass"] else "FAIL",
            "official_arm_prefix": _json_list(official["arm_prefix"]),
            "clean_arm_prefix": _json_list(clean_prefix),
            "arm_prefix_match_count": int(arm_match),
            "arm_prefix_match_denominator": int(len(clean_prefix)),
            "surrogate_arm_prefix_pre_update": _json_list(surrogate_prefix),
            "surrogate_logratio_margin_pre_update": surrogate_stats.get("target_minus_competitor_logsumexp_margin", ""),
            "surrogate_best_competitor_margin_pre_update": surrogate_stats.get("target_minus_best_competitor_margin", ""),
            "gradient_l1": gradient_norm.get("l1", ""),
            "gradient_l2": gradient_norm.get("l2", ""),
            "gradient_linf": gradient_norm.get("linf", ""),
            "delta_sha256": tensor_sha256(delta_model),
            "processor_input_sha256": tensor_sha256(pixel_values.detach()),
            "processor_linf": float(delta_model.abs().max().cpu()) if delta_model.numel() else 0.0,
        }

    record_candidate(iteration=0, update_step="delta0", pixel_values=delta0_adv_model)

    generated_arm_prefix_token_ids = None
    target_token_id = int(cfg["attack_optimizer"]["target_token_id"])
    margin = float(cfg["attack_optimizer"]["gripper_margin"])
    prefix_refresh_interval = int(cfg["attack_optimizer"].get("prefix_refresh_interval", 1))
    for step in range(int(cfg["attack_optimizer"]["num_steps"])):
        adv = adv.detach().requires_grad_(True)
        if generated_arm_prefix_token_ids is None or (step % prefix_refresh_interval) == 0:
            adv_for_gen = adapter._cast_projected_pixel_values(adv, x_orig_model)
            generated_arm_prefix_token_ids = adapter._generate_action_prefix_tokens(
                clean_ids,
                adv_for_gen,
                prefix_len=max(int(target_ids.numel()) - 1, 0),
            )
        adv_g = adapter._cast_projected_pixel_values(adv, x_orig_model)
        gripper_loss, gripper_stats = adapter._generated_prefix_target_token_loss_and_stats(
            clean_ids,
            generated_arm_prefix_token_ids,
            adv_g,
            target_token_id=target_token_id,
            margin=margin,
        )
        grad = torch.autograd.grad(gripper_loss, adv, retain_graph=False, create_graph=False)[0]
        if str(adapter.gradient_transform or "none").strip().lower() not in {"", "none"}:
            grad = shuffled_grad_direction(
                grad,
                seed=int(adapter.gradient_transform_seed) + int(step),
                mode=str(adapter.gradient_transform),
            )
        grad_norm = {
            "l1": float(grad.detach().abs().sum().cpu()),
            "l2": float(torch.linalg.vector_norm(grad.detach().reshape(-1)).cpu()),
            "linf": float(grad.detach().abs().max().cpu()) if grad.numel() else 0.0,
        }
        prefix = [int(x) for x in generated_arm_prefix_token_ids.detach().cpu().tolist()]
        adv = adv.detach() - float(adapter.step_size) * grad.detach().sign()
        adv = adapter._project_pixel_master(adv, x_orig).detach()
        adv_model = adapter._cast_projected_pixel_values(adv.detach(), x_orig_model)
        record_candidate(
            iteration=step + 1,
            update_step=step,
            pixel_values=adv_model,
            surrogate_stats=gripper_stats,
            surrogate_prefix=prefix,
            gradient_norm=grad_norm,
        )
        del grad, gripper_loss, adv_g
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    replay_meta = {
        "clean_tokens": clean_official["tokens"],
        "clean_arm_prefix": clean_prefix,
        "attack_seed": int(attack_seed),
        "config_objective": cfg["attack_optimizer"]["objective"],
        "target_token_id": int(cfg["attack_optimizer"]["target_token_id"]),
        "epsilon": float(cfg["attack_optimizer"]["epsilon"]),
        "num_steps": int(cfg["attack_optimizer"]["num_steps"]),
        "step_size": float(cfg["attack_optimizer"]["step_size"]),
        "commit": git_value(["rev-parse", "HEAD"]),
        "official_decode_schedule": "after_full_pgd_replay",
    }
    rows = [official_row(state) for state in candidate_states]
    return rows, replay_meta


def write_report(
    path: Path,
    *,
    mode: str,
    reconstruction_status: str,
    result_class: str,
    issues: list[str],
    output_tables: list[str],
) -> None:
    body = f"""# M3 v2 seed81 trajectory feasibility audit

## Status

- Mode: `{mode}`
- Reconstruction status: `{reconstruction_status}`
- Result class: `{result_class}`

## Output Tables

"""
    for table in output_tables:
        body += f"- `{table}`\n"
    body += "\n## Reconstruction Issues\n\n"
    if issues:
        for issue in issues:
            body += f"- {issue}\n"
    else:
        body += "- None recorded.\n"
    body += """
## Allowed Claim

This artifact may be used to decide whether the existing v2 seed81 20-step
trajectory contains a selective intermediate candidate after deterministic
replay validates the frozen final hashes and official output.

## Forbidden Claim

Do not use surrogate-only telemetry, unvalidated replay, or any intermediate
candidate to claim closed-loop Layer3 success.  This audit is fixed-frame only
and does not run LIBERO.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline_telemetry", "replay_candidates"], required=True)
    ap.add_argument("--debug_json", default="")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_step78_true_pgd_31744_logratio_v2.yaml"))
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tables_dir", default="")
    ap.add_argument("--reports_dir", default="")
    ap.add_argument("--attack_seed", type=int, default=81)
    ap.add_argument("--model_gpu_device_id", type=int, default=-1)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--margin_threshold", type=float, default=6.0)
    ap.add_argument("--arm_gate_min_match_count", type=int, default=5)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else output_dir
    reports_dir = Path(args.reports_dir) if args.reports_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cfg: Mapping[str, Any] = {}
    if args.mode == "replay_candidates":
        from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import load_config

        cfg = load_config(Path(args.config))
    output_tables: list[str] = []
    reconstruction_status = "NOT_RUN"
    result_class = "OFFLINE_TELEMETRY_ONLY"
    issues: list[str] = []

    if args.mode == "offline_telemetry":
        if not args.debug_json:
            raise SystemExit("--debug_json is required for offline_telemetry")
        rows = extract_offline_telemetry(load_json(Path(args.debug_json)), condition="true_pgd")
        telemetry_path = tables_dir / "m3_v2_seed81_trajectory_telemetry.csv"
        write_csv(telemetry_path, rows, list(rows[0].keys()) if rows else [
            "iteration",
            "surrogate_logratio_margin",
            "surrogate_best_competitor_margin",
            "arm_prefix",
            "arm_match_count",
            "gradient_l1",
            "gradient_l2",
            "gradient_linf",
        ])
        output_tables.append(str(telemetry_path))
    else:
        if not args.input_dir:
            raise SystemExit("--input_dir is required for replay_candidates")
        rows, replay_meta = replay_true_pgd_candidates(
            cfg=cfg,
            input_dir=Path(args.input_dir),
            attack_seed=int(args.attack_seed),
            model_gpu_device_id=int(args.model_gpu_device_id),
            tolerance=float(args.tolerance),
        )
        reconstruction_status, issues = validate_reconstruction(rows, margin_tolerance=float(args.tolerance))
        for row in rows:
            row["reconstruction_status"] = reconstruction_status
        result_class = (
            "RECONSTRUCTION_INVALID"
            if reconstruction_status != "RECONSTRUCTION_VALID"
            else classify_feasible_intermediate(
                rows,
                target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
                arm_gate_min_match_count=int(args.arm_gate_min_match_count),
                margin_threshold=float(args.margin_threshold),
            )
        )
        full_path = tables_dir / "m3_v2_seed81_full_trajectory_official_audit.csv"
        write_csv(full_path, rows, list(rows[0].keys()) if rows else [])
        claim_rows = [
            {
                "claim": "reconstruction_status",
                "status": reconstruction_status,
                "evidence": str(full_path),
                "notes": "; ".join(issues) if issues else "delta0/final hashes and final official output match expected seed81 v2 artifact",
            },
            {
                "claim": "feasible_intermediate",
                "status": result_class,
                "evidence": str(full_path),
                "notes": f"requires iteration>0, token 31744, margin>{float(args.margin_threshold)}, arm>={int(args.arm_gate_min_match_count)}/6",
            },
            {
                "claim": "libero_rollout",
                "status": "NOT_RUN",
                "evidence": "",
                "notes": "fixed-frame audit only",
            },
        ]
        claim_path = tables_dir / "m3_v2_seed81_trajectory_claim_matrix.csv"
        write_csv(claim_path, claim_rows, list(claim_rows[0].keys()))
        write_json(output_dir / "m3_v2_seed81_replay_meta.json", replay_meta)
        output_tables.extend([str(full_path), str(claim_path)])

    report_path = reports_dir / "M3_V2_SEED81_TRAJECTORY_FEASIBILITY_AUDIT.md"
    write_report(
        report_path,
        mode=args.mode,
        reconstruction_status=reconstruction_status,
        result_class=result_class,
        issues=issues,
        output_tables=output_tables,
    )
    print(json.dumps({"mode": args.mode, "result_class": result_class, "reconstruction_status": reconstruction_status, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
