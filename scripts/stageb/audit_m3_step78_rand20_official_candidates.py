from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
from gripper_attack.m3_controls import (
    project_and_cast_processor_values,
    rand_seed_schedule,
    sample_processor_delta,
    tensor_sha256,
)
from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
    git_value,
    load_config,
    load_frozen_input,
    load_model,
    official_decode,
    preprocess_raw_image,
    surrogate_stats_from_generated_prefix,
    write_csv,
)


STAGE = "M3_STEP78_RAND20_SELECTIVITY_AUDIT"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def arm_match_count(candidate_prefix: list[int], clean_prefix: list[int]) -> int:
    return sum(1 for a, b in zip(candidate_prefix, clean_prefix) if int(a) == int(b))


def classify_rand20_selectivity(rows: list[Mapping[str, Any]], *, arm_gate_min_match_count: int) -> str:
    emitted_31744 = [row for row in rows if int(row["official_gripper_token"]) == 31744]
    if any(int(row["arm_prefix_match_count"]) >= int(arm_gate_min_match_count) for row in emitted_31744):
        return "RANDOM_SELECTIVE_MATCH_EXISTS"
    if emitted_31744:
        return "RANDOM_ONLY_NONSELECTIVE_MATCH"
    return "NO_RANDOM_MATCH"


def verify_candidate_hashes(
    frozen_rows: list[Mapping[str, Any]],
    reconstructed_rows: list[Mapping[str, Any]],
) -> None:
    if len(frozen_rows) != len(reconstructed_rows):
        raise RuntimeError(f"candidate count mismatch: frozen={len(frozen_rows)} reconstructed={len(reconstructed_rows)}")
    frozen_by_id = {int(row["candidate_id"]): row for row in frozen_rows}
    for recon in reconstructed_rows:
        cid = int(recon["candidate_id"])
        if cid not in frozen_by_id:
            raise RuntimeError(f"reconstructed candidate {cid} missing from frozen CSV")
        frozen = frozen_by_id[cid]
        checks = {
            "candidate_seed": int(frozen["candidate_seed"]) == int(recon["candidate_seed"]),
            "delta_sha256": str(frozen["delta_sha256"]) == str(recon["delta_sha256"]),
            "processor_input_sha256": str(frozen["processor_input_sha256"]) == str(recon["processor_input_sha256"]),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise RuntimeError(f"candidate {cid} frozen hash/schedule mismatch: {failed}")


def reconstruct_candidates(
    *,
    model: Any,
    processor: Any,
    cfg: Mapping[str, Any],
    base_inputs: Mapping[str, torch.Tensor],
    device: str,
    attack_seed: int,
    action_dim: int,
) -> tuple[list[dict[str, Any]], dict[int, Mapping[str, torch.Tensor]]]:
    adapter = TokenPrefixPGDAttacker(
        model,
        processor,
        {"attack_optimizer": cfg["attack_optimizer"]},
        seed=int(attack_seed),
        preprocess_kwargs=dict(cfg.get("preprocess", {})),
        device=device,
    )
    x = base_inputs["pixel_values"]
    candidate_rows: list[dict[str, Any]] = []
    inputs_by_id: dict[int, Mapping[str, torch.Tensor]] = {}
    seeds = rand_seed_schedule(int(attack_seed), count=int(cfg["controls"]["rand20_count"]))
    for idx, cand_seed in enumerate(seeds):
        delta = sample_processor_delta(
            x.shape,
            epsilon=float(cfg["attack_optimizer"]["epsilon"]),
            seed=int(cand_seed),
            dtype=torch.float32,
            device=x.device,
        )
        projected, corrections = project_and_cast_processor_values(
            x,
            delta,
            epsilon=float(cfg["attack_optimizer"]["epsilon"]),
            candidate_is_delta=True,
        )
        cand_inputs = {"input_ids": base_inputs["input_ids"], "pixel_values": projected.detach()}
        stats = surrogate_stats_from_generated_prefix(
            adapter,
            cand_inputs["input_ids"],
            cand_inputs["pixel_values"],
            action_dim=int(action_dim),
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
        )
        inputs_by_id[idx] = cand_inputs
        candidate_rows.append(
            {
                "candidate_id": idx,
                "candidate_seed": int(cand_seed),
                "surrogate_target31744_margin_recomputed": float(stats["target_minus_best_competitor_margin"]),
                "delta_sha256": tensor_sha256((projected - x).detach().float()),
                "processor_input_sha256": tensor_sha256(projected.detach()),
                "budget_quantized_correction_count": int(corrections),
            }
        )
    return candidate_rows, inputs_by_id


def write_report(path: Path, *, summary: Mapping[str, Any], output_dir: Path, candidate_csv: Path) -> None:
    body = f"""# M3 Step78 RAND20 Selectivity Audit

## Result

`{summary["result_class"]}`

This exploratory audit official-decodes all 20 frozen RAND20 candidates from
the original M3 step78 canary. It does not change the preregistered v1 result:
`RANDOM_NOT_BEATEN`.

## Counts

- candidate count: `{summary["candidate_count"]}`
- official 31744 count: `{summary["official_31744_count"]}`
- selective 31744 count, arm prefix >= {summary["arm_gate_min_match_count"]}/6: `{summary["selective_31744_count"]}`
- nonselective 31744 count: `{summary["nonselective_31744_count"]}`
- best official margin: `{summary["best_official_margin"]}`
- best official margin candidate: `{summary["best_official_margin_candidate_id"]}`

## Provenance

- source candidate CSV: `{candidate_csv}`
- output directory: `{output_dir}`
- commit: `{summary["commit"]}`
- input hash check: `{summary["hash_verification"]}`

## Claim Boundary

Allowed claim: the frozen RAND20 candidate set was official-decoded for
selectivity auditing with hash-verified candidate reconstruction.

Forbidden claim: this does not establish true-PGD superiority, closed-loop
critical-closure disruption, paired task effect, or held-out transfer.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_step78_true_pgd_31744.yaml"))
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--candidate_csv", default=str(REPO_ROOT / "tables" / "m3_step78_canary_candidate_controls_af545e1_seed80.csv"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--attack_seed", type=int, default=80)
    ap.add_argument("--model_gpu_device_id", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    frozen_rows = read_csv(Path(args.candidate_csv))
    raw_image, clean_json = load_frozen_input(Path(args.input_dir))
    model, processor, device = load_model(cfg["model"]["path"], args.model_gpu_device_id)
    model_dtype = next(model.parameters()).dtype
    instruction = str(clean_json["instruction"])
    clean_tokens = [int(x) for x in clean_json["clean_exact_7_tokens"]]
    clean_arm_prefix = clean_tokens[:6]
    action_dim = int(model.get_action_dim(cfg["model"]["unnorm_key"]))
    base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)

    reconstructed_rows, inputs_by_id = reconstruct_candidates(
        model=model,
        processor=processor,
        cfg=cfg,
        base_inputs=base_inputs,
        device=device,
        attack_seed=int(args.attack_seed),
        action_dim=action_dim,
    )
    verify_candidate_hashes(frozen_rows, reconstructed_rows)

    frozen_by_id = {int(row["candidate_id"]): row for row in frozen_rows}
    audit_rows: list[dict[str, Any]] = []
    for recon in reconstructed_rows:
        cid = int(recon["candidate_id"])
        official = official_decode(
            model,
            inputs_by_id[cid],
            action_dim=action_dim,
            unnorm_key=cfg["model"]["unnorm_key"],
            target_token_id=int(cfg["attack_optimizer"]["target_token_id"]),
            margin=float(cfg["attack_optimizer"]["gripper_margin"]),
            tolerance=float(cfg["gates"]["score_tie_tolerance"]),
        )
        match_count = arm_match_count(official["arm_prefix"], clean_arm_prefix)
        emitted_31744 = int(official["gripper_token"]) == 31744
        selective = emitted_31744 and match_count >= int(cfg["attack_optimizer"]["arm_gate_min_match_count"])
        audit_rows.append(
            {
                "stage": STAGE,
                "commit": git_value(["rev-parse", "HEAD"]),
                "attack_seed": int(args.attack_seed),
                "candidate_id": cid,
                "candidate_seed": int(recon["candidate_seed"]),
                "frozen_selected": int(frozen_by_id[cid].get("selected", 0) or 0),
                "surrogate_margin_frozen": frozen_by_id[cid]["surrogate_target31744_margin"],
                "surrogate_margin_recomputed": recon["surrogate_target31744_margin_recomputed"],
                "delta_sha256": recon["delta_sha256"],
                "processor_input_sha256": recon["processor_input_sha256"],
                "hash_verification": "PASS",
                "official_tokens": json.dumps(official["tokens"]),
                "official_gripper_token": int(official["gripper_token"]),
                "official_target31744_score": float(official["target_stats"]["target_token_score"]),
                "official_best_competitor_token": int(official["target_stats"]["best_competitor_token_id"]),
                "official_best_competitor_score": float(official["target_stats"]["best_competitor_score"]),
                "official_target31744_margin": float(official["target_stats"]["target_minus_best_competitor_margin"]),
                "official_score_argmax": int(official["score_invariant"]["argmax_token"]),
                "score_invariant_status": "PASS" if official["score_invariant"]["tie_aware_pass"] else "FAIL",
                "arm_prefix": json.dumps(official["arm_prefix"]),
                "clean_arm_prefix": json.dumps(clean_arm_prefix),
                "arm_prefix_match_count": int(match_count),
                "arm_prefix_match_denominator": 6,
                "emitted_31744": int(emitted_31744),
                "selective_31744": int(selective),
                "candidate_selectivity_class": "SELECTIVE_31744" if selective else ("NONSELECTIVE_31744" if emitted_31744 else "NO_31744"),
            }
        )

    result_class = classify_rand20_selectivity(
        audit_rows,
        arm_gate_min_match_count=int(cfg["attack_optimizer"]["arm_gate_min_match_count"]),
    )
    official_31744_rows = [row for row in audit_rows if int(row["emitted_31744"]) == 1]
    selective_rows = [row for row in audit_rows if int(row["selective_31744"]) == 1]
    best_row = max(audit_rows, key=lambda row: float(row["official_target31744_margin"]))
    summary = {
        "stage": STAGE,
        "commit": git_value(["rev-parse", "HEAD"]),
        "attack_seed": int(args.attack_seed),
        "candidate_count": len(audit_rows),
        "official_31744_count": len(official_31744_rows),
        "selective_31744_count": len(selective_rows),
        "nonselective_31744_count": len(official_31744_rows) - len(selective_rows),
        "arm_gate_min_match_count": int(cfg["attack_optimizer"]["arm_gate_min_match_count"]),
        "best_official_margin": float(best_row["official_target31744_margin"]),
        "best_official_margin_candidate_id": int(best_row["candidate_id"]),
        "result_class": result_class,
        "hash_verification": "PASS",
        "source_candidate_csv": str(Path(args.candidate_csv)),
        "output_dir": str(Path(args.output_dir)),
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "m3_step78_rand20_full_official_audit.csv", audit_rows, list(audit_rows[0].keys()))
    write_csv(output_dir / "m3_step78_rand20_selectivity_summary.csv", [summary], list(summary.keys()))
    write_report(
        output_dir / "M3_STEP78_RAND20_SELECTIVITY_AUDIT.md",
        summary=summary,
        output_dir=output_dir,
        candidate_csv=Path(args.candidate_csv),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
