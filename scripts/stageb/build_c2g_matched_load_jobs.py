#!/usr/bin/env python3
"""Build the frozen five-condition C2g matched-load job manifest.

Inputs are a preregistered parent manifest and detector timing extracted from a
clean detector-only pass. The random-time start is deterministic, burst-feasible,
and different from detector timing. No attacked outcome is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_matched_load_manifest import (
    AttackLoadSpec,
    CORE_CONDITIONS,
    deterministic_objective_seed,
    validate_core_2x2_manifest,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("parents", value.get("episodes", value)) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a list of objects")
    return [dict(row) for row in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_random_start(
    parent_key: str,
    *,
    minimum: int,
    maximum_inclusive: int,
    detector_start: int,
    master_seed: int,
) -> int:
    candidates = [step for step in range(minimum, maximum_inclusive + 1) if step != detector_start]
    if not candidates:
        raise ValueError(f"no random-time candidate for {parent_key}")
    material = f"C2G_RANDOM_TIME|{master_seed}|{parent_key}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(candidates)
    return int(candidates[index])


def normalize_parent(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "parent_key", "suite", "task_index", "state_id", "eval_seed",
        "clean_parent_sha256", "initial_state_sha256",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError("parent missing fields: " + ", ".join(missing))
    return {
        "parent_key": str(row["parent_key"]),
        "suite": str(row["suite"]),
        "task_index": int(row["task_index"]),
        "state_id": int(row["state_id"]),
        "eval_seed": int(row["eval_seed"]),
        "clean_parent_sha256": str(row["clean_parent_sha256"]),
        "initial_state_sha256": str(row["initial_state_sha256"]),
        "max_steps": int(row.get("max_steps", 300)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--detector-timing", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=6.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=(6.0 / 255.0) * 0.075)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument(
        "--num-loss-forwards-per-frame",
        type=int,
        default=0,
        help="0 selects K+1 for the audited target-token VIS-PGD path",
    )
    parser.add_argument("--num-backwards-per-frame", type=int, default=0, help="0 selects K")
    parser.add_argument("--projection", default="processor_space_linf_fp32_then_model_cast")
    parser.add_argument("--cast-policy", default="budget_safe_bf16_or_fp16")
    parser.add_argument("--preprocessing", default="official_pil_lanczos_center_crop_224")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--random-start-policy", default="uniform_linf_seeded")
    parser.add_argument("--temporal-init-policy", default="prev_delta")
    parser.add_argument(
        "--control-objective",
        choices=("SHUFFLED_GRIPPER_GRADIENT", "RANDOM_DIRECTION_PGD_LOOP", "NONGRIPPER_VIS_PGD"),
        default="SHUFFLED_GRIPPER_GRADIENT",
    )
    args = parser.parse_args(argv)

    parents = [normalize_parent(row) for row in read_rows(args.parents)]
    if len({row["parent_key"] for row in parents}) != len(parents):
        raise ValueError("parent manifest contains duplicate parent_key")
    timing_rows = read_rows(args.detector_timing)
    timing = {str(row["parent_key"]): int(row["detector_start_step"]) for row in timing_rows}
    if len(timing) != len(timing_rows):
        raise ValueError("detector timing manifest contains duplicate parent_key")
    checkpoint_sha = sha256_file(args.checkpoint.resolve())
    config_sha = sha256_file(args.detector_config.resolve())
    num_loss_forwards = (
        args.num_loss_forwards_per_frame
        if args.num_loss_forwards_per_frame > 0
        else args.pgd_steps + 1
    )
    num_backwards = (
        args.num_backwards_per_frame
        if args.num_backwards_per_frame > 0
        else args.pgd_steps
    )
    load = AttackLoadSpec(
        burst_length=args.burst_length,
        epsilon=args.epsilon,
        step_size=args.step_size,
        pgd_steps=args.pgd_steps,
        projection=args.projection,
        cast_policy=args.cast_policy,
        preprocessing=args.preprocessing,
        image_height=args.image_height,
        image_width=args.image_width,
        random_start_policy=args.random_start_policy,
        temporal_init_policy=args.temporal_init_policy,
        num_loss_forwards_per_frame=num_loss_forwards,
        num_backwards_per_frame=num_backwards,
        num_adv_decodes_per_frame=1,
    )
    load.validate()

    jobs: list[dict[str, Any]] = []
    for parent in parents:
        key = parent["parent_key"]
        if key not in timing:
            raise KeyError(f"detector timing missing parent {key}")
        detector_start = timing[key]
        latest = parent["max_steps"] - args.burst_length
        if detector_start < 0 or detector_start > latest:
            raise ValueError(f"detector start for {key} is not burst-feasible")
        random_start = deterministic_random_start(
            key,
            minimum=0,
            maximum_inclusive=latest,
            detector_start=detector_start,
            master_seed=args.master_seed,
        )
        for condition in CORE_CONDITIONS:
            clean = condition == "CLEAN"
            detector_timing = condition.startswith("DET_")
            gripper = "GRIPPER" in condition and "RANDOM" not in condition
            objective = "NONE" if clean else (
                "GRIPPER_TARGETED_VIS_PGD" if gripper else args.control_objective
            )
            # Random initialization and any objective-specific stochasticity are
            # paired across detector and random-time rows of the same objective.
            seed_family = "CLEAN" if clean else objective
            job = {
                **{key_name: parent[key_name] for key_name in (
                    "parent_key", "suite", "task_index", "state_id", "eval_seed",
                    "clean_parent_sha256", "initial_state_sha256",
                )},
                "condition": condition,
                "detector_checkpoint_sha256": checkpoint_sha,
                "detector_config_sha256": config_sha,
                "timing_source": "NONE" if clean else (
                    "DETECTOR" if detector_timing else "RANDOM_TIME_MATCHED"
                ),
                "objective_family": objective,
                "objective_seed": deterministic_objective_seed(key, seed_family, args.master_seed),
                "attack_enabled": not clean,
                "expected_attacked_frames": 0 if clean else args.burst_length,
                "planned_start_step": None if clean else (
                    detector_start if detector_timing else random_start
                ),
                "load_spec": asdict(load),
                "max_steps": parent["max_steps"],
                "checkpoint_path": str(args.checkpoint.resolve()),
                "detector_config_path": str(args.detector_config.resolve()),
            }
            jobs.append(job)

    summary = validate_core_2x2_manifest(jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    report = {
        "status": "PASS_C2G_MATCHED_LOAD_JOBS_BUILT",
        "jobs_path": str(args.output.resolve()),
        "jobs_sha256": sha256_file(args.output.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "detector_config_sha256": config_sha,
        "master_seed": args.master_seed,
        "validation": summary,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
