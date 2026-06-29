#!/usr/bin/env python3
"""Generate manifests for next Table 1 conditions from TRUE_T10 canonical manifest.

Conditions:
  RANDOM_TIME  — same VIS payload, random trigger timing
  RAND_LINF    — random Linf perturbation, student trigger
  SHUFFLED     — shuffled gradient, student trigger
  TMA          — adapted TMA gripper attack, student trigger
  UMA          — untargeted CE-PGD, student trigger
  EARLY_SHIFT  — VIS payload, K steps BEFORE student emit

Reuses: same 54 parents, 3 perturbation replicates, same states, same anchors.
Each condition gets its own output root and condition_id.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = {
    "RANDOM_TIME": {
        "condition_id": "RANDOM_TIME",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Random-Time Control",
        "random_trigger": True,
        "trigger_range": "uniform(guard, episode_length - K)",
    },
    "RAND_LINF": {
        "condition_id": "RAND_LINF",
        "bridge_condition": "RAND_T10",
        "attack_objective": None,
        "description": "RAND Linf + Student Trigger",
        "random_trigger": False,
    },
    "SHUFFLED": {
        "condition_id": "SHUFFLED",
        "bridge_condition": "SHUFFLED_T10",
        "attack_objective": None,
        "description": "Shuffled Gradient + Student Trigger",
        "random_trigger": False,
    },
    "TMA": {
        "condition_id": "TMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "description": "Adapted TMA + Student Trigger",
        "random_trigger": False,
    },
    "UMA": {
        "condition_id": "UMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "untargeted_clean_token_ce",
        "description": "UMA Untargeted CE-PGD + Student Trigger",
        "random_trigger": False,
    },
    "EARLY_SHIFT": {
        "condition_id": "EARLY_SHIFT",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Early-Shift Control",
        "random_trigger": False,
        "early_shift": True,
        "shift_by": "K",
    },
}

EVIDENCE_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
TRUE_T10_ROOT = f"{EVIDENCE_ROOT}/TRUE_T10"


def load_true_t10_manifest(path: str) -> list[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def generate_random_trigger_steps(n_jobs: int, guard: int = 5,
                                   episode_len: int = 400, K: int = 10,
                                   seed: int = 42) -> list[int]:
    """Pre-generate random trigger steps uniform in [guard, episode_len - K]."""
    rng = np.random.RandomState(seed)
    lo = guard
    hi = episode_len - K
    return [int(rng.randint(lo, hi + 1)) for _ in range(n_jobs)]


def build_condition_manifest(true_t10_jobs: list[dict], condition_spec: dict,
                              output_root: str, seed: int = 42) -> list[dict]:
    """Build manifest for one condition from TRUE_T10 job templates."""
    cond_id = condition_spec["condition_id"]
    rand_trigger = condition_spec.get("random_trigger", False)
    early_shift = condition_spec.get("early_shift", False)

    # Pre-generate random trigger steps if needed
    if rand_trigger:
        trigger_steps = generate_random_trigger_steps(len(true_t10_jobs), seed=seed)
    elif early_shift:
        trigger_steps = generate_random_trigger_steps(len(true_t10_jobs), seed=seed + 1000)
    else:
        trigger_steps = [-1] * len(true_t10_jobs)

    new_jobs = []
    for i, job in enumerate(true_t10_jobs):
        # Derive output dir: replace TRUE_T10 with condition_id in path
        orig_out = job["output_dir"]
        new_out = orig_out.replace("/TRUE_T10/formal_v1/", f"/{cond_id}/formal_v1/")

        new_job = {
            "condition_id": cond_id,
            "bridge_condition": condition_spec["bridge_condition"],
            "attack_objective": condition_spec.get("attack_objective"),
            "fold": job["fold"],
            "state_id": job["state_id"],
            "task_id": job["task_id"],
            "detector_seed": job["detector_seed"],
            "perturbation_seed": job["perturbation_seed"],
            "output_dir": new_out,
            "trigger_step_override": trigger_steps[i] if trigger_steps[i] >= 0 else None,
            "job_key": job["job_key"].replace("TRUE_T10", cond_id),
            # Inherit from TRUE_T10 spec
            "K": job.get("K", 10),
            "arm_lock": job.get("arm_lock", False),
            "epsilon": job.get("epsilon", 0.023529411764705882),
            "optimization_steps": job.get("optimization_steps", 20),
            "preprocessing_backend": job.get("preprocessing_backend", "upstream_tf_jpeg"),
            "timing_policy": job.get("timing_policy", "student_trigger"),
            "no_emission_policy": job.get("no_emission_policy", "ITT_RETAIN"),
            "termination_policy": job.get("termination_policy", "episode_end"),
            "target_token": job.get("target_token", 31744),
            # Provenance (to be frozen in condition spec)
            "source_true_t10_job_key": job["job_key"],
            "bridge_sha256": job.get("bridge_sha256", ""),
            "worker_sha256": job.get("worker_sha256", ""),
            "detector_global_freeze_sha256": job.get("detector_global_freeze_sha256", ""),
            "state_selection_sha256": job.get("state_selection_sha256", ""),
        }
        new_jobs.append(new_job)

    return new_jobs


def main():
    ap = argparse.ArgumentParser(description="Generate next-condition manifests")
    ap.add_argument("--true_t10_manifest",
                    default=f"{TRUE_T10_ROOT}/formal_manifest.jsonl",
                    help="Path to TRUE_T10 canonical manifest")
    ap.add_argument("--conditions", nargs="*",
                    default=["RANDOM_TIME", "RAND_LINF"],
                    choices=list(CONDITIONS.keys()),
                    help="Conditions to generate")
    ap.add_argument("--output_root", default=EVIDENCE_ROOT,
                    help="Root for condition output directories")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for random trigger generation")
    ap.add_argument("--dry_run", action="store_true",
                    help="Preview only, don't write files")
    args = ap.parse_args()

    print(f"Loading TRUE_T10 manifest: {args.true_t10_manifest}")
    true_t10_jobs = load_true_t10_manifest(args.true_t10_manifest)
    print(f"  {len(true_t10_jobs)} jobs loaded")

    # Verify 162 jobs, 54 unique (fold,state,det,pert) combos
    parents = set()
    for j in true_t10_jobs:
        parents.add((j["fold"], j["state_id"], j["detector_seed"]))
    print(f"  {len(parents)} unique (fold, state, detector_seed) combos")
    if len(parents) != 54:
        print(f"  WARNING: expected 54 parents, got {len(parents)}")

    for cond_id in args.conditions:
        spec = CONDITIONS[cond_id]
        output_root = f"{args.output_root}/{cond_id}"
        manifest_path = f"{output_root}/formal_manifest.jsonl"

        print(f"\n{'='*60}")
        print(f"Condition: {cond_id} — {spec['description']}")
        print(f"  Bridge: --condition {spec['bridge_condition']}")
        if spec.get("attack_objective"):
            print(f"  Attack: --attack_objective {spec['attack_objective']}")
        if spec.get("random_trigger"):
            print(f"  Trigger: random uniform [guard, N-K]")
        if spec.get("early_shift"):
            print(f"  Trigger: K steps BEFORE student emit")
        print(f"  Output: {output_root}")

        new_jobs = build_condition_manifest(true_t10_jobs, spec, output_root, args.seed)
        print(f"  Jobs: {len(new_jobs)}")

        # Verify uniqueness
        job_keys = [j["job_key"] for j in new_jobs]
        if len(set(job_keys)) != len(job_keys):
            print("  ERROR: duplicate job_keys!")
        output_dirs = [j["output_dir"] for j in new_jobs]
        if len(set(output_dirs)) != len(output_dirs):
            print("  ERROR: duplicate output_dirs!")

        if not args.dry_run:
            os.makedirs(output_root, exist_ok=True)
            with open(manifest_path, "w") as f:
                for j in new_jobs:
                    f.write(json.dumps(j) + "\n")
            print(f"  Written: {manifest_path}")
        else:
            print(f"  DRY_RUN: would write {manifest_path}")
            # Show sample job
            print(f"  Sample: {json.dumps(new_jobs[0], indent=2)[:300]}...")


if __name__ == "__main__":
    main()
