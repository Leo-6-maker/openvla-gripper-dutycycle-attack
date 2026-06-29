#!/usr/bin/env python3
"""Generate manifests for next Table 1 conditions from TRUE_T10 canonical manifest.

P0 fixes:
- EARLY_SHIFT: trigger_step = student_emit - K (not random)
- RANDOM_TIME: uses per-episode n_valid_steps from episode_summary
- Output root containment enforced
- Duplicate/collision → exit
- Default dry_run
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

import numpy as np

CONDITIONS = {
    "RANDOM_TIME": {
        "condition_id": "RANDOM_TIME",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Random-Time Control",
        "random_trigger": True,
    },
    "RAND_LINF": {
        "condition_id": "RAND_LINF",
        "bridge_condition": "RAND_T10",
        "attack_objective": None,
        "description": "RAND Linf + Student Trigger",
        "random_trigger": False,
    },
    "EARLY_SHIFT": {
        "condition_id": "EARLY_SHIFT",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Early-Shift (K steps BEFORE student emit)",
        "random_trigger": False,
        "early_shift": True,
    },
}

EVIDENCE_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
TRUE_T10_ROOT = f"{EVIDENCE_ROOT}/TRUE_T10"
K_DEFAULT = 10
GUARD = 5


def load_true_t10_jobs(path: str) -> list[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def get_episode_metadata(job: dict) -> dict:
    """Read per-episode metadata from TRUE_T10 episode_summary."""
    ep_path = os.path.join(job["output_dir"], "episode_summary.json")
    meta = {"n_steps": 400, "mlp_emit_step": -1, "mlp_triggered": False}
    if os.path.exists(ep_path):
        try:
            d = json.loads(open(ep_path).read())
            meta["n_steps"] = d.get("n_steps", 400)
            meta["mlp_emit_step"] = d.get("mlp_emit_step", -1)
            meta["mlp_triggered"] = d.get("mlp_triggered", False)
        except Exception:
            pass
    return meta


def generate_random_trigger(n_jobs: int, n_steps_list: list[int], seed: int = 42) -> list[int]:
    """Generate random trigger steps uniform in [guard, n_steps - K] per episode."""
    rng = np.random.RandomState(seed)
    return [int(rng.randint(GUARD, max(GUARD + 1, ns - K_DEFAULT + 1))) for ns in n_steps_list]


def generate_early_shift(emit_steps: list[int], n_steps_list: list[int]) -> list[int | None]:
    """EARLY_SHIFT: trigger_step = mlp_emit - K. Returns None if invalid."""
    results = []
    for emit, ns in zip(emit_steps, n_steps_list):
        if emit < 0:
            results.append(None)  # No emission → cannot early-shift, ITT retains
        else:
            ts = emit - K_DEFAULT
            if ts >= GUARD and ts + K_DEFAULT <= ns:
                results.append(ts)
            else:
                results.append(None)  # Shift would go out of bounds
    return results


def build_manifest(true_t10_jobs: list[dict], condition_spec: dict,
                    output_root: str, seed: int = 42) -> list[dict]:
    """Build manifest with strict output root containment."""
    cond_id = condition_spec["condition_id"]

    # Gather per-episode metadata
    metas = [get_episode_metadata(j) for j in true_t10_jobs]
    n_steps_list = [m["n_steps"] for m in metas]
    emit_steps = [m["mlp_emit_step"] for m in metas]

    if condition_spec.get("random_trigger"):
        trigger_steps = generate_random_trigger(len(true_t10_jobs), n_steps_list, seed=seed)
    elif condition_spec.get("early_shift"):
        trigger_steps = generate_early_shift(emit_steps, n_steps_list)
    else:
        trigger_steps = [None] * len(true_t10_jobs)

    new_jobs = []
    seen_keys = set()
    seen_dirs = set()

    for i, job in enumerate(true_t10_jobs):
        orig_out = job["output_dir"]
        # Replace only the condition name in the path
        new_out = orig_out.replace("/TRUE_T10/formal_v1/", f"/{cond_id}/formal_v1/")
        if new_out == orig_out:
            sys.exit(f"ERROR: output_dir replacement failed for {job.get('job_key','?')}: {orig_out}")
        if not new_out.startswith(f"{EVIDENCE_ROOT}/{cond_id}/"):
            sys.exit(f"ERROR: output_dir outside condition root: {new_out}")

        jk = job["job_key"].replace("TRUE_T10", cond_id)
        if jk in seen_keys:
            sys.exit(f"ERROR: duplicate job_key: {jk}")
        seen_keys.add(jk)
        if new_out in seen_dirs:
            sys.exit(f"ERROR: duplicate output_dir: {new_out}")
        seen_dirs.add(new_out)

        ts = trigger_steps[i] if i < len(trigger_steps) else None
        new_job = {
            "condition_id": cond_id,
            "bridge_condition": condition_spec["bridge_condition"],
            "attack_objective": condition_spec.get("attack_objective"),
            "fold": job["fold"], "state_id": job["state_id"],
            "task_id": job.get("task_id", 0),
            "detector_seed": job["detector_seed"], "perturbation_seed": job["perturbation_seed"],
            "output_dir": new_out,
            "trigger_step_override": ts,
            "job_key": jk,
            "K": K_DEFAULT, "arm_lock": False,
            "epsilon": 0.023529411764705882, "optimization_steps": 20,
            "preprocessing_backend": "upstream_tf_jpeg",
            "timing_policy": "student_trigger",
            "no_emission_policy": "ITT_RETAIN",
            "termination_policy": "episode_end",
            "target_token": 31744,
            "source_true_t10_job_key": job["job_key"],
        }
        # For EARLY_SHIFT no-emission episodes: no trigger override (ITT)
        if ts is None:
            new_job["trigger_step_override"] = -1
            new_job["early_shift_skip_reason"] = "no_emission_or_out_of_bounds"

        new_jobs.append(new_job)

    # Verify counts
    parents = set()
    for j in new_jobs:
        parents.add((str(j["fold"]), str(j["state_id"]), str(j["detector_seed"])))
    if len(parents) != 54:
        sys.exit(f"ERROR: {len(parents)} parents != 54")
    if len(new_jobs) != 162:
        sys.exit(f"ERROR: {len(new_jobs)} jobs != 162")

    return new_jobs


def main():
    ap = argparse.ArgumentParser(description="Generate next-condition manifests")
    ap.add_argument("--true_t10_manifest", default=f"{TRUE_T10_ROOT}/formal_manifest.jsonl")
    ap.add_argument("--conditions", nargs="*", default=["RANDOM_TIME"],
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--output_root", default=EVIDENCE_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true", help="Actually write manifests")
    args = ap.parse_args()

    true_t10_jobs = load_true_t10_jobs(args.true_t10_manifest)
    print(f"TRUE_T10: {len(true_t10_jobs)} jobs")
    if len(true_t10_jobs) != 162:
        sys.exit(f"ERROR: TRUE_T10 manifest has {len(true_t10_jobs)} jobs, expected 162")

    for cond_id in args.conditions:
        spec = CONDITIONS[cond_id]
        output_root = f"{args.output_root}/{cond_id}"
        manifest_path = f"{output_root}/formal_manifest.jsonl"

        print(f"\n{'='*60}")
        print(f"Condition: {cond_id} — {spec['description']}")
        print(f"  Output: {output_root}")

        new_jobs = build_manifest(true_t10_jobs, spec, output_root, args.seed)
        print(f"  Jobs: {len(new_jobs)}, Parents: 54")

        # Summary stats
        has_trigger = sum(1 for j in new_jobs if j.get("trigger_step_override", -1) >= 0)
        no_trigger = len(new_jobs) - has_trigger
        print(f"  With trigger override: {has_trigger}, Without: {no_trigger}")

        if args.execute:
            if os.path.exists(manifest_path):
                sys.exit(f"ERROR: manifest already exists: {manifest_path}")
            os.makedirs(output_root, exist_ok=True)
            with open(manifest_path, "w") as f:
                for j in new_jobs:
                    f.write(json.dumps(j) + "\n")
            import hashlib
            sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
            print(f"  WRITTEN: {manifest_path}")
            print(f"  SHA256: {sha}")
        else:
            print(f"  DRY_RUN: would write {manifest_path} ({len(new_jobs)} jobs)")
            print(f"  Sample job: {json.dumps(new_jobs[0])[:200]}...")


if __name__ == "__main__":
    main()
