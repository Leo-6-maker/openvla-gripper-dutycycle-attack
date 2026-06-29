#!/usr/bin/env python3
"""Generate manifests for next Table 1 conditions from TRUE_T10 canonical manifest.

Strict field-diff: only approved fields may change from TRUE_T10 template.
Parameters inherited from TRUE_T10 manifest, not hardcoded.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np

# Fields that MAY differ between TRUE_T10 and matched controls
APPROVED_CHANGE_ALLOWLIST = {
    "condition_id", "bridge_condition", "attack_objective",
    "job_key", "output_dir", "trigger_step_override",
    "source_true_t10_job_key", "n_valid_steps",
    "trigger_skip_reason",
}

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
    },
    "EARLY_SHIFT": {
        "condition_id": "EARLY_SHIFT",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Early-Shift (K steps BEFORE student emit)",
        "early_shift": True,
    },
    "SHUFFLED": {
        "condition_id": "SHUFFLED",
        "bridge_condition": "SHUFFLED_T10",
        "attack_objective": None,
        "description": "Shuffled Gradient + Student Trigger",
        "note": "Spec not yet frozen — dry_run only",
    },
    "TMA": {
        "condition_id": "TMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "description": "Adapted TMA + Student Trigger",
        "note": "Spec not yet frozen — dry_run only",
    },
    "UMA": {
        "condition_id": "UMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "untargeted_clean_token_ce",
        "description": "UMA Untargeted CE-PGD + Student Trigger",
        "note": "Spec not yet frozen — dry_run only",
    },
}

DEFAULT_EVIDENCE_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
K_DEFAULT = 10
GUARD = 5


def load_true_t10_jobs(path: str) -> list[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def get_episode_metadata(job: dict) -> dict:
    """Read per-episode metadata STRICTLY. Missing/bad → raises."""
    ep_path = os.path.join(job["output_dir"], "episode_summary.json")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"episode_summary.json missing: {ep_path}")
    d = json.loads(open(ep_path).read())
    n_valid = d.get("n_valid_steps", d.get("n_steps"))
    if n_valid is None or not isinstance(n_valid, (int, float)) or n_valid < 1:
        raise ValueError(f"Invalid n_valid_steps in {job.get('job_key','?')}: {n_valid}")
    n_valid = int(n_valid)
    mlp_emit = int(d.get("mlp_emit_step", -1))
    triggered = bool(d.get("mlp_triggered", False))
    return {"n_valid_steps": n_valid, "mlp_emit_step": mlp_emit, "mlp_triggered": triggered}


def resolve_output_dir(true_t10_output: str, condition_id: str, evidence_root: str) -> str:
    """Resolve output_dir for new condition from TRUE_T10 path."""
    cond_root = os.path.join(evidence_root, condition_id)
    # Replace TRUE_T10 with condition_id in the path
    new_path = true_t10_output.replace("/TRUE_T10/", f"/{condition_id}/")
    if new_path == true_t10_output:
        raise ValueError(f"Path replacement failed: {true_t10_output}")
    resolved = os.path.normpath(new_path)
    cond_root_resolved = os.path.normpath(cond_root)
    if not resolved.startswith(cond_root_resolved + os.sep):
        raise ValueError(f"Output {resolved} not under {cond_root_resolved}")
    return resolved


def generate_random_trigger(n_valid_list: list[int], seed: int = 42) -> list[int | None]:
    """Generate random trigger steps. Returns None for episodes too short."""
    rng = np.random.RandomState(seed)
    results = []
    for ns in n_valid_list:
        if ns < GUARD + K_DEFAULT:
            results.append(None)  # Episode too short for random trigger
        else:
            results.append(int(rng.randint(GUARD, ns - K_DEFAULT + 1)))
    return results


def generate_early_shift(emit_steps: list[int], n_valid_list: list[int]) -> list[int | None]:
    """EARLY_SHIFT: trigger = mlp_emit - K. None if invalid."""
    results = []
    for emit, ns in zip(emit_steps, n_valid_list):
        if emit < 0:
            results.append(None)
        else:
            ts = emit - K_DEFAULT
            if ts >= GUARD and ts + K_DEFAULT <= ns:
                results.append(ts)
            else:
                results.append(None)
    return results


def build_manifest(true_t10_jobs: list[dict], condition_spec: dict,
                    evidence_root: str, seed: int = 42) -> tuple[list[dict], dict]:
    """Build manifest with strict validation. Returns (jobs, report)."""
    cond_id = condition_spec["condition_id"]

    # Strict metadata collection
    metas = []
    errors = []
    for j in true_t10_jobs:
        try:
            metas.append(get_episode_metadata(j))
        except Exception as e:
            errors.append(f"{j.get('job_key','?')}: {e}")
    if errors:
        for e in errors[:10]:
            print(f"METADATA ERROR: {e}")
        raise RuntimeError(f"{len(errors)} episodes failed metadata extraction")

    n_valid_list = [m["n_valid_steps"] for m in metas]
    emit_steps = [m["mlp_emit_step"] for m in metas]

    if condition_spec.get("random_trigger"):
        trigger_steps = generate_random_trigger(n_valid_list, seed=seed)
    elif condition_spec.get("early_shift"):
        trigger_steps = generate_early_shift(emit_steps, n_valid_list)
    else:
        trigger_steps = [None] * len(true_t10_jobs)

    # Inherit attack parameters from TRUE_T10 manifest (first job as template)
    t10_j0 = true_t10_jobs[0]
    inherited = {
        "K": t10_j0.get("K", K_DEFAULT),
        "arm_lock": t10_j0.get("arm_lock", False),
        "epsilon": t10_j0.get("epsilon", 0.023529411764705882),
        "optimization_steps": t10_j0.get("optimization_steps", 20),
        "preprocessing_backend": t10_j0.get("preprocessing_backend", "upstream_tf_jpeg"),
        "target_token": t10_j0.get("target_token", 31744),
    }

    new_jobs = []
    seen_keys = set()
    seen_dirs = set()

    for i, job in enumerate(true_t10_jobs):
        new_out = resolve_output_dir(job["output_dir"], cond_id, evidence_root)
        jk = job["job_key"].replace("TRUE_T10", cond_id)
        if jk in seen_keys:
            raise ValueError(f"Duplicate job_key: {jk}")
        seen_keys.add(jk)
        if new_out in seen_dirs:
            raise ValueError(f"Duplicate output_dir: {new_out}")
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
            "trigger_step_override": ts if ts is not None else -1,
            "job_key": jk,
            "source_true_t10_job_key": job["job_key"],
            "n_valid_steps": n_valid_list[i],
        }
        # Inherit attack parameters (not hardcoded)
        for k, v in inherited.items():
            new_job[k] = v
        if ts is None:
            new_job["trigger_step_override"] = -1
            new_job["trigger_skip_reason"] = "episode_too_short" if condition_spec.get("random_trigger") else \
                ("no_emission_or_out_of_bounds" if condition_spec.get("early_shift") else "not_applicable")
        new_jobs.append(new_job)

    # Verify counts
    parents = set()
    for j in new_jobs:
        parents.add((str(j["fold"]), str(j["state_id"]), str(j["detector_seed"])))
    if len(parents) != 54:
        raise ValueError(f"Parent count: {len(parents)} != 54")
    if len(new_jobs) != 162:
        raise ValueError(f"Job count: {len(new_jobs)} != 162")

    # Field-diff: verify only approved fields changed
    t10_keys = set(true_t10_jobs[0].keys())
    new_keys = set(new_jobs[0].keys())
    added = new_keys - t10_keys
    removed = t10_keys - new_keys
    changed = set()
    for k in t10_keys & new_keys:
        if k in ("job_key", "output_dir", "condition_id"):
            continue  # Expected to differ
        if true_t10_jobs[0].get(k) != new_jobs[0].get(k):
            changed.add(k)
    unauthorized = (added | changed) - APPROVED_CHANGE_ALLOWLIST
    if unauthorized:
        raise ValueError(
            f"Unauthorized field changes (not in allowlist): {sorted(unauthorized)}. "
            f"Added: {sorted(added)}, Changed: {sorted(changed)}")

    report = {
        "n_jobs": len(new_jobs), "n_parents": len(parents),
        "n_with_trigger": sum(1 for j in new_jobs if j.get("trigger_step_override", -1) >= 0),
        "n_skip": sum(1 for j in new_jobs if j.get("trigger_step_override", -1) < 0),
        "inherited_params": inherited,
        "field_diff": {"added": sorted(added), "removed": sorted(removed),
                       "changed": sorted(changed), "unauthorized": sorted(unauthorized),
                       "allowlist": sorted(APPROVED_CHANGE_ALLOWLIST)},
    }
    return new_jobs, report


def main():
    ap = argparse.ArgumentParser(description="Generate next-condition manifests")
    ap.add_argument("--true_t10_manifest", required=True)
    ap.add_argument("--conditions", nargs="*", default=["RANDOM_TIME"],
                    choices=list(CONDITIONS.keys()))
    ap.add_argument("--evidence_root", default=DEFAULT_EVIDENCE_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true", help="Actually write manifests")
    args = ap.parse_args()

    true_t10_jobs = load_true_t10_jobs(args.true_t10_manifest)
    print(f"TRUE_T10 manifest: {len(true_t10_jobs)} jobs")
    if len(true_t10_jobs) != 162:
        sys.exit(f"ERROR: expected 162 jobs, got {len(true_t10_jobs)}")

    for cond_id in args.conditions:
        spec = CONDITIONS[cond_id]
        cond_root = os.path.join(args.evidence_root, cond_id)
        manifest_path = os.path.join(cond_root, "formal_manifest.jsonl")

        print(f"\n{'='*60}")
        print(f"Condition: {cond_id} — {spec['description']}")
        if spec.get("note"):
            print(f"  NOTE: {spec['note']}")
        print(f"  Root: {cond_root}")

        new_jobs, report = build_manifest(true_t10_jobs, spec, args.evidence_root, args.seed)
        print(f"  Jobs: {report['n_jobs']}, Parents: {report['n_parents']}")
        print(f"  Trigger: {report['n_with_trigger']}, Skip: {report['n_skip']}")
        print(f"  Inherited: K={report['inherited_params']['K']}, "
              f"eps={report['inherited_params']['epsilon']:.4f}, "
              f"arm_lock={report['inherited_params']['arm_lock']}")

        if args.execute:
            if os.path.exists(manifest_path):
                sys.exit(f"ERROR: manifest already exists: {manifest_path}")
            os.makedirs(cond_root, exist_ok=True)
            with open(manifest_path, "w") as f:
                for j in new_jobs:
                    f.write(json.dumps(j) + "\n")
            sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
            print(f"  WRITTEN: {manifest_path}")
            print(f"  SHA256: {sha}")
        else:
            print(f"  DRY_RUN: would write {manifest_path}")
            # Show first job sample
            j0 = new_jobs[0]
            print(f"  Sample: {j0['job_key']} trigger={j0['trigger_step_override']} "
                  f"n_valid={j0['n_valid_steps']}")


if __name__ == "__main__":
    main()
