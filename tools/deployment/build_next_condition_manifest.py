#!/usr/bin/env python3
"""Generate manifests for next Table 1 conditions from TRUE_T10 canonical manifest.

Architecture: deepcopy source job → patch only approved fields → per-job diff validation.
No field removal. No parameter hardcoding. No silent defaults.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys
from pathlib import Path

import numpy as np

# Fields that MAY be ADDED (don't exist in TRUE_T10 source manifest)
APPROVED_ADDITIONS = {
    "n_valid_steps", "trigger_skip_reason", "source_true_t10_job_key",
    "bridge_condition", "attack_objective", "trigger_step_override",
}

# Fields that MAY be CHANGED from source values
APPROVED_CHANGES = {
    "condition_id", "job_key", "output_dir",
}

CONDITIONS = {
    "RANDOM_TIME": {
        "condition_id": "RANDOM_TIME",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Random-Time Control",
        "execution_status": "FROZEN",
        "random_trigger": True,
    },
    "RAND_LINF": {
        "condition_id": "RAND_LINF",
        "bridge_condition": "RAND_T10",
        "attack_objective": None,
        "description": "RAND Linf + Student Trigger",
        "execution_status": "DRY_RUN_ONLY",
        "note": "Spec not yet frozen",
    },
    "EARLY_SHIFT": {
        "condition_id": "EARLY_SHIFT",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Early-Shift (K steps BEFORE student emit)",
        "execution_status": "DRY_RUN_ONLY",
        "note": "Spec not yet frozen",
        "early_shift": True,
    },
    "SHUFFLED": {
        "condition_id": "SHUFFLED",
        "bridge_condition": "SHUFFLED_T10",
        "attack_objective": None,
        "description": "Shuffled Gradient + Student Trigger",
        "execution_status": "DRY_RUN_ONLY",
        "note": "Spec not yet frozen",
    },
    "TMA": {
        "condition_id": "TMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "description": "Adapted TMA + Student Trigger",
        "execution_status": "DRY_RUN_ONLY",
        "note": "Spec not yet frozen",
    },
    "UMA": {
        "condition_id": "UMA",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "untargeted_clean_token_ce",
        "description": "UMA Untargeted CE-PGD + Student Trigger",
        "execution_status": "DRY_RUN_ONLY",
        "note": "Spec not yet frozen",
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
    """Read per-episode metadata STRICTLY. Missing/bad → raise."""
    ep_path = os.path.join(job["output_dir"], "episode_summary.json")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"episode_summary.json missing: {ep_path}")
    d = json.loads(open(ep_path).read())
    n_steps_raw = d.get("n_steps")
    invalid = int(d.get("invalid_feature_steps", 0))
    if n_steps_raw is None or not isinstance(n_steps_raw, (int, float)) or int(n_steps_raw) < 1:
        raise ValueError(f"Invalid n_steps in {job.get('job_key','?')}: {n_steps_raw}")
    n_valid = int(n_steps_raw) - invalid
    if n_valid < 1:
        raise ValueError(f"n_valid_steps={n_valid} (n_steps={n_steps_raw} - invalid={invalid}) "
                         f"too small in {job.get('job_key','?')}")
    return {
        "n_valid_steps": n_valid,
        "n_steps_raw": int(n_steps_raw),
        "invalid_feature_steps": invalid,
        "mlp_emit_step": int(d.get("mlp_emit_step", -1)),
        "mlp_triggered": bool(d.get("mlp_triggered", False)),
    }


def resolve_output_dir(source_output: str, condition_id: str, evidence_root: str) -> str:
    cond_root = os.path.join(evidence_root, condition_id)
    new_path = source_output.replace("/TRUE_T10/", f"/{condition_id}/")
    if new_path == source_output:
        raise ValueError(f"Path replacement failed: {source_output}")
    resolved = os.path.normpath(new_path)
    if not resolved.startswith(os.path.normpath(cond_root) + os.sep):
        raise ValueError(f"Output {resolved} not under {cond_root}")
    return resolved


def generate_random_trigger(n_valid_list: list[int], seed: int = 42) -> list[int | None]:
    rng = np.random.RandomState(seed)
    results = []
    for ns in n_valid_list:
        if ns < GUARD + K_DEFAULT:
            results.append(None)
        else:
            results.append(int(rng.randint(GUARD, ns - K_DEFAULT + 1)))
    return results


def generate_early_shift(emit_steps: list[int], n_valid_list: list[int]) -> list[int | None]:
    results = []
    for emit, ns in zip(emit_steps, n_valid_list):
        if emit < 0:
            results.append(None)
        else:
            ts = emit - K_DEFAULT
            results.append(ts if (ts >= GUARD and ts + K_DEFAULT <= ns) else None)
    return results


def build_one_job(source: dict, cond_id: str, evidence_root: str,
                   cond_spec: dict, trigger_step: int | None,
                   n_valid: int) -> tuple[dict, dict]:
    """Build one new job by deepcopy+patch. Returns (job, diff_report)."""
    new = copy.deepcopy(source)

    # Patch approved fields
    new["condition_id"] = cond_id
    new["bridge_condition"] = cond_spec["bridge_condition"]
    if cond_spec.get("attack_objective"):
        new["attack_objective"] = cond_spec["attack_objective"]
    new["job_key"] = source["job_key"].replace("TRUE_T10", cond_id)
    new["output_dir"] = resolve_output_dir(source["output_dir"], cond_id, evidence_root)
    new["trigger_step_override"] = trigger_step if trigger_step is not None else -1

    # Add approved metadata
    new["source_true_t10_job_key"] = source["job_key"]
    new["n_valid_steps"] = n_valid
    if trigger_step is None:
        new["trigger_skip_reason"] = "episode_too_short_or_no_emission"

    # Diff report: compare with source
    source_keys = set(source.keys())
    new_keys = set(new.keys())
    added = new_keys - source_keys
    removed = source_keys - new_keys
    changed = []
    for k in source_keys & new_keys:
        if source.get(k) != new.get(k):
            changed.append(k)

    # Verify: no removed fields, only approved additions/changes
    errors = []
    if removed:
        errors.append(f"REMOVED fields (forbidden): {sorted(removed)}")
    unauthorized_add = added - APPROVED_ADDITIONS
    if unauthorized_add:
        errors.append(f"Unauthorized ADDITIONS: {sorted(unauthorized_add)}")
    unauthorized_change = set(changed) - APPROVED_CHANGES
    if unauthorized_change:
        errors.append(f"Unauthorized CHANGES: {sorted(unauthorized_change)}")

    if errors:
        raise ValueError(f"Job {source.get('job_key','?')}: {'; '.join(errors)}")

    diff = {"added": sorted(added), "changed": sorted(changed),
            "removed": sorted(removed), "errors": errors}
    return new, diff


def build_manifest(source_jobs: list[dict], cond_spec: dict,
                    evidence_root: str, seed: int = 42) -> tuple[list[dict], dict]:
    """Build full manifest with per-job deepcopy+diff validation."""
    cond_id = cond_spec["condition_id"]

    # Strict metadata
    metas = [get_episode_metadata(j) for j in source_jobs]
    n_valid_list = [m["n_valid_steps"] for m in metas]
    emit_steps = [m["mlp_emit_step"] for m in metas]

    if cond_spec.get("random_trigger"):
        trigger_steps = generate_random_trigger(n_valid_list, seed=seed)
    elif cond_spec.get("early_shift"):
        trigger_steps = generate_early_shift(emit_steps, n_valid_list)
    else:
        trigger_steps = [None] * len(source_jobs)

    new_jobs = []
    all_diffs = []
    seen_keys = set()
    seen_dirs = set()

    for i, src in enumerate(source_jobs):
        ts = trigger_steps[i] if i < len(trigger_steps) else None
        new_job, diff = build_one_job(src, cond_id, evidence_root, cond_spec,
                                       ts, n_valid_list[i])
        if new_job["job_key"] in seen_keys:
            raise ValueError(f"Duplicate job_key: {new_job['job_key']}")
        seen_keys.add(new_job["job_key"])
        if new_job["output_dir"] in seen_dirs:
            raise ValueError(f"Duplicate output_dir: {new_job['output_dir']}")
        seen_dirs.add(new_job["output_dir"])
        new_jobs.append(new_job)
        all_diffs.append(diff)

    # Verify counts
    parents = set()
    for j in new_jobs:
        parents.add((str(j["fold"]), str(j["state_id"]), str(j["detector_seed"])))
    if len(parents) != 54:
        raise ValueError(f"Parent count: {len(parents)} != 54")
    if len(new_jobs) != 162:
        raise ValueError(f"Job count: {len(new_jobs)} != 162")

    # Aggregate diff report
    all_added = set()
    all_changed = set()
    for d in all_diffs:
        all_added.update(d["added"])
        all_changed.update(d["changed"])
    any_errors = [d for d in all_diffs if d["errors"]]

    report = {
        "n_jobs": len(new_jobs), "n_parents": len(parents),
        "n_with_trigger": sum(1 for j in new_jobs if j.get("trigger_step_override", -1) >= 0),
        "n_skip": sum(1 for j in new_jobs if j.get("trigger_step_override", -1) < 0),
        "execution_status": cond_spec.get("execution_status", "DRY_RUN_ONLY"),
        "field_diff": {
            "added_fields": sorted(all_added),
            "changed_fields": sorted(all_changed),
            "approved_additions": sorted(APPROVED_ADDITIONS),
            "approved_changes": sorted(APPROVED_CHANGES),
            "jobs_with_errors": len(any_errors),
        },
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

    source_jobs = load_true_t10_jobs(args.true_t10_manifest)
    print(f"TRUE_T10 manifest: {len(source_jobs)} jobs")
    if len(source_jobs) != 162:
        sys.exit(f"ERROR: expected 162 jobs, got {len(source_jobs)}")

    for cond_id in args.conditions:
        spec = CONDITIONS[cond_id]
        cond_root = os.path.join(args.evidence_root, cond_id)
        manifest_path = os.path.join(cond_root, "formal_manifest.jsonl")

        print(f"\n{'='*60}")
        print(f"Condition: {cond_id} — {spec['description']}")
        print(f"  Status: {spec.get('execution_status', 'DRY_RUN_ONLY')}")
        if spec.get("note"):
            print(f"  NOTE: {spec['note']}")

        new_jobs, report = build_manifest(source_jobs, spec, args.evidence_root, args.seed)
        print(f"  Jobs: {report['n_jobs']}, Parents: {report['n_parents']}")
        print(f"  Trigger: {report['n_with_trigger']}, Skip: {report['n_skip']}")
        diff = report["field_diff"]
        print(f"  Added: {diff['added_fields']}, Changed: {diff['changed_fields']}")
        if diff["jobs_with_errors"] > 0:
            print(f"  ERRORS: {diff['jobs_with_errors']} jobs")

        if args.execute:
            if spec.get("execution_status") != "FROZEN":
                sys.exit(f"ERROR: {cond_id} execution_status={spec.get('execution_status')} "
                         f"(must be FROZEN for --execute)")
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


if __name__ == "__main__":
    main()
