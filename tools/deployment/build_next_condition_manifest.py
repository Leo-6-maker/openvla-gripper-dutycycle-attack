#!/usr/bin/env python3
"""Generate manifests for next Table 1 conditions from TRUE_T10 canonical manifest.

deepcopy source → patch approved fields → per-job diff.
Canary mode: explicit job/parent allowlist, isolated artifact root.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys
from pathlib import Path
import numpy as np

APPROVED_ADDITIONS = {
    "n_valid_steps", "trigger_skip_reason", "source_true_t10_job_key",
    "bridge_condition", "attack_objective", "trigger_step_override",
}
APPROVED_CHANGES = {"condition_id", "job_key", "output_dir"}

CONDITIONS = {
    "RANDOM_TIME": {
        "condition_id": "RANDOM_TIME", "execution_status": "FROZEN",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Random-Time Control",
        "random_trigger": True, "require_invalid_steps_zero": True,
    },
    "RAND_LINF": {
        "condition_id": "RAND_LINF", "execution_status": "DRY_RUN_ONLY",
        "bridge_condition": "RAND_T10", "attack_objective": None,
        "description": "RAND Linf + Student Trigger", "note": "Spec not frozen",
    },
    "EARLY_SHIFT": {
        "condition_id": "EARLY_SHIFT", "execution_status": "DRY_RUN_ONLY",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "description": "VIS + Early-Shift", "early_shift": True, "note": "Spec not frozen",
    },
    "SHUFFLED": {
        "condition_id": "SHUFFLED", "execution_status": "DRY_RUN_ONLY",
        "bridge_condition": "SHUFFLED_T10", "attack_objective": None,
        "description": "Shuffled Gradient", "note": "Spec not frozen",
    },
    "TMA": {
        "condition_id": "TMA", "execution_status": "DRY_RUN_ONLY",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "description": "Adapted TMA", "note": "Spec not frozen",
    },
    "UMA": {
        "condition_id": "UMA", "execution_status": "DRY_RUN_ONLY",
        "bridge_condition": "TRUE_T10",
        "attack_objective": "untargeted_clean_token_ce",
        "description": "UMA Untargeted CE-PGD", "note": "Spec not frozen",
    },
}

DEFAULT_EVIDENCE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1"
K_DEFAULT, GUARD = 10, 5

# ── helpers ──

def _strict_int(d, field):
    v = d.get(field)
    if v is None: raise ValueError(f"{field} MISSING")
    if type(v) is not int: raise ValueError(f"{field}={v} ({type(v).__name__}), must be exact int")
    return v

def load_jobs(path): return [json.loads(l) for l in open(path)]

def get_meta(job, require_invalid_zero=False):
    ep = os.path.join(job["output_dir"], "episode_summary.json")
    if not os.path.exists(ep): raise FileNotFoundError(f"Missing: {ep}")
    d = json.loads(open(ep).read())
    ns = _strict_int(d, "n_steps")
    inv = _strict_int(d, "invalid_feature_steps")
    if ns < 1: raise ValueError(f"n_steps={ns} < 1")
    if inv < 0 or inv > ns: raise ValueError(f"invalid={inv} out of [0,{ns}]")
    if require_invalid_zero and inv != 0:
        raise ValueError(f"invalid_feature_steps={inv} != 0 — contiguous valid-step contract required")
    return {"n_valid_steps": ns - inv, "n_steps_raw": ns, "invalid_feature_steps": inv,
            "mlp_emit_step": int(d.get("mlp_emit_step", -1)), "mlp_triggered": bool(d.get("mlp_triggered", False))}

def resolve_output(source_out, cond_id, artifact_root):
    new = source_out.replace("/TRUE_T10/formal_v1/", f"/{cond_id}/{artifact_root}/")
    if new == source_out: raise ValueError(f"Path replacement failed: {source_out}")
    return os.path.normpath(new)

def random_triggers(n_valid_list, seed=42):
    rng = np.random.RandomState(seed)
    return [int(rng.randint(GUARD, ns - K_DEFAULT + 1)) if ns >= GUARD + K_DEFAULT else None
            for ns in n_valid_list]

def early_shift_triggers(emit_list, n_valid_list):
    return [(e - K_DEFAULT) if (e >= 0 and (e - K_DEFAULT) >= GUARD and e <= ns) else None
            for e, ns in zip(emit_list, n_valid_list)]

def build_one(src, cond_id, evidence_root, artifact_root, cond_spec, ts, n_valid):
    new = copy.deepcopy(src)
    new["condition_id"] = cond_id
    new["bridge_condition"] = cond_spec["bridge_condition"]
    if cond_spec.get("attack_objective"): new["attack_objective"] = cond_spec["attack_objective"]
    new["job_key"] = src["job_key"].replace("TRUE_T10", cond_id)
    new["output_dir"] = resolve_output(src["output_dir"], cond_id,
                                        os.path.join(evidence_root, cond_id, artifact_root))
    new["trigger_step_override"] = ts if ts is not None else -1
    new["source_true_t10_job_key"] = src["job_key"]
    new["n_valid_steps"] = n_valid
    if ts is None: new["trigger_skip_reason"] = "too_short_or_no_emission"

    added = set(new.keys()) - set(src.keys())
    removed = set(src.keys()) - set(new.keys())
    changed = [k for k in (set(src.keys()) & set(new.keys())) if src.get(k) != new.get(k)]
    errors = []
    if removed: errors.append(f"REMOVED: {sorted(removed)}")
    ua = added - APPROVED_ADDITIONS
    if ua: errors.append(f"Unauthorized additions: {sorted(ua)}")
    uc = set(changed) - APPROVED_CHANGES
    if uc: errors.append(f"Unauthorized changes: {sorted(uc)}")
    if errors: raise ValueError(f"Job {src.get('job_key','?')}: {'; '.join(errors)}")
    return new, {"added": sorted(added), "changed": sorted(changed), "removed": sorted(removed)}

def build_manifest(src_jobs, cond_spec, evidence_root, artifact_root, seed=42):
    cond_id = cond_spec["condition_id"]
    req_zero = cond_spec.get("require_invalid_steps_zero", False)
    metas = [get_meta(j, require_invalid_zero=req_zero) for j in src_jobs]
    nv = [m["n_valid_steps"] for m in metas]
    es = [m["mlp_emit_step"] for m in metas]

    if cond_spec.get("random_trigger"): triggers = random_triggers(nv, seed)
    elif cond_spec.get("early_shift"): triggers = early_shift_triggers(es, nv)
    else: triggers = [None] * len(src_jobs)

    jobs, diffs, seen_k, seen_d = [], [], set(), set()
    for i, src in enumerate(src_jobs):
        j, diff = build_one(src, cond_id, evidence_root, artifact_root, cond_spec,
                            triggers[i] if i < len(triggers) else None, nv[i])
        if j["job_key"] in seen_k: raise ValueError(f"Dup key: {j['job_key']}")
        if j["output_dir"] in seen_d: raise ValueError(f"Dup dir: {j['output_dir']}")
        seen_k.add(j["job_key"]); seen_d.add(j["output_dir"])
        jobs.append(j); diffs.append(diff)

    parents = set((str(j["fold"]), str(j["state_id"]), str(j["detector_seed"])) for j in jobs)
    added_all = set(); changed_all = set()
    for d in diffs: added_all.update(d["added"]); changed_all.update(d["changed"])
    errs = [d for d in diffs if d.get("errors")]

    return jobs, {
        "n_jobs": len(jobs), "n_parents": len(parents),
        "n_trigger": sum(1 for j in jobs if j.get("trigger_step_override", -1) >= 0),
        "n_skip": sum(1 for j in jobs if j.get("trigger_step_override", -1) < 0),
        "added": sorted(added_all), "changed": sorted(changed_all),
        "job_errors": len(errs),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_t10_manifest", required=True)
    ap.add_argument("--conditions", nargs="*", default=["RANDOM_TIME"], choices=list(CONDITIONS.keys()))
    ap.add_argument("--evidence_root", default=DEFAULT_EVIDENCE)
    ap.add_argument("--canary", action="store_true", help="Canary mode: use canary_v1 artifact root")
    ap.add_argument("--canary_job_keys", nargs="*", help="Approved job keys for canary (exact match required)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    src_jobs = load_jobs(args.true_t10_manifest)
    print(f"Source: {len(src_jobs)} jobs")
    if len(src_jobs) != 162: sys.exit(f"Expected 162 source jobs, got {len(src_jobs)}")

    artifact_root = "canary_v1/formal_v1" if args.canary else "formal_v1"
    manifest_name = "manifest.jsonl"

    for cond_id in args.conditions:
        spec = CONDITIONS[cond_id]
        cond_root = os.path.join(args.evidence_root, cond_id)
        if args.canary:
            cond_root = os.path.join(cond_root, "canary_v1")
        manifest_path = os.path.join(cond_root, manifest_name)

        print(f"\n{'='*60}")
        print(f"Condition: {cond_id} [{spec['execution_status']}] — {spec['description']}")
        print(f"  Root: {cond_root}")

        jobs, report = build_manifest(src_jobs, spec, args.evidence_root, artifact_root, args.seed)
        print(f"  Jobs: {report['n_jobs']}, Parents: {report['n_parents']}")
        print(f"  Trigger: {report['n_trigger']}, Skip: {report['n_skip']}")
        print(f"  Added: {report['added']}, Changed: {report['changed']}")

        # Canary job key validation
        if args.canary_job_keys:
            jk_set = set(j["job_key"] for j in jobs)
            approved = set(args.canary_job_keys)
            if jk_set != approved:
                extra = jk_set - approved; missing = approved - jk_set
                parts = []
                if extra: parts.append(f"extra in manifest: {sorted(extra)}")
                if missing: parts.append(f"missing from manifest: {sorted(missing)}")
                sys.exit("Canary allowlist mismatch: " + "; ".join(parts))
            print(f"  Canary allowlist: {len(approved)} jobs — MATCH")

        if args.execute:
            if spec["execution_status"] != "FROZEN":
                sys.exit(f"ERROR: {cond_id} status={spec['execution_status']} (must be FROZEN)")
            if os.path.exists(manifest_path):
                sys.exit(f"ERROR: manifest exists: {manifest_path}")
            os.makedirs(cond_root, exist_ok=True)
            with open(manifest_path, "w") as f:
                for j in jobs: f.write(json.dumps(j) + "\n")
            sha = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
            print(f"  WRITTEN: {manifest_path}")
            print(f"  SHA256: {sha}")
        else:
            print(f"  DRY_RUN: {manifest_path}")

if __name__ == "__main__": main()
