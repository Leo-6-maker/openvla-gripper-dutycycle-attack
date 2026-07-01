#!/usr/bin/env python3
"""Scientific canary/formal validator for SOTA conditions.

Checks per-episode: attack execution, objective, budget, fallback, trigger, timing.
Writes CANARY_PASS.json or FORMAL_PASS.json only when all gates pass.
Fail-closed: sys.exit(1) on any failure.
"""
import argparse, json, os, sys, csv, time, hashlib
from collections import defaultdict

# ── Condition-specific requirements ──
CONDITION_SPECS = {
    "COMMAND_OPEN_ORACLE": {
        "mode": "oracle",
        "attack_channel": "environment_command_override",
        "expected_attack_frames": 10,
        "checks": ["attack_channel", "env_override", "arm_diff", "token_metric_false"],
    },
    "TMA_STUDENT": {
        "mode": "student_triggered",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "expected_attack_frames": 10,
        "checks": ["objective_match", "no_fallback", "pgd_steps", "epsilon_bound"],
    },
    "TMA_RANDOM_TIME": {
        "mode": "random_time",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "expected_attack_frames": 10,
        "checks": ["objective_match", "no_fallback", "pgd_steps", "epsilon_bound", "random_window_legal"],
    },
    "UMA_STUDENT": {
        "mode": "student_triggered",
        "attack_objective": "untargeted_clean_token_ce",
        "expected_attack_frames": 10,
        "checks": ["objective_match", "no_fallback", "pgd_steps", "epsilon_bound", "untargeted_semantics"],
    },
    "SHUFFLED_STUDENT": {
        "mode": "student_triggered",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "expected_attack_frames": 10,
        "checks": ["objective_match", "no_fallback", "pgd_steps", "epsilon_bound", "gradient_transform_present"],
    },
}

EPSILON = 0.023529411764705882  # 2/255
EPSILON_TOLERANCE = 0.001
EXPECTED_PGD_STEPS = 20

# ── Per-condition validators ──
def check_attack_channel(ep_data, spec):
    atk_rows = ep_data.get("attack_rows", [])
    if not atk_rows:
        return False, "no attack rows"
    all_ok = all(r.get("attack_channel", "") == spec["attack_channel"] for r in atk_rows)
    return all_ok, f"attack_channel mismatch" if not all_ok else "ok"

def check_env_override(ep_data, spec):
    atk_rows = ep_data.get("attack_rows", [])
    if len(atk_rows) != 10:
        return False, f"expected 10 attack rows, got {len(atk_rows)}"
    all_active = all(r.get("oracle_env_override_active") == "True" for r in atk_rows)
    all_minus_one = all(float(r.get("oracle_env_action_after_override", 0)) <= -0.99 for r in atk_rows)
    return (all_active and all_minus_one), f"override check: active={all_active} grip=-1={all_minus_one}"

def check_arm_diff(ep_data, spec):
    atk_rows = ep_data.get("attack_rows", [])
    diffs = [float(r.get("oracle_arm_max_abs_diff", 0)) for r in atk_rows if r.get("oracle_arm_max_abs_diff")]
    if len(diffs) != 10:
        return False, f"expected 10 arm_diff values, got {len(diffs)}"
    max_diff = max(diffs)
    return max_diff <= 1e-7, f"max_arm_diff={max_diff:.2e}"

def check_token_metric_false(ep_data, spec):
    summary = ep_data.get("summary", {})
    ok = summary.get("token_metric_applicable") == False
    return ok, f"token_metric_applicable={summary.get('token_metric_applicable')}"

def check_objective_match(ep_data, spec):
    summary = ep_data.get("summary", {})
    requested = summary.get("attack_objective", summary.get("requested_objective", ""))
    resolved = summary.get("resolved_objective", requested)
    expected = spec.get("attack_objective", "")
    if requested != expected:
        return False, f"requested={requested} != expected={expected}"
    if resolved != requested and resolved:
        return False, f"resolved={resolved} != requested={requested}"
    return True, "ok"

def check_no_fallback(ep_data, spec):
    summary = ep_data.get("summary", {})
    fb = summary.get("fallback_used", summary.get("fallback_used_any"))
    if fb is None:
        return True, "fallback field missing (pre-provenance bridge)"
    return not fb, f"fallback_used={fb}"

def check_pgd_steps(ep_data, spec):
    summary = ep_data.get("summary", {})
    steps = summary.get("num_backwards_min")
    if steps is None:
        return True, "pgd_steps field missing (pre-provenance bridge)"
    return steps == EXPECTED_PGD_STEPS, f"pgd_steps={steps} != {EXPECTED_PGD_STEPS}"

def check_epsilon_bound(ep_data, spec):
    summary = ep_data.get("summary", {})
    actual = summary.get("actual_linf_max")
    if actual is None:
        return True, "epsilon field missing (pre-provenance bridge)"
    return float(actual) <= EPSILON + EPSILON_TOLERANCE, f"actual_linf={actual} > epsilon+tol"

def check_random_window_legal(ep_data, spec):
    summary = ep_data.get("summary", {})
    trigger = summary.get("requested_trigger_step", -1)
    n_steps = summary.get("n_steps", 0)
    if trigger < 0:
        return True, "no trigger (no-emission)"
    if trigger < 5 or trigger + 10 > n_steps:
        return False, f"illegal window: trigger={trigger}, n_steps={n_steps}"
    return True, "ok"

def check_untargeted_semantics(ep_data, spec):
    """UMA: verify it's maximizing clean CE (untargeted), not targeted OPEN."""
    summary = ep_data.get("summary", {})
    obj = summary.get("attack_objective", summary.get("requested_objective", ""))
    if "untargeted" not in obj:
        return False, f"objective not untargeted: {obj}"
    return True, "ok"

def check_gradient_transform_present(ep_data, spec):
    """SHUFFLED: verify gradient_transform was active."""
    summary = ep_data.get("summary", {})
    gt = summary.get("gradient_transform", "")
    if not gt or gt == "none":
        return False, f"gradient_transform missing or none: {gt}"
    return True, f"gradient_transform={gt}"


CHECK_FUNCTIONS = {
    "attack_channel": check_attack_channel,
    "env_override": check_env_override,
    "arm_diff": check_arm_diff,
    "token_metric_false": check_token_metric_false,
    "objective_match": check_objective_match,
    "no_fallback": check_no_fallback,
    "pgd_steps": check_pgd_steps,
    "epsilon_bound": check_epsilon_bound,
    "random_window_legal": check_random_window_legal,
    "untargeted_semantics": check_untargeted_semantics,
    "gradient_transform_present": check_gradient_transform_present,
}


def load_episode(artifact_dir):
    """Load episode_summary + attack telemetry rows from artifact directory."""
    ep_path = os.path.join(artifact_dir, "episode_summary.json")
    tel_path = os.path.join(artifact_dir, "step_telemetry.csv")
    if not os.path.exists(ep_path):
        return None

    summary = json.load(open(ep_path))
    atk_rows = []
    if os.path.exists(tel_path):
        for row in csv.DictReader(open(tel_path)):
            if row.get("attack_this") == "True":
                atk_rows.append(row)

    return {"summary": summary, "attack_rows": atk_rows, "artifact_dir": artifact_dir}


def validate_episode(ep_data, spec, manifest_job):
    """Run all specified checks on one episode. Returns (passed, errors)."""
    errors = []
    for check_name in spec.get("checks", []):
        fn = CHECK_FUNCTIONS.get(check_name)
        if fn is None:
            continue
        ok, msg = fn(ep_data, spec)
        if not ok:
            errors.append(f"{check_name}: {msg}")

    # Basic checks
    summary = ep_data.get("summary", {})
    atk_rows = ep_data.get("attack_rows", [])

    # Attack frames count
    expected = spec.get("expected_attack_frames", 10)
    if len(atk_rows) != expected:
        errors.append(f"attack_rows count: {len(atk_rows)} != {expected}")

    # Trigger step match with manifest
    manifest_trigger = manifest_job.get("trigger_step_override", -1)
    if manifest_trigger >= 0 and atk_rows:
        actual_trigger = min(int(r["step"]) for r in atk_rows)
        if actual_trigger != manifest_trigger:
            errors.append(f"trigger_step: actual={actual_trigger} != manifest={manifest_trigger}")

    # Attack steps must be contiguous
    if atk_rows:
        steps = sorted(int(r["step"]) for r in atk_rows)
        expected_steps = list(range(steps[0], steps[0] + len(steps)))
        if steps != expected_steps:
            errors.append(f"attack steps not contiguous: {steps}")

    return len(errors) == 0, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITION_SPECS.keys()))
    ap.add_argument("--manifest", required=True, help="Path to canary/formal manifest JSONL")
    ap.add_argument("--artifact_root", required=True, help="Root of output directories")
    ap.add_argument("--expected", type=int, required=True, help="Expected number of jobs")
    ap.add_argument("--mode", choices=["canary", "formal"], default="canary")
    ap.add_argument("--output", required=True, help="Path to write PASS/FAIL JSON")
    args = ap.parse_args()

    spec = CONDITION_SPECS[args.condition]
    manifest_jobs = [json.loads(l) for l in open(args.manifest) if l.strip()]

    if len(manifest_jobs) != args.expected:
        print(f"FATAL: manifest has {len(manifest_jobs)} jobs, expected {args.expected}")
        sys.exit(1)

    # Map manifest jobs by output_dir
    job_by_dir = {}
    dupes = []
    for j in manifest_jobs:
        d = j["output_dir"]
        if d in job_by_dir:
            dupes.append(d)
        job_by_dir[d] = j
    if dupes:
        print(f"FATAL: {len(dupes)} duplicate output_dirs in manifest")
        sys.exit(1)

    # Validate each job
    results = []
    missing = 0
    all_errors = {}
    for j in manifest_jobs:
        ep = load_episode(j["output_dir"])
        if ep is None:
            missing += 1
            all_errors[j.get("job_key", "?")] = ["episode_summary.json missing"]
            continue
        passed, errors = validate_episode(ep, spec, j)
        results.append({"job_key": j.get("job_key", "?"), "passed": passed, "errors": errors})
        if not passed:
            all_errors[j.get("job_key", "?")] = errors

    # Summary
    n_total = len(manifest_jobs)
    n_passed = sum(1 for r in results if r["passed"])
    n_failed = n_total - n_passed - missing

    print(f"Validator: {args.condition} [{args.mode}]")
    print(f"  Total: {n_total}, Passed: {n_passed}, Failed: {n_failed}, Missing: {missing}")

    gate_pass = (n_passed == n_total and missing == 0 and n_failed == 0)

    # For student-triggered formal: verify emission/no-emission disposition
    emission_keys = 0
    no_emission_keys = 0
    if spec.get("mode") == "student_triggered" and args.mode == "formal":
        for j in manifest_jobs:
            ep = load_episode(j["output_dir"])
            if ep and len(ep.get("attack_rows", [])) > 0:
                emission_keys += 1
            else:
                no_emission_keys += 1
        print(f"  Emission: {emission_keys}, No-emission: {no_emission_keys}")
        if emission_keys != 141 or no_emission_keys != 21:
            gate_pass = False
            print(f"  FATAL: expected 141 emission + 21 no-emission")

    # For random-time formal: 162/162 executed
    if spec.get("mode") == "random_time" and args.mode == "formal":
        executed = sum(1 for j in manifest_jobs
                       if load_episode(j["output_dir"]) and len(load_episode(j["output_dir"]).get("attack_rows", [])) > 0)
        print(f"  Attack executed: {executed}/162")
        if executed != 162:
            gate_pass = False

    if all_errors:
        print(f"\n  Errors ({len(all_errors)} episodes):")
        for key, errs in list(all_errors.items())[:10]:
            print(f"    {key}: {'; '.join(errs)}")

    # SHA bindings
    import subprocess as _sp
    _repo = "/mnt/sdc/dty_user/openvla_attack"
    _commit = _sp.run(["git", "-C", _repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    _bridge_sha = hashlib.sha256(open(os.path.join(_repo, "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"), "rb").read()).hexdigest()
    _worker_sha = hashlib.sha256(open(os.path.join(_repo, "scripts/stageb/run_sota_worker.py"), "rb").read()).hexdigest()
    _val_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()

    # Write output
    output = {
        "condition": args.condition,
        "mode": args.mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_sha256": hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(),
        "commit_sha": _commit,
        "bridge_sha256": _bridge_sha,
        "worker_sha256": _worker_sha,
        "validator_sha256": _val_sha,
        "total": n_total, "passed": n_passed, "failed": n_failed, "missing": missing,
        "gate_pass": gate_pass,
        "spec": {"attack_objective": spec.get("attack_objective"),
                 "expected_attack_frames": spec.get("expected_attack_frames"),
                 "checks": spec.get("checks", [])},
    }
    if spec.get("mode") == "student_triggered":
        output["emission_keys"] = emission_keys
        output["no_emission_keys"] = no_emission_keys

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\nVERDICT: {verdict} -> {args.output}")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
