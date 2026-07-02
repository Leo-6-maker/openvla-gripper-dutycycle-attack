#!/usr/bin/env python3
"""Scientific canary/formal validator for SOTA conditions — fail-closed.

Validates per-episode: attack execution, objective, budget, fallback, trigger, timing.
Consumes runtime provenance from episode_summary (bridge writes from attack_result.debug).
Writes CANARY_PASS.json / FORMAL_PASS.json only when all gates pass. sys.exit(1) on fail.
"""
import argparse, json, os, sys, csv, time, hashlib, subprocess as _sp
from collections import defaultdict

EPSILON = 0.023529411764705882  # 2/255
EPSILON_TOL = 1e-6
EXPECTED_PGD = 20
K = 10

CONDITION_SPECS = {
    "COMMAND_OPEN_ORACLE": {
        "mode": "oracle", "attack_channel": "environment_command_override",
        "checks": ["oracle_per_row"],
    },
    "TMA_STUDENT": {
        "mode": "student_triggered", "attack_objective": "vanilla_tma_gripper_open_ce",
        "checks": ["vis_attack_runtime", "tma_semantics"],
    },
    "TMA_RANDOM_TIME": {
        "mode": "random_time", "attack_objective": "vanilla_tma_gripper_open_ce",
        "checks": ["vis_attack_runtime", "tma_semantics", "random_window_legal"],
    },
    "UMA_STUDENT": {
        "mode": "student_triggered", "attack_objective": "untargeted_clean_token_ce",
        "checks": ["vis_attack_runtime", "uma_semantics"],
    },
    "SHUFFLED_STUDENT": {
        "mode": "student_triggered",
        "attack_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "checks": ["vis_attack_runtime", "gradient_transform_is_permute"],
    },
}


def load_episode(artifact_dir):
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


def validate_oracle(ep_data, spec, job):
    """Oracle: per-row validation on all 10 attack rows."""
    errors = []
    atk_rows = ep_data.get("attack_rows", [])
    summary = ep_data.get("summary", {})

    if len(atk_rows) != 10:
        errors.append(f"attack_rows={len(atk_rows)} != 10"); return errors

    # Per-row
    if not all(r.get("attack_channel", "") == "environment_command_override" for r in atk_rows):
        errors.append("not all rows have correct attack_channel")
    if not all(r.get("oracle_env_override_active") in ("True", "true", True) for r in atk_rows):
        errors.append("not all rows have override_active")
    if not all(float(r.get("oracle_env_action_after_override", 0)) <= -0.99 for r in atk_rows):
        errors.append("not all rows have grip==-1.0")
    diffs = [float(r["oracle_arm_max_abs_diff"]) for r in atk_rows if r.get("oracle_arm_max_abs_diff") not in (None, "")]
    if len(diffs) != 10:
        errors.append(f"arm_diff count={len(diffs)} != 10")
    if diffs and max(diffs) > 1e-7:
        errors.append(f"max_arm_diff={max(diffs):.2e} > 1e-7")
    # Contiguous steps
    steps = sorted(int(r["step"]) for r in atk_rows)
    if steps != list(range(steps[0], steps[0] + 10)):
        errors.append(f"steps not contiguous: {steps}")
    # Trigger match
    tt_emit = summary.get("mlp_emit_step", -1)
    if steps[0] != tt_emit:
        errors.append(f"trigger={steps[0]} != tt_emit={tt_emit}")
    # Condition + token_metric
    if summary.get("condition") != "COMMAND_OPEN_ORACLE":
        errors.append("condition mismatch")
    if summary.get("token_metric_applicable") is not False:
        errors.append("token_metric_applicable != false")
    return errors


def validate_vis_attack(ep_data, spec, job):
    """VIS attack: strict runtime provenance consumption. All fields required."""
    errors = []
    summary = ep_data.get("summary", {})
    atk_rows = ep_data.get("attack_rows", [])

    # Attack frames count
    if len(atk_rows) != K:
        errors.append(f"attack_rows={len(atk_rows)} != {K}")
        return errors

    # Contiguous steps
    steps = sorted(int(r["step"]) for r in atk_rows)
    if steps != list(range(steps[0], steps[0] + K)):
        errors.append(f"steps not contiguous: {steps}")

    # Trigger match with manifest
    manifest_trigger = job.get("trigger_step_override", -1)
    if manifest_trigger >= 0 and steps[0] != manifest_trigger:
        errors.append(f"trigger mismatch: actual={steps[0]} manifest={manifest_trigger}")

    # ── Runtime provenance (STRICT: missing = FAIL) ──
    req_obj_set = summary.get("requested_objective_set")
    res_obj_set = summary.get("resolved_objective_set")
    fallback = summary.get("fallback_used_any")
    nb_set = summary.get("num_backwards_set")
    adapter_set = summary.get("resolved_adapter_class_set")
    linf_max = summary.get("actual_linf_max")
    gt = summary.get("gradient_transform")
    delta_shas = summary.get("delta_final_sha256_set", [])
    method_set = summary.get("attack_method_set", [])

    expected_obj = spec.get("attack_objective", "")
    if req_obj_set is None:
        errors.append("requested_objective_set: MISSING")
    elif req_obj_set != [expected_obj]:
        errors.append(f"requested_objective_set={req_obj_set} != [{expected_obj}]")

    if res_obj_set is None:
        errors.append("resolved_objective_set: MISSING")
    elif res_obj_set != [expected_obj]:
        errors.append(f"resolved_objective_set={res_obj_set} != [{expected_obj}]")

    if fallback is None:
        errors.append("fallback_used_any: MISSING")
    elif fallback is not False:
        reasons = summary.get("fallback_reasons", [])
        errors.append(f"fallback_used_any=true, reasons={reasons}")

    if nb_set is None:
        errors.append("num_backwards_set: MISSING")
    elif nb_set != [EXPECTED_PGD]:
        errors.append(f"num_backwards_set={nb_set} != [{EXPECTED_PGD}]")

    if adapter_set is None:
        errors.append("resolved_adapter_class_set: MISSING")
    elif "TokenPrefixPGDAttacker" not in str(adapter_set):
        errors.append(f"adapter_class_set={adapter_set}, expected TokenPrefixPGDAttacker")

    if linf_max is None:
        errors.append("actual_linf_max: MISSING")
    elif not (0 < float(linf_max) <= EPSILON + EPSILON_TOL):
        errors.append(f"actual_linf_max={linf_max} not in (0, {EPSILON + EPSILON_TOL}]")

    if gt is None:
        errors.append("gradient_transform: MISSING")
    # Specific checks per condition handled below

    if method_set is None:
        errors.append("attack_method_set: MISSING")
    elif not any("token_prefix_pgd" in str(m).lower() for m in method_set):
        errors.append(f"attack_method_set={method_set}, expected token_prefix_pgd")

    if not delta_shas:
        errors.append("delta_final_sha256_set: empty")
    if len(delta_shas) != len(set(delta_shas)):
        errors.append("delta_final_sha256_set: not all frames have same delta SHA")

    # Provenance frame count must equal attack frame count
    prov_count = summary.get("provenance_frame_count")
    if prov_count is None:
        errors.append("provenance_frame_count: MISSING")
    elif prov_count != K:
        errors.append(f"provenance_frame_count={prov_count} != {K}")

    # Per-frame linf: all 10 values must be present and bounded
    linf_per_frame = summary.get("actual_linf_per_frame")
    if linf_per_frame is None:
        errors.append("actual_linf_per_frame: MISSING")
    elif len(linf_per_frame) != K:
        errors.append(f"actual_linf_per_frame count={len(linf_per_frame)} != {K}")
    elif not all(0 < v <= EPSILON + EPSILON_TOL for v in linf_per_frame):
        errors.append(f"actual_linf_per_frame out of bounds")

    # Delta SHA: must have 10 present values (not necessarily identical)
    dsha_count = summary.get("delta_sha_present_count")
    if dsha_count is None:
        errors.append("delta_sha_present_count: MISSING")
    elif dsha_count != K:
        errors.append(f"delta_sha_present_count={dsha_count} != {K}")

    return errors


def validate_tma_semantics(ep_data, spec, job):
    """TMA: strict semantics — token 31744 CE, minimize loss, gripper_only."""
    errors = []
    summary = ep_data.get("summary", {})
    # Loss direction: minimize
    ld_set = summary.get("loss_direction_set")
    if ld_set is None: errors.append("loss_direction_set: MISSING")
    elif ld_set != ["minimize"]: errors.append(f"loss_direction_set={ld_set} != [minimize]")
    # Target token
    tok_set = summary.get("attack_target_gripper_token_id_set")
    if tok_set is None: errors.append("attack_target_gripper_token_id_set: MISSING")
    elif tok_set != [31744]: errors.append(f"target_token_id_set={tok_set} != [31744]")
    # Label source
    ls_set = summary.get("token_label_source_set")
    if ls_set is None: errors.append("token_label_source_set: MISSING")
    elif not any("vanilla_tma_gripper_open_ce" in s for s in ls_set):
        errors.append(f"token_label_source_set={ls_set}, expected vanilla_tma_gripper_open_ce")
    # Gripper only
    go = summary.get("gripper_only_loss")
    if go is None: errors.append("gripper_only_loss: MISSING")
    elif go is not True: errors.append(f"gripper_only_loss={go} != True")
    return errors


def validate_uma_semantics(ep_data, spec, job):
    """UMA: strict semantics — maximize clean CE, untargeted, clean labels."""
    errors = []
    summary = ep_data.get("summary", {})
    ld_set = summary.get("loss_direction_set")
    if ld_set is None: errors.append("loss_direction_set: MISSING")
    elif ld_set != ["maximize"]: errors.append(f"loss_direction_set={ld_set} != [maximize]")
    ls_set = summary.get("token_label_source_set")
    if ls_set is None: errors.append("token_label_source_set: MISSING")
    elif ls_set != ["clean_model_output_sequences"]:
        errors.append(f"token_label_source_set={ls_set} != [clean_model_output_sequences]")
    cl_count = summary.get("clean_token_label_ids_present_count")
    if cl_count is None: errors.append("clean_token_label_ids_present_count: MISSING")
    elif cl_count != 10: errors.append(f"clean_token_label_ids_present_count={cl_count} != 10")
    return errors


def validate_permute(ep_data, spec, job):
    """SHUFFLED: gradient_transform must be exactly 'permute'."""
    errors = []
    summary = ep_data.get("summary", {})
    gt = summary.get("gradient_transform")
    if gt is None:
        errors.append("gradient_transform: MISSING")
    elif gt != "permute":
        errors.append(f"gradient_transform={gt} != permute")
    gts = summary.get("gradient_transform_seed_set")
    if gts is None:
        errors.append("gradient_transform_seed_set: MISSING")
    elif len(gts) == 0 or -1 in gts:
        errors.append(f"gradient_transform_seed invalid: {gts}")
    return errors


def validate_random_window(ep_data, spec, job):
    """TMA Random-Time: verify V3 schedule window is legal."""
    errors = []
    summary = ep_data.get("summary", {})
    trigger = job.get("trigger_step_override", -1)
    n_steps = summary.get("n_steps", 0)
    # V3 schedule already frozen & validated; only check window bounds
    if trigger + K > n_steps:
        errors.append(f"window [{trigger}, {trigger+K}) exceeds n_steps={n_steps}")
    tp = job.get("trigger_policy", summary.get("timing_policy", ""))
    if tp != "v3_frozen_random_schedule":
        errors.append(f"trigger_policy={tp} != v3_frozen_random_schedule")
    return errors


CHECK_FNS = {
    "oracle_per_row": validate_oracle,
    "vis_attack_runtime": validate_vis_attack,
    "tma_semantics": validate_tma_semantics,
    "uma_semantics": validate_uma_semantics,
    "gradient_transform_is_permute": validate_permute,
    "random_window_legal": validate_random_window,
}


def validate_no_emission(ep_data, job):
    """Validate a no-emission-disposition episode."""
    errors = []
    summary = ep_data.get("summary", {})
    atk_rows = ep_data.get("attack_rows", [])

    if len(atk_rows) != 0:
        errors.append(f"no-emission episode has {len(atk_rows)} attack rows")
    if job.get("attack_enabled") is not False:
        errors.append("attack_enabled != false")
    tp = job.get("trigger_policy", "")
    if tp != "disabled_no_emission_disposition":
        errors.append(f"trigger_policy={tp} != disabled_no_emission_disposition")
    ts = job.get("trigger_step_override", -1)
    if ts >= 0:
        errors.append(f"trigger_step_override={ts} >= 0 on no-emission")
    # Summary should have no attack provenance
    if summary.get("attack_frames", 0) != 0:
        errors.append("attack_frames != 0 on no-emission")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITION_SPECS.keys()))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--artifact_root", required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--mode", choices=["canary", "formal"], default="canary")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    spec = CONDITION_SPECS[args.condition]
    manifest_jobs = [json.loads(l) for l in open(args.manifest) if l.strip()]

    if len(manifest_jobs) != args.expected:
        print(f"FATAL: manifest has {len(manifest_jobs)} jobs, expected {args.expected}")
        sys.exit(1)

    # Dedup check
    seen = set(); dupes = []
    for j in manifest_jobs:
        k = (j["fold"], str(j["state_id"]), str(j["detector_seed"]), str(j["perturbation_seed"]))
        if k in seen: dupes.append(str(k))
        seen.add(k)
    if dupes:
        print(f"FATAL: {len(dupes)} duplicate keys in manifest")
        sys.exit(1)

    # ── Separate emission vs no-emission for student-triggered ──
    is_student = spec.get("mode") == "student_triggered"
    emit_jobs = []; noemit_jobs = []
    if is_student:
        for j in manifest_jobs:
            if j.get("attack_enabled", True):
                emit_jobs.append(j)
            else:
                noemit_jobs.append(j)
        print(f"Student disposition: {len(emit_jobs)} emission + {len(noemit_jobs)} no-emission")
        if args.mode == "formal":
            if len(emit_jobs) != 141 or len(noemit_jobs) != 21:
                print(f"FATAL: expected 141+21, got {len(emit_jobs)}+{len(noemit_jobs)}")
                sys.exit(1)

    # ── Validate emission episodes ──
    all_errors = {}
    n_emit_pass = 0
    for j in (emit_jobs if is_student else manifest_jobs):
        ep = load_episode(j["output_dir"])
        if ep is None:
            all_errors[j.get("job_key", "?")] = ["episode_summary missing"]
            continue
        errors = []
        for ck in spec.get("checks", []):
            fn = CHECK_FNS.get(ck)
            if fn:
                errors.extend(fn(ep, spec, j))
        if errors:
            all_errors[j.get("job_key", "?")] = errors
        else:
            n_emit_pass += 1

    # ── Validate no-emission episodes ──
    n_noemit_pass = 0
    for j in noemit_jobs:
        ep = load_episode(j["output_dir"])
        if ep is None:
            all_errors[j.get("job_key", "?")] = ["episode_summary missing (no-emit)"]
            continue
        errors = validate_no_emission(ep, j)
        if errors:
            all_errors[j.get("job_key", "?")] = errors
        else:
            n_noemit_pass += 1

    # ── Random-time: all 162 must have executed ──
    if spec.get("mode") == "random_time" and args.mode == "formal":
        executed = sum(1 for j in manifest_jobs
                       if load_episode(j["output_dir"]) and len(load_episode(j["output_dir"]).get("attack_rows", [])) > 0)
        if executed != 162:
            all_errors["random_time"] = [f"{executed}/162 executed"]
        print(f"  Random-time executed: {executed}/162")

    n_total = len(manifest_jobs)
    n_passed = n_emit_pass + n_noemit_pass
    n_failed = n_total - n_passed
    gate_pass = (n_passed == n_total and len(all_errors) == 0)

    print(f"Validator: {args.condition} [{args.mode}]")
    print(f"  Total: {n_total}, Passed: {n_passed}, Failed: {n_failed}")
    if is_student:
        print(f"  Emission pass: {n_emit_pass}/{len(emit_jobs)}, No-emission pass: {n_noemit_pass}/{len(noemit_jobs)}")

    if all_errors:
        print(f"\n  Errors ({len(all_errors)} episodes):")
        for key, errs in list(all_errors.items())[:15]:
            print(f"    {key}: {'; '.join(errs[:3])}")

    # ── SHA bindings ──
    _repo = "/mnt/sdc/dty_user/openvla_attack"
    _commit = _sp.run(["git", "-C", _repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    _bridge_sha = hashlib.sha256(open(os.path.join(_repo, "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"), "rb").read()).hexdigest()
    _worker_sha = hashlib.sha256(open(os.path.join(_repo, "scripts/stageb/run_sota_worker.py"), "rb").read()).hexdigest()
    _val_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    _mf_sha = hashlib.sha256(open(args.manifest, "rb").read()).hexdigest()

    output = {
        "condition": args.condition, "mode": args.mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_sha256": _mf_sha, "commit_sha": _commit,
        "bridge_sha256": _bridge_sha, "worker_sha256": _worker_sha,
        "validator_sha256": _val_sha,
        "total": n_total, "passed": n_passed, "failed": n_failed,
        "emit_pass": n_emit_pass, "noemit_pass": n_noemit_pass,
        "gate_pass": gate_pass,
        "spec": {"attack_objective": spec.get("attack_objective"),
                 "checks": spec.get("checks", []), "mode": spec.get("mode")},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\nVERDICT: {verdict} -> {args.output}")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
