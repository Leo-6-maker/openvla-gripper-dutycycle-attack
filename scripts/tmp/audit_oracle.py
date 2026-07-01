#!/usr/bin/env python3
"""Audit COMMAND_OPEN_ORACLE results with hard validation gates. Fail-closed."""
import os, json, csv, sys, time

ORACLE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/formal_v1"
TRUE_T10 = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/formal_v1"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/COMMAND_OPEN_ORACLE_V1"
EXPECTED_KEYS = 141

os.makedirs(OUT, exist_ok=True)

# ── Collect TRUE_T10 emission keys and emit steps ──
tt_emit_steps = {}
for fold in sorted(os.listdir(TRUE_T10)):
    fp = os.path.join(TRUE_T10, fold)
    if not os.path.isdir(fp): continue
    fold_id = fold.split("_")[1]
    for sd in sorted(os.listdir(fp)):
        sid = int(sd.split("_")[1])
        sp = os.path.join(fp, sd)
        for dd in sorted(os.listdir(sp)):
            did = int(dd.split("_")[2])
            dp = os.path.join(sp, dd)
            for pd in sorted(os.listdir(dp)):
                pid = int(pd.split("_")[2])
                ep = os.path.join(dp, pd, "episode_summary.json")
                if not os.path.exists(ep): continue
                d = json.load(open(ep))
                emit = d.get("mlp_emit_step", -1)
                if emit >= 0:
                    tt_emit_steps[(fold_id, sid, did, pid)] = emit

# ── Collect oracle results with full validation ──
oracle_data = {}
duplicates = []
validation_errors = []

for fold in sorted(os.listdir(ORACLE)):
    fp = os.path.join(ORACLE, fold)
    if not os.path.isdir(fp): continue
    fold_id = fold.split("_")[1]
    for sd in sorted(os.listdir(fp)):
        sid = int(sd.split("_")[1])
        sp = os.path.join(fp, sd)
        for dd in sorted(os.listdir(sp)):
            did = int(dd.split("_")[2])
            dp = os.path.join(sp, dd)
            for pd in sorted(os.listdir(dp)):
                pid = int(pd.split("_")[2])
                k = (fold_id, sid, did, pid)

                if k in oracle_data:
                    duplicates.append(str(k))
                    continue

                ep = os.path.join(dp, pd, "episode_summary.json")
                if not os.path.exists(ep): continue
                d = json.load(open(ep))

                tel = os.path.join(dp, pd, "step_telemetry.csv")
                atk_rows = []
                if os.path.exists(tel):
                    for row in csv.DictReader(open(tel)):
                        if row.get("attack_this") == "True":
                            atk_rows.append(row)

                # ── Per-episode per-row validation ──
                errors = []
                if len(atk_rows) != 10:
                    errors.append(f"attack_rows={len(atk_rows)} (expected 10)")

                # Per-row checks on all 10 attack rows
                all_channel_ok = all(r.get("attack_channel", "") == "environment_command_override" for r in atk_rows)
                all_override_active = all(r.get("oracle_env_override_active") in ("True", "true", True) for r in atk_rows)
                all_grip_minus_one = all(float(r.get("oracle_env_action_after_override", 0)) <= -0.99 for r in atk_rows)
                arm_diffs = [float(r["oracle_arm_max_abs_diff"]) for r in atk_rows if r.get("oracle_arm_max_abs_diff") not in (None, "")]
                attack_steps = sorted(int(r["step"]) for r in atk_rows)

                if not all_channel_ok:
                    errors.append("not all rows have attack_channel==environment_command_override")
                if not all_override_active:
                    errors.append("not all rows have oracle_env_override_active==True")
                if not all_grip_minus_one:
                    errors.append("not all rows have oracle_env_action_after_override==-1.0")
                if len(arm_diffs) != 10:
                    errors.append(f"arm_diff values={len(arm_diffs)} (expected 10)")
                max_ad = max(arm_diffs) if arm_diffs else 0.0
                if max_ad > 1e-7:
                    errors.append(f"arm_max_abs_diff={max_ad:.2e} > 1e-7")

                # Attack steps must be contiguous [emit, emit+1, ..., emit+9]
                tt_emit = tt_emit_steps.get(k, -1)
                oracle_trigger = attack_steps[0] if attack_steps else -1
                expected_steps = list(range(oracle_trigger, oracle_trigger + 10)) if oracle_trigger >= 0 else []
                if attack_steps and attack_steps != expected_steps:
                    errors.append(f"attack steps not contiguous: {attack_steps}, expected {expected_steps}")
                if tt_emit >= 0 and oracle_trigger >= 0 and tt_emit != oracle_trigger:
                    errors.append(f"trigger_step mismatch: oracle={oracle_trigger} tt_emit={tt_emit}")

                # Summary-level checks
                if d.get("token_metric_applicable") is not False:
                    errors.append("token_metric_applicable != false")
                if d.get("condition") != "COMMAND_OPEN_ORACLE":
                    errors.append(f"condition={d.get('condition')} != COMMAND_OPEN_ORACLE")

                if errors:
                    validation_errors.append({"key": str(k), "errors": errors})

                oracle_data[k] = {
                    "success": d.get("task_success", False),
                    "n_steps": d.get("n_steps", 0),
                    "attack_frames": len(atk_rows),
                    "env_open_frames": sum(1 for r in atk_rows if float(r.get("env_gripper", 0)) < 0),
                    "arm_max_abs_diff": max_ad,
                    "trigger_step": oracle_trigger,
                    "condition": d.get("condition", ""),
                }

# ── Gate checks ──
checks = {
    "observed_keys == 141": len(oracle_data) == EXPECTED_KEYS,
    "matched_emission_keys == 141": len(set(oracle_data.keys()) & set(tt_emit_steps.keys())) == EXPECTED_KEYS,
    "extra_keys == 0": len(set(oracle_data.keys()) - set(tt_emit_steps.keys())) == 0,
    "missing_keys == 0": len(set(tt_emit_steps.keys()) - set(oracle_data.keys())) == 0,
    "duplicates == 0": len(duplicates) == 0,
    "validation_errors == 0": len(validation_errors) == 0,
}

all_pass = all(checks.values())

# ── Report ──
print(f"Oracle audit: {len(oracle_data)} episodes")
for name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {name}")
if validation_errors:
    print(f"  Validation errors ({len(validation_errors)}):")
    for ve in validation_errors[:10]:
        print(f"    {ve['key']}: {'; '.join(ve['errors'])}")
if duplicates:
    print(f"  Duplicates: {duplicates}")

# ── Write outputs ──
oracle_sr = sum(1 for v in oracle_data.values() if v["success"]) / max(1, len(oracle_data))
atk_exec = sum(1 for v in oracle_data.values() if v["attack_frames"] > 0) / max(1, len(oracle_data))

envelope = {
    "gate": "COMMAND_OPEN_ORACLE_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_episodes": len(oracle_data),
    "matched_emission_keys": len(set(oracle_data.keys()) & set(tt_emit_steps.keys())),
    "oracle_success_rate": round(oracle_sr, 4),
    "attack_execution_rate": round(atk_exec, 4),
    "oracle_protocol": "env_gripper_force_open_continuous",
    "validation_checks": checks,
    "all_checks_pass": all_pass,
    "status": "FROZEN" if all_pass else "FAILED_VALIDATION",
}
with open(os.path.join(OUT, "FREEZE_ENVELOPE.json"), "w") as f:
    json.dump(envelope, f, indent=2)
with open(os.path.join(OUT, "VALIDATION_REPORT.json"), "w") as f:
    json.dump({"checks": checks, "validation_errors": validation_errors,
               "duplicates": duplicates}, f, indent=2)

# Per-episode ledger
with open(os.path.join(OUT, "ORACLE_LEDGER.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fold","state_id","det_seed","pert_seed",
        "oracle_success","oracle_n_steps","oracle_atk_frames","oracle_env_open_frames",
        "arm_max_abs_diff","trigger_step","condition"])
    w.writeheader()
    for k in sorted(oracle_data.keys()):
        v = oracle_data[k]
        w.writerow({"fold": k[0], "state_id": k[1], "det_seed": k[2], "pert_seed": k[3],
                     "oracle_success": v["success"], "oracle_n_steps": v["n_steps"],
                     "oracle_atk_frames": v["attack_frames"],
                     "oracle_env_open_frames": v["env_open_frames"],
                     "arm_max_abs_diff": v["arm_max_abs_diff"],
                     "trigger_step": v["trigger_step"],
                     "condition": v["condition"]})

print(f"\nVERDICT: {'PASS' if all_pass else 'FAIL'}")

if not all_pass:
    print("Oracle validation FAILED — envelope status = FAILED_VALIDATION")
    sys.exit(1)
