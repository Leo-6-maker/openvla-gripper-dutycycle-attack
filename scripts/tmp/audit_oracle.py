#!/usr/bin/env python3
"""Audit COMMAND_OPEN_ORACLE results and freeze envelope."""
import os, json, csv, time

ORACLE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/formal_v1"
TRUE_T10 = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/formal_v1"
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/COMMAND_OPEN_ORACLE_V1"
os.makedirs(OUT, exist_ok=True)

def collect(base):
    data = {}
    for fold in sorted(os.listdir(base)):
        fp = os.path.join(base, fold)
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
                    k = (fold_id, sid, did, pid)
                    tel = os.path.join(dp, pd, "step_telemetry.csv")
                    atk = 0; env_open = 0
                    if os.path.exists(tel):
                        for row in csv.DictReader(open(tel)):
                            if row.get("attack_this") == "True":
                                atk += 1
                                if float(row.get("env_gripper", 0)) < 0:
                                    env_open += 1
                    data[k] = {"success": d.get("task_success", False),
                               "n_steps": d.get("n_steps", 0),
                               "attack_frames": atk, "env_open_frames": env_open}
    return data

oracle = collect(ORACLE)
tt = collect(TRUE_T10)
tt_emit = {k: v for k, v in tt.items() if v.get("attack_frames", 0) > 0}

common = set(tt_emit.keys()) & set(oracle.keys())
oracle_sr = sum(1 for k in common if oracle[k]["success"]) / max(1, len(common))
atk_exec = sum(1 for k in common if oracle[k]["attack_frames"] > 0) / max(1, len(common))
env_open_rate = sum(1 for k in common if oracle[k]["env_open_frames"] > 0) / max(1, len(common))

envelope = {
    "gate": "COMMAND_OPEN_ORACLE_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_episodes": len(oracle), "matched_emission_keys": len(common),
    "oracle_success_rate": round(oracle_sr, 4),
    "attack_execution_rate": round(atk_exec, 4),
    "env_open_rate": round(env_open_rate, 4),
    "oracle_protocol": "env_gripper_force_open_continuous",
    "status": "FROZEN",
}
with open(os.path.join(OUT, "FREEZE_ENVELOPE.json"), "w") as f:
    json.dump(envelope, f, indent=2)

# Per-episode ledger
with open(os.path.join(OUT, "ORACLE_LEDGER.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["fold","state_id","det_seed","pert_seed",
        "oracle_success","oracle_n_steps","oracle_atk_frames","oracle_env_open_frames"])
    w.writeheader()
    for k in sorted(common):
        w.writerow({"fold": k[0], "state_id": k[1], "det_seed": k[2], "pert_seed": k[3],
                     "oracle_success": oracle[k]["success"], "oracle_n_steps": oracle[k]["n_steps"],
                     "oracle_atk_frames": oracle[k]["attack_frames"],
                     "oracle_env_open_frames": oracle[k]["env_open_frames"]})

print(f"Oracle audit: {len(oracle)} total, {len(common)} emission-matched")
print(f"  SR={oracle_sr:.3f} atk_exec={atk_exec:.3f} env_open={env_open_rate:.3f}")
print(f"  Freeze: {OUT}")
