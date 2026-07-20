"""F1.1: Action semantic parity check."""
import json, sys
from pathlib import Path

S1 = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_S1_FIT_V1_5e27d7c/libero_10/task_00/state_00")
T = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels/libero_10/task_00/state_00")

s_recs = [json.loads(l) for l in (S1 / "student_input_records.jsonl").read_text().splitlines() if l.strip()]
t_recs = [json.loads(l) for l in (T / "physics_teacher_v21.jsonl").read_text().splitlines() if l.strip()]

# Feature indices: 0=gripper_command(raw) 12=action_gripper(env) 13=close_streak
# 14=open_streak 16=close_onset
print("step  raw_cmd    raw<=0.5  env_cmd    env>=0.5  T_cc      streak_c  streak_o  close_onset")
print("-" * 95)
for i in range(0, min(50, len(s_recs), len(t_recs)), 5):
    r = s_recs[i]
    t = t_recs[i]
    rc = float(r["features_25d"][0])
    ec = float(r["features_25d"][12])
    cs = float(r["features_25d"][13])
    os_val = float(r["features_25d"][14])
    co = float(r["features_25d"][16])
    t_cc = bool(t.get("candidate_close", False))
    print("{:4d}  {:9.4f}  {:8s}  {:9.4f}  {:8s}  {:8s}  {:8.1f}  {:8.1f}  {:11.1f}".format(
        i, rc, str(rc <= 0.5), ec, str(ec >= 0.5), str(t_cc), cs, os_val, co))

# Summary
N = min(len(s_recs), len(t_recs))
raw_close = [float(s_recs[i]["features_25d"][0]) <= 0.5 for i in range(N)]
env_close = [float(s_recs[i]["features_25d"][12]) >= 0.5 for i in range(N)]
t_cc = [bool(t_recs[i].get("candidate_close", False)) for i in range(N)]

raw_env_agree = sum(1 for i in range(N) if raw_close[i] == env_close[i])
raw_env_opp = sum(1 for i in range(N) if raw_close[i] != env_close[i])

t_raw_agree = sum(1 for i in range(N) if t_cc[i] == raw_close[i])
t_raw_opp = sum(1 for i in range(N) if t_cc[i] != raw_close[i])

t_env_agree = sum(1 for i in range(N) if t_cc[i] == env_close[i])
t_env_opp = sum(1 for i in range(N) if t_cc[i] != env_close[i])

print()
print("N={}".format(N))
print("Raw <= 0.5 steps: {} / Env >= 0.5 steps: {} / T_cc steps: {}".format(
    sum(raw_close), sum(env_close), sum(t_cc)))
print("Raw-Env agreement: {} / opposite: {}".format(raw_env_agree, raw_env_opp))
print("Teacher-Raw agreement: {} / opposite: {}".format(t_raw_agree, t_raw_opp))
print("Teacher-Env agreement: {} / opposite: {}".format(t_env_agree, t_env_opp))

if t_env_opp > 0:
    print("\nTeacher-Env OPPOSITE detected — candidate_close uses different action space")
if t_raw_opp > 0:
    print("Teacher-Raw OPPOSITE detected — candidate_close uses different action space")
