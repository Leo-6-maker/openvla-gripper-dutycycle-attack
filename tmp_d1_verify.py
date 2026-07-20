"""Gate D1: Verify Teacher threshold root cause."""
import json
from pathlib import Path

CLEAN = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean")
TEACHER = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels")

recs = [json.loads(l) for l in (CLEAN / "libero_10/task_00/state_00/step_records.jsonl").read_text().splitlines() if l.strip()]
t_recs = [json.loads(l) for l in (TEACHER / "libero_10/task_00/state_00/physics_teacher_v21.jsonl").read_text().splitlines() if l.strip()]

print("Teacher used: clean_action_raw_7d[-1] >= 0.5 → cc=True")
print("Production uses: raw_action[-1] <= 0.5 → close")
print("Correct (OpenVLA space 0=close,1=open): raw < 0.5 → close")

agree_correct = 0
agree_wrong = 0
total = min(20, len(recs), len(t_recs))

print("\nStep  raw    raw<0.5(correct)  raw>=0.5(wrong)  env   env>0  T_cc")
for i in range(total):
    raw = recs[i]["clean_action_raw_7d"][-1]
    env = recs[i]["applied_action_7d"][-1]
    correct_close = raw < 0.5
    wrong_close = raw >= 0.5
    env_close = env > 0
    t_cc = bool(t_recs[i]["candidate_close"])

    match_correct = "FIX!" if t_cc == correct_close else "     "
    match_wrong = "OLD" if t_cc == wrong_close else "   "
    print("  {:3d}  {:.4f}  {:6s}           {:6s}       {:5.1f}  {:5s}  {:5s}  {} {}".format(
        i, raw, str(correct_close), str(wrong_close), env, str(env_close), str(t_cc),
        match_correct, match_wrong))
    if t_cc == correct_close:
        agree_correct += 1
    if t_cc == wrong_close:
        agree_wrong += 1

print("\nTeacher cc matches correct (< 0.5): {}/{}".format(agree_correct, total))
print("Teacher cc matches wrong   (>= 0.5): {}/{}".format(agree_wrong, total))
print()
if agree_wrong == total:
    print("CONFIRMED: Teacher threshold is INVERTED. Fix: use raw < 0.5.")
    print("All Teacher labels must be regenerated with correct threshold.")
