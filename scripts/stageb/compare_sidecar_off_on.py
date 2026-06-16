#!/usr/bin/env python3
"""Compare reference vs shadow episodes for sidecar non-invasiveness."""
import csv, json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/liuyu/outputs/l12_sidecar_smoke"
TASKS = [
    ("alphabet_soup", "2"),
    ("bbq_sauce", "27"),
    ("butter", "2"),
    ("orange_juice", "8"),
    ("tomato_sauce", "2"),
]

results = []
for task, state in TASKS:
    ref_dir = os.path.join(OUT, f"{task}_s{state}_reference_attempt1")
    sh_dir = os.path.join(OUT, f"{task}_s{state}_shadow_attempt1")

    ref_rows = list(csv.DictReader(open(os.path.join(ref_dir, "step_trace.csv"))))
    sh_rows = list(csv.DictReader(open(os.path.join(sh_dir, "step_trace.csv"))))
    ref_act = list(csv.DictReader(open(os.path.join(ref_dir, "action_identity.csv"))))
    sh_act = list(csv.DictReader(open(os.path.join(sh_dir, "action_identity.csv"))))

    n_steps = len(ref_rows)

    qpos_mismatch = 0
    eef_mismatch = 0
    gripper_mismatch = 0
    env_gripper_mismatch = 0

    for i, (rr, sr) in enumerate(zip(ref_rows, sh_rows)):
        if rr.get("gripper_qpos_before") != sr.get("gripper_qpos_before"):
            qpos_mismatch += 1
        for axis in ["eef_x", "eef_y", "eef_z"]:
            rv = float(rr.get(axis, "nan") or "nan")
            sv = float(sr.get(axis, "nan") or "nan")
            if abs(rv - sv) > 1e-7:
                eef_mismatch += 1
                break
        if rr.get("raw_gripper") != sr.get("raw_gripper"):
            gripper_mismatch += 1
        if rr.get("env_gripper") != sr.get("env_gripper"):
            env_gripper_mismatch += 1

    # Use SHA256 hashes for exact action identity (stored in action_identity.csv)
    action_hash_diffs = 0
    env_hash_diffs = 0
    for i, (ra, sa) in enumerate(zip(ref_act, sh_act)):
        if ra.get("action_hash_post", "") != sa.get("action_hash_post", ""):
            action_hash_diffs += 1
        if ra.get("env_action_hash", "") != sa.get("env_action_hash", ""):
            env_hash_diffs += 1
    action_match = (action_hash_diffs == 0 and env_hash_diffs == 0)
    action_diffs = f"action_hash_diffs={action_hash_diffs} env_hash_diffs={env_hash_diffs}"

    ref_success_done = ref_rows[-1].get("success_done", "")
    sh_success_done = sh_rows[-1].get("success_done", "")
    ref_success_check = ref_rows[-1].get("success_check", "")
    sh_success_check = sh_rows[-1].get("success_check", "")

    emit_path = os.path.join(sh_dir, "detector_emission.json")
    emit_info = json.load(open(emit_path)) if os.path.exists(emit_path) else {}
    emit_step = emit_info.get("emit_step", -99)
    cands_path = os.path.join(sh_dir, "detector_candidates.csv")
    n_candidates = len(list(csv.DictReader(open(cands_path)))) if os.path.exists(cands_path) else 0

    results.append({
        "task": task, "state_id": state,
        "n_steps_ref": n_steps, "n_steps_sh": len(sh_rows),
        "n_actions_ref": len(ref_act), "n_actions_sh": len(sh_act),
        "steps_match": n_steps == len(sh_rows),
        "action_match": action_match,
        "action_diffs": action_diffs,
        "qpos_mismatch": qpos_mismatch,
        "eef_mismatch": eef_mismatch,
        "gripper_mismatch": gripper_mismatch,
        "env_gripper_mismatch": env_gripper_mismatch,
        "success_done_match": ref_success_done == sh_success_done,
        "success_check_match": ref_success_check == sh_success_check,
        "ref_success_done": ref_success_done,
        "sh_success_done": sh_success_done,
        "emit_step": emit_step,
        "n_candidates": n_candidates,
        "infra": "ok",
    })

out_csv = os.path.join(OUT, "l12_sidecar_off_on_identity.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

print("=== Sidecar OFF/ON Identity Results ===")
all_action_match = all(r["action_match"] for r in results)
all_steps_match = all(r["steps_match"] for r in results)
all_success_match = all(r["success_done_match"] for r in results)
all_qpos_clean = all(r["qpos_mismatch"] == 0 for r in results)
all_eef_clean = all(r["eef_mismatch"] == 0 for r in results)
all_gripper_clean = all(r["gripper_mismatch"] == 0 for r in results)

for r in results:
    status = "MATCH" if r["action_match"] else "DIFF:" + r["action_diffs"]
    print(f"{r['task']}_s{r['state_id']}: steps={r['n_steps_ref']}/{r['n_steps_sh']} actions={status} qpos={r['qpos_mismatch']} eef={r['eef_mismatch']} emit={r['emit_step']}")

print()
print("Action identity:  " + ("ALL 5/5 PASS" if all_action_match else "FAIL"))
print("Steps match:      " + ("ALL 5/5 PASS" if all_steps_match else "FAIL"))
print("Success match:    " + ("ALL 5/5 PASS" if all_success_match else "FAIL"))
print("Qpos clean:       " + ("ALL 5/5 PASS" if all_qpos_clean else "FAIL"))
print("EEF clean:        " + ("ALL 5/5 PASS" if all_eef_clean else "FAIL"))
print("Gripper clean:    " + ("ALL 5/5 PASS" if all_gripper_clean else "FAIL"))
print("CSV: " + out_csv)
