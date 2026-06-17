#!/usr/bin/env python3
"""Repair handoff v2 evidence: real scores, exact splits, repeatability table."""
import csv, json, os

PANEL = "/data/liuyu/outputs/l12_timing_panel_v2"
RESUME = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"

acc = {}
for r in csv.DictReader(open(MANIFEST)):
    if r.get("status") == "BOUND":
        acc[(r["task"], int(r["state_id"]))] = r

labels = {}
for r in csv.DictReader(open(LABELS)):
    labels[(r["task"], int(r["state_id"]))] = r

handoff = list(csv.DictReader(open(os.path.join(PANEL, "postprocess", "l12_to_l3_timing_handoff_v2.csv"))))
print("Loaded " + str(len(handoff)) + " handoff rows")

for row in handoff:
    task = row["task"]
    sid = int(row["state_id"])
    key = (task, sid)

    if key in acc:
        row["split"] = acc[key].get("split", "train")
    else:
        row["split"] = "train"

    ep_dir = os.path.join(PANEL, task + "_s" + str(sid) + "_shadow_attempt1")
    if not os.path.exists(ep_dir):
        ep_dir = os.path.join(RESUME, task + "_s" + str(sid) + "_shadow_attempt1")

    cands_path = os.path.join(ep_dir, "detector_candidates.csv")
    if os.path.exists(cands_path):
        cands = list(csv.DictReader(open(cands_path)))
        emit_step = int(row.get("d5_emit", -1))
        score = 0.0
        for c in cands:
            if int(c.get("step", -1)) == emit_step:
                score = float(c.get("score", 0))
                break
        if score > 0:
            row["d5_score"] = str(round(score, 6))
            row["d5_tau_margin"] = str(round(score - 0.050, 6))

    succ = int(row.get("clean_success", "1"))
    if succ == 0:
        row["primary_or_diagnostic"] = "diagnostic"

    teacher = labels.get(key, {})
    if teacher:
        anchor = int(teacher.get("anchor", -1))
        row["teacher_ws"] = teacher.get("ws", row.get("teacher_ws", ""))
        row["teacher_anchor"] = teacher.get("anchor", row.get("teacher_anchor", ""))
        row["teacher_we"] = teacher.get("we", row.get("teacher_we", ""))
        emit = int(row.get("d5_emit", -1))
        if emit >= 0 and anchor >= 0:
            row["emit_anchor_offset"] = str(emit - anchor)

out_csv = os.path.join(PANEL, "postprocess", "l12_to_l3_timing_handoff_v2_fixed.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(handoff[0].keys()))
    w.writeheader()
    w.writerows(handoff)
print("Fixed handoff: " + out_csv)

# Repeatability table
REPEATS = [("butter", "11"), ("tomato_sauce", "23"), ("salad_dressing", "11")]
rep_rows = []
for task, sid in REPEATS:
    orig_dir = os.path.join(PANEL, task + "_s" + str(sid) + "_shadow_attempt1")
    rep_dir = os.path.join(PANEL, task + "_s" + str(sid) + "_shadow_attempt2")
    if not os.path.exists(orig_dir):
        orig_dir = os.path.join(RESUME, task + "_s" + str(sid) + "_shadow_attempt1")
    if not os.path.exists(rep_dir):
        rep_dir = os.path.join(RESUME, task + "_s" + str(sid) + "_shadow_attempt2")

    orig_rows = list(csv.DictReader(open(os.path.join(orig_dir, "step_trace.csv"))))
    rep_rows_data = list(csv.DictReader(open(os.path.join(rep_dir, "step_trace.csv"))))
    orig_act = list(csv.DictReader(open(os.path.join(orig_dir, "action_identity.csv"))))
    rep_act = list(csv.DictReader(open(os.path.join(rep_dir, "action_identity.csv"))))

    n_orig = len(orig_rows)
    n_rep = len(rep_rows_data)
    orig_succ = orig_rows[-1].get("success_done", "?") if orig_rows else "?"
    rep_succ = rep_rows_data[-1].get("success_done", "?") if rep_rows_data else "?"

    n = min(len(orig_act), len(rep_act))
    ad = sum(1 for i in range(n) if orig_act[i].get("action_hash_post") != rep_act[i].get("action_hash_post"))
    ed = sum(1 for i in range(n) if orig_act[i].get("env_action_hash") != rep_act[i].get("env_action_hash"))
    od = sum(1 for i in range(n) if orig_act[i].get("obs_hash", "") != rep_act[i].get("obs_hash", ""))

    orig_cands = list(csv.DictReader(open(os.path.join(orig_dir, "detector_candidates.csv"))))
    rep_cands = list(csv.DictReader(open(os.path.join(rep_dir, "detector_candidates.csv"))))
    cand_diff = 0
    abst_diff = 0
    max_s = 0.0
    if len(orig_cands) == len(rep_cands):
        for oc, rc in zip(orig_cands, rep_cands):
            if oc.get("step") != rc.get("step"): cand_diff += 1
            if int(oc.get("abstained", 0) or 0) != int(rc.get("abstained", 0) or 0): abst_diff += 1
            try:
                s1 = float(oc.get("score", "0"))
                s2 = float(rc.get("score", "0"))
                d = abs(s1 - s2)
                if d > max_s: max_s = d
            except: pass
    else:
        cand_diff = abs(len(orig_cands) - len(rep_cands))

    orig_emit = json.load(open(os.path.join(orig_dir, "detector_emission.json")))
    rep_emit = json.load(open(os.path.join(rep_dir, "detector_emission.json")))
    emit_match = orig_emit.get("emit_step") == rep_emit.get("emit_step")

    ok = (n_orig == n_rep and orig_succ == rep_succ and ad == 0 and ed == 0
          and cand_diff == 0 and abst_diff == 0 and emit_match)

    rep_rows.append({
        "task": task, "state_id": str(sid),
        "orig_steps": str(n_orig), "rep_steps": str(n_rep),
        "orig_success": str(orig_succ), "rep_success": str(rep_succ),
        "success_match": str(orig_succ == rep_succ),
        "action_diffs": str(ad), "env_diffs": str(ed), "obs_diffs": str(od),
        "candidate_step_diffs": str(cand_diff), "abstain_diffs": str(abst_diff),
        "max_score_diff": str(round(max_s, 10)),
        "orig_emit": str(orig_emit.get("emit_step", -1)),
        "rep_emit": str(rep_emit.get("emit_step", -1)),
        "emit_match": str(emit_match),
        "status": "PASS" if ok else "FAIL",
    })
    print("  " + task + "_s" + str(sid) + ": " + ("PASS" if ok else "FAIL")
          + " act=" + str(ad) + " env=" + str(ed) + " cand=" + str(cand_diff)
          + " emit=" + str(emit_match))

rep_csv = os.path.join(PANEL, "postprocess", "l12_timing_repeatability_v2_fixed.csv")
with open(rep_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
    w.writeheader()
    w.writerows(rep_rows)
print("Repeatability: " + rep_csv)

print("\n=== Fixed scores ===")
for r in handoff[:6]:
    print("  " + r["task"] + "_s" + r["state_id"] + ": score=" + r["d5_score"]
          + " margin=" + r["d5_tau_margin"] + " split=" + r["split"]
          + " offset=" + r.get("emit_anchor_offset", "?"))
