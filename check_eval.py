"""CHECK evaluation on selected B2_seed456_e025 checkpoint."""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_deepseek_r9q_retrain_20260713")
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig, C2gGripperCriticalWindowDetector, FixedBurstTriggerScheduler)
from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9P_HEAD_NAMES, _hash_language_embedding)

DS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2")
MODELS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_correct_models_c15fa976_20260713_v1")
CKPT = MODELS / "B2_seed456" / "epoch_025.pt"
THRESH = {"tau_critical": 0.4, "tau_release": 0.3, "tau_ground": 0.7,
          "persistence_window": 5, "persistence_required": 3, "burst_length": 10}

ckpt_data = torch.load(str(CKPT), map_location="cpu")
cfg = C2gDetectorConfig(**ckpt_data["model_config"])
device = torch.device("cuda:4")
model = C2gGripperCriticalWindowDetector(cfg).to(device)
model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
model.eval()

idx = [json.loads(l) for l in (DS / "dataset_index.jsonl").read_text().splitlines() if l.strip()]
check_rows = [r for r in idx if r["preview_split"] == "CHECK"]
norm = json.loads((DS / "normalization.json").read_text())
print(f"CHECK episodes: {len(check_rows)}")
print(f"Thresholds: {THRESH}")

use_pol = cfg.use_policy_intent
pm = torch.tensor(norm["proprio_mean"]).to(device)
ps = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
im = torch.tensor(norm["policy_intent_mean"]).to(device)
istd = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)

suites = {}
for r in check_rows:
    d = np.load(r["npz_path"], allow_pickle=False)
    p = (torch.from_numpy(d["features_25d"]).unsqueeze(0).to(device) - pm) / ps
    pg = (torch.from_numpy(d["features_9d"]).unsqueeze(0).to(device) - im) / istd if use_pol else None
    lang = torch.from_numpy(_hash_language_embedding(r.get("task_language", ""))).unsqueeze(0).to(device)
    with torch.no_grad():
        o = model(p, lang, policy_intent=pg, return_sequence=True)

    suite = r["suite"]
    if suite not in suites:
        suites[suite] = {"feas": 0, "t10": 0, "false": 0, "rel": 0,
                         "pos": 0, "neg": 0, "trig": 0, "multi": 0, "n": 0}
    s = suites[suite]; s["n"] += 1

    crit = torch.sigmoid(o["critical_window"]).squeeze(0).cpu().numpy()
    rel = torch.sigmoid(o["release_safe"]).squeeze(0).cpu().numpy()
    grnd = torch.sigmoid(o["grounding_confidence"]).squeeze(0).cpu().numpy()

    sched = FixedBurstTriggerScheduler(**THRESH)
    T = len(crit); trig = False; tt = -1; trig_count = 0
    for t in range(T):
        dec = sched.update(float(crit[t]), float(rel[t]), float(grnd[t]), True)
        if dec.trigger_started:
            trig = True; tt = t; trig_count += 1

    has_start = d["y_window_start"].any() and d["m_window_start"].any()
    if has_start: s["pos"] += 1
    else: s["neg"] += 1

    if trig:
        s["trig"] += 1
        if trig_count > 1: s["multi"] += 1
        if tt < T and d["m_burst_feasible"][tt] and d["y_burst_feasible"][tt] > 0.5:
            s["feas"] += 1
        end = min(tt + 10, T)
        wc = d["y_critical_window"][tt:end]; wm = d["m_critical_window"][tt:end]
        if len(wc) == 10 and wm.all() and (wc > 0.5).all():
            s["t10"] += 1
        if tt < T and d["m_release_safe"][tt] and d["y_release_safe"][tt] > 0.5:
            s["rel"] += 1
        if not has_start:
            s["false"] += 1

# Aggregate
totals = {k: sum(s[k] for s in suites.values()) for k in ["feas", "t10", "false", "rel", "pos", "neg", "trig", "multi"]}
totals["n"] = len(check_rows)
fr = totals["false"] / max(totals["neg"], 1)
rr = totals["rel"] / max(len(check_rows), 1)
feas_rate = totals["feas"] / max(totals["pos"], 1)

print(f"\n=== CHECK Results (B2_seed456_e025) ===")
print(f"Overall: feas={totals['feas']}/{totals['pos']} ({feas_rate*100:.1f}%) "
      f"T10={totals['t10']} false={totals['false']}/{totals['neg']} "
      f"rel_emit={totals['rel']} multi_trig={totals['multi']}")
for suite in sorted(suites.keys()):
    s = suites[suite]
    label = " LOW_SUPPORT" if s["n"] < 5 else ""
    print(f"  {suite}: n={s['n']} feas={s['feas']}/{s['pos']} T10={s['t10']} "
          f"false={s['false']}/{s['neg']} rel={s['rel']} multi={s['multi']}{label}")

gate_pass = (feas_rate >= 0.55 and fr <= 0.15 and rr <= 0.03 and totals["multi"] == 0)
checks = {
    "feas>=0.55": feas_rate >= 0.55,
    "false<=0.15": fr <= 0.15,
    "rel<=0.03": rr <= 0.03,
    "multi=0": totals["multi"] == 0,
}
print(f"\nCHECK Gate: {'PASS' if gate_pass else 'HOLD'}")
for k, v in checks.items():
    print(f"  {k}: {'OK' if v else 'FAIL'}")

report = {"checkpoint": str(CKPT), "thresholds": THRESH, "overall": totals,
          "per_suite": suites, "checks": checks, "gate": "PASS" if gate_pass else "HOLD"}
with open(str(MODELS / "check_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Report saved to check_report.json")
