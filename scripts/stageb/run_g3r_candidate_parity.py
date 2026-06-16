#!/usr/bin/env python3
"""G3-R: Full candidate-level historical parity."""
import csv, json, math, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from train_d1b_detector import CandidateRanker, FEATURE_NAMES
from evaluate_d5_frozen import online_detect
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1

CKPT = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"
CONFIG = "/data/liuyu/outputs/d5_training/d5_frozen_config.json"
OUT = "/data/liuyu/outputs/g3r_candidate_parity"
ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}
EXT_ROOT = "/data/liuyu/outputs/d4_34_privileged_replay"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"

config = json.load(open(CONFIG))
tau = float(config["tau"])
os.makedirs(OUT, exist_ok=True)

def parse_flag(v):
    s = str(v).strip() if v is not None else ""
    if s == "": return False
    try: return bool(int(float(s)))
    except: return False

def process_episode(edir, task, sid):
    cands_path = os.path.join(edir, "detector_candidates.csv")
    trace_path = os.path.join(edir, "step_trace.csv")
    if not os.path.exists(cands_path) or not os.path.exists(trace_path):
        return None, None

    csv_cands = list(csv.DictReader(open(cands_path)))
    rows = list(csv.DictReader(open(trace_path)))

    det = D5FrozenOnlineDetectorV1(CKPT, CONFIG)
    det.reset()
    for r in rows:
        det.update(
            int(r["step"]),
            float(r.get("raw_gripper", 0) or 0), float(r.get("env_gripper", 0) or 0),
            float(r.get("gripper_qpos_before", 0) or 0),
            float(r.get("eef_x", 0) or 0), float(r.get("eef_y", 0) or 0), float(r.get("eef_z", 0) or 0),
            int(float(r.get("decoded_open", 0) or 0)),
            parse_flag(r.get("raw_valid", "")), parse_flag(r.get("env_valid", "")),
            parse_flag(r.get("qpos_valid", "")), parse_flag(r.get("eef_valid", "")),
            parse_flag(r.get("semantics_ok", "")))

    live_records = det.audit_records
    frozen = online_detect(edir, model, means, stdevs, impute, tau)
    frozen_scores = frozen.get("all_scores", [])

    n_csv = len(csv_cands)
    n_live = len(live_records)
    n_frozen = len(frozen_scores)
    count_ok = (n_csv == n_live == n_frozen)

    feat_mismatches = 0; max_feat_diff = 0.0
    abstain_mismatches = 0
    score_mismatches = 0; max_score_diff = 0.0
    step_mismatches = 0

    detail_rows = []
    for i in range(min(n_live, n_csv)):
        lr = live_records[i]
        cc = csv_cands[i]

        if lr["step"] != int(cc.get("step", -1)):
            step_mismatches += 1

        fc = frozen_scores[i] if i < len(frozen_scores) else {}
        csv_abstain = int(cc.get("abstained", 0) or 0) == 1
        if lr["abstained"] != csv_abstain:
            abstain_mismatches += 1

        for fn in FEATURE_NAMES:
            cv = float(cc.get(f"feat_{fn}", cc.get(fn, "nan")) or "nan")
            lv = lr["features"].get(fn, "")
            try:
                lv_f = float(lv) if lv != "" else float("nan")
                if math.isfinite(cv) and math.isfinite(lv_f):
                    d = abs(cv - lv_f)
                    if d > max_feat_diff: max_feat_diff = d
                    if d > 2e-6: feat_mismatches += 1
            except: pass

        # Compare against frozen replay score (both use D5 model)
        frozen_score = fc.get("score", float("nan")) if i < len(frozen_scores) else float("nan")
        live_score = lr["score"]
        if math.isfinite(frozen_score):
            d = abs(frozen_score - live_score)
            if d > max_score_diff: max_score_diff = d
            if d > 1e-6: score_mismatches += 1

        # CSV score is D1b model (different model!) — only for reference
        csv_score = float(cc.get("score", "nan") or "nan")

        if task == "alphabet_soup" and sid == 17:
            frozen_score = fc.get("score", float("nan")) if i < len(frozen_scores) else float("nan")
            detail_rows.append({
                "step": lr["step"],
                "live_total_score": lr["features"].get("total_score", ""),
                "csv_total_score": cc.get("feat_total_score", cc.get("total_score", "")),
                "live_eef_speed_now": lr["features"].get("eef_speed_now", ""),
                "csv_eef_speed_now": cc.get("feat_eef_speed_now", cc.get("eef_speed_now", "")),
                "live_score": lr["score"],
                "frozen_d5_score": frozen_score,
                "live_abstain": lr["abstain"],
                "csv_abstain": cc.get("abstained", ""),
            })

    emit_match = (det.emit_step == frozen.get("emit_step", -1))

    row = {
        "task": task, "state_id": sid,
        "n_csv": n_csv, "n_live": n_live, "n_frozen": n_frozen,
        "count_ok": count_ok,
        "step_mismatches": step_mismatches,
        "abstain_mismatches": abstain_mismatches,
        "feat_mismatches": feat_mismatches, "max_feat_diff": max_feat_diff,
        "score_mismatches": score_mismatches, "max_score_diff": max_score_diff,
        "emit_match": emit_match,
        "live_emit": det.emit_step, "frozen_emit": frozen.get("emit_step", -1),
    }
    return row, detail_rows

# Load D5
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = CandidateRanker(n_features=16)
model.load_state_dict(ckpt["model_state"]); model.eval()
means = ckpt["means"]; stdevs = ckpt["stdevs"]; impute = ckpt["impute"]

# Manifest
manifest = list(csv.DictReader(open(MANIFEST)))
accepted = {(r["task"], int(r["state_id"])): r for r in manifest if r.get("status") == "BOUND"}

results = []
mismatches = []
s17_detail = None

print("Internal 120...")
for (task, sid), acc in sorted(accepted.items()):
    rname = acc["accepted_root"]
    edir = os.path.join(ROOTS[rname], acc["accepted_episode_dir"])
    row, detail = process_episode(edir, task, sid)
    if row: results.append(row)
    if detail: s17_detail = detail
    is_m = (not row["count_ok"] or row["step_mismatches"] > 0 or row["abstain_mismatches"] > 0
            or row["feat_mismatches"] > 0 or row["score_mismatches"] > 0 or not row["emit_match"])
    if is_m: mismatches.append(f"{task}_s{sid}: count={row['count_ok']} step={row['step_mismatches']} abstain={row['abstain_mismatches']} feat={row['feat_mismatches']} score={row['score_mismatches']} emit={row['emit_match']}")

print("External 34...")
for d in sorted(os.listdir(EXT_ROOT)):
    dp = os.path.join(EXT_ROOT, d)
    if not os.path.isdir(dp): continue
    m = re.match(r"(.+)_s(\d+)_shadow_attempt1", d)
    if not m: continue
    row, _ = process_episode(dp, m.group(1), int(m.group(2)))
    if row: results.append(row)
    is_m = (not row["count_ok"] or row["step_mismatches"] > 0 or row["abstain_mismatches"] > 0
            or row["feat_mismatches"] > 0 or row["score_mismatches"] > 0 or not row["emit_match"])
    if is_m: mismatches.append(f"{m.group(1)}_s{m.group(2)}: count={row['count_ok']} step={row['step_mismatches']} abstain={row['abstain_mismatches']} feat={row['feat_mismatches']} score={row['score_mismatches']} emit={row['emit_match']}")

# Summary
n_ok = len(results) - len(mismatches)
print(f"\n=== G3-R Candidate-Level Parity ===")
print(f"Total: {len(results)} episodes, PASS: {n_ok}")
for label, key in [("Count", "count_ok"), ("Step", "step_mismatches"), ("Abstain", "abstain_mismatches"), ("Emit", "emit_match")]:
    pct = sum(1 for r in results if (r[key] == True if isinstance(r[key], bool) else r[key] == 0)) / len(results) * 100
    print(f"  {label}: {pct:.0f}%")

max_f = max(r["max_feat_diff"] for r in results)
max_s = max(r["max_score_diff"] for r in results)
print(f"  Max feat diff: {max_f:.2e}")
print(f"  Max score diff: {max_s:.2e}")
print(f"  Total feat mismatches: {sum(r['feat_mismatches'] for r in results)}")
print(f"  Total abstain mismatches: {sum(r['abstain_mismatches'] for r in results)}")
print(f"  Total score mismatches: {sum(r['score_mismatches'] for r in results)}")

if mismatches:
    print(f"\nMismatches ({len(mismatches)}):")
    for m in mismatches[:20]: print(f"  {m}")

# Write
csv_path = os.path.join(OUT, "g3r_candidate_parity.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nCSV: {csv_path}")

if s17_detail:
    s17_path = os.path.join(OUT, "g3r_s17_diagnostic.csv")
    with open(s17_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s17_detail[0].keys()))
        w.writeheader(); w.writerows(s17_detail)
    print(f"s17: {s17_path}")
    for d in s17_detail:
        if float(str(d["live_total_score"]).strip() or 0) != float(str(d["csv_total_score"]).strip() or 0):
            print(f"  DIFF step={d['step']} total_score: live={d['live_total_score']} csv={d['csv_total_score']}")

print(f"Output: {OUT}")
