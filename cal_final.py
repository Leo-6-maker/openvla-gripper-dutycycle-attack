"""CAL calibration with safety-first T10 metrics on all epoch checkpoints."""
import json, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_deepseek_r9q_retrain_20260713")
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig, C2gGripperCriticalWindowDetector, FixedBurstTriggerScheduler)
from scripts.stageb.train_c2g_r9p_preview_detector import R9P_HEAD_NAMES, _hash_language_embedding

DS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2")
MODELS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_correct_models_c15fa976_20260713_v1")
OUT = MODELS

idx = [json.loads(l) for l in (DS / "dataset_index.jsonl").read_text().splitlines() if l.strip()]
cal_rows = [r for r in idx if r["preview_split"] == "CAL"]
norm = json.loads((DS / "normalization.json").read_text())
print(f"CAL episodes: {len(cal_rows)}")

def calibrate_one(ckpt_path, device_str):
    device = torch.device(device_str)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = C2gDetectorConfig(**ckpt["model_config"])
    model = C2gGripperCriticalWindowDetector(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    use_policy = cfg.use_policy_intent

    p_mean = torch.tensor(norm["proprio_mean"]).to(device)
    p_std = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.tensor(norm["policy_intent_mean"]).to(device)
    pi_std = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)

    # Pre-compute all CAL outputs
    cal_data = []
    for r in cal_rows:
        d = np.load(r["npz_path"], allow_pickle=False)
        p = (torch.from_numpy(d["features_25d"]).unsqueeze(0).to(device) - p_mean) / p_std
        pg = (torch.from_numpy(d["features_9d"]).unsqueeze(0).to(device) - pi_mean) / pi_std if use_policy else None
        lang = torch.from_numpy(_hash_language_embedding(r.get("task_language", ""))).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(p, lang, policy_intent=pg, return_sequence=True)
        cal_data.append({
            "suite": r["suite"],
            "crit": torch.sigmoid(out["critical_window"]).squeeze(0).cpu().numpy(),
            "rel": torch.sigmoid(out["release_safe"]).squeeze(0).cpu().numpy(),
            "grnd": torch.sigmoid(out["grounding_confidence"]).squeeze(0).cpu().numpy(),
            "tgt_burst": d["y_burst_feasible"], "msk_burst": d["m_burst_feasible"],
            "tgt_crit": d["y_critical_window"], "msk_crit": d["m_critical_window"],
            "tgt_rel": d["y_release_safe"], "msk_rel": d["m_release_safe"],
            "tgt_start": d["y_window_start"], "msk_start": d["m_window_start"],
        })
    del model; torch.cuda.empty_cache()

    # Grid search
    taus_crit = [0.3, 0.4, 0.5, 0.6, 0.7]
    taus_rel = [0.3, 0.4, 0.5, 0.6]
    taus_grnd = [0.3, 0.5, 0.7]
    persist = [(1, 1), (3, 2), (5, 3)]
    MAX_FALSE, MAX_REL = 0.10, 0.02
    feasible = []

    for tc in taus_crit:
        for tr in taus_rel:
            for tg in taus_grnd:
                for pw, pr in persist:
                    n_feas = n_t10 = n_false = n_rel_emit = n_pos = n_neg = 0
                    for ep in cal_data:
                        s = FixedBurstTriggerScheduler(burst_length=10, tau_critical=tc, tau_release=tr, tau_ground=tg, persistence_window=pw, persistence_required=pr)
                        T = len(ep["crit"]); trig = False; tt = -1
                        for t in range(T):
                            d = s.update(critical_probability=float(ep["crit"][t]), release_safe_probability=float(ep["rel"][t]), grounding_confidence_probability=float(ep["grnd"][t]), valid=True)
                            if d.trigger_started: trig = True; tt = t; break
                        has_start = ep["tgt_start"].any() and ep["msk_start"].any()
                        if has_start: n_pos += 1
                        else: n_neg += 1
                        if trig:
                            if tt < T and ep["msk_burst"][tt] and ep["tgt_burst"][tt] > 0.5: n_feas += 1
                            end = min(tt + 10, T)
                            wc = ep["tgt_crit"][tt:end]; wm = ep["msk_crit"][tt:end]
                            if len(wc) == 10 and wm.all() and (wc > 0.5).all(): n_t10 += 1
                            if tt < T and ep["msk_rel"][tt] and ep["tgt_rel"][tt] > 0.5: n_rel_emit += 1
                            if not has_start: n_false += 1
                    fr = n_false / max(n_neg, 1); rr = n_rel_emit / max(len(cal_data), 1)
                    if fr > MAX_FALSE or rr > MAX_RELEASE: continue
                    feasible.append({"tc": tc, "tr": tr, "tg": tg, "pw": pw, "pr": pr, "feas": n_feas/max(n_pos,1), "t10": n_t10, "fr": fr, "rr": rr})

    if not feasible: return None
    best = max(feasible, key=lambda c: (c["feas"], c["t10"], -c["fr"], -c["rr"]))
    return {"feasible_hit": best["feas"], "full_t10": best["t10"], "false_rate": best["fr"], "release_rate": best["rr"], "config": {"tau_critical": best["tc"], "tau_release": best["tr"], "tau_ground": best["tg"], "persistence_window": best["pw"], "persistence_required": best["pr"]}, "n_feasible_configs": len(feasible)}

# Main: calibrate all checkpoints
checkpoints = sorted(MODELS.glob("*/epoch_*.pt"))
print(f"Total checkpoints: {len(checkpoints)}")
results = {}

# Process sequentially cycling GPUs
gpus = [4, 5, 6, 7]
for i, ckpt in enumerate(checkpoints):
    name = f"{ckpt.parent.name}_{ckpt.stem}"
    gpu = gpus[i % 4]
    print(f"  [{i+1}/{len(checkpoints)}] {name} GPU{gpu}...", end=" ", flush=True)
    try:
        r = calibrate_one(ckpt, f"cuda:{gpu}")
        if r: results[name] = r; print(f"feas={r['feasible_hit']:.3f} T10={r['full_t10']}")
        else: results[name] = {"error": "no feasible config"}; print("NO_FEASIBLE")
    except Exception as e:
        results[name] = {"error": str(e)}; print(f"ERR: {e}")

# Find best per config
print("\n=== BEST PER CONFIG ===")
for cfg in ["A2", "B2"]:
    cfg_r = {k: v for k, v in results.items() if k.startswith(cfg) and "error" not in v}
    if cfg_r:
        best_k = max(cfg_r, key=lambda k: (cfg_r[k]["feasible_hit"], cfg_r[k]["full_t10"]))
        best_v = cfg_r[best_k]
        print(f"  {cfg}: {best_k} feas={best_v['feasible_hit']:.3f} T10={best_v['full_t10']} config={best_v['config']}")

with open(OUT / "calibration_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUT}/calibration_results.json")
print("DONE")
