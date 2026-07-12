"""Proper CAL calibration with T10 metrics on GPU 4/5/6/7."""
import json, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_codex_c2g_strict_resume_a334891_20260711")
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig, C2gGripperCriticalWindowDetector, FixedBurstTriggerScheduler
)

OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_overnight_models_f47cb75_20260713_v1")
DATASET = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2")
HEAD_NAMES = ("window_start", "burst_feasible", "critical_window", "release_safe", "contact_grasp", "grounding_confidence")

def hash_lang(text):
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    proj = rng.randn(32, 128).astype(np.float32)
    vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    if len(vals) < 32: vals = np.pad(vals, (0, 32-len(vals)))
    emb = vals[:32] @ proj
    n = np.linalg.norm(emb)
    return (emb / n).astype(np.float32) if n > 1e-8 else emb

def calibrate_checkpoint(ckpt_path, device_str):
    """Full CAL calibration: grid search thresholds, return best config + metrics."""
    device = torch.device(device_str)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = C2gDetectorConfig(**ckpt["model_config"])
    model = C2gGripperCriticalWindowDetector(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    use_policy = cfg.use_policy_intent

    idx = [json.loads(l) for l in (DATASET / "dataset_index.jsonl").read_text().splitlines() if l.strip()]
    cal_rows = [r for r in idx if r["preview_split"] == "CAL"]
    norm = json.loads((DATASET / "normalization.json").read_text())
    p_mean = torch.tensor(norm["proprio_mean"]).to(device)
    p_std = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.tensor(norm["policy_intent_mean"]).to(device)
    pi_std = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)

    # Pre-compute model outputs for all CAL episodes
    print(f"    Loading {len(cal_rows)} CAL episodes...", flush=True)
    cal_data = []
    for r in cal_rows:
        d = np.load(r["npz_path"], allow_pickle=False)
        p = (torch.from_numpy(d["features_25d"]).unsqueeze(0).to(device) - p_mean) / p_std
        pg = (torch.from_numpy(d["features_9d"]).unsqueeze(0).to(device) - pi_mean) / pi_std if use_policy else None
        lang = torch.from_numpy(hash_lang(r.get("task_language", ""))).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(p, lang, policy_intent=pg, return_sequence=True)
        cal_data.append({
            "suite": r["suite"],
            "critical": torch.sigmoid(out["critical_window"]).squeeze(0).cpu().numpy(),
            "release": torch.sigmoid(out["release_safe"]).squeeze(0).cpu().numpy(),
            "grounding": torch.sigmoid(out["grounding_confidence"]).squeeze(0).cpu().numpy(),
            "start": torch.sigmoid(out["window_start"]).squeeze(0).cpu().numpy(),
            "burst": torch.sigmoid(out["burst_feasible"]).squeeze(0).cpu().numpy(),
            "tgt_burst": d["y_burst_feasible"], "msk_burst": d["m_burst_feasible"],
            "tgt_start": d["y_window_start"], "msk_start": d["m_window_start"],
            "tgt_critical": d["y_critical_window"], "msk_critical": d["m_critical_window"],
            "tgt_release": d["y_release_safe"], "msk_release": d["m_release_safe"],
        })

    del model; torch.cuda.empty_cache()

    # Grid search with safety-first filtering
    tau_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    persistence_cfgs = [(3,2), (5,3), (1,1)]
    MAX_FALSE = 0.10; MAX_RELEASE = 0.02
    feasible = []

    for tau_crit in tau_values:
        for tau_rel in [0.3, 0.4, 0.5, 0.6]:
            for tau_grnd in [0.3, 0.5, 0.7]:
                for pw, pr in persistence_cfgs:
                    n_feas = 0; n_full_t10 = 0; n_false = 0; n_rel_emit = 0
                    n_pos = 0; n_neg = 0
                    for ep in cal_data:
                        sched = FixedBurstTriggerScheduler(burst_length=10, tau_critical=tau_crit,
                            tau_release=tau_rel, tau_ground=tau_grnd, persistence_window=pw, persistence_required=pr)
                        T = len(ep["critical"])
                        triggered = False; trig_t = -1
                        for t in range(T):
                            d = sched.update(critical_probability=float(ep["critical"][t]),
                                release_safe_probability=float(ep["release"][t]),
                                grounding_confidence_probability=float(ep["grounding"][t]), valid=True)
                            if d.trigger_started:
                                triggered = True; trig_t = t; break
                        has_start = ep["tgt_start"].any() and ep["msk_start"].any()
                        if has_start and ep["msk_start"].any():
                            n_pos += 1
                        else:
                            n_neg += 1
                        if triggered:
                            # feasible_hit: y_burst_feasible at trigger step
                            if trig_t < T and ep["msk_burst"][trig_t]:
                                if ep["tgt_burst"][trig_t] > 0.5:
                                    n_feas += 1
                            # full T10 containment: critical[t:t+10] all known and True
                            end = min(trig_t + 10, T)
                            window_crit = ep["tgt_critical"][trig_t:end]
                            window_msk = ep["msk_critical"][trig_t:end]
                            if len(window_crit) == 10 and window_msk.all() and (window_crit > 0.5).all():
                                n_full_t10 += 1
                            # release_safe at trigger
                            if trig_t < T and ep["msk_release"][trig_t] and ep["tgt_release"][trig_t] > 0.5:
                                n_rel_emit += 1
                            if not has_start:
                                n_false += 1

                    false_rate = n_false / max(n_neg, 1)
                    rel_rate = n_rel_emit / max(len(cal_data), 1)
                    if false_rate > MAX_FALSE or rel_rate > MAX_RELEASE:
                        continue
                    feasible.append({
                        "tau_critical": tau_crit, "tau_release": tau_rel, "tau_ground": tau_grnd,
                        "pw": pw, "pr": pr,
                        "feasible_rate": n_feas / max(n_pos, 1),
                        "full_t10": n_full_t10, "false_rate": false_rate, "rel_rate": rel_rate,
                        "n_pos": n_pos, "n_feas": n_feas
                    })

    if not feasible:
        return {"error": "no feasible config", "cal_episodes": len(cal_data)}

    best = max(feasible, key=lambda c: (c["feasible_rate"], c["full_t10"], -c["false_rate"], -c["rel_rate"]))
    return {
        "cal_episodes": len(cal_data), "feasible_configs": len(feasible),
        "best_feasible_rate": best["feasible_rate"],
        "best_full_t10": best["full_t10"],
        "best_false_rate": best["false_rate"],
        "best_rel_rate": best["rel_rate"],
        "best_config": {k: best[k] for k in ["tau_critical","tau_release","tau_ground","pw","pr"]},
        "n_pos": best["n_pos"], "n_feas": best["n_feas"],
    }

# Main: calibrate all checkpoints in parallel across GPUs
import subprocess, os
checkpoints = sorted(OUT.glob("*/checkpoint.pt"))
print(f"Calibrating {len(checkpoints)} checkpoints on GPUs 4/5/6/7...")
print(f"Dataset: {DATASET}")

# Process in batches of 4 (one per GPU)
results = {}
for i in range(0, len(checkpoints), 4):
    batch = checkpoints[i:i+4]
    # Launch in parallel
    gpu_idx = 4
    for ckpt in batch:
        name = ckpt.parent.name
        print(f"  {name} on cuda:{gpu_idx}...", end=" ", flush=True)
        try:
            r = calibrate_checkpoint(ckpt, f"cuda:{gpu_idx}")
            results[name] = r
            if "error" in r:
                print(f"FAIL: {r['error']}")
            else:
                print(f"feas={r['best_feasible_rate']:.3f} T10={r['best_full_t10']} false={r['best_false_rate']:.3f}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"CRASH: {e}")
        gpu_idx += 1

# Best per config
print("\n=== BEST PER CONFIG ===")
for cfg in ["A1", "A2", "B1", "B2"]:
    cfg_results = {k: v for k, v in results.items() if k.startswith(cfg) and "error" not in v}
    if cfg_results:
        best_name = max(cfg_results, key=lambda k: (cfg_results[k].get("best_feasible_rate",0), cfg_results[k].get("best_full_t10",0)))
        best_r = cfg_results[best_name]
        print(f"  {cfg}: {best_name} feas={best_r['best_feasible_rate']:.3f} T10={best_r['best_full_t10']} config={best_r.get('best_config',{})}")

# Save results
cal_out = OUT / "calibration_results.json"
with open(cal_out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {cal_out}")
print("DONE")
