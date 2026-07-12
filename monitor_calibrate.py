"""Monitor training, auto-calibrate all checkpoints when done."""
import json, sys, time, hashlib, subprocess, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_codex_c2g_strict_resume_a334891_20260711")
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig, C2gGripperCriticalWindowDetector, FixedBurstTriggerScheduler
)

OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_overnight_models_f47cb75_20260713_v1")
DATASET = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2")
TARGET = 12
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

def calibrate_one(ckpt_path, cal_dir):
    """Quick CAL evaluation for a single checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = C2gDetectorConfig(**ckpt["model_config"])
    device = torch.device("cuda")
    model = C2gGripperCriticalWindowDetector(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    idx = [json.loads(l) for l in (DATASET / "dataset_index.jsonl").read_text().splitlines() if l.strip()]
    cal_rows = [r for r in idx if r["preview_split"] == "CAL"]
    norm = json.loads((DATASET / "normalization.json").read_text())
    p_mean = torch.tensor(norm["proprio_mean"]).to(device)
    p_std = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.tensor(norm["policy_intent_mean"]).to(device)
    pi_std = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)
    use_policy = cfg.use_policy_intent

    total_loss = 0.0; n_eps = 0
    for r in cal_rows:
        d = np.load(r["npz_path"], allow_pickle=False)
        p = (torch.from_numpy(d["features_25d"]).unsqueeze(0).to(device) - p_mean) / p_std
        pg = (torch.from_numpy(d["features_9d"]).unsqueeze(0).to(device) - pi_mean) / pi_std if use_policy else None
        lang = torch.from_numpy(hash_lang(r.get("task_language", ""))).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(p, lang, policy_intent=pg, return_sequence=True)
        ep_loss = 0.0
        for h in HEAD_NAMES:
            tgt = torch.from_numpy(d[f"y_{h}"]).unsqueeze(0).to(device)
            msk = torch.from_numpy(d[f"m_{h}"]).unsqueeze(0).to(device)
            if msk.any():
                ep_loss += torch.nn.functional.binary_cross_entropy_with_logits(out[h][msk], tgt[msk]).item()
        total_loss += ep_loss; n_eps += 1

    del model; torch.cuda.empty_cache()
    avg_loss = total_loss / max(n_eps, 1)
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "cal_quick.json", "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "cal_episodes": n_eps, "avg_bce": avg_loss}, f)
    return avg_loss

print(f"=== R9Q Monitor Started {time.strftime('%H:%M:%S')} ===")
print(f"Target: {TARGET} checkpoints, Dataset: {DATASET}")

while True:
    done = len(list(OUT.glob("*/checkpoint.pt")))
    running = int(subprocess.run(["pgrep", "-cf", "train_r9q.py"], capture_output=True, text=True).stdout.strip() or 0)
    print(f"{time.strftime('%H:%M:%S')} Done: {done}/{TARGET} Running: {running}")

    if done >= TARGET and running <= 1:
        print(f"\n=== TRAINING COMPLETE at {time.strftime('%H:%M:%S')} ===")
        for ckpt in sorted(OUT.glob("*/checkpoint.pt")):
            print(f"  {ckpt.parent.name}: {ckpt.stat().st_size} bytes")

        print("\n=== STARTING CALIBRATION ===")
        results = {}
        for ckpt_path in sorted(OUT.glob("*/checkpoint.pt")):
            name = ckpt_path.parent.name
            if "_cal" in name:
                continue
            cal_dir = OUT / f"{name}_cal"
            print(f"  {name}...", end=" ", flush=True)
            try:
                loss = calibrate_one(ckpt_path, cal_dir)
                results[name] = loss
                print(f"loss={loss:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")

        # Find best checkpoint per config
        print("\n=== BEST PER CONFIG ===")
        for cfg in ["A1", "A2", "B1", "B2"]:
            cfg_results = {k: v for k, v in results.items() if k.startswith(cfg)}
            if cfg_results:
                best = min(cfg_results, key=cfg_results.get)
                print(f"  {cfg}: {best} loss={cfg_results[best]:.4f}")

        print(f"\n=== DONE at {time.strftime('%H:%M:%S')} ===")
        break

    time.sleep(30)
