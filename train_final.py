"""R9Q correct training using PR#71 pipeline — A2/B2, save all epoch checkpoints."""
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_deepseek_r9q_retrain_20260713")
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig, C2gGripperCriticalWindowDetector)
from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9P_HEAD_NAMES, r9p_preview_loss, R9PEpisodeDataset, collate_episodes)

DS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2")
OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_correct_models_c15fa976_20260713_v1")

def train_one(label, use_policy, seed, device_str, epochs=30, bs=4):
    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = [json.loads(l) for l in (DS / "dataset_index.jsonl").read_text().splitlines() if l.strip()]
    norm = json.loads((DS / "normalization.json").read_text())

    train_ds = R9PEpisodeDataset(idx, DS, split_filter="FIT")
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_episodes)

    cfg = C2gDetectorConfig(
        visual_dim=1152, language_dim=128, policy_intent_dim=9,
        hidden=128, dropout=0.1, use_policy_intent=use_policy,
        use_visual=False, use_language_conditioning=True,
        head_names=R9P_HEAD_NAMES)
    model = C2gGripperCriticalWindowDetector(cfg).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    p_mean = torch.tensor(norm["proprio_mean"]).to(device)
    p_std = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.tensor(norm["policy_intent_mean"]).to(device)
    pi_std = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)

    out_dir = OUT / f"{label}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for ep in range(epochs):
        model.train()
        losses = defaultdict(float)
        nb = 0
        for batch in train_dl:
            p = (batch["proprio_25d"].to(device) - p_mean) / p_std
            pg = (batch["policy_intent"].to(device) - pi_mean) / pi_std if use_policy else None
            l = batch["language"].to(device)
            t = {k: v.to(device) for k, v in batch["targets"].items()}
            m = {k: v.to(device) for k, v in batch["masks"].items()}
            out = model(p, l, policy_intent=pg, return_sequence=True)
            pm_dev = batch["padding_mask"].to(device)
            for h in R9P_HEAD_NAMES:
                out[h] = out[h] * pm_dev.float()
            ld = r9p_preview_loss(out, t, m, sample_weight=pm_dev.float())
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            for k, v in ld.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    losses[k] += v.item()
            nb += 1

        # Save epoch checkpoint
        ckpt = {
            "schema_version": "c2g.r9q.correct.2026-07-13.v1",
            "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "model_config": {
                "visual_dim": 1152, "language_dim": 128, "policy_intent_dim": 9,
                "hidden": 128, "dropout": 0.1,
                "use_policy_intent": use_policy, "use_visual": False,
                "use_language_conditioning": True,
                "head_names": list(R9P_HEAD_NAMES)},
            "epoch": ep, "seed": seed, "config": label}
        torch.save(ckpt, out_dir / f"epoch_{ep+1:03d}.pt")

        avg_loss = {k: v / max(nb, 1) for k, v in losses.items()}
        history.append({"epoch": ep + 1, "loss": avg_loss})
        if (ep + 1) % 5 == 0:
            print(f"  {label} s{seed} epoch {ep+1}/{epochs} loss={avg_loss['total']:.4f}", flush=True)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "training_report.json", "w") as f:
        json.dump({"config": label, "seed": seed, "epochs": len(history)}, f)
    print(f"DONE {label} seed={seed} epochs={len(history)}", flush=True)
    return len(history)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:4")
    args = p.parse_args()
    use_pol = args.config in ("A2", "B2")
    r = train_one(args.config, use_pol, args.seed, args.device)
