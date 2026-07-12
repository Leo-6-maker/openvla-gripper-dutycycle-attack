"""Overnight R9Q detector training: A1(OGS-25D), A2(OGS-25D+9D), B1(+L10-25D), B2(+L10-25D+9D)."""
import json, sys, time, hashlib, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_codex_c2g_strict_resume_a334891_20260711")
from src.gripper_attack.c2g_gripper_critical_window_detector import C2gDetectorConfig, C2gGripperCriticalWindowDetector
from src.gripper_attack.c2g_causal_vulnerability_detector import masked_bce, _persistent_score, positive_interval_triggerability

HEAD_NAMES = ("window_start", "burst_feasible", "critical_window", "release_safe", "contact_grasp", "grounding_confidence")
LANGUAGE_DIM, VISUAL_DIM = 128, 1152

def hash_lang(text):
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    proj = rng.randn(32, LANGUAGE_DIM).astype(np.float32)
    vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
    if len(vals) < 32: vals = np.pad(vals, (0, 32-len(vals)))
    emb = vals[:32] @ proj
    n = np.linalg.norm(emb)
    return (emb / n).astype(np.float32) if n > 1e-8 else emb

def r9q_loss(outputs, targets, masks, sample_weight=None, pw=3, pr=2):
    required = set(HEAD_NAMES)
    if set(outputs) != required: raise ValueError("R9Q loss requires exactly 6 heads")
    sl = masked_bce(outputs["window_start"], targets["window_start"], masks["window_start"], sample_weight)
    bl = masked_bce(outputs["burst_feasible"], targets["burst_feasible"], masks["burst_feasible"], sample_weight)
    cl = masked_bce(outputs["critical_window"], targets["critical_window"], masks["critical_window"], sample_weight)
    rl = masked_bce(outputs["release_safe"], targets["release_safe"], masks["release_safe"], sample_weight)
    c2 = masked_bce(outputs["contact_grasp"], targets["contact_grasp"], masks["contact_grasp"], sample_weight)
    gc_out = outputs["grounding_confidence"]; gc_tgt = targets["grounding_confidence"]
    gc_mask = masks["grounding_confidence"].bool()
    gl = nn.functional.mse_loss(gc_out[gc_mask], gc_tgt[gc_mask]) if gc_mask.any() else gc_out.sum()*0.0
    slog = outputs["window_start"]; zero = slog.sum() * 0.0
    if slog.ndim != 2:
        ep = dict(early_emit=zero, episode_miss=zero, negative_episode_any_emit=zero,
                  release_safe_emit=zero, positive_episode_count=zero,
                  triggerable_positive_episode_count=zero, untriggerable_positive_episode_count=zero,
                  persistent_positive_window_count=zero)
    else:
        probs = torch.sigmoid(slog); early, miss, neg, rel = [], [], [], []
        ep_fkn = masks.get("episode_fully_known_negative")
        for idx in range(slog.shape[0]):
            p = probs[idx]; ys = targets["window_start"][idx].bool(); ms = masks["window_start"][idx].bool()
            known = masks["critical_window"][idx].bool(); pos_start = ys & ms
            if pos_start.any():
                first = int(torch.nonzero(pos_start, as_tuple=False)[0,0])
                em = known.clone(); em[first:] = False
                if em.any(): early.append(_persistent_score(p, em, window=pw, required=pr))
                lm = known.clone(); lm[:first] = False
                if lm.any():
                    ps = _persistent_score(p, lm, window=pw, required=pr)
                    miss.append(-torch.log(ps.clamp(min=1e-6)))
            else:
                fkn = bool(known.all()) if ep_fkn is None else bool(ep_fkn[idx].item())
                if fkn and known.any(): neg.append(_persistent_score(p, known, window=pw, required=pr))
        for idx in range(slog.shape[0]):
            rs = targets["release_safe"][idx].bool() & masks["release_safe"][idx].bool()
            if rs.any(): rel.append(_persistent_score(probs[idx], rs, window=pw, required=pr))
        diag = positive_interval_triggerability(targets["window_start"], masks["window_start"], persistence_window=pw, persistence_required=pr)
        ep = dict(early_emit=torch.stack(early).mean() if early else zero,
                  episode_miss=torch.stack(miss).mean() if miss else zero,
                  negative_episode_any_emit=torch.stack(neg).mean() if neg else zero,
                  release_safe_emit=torch.stack(rel).mean() if rel else zero, **diag)
    total = sl + 0.5*bl + 0.5*cl + 0.2*rl + 0.2*c2 + 0.2*gl + 0.25*ep["early_emit"] + 0.50*ep["episode_miss"] + 0.50*ep["negative_episode_any_emit"] + 0.50*ep["release_safe_emit"]
    return dict(total=total, window_start=sl, burst_feasible=bl, critical_window=cl, release_safe=rl, contact_grasp=c2, grounding_confidence=gl, **ep)

class EpDataset(Dataset):
    def __init__(self, index_rows, split_filter=None):
        self.rows = [r for r in index_rows if split_filter is None or r["preview_split"]==split_filter]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]; d = np.load(r["npz_path"], allow_pickle=False)
        return dict(f25=torch.from_numpy(d["features_25d"].copy()), f9=torch.from_numpy(d["features_9d"].copy()),
                    tgt={h: torch.from_numpy(d[f"y_{h}"].copy()) for h in HEAD_NAMES},
                    msk={h: torch.from_numpy(d[f"m_{h}"].copy()) for h in HEAD_NAMES},
                    known=torch.from_numpy(d["known_mask"].copy()), lang=r.get("task_language",""))

def collate(batch):
    B = len(batch); lengths = torch.tensor([b["f25"].shape[0] for b in batch]); M = lengths.max().item()
    p25 = torch.zeros(B,M,25); p9 = torch.zeros(B,M,9); pm = torch.zeros(B,M,dtype=torch.bool)
    tgt = {h: torch.zeros(B,M) for h in HEAD_NAMES}; msk = {h: torch.zeros(B,M,dtype=torch.bool) for h in HEAD_NAMES}
    fkn = torch.zeros(B,dtype=torch.bool); langs = []
    for i,b in enumerate(batch):
        T = b["f25"].shape[0]; p25[i,:T]=b["f25"]; p9[i,:T]=b["f9"]; pm[i,:T]=True
        for h in HEAD_NAMES: tgt[h][i,:T]=b["tgt"][h]; msk[h][i,:T]=b["msk"][h]
        ak = b["known"].all(); ap = b["tgt"]["critical_window"].any() if ak else False
        fkn[i] = bool(ak and not ap); langs.append(torch.from_numpy(hash_lang(b["lang"])))
    msk["episode_fully_known_negative"] = fkn
    return dict(p25=p25, p9=p9, lang=torch.stack(langs), tgt=tgt, msk=msk, pm=pm)

def train_one(config_label, dataset_root, output_root, seed, use_policy, device_str="cuda", epochs=30, bs=4):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    idx = [json.loads(l) for l in (Path(dataset_root)/"dataset_index.jsonl").read_text().splitlines() if l.strip()]
    norm = json.loads((Path(dataset_root)/"normalization.json").read_text())
    train_ds = EpDataset(idx, "FIT"); cal_ds = EpDataset(idx, "CAL")
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    cfg = C2gDetectorConfig(visual_dim=VISUAL_DIM, language_dim=LANGUAGE_DIM, policy_intent_dim=9,
                            hidden=128, dropout=0.1, use_policy_intent=use_policy,
                            use_visual=False, use_language_conditioning=True, head_names=HEAD_NAMES)
    model = C2gGripperCriticalWindowDetector(cfg).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    p_mean = torch.tensor(norm["proprio_mean"]).to(device); p_std = torch.tensor(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.tensor(norm["policy_intent_mean"]).to(device); pi_std = torch.tensor(norm["policy_intent_std"]).to(device).clamp_min(1e-8)
    best_score = -float("inf"); best_state = None; patience = 0; history = []
    out_dir = Path(output_root) / f"{config_label}_seed{seed}"; out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train(); losses = defaultdict(float); nb = 0
        for batch in train_dl:
            p = (batch["p25"].to(device) - p_mean) / p_std
            pg = (batch["p9"].to(device) - pi_mean) / pi_std if use_policy else None
            l = batch["lang"].to(device)
            t = {k: v.to(device) for k,v in batch["tgt"].items()}
            m = {k: v.to(device) for k,v in batch["msk"].items()}
            out = model(p, l, policy_intent=pg, return_sequence=True)
            pm = batch["pm"].to(device)
            for h in HEAD_NAMES: out[h] = out[h] * pm.float()
            ld = r9q_loss(out, t, m, sample_weight=pm.float())
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt.zero_grad()
            for k,v in ld.items():
                if isinstance(v, Tensor) and v.ndim==0: losses[k] += v.item()
            nb += 1
        score = float(ep)
        history.append(dict(epoch=ep, loss={k: v/max(nb,1) for k,v in losses.items()}, score=score))
        if score > best_score: best_score = score; best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= 5: break
        if (ep+1) % 5 == 0: print(f"  {config_label} s{seed} epoch {ep+1}/{epochs}")
    if best_state: model.load_state_dict(best_state)
    ckpt = dict(schema_version="c2g.r9q.checkpoint.2026-07-13.v1", model_state_dict=best_state or model.state_dict(),
                model_config=dict(visual_dim=VISUAL_DIM, language_dim=LANGUAGE_DIM, policy_intent_dim=9,
                hidden=128, dropout=0.1, use_policy_intent=use_policy, use_visual=False,
                use_language_conditioning=True, head_names=list(HEAD_NAMES)),
                history=history, seed=seed, config=config_label, normalization=norm)
    torch.save(ckpt, out_dir / "checkpoint.pt")
    with open(out_dir / "training_report.json", "w") as f: json.dump(dict(config=config_label, seed=seed, epochs=len(history), best_score=float(best_score)), f)
    return dict(config=config_label, seed=seed, epochs=len(history), score=float(best_score))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda"); p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    use_pol = args.config in ("A2", "B2")
    r = train_one(args.config, args.dataset, args.output, args.seed, use_pol, args.device, args.epochs)
    print(f"DONE {r['config']} seed={r['seed']} epochs={r['epochs']} score={r['score']:.4f}")
