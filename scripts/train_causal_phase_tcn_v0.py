#!/usr/bin/env python3
"""train_causal_phase_tcn_v0.py — Offline causal phase detector smoke on GPU compute-only.

Uses CUDA_VISIBLE_DEVICES=0,3 for training only — NO rendering, NO MuJoCo.
Target: phase_bin_proxy (multiclass). NOT vulnerability detector.

Output: tables/phase_tcn_v0_*.csv, reports/PHASE_TCN_V0_OFFLINE_AUDIT.md
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Causal TCN (same architecture as train_early_grasp_detector.py) ──
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    def forward(self, x):
        xt = x.transpose(1, 2)
        return self.conv(F.pad(xt, (self.pad, 0))).transpose(1, 2)

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.n1 = nn.LayerNorm(ch); self.n2 = nn.LayerNorm(ch)
    def forward(self, x):
        r = x
        out = self.n1(F.relu(self.c1(x))); out = self.dropout(out)
        out = self.n2(F.relu(self.c2(out))); out = self.dropout(out)
        return out + r

class PhaseTCN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=3, kernel_size=3,
                 num_classes=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, kernel_size, 2**i, dropout)
                                     for i in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = self.input_proj(x)
        for blk in self.blocks: out = blk(out)
        return self.output_proj(out)

# ── Dataset: per-timestep labels from descriptors + phase events ──
class SequencePhaseDataset(Dataset):
    def __init__(self, npz_path, meta_csv, phase_csv, desc_csv, split="train",
                 split_col="split_state_holdout"):
        data = np.load(npz_path, allow_pickle=True)
        X_key = "X_norm" if "X_norm" in data else "X"
        self.X = torch.from_numpy(data[X_key]).float()
        self.mask = torch.from_numpy(data["mask"]).bool()
        self.ep_ids = list(data.get("episode_ids", []))

        # Load meta, phase events, descriptors
        self.meta = {}; self.tg_form = {}
        if os.path.exists(meta_csv):
            with open(meta_csv, newline="") as f:
                for r in csv.DictReader(f):
                    self.meta[r["episode_id"]] = r
                    if r.get("T_gform"): self.tg_form[r["episode_id"]] = int(r["T_gform"])

        # Build per-timestep phase labels from descriptor windows
        self.phase_labels = {}
        if os.path.exists(desc_csv):
            with open(desc_csv, newline="") as f:
                for d in csv.DictReader(f):
                    eid = d.get("episode_id","")
                    ws_str = d.get("window_start",""); we_str = d.get("window_end","")
                    if not ws_str or not we_str: continue
                    ws = int(ws_str); we = int(we_str)
                    phase = d.get("phase_bin_proxy","")
                    if phase not in PHASE_MAP: continue
                    if eid not in self.phase_labels: self.phase_labels[eid] = []
                    self.phase_labels[eid].append((ws, we, PHASE_MAP[phase]))

        # Filter by split
        if os.path.exists("tables/object_detector_split_plan_clean.csv"):
            with open("tables/object_detector_split_plan_clean.csv", newline="") as f:
                split_map = {r["episode_id"]: r[split_col] for r in csv.DictReader(f)}
        else:
            split_map = {eid: "train" for eid in self.ep_ids}

        indices = [i for i, eid in enumerate(self.ep_ids) if split_map.get(str(eid), "train") == split]
        self.X = self.X[indices]; self.mask = self.mask[indices]
        self.ep_ids = [self.ep_ids[i] for i in indices]
        print(f"  {split}: {len(self)} episodes")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        eid = str(self.ep_ids[idx])
        T = int(self.mask[idx].sum())
        X = self.X[idx, :T]
        y = torch.full((T,), -100, dtype=torch.long)

        # Fill labels from descriptor windows
        if eid in self.phase_labels:
            for ws, we, phase_id in self.phase_labels[eid]:
                ws_c = max(0, ws); we_c = min(we, T - 1)
                if ws_c <= we_c: y[ws_c:we_c+1] = phase_id

        return X, y[:T], self.mask[idx, :T]


PHASE_MAP = {
    "approach_far_closed_proxy": 0,
    "approach_near_closed_proxy": 1,
    "pre_lock_closed_proxy": 2,
    "grasp_formation_pre_lock_proxy": 3,
    "stable_grasp_or_lift_proxy": 4,
    "natural_open_or_release_proxy": 5,
}
ID_TO_PHASE = {v: k for k, v in PHASE_MAP.items()}


def pad_collate(batch):
    """Pad variable-length sequences to max in batch."""
    max_len = max(b[0].size(0) for b in batch)
    D = batch[0][0].size(1)
    X_pad = torch.zeros(len(batch), max_len, D)
    y_pad = torch.full((len(batch), max_len), -100, dtype=torch.long)
    m_pad = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, m) in enumerate(batch):
        T = x.size(0)
        X_pad[i, :T] = x; y_pad[i, :T] = y; m_pad[i, :T] = m
    return X_pad, y_pad, m_pad


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v3.npz")
    ap.add_argument("--meta-csv", default="data/detector/object_clean_sequences_v3_meta.csv")
    ap.add_argument("--phase-csv", default="tables/object_runtime_phase_events.csv")
    ap.add_argument("--desc-csv", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="outputs/phase_tcn_v0")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def train_epoch(model, loader, optimizer, device):
    model.train(); total_loss = 0.0; n = 0
    for Xb, yb, mb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = 0; total = 0; per_class = defaultdict(lambda: [0, 0])
    for Xb, yb, mb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        pred = logits.argmax(dim=-1)
        mask = (yb != -100)
        correct += (pred[mask] == yb[mask]).sum().item()
        total += mask.sum().item()
        for cid in range(len(PHASE_MAP)):
            cm = mask & (yb == cid)
            per_class[cid][0] += (pred[cm] == cid).sum().item()
            per_class[cid][1] += cm.sum().item()
    f1s = {}
    for cid, (tp, support) in per_class.items():
        if support > 0: f1s[ID_TO_PHASE.get(cid, f"c{cid}")] = round(tp / support, 4)
    macro_f1 = np.mean(list(f1s.values())) if f1s else 0.0
    return correct / max(total, 1), macro_f1, f1s


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_kwargs = dict(npz_path=args.npz_path, meta_csv=args.meta_csv,
                     phase_csv=args.phase_csv, desc_csv=args.desc_csv)
    train_ds = SequencePhaseDataset(**ds_kwargs, split="train")
    val_ds = SequencePhaseDataset(**ds_kwargs, split="val")
    test_ds = SequencePhaseDataset(**ds_kwargs, split="test")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    D = train_ds.X.shape[-1]
    model = PhaseTCN(input_dim=D, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, num_classes=len(PHASE_MAP)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, {D} features, {len(PHASE_MAP)} classes")

    if args.dry_run:
        print("DRY RUN complete."); return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_f1 = 0.0; best_state = None; history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_macro_f1, val_f1s = eval_epoch(model, val_loader, device)
        dt = time.time() - t0

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append(dict(epoch=epoch+1, train_loss=round(train_loss,4),
                           val_acc=round(val_acc,4), val_macro_f1=round(val_macro_f1,4), dt_s=round(dt,2)))
        print(f"  E{epoch+1:3d}: loss={train_loss:.4f} val_acc={val_acc:.3f} val_f1={val_macro_f1:.3f}")

    if best_state: model.load_state_dict(best_state)
    test_acc, test_f1, test_f1s = eval_epoch(model, test_loader, device)
    print(f"\nTest: acc={test_acc:.3f} macro_f1={test_f1:.3f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model_state": best_state or model.state_dict(), "config": vars(args)}, os.path.join(out_dir, "model.pt"))

    # Metrics CSV
    rows = [dict(phase=ID_TO_PHASE.get(i, f"c{i}"), f1=f1) for i, f1 in sorted(test_f1s.items())]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase","f1"])
        w.writeheader(); w.writerows(rows)

    print(f"Saved to {out_dir}")
    print(f"Test per-phase F1: {dict((k[:25],v) for k,v in test_f1s.items())}")


if __name__ == "__main__":
    main()
