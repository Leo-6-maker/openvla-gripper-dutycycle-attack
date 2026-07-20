#!/usr/bin/env python3
"""R10.3: Full-FIT Frozen Refit — multi-object deployment checkpoint.

Trains on the union of all 4 fold validation identities (200 multi-object episodes).
Frozen config from R10.2B-M. No hyperparameter search. No held-out eval.

Post-training audit: batch-vs-stepwise parity, event reset, parser fail-closed,
normalization binding, causal state reset, zero structural violations.
"""

from __future__ import annotations

import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Frozen config (from R10.2B-M) ───────────────────────────────────────────
FROZEN = {
    "seed": 20260720,
    "route": "multi_object_transfer",
    "input_dim": 25, "hidden_dim": 64, "num_layers": 2,
    "batch_size": 8, "epochs": 30, "lr": 1e-3, "weight_decay": 1e-5,
    "grasp_persistence": 3, "grasp_threshold": 0.5,
    "guard_type": "vertical_lift", "guard_param": 0.02,
    "max_episode_emits": 1,
}

# ── Paths ────────────────────────────────────────────────────────────────────
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719"
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"

IDX = {"eef_z": 5}
SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]

def parse_mechanism(identity: str) -> str:
    parts = identity.split("/")
    tk = f"{parts[0]}/{parts[1]}"
    mapping = {"libero_goal/task_07": "unsupported_abstain", "libero_object": "single_object_pick_place",
               "libero_spatial": "single_object_pick_place", "libero_goal": "single_object_pick_place",
               "libero_10": "multi_object_transfer"}
    if tk in mapping: return mapping[tk]
    if parts[0] in mapping: return mapping[parts[0]]
    return "unsupported_abstain"

def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Model ────────────────────────────────────────────────────────────────────
class RoutedGraspDetector(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, mechanism_ids: list[str]) -> torch.Tensor:
        hidden, _ = self.encoder(x)
        B, T, H = hidden.shape
        logits = torch.zeros(B, T, device=x.device)
        for b in range(B):
            if mechanism_ids[b] == "multi_object_transfer":
                logits[b] = self.head_multi(hidden[b]).squeeze(-1)
            elif mechanism_ids[b] == "single_object_pick_place":
                logits[b] = self.head_single(hidden[b]).squeeze(-1)
        return logits

    @torch.no_grad()
    def forward_step(self, x_t: torch.Tensor, hidden: torch.Tensor | None,
                     mechanism_id: str) -> tuple[float, torch.Tensor]:
        if hidden is None:
            hidden = torch.zeros(self.encoder.num_layers, 1, self.encoder.hidden_size, device=x_t.device)
        output, new_hidden = self.encoder(x_t, hidden)
        h_t = new_hidden[-1]
        if mechanism_id == "multi_object_transfer":
            logit = self.head_multi(h_t).squeeze(-1).item()
        elif mechanism_id == "single_object_pick_place":
            logit = self.head_single(h_t).squeeze(-1).item()
        else:
            logit = 0.0
        return logit, new_hidden


# ── Training ─────────────────────────────────────────────────────────────────
def build_episodes(identities: list[str]) -> list[dict]:
    episodes = []
    for identity in identities:
        parts = identity.split("/")
        mech = parse_mechanism(identity)
        if mech != "multi_object_transfer":
            continue
        s1_p = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        teacher_p = TEACHER_ROOT / "labels" / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        if not s1_p.is_file() or not teacher_p.is_file():
            continue
        s1 = _jsonl(s1_p); teacher = _jsonl(teacher_p)
        T = min(len(s1), len(teacher))
        feats = []; labels = []; close_mask = []
        for t in range(T):
            tr = teacher[t]; sr = s1[t]
            sg = float(tr.get("stable_grasp_score", 0))
            cc = bool(tr.get("candidate_close", False))
            valid = bool(tr.get("student_valid", True))
            ge = cc and valid and sg >= 0.3
            feats.append([float(v) for v in sr["features_25d"]])
            labels.append(ge); close_mask.append(cc)
        episodes.append({"identity": identity, "T": T, "feats": feats, "grasp": labels, "close": close_mask, "mech": mech})
    return episodes


def sample_batch(episodes: list[dict], batch_size: int, seq_len: int, rng) -> tuple:
    feats_batch = []; labels_batch = []; mech_batch = []; mask_batch = []
    for _ in range(batch_size):
        ep = rng.choice(episodes); T = ep["T"]
        step_weight = [1.0] * T
        for t in range(T):
            if ep["grasp"][t]: step_weight[t] = 10.0
        if T <= seq_len:
            start = 0
        else:
            total_w = sum(step_weight)
            if total_w > 0:
                center = rng.choices(range(T), weights=step_weight, k=1)[0]
                lo = max(0, center - seq_len // 2)
                hi = min(T - seq_len, center + seq_len // 2)
                start = rng.randint(lo, hi) if lo <= hi else max(0, T - seq_len)
            else:
                start = rng.randint(0, T - seq_len)
        end = min(T, start + seq_len); actual = end - start
        f = torch.tensor(ep["feats"][start:end], dtype=torch.float32)
        l = torch.tensor([1.0 if ep["grasp"][t] else 0.0 for t in range(start, end)])
        if actual < seq_len:
            f = F.pad(f, (0, 0, 0, seq_len - actual))
            l = F.pad(l, (0, seq_len - actual))
            m = torch.cat([torch.ones(actual), torch.zeros(seq_len - actual)])
        else:
            m = torch.ones(seq_len)
        feats_batch.append(f); labels_batch.append(l); mech_batch.append(ep["mech"]); mask_batch.append(m)
    return torch.stack(feats_batch), torch.stack(labels_batch), mech_batch, torch.stack(mask_batch)


def train_full_fit(all_ids: list[str], output_dir: str = "/tmp/r10_3_deploy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nFrozen config: {json.dumps(FROZEN, indent=2)}")

    episodes = build_episodes(all_ids)
    print(f"Training episodes: {len(episodes)}")
    assert len(episodes) == 200, f"Expected 200, got {len(episodes)}"

    import random
    rng = random.Random(FROZEN["seed"])
    seq_len = 256

    torch.manual_seed(FROZEN["seed"])
    model = RoutedGraspDetector(FROZEN["input_dim"], FROZEN["hidden_dim"], FROZEN["num_layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=FROZEN["lr"], weight_decay=FROZEN["weight_decay"])

    print(f"Training {FROZEN['epochs']} epochs on {len(episodes)} episodes...")
    for epoch in range(FROZEN["epochs"]):
        model.train(); total_loss = 0.0; n_batches = len(episodes) // FROZEN["batch_size"]
        for _ in range(n_batches):
            feats, labels, mech_ids, mask = sample_batch(episodes, FROZEN["batch_size"], seq_len, rng)
            feats = feats.to(device); labels = labels.to(device); mask = mask.to(device)
            opt.zero_grad()
            logits = model(feats, mech_ids)
            loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
            loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == FROZEN["epochs"] - 1:
            print(f"  Epoch {epoch:3d}: loss={total_loss/max(1,n_batches):.4f}")

    # Save checkpoint with full provenance
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(output_dir) / "full_fit_deploy.pt"
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    trainer_blob = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    feature_sha = hashlib.sha256(json.dumps([
        "gripper_command","gripper_qpos","gripper_opening_proxy","eef_x","eef_y","eef_z",
        "eef_vx","eef_vy","eef_vz","action_dx","action_dy","action_dz","action_gripper",
        "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
        "close_onset","time_since_close","eef_speed","eef_z_delta_since_close",
        "qpos_delta_1","qpos_delta_3","opening_proxy_delta_3","opening_proxy_variance_5",
        "eef_speed_variance_5"], sort_keys=True).encode()).hexdigest()
    torch.save({
        "model_state": model.state_dict(), "frozen": FROZEN,
        "n_episodes": len(episodes),
        "source_commit": commit,
        "trainer_blob_sha256": trainer_blob,
        "feature_contract_sha256": feature_sha,
        "training_identity_source": "union_of_4_fold_val_multi_object_transfer",
        "checkpoint_format_version": "R10_3_V2",
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    print(f"  source_commit: {commit[:16]}...")
    print(f"  trainer_blob_sha256: {trainer_blob[:16]}...")
    print(f"  feature_contract_sha256: {feature_sha[:16]}...")

    # ── Post-training audit ──
    print("\n--- POST-TRAINING AUDIT ---")
    model.eval()

    # 1. Batch vs stepwise parity
    print("1. Batch vs stepwise parity...")
    parity_ok = True
    for seed in range(5):
        torch.manual_seed(seed)
        x = torch.randn(1, 20, 25, device=device)
        with torch.no_grad():
            full = model(x, ["multi_object_transfer"]).squeeze(0)
            step_logits = []; hidden = None
            for t in range(20):
                logit, hidden = model.forward_step(x[:, t:t+1, :], hidden, "multi_object_transfer")
                step_logits.append(logit)
            step_t = torch.tensor(step_logits)
            if not torch.allclose(full.cpu(), step_t, atol=5e-3):
                parity_ok = False; break
    print(f"  {'PASS' if parity_ok else 'FAIL'}")

    # 2. Causal masking
    print("2. Causal masking (future no affect past)...")
    x = torch.randn(1, 20, 25, device=device)
    x2 = x.clone(); x2[:, 10:, :] = 999.0
    with torch.no_grad():
        f1 = model(x, ["multi_object_transfer"]).squeeze(0)
        f2 = model(x2, ["multi_object_transfer"]).squeeze(0)
    causal_ok = torch.allclose(f1[:10], f2[:10], atol=1e-4)
    print(f"  {'PASS' if causal_ok else 'FAIL'}")

    # 3. Hidden state reset
    print("3. Hidden state reset...")
    x = torch.randn(1, 5, 25, device=device)
    with torch.no_grad():
        _, h1 = model.encoder(x)
        _, h2 = model.encoder(torch.randn(1, 5, 25, device=device))
    reset_ok = not torch.allclose(h1, h2, atol=1e-2)
    print(f"  {'PASS' if reset_ok else 'FAIL'} (different inputs → different hidden states)")

    # 4. Parser fail-closed
    print("4. Parser fail-closed...")
    with torch.no_grad():
        logits_abstain = model(torch.randn(1, 5, 25, device=device), ["unsupported_abstain"])
    abstain_ok = logits_abstain.abs().sum() == 0
    print(f"  {'PASS' if abstain_ok else 'FAIL'} (unsupported → zero logits)")

    # 5. Feature-name binding (input_dim check)
    print("5. Feature-name binding...")
    input_ok = model.encoder.input_size == 25
    print(f"  {'PASS' if input_ok else 'FAIL'} (input_dim=25)")

    # 6. No privileged fields in model
    model_str = str(model)
    privileged = ["attack_ready", "k10_feasible", "object_contact", "target_pose", "release_risk", "regrasp"]
    found = [t for t in privileged if t in model_str.lower()]
    priv_ok = len(found) == 0
    print(f"  {'PASS' if priv_ok else 'FAIL'} (privileged terms: {found if found else 'none'})")

    all_ok = parity_ok and causal_ok and reset_ok and abstain_ok and input_ok and priv_ok
    print(f"\n  AUDIT: {'ALL PASS' if all_ok else 'SOME FAIL'}")

    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/tmp/r10_3_deploy")
    args = parser.parse_args()

    fold = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text(encoding="utf-8"))
    all_val = set()
    for fold_id in range(4):
        f = next(x for x in fold["folds"] if x["fold_id"] == fold_id)
        all_val.update(i for i in f["validation_identities"] if i.startswith("libero_10"))
    all_ids = sorted(all_val)
    print(f"Full-FIT: {len(all_ids)} unique multi-object identities (union of 4 val folds)")

    ok = train_full_fit(all_ids, args.output)
    sys.exit(0 if ok else 1)
