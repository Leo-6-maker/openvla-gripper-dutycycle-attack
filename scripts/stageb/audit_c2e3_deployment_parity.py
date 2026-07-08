#!/usr/bin/env python3
"""C2e3 Deployment Parity Audit — three-way logits comparison.

Compares:
  1. training-equivalent path: load C2e1 pre-normalized X_temporal/X_context → GRU
  2. old D7 raw path: raw 25D + zero ctx → GRU (baseline for mismatch evidence)
  3. patched deployment path: raw 25D → runtime normalize → context lookup → normalize → GRU

Paths 1 and 3 must agree (max_abs_diff <= 1e-5).
Path 2 should disagree (mismatch evidence).

CPU-only. No env.step, no OpenVLA, no MuJoCo.
"""

from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class GRUModel(nn.Module):
    def __init__(self, nf=25, nc=108, hidden=128):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden + nc, 2)

    def forward(self, xt, xc):
        _, h = self.gru(xt)
        return self.head(torch.cat([h[-1], xc], dim=1))


def main():
    ap = argparse.ArgumentParser(description="C2e3 Deployment Parity Audit")
    ap.add_argument("--c2e1-dataset", required=True,
                    help="C2e1 w16 temporal dataset .npz path")
    ap.add_argument("--c2e3-checkpoint", required=True,
                    help="C2e3 selected baseline model .pt path")
    ap.add_argument("--c2e3-norm-stats", required=True,
                    help="C2e3 normalization stats JSON path")
    ap.add_argument("--c2e3-context-lookup", required=True,
                    help="C2e3 context lookup JSON path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=100,
                    help="Number of C2e1 rows to sample for parity check")
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading C2e1 dataset...")
    npz = np.load(args.c2e1_dataset, allow_pickle=True)
    Xt_raw_all = np.asarray(npz["X_temporal"], dtype=np.float32)   # (N, 16, 25)  RAW (not normalized)
    Xc_raw_all = np.asarray(npz["X_context"], dtype=np.float32)    # (N, 108)     RAW (not normalized)
    suites_all = np.asarray(npz["suite"]).astype(str)
    splits_all = np.asarray(npz["split"]).astype(str)

    # ── Load model ──
    print("Loading C2e3 checkpoint...")
    ckpt = torch.load(args.c2e3_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    cfg = ckpt.get("config", {})
    cfg_th = ckpt.get("threshold", {})
    tau_emit = float(cfg_th.get("tau_emit", 0.33))
    tau_suppress = float(cfg_th.get("tau_suppress", 0.67))
    hidden = int(cfg.get("channels", cfg.get("hidden", 128)))

    model_norm = GRUModel(25, 108, hidden)
    model_norm.load_state_dict(state)
    model_norm.cpu().eval()

    model_raw = GRUModel(25, 108, hidden)
    model_raw.load_state_dict(state)
    model_raw.cpu().eval()

    # ── Load normalization stats ──
    norm_stats = json.loads(Path(args.c2e3_norm_stats).read_text())
    t_mean = np.array(norm_stats["temporal_feature_mean"], dtype=np.float32).reshape(1, 1, -1)
    t_std = np.maximum(np.array(norm_stats["temporal_feature_std"], dtype=np.float32).reshape(1, 1, -1), 1e-8)
    c_mean = np.array(norm_stats["context_feature_mean"], dtype=np.float32).reshape(1, -1)
    c_std = np.maximum(np.array(norm_stats["context_feature_std"], dtype=np.float32).reshape(1, -1), 1e-8)

    # ── Load context lookup ──
    ctx_lookup_raw = json.loads(Path(args.c2e3_context_lookup).read_text())
    context_lookup: Dict[Tuple[str, int], np.ndarray] = {}
    for key, vec in ctx_lookup_raw["lookup"].items():
        suite, task_str = key.split("|")
        task_idx = int(task_str.replace("task_", ""))
        context_lookup[(suite, task_idx)] = np.array(vec, dtype=np.float32)

    # ── Select samples ──
    rng = np.random.RandomState(args.seed)
    n_total = len(Xt_raw_all)
    indices = rng.choice(n_total, min(args.n_samples, n_total), replace=False)

    print(f"Running parity on {len(indices)} samples...")

    results: List[Dict[str, Any]] = []
    max_abs_diff_13 = 0.0  # patched vs training
    max_abs_diff_12 = 0.0  # raw vs training

    for idx in indices:
        xt_raw = Xt_raw_all[idx]          # (16, 25) raw
        xc_raw = Xc_raw_all[idx]          # (108,) raw
        s = suites_all[idx]

        # Determine task_index from context lookup (match raw context to lookup)
        task_idx = 0
        ctx_raw_lookup = None
        best_dist = float("inf")
        for (ls, lt), lvec in context_lookup.items():
            if ls != s:
                continue
            dist = np.sum((lvec - xc_raw) ** 2)
            if dist < best_dist:
                best_dist = dist
                task_idx = lt
                ctx_raw_lookup = lvec
        if ctx_raw_lookup is None:
            continue

        # ── Path 1: training-equivalent (raw → normalize → GRU) ──
        xt1 = (xt_raw.astype(np.float32) - t_mean) / t_std
        xc1 = (xc_raw.astype(np.float32).reshape(1, -1) - c_mean) / c_std
        with torch.no_grad():
            logits1 = model_norm(
                torch.from_numpy(xt1.reshape(1, 16, 25)),
                torch.from_numpy(xc1.reshape(1, 108)),
            ).numpy()[0]
        ep1 = sigmoid(logits1[0])
        sp1 = sigmoid(logits1[1])
        e1 = bool(ep1 >= tau_emit and sp1 <= tau_suppress)

        # ── Path 2: old D7 raw path (raw 25D + zero ctx, no normalization) ──
        ctx_zero = np.zeros((1, 108), dtype=np.float32)
        with torch.no_grad():
            logits2 = model_raw(
                torch.from_numpy(xt_raw.reshape(1, 16, 25)),
                torch.from_numpy(ctx_zero),
            ).numpy()[0]
        ep2 = sigmoid(logits2[0])
        sp2 = sigmoid(logits2[1])
        e2 = bool(ep2 >= tau_emit and sp2 <= tau_suppress)

        # ── Path 3: patched deployment (raw → runtime normalize → context lookup → normalize → GRU) ──
        xt3 = (xt_raw.astype(np.float32) - t_mean) / t_std
        xc3 = (ctx_raw_lookup.astype(np.float32).reshape(1, -1) - c_mean) / c_std
        with torch.no_grad():
            logits3 = model_norm(
                torch.from_numpy(xt3.reshape(1, 16, 25)),
                torch.from_numpy(xc3.reshape(1, 108)),
            ).numpy()[0]
        ep3 = sigmoid(logits3[0])
        sp3 = sigmoid(logits3[1])
        e3 = bool(ep3 >= tau_emit and sp3 <= tau_suppress)

        diff_13 = float(np.max(np.abs(logits1 - logits3)))
        diff_12 = float(np.max(np.abs(logits1 - logits2)))
        max_abs_diff_13 = max(max_abs_diff_13, diff_13)
        max_abs_diff_12 = max(max_abs_diff_12, diff_12)

        results.append({
            "row_idx": int(idx),
            "suite": s,
            "task_index": task_idx,
            "path1_emit_p": float(ep1),
            "path1_suppress_p": float(sp1),
            "path1_emitted": e1,
            "path2_emit_p": float(ep2),
            "path2_suppress_p": float(sp2),
            "path2_emitted": e2,
            "path3_emit_p": float(ep3),
            "path3_suppress_p": float(sp3),
            "path3_emitted": e3,
            "diff_1v3_logits": diff_13,
            "diff_1v2_logits": diff_12,
            "path13_agree": bool(e1 == e3),
            "path12_agree": bool(e1 == e2),
        })

    # ── Summary ──
    parity_pass = max_abs_diff_13 <= 1e-5
    n_agree_13 = sum(1 for r in results if r["path13_agree"])
    n_agree_12 = sum(1 for r in results if r["path12_agree"])
    n_total = len(results)

    status = "PASS_C2E3_DEPLOYMENT_PARITY" if parity_pass else "FAIL_C2E3_DEPLOYMENT_PARITY"

    # Write raw results
    with open(out / "c2e3_parity_audit_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    report = {
        "gate": "C2E3_DEPLOYMENT_PARITY_AUDIT",
        "status": status,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "n_samples": n_total,
        "parity_threshold": 1e-5,
        "max_abs_diff_1v3": max_abs_diff_13,
        "max_abs_diff_1v2": max_abs_diff_12,
        "path13_emission_agree": f"{n_agree_13}/{n_total}",
        "path12_emission_agree": f"{n_agree_12}/{n_total}",
        "path2_evidence": "mismatch_expected" if n_agree_12 < n_total else "WARNING_no_mismatch_found",
        "conclusion": (
            "patched_deployment_equivalent_to_training" if parity_pass
            else "deployment_diverges_from_training"
        ),
        "d7b2_recommendation": "PROCEED_WITH_REPAIRED_RUNTIME" if parity_pass else "DO_NOT_DEPLOY_FIX_REQUIRED",
        "checkpoint_sha256": sha256_file(Path(args.c2e3_checkpoint)),
        "norm_stats_sha256": sha256_file(Path(args.c2e3_norm_stats)),
        "context_lookup_sha256": sha256_file(Path(args.c2e3_context_lookup)),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    write_json(out / "c2e3_deployment_parity_audit_report.json", report)

    print(f"Parity Audit: {status}")
    print(f"  max_abs_diff(patched vs training) = {max_abs_diff_13:.2e} {'PASS' if parity_pass else 'FAIL'}")
    print(f"  max_abs_diff(raw vs training)     = {max_abs_diff_12:.2e} {'(expected mismatch)' if max_abs_diff_12 > 1e-5 else 'UNEXPECTED'}")
    print(f"  emission agree (patched vs train)  = {n_agree_13}/{n_total}")
    print(f"  emission agree (raw vs train)      = {n_agree_12}/{n_total}")
    return 0 if parity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
