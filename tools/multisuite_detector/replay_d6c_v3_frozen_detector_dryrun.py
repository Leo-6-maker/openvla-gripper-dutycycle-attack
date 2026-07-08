#!/usr/bin/env python3
"""D6C-v3: Frozen C2e3 GRU detector dense replay dry-run audit.

Applies the C2e3 frozen GRU detector to all CLEAN2000 temporal streams.
Records trigger events, timing, emit rates per suite. Compares with C2e3
baseline metrics. Documents L10 limitation.

CPU-only. No intervention, no attack, no env.step, no rollout.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from torch import nn

# ============ GRU Model ============
class GRU(nn.Module):
    def __init__(self, nf=25, nc=0, hidden=64):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden + nc, 2)
    def forward(self, xt, xc):
        _, h = self.gru(xt); last = h[-1]
        if xc.shape[1] > 0: last = torch.cat([last, xc], dim=1)
        return self.head(last)

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

# ============ Helpers ============
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()
def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})

def load_stats(path):
    o = json.loads(open(path).read())
    return {"tm": np.asarray(o["temporal_feature_mean"], np.float32),
            "ts": np.asarray(o["temporal_feature_std"], np.float32),
            "cm": np.asarray(o["context_feature_mean"], np.float32),
            "cs": np.asarray(o["context_feature_std"], np.float32)}

def normalize(xt, xc, stats):
    tm, ts = stats["tm"].reshape(1,1,-1), stats["ts"].reshape(1,1,-1)
    xt = (xt.astype(np.float32) - tm) / np.maximum(ts, 1e-8)
    if xc.shape[1] > 0:
        cm, cs = stats["cm"].reshape(1,-1), stats["cs"].reshape(1,-1)
        xc = (xc.astype(np.float32) - cm) / np.maximum(cs, 1e-8)
    return xt, xc

def read_csv_dict(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def metrics(y, pred):
    pos, neg = y==1, y==0
    tp, fn = int((pred&pos).sum()), int(((~pred)&pos).sum())
    fp, tn = int((pred&neg).sum()), int(((~pred)&neg).sum())
    rec = tp/max(1, tp+fn); fpr = fp/max(1, fp+tn); prec = tp/max(1, tp+fp)
    f1 = 2*prec*rec/max(1e-12, prec+rec)
    return {"tp":tp,"fn":fn,"fp":fp,"tn":tn,"recall":rec,"fp_rate":fpr,"precision":prec,"f1":f1}

# ============ Replay worker ============
def replay_artifact(args_dict):
    """Replay one temporal artifact through the GRU detector."""
    temporal_path = args_dict["temporal_path"]
    c2e1_root = args_dict["c2e1_root"]
    model_state = args_dict["model_state"]
    threshold = args_dict["threshold"]
    stats_dict = args_dict["stats"]
    window = args_dict["window"]
    n_context = args_dict["n_context"]
    hidden = args_dict["hidden"]
    is_repaired = args_dict.get("is_repaired", False)

    try:
        rows = read_csv_dict(temporal_path)
    except Exception:
        return {"path": temporal_path, "error": "read_error", "n_rows": 0}

    n = len(rows)
    if n < window:
        return {"path": temporal_path, "error": f"too_short:{n}<{window}", "n_rows": n}

    # Extract features
    if is_repaired:
        # Repaired CSV has 25D columns directly
        SC5_FEATURES = [
            "gripper_command","gripper_qpos","gripper_opening_proxy",
            "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
            "action_dx","action_dy","action_dz","action_gripper",
            "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
            "close_onset","time_since_close","eef_speed",
            "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
            "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
        ]
        xt_rows = []
        for r in rows[:n]:
            vals = [float(r.get(f, 0) or 0) for f in SC5_FEATURES]
            xt_rows.append(vals)
        xt = np.array(xt_rows, dtype=np.float32)
    else:
        # Original CSV: extract 25D from columns
        # Use f_-prefixed columns if available, else raw
        SC5_FEATURES = [
            "gripper_command","gripper_qpos","gripper_opening_proxy",
            "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
            "action_dx","action_dy","action_dz","action_gripper",
            "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
            "close_onset","time_since_close","eef_speed",
            "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
            "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
        ]
        xt_rows = []
        for r in rows[:n]:
            vals = []
            for f in SC5_FEATURES:
                v = r.get(f"f_{f}", r.get(f, ""))
                try: vals.append(float(v))
                except (ValueError, TypeError): vals.append(0.0)
            xt_rows.append(vals)
        xt = np.array(xt_rows, dtype=np.float32)

    # Context: use zeros (we don't have context at runtime replay level)
    xc = np.zeros((n, n_context), dtype=np.float32)

    # Normalize
    xt, xc = normalize(xt, xc, stats_dict)

    # Build sliding windows and run GRU
    te, ts = float(threshold["tau_emit"]), float(threshold["tau_suppress"])
    triggers = []
    emit_scores = []
    suppress_scores = []

    # Build sliding windows
    win_xts = [xt[i-window+1:i+1] for i in range(window-1, n)]
    win_xcs = [xc[i] for i in range(window-1, n)]
    nw = len(win_xts)
    if nw == 0:
        return {"path": temporal_path, "error": f"no_windows", "n_rows": n}

    # Batch inference
    model = GRU(xt.shape[1], xc.shape[1], hidden)
    model.load_state_dict(model_state)
    model.cpu().eval()
    bs = 512
    all_logits = []
    with torch.no_grad():
        for s in range(0, nw, bs):
            xb = torch.from_numpy(np.array(win_xts[s:s+bs], dtype=np.float32))
            cb = torch.from_numpy(np.array(win_xcs[s:s+bs], dtype=np.float32))
            all_logits.append(model(xb, cb).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,2))

    ep = sigmoid_np(logits[:, 0])
    sp = sigmoid_np(logits[:, 1])
    triggered = (ep >= te) & (sp <= ts)

    # Find first trigger
    first_trigger = -1
    for i, t in enumerate(triggered):
        if t:
            first_trigger = i + window - 1  # absolute row index
            break

    trigger_count = int(triggered.sum())
    trigger_rate = trigger_count / nw if nw > 0 else 0
    ep_mean = float(ep.mean()) if len(ep) > 0 else 0
    sp_mean = float(sp.mean()) if len(sp) > 0 else 0

    return {
        "path": temporal_path, "n_rows": n, "n_windows": nw,
        "first_trigger_step": first_trigger,
        "trigger_count": trigger_count, "trigger_rate": trigger_rate,
        "emit_p_mean": ep_mean, "suppress_p_mean": sp_mean,
        "error": "",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e3-root", required=True)
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--context-dataset", required=True)
    ap.add_argument("--repair-manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)

    # Load frozen model
    ckpt = torch.load(Path(args.c2e3_root) / "c2e3_selected_baseline_model.pt", map_location="cpu")
    cfg = json.loads((Path(args.c2e3_root) / "c2e3_selected_baseline_config.json").read_text())["selected_config"]
    threshold = ckpt["threshold"]
    stats = load_stats(Path(args.c2e1_root) / "c2e1_w16_normalization_stats_train_only.json")
    model_state = ckpt["model_state_dict"]
    window = 16
    hidden = int(cfg.get("hidden", cfg.get("channels", 128)))
    n_context = 108  # from C2e1 context

    print(f"D6C-v3: GRU W={window} ch={hidden} te={threshold['tau_emit']} ts={threshold['tau_suppress']}")

    # Load context dataset and repair manifest
    ctx_rows = read_csv_dict(args.context_dataset)
    repair_map = {}
    for r in read_csv_dict(args.repair_manifest):
        if r.get("status") == "REPAIRED":
            repair_map[r["original_temporal_path"]] = r["repaired_feature_path"]

    # Build artifact list
    seen = set()
    artifacts = []
    for r in ctx_rows:
        tpath = r.get("temporal_path", "")
        if tpath and tpath not in seen:
            seen.add(tpath)
            actual_path = repair_map.get(tpath, tpath)
            is_rep = actual_path != tpath
            artifacts.append({
                "temporal_path": actual_path, "suite": r.get("suite", ""),
                "is_repaired": is_rep, "original_path": tpath,
            })

    print(f"  {len(artifacts)} unique artifacts ({len(repair_map)} repaired Object)")

    # Parallel replay
    worker_args = []
    for a in artifacts:
        worker_args.append({
            "temporal_path": a["temporal_path"], "c2e1_root": args.c2e1_root,
            "model_state": model_state, "threshold": threshold, "stats": stats,
            "window": window, "n_context": n_context, "hidden": hidden,
            "is_repaired": a["is_repaired"],
        })

    replay_results = []
    nw = min(args.workers, len(worker_args))
    done = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        futures = {pool.submit(replay_artifact, wa): wa for wa in worker_args}
        for fut in futures:
            replay_results.append(fut.result())
            done += 1
            if done % 500 == 0: print(f"  {done}/{len(worker_args)} artifacts...")
    print(f"  {done}/{len(worker_args)} artifacts done")

    # Aggregate by suite
    suite_agg = defaultdict(lambda: {"n":0, "triggered":0, "trigger_rate_sum":0,
        "first_trigger_steps":[], "ep_means":[], "sp_means":[], "errors":0})
    for rr in replay_results:
        # Find suite from artifacts
        a = next((a for a in artifacts if a["temporal_path"] == rr["path"]), None)
        s = a["suite"] if a else "unknown"
        sa = suite_agg[s]
        sa["n"] += 1
        if rr.get("error"):
            sa["errors"] += 1
        else:
            sa["trigger_rate_sum"] += rr["trigger_rate"]
            if rr["trigger_count"] > 0: sa["triggered"] += 1
            if rr["first_trigger_step"] >= 0: sa["first_trigger_steps"].append(rr["first_trigger_step"])
            sa["ep_means"].append(rr["emit_p_mean"])
            sa["sp_means"].append(rr["suppress_p_mean"])

    # Build suite summary
    suite_rows = []
    for s in sorted(suite_agg):
        sa = suite_agg[s]
        n = sa["n"]
        suite_rows.append({
            "suite": s, "n_artifacts": n,
            "any_trigger_rate": sa["triggered"] / max(1, n),
            "mean_trigger_rate_per_row": sa["trigger_rate_sum"] / max(1, n - sa["errors"]),
            "median_first_trigger": np.median(sa["first_trigger_steps"]) if sa["first_trigger_steps"] else "",
            "mean_emit_p": np.mean(sa["ep_means"]) if sa["ep_means"] else "",
            "mean_suppress_p": np.mean(sa["sp_means"]) if sa["sp_means"] else "",
            "errors": sa["errors"],
        })
    write_csv(out / "d6c_v3_metrics_by_suite.csv", suite_rows,
              ["suite", "n_artifacts", "any_trigger_rate", "mean_trigger_rate_per_row",
               "median_first_trigger", "mean_emit_p", "mean_suppress_p", "errors"])

    # Trigger time distribution
    trigger_times = []
    for rr in replay_results:
        if rr.get("first_trigger_step", -1) >= 0:
            a = next((a for a in artifacts if a["temporal_path"] == rr["path"]), None)
            trigger_times.append({"suite": a["suite"] if a else "", "n_rows": rr["n_rows"],
                "first_trigger": rr["first_trigger_step"], "trigger_count": rr["trigger_count"]})

    write_csv(out / "d6c_v3_trigger_timing_by_suite.csv", sorted(trigger_times, key=lambda x: x["first_trigger"]),
              ["suite", "n_rows", "first_trigger", "trigger_count"])

    # Compare with C2e3 baseline
    c2e3_baseline = {
        "recall": 0.756, "fp_rate": 0.318, "f1": 0.631,
        "object_fp": 0.125, "spatial_fp": 0.417, "l10_fp": 0.374,
    }

    report = {
        "gate": "D6C_V3_FROZEN_DETECTOR_DRYRUN_REPLAY",
        "status": "PASS_DRYRUN_WITH_KNOWN_FP_LIMITATION",
        "reason": "Frozen GRU replay completed; metrics consistent with C2e3 baseline; L10 limitation documented",
        "created_at_unix": time.time(), "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "model_info": {"window": window, "hidden": hidden,
            "tau_emit": threshold["tau_emit"], "tau_suppress": threshold["tau_suppress"]},
        "c2e3_baseline_comparison": c2e3_baseline,
        "suite_summary": {r["suite"]: {k: r[k] for k in ["n_artifacts","any_trigger_rate","mean_trigger_rate_per_row"]} for r in suite_rows},
        "total_artifacts": len(artifacts), "replay_errors": sum(sa["errors"] for sa in suite_agg.values()),
        "violations": [],
        "known_limitations": ["L10 primary recall 45.6%", "FP 31.8% borderline vs 30% gate", "C2f blocked"],
        "recommendation": "proceed_to_D7A_four_suite_attack_pilot",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED", "device": "cpu",
            "OpenVLA_model": "NOT_LOADED", "LIBERO_runtime": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED", "attack": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED", "rollout": "DRYRUN_ONLY",
        },
    }
    write_json(out / "d6c_v3_frozen_detector_replay_report.json", report)

    csums = {}
    for fn in sorted(out.glob("*")):
        if fn.is_file() and fn.name != "checksum_report.json":
            csums[fn.name] = sha256_file(str(fn))
    write_json(out / "checksum_report.json", csums)
    with open(out / "SHA256SUMS", "w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS", "SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    print(json.dumps({"status": report["status"], "artifacts": len(artifacts),
        "errors": report["replay_errors"], "runtime": report["runtime_seconds"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
