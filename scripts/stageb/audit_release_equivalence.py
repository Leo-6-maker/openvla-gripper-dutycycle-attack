#!/usr/bin/env python3
"""Release equivalence audit: V1 full release path vs injected V1 head on V2 trunk."""
import csv, json, hashlib, math, os, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5_FEATURES, SC5DetectorRuntime

V1_CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
V2_CKPTS = {
    42: REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
    123: REPO / "outputs/sc5_v2_seed123/sc5_mlp_v2.pt",
    456: REPO / "outputs/sc5_v2_seed456/sc5_mlp_v2.pt",
    789: REPO / "outputs/sc5_v2_seed789/sc5_mlp_v2.pt",
    1024: REPO / "outputs/sc5_v2_seed1024/sc5_mlp_v2.pt",
}
DATASET_CSV = REPO / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"
TAU_RELEASE = 0.3

def load_runtime(ckpt_path):
    return SC5DetectorRuntime(str(ckpt_path), tau_corridor=0.3, tau_release=0.3, guard=5)

def extract_release_logit(model, x_norm):
    """Get raw release logit from model (before sigmoid)."""
    with torch.no_grad():
        out = model(torch.tensor(x_norm, dtype=torch.float32))
        return float(out["release_logit"].item())

def main():
    print("Loading V1 runtime...")
    v1_rt = load_runtime(V1_CKPT)
    v1_model = v1_rt.model
    v1_mean = v1_rt.mean; v1_std = v1_rt.std

    print("Loading V2 runtimes...")
    v2_rts = {}
    for seed, path in V2_CKPTS.items():
        v2_rts[seed] = load_runtime(path)
        print(f"  seed{seed}: loaded")

    print("Loading step dataset...")
    val_rows = [r for r in csv.DictReader(open(DATASET_CSV)) if r['split'] == 'val']
    print(f"  val steps: {len(val_rows)}")

    # Group by episode
    ep_rows = defaultdict(list)
    for r in val_rows:
        ep_rows[r['episode_id']].append(r)

    results = {}
    for seed, v2_rt in v2_rts.items():
        print(f"\n=== Seed {seed} ===")
        v2_model = v2_rt.model
        v2_mean = v2_rt.mean; v2_std = v2_rt.std

        all_v1 = []; all_v2 = []
        false_low = 0; false_high = 0; total = 0

        for eid, rows in sorted(ep_rows.items()):
            for r in rows:
                x = np.array([[float(r[fn]) for fn in SC5_FEATURES]], dtype=np.float32)
                if not np.all(np.isfinite(x)):
                    continue

                x_v1 = (x - v1_mean) / (v1_std + 1e-8)
                x_v2 = (x - v2_mean) / (v2_std + 1e-8)

                lp_v1 = extract_release_logit(v1_model, x_v1)
                lp_v2 = extract_release_logit(v2_model, x_v2)

                p_v1 = 1.0 / (1.0 + math.exp(-lp_v1))
                p_v2 = 1.0 / (1.0 + math.exp(-lp_v2))

                all_v1.append(p_v1); all_v2.append(p_v2)

                if p_v1 >= TAU_RELEASE and p_v2 < TAU_RELEASE:
                    false_low += 1
                if p_v1 < TAU_RELEASE and p_v2 >= TAU_RELEASE:
                    false_high += 1
                total += 1

        v1_arr = np.array(all_v1); v2_arr = np.array(all_v2)
        mae = float(np.mean(np.abs(v1_arr - v2_arr)))
        rmse = float(np.sqrt(np.mean((v1_arr - v2_arr)**2)))
        p95 = float(np.percentile(np.abs(v1_arr - v2_arr), 95))
        corr = float(np.corrcoef(v1_arr, v2_arr)[0, 1]) if len(v1_arr) > 1 else 0.0
        agree = 1.0 - (false_low + false_high) / max(total, 1)

        results[seed] = {
            'n_steps': total, 'mae': mae, 'rmse': rmse, 'p95_ae': p95,
            'pearson_corr': corr, 'threshold_agreement': agree,
            'false_low': false_low, 'false_high': false_high,
            'mean_v1': float(np.mean(v1_arr)), 'mean_v2': float(np.mean(v2_arr)),
        }
        print(f"  Steps: {total}  MAE: {mae:.6f}  RMSE: {rmse:.6f}  P95: {p95:.6f}")
        print(f"  Corr: {corr:.6f}  ThreshAgree: {agree:.6f}")
        print(f"  FalseLow: {false_low}  FalseHigh: {false_high}")

    # Summary
    print("\n=== RELEASE EQUIVALENCE SUMMARY ===")
    all_pass = True
    for seed, r in results.items():
        mae_ok = r['mae'] <= 0.05
        p95_ok = r['p95_ae'] <= 0.15
        agree_ok = r['threshold_agreement'] >= 0.99
        fl_ok = r['false_low'] <= max(1, r['n_steps'] * 0.001)
        seed_pass = mae_ok and p95_ok and agree_ok and fl_ok
        status = "PASS" if seed_pass else "FAIL"
        if not seed_pass: all_pass = False
        print(f"  seed{seed}: {status}  MAE={r['mae']:.4f}(<0.05={mae_ok}) P95={r['p95_ae']:.4f}(<0.15={p95_ok}) Agree={r['threshold_agreement']:.4f}(>0.99={agree_ok}) FL={r['false_low']}(ok={fl_ok})")

    # Save
    out = {
        'gate': 'RELEASE_EQUIVALENCE_AUDIT',
        'v1_checkpoint_sha256': hashlib.sha256(open(V1_CKPT,'rb').read()).hexdigest(),
        'tau_release': TAU_RELEASE,
        'val_steps': sum(r['n_steps'] for r in results.values()),
        'per_seed': {str(k): v for k, v in results.items()},
        'all_pass': all_pass,
        'gates': {'mae_le_0.05': all(r['mae'] <= 0.05 for r in results.values()),
                  'p95_le_0.15': all(r['p95_ae'] <= 0.15 for r in results.values()),
                  'threshold_agreement_ge_0.99': all(r['threshold_agreement'] >= 0.99 for r in results.values())},
    }
    out_path = REPO / "evidence/m1c/release_equivalence_audit.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"ALL_PASS: {all_pass}")

if __name__ == "__main__":
    main()
