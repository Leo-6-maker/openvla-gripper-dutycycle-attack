#!/usr/bin/env python3
import numpy as np, hashlib, os, json, glob, time
from collections import Counter
base = "/mnt/sdc/dty_user/openvla_attack_evidence/c2f"
t0 = time.time()
npz_paths = sorted(glob.glob(base + "/siglip_full_shard_*/c2f_w16_openvla_siglip_dataset.npz"))
print("Found {} NPZs".format(len(npz_paths)))
all_data = {}
for p in npz_paths:
    npz = np.load(p)
    for k in npz.files:
        if k not in all_data: all_data[k] = []
        all_data[k].append(npz[k])
merged = {}
for k, arrs in all_data.items():
    merged[k] = np.concatenate(arrs, axis=0)
out_dir = base + "/siglip_full_final"
os.makedirs(out_dir, exist_ok=True)
path = out_dir + "/c2f_w16_openvla_siglip_full_dataset.npz"
np.savez_compressed(path, **merged)
sha = hashlib.sha256(open(path,"rb").read()).hexdigest()
sz_mb = os.path.getsize(path) / 1024 / 1024
n_wins = len(merged["y_primary"])
n_eps = len(set(str(e) for e in merged["episode_id"]))
suites = Counter(str(s) for s in merged["suite"])
nans = int(np.isnan(merged["X_visual"]).sum()) + int(np.isnan(merged["X_language"]).sum())
y_prim = merged["y_primary"]
suites_arr = merged["suite"]
print("=== FINAL FULL SIGLIP NPZ ===")
print("Episodes: {} (unique)".format(n_eps))
print("Windows: {}".format(n_wins))
print("Size: {:.0f} MB  SHA: {}".format(sz_mb, sha[:16]))
print("NaN: {}".format(nans))
for s in sorted(suites.keys()):
    mask = np.array([str(x)==s for x in suites_arr])
    n = mask.sum(); p = y_prim[mask].sum()
    print("  {}: {} win primary={} ({:.1f}%)".format(s, n, p, p/max(n,1)*100))
print("Time: {:.0f}s".format(time.time()-t0))
