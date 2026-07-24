#!/usr/bin/env python3
"""A1: Audit grip per boundary cases in existing clean30 traces."""
import json, os, csv, sys

DIRS = [
    ("fp32_upstream", "evidence/fp32_eager_upstream_tfjpeg_clean30"),
    ("flash2_upstream", "evidence/bf16_flash2_upstream_tfjpeg_clean30"),
    ("fp32_pil", "evidence/fp32_clean30"),
]

results = {}
for name, d in DIRS:
    exact = 0; eps = 0; near = 0; total = 0; affected = []
    if not os.path.isdir(d):
        print(f"SKIP {name}: dir not found")
        continue
    for ep in sorted(os.listdir(d)):
        tf = os.path.join(d, ep, "trace.csv")
        if not os.path.exists(tf):
            continue
        with open(tf) as f:
            for row in csv.DictReader(f):
                total += 1
                a = row.get("action", row.get("final_action", ""))
                if not a:
                    continue
                p = a.strip().split()
                if len(p) < 7:
                    continue
                try:
                    rg = float(p[6])
                except Exception:
                    continue
                if abs(rg - 0.5) == 0.0:
                    exact += 1
                    if len(affected) < 10:
                        affected.append("%s step%s" % (ep, row.get("step", "?")))
                elif abs(rg - 0.5) < 1e-6:
                    eps += 1
                elif abs(rg - 0.5) < 1e-3:
                    near += 1
    results[name] = {
        "exact_eq": exact, "lt_1e_6": eps, "lt_1e_3": near, "total": total,
        "affected": affected,
        "has_boundary_exact": exact > 0,
    }
    print("%s: total=%d exact=%.0f lt1e-6=%d lt1e-3=%d boundary=%s" % (
        name, total, exact, eps, near, "YES" if exact > 0 else "no"))

out_dir = "migration_audit/detector"
os.makedirs(out_dir, exist_ok=True)
json.dump(results, open(os.path.join(out_dir, "gripper_boundary_audit.json"), "w"), indent=2)

has_boundary = any(r.get("has_boundary_exact", False) for r in results.values())
if has_boundary:
    print("STOP: boundary cases found")
    sys.exit(1)
else:
    print("PASS: no exact boundary cases")
    sys.exit(0)
