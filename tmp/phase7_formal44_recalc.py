#!/usr/bin/env python3
"""Phase 7 Object: Formal-44 NC recalculation + TV paired fourfold.

Key requirements:
  - Denominator = 44 formal eligible NC (46 total minus 2 overlap: t1_s0, t8_s2)
  - Report M1, M1-OS, M2 all on same formal-44
  - TV paired fourfold table
  - McNemar exact test on paired counts
"""
import csv, json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
SERVER_REPO = Path("/mnt/sdc/dty_user/openvla_attack")

sys.path.insert(0, str(SERVER_REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C, TAU_R, GUARD = 0.3, 0.3, 5

DETECTORS = {
    "M1": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_primary_seed42/sc5_mlp_v2.pt",
    "M1_OS": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_oversampled_seed42/sc5_mlp_v2.pt",
    "M2": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
}

CENSUS_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase7b_nc_census"
RELABEL_CSV = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/NC_CENSUS_TEACHER_RELABEL.csv"
OUT_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object"

# Excluded cells (overlap with old_object11)
EXCLUDED_CELLS = {"census_t1_s0", "census_t8_s2"}


def load_telemetry(cell):
    tel = os.path.join(CENSUS_DIR, cell, "step_telemetry.csv")
    if not os.path.exists(tel): return None
    rows = list(csv.DictReader(open(tel)))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def replay_detector(rt, rows):
    rt.reset()
    arm_step = -1; emit_step = -1
    for r in rows:
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            val = r.get(f"f_{fn}", r.get(fn, ""))
            if val in ("", "nan", "NaN", None):
                ok = False; break
            try: feats[fn] = float(val)
            except: ok = False; break
        if not ok: continue
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)): continue
        step = int(r.get("step", 0))
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)
        if rt.state == "ARMED" and arm_step < 0: arm_step = step
        if dec.get("emitted") and emit_step < 0: emit_step = step
    return {"emitted": rt.emitted, "emit_step": emit_step, "armed": arm_step >= 0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Teacher relabel
    relabel = {}
    if os.path.exists(RELABEL_CSV):
        for r in csv.DictReader(open(RELABEL_CSV)):
            relabel[r["cell_id"]] = r

    # Identify formal NC cells (Teacher NC, not excluded)
    all_nc = [c for c, r in relabel.items() if r.get("teacher_category") == "NC"]
    formal_nc = [c for c in all_nc if c not in EXCLUDED_CELLS]
    excluded = [c for c in all_nc if c in EXCLUDED_CELLS]

    print(f"All NC: {len(all_nc)}")
    print(f"Excluded (overlap): {len(excluded)}: {excluded}")
    print(f"Formal eligible NC: {len(formal_nc)}")

    # Load detectors
    runtimes = {}
    for name, path in DETECTORS.items():
        if os.path.exists(path):
            runtimes[name] = SC5DetectorRuntime(path, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
            print(f"  {name}: SHA={runtimes[name].checkpoint_sha256[:16]}")
        else:
            print(f"  {name}: MISSING")

    # Replay all formal NC cells
    print(f"\nReplaying {len(formal_nc)} formal NC cells...")
    nc_results = []
    for cell in sorted(formal_nc):
        rows = load_telemetry(cell)
        if rows is None:
            print(f"  {cell}: NO TELEMETRY")
            continue
        rl = relabel.get(cell, {})
        row = {"cell_id": cell, "task": rl.get("task", -1), "state": rl.get("state", -1),
               "teacher_category": "NC"}
        for name, rt in runtimes.items():
            res = replay_detector(rt, rows)
            row[f"{name}_emitted"] = res["emitted"]
            row[f"{name}_emit_step"] = res["emit_step"]
            row[f"{name}_armed"] = res["armed"]
        nc_results.append(row)

    # ── Formal-44 NC FT counts ──
    print("\n=== Formal-44 NC Results ===")
    for name in DETECTORS:
        ft = [r for r in nc_results if r.get(f"{name}_emitted")]
        print(f"  {name}: {len(ft)}/{len(nc_results)} FT ({round(100*len(ft)/len(nc_results),2)}%)")
        if ft:
            for f in ft:
                print(f"    FT: {f['cell_id']} emit={f[f'{name}_emit_step']}")

    # ── TV paired fourfold ──
    # Load TV cells from relabel
    tv_cells = [c for c, r in relabel.items() if r.get("teacher_category") == "TV"]
    print(f"\nReplaying {len(tv_cells)} TV cells...")
    tv_results = []
    for cell in sorted(tv_cells):
        rows = load_telemetry(cell)
        if rows is None: continue
        rl = relabel.get(cell, {})
        row = {"cell_id": cell, "task": rl.get("task", -1), "state": rl.get("state", -1),
               "teacher_category": "TV"}
        for name, rt in runtimes.items():
            res = replay_detector(rt, rows)
            row[f"{name}_emitted"] = res["emitted"]
            row[f"{name}_emit_step"] = res["emit_step"]
        tv_results.append(row)

    print("\n=== TV Recall ===")
    for name in DETECTORS:
        hit = [r for r in tv_results if r.get(f"{name}_emitted")]
        print(f"  {name}: {len(hit)}/{len(tv_results)} ({round(100*len(hit)/len(tv_results),1)}%)")

    # ── Paired fourfold: M1 vs M2 on TV ──
    print("\n=== TV Paired Fourfold: M1 vs M2 ===")
    m1m2_tv = {"both_emit": 0, "both_miss": 0, "m1_only": 0, "m2_only": 0}
    for r in tv_results:
        m1e = r.get("M1_emitted", False)
        m2e = r.get("M2_emitted", False)
        if m1e and m2e: m1m2_tv["both_emit"] += 1
        elif not m1e and not m2e: m1m2_tv["both_miss"] += 1
        elif m1e and not m2e: m1m2_tv["m1_only"] += 1
        elif not m1e and m2e: m1m2_tv["m2_only"] += 1

    for k, v in m1m2_tv.items():
        print(f"  {k}: {v}")

    # McNemar exact test on discordant pairs
    b = m1m2_tv["m1_only"]
    c = m1m2_tv["m2_only"]
    if b + c > 0:
        p_mcnemar = binomtest(min(b, c), n=b+c, p=0.5, alternative="two-sided").pvalue
        print(f"  McNemar exact p={round(p_mcnemar, 4)} (discordant: M1_only={b}, M2_only={c})")
    else:
        p_mcnemar = 1.0
        print(f"  McNemar: no discordant pairs (p=1.0)")

    # ── Paired fourfold: M1 vs M2 on formal-44 NC ──
    print("\n=== NC Paired Fourfold: M1 vs M2 on Formal-44 ===")
    m1m2_nc = {"both_abstain": 0, "both_emit": 0, "m1_only": 0, "m2_only": 0}
    for r in nc_results:
        m1e = r.get("M1_emitted", False)
        m2e = r.get("M2_emitted", False)
        if m1e and m2e: m1m2_nc["both_emit"] += 1
        elif not m1e and not m2e: m1m2_nc["both_abstain"] += 1
        elif m1e and not m2e: m1m2_nc["m1_only"] += 1
        elif not m1e and m2e: m1m2_nc["m2_only"] += 1

    for k, v in m1m2_nc.items():
        print(f"  {k}: {v}")

    # ── Save outputs ──
    # NC formal-44 CSV
    nc_fields = ["cell_id", "task", "state", "teacher_category"]
    for name in DETECTORS:
        nc_fields += [f"{name}_emitted", f"{name}_emit_step", f"{name}_armed"]
    with open(os.path.join(OUT_DIR, "FORMAL44_NC_DETECTOR_MATRIX.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=nc_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(nc_results)

    # TV CSV
    tv_fields = nc_fields
    with open(os.path.join(OUT_DIR, "TV_DETECTOR_MATRIX.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tv_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(tv_results)

    # Summary JSON
    summary = {
        "gate": "FORMAL44_NC_AND_TV_RECALCULATION",
        "nc_total": len(all_nc),
        "nc_excluded": excluded,
        "nc_formal_eligible": len(formal_nc),
        "nc_formal_replayed": len(nc_results),
        "nc_ft": {},
        "tv_total": len(tv_cells),
        "tv_replayed": len(tv_results),
        "tv_recall": {},
        "tv_paired_m1_vs_m2": m1m2_tv,
        "tv_mcnemar_p": round(p_mcnemar, 6) if b+c > 0 else 1.0,
        "nc_paired_m1_vs_m2": m1m2_nc,
        "detector_checkpoints": {n: runtimes[n].checkpoint_sha256 for n in runtimes},
    }
    for name in DETECTORS:
        ft_cells = [r["cell_id"] for r in nc_results if r.get(f"{name}_emitted")]
        summary["nc_ft"][name] = {"count": len(ft_cells), "rate": f"{len(ft_cells)}/{len(nc_results)}",
                                   "cells": ft_cells}
        tv_hit = len([r for r in tv_results if r.get(f"{name}_emitted")])
        summary["tv_recall"][name] = {"count": tv_hit, "rate": f"{tv_hit}/{len(tv_results)}",
                                       "pct": round(100*tv_hit/len(tv_results), 1)}

    with open(os.path.join(OUT_DIR, "FORMAL44_NC_TV_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== OUTPUTS SAVED to {OUT_DIR} ===")
    print(f"  FORMAL44_NC_DETECTOR_MATRIX.csv")
    print(f"  TV_DETECTOR_MATRIX.csv")
    print(f"  FORMAL44_NC_TV_SUMMARY.json")


if __name__ == "__main__":
    main()
