#!/usr/bin/env python3
"""Phase 7A: P0 evidence closeout — exact manifest, parity audit, artifact hashes."""
import csv, hashlib, json, os, sys, glob, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C, TAU_R, GUARD = 0.3, 0.3, 5
NC_CLEAN = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase6c_nc_clean_shadow"
NC_ATTACK = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase6c_nc_controls"
V2_CKPT = "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt"
OUT = REPO / "evidence/phase6_gpu"


def sha256_file(path):
    if not os.path.exists(path): return "MISSING"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def exact_member_manifest():
    """Build exact manifest of all NC CLEAN shadow cells."""
    rows = []
    for d in sorted(glob.glob(NC_CLEAN + "/*")):
        if not os.path.isdir(d): continue
        cell = os.path.basename(d)
        tel = os.path.join(d, "step_telemetry.csv")
        summary = os.path.join(d, "episode_summary.json")
        stdout = os.path.join(d, "stdout.log")

        n_steps = 0
        n_valid = 0; n_invalid = 0; n_skipped = 0
        if os.path.exists(tel):
            with open(tel) as f:
                for r in csv.DictReader(f):
                    n_steps += 1
                    fv = r.get("feat_valid", "")
                    if fv == "True": n_valid += 1
                    elif fv == "False": n_invalid += 1
                    else: n_skipped += 1

        # Parse task/state from cell name
        # Format: nc_[p|r|x]_t<task>_s<state>
        parts = cell.split("_")
        task = -1; state = -1; pool = "?"
        if len(parts) >= 4:
            pool = parts[1]
            task = int(parts[2][1:]) if parts[2].startswith("t") else -1
            state = int(parts[3][1:]) if parts[3].startswith("s") else -1

        # Check attacked version
        has_attacked = os.path.exists(os.path.join(NC_ATTACK, cell, "step_telemetry.csv"))

        rows.append({
            "cell_id": cell, "pool": pool, "task": task, "state": state,
            "n_steps": n_steps, "n_valid_features": n_valid,
            "n_invalid_features": n_invalid, "has_attacked": has_attacked,
            "tel_sha256": sha256_file(tel)[:16],
            "summary_sha256": sha256_file(summary)[:16],
        })
    return rows


def compute_parity(manifest_rows):
    """Compute online/offline replay parity for each cell."""
    rt = SC5DetectorRuntime(V2_CKPT, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
    parity_rows = []

    for mrow in manifest_rows:
        cell = mrow["cell_id"]
        tel_path = os.path.join(NC_CLEAN, cell, "step_telemetry.csv")
        if not os.path.exists(tel_path):
            parity_rows.append({"cell_id": cell, "parity": "MISSING_TELEMETRY"})
            continue

        rows = list(csv.DictReader(open(tel_path)))
        rows.sort(key=lambda r: int(r.get("step", 0)))

        # Online data from telemetry
        online_arm = -1; online_emit = -1; online_fsm_seq = []
        for r in rows:
            ds = r.get("detector_state", "")
            if ds: online_fsm_seq.append(ds)
            if ds == "ARMED" and online_arm < 0:
                online_arm = int(r.get("step", 0))
            mlp_emit = r.get("mlp_emit", "-1")
            if mlp_emit and int(mlp_emit) >= 0 and online_emit < 0:
                online_emit = int(r.get("step", 0))

        # Offline replay
        rt.reset()
        offline_arm = -1; offline_emit = -1; offline_fsm_seq = []
        cp_errors = []; rp_errors = []
        phase_matches = 0; phase_total = 0

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
            offline_fsm_seq.append(rt.state)

            if rt.state == "ARMED" and offline_arm < 0:
                offline_arm = step
            if dec.get("emitted") and offline_emit < 0:
                offline_emit = step

            # Compare probabilities
            online_cp = float(r.get("corridor_p", "nan"))
            online_rp = float(r.get("release_p", "nan"))
            offline_cp = dec.get("corridor_p", float("nan"))
            offline_rp = dec.get("release_p", float("nan"))
            if not np.isnan(online_cp) and not np.isnan(offline_cp):
                cp_errors.append(abs(online_cp - offline_cp))
            if not np.isnan(online_rp) and not np.isnan(offline_rp):
                rp_errors.append(abs(online_rp - offline_rp))

            online_phase = r.get("pred_phase", "")
            offline_phase = dec.get("pred_phase", "")
            if online_phase and offline_phase:
                phase_total += 1
                if online_phase == offline_phase:
                    phase_matches += 1

        emit_match = online_emit == offline_emit
        arm_match = online_arm == offline_arm
        max_cp_err = max(cp_errors) if cp_errors else 0
        max_rp_err = max(rp_errors) if rp_errors else 0
        phase_acc = phase_matches / max(phase_total, 1)

        parity_rows.append({
            "cell_id": cell,
            "total_steps": mrow["n_steps"],
            "valid_features": mrow["n_valid_features"],
            "invalid_features": mrow["n_invalid_features"],
            "online_arm": online_arm, "offline_arm": offline_arm, "arm_match": arm_match,
            "online_emit": online_emit, "offline_emit": offline_emit, "emit_match": emit_match,
            "max_cp_abs_err": round(max_cp_err, 8),
            "max_rp_abs_err": round(max_rp_err, 8),
            "phase_accuracy": round(phase_acc, 4),
            "parity": "PASS" if (emit_match and arm_match and max_cp_err < 1e-4 and phase_acc > 0.99) else "FAIL",
        })
    return parity_rows


def main():
    os.makedirs(OUT, exist_ok=True)

    print("=== Phase 7A: P0 Evidence Closeout ===")

    # 1. Exact member manifest
    print("\n1. Building exact member manifest...")
    manifest = exact_member_manifest()
    n_total = len(manifest)
    n_valid_feat = sum(1 for m in manifest if m["n_invalid_features"] == 0)
    n_attacked = sum(1 for m in manifest if m["has_attacked"])
    print(f"   Total cells: {n_total}")
    print(f"   Cells with 0 invalid features: {n_valid_feat}")
    print(f"   Cells with attacked versions: {n_attacked}")

    manifest_csv = OUT / "P0_EXACT_MEMBER_MANIFEST.csv"
    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader(); w.writerows(manifest)
    print(f"   Saved: {manifest_csv}")

    # 2. Parity audit
    print("\n2. Computing online/offline parity...")
    parity = compute_parity(manifest)
    n_pass = sum(1 for p in parity if p["parity"] == "PASS")
    n_fail = sum(1 for p in parity if p["parity"] == "FAIL")
    n_missing = sum(1 for p in parity if p["parity"] == "MISSING_TELEMETRY")
    print(f"   PASS: {n_pass}, FAIL: {n_fail}, MISSING: {n_missing}")

    parity_csv = OUT / "P0_ONLINE_OFFLINE_PARITY.csv"
    with open(parity_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=parity[0].keys())
        w.writeheader(); w.writerows(parity)
    print(f"   Saved: {parity_csv}")

    # 3. Artifact hashes
    print("\n3. Computing artifact hashes...")
    hashes = {
        "v2_checkpoint_sha256": sha256_file(V2_CKPT),
        "relabel_csv_sha256": sha256_file(str(OUT / "NC_OFFICIAL_TEACHER_RELABEL.csv")),
        "detector_matrix_csv_sha256": sha256_file(str(OUT / "NC_SAME_TRAJECTORY_DETECTOR_MATRIX.csv")),
        "manifest_csv_sha256": sha256_file(str(manifest_csv)),
        "parity_csv_sha256": sha256_file(str(parity_csv)),
    }
    hashes_path = OUT / "P0_ARTIFACT_HASHES.json"
    with open(hashes_path, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"   Saved: {hashes_path}")

    # 4. Audit JSON
    audit = {
        "gate": "P0_EXACT_MEMBER_AUDIT",
        "total_cells_in_manifest": n_total,
        "duplicate_cells": n_total - len(set(m["cell_id"] for m in manifest)),
        "cells_with_invalid_features": n_total - n_valid_feat,
        "parity_pass": n_pass,
        "parity_fail": n_fail,
        "parity_missing": n_missing,
        "all_parity_pass": n_fail == 0,
        "v2_checkpoint": hashes["v2_checkpoint_sha256"],
        "v2_checkpoint_expected": "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c",
        "v2_checkpoint_match": hashes["v2_checkpoint_sha256"] == "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c",
    }
    audit_path = OUT / "P0_EXACT_MEMBER_AUDIT.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"   Saved: {audit_path}")

    # 5. Summary
    print(f"\n=== P0 CLOSEOUT SUMMARY ===")
    print(f"  Manifest: {n_total} cells, {n_valid_feat} clean")
    print(f"  Parity: {n_pass}P/{n_fail}F/{n_missing}M")
    print(f"  V2 checkpoint: {hashes['v2_checkpoint_sha256'][:16]} match={audit['v2_checkpoint_match']}")
    parity_ok = n_fail == 0
    print(f"  Full Attack Benchmark: {'GO' if parity_ok else 'HOLD (parity)'}")

    parity_json = OUT / "P0_ONLINE_OFFLINE_PARITY.json"
    with open(parity_json, "w") as f:
        json.dump({"gate": "P0_ONLINE_OFFLINE_PARITY", "all_pass": parity_ok,
                    "per_cell": {p["cell_id"]: p for p in parity}}, f, indent=2, default=str)
    print(f"  Saved: {parity_json}")


if __name__ == "__main__":
    main()
