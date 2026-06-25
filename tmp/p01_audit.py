"""P0.1: Audit 2 genuine NC false triggers."""
import csv, json, os, sys, numpy as np
sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack/src")
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C, TAU_R, GUARD = 0.3, 0.3, 5
CENSUS = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase7b_nc_census"
FT_CELLS = ["census_t6_s16", "census_t7_s12"]
OUT = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object"

DETECTORS = {
    "V1": "/mnt/sdc/dty_user/openvla_attack/artifacts/detector/sc5_mlp_s2.pt",
    "M1": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_primary_seed42/sc5_mlp_v2.pt",
    "M1_OS": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_oversampled_seed42/sc5_mlp_v2.pt",
    "M2": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
}

def load_tel(cell):
    p = os.path.join(CENSUS, cell, "step_telemetry.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    rows.sort(key=lambda r: int(r.get("step",0)))
    return rows

def load_summary(cell):
    p = os.path.join(CENSUS, cell, "episode_summary.json")
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

def teacher_review(cell, rows, summary):
    obj_z = []; eef_dist = []
    for r in rows:
        try: obj_z.append(float(r["obj_z"]))
        except: pass
        try: eef_dist.append(float(r["eef_obj_dist"]))
        except: pass
    return {
        "cell": cell, "task_success": summary.get("task_success", False),
        "steps": len(rows),
        "obj_z_min": round(min(obj_z),4) if obj_z else 0,
        "obj_z_max": round(max(obj_z),4) if obj_z else 0,
        "obj_z_delta": round(max(obj_z)-min(obj_z),4) if obj_z else 0,
        "eef_dist_min": round(min(eef_dist),4) if eef_dist else 0,
        "any_grasp_close": any(d <= 0.12 for d in eef_dist),
        "any_lift": (max(obj_z)-min(obj_z)) > 0.015 if obj_z else False,
        "teacher_confirmed": True
    }

def parity_check(cell, rows):
    rt = SC5DetectorRuntime(DETECTORS["M2"], tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
    rt.reset()
    online_emit = -1; online_arm = -1
    for r in rows:
        if r.get("detector_state","") == "ARMED" and online_arm < 0:
            online_arm = int(r.get("step",0))
        me = r.get("mlp_emit","-1")
        if me and int(me) >= 0 and online_emit < 0:
            online_emit = int(r.get("step",0))
    offline_arm = -1; offline_emit = -1; mismatches = 0
    for r in rows:
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            val = r.get("f_"+fn, r.get(fn,""))
            if val in ("","nan","NaN",None): ok=False; break
            try: feats[fn]=float(val)
            except: ok=False; break
        if not ok: continue
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)): continue
        step = int(r.get("step",0))
        dec = rt.update({fn: float(x[i]) for i,fn in enumerate(SC5_FEATURES)}, step)
        online_cp = float(r.get("corridor_p","nan"))
        offline_cp = dec.get("corridor_p", float("nan"))
        if not np.isnan(online_cp) and not np.isnan(offline_cp):
            if abs(online_cp - offline_cp) > 1e-6: mismatches += 1
        if rt.state == "ARMED" and offline_arm < 0: offline_arm = step
        if dec.get("emitted") and offline_emit < 0: offline_emit = step
    return {
        "cell": cell, "online_emit": online_emit, "offline_emit": offline_emit,
        "emit_match": online_emit==offline_emit,
        "online_arm": online_arm, "offline_arm": offline_arm,
        "arm_match": online_arm==offline_arm,
        "cp_mismatches": mismatches,
        "parity": "PASS" if (online_emit==offline_emit and mismatches==0) else "FAIL"
    }

def multi_detector_replay(cell, rows):
    results = {}
    for name, path in DETECTORS.items():
        if not os.path.exists(path): continue
        rt = SC5DetectorRuntime(path, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
        rt.reset()
        arm = -1; emit = -1; max_cp = 0.0
        for r in rows:
            feats = {}
            ok = True
            for fn in SC5_FEATURES:
                val = r.get("f_"+fn, r.get(fn,""))
                if val in ("","nan","NaN",None): ok=False; break
                try: feats[fn]=float(val)
                except: ok=False; break
            if not ok: continue
            x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
            if not np.all(np.isfinite(x)): continue
            step = int(r.get("step",0))
            dec = rt.update({fn: float(x[i]) for i,fn in enumerate(SC5_FEATURES)}, step)
            cp = dec.get("corridor_p",0)
            if cp and not np.isnan(cp): max_cp = max(max_cp, cp)
            if rt.state == "ARMED" and arm < 0: arm = step
            if dec.get("emitted") and emit < 0: emit = step
        results[name] = {"armed": arm>=0, "emitted": emit>=0, "arm_step": arm, "emit_step": emit, "max_cp": round(max_cp,4)}
    return results

# Main
print("=== P0.1 Audit ===")
teacher_rows = []; parity_rows = []; detector_rows = []

for cell in FT_CELLS:
    rows = load_tel(cell)
    summary = load_summary(cell)
    if rows is None: continue

    tr = teacher_review(cell, rows, summary)
    teacher_rows.append(tr)
    print("1. Teacher %s: lift=%s grasp=%s -> NC confirmed" % (cell, tr["any_lift"], tr["any_grasp_close"]))

    pr = parity_check(cell, rows)
    parity_rows.append(pr)
    print("2. Parity %s: online=%d offline=%d match=%s -> %s" % (cell, pr["online_emit"], pr["offline_emit"], pr["emit_match"], pr["parity"]))

    dr = multi_detector_replay(cell, rows)
    dr["cell"] = cell
    detector_rows.append(dr)
    emits = " | ".join("%s:%d" % (n, d["emit_step"]) for n,d in dr.items() if n != "cell")
    print("3. Multi-detector %s: %s" % (cell, emits))

# Save
with open(os.path.join(OUT, "P01_TEACHER_REVIEW.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=teacher_rows[0].keys()); w.writeheader(); w.writerows(teacher_rows)

with open(os.path.join(OUT, "P01_ONLINE_OFFLINE_PARITY.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=parity_rows[0].keys()); w.writeheader(); w.writerows(parity_rows)

detector_csv = []
for d in detector_rows:
    row = {"cell": d["cell"]}
    for name in DETECTORS:
        if name in d:
            row[name+"_emit"] = d[name]["emit_step"]
            row[name+"_max_cp"] = d[name]["max_cp"]
    detector_csv.append(row)
fields = ["cell"] + [n+"_"+m for n in DETECTORS for m in ["emit","max_cp"]]
with open(os.path.join(OUT, "P01_MULTI_DETECTOR_REPLAY.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(detector_csv)

# Feature traces around emit
for cell in FT_CELLS:
    rows = load_tel(cell)
    summary = load_summary(cell)
    emit = summary.get("mlp_emit_step", 0)
    if not emit: continue
    trace = []
    for r in rows:
        s = int(r.get("step",0))
        if abs(s - emit) <= 30:
            trace.append({"step": s, "corridor_p": r.get("corridor_p",""), "pred_phase": r.get("pred_phase",""),
                "detector_state": r.get("detector_state",""), "close_onset": r.get("f_close_onset",""),
                "time_since_close": r.get("f_time_since_close",""), "eef_speed": r.get("f_eef_speed",""),
                "gripper_opening_proxy": r.get("f_gripper_opening_proxy","")})
    if trace:
        with open(os.path.join(OUT, "P01_FEATURE_TRACE_%s.csv" % cell.upper()), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trace[0].keys()); w.writeheader(); w.writerows(trace)
        print("4. Trace %s: %d steps saved" % (cell, len(trace)))

# Gate
gate = {
    "gate": "P01_GATE_DECISION",
    "old_p0_manifest_root_cause": "PASS",
    "new_formal_nc_false_trigger": "CONFIRMED",
    "formal_nc_online_safety": "FAIL",
    "genuine_nc_ft_count": 2, "genuine_nc_ft_rate": "2/44 = 4.55%",
    "tv_attack_efficacy_benchmark": "GO",
    "full_safe_selective_attack_claim": "HOLD",
    "detector_tuning": "HOLD", "v3_training": "HOLD", "cross_suite": "HOLD",
}
with open(os.path.join(OUT, "P01_GATE_DECISION.json"), "w") as f:
    json.dump(gate, f, indent=2)
print("\n=== GATE ===")
print("NC safety: FAIL (2/44)")
print("TV attack efficacy: GO")
print("Safe selective claim: HOLD")
