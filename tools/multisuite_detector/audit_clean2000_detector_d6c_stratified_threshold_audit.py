#!/usr/bin/env python3
"""D6C: Stratified threshold/gate robustness audit with suite sharding for parallel CPU execution.

Usage per suite (4 parallel processes):
  python3 d6c_...py --suite libero_10 --suite-shard-key 0 --output-root /path/libero_10
  python3 d6c_...py --suite libero_spatial --suite-shard-key 1 --output-root /path/libero_spatial
  ...
Then merge with --merge-only.

CPU-only. No GPU. No LIBERO. No OpenVLA. No training.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pathlib, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))

from gripper_attack.sc5_multisuite_detector_runtime import (
    SC5MultiSuiteDetectorRuntime, SC5_V2_FEATURES, validate_no_forbidden_inputs,
)
import probe_clean2000_detector_25d_feature_extraction_v3 as v3

THRESHOLD_GRID = {
    "tau_corridor": [0.20, 0.30, 0.40],
    "tau_primary": [0.40, 0.50, 0.60],
}
TAU_RELEASE = 0.30
GUARD = 5
STRIDE = 10
MAX_STEPS = 120
MAX_GROUPS_PER_SUITE = 25

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def classify_group(rows):
    has_primary = any(r.get("teacher_label_status") == "VALID_PRIMARY" for r in rows)
    return "has_primary" if has_primary else "no_event_or_unsupported"


def group_key(row):
    for k in ["group_key", "record_id"]:
        if row.get(k): return str(row[k])
    return ""


def extract_stream_features(trows, idx, allow_impute):
    """Extract 25D features using v3 compute_features, with imputation fallback."""
    values, methods, fields = v3.compute_features(trows, idx, "NO_EVENT")
    missing = []
    imputed = []
    residual = {"eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
                 "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5"}
    for feat in SC5_V2_FEATURES:
        val = values.get(feat, None)
        if val is not None and math.isfinite(val):
            continue
        if allow_impute and feat in residual:
            values[feat] = 0.0
            imputed.append(feat)
        else:
            missing.append(feat)
    return values, missing, imputed


def run_config(dataset_rows, checkpoint_path, suite, tc, tp, out):
    detector = SC5MultiSuiteDetectorRuntime(
        str(checkpoint_path), tau_corridor=tc, tau_release=TAU_RELEASE,
        tau_primary=tp, guard=GUARD, require_primary_event_role=True,
    )
    groups = defaultdict(list)
    for r in dataset_rows:
        if r.get("suite") != suite: continue
        groups[group_key(r)].append(r)

    selected = sorted(groups.items())
    # Take up to MAX_GROUPS, balancing primary/no-primary
    primary_groups = [(k, v) for k, v in selected if classify_group(v) == "has_primary"]
    no_primary_groups = [(k, v) for k, v in selected if classify_group(v) != "has_primary"]
    n_each = min(MAX_GROUPS_PER_SUITE // 2, len(primary_groups), len(no_primary_groups)) if no_primary_groups else MAX_GROUPS_PER_SUITE
    if not no_primary_groups:
        selected = primary_groups[:MAX_GROUPS_PER_SUITE]
    else:
        selected = primary_groups[:n_each] + no_primary_groups[:n_each]
        selected = selected[:MAX_GROUPS_PER_SUITE]

    group_replay = []
    row_sample = []
    temporal_cache = {}
    stream_attempted = stream_used = stream_missing = stream_imputed = 0
    temporal_failures = 0

    for g, grows in selected:
        tpath = next((str(r.get("temporal_path", "")) for r in grows if r.get("temporal_path")), "")
        detector.reset()
        emitted = False
        first_emit_step = ""
        max_cp, max_pp, min_rp = -1.0, -1.0, 1e9
        phase_ct, role_ct = Counter(), Counter()
        local_used = local_missing = local_imputed = 0
        last_state = "IDLE"

        if not tpath:
            group_replay.append({"group_key": g, "suite": suite, "group_class": classify_group(grows),
                "stream_rows_used": 0, "emitted": 0, "last_state": "NO_TEMPORAL_PATH"})
            continue

        try:
            if tpath not in temporal_cache:
                temporal_cache[tpath] = v3.read_temporal(tpath)
            trows = temporal_cache[tpath]
        except Exception:
            temporal_failures += 1
            continue

        max_n = min(len(trows), MAX_STEPS)
        for idx in range(0, max_n, STRIDE):
            stream_attempted += 1
            vals, missing, imputed = extract_stream_features(trows, idx, True)
            if missing:
                stream_missing += 1; local_missing += 1; continue
            stream_used += 1; local_used += 1
            local_imputed += len(imputed); stream_imputed += len(imputed)

            d = detector.update(vals, step=idx)
            cp_v = d.get("corridor_p"); pp_v = d.get("primary_p"); rp_v = d.get("release_p")
            if cp_v is not None: max_cp = max(max_cp, float(cp_v))
            if pp_v is not None: max_pp = max(max_pp, float(pp_v))
            if rp_v is not None: min_rp = min(min_rp, float(rp_v))
            phase_ct[str(d.get("pred_phase", ""))] += 1
            role_ct[str(d.get("pred_event_role", ""))] += 1
            if bool(d.get("emitted")) and not emitted:
                emitted = True; first_emit_step = str(d.get("emit_step", idx))
            last_state = str(d.get("state", ""))
            if len(row_sample) < 200:
                row_sample.append({"group_key": g, "suite": suite, "row_index": idx,
                    "runtime_state": last_state, "emitted": int(emitted),
                    "corridor_p": cp_v, "release_p": rp_v, "primary_p": pp_v,
                    "pred_phase": str(d.get("pred_phase", "")),
                    "pred_event_role": str(d.get("pred_event_role", ""))})

        gclass = classify_group(grows)
        group_replay.append({"group_key": g, "suite": suite, "group_class": gclass,
            "stream_rows_used": local_used, "stream_rows_missing": local_missing,
            "stream_imputation_count": local_imputed,
            "has_primary_truth": int(gclass == "has_primary"),
            "no_primary_truth": int(gclass != "has_primary"),
            "emitted": int(emitted), "first_emit_step": first_emit_step,
            "last_state": last_state,
            "max_corridor_p": "" if max_cp < 0 else f"{max_cp:.4f}",
            "max_primary_p": "" if max_pp < 0 else f"{max_pp:.4f}",
            "min_release_p": "" if min_rp > 1e8 else f"{min_rp:.4f}",
            "pred_phase_counts": json.dumps(dict(phase_ct)),
            "pred_event_role_counts": json.dumps(dict(role_ct))})

    # Metrics
    pg = [r for r in group_replay if r["has_primary_truth"]]
    npg = [r for r in group_replay if r["no_primary_truth"]]
    tp = sum(1 for r in pg if r["emitted"])
    fn = len(pg) - tp
    fp = sum(1 for r in npg if r["emitted"])
    tn = len(npg) - fp
    pr = tp / len(pg) if pg else 0.0
    npr = fp / len(npg) if npg else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * pr / (prec + pr) if (prec + pr) > 0 else 0.0

    config_result = {
        "tau_corridor": tc, "tau_primary": tp, "tau_release": TAU_RELEASE, "guard": GUARD,
        "suite": suite, "n_groups": len(group_replay),
        "primary_groups": len(pg), "no_primary_groups": len(npg),
        "tp_emit": tp, "fn_no_emit": fn, "fp_emit": fp, "tn_no_emit": tn,
        "primary_recall": pr, "no_primary_emit_rate": npr,
        "emit_precision": prec, "emit_f1": f1,
        "stream_rows_used": stream_used, "stream_rows_missing": stream_missing,
        "stream_imputed": stream_imputed, "temporal_failures": temporal_failures,
    }

    # Write per-config outputs
    cfg_dir = out / f"tc{tc:.2f}_tp{tp:.2f}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cfg_dir / "group_replay.csv", group_replay,
              ["group_key","suite","group_class","stream_rows_used","emitted","first_emit_step","last_state","max_corridor_p","max_primary_p","min_release_p","has_primary_truth","no_primary_truth"])
    write_json(cfg_dir / "metrics.json", config_result)

    return config_result, group_replay, row_sample


def main():
    parser = argparse.ArgumentParser(description="D6C: Stratified threshold audit")
    parser.add_argument("--frozen-dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", choices=SUITES, help="Single suite to process")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-rows", type=int, default=3717)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--merge-dirs", nargs="*")
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    if args.merge_only and args.merge_dirs:
        merge_results([Path(d) for d in args.merge_dirs], out)
        return 0

    dataset_rows = read_csv(Path(args.frozen_dataset))
    assert len(dataset_rows) == args.expected_rows
    checkpoint_path = Path(args.checkpoint)
    suite = args.suite

    all_results = []
    all_group_replay = []
    all_row_sample = []

    for tc in THRESHOLD_GRID["tau_corridor"]:
        for tp in THRESHOLD_GRID["tau_primary"]:
            print(f"  [{suite}] tc={tc:.2f} tp={tp:.2f} ...", end=" ", flush=True)
            t0 = time.time()
            cr, gr, rs = run_config(dataset_rows, checkpoint_path, suite, tc, tp, out)
            dt = time.time() - t0
            print(f"done ({dt:.0f}s) recall={cr['primary_recall']:.2f} emit_rate={cr['no_primary_emit_rate']:.2f}")
            all_results.append(cr)
            all_group_replay.extend(gr)
            all_row_sample.extend(rs)

    # Write suite-level summary
    write_csv(out / "d6c_threshold_grid_summary.csv", all_results,
              ["tau_corridor","tau_primary","tau_release","guard","suite","n_groups",
               "primary_groups","no_primary_groups","tp_emit","fn_no_emit","fp_emit","tn_no_emit",
               "primary_recall","no_primary_emit_rate","emit_precision","emit_f1",
               "stream_rows_used","stream_rows_missing","stream_imputed"])
    write_csv(out / "d6c_group_replay.csv", all_group_replay,
              ["group_key","suite","group_class","stream_rows_used","emitted","first_emit_step","last_state"])
    write_json(out / "d6c_threshold_grid_report.json", {
        "suite": suite, "configs": len(all_results),
        "results": all_results,
        "best_by_recall": max(all_results, key=lambda r: (r["primary_recall"], -r["no_primary_emit_rate"])),
        "best_by_f1": max(all_results, key=lambda r: (r["emit_f1"], -r["no_primary_emit_rate"])),
    })

    # SHA256SUMS
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file(): csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)
    with open(out / "SHA256SUMS", "w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS", "SHA256SUMS.sha256"): f.write(f"{sha}  {fn}\n")
    ss = sha256_file(str(out / "SHA256SUMS"))
    with open(out / "SHA256SUMS.sha256", "w") as f: f.write(f"{ss}  SHA256SUMS\n")

    print(f"D6C [{suite}] DONE: {len(all_results)} configs")
    return 0


def merge_results(dirs, out):
    all_results = []
    for d in dirs:
        rp = d / "d6c_threshold_grid_report.json"
        if rp.exists():
            all_results.extend(json.loads(open(rp))["results"])
    write_csv(out / "d6c_merged_threshold_grid_summary.csv", all_results,
              ["tau_corridor","tau_primary","tau_release","guard","suite","n_groups",
               "primary_groups","no_primary_groups","tp_emit","fn_no_emit","fp_emit","tn_no_emit",
               "primary_recall","no_primary_emit_rate","emit_precision","emit_f1"])
    write_json(out / "d6c_merged_report.json", {
        "total_configs": len(all_results),
        "best_by_suite": {},
        "all_results": all_results,
    })
    print(f"D6C merge DONE: {len(all_results)} configs from {len(dirs)} dirs")


if __name__ == "__main__":
    raise SystemExit(main())
