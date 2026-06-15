#!/usr/bin/env python3
"""E4C.2: Deterministic RC1a remap + Teacher-P eligibility construction.

Runs DIRECTLY on the GPU server (vla) with local file access.
No SSH round-trips. GPU acceleration authorized as implementation detail
only — all algorithms, thresholds, and abstain rules are frozen.

Processes each trace through:
  1. Provenance gate (file SHA == manifest SHA)
  2. Full row-level field validity (ALL rows, NOT 50-row sample)
  3. Open-convention invariant (decoded_open_bool iff env < -0.5)
  4. RC1a remap (rc1a_corrected_v2_e1_5)
  5. Teacher-P grasp-privilege evaluation
  6. CLOSE candidate enumeration

Usage (on vla):
  /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python \
    scripts/stageb/run_l12_e4c2_eligibility.py \
    --manifest /data/liuyu/l12_e4c2_pipeline/l12_e4c2_input_manifest.csv \
    --output-dir /data/liuyu/l12_e4c2_output \
    [--parity 10] [--start N --end M]
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure pipeline source files are importable ──
PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.phase_detector import (
    teacher_privileged_critical_close_anchor,
    teacher_rule_critical_close_anchor,
    check_teacher_p_privilege_capability,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    PREDICTION_HORIZON,
)

# ── Frozen constants ──
CONFIG_SHA = "2BFDCC4222298D6E803A3287A39E5F422DDCCCC8969FE830166E872452598F80"
REQUIRED_HEADER = [
    "obj_x", "obj_y", "obj_z",
    "eef_x", "eef_y", "eef_z",
    "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before",
]

# ── Gate 1: Provenance ──
def check_provenance(fp: str, expected_sha: str, expected_rows: int) -> dict:
    result = {"provenance_pass": False, "file_exists": False, "sha_match": False,
              "row_count_match": False, "current_sha": "", "current_row_count": -1,
              "failure_reason": ""}
    if not os.path.isfile(fp):
        result["failure_reason"] = "file_missing"; return result
    result["file_exists"] = True

    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    result["current_sha"] = h.hexdigest()
    if result["current_sha"] != expected_sha:
        result["failure_reason"] = "sha_mismatch"; return result
    result["sha_match"] = True

    with open(fp, "r") as f:
        n = sum(1 for _ in f) - 1  # minus header
    result["current_row_count"] = n
    if n != expected_rows:
        result["failure_reason"] = "row_count_mismatch"; return result
    result["row_count_match"] = True
    result["provenance_pass"] = True
    return result


# ── Gate 2: Full row-level field validity ──
def check_field_validity(fp: str) -> dict:
    result = {"field_validity_pass": True, "n_rows_checked": 0,
              "fields_checked": {}, "first_invalid_row": -1}
    try:
        with open(fp, "r") as f:
            header_line = f.readline()
            lines = [l for l in f if l.strip()]
    except Exception:
        result["field_validity_pass"] = False; return result

    if not header_line:
        result["field_validity_pass"] = False; return result

    header_fields = header_line.strip().split(",")
    idx = {}
    for fld in REQUIRED_HEADER:
        try: idx[fld] = header_fields.index(fld)
        except ValueError:
            result["field_validity_pass"] = False; return result

    for fld in REQUIRED_HEADER:
        result["fields_checked"][fld] = {"missing": 0, "parse_fail": 0, "non_finite": 0}

    result["n_rows_checked"] = len(lines)
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < len(header_fields):
            result["field_validity_pass"] = False
            if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
            continue
        for fld in REQUIRED_HEADER:
            j = idx[fld]
            if j >= len(parts):
                result["fields_checked"][fld]["missing"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
                continue
            val = parts[j].strip()
            if val == "":
                result["fields_checked"][fld]["missing"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
                continue
            try:
                fv = float(val)
                if math.isnan(fv) or math.isinf(fv):
                    result["fields_checked"][fld]["non_finite"] += 1
                    if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
            except ValueError:
                result["fields_checked"][fld]["parse_fail"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1

    for fld in REQUIRED_HEADER:
        s = result["fields_checked"][fld]
        if s["missing"] > 0 or s["parse_fail"] > 0 or s["non_finite"] > 0:
            result["field_validity_pass"] = False
    return result


# ── Gate 3: Open convention invariant ──
def check_open_convention(fp: str) -> dict:
    result = {"open_convention_pass": True, "n_violations": 0, "first_violation_row": -1}
    try:
        with open(fp, "r") as f:
            header_line = f.readline()
            lines = [l for l in f if l.strip()]
    except Exception:
        result["open_convention_pass"] = False; return result

    if not header_line:
        result["open_convention_pass"] = False; return result

    hdr = header_line.strip().split(",")
    try: env_i, dec_i = hdr.index("clean_gripper_env"), hdr.index("decoded_open_bool")
    except ValueError:
        result["open_convention_pass"] = False; return result

    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < max(env_i, dec_i) + 1: continue
        try:
            env_v = float(parts[env_i].strip())
            dec_v = int(float(parts[dec_i].strip()))
        except (ValueError, IndexError):
            result["n_violations"] += 1
            if result["first_violation_row"] < 0: result["first_violation_row"] = i + 1
            continue
        if abs(env_v) <= 0.5: continue  # neutral
        expected = 1 if env_v < -0.5 else 0
        if dec_v != expected:
            result["n_violations"] += 1
            if result["first_violation_row"] < 0: result["first_violation_row"] = i + 1

    if result["n_violations"] > 0: result["open_convention_pass"] = False
    return result


# ── Gate 4+5: RC1a remap + Teacher-P ──
def run_rc1a_and_teacher_p(fp: str) -> dict:
    """Run remap and Teacher-P on a single trace. Returns combined result."""
    result = {
        "remap_pass": False, "remapper_version": REMAPPER_VERSION,
        "valid_row_count": -1, "invariant_violations": 0, "remap_abstain": "",
        "teacher_p_available": False, "teacher_p_step": -1,
        "teacher_p_abstain_reason": "", "grasp_privilege_valid": False,
        "placement_privilege_valid": False, "n_close_candidates": -1,
        "teacher_r_step": -1,
    }
    rows, inv, fi = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
    result["valid_row_count"] = len(rows) if rows else 0
    result["invariant_violations"] = len(inv) if inv else 0

    if not rows or result["valid_row_count"] == 0:
        result["remap_abstain"] = "remap_zero_rows"; return result
    if result["invariant_violations"] > 0:
        result["remap_abstain"] = f"invariant_violations_{result['invariant_violations']}"; return result
    result["remap_pass"] = True

    cap = check_teacher_p_privilege_capability(rows)
    result["grasp_privilege_valid"] = cap["grasp_privilege_valid"]
    result["placement_privilege_valid"] = cap["placement_privilege_valid"]

    p_anchor = teacher_privileged_critical_close_anchor(rows)
    r_anchor = teacher_rule_critical_close_anchor(rows)
    result["teacher_r_step"] = r_anchor

    preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                        teacher_anchor=p_anchor if p_anchor >= 0 else -1)
    close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
    result["n_close_candidates"] = len(close_cands)

    if p_anchor >= 0:
        result["teacher_p_available"] = True
        result["teacher_p_step"] = p_anchor
    elif not result["grasp_privilege_valid"]:
        result["teacher_p_abstain_reason"] = "grasp_privilege_invalid"
    else:
        result["teacher_p_abstain_reason"] = "no_close_satisfies_teacher_p"

    return result


def get_candidate_details(fp: str, p_anchor: int) -> list[dict]:
    """Enumerate all CLOSE candidates with full feature details."""
    rows, _, _ = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
    preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                        teacher_anchor=p_anchor)
    close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
    result = []
    open_steps = [p["step"] for p in preds if p.get("decoded_open_bool")]
    for idx, c in enumerate(close_cands):
        step = c["step"]
        prev_close = close_cands[idx - 1]["step"] if idx > 0 else None
        last_open = max([s for s in open_steps if s < step]) if [s for s in open_steps if s < step] else None
        result.append({
            "candidate_step": step, "candidate_index": idx,
            "is_teacher_p": int(step == p_anchor),
            "distance_to_p": step - p_anchor,
            "total_score": round(c["score"], 4),
            "raw_crossing": int(c.get("raw_open_to_close_crossing", 0)),
            "close_onset": int(c.get("close_onset", 0)),
            "close_streak": int(c.get("close_streak_value", 0)),
            "qpos": round(c.get("qpos", 0), 6),
            "eef_speed_now": round(c.get("eef_speed_now", 0) or 0, 6),
            "eef_speed_prev": round(c.get("eef_speed_prev", 0) or 0, 6),
            "eef_deceleration_delta": round((c.get("eef_speed_now", 0) or 0) - (c.get("eef_speed_prev", 0) or 0), 6),
            "time_since_prev_close": step - prev_close if prev_close is not None else "",
            "time_since_last_open": step - last_open if last_open is not None else "",
        })
    return result


def classify_trace(gates: dict) -> str:
    if not gates["prov"]["provenance_pass"]:       return "PROVENANCE_FAIL"
    if not gates["fv"]["field_validity_pass"]:     return "FIELD_VALIDITY_FAIL"
    if not gates["oc"]["open_convention_pass"]:    return "OPEN_CONVENTION_FAIL"
    if not gates["tp"]["remap_pass"]:              return "RC1A_REMAP_FAIL"
    tp = gates["tp"]
    if not tp["teacher_p_available"]:
        return "TEACHER_P_UNAVAILABLE"
    nc = tp["n_close_candidates"]
    if nc == 0: return "NO_CLOSE_CANDIDATE"
    if nc == 1: return "ELIGIBLE_SINGLE_CANDIDATE"
    return "ELIGIBLE_MULTI_CANDIDATE"


def get_gpu_info() -> dict:
    """Record CUDA environment for provenance."""
    info = {"cuda_available": False, "gpu_count": 0, "devices": [], "env_cuda_visible": ""}
    info["env_cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
            text=True, timeout=10
        ).strip()
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["devices"].append({"index": parts[0], "name": parts[1]})
        info["gpu_count"] = len(info["devices"])
    except Exception:
        pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["torch_version"] = torch.__version__
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--parity", type=int, default=0,
                    help="Process only N traces for parity validation")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # Record environment
    gpu_info = get_gpu_info()
    hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    total = len(manifest_rows)
    if args.parity > 0:
        end_idx = args.start + args.parity
        batch = manifest_rows[args.start:end_idx]
        print(f"E4C.2 PARITY MODE: {len(batch)} traces")
    elif args.end > 0:
        end_idx = args.end
        batch = manifest_rows[args.start:end_idx]
    else:
        end_idx = total
        batch = manifest_rows[args.start:end_idx]
    print(f"E4C.2: {len(batch)}/{total} traces [{args.start}:{end_idx}]")
    print(f"Host: {hostname}  CUDA: {gpu_info['cuda_available']}  GPUs: {gpu_info['gpu_count']}")
    print(f"Remapper: {REMAPPER_VERSION}")

    trace_status = []
    trace_candidates = []
    failures = Counter()
    task_counts = defaultdict(lambda: Counter())
    t0 = time.time()

    for i, mr in enumerate(batch):
        tid = mr["trace_id"]; task = mr["task_key"]; state = mr["state_id"]
        fp = mr["source_path"]

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{args.start + i + 1}/{end_idx}] {task}_s{state}  ({rate:.1f} traces/s)")

        gates = {}

        # Gate 1
        gates["prov"] = check_provenance(fp, mr["source_sha256"], int(mr["row_count"]))
        if not gates["prov"]["provenance_pass"]:
            failures["PROVENANCE_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": False, "field_validity_pass": "", "open_convention_pass": "",
                "remap_pass": "", "grasp_privilege_valid": "", "teacher_p_available": "",
                "teacher_p_step": -1, "teacher_r_step": -1, "n_close_candidates": -1,
                "category": "PROVENANCE_FAIL",
                "failure_detail": gates["prov"]["failure_reason"],
            }); continue

        # Gate 2
        gates["fv"] = check_field_validity(fp)
        if not gates["fv"]["field_validity_pass"]:
            failures["FIELD_VALIDITY_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": False,
                "open_convention_pass": "", "remap_pass": "", "grasp_privilege_valid": "",
                "teacher_p_available": "", "teacher_p_step": -1, "teacher_r_step": -1,
                "n_close_candidates": -1, "category": "FIELD_VALIDITY_FAIL",
                "failure_detail": f"first_invalid_row={gates['fv']['first_invalid_row']}",
            }); continue

        # Gate 3
        gates["oc"] = check_open_convention(fp)
        if not gates["oc"]["open_convention_pass"]:
            failures["OPEN_CONVENTION_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": True,
                "open_convention_pass": False, "remap_pass": "", "grasp_privilege_valid": "",
                "teacher_p_available": "", "teacher_p_step": -1, "teacher_r_step": -1,
                "n_close_candidates": -1, "category": "OPEN_CONVENTION_FAIL",
                "failure_detail": f"n_violations={gates['oc']['n_violations']}",
            }); continue

        # Gates 4+5
        gates["tp"] = run_rc1a_and_teacher_p(fp)
        if not gates["tp"]["remap_pass"]:
            failures["RC1A_REMAP_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": True,
                "open_convention_pass": True, "remap_pass": False,
                "grasp_privilege_valid": "", "teacher_p_available": "",
                "teacher_p_step": -1, "teacher_r_step": gates["tp"]["teacher_r_step"],
                "n_close_candidates": -1, "category": "RC1A_REMAP_FAIL",
                "failure_detail": gates["tp"]["remap_abstain"],
            }); continue

        # Classification
        category = classify_trace(gates)
        failures[category] += 1
        if category.startswith("ELIGIBLE"):
            task_counts[task]["eligible"] += 1
            if category == "ELIGIBLE_MULTI_CANDIDATE": task_counts[task]["multi"] += 1
            else: task_counts[task]["single"] += 1

        trace_status.append({
            "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
            "provenance_pass": True, "field_validity_pass": True,
            "open_convention_pass": True, "remap_pass": True,
            "grasp_privilege_valid": gates["tp"]["grasp_privilege_valid"],
            "teacher_p_available": gates["tp"]["teacher_p_available"],
            "teacher_p_step": gates["tp"]["teacher_p_step"],
            "teacher_r_step": gates["tp"]["teacher_r_step"],
            "n_close_candidates": gates["tp"]["n_close_candidates"],
            "category": category,
            "failure_detail": gates["tp"].get("teacher_p_abstain_reason", ""),
        })

        # Candidate details for eligible traces
        if gates["tp"]["teacher_p_available"] and gates["tp"]["n_close_candidates"] > 0:
            try:
                cands = get_candidate_details(fp, gates["tp"]["teacher_p_step"])
                for c in cands:
                    c["trace_id"] = tid; c["task_key"] = task; c["state_id"] = state
                    trace_candidates.append(c)
            except Exception:
                pass

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s  Rate: {len(batch)/elapsed:.2f} traces/s")

    # ── Write outputs ──
    status_fields = [
        "trace_id", "task_key", "state_id", "seed",
        "provenance_pass", "field_validity_pass", "open_convention_pass",
        "remap_pass", "grasp_privilege_valid", "teacher_p_available",
        "teacher_p_step", "teacher_r_step", "n_close_candidates",
        "category", "failure_detail",
    ]
    with open(out / "l12_e4c2_trace_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=status_fields); w.writeheader(); w.writerows(trace_status)

    if trace_candidates:
        cfields = list(trace_candidates[0].keys())
        with open(out / "l12_e4c2_close_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(trace_candidates)

    p_cov = []
    for r in trace_status:
        if r["provenance_pass"] and r["field_validity_pass"] and r["open_convention_pass"] and r["remap_pass"]:
            p_cov.append({
                "trace_id": r["trace_id"], "task_key": r["task_key"], "state_id": r["state_id"],
                "grasp_privilege_valid": r["grasp_privilege_valid"],
                "teacher_p_available": r["teacher_p_available"],
                "teacher_p_step": r["teacher_p_step"],
                "teacher_r_step": r["teacher_r_step"],
                "n_close_candidates": r["n_close_candidates"],
                "category": r["category"],
            })
    if p_cov:
        with open(out / "l12_e4c2_teacher_p_coverage.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(p_cov[0].keys())); w.writeheader(); w.writerows(p_cov)

    tasks_seen = set(r["task_key"] for r in trace_status)
    task_rows = []
    for t in sorted(tasks_seen):
        trs = [r for r in trace_status if r["task_key"] == t]
        task_rows.append({
            "task_key": t, "n_total": len(trs),
            "n_provenance_pass": sum(1 for r in trs if r["provenance_pass"]),
            "n_field_valid": sum(1 for r in trs if r["field_validity_pass"] == True),
            "n_open_convention_pass": sum(1 for r in trs if r["open_convention_pass"] == True),
            "n_remap_pass": sum(1 for r in trs if r["remap_pass"]),
            "n_grasp_privilege_valid": sum(1 for r in trs if r["grasp_privilege_valid"] == True),
            "n_teacher_p_available": sum(1 for r in trs if r["teacher_p_available"] == True),
            "n_multi_candidate": sum(1 for r in trs if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"),
            "n_single_candidate": sum(1 for r in trs if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE"),
            "n_no_candidate": sum(1 for r in trs if r["category"] == "NO_CLOSE_CANDIDATE"),
            "n_teacher_p_unavailable": sum(1 for r in trs if r["category"] == "TEACHER_P_UNAVAILABLE"),
        })
    with open(out / "l12_e4c2_task_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys())); w.writeheader(); w.writerows(task_rows)

    tax_rows = [{"category": cat, "count": count} for cat, count in failures.most_common()]
    with open(out / "l12_e4c2_failure_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "count"]); w.writeheader(); w.writerows(tax_rows)

    hash_rows = []
    for fname in sorted(os.listdir(str(out))):
        fpath = out / fname
        if fpath.suffix == ".csv":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
            hash_rows.append({"file": fname, "sha256": h.hexdigest()})
    with open(out / "l12_e4c2_output_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256"]); w.writeheader(); w.writerows(hash_rows)

    end_time = datetime.now(timezone.utc)
    with open(out / "l12_e4c2_run_log.txt", "w") as f:
        f.write(f"E4C.2 RUN LOG\n")
        f.write(f"hostname: {hostname}\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"config_sha256: {CONFIG_SHA}\n")
        f.write(f"input_manifest: {args.manifest}\n")
        f.write(f"n_input: {total}\nn_processed: {len(batch)}\n")
        f.write(f"n_trace_status_rows: {len(trace_status)}\n")
        f.write(f"n_candidate_rows: {len(trace_candidates)}\n")
        f.write(f"cuda_available: {gpu_info['cuda_available']}\n")
        f.write(f"gpu_count: {gpu_info['gpu_count']}\n")
        f.write(f"env_cuda_visible_devices: {gpu_info['env_cuda_visible']}\n")
        for dev in gpu_info.get("devices", []):
            f.write(f"gpu_{dev['index']}: {dev['name']}\n")
        f.write(f"parity_mode: {args.parity > 0}\n")
        for cat, count in failures.most_common():
            f.write(f"category_{cat}: {count}\n")

    print(f"\n=== E4C.2 SUMMARY ===")
    print(f"Traces processed: {len(trace_status)}")
    for cat, count in failures.most_common():
        print(f"  {cat}: {count}")
    n_tp = sum(1 for r in trace_status if r["teacher_p_available"] == True)
    n_multi = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE")
    n_single = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE")
    print(f"\nTeacher-P available: {n_tp}")
    print(f"Multi-candidate: {n_multi}")
    print(f"Single-candidate: {n_single}")
    print(f"Output: {out}")
    print("E4C.2 COMPLETE")
    print("TRAINING_STARTED: NO")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
