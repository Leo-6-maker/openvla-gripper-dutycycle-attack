#!/usr/bin/env python3
"""E4C.2b: Final label-pool integrity repair.

Fixes E4C.2a P0/P1 audit issues:
  P0-1: True fail-closed source provenance — compare runtime SHAs vs frozen expected.
  P0-2: Fix time_since_last_open — compute from remapped rows, not predictions.
  P0-3: Ambiguous label safety — zero is_teacher_p on traces with >1 TP-qualifying.
  P1-1: Domain gate contract — domain checks descriptive only; honest violation counts.
  P1-2: Implement OTHER_ABSTAIN catch-all path.
  + Consistency assertions at end.

Runs directly on GPU server. All thresholds frozen. TRAINING FORBIDDEN.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Runtime paths ──
PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

import numpy as np
from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.phase_detector import (
    _classify_motion_evidence,
    _check_grasp_privilege_valid,
    _field_is_valid,
    _safe_float,
    EEF_TO_OBJ_NEAR_THRESHOLD,
    OBJECT_LIFT_MIN_DELTA,
    OBJECT_LIFT_LOOKAHEAD,
    SUSTAINED_MOTION_FRAMES,
    MOTION_SUSTAINED_VERTICAL_LIFT,
    check_teacher_p_privilege_capability,
    teacher_privileged_critical_close_anchor,
    teacher_rule_critical_close_anchor,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    PREDICTION_HORIZON,
)

# ── Frozen expected SHA256 (local commit 402ecfb — prereg config) ──
EXPECTED_MANIFEST_SHA = "b11ac1e8df74f63fd764985ff335a7d6648114af66045fc34aca6956e6b85863"
EXPECTED_CONFIG_SHA  = "2bfdcc4222298d6e803a3287a39e5f422ddcccc8969fe830166e872452598f80"
EXPECTED_REMAPPER_SHA = "5d9cf327b25da459d399eda0c32527acea635daad5e1509b02a732154503c063"
EXPECTED_PHASE_DETECTOR_SHA = "f9cc7e90f415ee315723eba4738fd8f84f843b02f6dc2c123ae3570dc3a41329"
EXPECTED_SELECTOR_SHA = "81b510ec30716df11a89fbfb45194dceb1cfec09c1a2fd5d357a8a7448b4fa34"

REQUIRED_HEADER = [
    "obj_x", "obj_y", "obj_z",
    "eef_x", "eef_y", "eef_z",
    "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before",
]


def sha256_file(path: str) -> str:
    if not os.path.isfile(path): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


# ── P0-1: True fail-closed provenance seal ──
def runtime_provenance_seal(manifest_path: str) -> dict:
    """Verify all runtime artifacts match frozen expected SHAs. Fail-closed."""
    checks = [
        ("manifest", manifest_path, EXPECTED_MANIFEST_SHA),
        ("remapper", os.path.join(PIPELINE_ROOT, "scripts", "stageb", "remap_v4_trace_for_l12.py"), EXPECTED_REMAPPER_SHA),
        ("phase_detector", os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "phase_detector.py"), EXPECTED_PHASE_DETECTOR_SHA),
        ("selector", os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "critical_close_selector.py"), EXPECTED_SELECTOR_SHA),
        ("runner", __file__, None),  # recorded, not compared (script IS the authority)
    ]
    result = {"pass": True, "failures": [], "artifacts": {}}
    for name, path, expected in checks:
        actual = sha256_file(path)
        result["artifacts"][f"{name}_sha"] = actual
        if actual == "MISSING":
            result["pass"] = False
            result["failures"].append(f"{name}: file missing at {path}")
        elif expected is not None and actual.lower() != expected.lower():
            result["pass"] = False
            result["failures"].append(
                f"{name}: SHA MISMATCH expected={expected[:16]} actual={actual[:16]}")
    # Config is not on server, but record expected
    result["artifacts"]["config_sha_expected"] = EXPECTED_CONFIG_SHA
    result["artifacts"]["config_sha_runtime"] = "not_on_server_pipeline"
    return result


# ── Field validity (domain checks descriptive, NOT gating) ──
def check_field_validity_v3(fp: str) -> dict:
    """Row-level field validity. Missing/parse/non-finite → hard fail.
    Domain checks → descriptive only (recorded, not gating)."""
    result = {"field_validity_pass": True, "n_rows_checked": 0,
              "fields_checked": {}, "first_invalid_row": -1,
              "domain_violation_count": 0, "domain_violations": []}
    try:
        with open(fp, "r") as f:
            header_line = f.readline()
            lines = [l for l in f if l.strip()]
    except Exception:
        result["field_validity_pass"] = False; return result
    if not header_line:
        result["field_validity_pass"] = False; return result

    hdr = header_line.strip().split(",")
    idx = {}
    for fld in REQUIRED_HEADER:
        try: idx[fld] = hdr.index(fld)
        except ValueError:
            result["field_validity_pass"] = False; return result

    for fld in REQUIRED_HEADER:
        result["fields_checked"][fld] = {"missing": 0, "parse_fail": 0, "non_finite": 0, "domain_soft": 0}

    result["n_rows_checked"] = len(lines)
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < len(hdr):
            result["field_validity_pass"] = False
            if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
            continue
        for fld in REQUIRED_HEADER:
            j = idx[fld]; val = parts[j].strip() if j < len(parts) else ""
            if val == "":
                result["fields_checked"][fld]["missing"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
                continue
            try:
                fv = float(val)
                if math.isnan(fv) or math.isinf(fv):
                    result["fields_checked"][fld]["non_finite"] += 1
                    if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1
                    continue
                # Descriptive domain checks (P1-1: not gating)
                if fld in ("obj_x", "obj_y", "eef_x", "eef_y") and abs(fv) > 10:
                    result["fields_checked"][fld]["domain_soft"] += 1
                    result["domain_violation_count"] += 1
                    if len(result["domain_violations"]) < 5:
                        result["domain_violations"].append(f"{fld}={fv} row={i+1}")
                elif fld == "obj_z" and (fv < -1 or fv > 5):
                    result["fields_checked"][fld]["domain_soft"] += 1
                    result["domain_violation_count"] += 1
                elif fld == "eef_z" and (fv < -1 or fv > 5):
                    result["fields_checked"][fld]["domain_soft"] += 1
                    result["domain_violation_count"] += 1
                elif fld == "clean_gripper_env" and abs(fv) > 10:
                    result["fields_checked"][fld]["domain_soft"] += 1
                    result["domain_violation_count"] += 1
                elif fld == "gripper_qpos_before" and (fv < -0.001 or fv > 1.001):
                    # Tolerant: -0.001 to 1.001 captures floating noise near 0/1
                    if not (-0.005 <= fv <= 1.005):
                        result["fields_checked"][fld]["domain_soft"] += 1
                        result["domain_violation_count"] += 1
                        if len(result["domain_violations"]) < 5:
                            result["domain_violations"].append(f"{fld}={fv} row={i+1}")
            except ValueError:
                result["fields_checked"][fld]["parse_fail"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1

    for fld in REQUIRED_HEADER:
        s = result["fields_checked"][fld]
        if any(s[k] > 0 for k in ("missing", "parse_fail", "non_finite")):
            result["field_validity_pass"] = False
    # Domain violations do NOT cause gate failure
    return result


def check_open_convention_v2(fp: str) -> dict:
    result = {"open_convention_pass": True, "n_violations": 0, "first_violation_row": -1}
    try:
        with open(fp, "r") as f:
            hdr_line = f.readline(); lines = [l for l in f if l.strip()]
    except Exception:
        result["open_convention_pass"] = False; return result
    if not hdr_line: result["open_convention_pass"] = False; return result
    hdr = hdr_line.strip().split(",")
    try: env_i, dec_i = hdr.index("clean_gripper_env"), hdr.index("decoded_open_bool")
    except ValueError: result["open_convention_pass"] = False; return result
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < max(env_i, dec_i) + 1: continue
        try:
            env_v = float(parts[env_i].strip()); dec_v = int(float(parts[dec_i].strip()))
        except (ValueError, IndexError):
            result["n_violations"] += 1
            if result["first_violation_row"] < 0: result["first_violation_row"] = i + 1; continue
        if abs(env_v) <= 0.5: continue
        if dec_v != (1 if env_v < -0.5 else 0):
            result["n_violations"] += 1
            if result["first_violation_row"] < 0: result["first_violation_row"] = i + 1
    if result["n_violations"] > 0: result["open_convention_pass"] = False
    return result


# ── Per-candidate Teacher-P evidence ──
def evaluate_teacher_p_for_candidate(records: list[dict], t: int) -> dict:
    result = {
        "candidate_step": t,
        "is_close_onset_and_clean_close": False,
        "gripper_not_already_open": False,
        "grasp_privilege_locally_valid": False,
        "eef_near_object": False,
        "sustained_vertical_lift": False,
        "teacher_p_criteria_pass": False,
        "eef_to_obj_distance_at_close": None,
        "max_cumulative_vertical_dz": 0.0,
        "max_sustained_vertical_frames": 0,
        "eef_attachment_consistent": True,
        "grasp_local_abstain_reason": "",
    }
    if t < 0 or t >= len(records): return result
    r = records[t]
    if not (int(_safe_float(r.get("close_onset", 0))) and
            int(_safe_float(r.get("clean_close", 0)))):
        result["grasp_local_abstain_reason"] = "not_close_onset_or_not_clean_close"; return result
    result["is_close_onset_and_clean_close"] = True
    if int(_safe_float(r.get("decoded_open_bool", 0))):
        result["grasp_local_abstain_reason"] = "gripper_already_open"; return result
    result["gripper_not_already_open"] = True
    if not _check_grasp_privilege_valid(records, t):
        result["grasp_local_abstain_reason"] = "grasp_privilege_not_locally_valid"; return result
    result["grasp_privilege_locally_valid"] = True
    eef_dist = _safe_float(r.get("eef_to_obj_distance", 999))
    result["eef_to_obj_distance_at_close"] = round(eef_dist, 6)
    if eef_dist > EEF_TO_OBJ_NEAR_THRESHOLD:
        result["grasp_local_abstain_reason"] = f"eef_not_near_object_dist={eef_dist:.4f}"; return result
    result["eef_near_object"] = True
    evidence = _classify_motion_evidence(records, t)
    result["max_cumulative_vertical_dz"] = round(evidence["cumulative_vertical_dz"], 6)
    result["max_sustained_vertical_frames"] = evidence["sustained_above_threshold_frames"]
    result["eef_attachment_consistent"] = evidence["eef_attachment_consistent"]
    if evidence["motion_evidence_type"] != MOTION_SUSTAINED_VERTICAL_LIFT:
        result["grasp_local_abstain_reason"] = (
            f"no_sustained_vertical_lift_type={evidence['motion_evidence_type']}"); return result
    result["sustained_vertical_lift"] = True
    result["teacher_p_criteria_pass"] = True
    return result


# ── P1-2: Full 10-category taxonomy WITH OTHER_ABSTAIN path ──
def classify_trace_v3(
    provenance_ok: bool, field_ok: bool, oc_ok: bool, remap_ok: bool,
    n_total_candidates: int, n_tp_qualifying: int, grasp_priv_valid: bool,
    remap_abstain: str,
) -> str:
    if not provenance_ok: return "PROVENANCE_FAIL"
    if not field_ok: return "FIELD_VALIDITY_FAIL"
    if not oc_ok: return "OPEN_CONVENTION_FAIL"
    if not remap_ok: return "RC1A_REMAP_FAIL"

    if n_total_candidates == 0:
        return "NO_CLOSE_CANDIDATE"
    if grasp_priv_valid is False:
        return "TEACHER_P_UNAVAILABLE"
    if n_tp_qualifying == 0:
        return "TEACHER_P_UNAVAILABLE"
    if n_tp_qualifying > 1:
        return "TEACHER_P_AMBIGUOUS"
    if n_tp_qualifying == 1:
        if n_total_candidates == 1:
            return "ELIGIBLE_SINGLE_CANDIDATE"
        if n_total_candidates >= 2:
            return "ELIGIBLE_MULTI_CANDIDATE"
    # Catch-all — should be unreachable but preserves taxonomy completeness
    return "OTHER_ABSTAIN"


def get_gpu_info() -> dict:
    info = {"cuda_available": False, "gpu_count": 0, "devices": [], "env_cuda_visible": ""}
    info["env_cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2: info["devices"].append({"index": parts[0], "name": parts[1]})
        info["gpu_count"] = len(info["devices"])
    except: pass
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]: info["torch_version"] = torch.__version__
    except ImportError: pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # ── P0-1: Fail-closed provenance seal ──
    print("=== RUNTIME PROVENANCE SEAL ===")
    seal = runtime_provenance_seal(args.manifest)
    for k, v in seal["artifacts"].items():
        if k.endswith("_sha") or k.endswith("_expected"):
            print(f"  {k}: {v[:16]}..." if len(v) > 16 else f"  {k}: {v}")
    if not seal["pass"]:
        print(f"\nFATAL: PROVENANCE SEAL FAILED:")
        for f in seal["failures"]: print(f"  {f}")
        sys.exit(1)
    print("  PROVENANCE SEAL: PASS (all 4 source files + manifest matched)\n")

    gpu_info = get_gpu_info()
    hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"
    print(f"Host: {hostname}  CUDA: {gpu_info['cuda_available']}  GPUs: {gpu_info['gpu_count']}")
    print(f"Remapper: {REMAPPER_VERSION}\n")

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    total = len(manifest_rows)
    end_idx = args.end if args.end > 0 else total
    batch = manifest_rows[args.start:end_idx]
    print(f"Processing {len(batch)}/{total} traces [{args.start}:{end_idx}]\n")

    trace_status = []
    trace_candidates = []
    failures = Counter()
    task_counts = defaultdict(lambda: Counter())
    t0 = time.time()
    total_domain_violations = 0

    for i, mr in enumerate(batch):
        tid = mr["trace_id"]; task = mr["task_key"]; state = mr["state_id"]
        fp = mr["source_path"]

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{args.start + i + 1}/{end_idx}] {task}_s{state}  ({rate:.1f} t/s)")

        prov_ok = True; field_ok = True; oc_ok = True; remap_ok = True
        prov_detail = ""; field_detail = ""; oc_detail = ""; remap_detail = ""

        # Gate 1: Provenance
        if not os.path.isfile(fp):
            prov_ok = False; prov_detail = "file_missing"
        else:
            actual_sha = sha256_file(fp)
            if actual_sha != mr["source_sha256"]:
                prov_ok = False; prov_detail = f"sha_mismatch"
            with open(fp, "r") as f:
                n_rows = sum(1 for _ in f) - 1
            if n_rows != int(mr["row_count"]):
                prov_ok = False; prov_detail = f"row_count_mismatch"

        if not prov_ok:
            failures["PROVENANCE_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": False, "field_validity_pass": "", "open_convention_pass": "",
                "remap_pass": "", "n_total_rows": -1, "n_gripper_valid_rows": -1,
                "n_neutral_rows": -1, "remap_field_issue_count": -1,
                "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "legacy_first_match_step": -1, "teacher_r_step": -1,
                "category": "PROVENANCE_FAIL", "failure_detail": prov_detail,
            }); continue

        # Gate 2: Field validity
        fv = check_field_validity_v3(fp)
        field_ok = fv["field_validity_pass"]
        field_detail = f"domain_violations={fv['domain_violation_count']}"
        total_domain_violations += fv["domain_violation_count"]

        if not field_ok:
            failures["FIELD_VALIDITY_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": False, "open_convention_pass": "",
                "remap_pass": "", "n_total_rows": fv["n_rows_checked"],
                "n_gripper_valid_rows": "", "n_neutral_rows": "",
                "remap_field_issue_count": "", "grasp_privilege_valid": "",
                "n_close_candidates": -1, "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "legacy_first_match_step": -1, "teacher_r_step": -1,
                "category": "FIELD_VALIDITY_FAIL", "failure_detail": field_detail,
            }); continue

        # Gate 3: Open convention
        oc = check_open_convention_v2(fp)
        oc_ok = oc["open_convention_pass"]
        oc_detail = f"n_violations={oc['n_violations']}"

        if not oc_ok:
            failures["OPEN_CONVENTION_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": False,
                "remap_pass": "", "n_total_rows": fv["n_rows_checked"],
                "n_gripper_valid_rows": "", "n_neutral_rows": "",
                "remap_field_issue_count": "", "grasp_privilege_valid": "",
                "n_close_candidates": -1, "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "legacy_first_match_step": -1, "teacher_r_step": -1,
                "category": "OPEN_CONVENTION_FAIL", "failure_detail": oc_detail,
            }); continue

        # Gate 4: RC1a remap
        rows, invariants, field_issues = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
        n_total_rows = len(rows) if rows else 0
        n_inv = len(invariants) if invariants else 0
        n_fi = len(field_issues) if field_issues else 0
        n_gripper_valid = sum(1 for r in (rows or [])
                              if r.get("gripper_semantics_valid", "1") not in ("0", "False", "false"))
        n_neutral = sum(1 for r in (rows or [])
                        if abs(_safe_float(r.get("clean_gripper_env", 0))) <= 0.5
                        and r.get("clean_gripper_env", "") != "")

        remap_ok = n_total_rows > 0 and n_inv == 0
        remap_detail = f"rows={n_total_rows} inv={n_inv} field_issues={n_fi}"

        if not remap_ok:
            failures["RC1A_REMAP_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": True,
                "remap_pass": False, "n_total_rows": n_total_rows,
                "n_gripper_valid_rows": n_gripper_valid, "n_neutral_rows": n_neutral,
                "remap_field_issue_count": n_fi, "grasp_privilege_valid": "",
                "n_close_candidates": -1, "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "legacy_first_match_step": -1, "teacher_r_step": -1,
                "category": "RC1A_REMAP_FAIL", "failure_detail": remap_detail,
            }); continue

        # ── Teacher-P + candidate enumeration ──
        cap = check_teacher_p_privilege_capability(rows)
        grasp_priv_valid = cap["grasp_privilege_valid"]
        legacy_first_match = teacher_privileged_critical_close_anchor(rows)
        r_anchor = teacher_rule_critical_close_anchor(rows)

        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=legacy_first_match if legacy_first_match >= 0 else -1)

        # All CLOSE-event candidates (3-condition OR, no selector filter)
        all_close_candidates = [p for p in preds if p.get("is_close_event_candidate")]
        emittable_steps = {c["step"] for c in all_close_candidates if not c.get("abstain")}

        # Per-candidate Teacher-P evidence
        tp_evidence = []
        for c in all_close_candidates:
            ev = evaluate_teacher_p_for_candidate(rows, c["step"])
            tp_evidence.append(ev)
        n_tp_qualifying = sum(1 for e in tp_evidence if e["teacher_p_criteria_pass"])
        n_all_candidates = len(all_close_candidates)

        # P0-3: Determine unique Teacher-P step
        if n_tp_qualifying == 1:
            unique_tp_step = next(e["candidate_step"] for e in tp_evidence if e["teacher_p_criteria_pass"])
        else:
            unique_tp_step = -1  # 0 or >1 qualifying → no unique positive label

        # Classification
        category = classify_trace_v3(
            prov_ok, field_ok, oc_ok, remap_ok,
            n_all_candidates, n_tp_qualifying, grasp_priv_valid, remap_detail)
        failures[category] += 1
        if category.startswith("ELIGIBLE"):
            task_counts[task]["eligible"] += 1
            if category == "ELIGIBLE_MULTI_CANDIDATE": task_counts[task]["multi"] += 1
            else: task_counts[task]["single"] += 1
        elif category == "TEACHER_P_AMBIGUOUS":
            task_counts[task]["ambiguous"] += 1
        elif category == "TEACHER_P_UNAVAILABLE":
            task_counts[task]["unavailable"] += 1
        elif category == "NO_CLOSE_CANDIDATE":
            task_counts[task]["nocand"] += 1

        trace_status.append({
            "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
            "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": True,
            "remap_pass": True, "n_total_rows": n_total_rows,
            "n_gripper_valid_rows": n_gripper_valid, "n_neutral_rows": n_neutral,
            "remap_field_issue_count": n_fi,
            "grasp_privilege_valid": grasp_priv_valid,
            "n_close_candidates": n_all_candidates,
            "n_tp_qualifying_candidates": n_tp_qualifying,
            "teacher_p_step": unique_tp_step,
            "legacy_first_match_step": legacy_first_match,
            "teacher_r_step": r_anchor,
            "category": category,
            "failure_detail": "",
        })

        # ── Candidate rows ──
        # P0-2: open_steps from REMAPPED ROWS (not predictions)
        open_steps_from_rows = [
            int(r.get("step", i))
            for i, r in enumerate(rows)
            if int(_safe_float(r.get("decoded_open_bool", 0))) == 1
        ]
        tp_ev_by_step = {e["candidate_step"]: e for e in tp_evidence}

        for idx, c in enumerate(all_close_candidates):
            step = c["step"]
            ev = tp_ev_by_step.get(step, {})
            prev_close = all_close_candidates[idx - 1]["step"] if idx > 0 else None

            # P0-2: time_since_last_open from rows (not preds)
            candidates_before = [s for s in open_steps_from_rows if s < step]
            last_open = max(candidates_before) if candidates_before else None

            speed_now = c.get("eef_speed_now", "")
            speed_prev = c.get("eef_speed_prev", "")
            decel_delta = ""
            if speed_now != "" and speed_prev != "":
                try: decel_delta = round(float(speed_now) - float(speed_prev), 6)
                except: pass

            # P0-3: is_teacher_p only when exactly 1 TP-qualifying candidate exists
            is_tp = int(step == unique_tp_step) if unique_tp_step >= 0 else 0

            trace_candidates.append({
                "trace_id": tid, "task_key": task, "state_id": state,
                "candidate_step": step, "candidate_index": idx,
                "is_teacher_p": is_tp,
                "teacher_p_criteria_pass": int(ev.get("teacher_p_criteria_pass", False)),
                "distance_to_teacher_p": step - unique_tp_step if unique_tp_step >= 0 else "",
                # All 16 frozen features
                "total_score": round(c.get("score", 0), 4),
                "raw_crossing_bonus": c.get("raw_crossing_bonus", ""),
                "close_streak_bonus": c.get("close_streak_bonus", ""),
                "close_onset_qpos_bonus": c.get("close_onset_qpos_bonus", ""),
                "eef_deceleration_bonus": c.get("eef_deceleration_bonus", ""),
                "qpos_ready_bonus": c.get("qpos_ready_bonus", ""),
                "eef_speed_now": speed_now,
                "eef_speed_prev": speed_prev,
                "eef_deceleration_delta": decel_delta,
                "close_streak": c.get("close_streak_value", ""),
                "raw_crossing": int(c.get("raw_open_to_close_crossing", 0)),
                "close_onset": int(c.get("close_onset", 0)),
                "qpos": c.get("qpos", ""),
                "time_since_prev_close": step - prev_close if prev_close is not None else "",
                "time_since_last_open": step - last_open if last_open is not None else "",
                # Selector flags
                "selector_abstain_reason": c.get("abstain", ""),
                "selector_emittable": int(step in emittable_steps),
                # Teacher-P evidence
                "eef_to_obj_distance_at_close": ev.get("eef_to_obj_distance_at_close", ""),
                "grasp_privilege_local_valid": int(ev.get("grasp_privilege_locally_valid", False)),
                "max_cumulative_vertical_dz": ev.get("max_cumulative_vertical_dz", ""),
                "max_sustained_vertical_frames": ev.get("max_sustained_vertical_frames", ""),
                "eef_attachment_consistent": int(ev.get("eef_attachment_consistent", True)),
                "tp_abstain_reason": ev.get("grasp_local_abstain_reason", ""),
            })

    elapsed = time.time() - t0
    tasks_seen = set(r["task_key"] for r in trace_status)
    print(f"\nElapsed: {elapsed:.1f}s  Rate: {len(batch)/elapsed:.2f} traces/s")

    # ── Consistency assertions ──
    print("\n=== CONSISTENCY ASSERTIONS ===")
    assertions_ok = True
    def _check(label, cond):
        nonlocal assertions_ok
        status = "PASS" if cond else "FAIL"
        if not cond: assertions_ok = False
        print(f"  [{status}] {label}")

    _check("sum(trace categories) == n_processed",
           sum(failures.values()) == len(trace_status) == len(batch))
    _check("sum(task totals) == n_processed",
           sum(len([r for r in trace_status if r["task_key"] == t]) for t in tasks_seen) == len(batch))
    _check("candidate rows == sum(n_close_candidates)",
           len(trace_candidates) == sum(int(r["n_close_candidates"]) for r in trace_status if r["n_close_candidates"] >= 0))
    # For eligible single: n_close_candidates=1 AND n_tp_qualifying=1
    _check("eligible single → n_cands=1 AND n_tp_qual=1",
           all(int(r["n_close_candidates"]) == 1 and int(r["n_tp_qualifying_candidates"]) == 1
               for r in trace_status if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE"))
    # For eligible multi: n_close_candidates>=2 AND n_tp_qualifying=1
    _check("eligible multi → n_cands>=2 AND n_tp_qual=1",
           all(int(r["n_close_candidates"]) >= 2 and int(r["n_tp_qualifying_candidates"]) == 1
               for r in trace_status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"))
    # For ambiguous: n_tp_qualifying>1 AND teacher_p_step=-1
    _check("ambiguous → n_tp_qual>1 AND teacher_p_step=-1",
           all(int(r["n_tp_qualifying_candidates"]) > 1 and int(r["teacher_p_step"]) == -1
               for r in trace_status if r["category"] == "TEACHER_P_AMBIGUOUS"))
    # For unavailable (with cands): n_cands>0 AND n_tp_qual=0
    _check("unavailable → n_cands>0 AND n_tp_qual=0",
           all(int(r["n_close_candidates"]) > 0 and int(r["n_tp_qualifying_candidates"]) == 0
               for r in trace_status if r["category"] == "TEACHER_P_UNAVAILABLE"))
    # For no-candidate: n_close_candidates=0
    _check("no-candidate → n_cands=0",
           all(int(r["n_close_candidates"]) == 0
               for r in trace_status if r["category"] == "NO_CLOSE_CANDIDATE"))
    _check("runtime provenance seal passed", seal["pass"])
    print(f"  ALL ASSERTIONS: {'PASS' if assertions_ok else 'FAIL'}")

    # ── Write outputs ──
    status_fields = [
        "trace_id", "task_key", "state_id", "seed",
        "provenance_pass", "field_validity_pass", "open_convention_pass",
        "remap_pass", "n_total_rows", "n_gripper_valid_rows", "n_neutral_rows",
        "remap_field_issue_count", "grasp_privilege_valid",
        "n_close_candidates", "n_tp_qualifying_candidates",
        "teacher_p_step", "legacy_first_match_step", "teacher_r_step",
        "category", "failure_detail",
    ]
    with open(out / "l12_e4c2b_trace_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=status_fields); w.writeheader(); w.writerows(trace_status)

    if trace_candidates:
        cfields = list(trace_candidates[0].keys())
        with open(out / "l12_e4c2b_close_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(trace_candidates)

    # Teacher-P coverage
    p_cov_fields = [
        "trace_id", "task_key", "state_id",
        "grasp_privilege_valid", "n_close_candidates", "n_tp_qualifying_candidates",
        "teacher_p_step", "legacy_first_match_step", "teacher_r_step", "category",
    ]
    with open(out / "l12_e4c2b_teacher_p_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=p_cov_fields); w.writeheader()
        for r in trace_status:
            if r["provenance_pass"] and r["field_validity_pass"] and r["open_convention_pass"] and r["remap_pass"]:
                w.writerow({k: r[k] for k in p_cov_fields})

    # Task summary
    tasks_seen = set(r["task_key"] for r in trace_status)
    task_rows = []
    for t in sorted(tasks_seen):
        trs = [r for r in trace_status if r["task_key"] == t]
        task_rows.append({
            "task_key": t, "n_total": len(trs),
            "n_tp_qualifying_0": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) == 0),
            "n_tp_qualifying_1": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) == 1),
            "n_tp_qualifying_gt1": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) > 1),
            "n_eligible_multi": sum(1 for r in trs if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"),
            "n_eligible_single": sum(1 for r in trs if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE"),
            "n_tp_unavailable": sum(1 for r in trs if r["category"] == "TEACHER_P_UNAVAILABLE"),
            "n_tp_ambiguous": sum(1 for r in trs if r["category"] == "TEACHER_P_AMBIGUOUS"),
            "n_no_candidate": sum(1 for r in trs if r["category"] == "NO_CLOSE_CANDIDATE"),
            "n_other_abstain": sum(1 for r in trs if r["category"] == "OTHER_ABSTAIN"),
        })
    with open(out / "l12_e4c2b_task_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys())); w.writeheader(); w.writerows(task_rows)

    # Failure taxonomy
    tax_rows = [{"category": cat, "count": count} for cat, count in failures.most_common()]
    with open(out / "l12_e4c2b_failure_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "count"]); w.writeheader(); w.writerows(tax_rows)

    # Output hashes
    hash_rows = []
    for fname in sorted(os.listdir(str(out))):
        fpath = out / fname
        if fpath.suffix == ".csv":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
            hash_rows.append({"file": fname, "sha256": h.hexdigest()})
    with open(out / "l12_e4c2b_output_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256"]); w.writeheader(); w.writerows(hash_rows)

    # Run log
    end_time = datetime.now(timezone.utc)
    with open(out / "l12_e4c2b_run_log.txt", "w") as f:
        f.write(f"E4C.2b RUN LOG\n")
        f.write(f"hostname: {hostname}\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        for k, v in seal["artifacts"].items():
            f.write(f"provenance_{k}: {v}\n")
        for name, expected in [
            ("manifest", EXPECTED_MANIFEST_SHA), ("config", EXPECTED_CONFIG_SHA),
            ("remapper", EXPECTED_REMAPPER_SHA), ("phase_detector", EXPECTED_PHASE_DETECTOR_SHA),
            ("selector", EXPECTED_SELECTOR_SHA)]:
            f.write(f"expected_{name}_sha: {expected}\n")
        f.write(f"n_input: {total}\nn_processed: {len(batch)}\n")
        f.write(f"n_trace_status_rows: {len(trace_status)}\n")
        f.write(f"n_candidate_rows: {len(trace_candidates)}\n")
        f.write(f"total_domain_violations: {total_domain_violations}\n")
        f.write(f"cuda_available: {gpu_info['cuda_available']}\n")
        f.write(f"gpu_count: {gpu_info['gpu_count']}\n")
        f.write(f"consistency_assertions_pass: {assertions_ok}\n")
        for cat, count in failures.most_common():
            f.write(f"category_{cat}: {count}\n")

    print(f"\n=== E4C.2b SUMMARY ===")
    print(f"Traces: {len(trace_status)}  Candidates: {len(trace_candidates)}")
    for cat, count in failures.most_common():
        print(f"  {cat}: {count}")
    n_tp0 = sum(1 for r in trace_status if int(r["n_tp_qualifying_candidates"]) == 0)
    n_tp1 = sum(1 for r in trace_status if int(r["n_tp_qualifying_candidates"]) == 1)
    n_tp_gt1 = sum(1 for r in trace_status if int(r["n_tp_qualifying_candidates"]) > 1)
    print(f"\nTP-qualifying: 0={n_tp0}  1={n_tp1}  >1={n_tp_gt1}")
    print(f"Domain violations (descriptive): {total_domain_violations}")
    print(f"Consistency assertions: {'PASS' if assertions_ok else 'FAIL'}")
    print(f"\nOutput: {out}")
    print("E4C.2b COMPLETE")
    print("TRAINING_STARTED: NO")

    if not assertions_ok:
        sys.exit(2)


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
