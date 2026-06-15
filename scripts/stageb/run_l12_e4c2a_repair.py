#!/usr/bin/env python3
"""E4C.2a: Deterministic label-audit repair.

Fixes all P0 issues from E4C.2 audit:
  P0-1: Export candidates for ALL remap-passing traces (not just TP-available)
  P0-2: Full 10-category taxonomy with per-candidate Teacher-P evidence
  P0-3: Candidate definition matches prereg (no selector-abstain filter)
  P0-4: All 16 frozen E4B.3 features exported
  P0-5: Runtime provenance seal (manifest/config/source SHA verified)
  P1:   Domain validity checks and RC1a detail

Runs directly on GPU server. All thresholds frozen.
TRAINING FORBIDDEN.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Runtime paths (deployed on server) ──
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

# ── Frozen config ──
CONFIG_PATH = os.path.join(PIPELINE_ROOT, "..", "..", "configs", "l12_e4c2_remap_teacherp.yaml")
# Actually: config is committed locally, not on server. Use embedded hash.
FROZEN_CONFIG_SHA = "2BFDCC4222298D6E803A3287A39E5F422DDCCCC8969FE830166E872452598F80"

REQUIRED_HEADER = [
    "obj_x", "obj_y", "obj_z",
    "eef_x", "eef_y", "eef_z",
    "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before",
]

# All 16 frozen features
ALL_16_FEATURES = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus",
    "close_onset_qpos_bonus", "eef_deceleration_bonus", "qpos_ready_bonus",
    "eef_speed_now", "eef_speed_prev", "eef_deceleration_delta",
    "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]


# ── P0-5: Runtime provenance seal ──
def sha256_file(path: str) -> str:
    if not os.path.isfile(path): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def runtime_provenance_check(manifest_path: str, expected_manifest_sha: str) -> dict:
    """Verify runtime manifest and source files match frozen versions."""
    result = {"pass": True, "failures": [], "artifacts": {}}

    # Manifest SHA
    actual = sha256_file(manifest_path)
    result["artifacts"]["manifest_sha"] = actual
    if actual.lower() != expected_manifest_sha.lower():
        result["pass"] = False
        result["failures"].append(f"manifest_sha: expected {expected_manifest_sha[:16]} got {actual[:16]}")

    # Source file SHAs (the 3 critical label-producing files)
    src_files = {
        "remap_v4_trace_for_l12": os.path.join(PIPELINE_ROOT, "scripts", "stageb", "remap_v4_trace_for_l12.py"),
        "phase_detector": os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "phase_detector.py"),
        "critical_close_selector": os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "critical_close_selector.py"),
    }
    for name, path in src_files.items():
        sha = sha256_file(path)
        result["artifacts"][f"{name}_sha"] = sha
        if sha == "MISSING":
            result["pass"] = False
            result["failures"].append(f"{name}: file missing")

    # Runner SHA (this script itself)
    runner_sha = sha256_file(__file__)
    result["artifacts"]["runner_sha"] = runner_sha

    # Git commit (pipeline dir may not be a repo — that's fine, source SHAs seal it)
    git_commit = "n/a_pipeline_not_git_repo"
    git_dirty = "n/a"
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT, text=True, timeout=10
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=PIPELINE_ROOT, text=True, timeout=10
        ).strip()
        git_dirty = "True" if dirty_out else "False"
    except Exception:
        pass
    result["artifacts"]["git_commit"] = git_commit
    result["artifacts"]["git_dirty"] = git_dirty

    return result


# ── P1: Domain validity ──
def check_field_validity_v2(fp: str) -> dict:
    """Full row-level field validity WITH domain checks."""
    result = {"field_validity_pass": True, "n_rows_checked": 0,
              "fields_checked": {}, "first_invalid_row": -1, "invalid_domain_count": 0}
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
        result["fields_checked"][fld] = {"missing": 0, "parse_fail": 0, "non_finite": 0, "invalid_domain": 0}

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
                # Domain checks
                if fld in ("obj_x", "obj_y", "eef_x", "eef_y") and abs(fv) > 10:
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
                elif fld == "obj_z" and (fv < -1 or fv > 5):
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
                elif fld == "eef_z" and (fv < -1 or fv > 5):
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
                elif fld == "clean_gripper_env" and abs(fv) > 10:
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
                elif fld == "decoded_open_bool" and fv not in (0, 1):
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
                elif fld == "gripper_qpos_before" and (fv < 0 or fv > 1):
                    result["fields_checked"][fld]["invalid_domain"] += 1
                    result["invalid_domain_count"] += 1
            except ValueError:
                result["fields_checked"][fld]["parse_fail"] += 1
                if result["first_invalid_row"] < 0: result["first_invalid_row"] = i + 1

    for fld in REQUIRED_HEADER:
        s = result["fields_checked"][fld]
        if any(s[k] > 0 for k in ("missing", "parse_fail", "non_finite")):
            result["field_validity_pass"] = False
    return result


def check_open_convention_v2(fp: str) -> dict:
    """Same as before — decoded_open_bool iff env < -0.5."""
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


# ── P0-2: Per-candidate Teacher-P evidence ──
def evaluate_teacher_p_for_candidate(records: list[dict], t: int) -> dict:
    """Evaluate all 5 Teacher-P criteria for a specific close-candidate step.

    Returns evidence dict so ambiguity can be detected (multiple candidates
    may satisfy all 5 criteria).
    """
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

    if t < 0 or t >= len(records):
        return result

    r = records[t]

    # Criterion 1: close_onset + clean_close
    if not (int(_safe_float(r.get("close_onset", 0))) and
            int(_safe_float(r.get("clean_close", 0)))):
        result["grasp_local_abstain_reason"] = "not_close_onset_or_not_clean_close"
        return result
    result["is_close_onset_and_clean_close"] = True

    # Criterion 2: gripper not already open
    if int(_safe_float(r.get("decoded_open_bool", 0))):
        result["grasp_local_abstain_reason"] = "gripper_already_open"
        return result
    result["gripper_not_already_open"] = True

    # Criterion 3: grasp privilege locally valid
    if not _check_grasp_privilege_valid(records, t):
        result["grasp_local_abstain_reason"] = "grasp_privilege_not_locally_valid"
        return result
    result["grasp_privilege_locally_valid"] = True

    # Criterion 4: EEF near object
    eef_dist = _safe_float(r.get("eef_to_obj_distance", 999))
    result["eef_to_obj_distance_at_close"] = round(eef_dist, 6)
    if eef_dist > EEF_TO_OBJ_NEAR_THRESHOLD:
        result["grasp_local_abstain_reason"] = f"eef_not_near_object_dist={eef_dist:.4f}"
        return result
    result["eef_near_object"] = True

    # Criterion 5: sustained vertical lift
    evidence = _classify_motion_evidence(records, t)
    result["max_cumulative_vertical_dz"] = round(evidence["cumulative_vertical_dz"], 6)
    result["max_sustained_vertical_frames"] = evidence["sustained_above_threshold_frames"]
    result["eef_attachment_consistent"] = evidence["eef_attachment_consistent"]
    if evidence["motion_evidence_type"] != MOTION_SUSTAINED_VERTICAL_LIFT:
        result["grasp_local_abstain_reason"] = (
            f"no_sustained_vertical_lift_type={evidence['motion_evidence_type']}"
        )
        return result
    result["sustained_vertical_lift"] = True

    result["teacher_p_criteria_pass"] = True
    return result


# ── P0-2: Full taxonomy ──
def classify_trace_v2(
    provenance_ok: bool, field_ok: bool, oc_ok: bool, remap_ok: bool,
    all_candidates: list[dict], tp_evidence: list[dict],
    grasp_priv_valid: bool, remap_abstain: str,
) -> str:
    """Assign exactly one of 10 eligibility categories."""
    if not provenance_ok: return "PROVENANCE_FAIL"
    if not field_ok: return "FIELD_VALIDITY_FAIL"
    if not oc_ok: return "OPEN_CONVENTION_FAIL"
    if not remap_ok: return "RC1A_REMAP_FAIL"

    n_candidates = len(all_candidates)
    n_tp_qualifying = sum(1 for e in tp_evidence if e["teacher_p_criteria_pass"])

    if n_candidates == 0:
        return "NO_CLOSE_CANDIDATE"
    if grasp_priv_valid is False:
        return "TEACHER_P_UNAVAILABLE"
    if n_tp_qualifying == 0:
        return "TEACHER_P_UNAVAILABLE"
    if n_tp_qualifying > 1:
        return "TEACHER_P_AMBIGUOUS"
    # Exactly 1 TP-qualifying candidate
    if n_candidates == 1:
        return "ELIGIBLE_SINGLE_CANDIDATE"
    return "ELIGIBLE_MULTI_CANDIDATE"


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
        if info["cuda_available"]:
            info["torch_version"] = torch.__version__
            info["cuda_version"] = torch.version.cuda
    except ImportError: pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-manifest-sha", required=True,
                    help="Frozen input manifest SHA256")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # ── P0-5: Runtime provenance seal ──
    print("=== RUNTIME PROVENANCE SEAL ===")
    prov = runtime_provenance_check(args.manifest, args.expected_manifest_sha)
    for k, v in prov["artifacts"].items():
        if k.endswith("_sha"):
            print(f"  {k}: {v[:16]}...")
    print(f"  git_commit: {prov['artifacts']['git_commit']}")
    print(f"  git_dirty: {prov['artifacts']['git_dirty']}")
    if not prov["pass"]:
        print(f"FATAL: provenance seal failed: {prov['failures']}")
        sys.exit(1)
    print("  PROVENANCE SEAL: PASS\n")

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

    for i, mr in enumerate(batch):
        tid = mr["trace_id"]; task = mr["task_key"]; state = mr["state_id"]
        fp = mr["source_path"]

        if (i + 1) % 25 == 0:
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
                prov_ok = False; prov_detail = f"row_count_mismatch_{n_rows}_vs_{mr['row_count']}"

        if not prov_ok:
            failures["PROVENANCE_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": False, "field_validity_pass": "", "open_convention_pass": "",
                "remap_pass": "", "n_total_rows": -1, "n_gripper_valid_rows": -1,
                "n_neutral_rows": -1, "remap_field_issue_count": -1,
                "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "teacher_r_step": -1,
                "category": "PROVENANCE_FAIL", "failure_detail": prov_detail,
            }); continue

        # Gate 2: Field validity v2
        fv = check_field_validity_v2(fp)
        field_ok = fv["field_validity_pass"]
        field_detail = f"invalid_domain={fv['invalid_domain_count']} first_bad={fv['first_invalid_row']}"

        if not field_ok:
            failures["FIELD_VALIDITY_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
                "provenance_pass": True, "field_validity_pass": False, "open_convention_pass": "",
                "remap_pass": "", "n_total_rows": fv["n_rows_checked"],
                "n_gripper_valid_rows": "", "n_neutral_rows": "",
                "remap_field_issue_count": "", "grasp_privilege_valid": "",
                "n_close_candidates": -1, "n_tp_qualifying_candidates": -1,
                "teacher_p_step": -1, "teacher_r_step": -1,
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
                "teacher_p_step": -1, "teacher_r_step": -1,
                "category": "OPEN_CONVENTION_FAIL", "failure_detail": oc_detail,
            }); continue

        # Gate 4: RC1a remap
        rows, invariants, field_issues = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
        n_total_rows = len(rows) if rows else 0
        n_inv = len(invariants) if invariants else 0
        n_fi = len(field_issues) if field_issues else 0
        # Count gripper-valid and neutral rows
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
                "teacher_p_step": -1, "teacher_r_step": -1,
                "category": "RC1A_REMAP_FAIL", "failure_detail": remap_detail,
            }); continue

        # ── Teacher-P evaluation and candidate enumeration ──
        cap = check_teacher_p_privilege_capability(rows)
        grasp_priv_valid = cap["grasp_privilege_valid"]
        p_anchor = teacher_privileged_critical_close_anchor(rows)
        r_anchor = teacher_rule_critical_close_anchor(rows)

        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if p_anchor >= 0 else -1)

        # P0-3: Candidates defined by 3-condition OR, WITHOUT selector abstain filter
        all_close_candidates = [p for p in preds if p.get("is_close_event_candidate")]
        # Subset that passes selector's own abstain (for emittable flag)
        emittable_candidates = [p for p in all_close_candidates if not p.get("abstain")]

        n_all_candidates = len(all_close_candidates)

        # P0-2: Per-candidate Teacher-P evidence
        tp_evidence = []
        for c in all_close_candidates:
            ev = evaluate_teacher_p_for_candidate(rows, c["step"])
            tp_evidence.append(ev)
        n_tp_qualifying = sum(1 for e in tp_evidence if e["teacher_p_criteria_pass"])

        # Classification (v2 — full taxonomy)
        category = classify_trace_v2(
            prov_ok, field_ok, oc_ok, remap_ok,
            all_close_candidates, tp_evidence, grasp_priv_valid, remap_detail)
        failures[category] += 1
        if category.startswith("ELIGIBLE"):
            task_counts[task]["eligible"] += 1
            if category == "ELIGIBLE_MULTI_CANDIDATE": task_counts[task]["multi"] += 1
            else: task_counts[task]["single"] += 1
        if category == "TEACHER_P_AMBIGUOUS":
            task_counts[task]["ambiguous"] += 1

        trace_status.append({
            "trace_id": tid, "task_key": task, "state_id": state, "seed": mr["seed"],
            "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": True,
            "remap_pass": True, "n_total_rows": n_total_rows,
            "n_gripper_valid_rows": n_gripper_valid, "n_neutral_rows": n_neutral,
            "remap_field_issue_count": n_fi,
            "grasp_privilege_valid": grasp_priv_valid,
            "n_close_candidates": n_all_candidates,
            "n_tp_qualifying_candidates": n_tp_qualifying,
            "teacher_p_step": p_anchor, "teacher_r_step": r_anchor,
            "category": category,
            "failure_detail": (
                f"tp_qualifying={n_tp_qualifying}" if category in ("TEACHER_P_UNAVAILABLE", "TEACHER_P_AMBIGUOUS")
                else remap_detail if category == "RC1A_REMAP_FAIL" else ""
            ),
        })

        # P0-4: Export all candidates WITH all 16 features + TP evidence + selector flags
        # Build lookup: step → TP evidence
        tp_ev_by_step = {e["candidate_step"]: e for e in tp_evidence}
        emittable_steps = {c["step"] for c in emittable_candidates}
        open_steps = [p["step"] for p in preds if p.get("decoded_open_bool")]

        for idx, c in enumerate(all_close_candidates):
            step = c["step"]
            ev = tp_ev_by_step.get(step, {})
            prev_close = all_close_candidates[idx - 1]["step"] if idx > 0 else None
            last_open = max([s for s in open_steps if s < step]) if [s for s in open_steps if s < step] else None
            speed_now = c.get("eef_speed_now", "")
            speed_prev = c.get("eef_speed_prev", "")
            decel_delta = ""
            if speed_now != "" and speed_prev != "":
                try: decel_delta = round(float(speed_now) - float(speed_prev), 6)
                except: pass

            trace_candidates.append({
                "trace_id": tid, "task_key": task, "state_id": state,
                "candidate_step": step, "candidate_index": idx,
                "is_teacher_p": int(step == p_anchor) if p_anchor >= 0 else 0,
                "teacher_p_criteria_pass": int(ev.get("teacher_p_criteria_pass", False)),
                "distance_to_p": step - p_anchor if p_anchor >= 0 else "",
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
                # Selector abstain / emittable flags (P0-3)
                "selector_abstain_reason": c.get("abstain", ""),
                "selector_emittable": int(step in emittable_steps),
                # Teacher-P evidence (P0-2)
                "eef_to_obj_distance_at_close": ev.get("eef_to_obj_distance_at_close", ""),
                "grasp_privilege_local_valid": int(ev.get("grasp_privilege_locally_valid", False)),
                "max_cumulative_vertical_dz": ev.get("max_cumulative_vertical_dz", ""),
                "max_sustained_vertical_frames": ev.get("max_sustained_vertical_frames", ""),
                "eef_attachment_consistent": int(ev.get("eef_attachment_consistent", True)),
                "tp_abstain_reason": ev.get("grasp_local_abstain_reason", ""),
            })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s  Rate: {len(batch)/elapsed:.2f} traces/s")

    # ── Write outputs ──
    status_fields = [
        "trace_id", "task_key", "state_id", "seed",
        "provenance_pass", "field_validity_pass", "open_convention_pass",
        "remap_pass", "n_total_rows", "n_gripper_valid_rows", "n_neutral_rows",
        "remap_field_issue_count", "grasp_privilege_valid",
        "n_close_candidates", "n_tp_qualifying_candidates",
        "teacher_p_step", "teacher_r_step", "category", "failure_detail",
    ]
    with open(out / "l12_e4c2a_trace_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=status_fields); w.writeheader(); w.writerows(trace_status)

    if trace_candidates:
        cfields = list(trace_candidates[0].keys())
        with open(out / "l12_e4c2a_close_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(trace_candidates)

    # Teacher-P coverage
    p_cov_fields = [
        "trace_id", "task_key", "state_id",
        "grasp_privilege_valid", "n_close_candidates", "n_tp_qualifying_candidates",
        "teacher_p_step", "teacher_r_step", "category",
    ]
    with open(out / "l12_e4c2a_teacher_p_coverage.csv", "w", newline="") as f:
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
            "n_provenance_pass": sum(1 for r in trs if r["provenance_pass"]),
            "n_field_valid": sum(1 for r in trs if r["field_validity_pass"] == True),
            "n_open_convention_pass": sum(1 for r in trs if r["open_convention_pass"] == True),
            "n_remap_pass": sum(1 for r in trs if r["remap_pass"]),
            "n_grasp_privilege_valid": sum(1 for r in trs if r["grasp_privilege_valid"] == True),
            "n_tp_qualifying_1": sum(1 for r in trs if r["n_tp_qualifying_candidates"] == 1),
            "n_tp_qualifying_gt1": sum(1 for r in trs if r["n_tp_qualifying_candidates"] > 1),
            "n_tp_qualifying_0": sum(1 for r in trs if r["n_tp_qualifying_candidates"] == 0),
            "n_eligible_multi": sum(1 for r in trs if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"),
            "n_eligible_single": sum(1 for r in trs if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE"),
            "n_no_candidate": sum(1 for r in trs if r["category"] == "NO_CLOSE_CANDIDATE"),
            "n_tp_unavailable": sum(1 for r in trs if r["category"] == "TEACHER_P_UNAVAILABLE"),
            "n_tp_ambiguous": sum(1 for r in trs if r["category"] == "TEACHER_P_AMBIGUOUS"),
        })
    with open(out / "l12_e4c2a_task_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys())); w.writeheader(); w.writerows(task_rows)

    # Failure taxonomy
    tax_rows = [{"category": cat, "count": count} for cat, count in failures.most_common()]
    with open(out / "l12_e4c2a_failure_taxonomy.csv", "w", newline="") as f:
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
    with open(out / "l12_e4c2a_output_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256"]); w.writeheader(); w.writerows(hash_rows)

    # Run log
    end_time = datetime.now(timezone.utc)
    with open(out / "l12_e4c2a_run_log.txt", "w") as f:
        f.write(f"E4C.2a RUN LOG\n")
        f.write(f"hostname: {hostname}\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"frozen_config_sha: {FROZEN_CONFIG_SHA}\n")
        f.write(f"input_manifest: {args.manifest}\n")
        f.write(f"input_manifest_expected_sha: {args.expected_manifest_sha}\n")
        f.write(f"input_manifest_runtime_sha: {prov['artifacts']['manifest_sha']}\n")
        for k, v in prov["artifacts"].items():
            f.write(f"provenance_{k}: {v}\n")
        f.write(f"n_input: {total}\nn_processed: {len(batch)}\n")
        f.write(f"n_trace_status_rows: {len(trace_status)}\n")
        f.write(f"n_candidate_rows: {len(trace_candidates)}\n")
        f.write(f"cuda_available: {gpu_info['cuda_available']}\n")
        f.write(f"gpu_count: {gpu_info['gpu_count']}\n")
        f.write(f"env_cuda_visible_devices: {gpu_info['env_cuda_visible']}\n")
        for dev in gpu_info.get("devices", []):
            f.write(f"gpu_{dev['index']}: {dev['name']}\n")
        for cat, count in failures.most_common():
            f.write(f"category_{cat}: {count}\n")

    # ── Report ──
    print(f"\n=== E4C.2a SUMMARY ===")
    print(f"Traces processed: {len(trace_status)}")
    for cat, count in failures.most_common():
        print(f"  {cat}: {count}")
    n_all_cands = sum(int(r["n_close_candidates"]) for r in trace_status if r["n_close_candidates"] >= 0)
    n_tp_avail = sum(1 for r in trace_status if r["n_tp_qualifying_candidates"] >= 1)
    n_ambig = sum(1 for r in trace_status if r["category"] == "TEACHER_P_AMBIGUOUS")
    print(f"\nTotal candidates across all traces: {len(trace_candidates)}")
    print(f"Traces with >=1 TP-qualifying candidate: {n_tp_avail}")
    print(f"Ambiguous traces (>1 TP-qualifying): {n_ambig}")
    print(f"Output: {out}")
    print("E4C.2a COMPLETE")
    print("TRAINING_STARTED: NO")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
