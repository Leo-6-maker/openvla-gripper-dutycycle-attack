#!/usr/bin/env python3
"""R7.1.2.2: K10 OPPORTUNITY LABELER — canonical-action-corrected rematerialization.

Source: Physics Teacher V2.1C (SHA256SUMS 9c3c97ab...)
        V2.1C fixes the action inversion: raw < 0.5 = CLOSE (correct),
        replacing V2.1's raw >= 0.5 which selected the OPEN region.

Narrow change from V1.2.1:
  - Source Teacher: V2.1 → V2.1C (sealed root b7cc5b8)
  - Label filename: k10_labels_v122_v21c.jsonl
  - Protocol schema: V1.2.2_V21C_CANONICAL
  - Candidate-close now uses canonical action contract (raw < 0.5)

All K10 formulas, thresholds, K=10, component gates IDENTICAL to V1.2.1.
"""

from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

K = 10
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))

STABLE_GRASP_MIN = 0.5
LIFT_MIN = 0.3
SUPPORT_REMOVED_MIN = 0.3
TARGET_PROGRESS_MIN = 0.05
RELEASE_RISK_MAX = 0.5
REGRASP_RISK_MAX = 0.5
TASK_ROLE_MIN = 0.5

SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1_2_2_V21C_CANONICAL"
TEACHER_SHA = "9c3c97abf5a2db0f70993d666bdca028981c28b93ffebbe353e18123fcedfce9"
TEACHER_COMMIT = "b7cc5b8988e6316c6257ead354c460f16fa425d4"

REQUIRED_TEACHER_FIELDS = (
    "known_mask", "student_valid", "candidate_close",
    "stable_grasp_score", "lift_score", "support_removed",
    "target_progress", "target_progress_known",
    "release_risk", "regrasp_or_instability_risk",
    "task_grasp_necessity", "component_valid_mask",
    "phase_name", "window_id", "step",
    "physics_protocol_schema",
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_head() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not commit or len(commit) != 40:
        raise RuntimeError(f"git rev-parse HEAD failed: {commit!r}")
    return commit


# ── source validation (per-row, all 800 identities) ───────────────────
def validate_all(teacher_root: Path):
    """Fail-closed: validate SHA, file-set, all 800 identities, every row."""
    sums_path = teacher_root / "SHA256SUMS"
    if not sums_path.exists():
        raise SystemExit("Teacher root has no SHA256SUMS")
    actual = sha256_file(sums_path)
    if actual != TEACHER_SHA:
        raise SystemExit(f"Teacher SHA mismatch: expected {TEACHER_SHA}, got {actual}")

    # Verify SHA256SUMS internal consistency
    sums_lines = sums_path.read_text().strip().splitlines()
    listed = set()
    for line in sums_lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise SystemExit(f"Bad SHA256SUMS line: {line[:60]}")
        h, rel = parts
        fpath = teacher_root / rel
        if not fpath.exists():
            raise SystemExit(f"SHA256SUMS lists missing file: {rel}")
        if sha256_file(fpath) != h:
            raise SystemExit(f"File hash mismatch: {rel}")
        listed.add(str(rel))

    # Verify SHA256SUMS.sha256
    sums_sha_path = teacher_root / "SHA256SUMS.sha256"
    if not sums_sha_path.exists():
        raise SystemExit("Teacher root has no SHA256SUMS.sha256")
    sums_sha_content = sums_sha_path.read_text().strip()
    if actual not in sums_sha_content:
        raise SystemExit("SHA256SUMS.sha256 does not contain actual SHA256SUMS hash")

    # Validate all 800 identities
    total_rows = 0
    total_bad = 0
    for suite in SUITES:
        for task in range(N_TASKS):
            for state in FIT_STATES:
                path = teacher_root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "physics_teacher_v21c.jsonl"
                rel = str(path.relative_to(teacher_root))
                if rel not in listed:
                    raise SystemExit(f"Identity not in SHA256SUMS: {rel}")
                if not path.exists():
                    raise SystemExit(f"Identity file missing: {rel}")
                records = jsonl(path)
                if not records:
                    raise SystemExit(f"Empty identity: {rel}")
                cid = f"{suite}/task_{task:02d}/state_{state:02d}"
                for row_idx, r in enumerate(records):
                    total_rows += 1
                    if r.get("physics_protocol_schema") != "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL":
                        raise SystemExit(f"Wrong schema at {cid}:{row_idx}")
                    missing = [f for f in REQUIRED_TEACHER_FIELDS if f not in r]
                    if missing:
                        raise SystemExit(f"Missing fields at {cid}:{row_idx}: {missing}")
                    step_val = r["step"]
                    if not isinstance(step_val, int) or step_val != row_idx:
                        raise SystemExit(f"Step discontinuity at {cid}: expected {row_idx}, got {step_val}")
                    cm = r.get("component_valid_mask", {})
                    if not isinstance(cm, dict):
                        raise SystemExit(f"component_valid_mask not dict at {cid}:{row_idx}")
                    for k in ("lift_score", "object_eef_comotion_score", "regrasp_or_instability_risk",
                              "relative_pose_stability", "release_risk"):
                        if k not in cm:
                            raise SystemExit(f"Missing validity key {k} at {cid}:{row_idx}")
                    # Numeric finite checks
                    for fld in ("stable_grasp_score", "lift_score", "support_removed", "target_progress",
                               "release_risk", "regrasp_or_instability_risk"):
                        v = r.get(fld, 0.0)
                        if not isinstance(v, (int, float)):
                            raise SystemExit(f"Non-numeric {fld}={v} at {cid}:{row_idx}")
                total_bad += sum(1 for r in records if not isinstance(r.get("known_mask"), bool))

    print(f"Source validated: {len(listed)} files, {total_rows} rows across 800 IDs, {total_bad} unknown-masked rows")


# ── critical_t ─────────────────────────────────────────────────────────
def compute_critical(records: list[dict[str, Any]]) -> tuple[list[bool], list[str], list[int]]:
    """Returns (critical, reasons, component_bitmask).

    bitmask: 1=lift, 2=support_removed, 4=target_progress
    """
    n = len(records)
    critical = [False] * n
    reasons = ["none"] * n
    bitmask = [0] * n

    for i, r in enumerate(records):
        km = r["known_mask"]
        if not km:
            reasons[i] = "unknown_mask"
            continue

        sv = r["student_valid"]
        if not sv:
            reasons[i] = "student_invalid"
            continue

        cc = r["candidate_close"]
        if not cc:
            reasons[i] = "not_candidate_close"
            continue

        tgn = r["task_grasp_necessity"]
        if tgn < TASK_ROLE_MIN:
            reasons[i] = "task_role_not_applicable"
            continue

        sg = r["stable_grasp_score"]
        cm = r["component_valid_mask"]
        sg_valid = cm["relative_pose_stability"]
        if not sg_valid or sg < STABLE_GRASP_MIN:
            reasons[i] = "not_stable_grasp" if sg_valid else "stable_grasp_unknown"
            continue

        # manipulation: check all three independently, build bitmask
        lift = r["lift_score"]
        sr = r["support_removed"]
        tp = r["target_progress"]
        tpk = r["target_progress_known"]
        lu = cm["lift_score"]
        sru = cm["support_removed"]
        tpu = cm["target_progress"]

        mask = 0
        if lu and lift >= LIFT_MIN:
            mask |= 1
        if sru and sr >= SUPPORT_REMOVED_MIN:
            mask |= 2
        if tpu and tpk and tp > TARGET_PROGRESS_MIN:
            mask |= 4

        if mask == 0:
            reasons[i] = "not_manipulation_active"
            continue

        # release/regrasp: validity MUST be True; unknown → non-positive
        rr = r["release_risk"]
        ri = r["regrasp_or_instability_risk"]
        rru = cm["release_risk"]
        riu = cm["regrasp_or_instability_risk"]

        if not rru:
            reasons[i] = "release_risk_unknown"
            continue
        if not riu:
            reasons[i] = "regrasp_risk_unknown"
            continue
        if rr > RELEASE_RISK_MAX:
            reasons[i] = "release_risk"
            continue
        if ri > REGRASP_RISK_MAX:
            reasons[i] = "regrasp_or_instability"
            continue

        critical[i] = True
        reasons[i] = "critical"
        bitmask[i] = mask

    return critical, reasons, bitmask


# ── K10 burst ──────────────────────────────────────────────────────────
def compute_burst(critical: list[bool], n: int,
                  records: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    burst = [False] * n
    is_start = [False] * n

    in_seg = [-1] * n
    for i, r in enumerate(records):
        wid = r.get("window_id", "")
        if isinstance(wid, str) and wid.startswith("candidate:"):
            try:
                in_seg[i] = int(wid.split(":")[1])
            except ValueError:
                pass

    for t in range(n - K + 1):
        if not all(critical[t + k] for k in range(K)):
            continue
        seg = in_seg[t]
        if seg < 0:
            continue
        if not all(in_seg[t + k] == seg for k in range(K)):
            continue
        burst[t] = True
        is_start[t] = True

    return burst, is_start


# ── process one episode ────────────────────────────────────────────────
def process(teacher_root: Path, suite: str, task: int, state: int
            ) -> Optional[dict[str, Any]]:
    path = teacher_root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "physics_teacher_v21.jsonl"
    if not path.exists():
        return None
    records = jsonl(path)
    n = len(records)
    if n == 0:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
    critical, reasons, bitmask = compute_critical(records)
    burst, is_start = compute_burst(critical, n, records)

    feasible_starts = [i for i, s in enumerate(is_start) if s]
    has_feas = len(feasible_starts) > 0
    tgn = records[0].get("task_grasp_necessity", 0.0) if records else 0.0

    # Component funnel
    funnel = defaultdict(int)
    for i, r in enumerate(records):
        if not r["known_mask"] or not r["student_valid"]:
            continue
        if not r["candidate_close"]:
            continue
        funnel["candidate_close"] += 1
        if r["task_grasp_necessity"] < TASK_ROLE_MIN:
            continue
        funnel["task_role"] += 1
        cm = r["component_valid_mask"]
        if not cm["relative_pose_stability"] or r["stable_grasp_score"] < STABLE_GRASP_MIN:
            continue
        funnel["stable_grasp"] += 1
        if cm["lift_score"] and r["lift_score"] >= LIFT_MIN:
            funnel["lift_pass"] += 1
        if cm["support_removed"] and r["support_removed"] >= SUPPORT_REMOVED_MIN:
            funnel["support_removed_pass"] += 1
        if cm["target_progress"] and r["target_progress_known"] and r["target_progress"] > TARGET_PROGRESS_MIN:
            funnel["target_progress_pass"] += 1
        if not cm["release_risk"]:
            funnel["release_unknown_veto"] += 1
        elif r["release_risk"] > RELEASE_RISK_MAX:
            funnel["release_veto"] += 1
        if not cm["regrasp_or_instability_risk"]:
            funnel["regrasp_unknown_veto"] += 1
        elif r["regrasp_or_instability_risk"] > REGRASP_RISK_MAX:
            funnel["regrasp_veto"] += 1
        if critical[i]:
            funnel["critical"] += 1

    # Start source breakdown using bitmask from K=10 window
    start_sources = defaultdict(int)
    for s in feasible_starts:
        mask_union = 0
        mask_intersection = 7  # all three
        for k in range(K):
            m = bitmask[s + k]
            mask_union |= m
            mask_intersection &= m
        has_lift = bool(mask_union & 1)
        has_sr = bool(mask_union & 2)
        has_tp = bool(mask_union & 4)
        if has_lift and has_sr and has_tp:
            start_sources["all_three"] += 1
        elif has_lift and has_sr:
            start_sources["lift_and_support"] += 1
        elif has_lift and has_tp:
            start_sources["lift_and_progress"] += 1
        elif has_sr and has_tp:
            start_sources["support_and_progress"] += 1
        elif has_lift:
            start_sources["lift_only"] += 1
        elif has_sr:
            start_sources["support_removed_only"] += 1
        elif has_tp:
            start_sources["target_progress_only"] += 1

    # Per-start component co-occurrence (any step in window has it)
    start_cooccur = {"lift": 0, "support_removed": 0, "target_progress": 0}
    for s in feasible_starts:
        mask_union = 0
        for k in range(K):
            mask_union |= bitmask[s + k]
        if mask_union & 1: start_cooccur["lift"] += 1
        if mask_union & 2: start_cooccur["support_removed"] += 1
        if mask_union & 4: start_cooccur["target_progress"] += 1

    # No-feasible reason
    no_reason = "N/A"
    if not has_feas:
        if tgn < TASK_ROLE_MIN:
            no_reason = "non_gripper_task"
        elif funnel["candidate_close"] == 0:
            no_reason = "no_close_segments"
        elif funnel["stable_grasp"] == 0:
            no_reason = "no_stable_grasp"
        elif funnel["critical"] == 0:
            no_reason = "no_critical_steps"
        else:
            no_reason = "critical_but_no_K10_contiguous"

    # Step labels (include window_id, student_valid for auditor)
    step_labels = []
    for i, r in enumerate(records):
        cm = r["component_valid_mask"]
        step_labels.append({
            "step": i, "episode_key": cid,
            "candidate_close": r["candidate_close"],
            "known_mask": r["known_mask"],
            "student_valid": r["student_valid"],
            "window_id": r.get("window_id", ""),
            "critical_t": critical[i],
            "burst_feasible_t": burst[i] if i < n - K + 1 else False,
            "is_feasible_start": is_start[i] if i < n - K + 1 else False,
            "component_bitmask": bitmask[i],
            "release_risk_valid": cm.get("release_risk", False),
            "regrasp_risk_valid": cm.get("regrasp_or_instability_risk", False),
            "teacher_reason_code": reasons[i],
        })

    return {
        "identity": cid, "suite": suite, "task_idx": task, "state_id": state,
        "fold_id": state // 5, "n_steps": n,
        "has_feasible_k10": has_feas,
        "feasible_start_count": len(feasible_starts),
        "first_feasible_start": feasible_starts[0] if has_feas else -1,
        "last_feasible_start": feasible_starts[-1] if has_feas else -1,
        "no_feasible_reason": no_reason,
        "task_grasp_necessity": tgn,
        "n_critical_steps": funnel["critical"],
        "component_funnel": dict(funnel),
        "start_sources": dict(start_sources),
        "start_cooccur": start_cooccur,
        "step_labels": step_labels,
    }


# ── audit with REAL gate computation ───────────────────────────────────
def audit_results(results: list[dict]) -> dict:
    n_seg_cross = 0; n_k10_oob = 0; n_unknown_pos = 0; n_invalid_pos = 0

    for ep in results:
        labels = ep["step_labels"]
        n = ep["n_steps"]

        # Build window_id index for segment crossing check
        wids = [lab.get("window_id", "") for lab in labels]

        for i, lab in enumerate(labels):
            if not lab["burst_feasible_t"]:
                continue

            if i >= n - K + 1:
                n_k10_oob += 1

            if not lab["known_mask"]:
                n_unknown_pos += 1

            if not lab["student_valid"]:
                n_invalid_pos += 1

            # Segment crossing: all K steps must share non-empty window_id
            wid = wids[i]
            if not wid:
                n_seg_cross += 1
            else:
                for k in range(1, K):
                    if i + k >= n or wids[i + k] != wid:
                        n_seg_cross += 1
                        break

    per_task = defaultdict(lambda: {"eps": 0, "feas": 0, "starts": 0, "no_corridor": 0,
                                     "funnel": defaultdict(int), "sources": defaultdict(int)})
    suite_funnel = defaultdict(lambda: defaultdict(int))
    suite_sources = defaultdict(lambda: defaultdict(int))
    suite_cooccur = defaultdict(lambda: defaultdict(int))

    for ep in results:
        tk = f"{ep['suite']}/t{ep['task_idx']:02d}"
        sk = ep["suite"]
        per_task[tk]["eps"] += 1
        if ep["has_feasible_k10"]:
            per_task[tk]["feas"] += 1
            per_task[tk]["starts"] += ep["feasible_start_count"]
        else:
            per_task[tk]["no_corridor"] += 1
        for k, v in ep.get("component_funnel", {}).items():
            per_task[tk]["funnel"][k] += v
            suite_funnel[sk][k] += v
        for k, v in ep.get("start_sources", {}).items():
            per_task[tk]["sources"][k] += v
            suite_sources[sk][k] += v
        for k, v in ep.get("start_cooccur", {}).items():
            suite_cooccur[sk][k] += v

    return {
        "n_episodes": len(results),
        "n_feasible": sum(1 for e in results if e["has_feasible_k10"]),
        "total_starts": sum(e["feasible_start_count"] for e in results),
        "segment_crossing": n_seg_cross,
        "k10_oob": n_k10_oob,
        "unknown_in_positive": n_unknown_pos,
        "student_invalid_in_positive": n_invalid_pos,
        "gates": {
            "identity_closure": len(results) == 800,
            "segment_crossing_zero": n_seg_cross == 0,
            "k10_oob_zero": n_k10_oob == 0,
            "unknown_in_positive_zero": n_unknown_pos == 0,
            "invalid_in_positive_zero": n_invalid_pos == 0,
        },
        "per_task": {k: dict(v) for k, v in sorted(per_task.items())},
        "suite_funnel": {k: dict(v) for k, v in sorted(suite_funnel.items())},
        "suite_sources": {k: dict(v) for k, v in sorted(suite_sources.items())},
        "suite_cooccur": {k: dict(v) for k, v in sorted(suite_cooccur.items())},
    }


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    validate_all(args.teacher_root)

    teacher = args.teacher_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)

    scope = [(s, t, st) for s in SUITES for t in range(N_TASKS) for st in FIT_STATES]
    all_eps = []

    for suite, task, state in scope:
        ep = process(teacher, suite, task, state)
        if ep is None:
            continue
        ident_out = out / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "k10_labels_v122_v21c.jsonl", "w", encoding="utf-8") as fh:
            for lab in ep["step_labels"]:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")
        all_eps.append(ep)

    aud = audit_results(all_eps)

    print(f"=== V1.2.1 AUDIT ===")
    print(f"Episodes: {aud['n_episodes']}")
    print(f"Feasible K10: {aud['n_feasible']}")
    print(f"Total starts: {aud['total_starts']}")
    for g, s in aud["gates"].items():
        print(f"  {g}: {'PASS' if s else 'FAIL'}")
    if aud["segment_crossing"] > 0:
        print(f"  ** segment_crossing={aud['segment_crossing']}")
    if aud["student_invalid_in_positive"] > 0:
        print(f"  ** student_invalid_in_positive={aud['student_invalid_in_positive']}")

    # Component funnel
    print(f"\n=== COMPONENT FUNNEL ===")
    stages = ["candidate_close", "task_role", "stable_grasp",
              "lift_pass", "support_removed_pass", "target_progress_pass",
              "release_veto", "release_unknown_veto", "regrasp_veto",
              "regrasp_unknown_veto", "critical"]
    header = f"{'Suite':<16}" + "".join(f" {s:>14}" for s in stages)
    print(header)
    for sk in SUITES:
        sf = aud["suite_funnel"].get(sk, {})
        print(f"{sk:<16}" + "".join(f" {sf.get(s, 0):>14}" for s in stages))

    # Start sources (from bitmask union)
    print(f"\n=== START SOURCES (bitmask union over K=10 window) ===")
    srcs = ["lift_only", "support_removed_only", "target_progress_only",
            "lift_and_support", "lift_and_progress", "support_and_progress", "all_three"]
    header2 = f"{'Suite':<16}" + "".join(f" {s:>22}" for s in srcs)
    print(header2)
    for sk in SUITES:
        ss = aud["suite_sources"].get(sk, {})
        print(f"{sk:<16}" + "".join(f" {ss.get(s, 0):>22}" for s in srcs))

    # Component co-occurrence
    print(f"\n=== COMPONENT CO-OCCURRENCE (fraction of starts with component) ===")
    co = ["lift", "support_removed", "target_progress"]
    header3 = f"{'Suite':<16}" + "".join(f" {c:>20}" for c in co)
    print(header3)
    total_starts = aud["total_starts"]
    for sk in SUITES:
        sc = aud["suite_cooccur"].get(sk, {})
        suite_starts = aud["suite_sources"].get(sk, {}).get("lift_only", 0) + \
                       aud["suite_sources"].get(sk, {}).get("support_removed_only", 0) + \
                       aud["suite_sources"].get(sk, {}).get("target_progress_only", 0) + \
                       aud["suite_sources"].get(sk, {}).get("lift_and_support", 0) + \
                       aud["suite_sources"].get(sk, {}).get("lift_and_progress", 0) + \
                       aud["suite_sources"].get(sk, {}).get("support_and_progress", 0) + \
                       aud["suite_sources"].get(sk, {}).get("all_three", 0)
        row = f"{sk:<16}"
        for c_name in co:
            val = sc.get(c_name, 0)
            pct = f"{val}/{suite_starts}" if suite_starts > 0 else "0/0"
            row += f" {pct:>20}"
        print(row)

    # Per-task
    print(f"\nSuite/Task:")
    for tk, c in sorted(aud["per_task"].items()):
        print(f"  {tk}: {c['feas']}/{c['eps']} feasible, {c['starts']} starts")

    # ── Write all outputs before seal ──
    protocol = {
        "schema": SCHEMA, "K": K,
        "teacher_sha256sums": TEACHER_SHA,
        "thresholds": {
            "stable_grasp_min": STABLE_GRASP_MIN, "lift_min": LIFT_MIN,
            "support_removed_min": SUPPORT_REMOVED_MIN,
            "target_progress_min": TARGET_PROGRESS_MIN,
            "release_risk_max": RELEASE_RISK_MAX,
            "regrasp_risk_max": REGRASP_RISK_MAX,
            "task_role_min": TASK_ROLE_MIN,
        },
    }
    with open(out / "PROTOCOL.json", "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2)
    with open(out / "AUDIT.json", "w", encoding="utf-8") as fh:
        json.dump(aud, fh, indent=2)

    # SOURCE_BINDING with actual git commit
    head = git_head()
    blob_sha = sha256_file(Path(__file__))
    with open(out / "SOURCE_BINDING.json", "w", encoding="utf-8") as fh:
        json.dump({
            "teacher_sha256sums": TEACHER_SHA,
            "teacher_source_commit": TEACHER_COMMIT,
            "teacher_source": "V2.1C (canonical action, raw < 0.5 = CLOSE)",
            "K": K, "labeler_schema": SCHEMA,
            "git_commit": head,
            "labeler_blob_sha256": blob_sha,
        }, fh, indent=2)

    with open(out / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "teacher_source": "Physics Teacher V2.1C (canonical action)",
                    "teacher_sha256sums": TEACHER_SHA, "teacher_commit": TEACHER_COMMIT}, fh, indent=2)

    # EPISODE_SUMMARY.csv
    with open(out / "EPISODE_SUMMARY.csv", "w", encoding="utf-8") as fh:
        fh.write("identity,suite,task_idx,state_id,fold_id,has_feasible_k10,feasible_start_count,first_feasible,last_feasible,no_feasible_reason,task_grasp_necessity,n_critical_steps\n")
        for ep in all_eps:
            fh.write(f"{ep['identity']},{ep['suite']},{ep['task_idx']},{ep['state_id']},{ep['fold_id']},{ep['has_feasible_k10']},{ep['feasible_start_count']},{ep['first_feasible_start']},{ep['last_feasible_start']},{ep['no_feasible_reason']},{ep['task_grasp_necessity']},{ep['n_critical_steps']}\n")

    # TASK_GEOMETRY.csv
    with open(out / "TASK_GEOMETRY.csv", "w", encoding="utf-8") as fh:
        stages_hdr = ["candidate_close", "task_role", "stable_grasp", "lift_pass",
                      "support_removed_pass", "target_progress_pass", "release_veto",
                      "release_unknown_veto", "regrasp_veto", "regrasp_unknown_veto", "critical"]
        fh.write("task,episodes,feasible,starts,no_corridor," + ",".join(stages_hdr) + "\n")
        for tk, c in sorted(aud["per_task"].items()):
            f_stages = ",".join(str(c["funnel"].get(s, 0)) for s in stages_hdr)
            fh.write(f"{tk},{c['eps']},{c['feas']},{c['starts']},{c['no_corridor']},{f_stages}\n")

    # ── Seal ──
    SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payload_files = sorted(
        [f for f in out.rglob("*") if f.is_file() and f.name not in SEAL_FILES],
        key=lambda f: str(f.relative_to(out)))
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in payload_files:
            rel = str(fp.relative_to(out))
            fh.write(f"{sha256_file(fp)}  {rel}\n")
    sha = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    print(f"\nRoot: {out}\nSHA256SUMS: {sha}\nLabeler commit: {head}")
    all_pass = all(aud["gates"].values())
    print(f"ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
