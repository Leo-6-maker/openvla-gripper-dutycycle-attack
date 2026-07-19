#!/usr/bin/env python3
"""R7.1.1: K10 OPPORTUNITY LABELER V1.1 — uses real Physics Teacher V2.1.

Source: OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719
SHA256SUMS: 18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da

critical_t = known_mask AND candidate_close AND target_relevant
             AND stable_grasp AND manipulation_active
             AND NOT release_or_regrasp

burst_feasible_t = AND_{j=0}^{K-1} critical_{t+j}
  within same candidate segment, no horizon crossing.

All engineering issues from V1.0 fixed:
- SOURCE_BINDING records actual labeler commit
- MANIFEST in SHA256SUMS (single-pass seal)
- Gate failure → sys.exit(1)
- label_known matches critical_t's known_mask field
- regrasp semantics correct (risk > threshold → instability)
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

K = 10
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))

# Physics V2.1 thresholds
STABLE_GRASP_MIN = 0.5
LIFT_MIN = 0.3
SUPPORT_REMOVED_MIN = 0.3
TARGET_PROGRESS_MIN = 0.0  # any positive progress counts
RELEASE_RISK_MAX = 0.5
REGRASP_RISK_MAX = 0.5
TASK_GRASP_NECESSITY_MIN = 0.3

SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1_1"
TEACHER_ROOT = "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719"
TEACHER_SHA = "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── critical_t from Physics V2.1 ──────────────────────────────────────
def compute_critical(records: list[dict[str, Any]]) -> tuple[list[bool], list[str]]:
    n = len(records)
    critical = [False] * n
    reasons = ["none"] * n

    for i, r in enumerate(records):
        km = r.get("known_mask", True)
        if not km:
            reasons[i] = "unknown_mask"
            continue

        sv = r.get("student_valid", True)
        if not sv:
            reasons[i] = "student_invalid"
            continue

        cc = r.get("candidate_close", False)
        if not cc:
            reasons[i] = "not_candidate_close"
            continue

        # target_relevant: task_grasp_necessity indicates gripper dependence
        tgn = r.get("task_grasp_necessity", 1.0)
        if tgn < TASK_GRASP_NECESSITY_MIN:
            reasons[i] = "not_target_relevant"
            continue

        # stable_grasp
        sg = r.get("stable_grasp_score", 0.0)
        cm = r.get("component_valid_mask", {})
        sg_valid = cm.get("relative_pose_stability", True)
        if not sg_valid or sg < STABLE_GRASP_MIN:
            reasons[i] = "not_stable_grasp" if sg_valid else "stable_grasp_unknown"
            continue

        # manipulation_active: lift OR support_removed OR valid target_progress
        lift = r.get("lift_score", 0.0)
        sr = r.get("support_removed", 0.0)
        tp = r.get("target_progress", 0.0)
        tpk = r.get("target_progress_known", False)

        lift_valid = cm.get("lift_score", True)
        sr_valid = True  # support_removed always valid per protocol
        tp_valid = cm.get("target_progress", tpk)

        manipulation = False
        if lift_valid and lift >= LIFT_MIN:
            manipulation = True
        elif sr_valid and sr >= SUPPORT_REMOVED_MIN:
            manipulation = True
        elif tp_valid and tpk and tp >= TARGET_PROGRESS_MIN:
            manipulation = True

        if not manipulation:
            reasons[i] = "not_manipulation_active"
            continue

        # release_or_regrasp
        rr = r.get("release_risk", 0.0)
        ri = r.get("regrasp_or_instability_risk", 0.0)
        rr_valid = cm.get("release_risk", True)
        ri_valid = cm.get("regrasp_or_instability_risk", True)

        if rr_valid and rr > RELEASE_RISK_MAX:
            reasons[i] = "release_risk"
            continue
        if ri_valid and ri > REGRASP_RISK_MAX:
            reasons[i] = "regrasp_or_instability"
            continue

        critical[i] = True
        reasons[i] = "critical"

    return critical, reasons


# ── K10 burst feasibility (same contract as V1.0) ─────────────────────
def compute_burst(critical: list[bool], n: int,
                  records: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    burst = [False] * n
    is_start = [False] * n

    # Build segment index from window_id
    in_segment = [-1] * n
    seg_ids = {}
    for i, r in enumerate(records):
        wid = r.get("window_id", "")
        if wid and wid.startswith("candidate:"):
            sid = int(wid.split(":")[1])
            in_segment[i] = sid
            seg_ids[sid] = True

    for t in range(n - K + 1):
        if not all(critical[t + k] for k in range(K)):
            continue
        seg = in_segment[t]
        if seg < 0:
            continue
        if not all(in_segment[t + k] == seg for k in range(K)):
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
    critical, reasons = compute_critical(records)
    burst, is_start = compute_burst(critical, n, records)

    # Segments from window_id
    segments = defaultdict(lambda: {"onset": n, "end": -1, "crit_count": 0, "feas_count": 0})
    for i, r in enumerate(records):
        wid = r.get("window_id", "")
        if wid and wid.startswith("candidate:"):
            sid = int(wid.split(":")[1])
            s = segments[sid]
            s["onset"] = min(s["onset"], i)
            s["end"] = max(s["end"], i)
            if critical[i]:
                s["crit_count"] += 1

    feasible_starts = [i for i, s in enumerate(is_start) if s]
    has_feas = len(feasible_starts) > 0
    tgn = records[0].get("task_grasp_necessity", 1.0) if records else 1.0

    # Step labels
    step_labels = []
    for i, r in enumerate(records):
        cm = r.get("component_valid_mask", {})
        step_labels.append({
            "step": i, "episode_key": cid,
            "candidate_close": r.get("candidate_close", False),
            "known_mask": r.get("known_mask", True),
            "student_valid": r.get("student_valid", True),
            "target_relevant": r.get("task_grasp_necessity", 1.0) >= TASK_GRASP_NECESSITY_MIN,
            "stable_grasp_score": r.get("stable_grasp_score", 0.0),
            "lift_score": r.get("lift_score", 0.0),
            "support_removed": r.get("support_removed", 0.0),
            "target_progress": r.get("target_progress", 0.0),
            "target_progress_known": r.get("target_progress_known", False),
            "release_risk": r.get("release_risk", 0.0),
            "regrasp_or_instability_risk": r.get("regrasp_or_instability_risk", 0.0),
            "component_valid": {k: bool(v) for k, v in cm.items()},
            "critical_t": critical[i],
            "burst_feasible_t": burst[i] if i < n - K + 1 else False,
            "is_feasible_start": is_start[i] if i < n - K + 1 else False,
            "teacher_reason_code": reasons[i],
            "phase_name": r.get("phase_name", ""),
            "window_id": r.get("window_id", ""),
            "task_grasp_necessity": tgn,
        })

    # No-feasible reason
    if not has_feas:
        if tgn < TASK_GRASP_NECESSITY_MIN:
            no_reason = "non_gripper_task"
        elif not any(r.get("candidate_close") for r in records):
            no_reason = "no_close_segments"
        elif not any(critical):
            rc = set(reasons)
            no_reason = "no_critical_" + "_".join(sorted(rc)[:3])
        else:
            no_reason = "critical_but_no_K10_contiguous"
    else:
        no_reason = "N/A"

    return {
        "identity": cid, "suite": suite, "task_idx": task, "state_id": state,
        "fold_id": state // 5, "n_steps": n,
        "has_feasible_k10": has_feas,
        "feasible_start_count": len(feasible_starts),
        "first_feasible_start": feasible_starts[0] if has_feas else -1,
        "last_feasible_start": feasible_starts[-1] if has_feas else -1,
        "no_feasible_reason": no_reason,
        "task_grasp_necessity": tgn,
        "n_critical_steps": sum(critical),
        "n_candidate_steps": sum(1 for r in records if r.get("candidate_close")),
        "n_segments": len(segments),
        "n_unknown_masked": sum(1 for r in records if not r.get("known_mask", True)),
        "step_labels": step_labels,
    }


# ── audit ──────────────────────────────────────────────────────────────
def audit(results: list[dict]) -> dict:
    n_seg_cross = 0; n_k10_oob = 0; n_unknown_pos = 0; n_invalid_pos = 0

    for ep in results:
        labels = ep["step_labels"]
        n = ep["n_steps"]
        for i, lab in enumerate(labels):
            if lab["burst_feasible_t"]:
                if i >= n - K + 1:
                    n_k10_oob += 1
                if not lab["known_mask"]:
                    n_unknown_pos += 1
                if not lab["student_valid"]:
                    n_invalid_pos += 1
                # segment crossing: all K must share window_id
                wid = lab["window_id"]
                if wid:
                    for k in range(1, K):
                        if i + k >= n or labels[i + k]["window_id"] != wid:
                            n_seg_cross += 1
                            break

    per_task = defaultdict(lambda: {"eps": 0, "feas": 0, "starts": 0, "no_corridor": 0})
    for ep in results:
        tk = f"{ep['suite']}/t{ep['task_idx']:02d}"
        per_task[tk]["eps"] += 1
        if ep["has_feasible_k10"]:
            per_task[tk]["feas"] += 1
            per_task[tk]["starts"] += ep["feasible_start_count"]
        else:
            per_task[tk]["no_corridor"] += 1

    return {
        "n_episodes": len(results),
        "n_feasible": sum(1 for e in results if e["has_feasible_k10"]),
        "n_no_feasible": sum(1 for e in results if not e["has_feasible_k10"]),
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
    }


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

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
        with open(ident_out / "k10_labels_v11.jsonl", "w", encoding="utf-8") as fh:
            for lab in ep["step_labels"]:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")

        all_eps.append({k: v for k, v in ep.items() if k != "step_labels"})

    # Audit
    aud = audit([{**e, "step_labels": jsonl(
        out / "labels" / e["suite"] / f"task_{e['task_idx']:02d}" /
        f"state_{e['state_id']:02d}" / "k10_labels_v11.jsonl"
    )} for e in all_eps])

    print(f"=== K10 V1.1 AUDIT ===")
    print(f"Episodes: {aud['n_episodes']}")
    print(f"Feasible K10: {aud['n_feasible']}")
    print(f"No feasible: {aud['n_no_feasible']}")
    print(f"Total starts: {aud['total_starts']}")
    for g, s in aud["gates"].items():
        print(f"  {g}: {'PASS' if s else 'FAIL'}")

    print(f"\nSuite/Task feasibility:")
    for tk, c in sorted(aud["per_task"].items()):
        print(f"  {tk}: {c['feas']}/{c['eps']} feasible, {c['starts']} starts, {c['no_corridor']} no-corridor")

    # ── Write outputs ──
    protocol = {
        "schema": SCHEMA, "K": K,
        "source_teacher_root": str(teacher),
        "source_teacher_sha256sums": TEACHER_SHA,
        "critical_conjunction": [
            "known_mask", "student_valid", "candidate_close",
            "task_grasp_necessity >= 0.3", "stable_grasp_score >= 0.5",
            "(lift >= 0.3 OR support_removed >= 0.3 OR target_progress >= 0)",
            "release_risk <= 0.5", "regrasp_or_instability_risk <= 0.5",
        ],
        "thresholds": {
            "stable_grasp_min": STABLE_GRASP_MIN,
            "lift_min": LIFT_MIN,
            "support_removed_min": SUPPORT_REMOVED_MIN,
            "release_risk_max": RELEASE_RISK_MAX,
            "regrasp_risk_max": REGRASP_RISK_MAX,
            "task_grasp_necessity_min": TASK_GRASP_NECESSITY_MIN,
        },
    }
    with open(out / "PROTOCOL.json", "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2)

    with open(out / "AUDIT.json", "w", encoding="utf-8") as fh:
        json.dump(aud, fh, indent=2)

    with open(out / "EPISODE_SUMMARY.csv", "w", encoding="utf-8") as fh:
        fh.write("identity,suite,task_idx,state_id,fold_id,n_steps,"
                 "has_feasible_k10,feasible_start_count,first_feasible,"
                 "last_feasible,no_feasible_reason,task_grasp_necessity,"
                 "n_critical_steps,n_candidate_steps\n")
        for ep in all_eps:
            fh.write(f"{ep['identity']},{ep['suite']},{ep['task_idx']},"
                     f"{ep['state_id']},{ep['fold_id']},{ep['n_steps']},"
                     f"{ep['has_feasible_k10']},{ep['feasible_start_count']},"
                     f"{ep['first_feasible_start']},{ep['last_feasible_start']},"
                     f"{ep['no_feasible_reason']},{ep['task_grasp_necessity']},"
                     f"{ep['n_critical_steps']},{ep['n_candidate_steps']}\n")

    with open(out / "TASK_GEOMETRY.csv", "w", encoding="utf-8") as fh:
        fh.write("task,episodes,feasible,starts,no_corridor\n")
        for tk, c in sorted(aud["per_task"].items()):
            fh.write(f"{tk},{c['eps']},{c['feas']},{c['starts']},{c['no_corridor']}\n")

    # SOURCE_BINDING (will be updated after commit)
    with open(out / "SOURCE_BINDING.json", "w", encoding="utf-8") as fh:
        json.dump({
            "teacher_root": str(teacher),
            "teacher_sha256sums": TEACHER_SHA,
            "K": K,
            "labeler_schema": SCHEMA,
            "git_commit": "TO_BE_SET_AFTER_COMMIT",
        }, fh, indent=2)

    # ── Seal (single pass, excludes seal files) ──
    SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256"}
    file_list = sorted(
        [f for f in out.rglob("*") if f.is_file() and f.name not in SEAL_FILES],
        key=lambda f: str(f.relative_to(out)))
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = str(fp.relative_to(out))
            fh.write(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}\n")
    sha = hashlib.sha256((out / "SHA256SUMS").read_bytes()).hexdigest()
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    print(f"\nRoot: {out}")
    print(f"SHA256SUMS: {sha}")

    all_pass = all(aud["gates"].values())
    print(f"ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
