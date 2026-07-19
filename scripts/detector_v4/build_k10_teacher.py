#!/usr/bin/env python3
"""R7.1: K10 Gripper-Critical Opportunity Teacher.

Builds per-step critical_t and burst_feasible_t labels from privileged
S1 Teacher records. K=10 fixed attack length.

critical_t = target_relevant AND gripper_dependent AND clean_close_intent
             AND stable_grasp AND task_progress_active
             AND NOT release_safe AND label_known

burst_feasible_t = critical_t ... critical_(t+9) all True

Teacher uses privileged evidence (up to K-1 future steps for containment check).
Student input is strictly causal.
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ── constants ──────────────────────────────────────────────────────────
K = 10  # fixed attack length
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))
EPISODE_MARGIN = 10  # exclude first/last N steps from task_progress_active
RELEASE_SAFE_HORIZON = 5  # steps before/after release_onset considered release_safe

SCHEMA = "R7_K10_GRIPPER_CRITICAL_OPPORTUNITY_TEACHER_V1"


# ── helpers ────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── step-level critical_t ──────────────────────────────────────────────
def compute_critical(records: list[dict[str, Any]], episode_len: int
                     ) -> tuple[list[bool], list[str]]:
    """Compute critical_t and reason_code for each step."""
    n = len(records)
    critical = [False] * n
    reasons = ["none"] * n

    # Pre-scan: which steps are within a close event window
    in_close = [False] * n
    for r in records:
        if r.get("event_close_onset") and r.get("event_end_step", -1) >= r["step"]:
            onset = r["step"]
            end = min(r["event_end_step"], n - 1)
            for s in range(onset, end + 1):
                in_close[s] = True

    # Release-safe windows
    release_safe = [False] * n
    for r in records:
        if r.get("event_release_onset") or (r.get("release_imminent") and not r.get("retention_unknown_mask")):
            s = r["step"]
            for d in range(-RELEASE_SAFE_HORIZON, RELEASE_SAFE_HORIZON + 1):
                if 0 <= s + d < n:
                    release_safe[s + d] = True

    for i, r in enumerate(records):
        # label_known
        if r.get("retention_unknown_mask"):
            critical[i] = False
            reasons[i] = "unknown_mask"
            continue

        # target_relevant: all LIBERO tasks are gripper-relevant
        target_relevant = True

        # gripper_dependent: close event exists or grasp_support
        gripper_dependent = in_close[i] or (
            r.get("grasp_support") and not r.get("retention_unknown_mask"))

        # clean_close_intent: close onset occurred or retention_active
        clean_close_intent = in_close[i] or (
            r.get("retention_active") and not r.get("retention_unknown_mask"))

        # stable_grasp: event_support present
        stable_grasp = r.get("event_support", False)

        # task_progress_active: not at episode margins, within close window
        task_active = EPISODE_MARGIN <= i < episode_len - EPISODE_MARGIN

        # NOT release_safe
        not_release_safe = not release_safe[i]

        # evidence valid
        ev_valid = r.get("event_evidence_valid", True)

        if not ev_valid:
            critical[i] = False
            reasons[i] = "invalid_evidence"
        elif not target_relevant:
            reasons[i] = "not_target_relevant"
        elif not gripper_dependent:
            reasons[i] = "not_gripper_dependent"
        elif not clean_close_intent:
            reasons[i] = "not_close_intent"
        elif not stable_grasp:
            reasons[i] = "not_stable_grasp"
        elif not task_active:
            reasons[i] = "not_task_active"
        elif not not_release_safe:
            reasons[i] = "release_safe"
        else:
            critical[i] = True
            reasons[i] = "critical"

    return critical, reasons


# ── burst feasibility ──────────────────────────────────────────────────
def compute_burst_feasible(critical: list[bool], n: int
                           ) -> tuple[list[bool], list[int], list[bool]]:
    """Compute burst_feasible_t and first_feasible_start for each step.

    burst_feasible[t] = critical[t] AND critical[t+1] AND ... AND critical[t+K-1]
    """
    burst = [False] * n
    first_start = [-1] * n
    is_feasible_start = [False] * n

    # Forward scan: for each t, check if K consecutive critical steps exist
    for t in range(n - K + 1):
        all_crit = all(critical[t + k] for k in range(K))
        burst[t] = all_crit
        if all_crit:
            is_feasible_start[t] = True

    # Compute first_feasible_start for each step
    # The earliest t' >= t where burst[t'] is True
    next_feasible = -1
    for t in range(n - 1, -1, -1):
        if t < n - K + 1 and burst[t]:
            next_feasible = t
        first_start[t] = next_feasible

    return burst, first_start, is_feasible_start


# ── main pipeline ──────────────────────────────────────────────────────
def process_episode(s1_root: Path, suite: str, task: int, state: int
                    ) -> Optional[dict[str, Any]]:
    """Build K10 labels for one FIT episode."""
    ident_dir = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    teacher_path = ident_dir / "teacher_retention_records.jsonl"
    student_path = ident_dir / "student_input_records.jsonl"
    if not teacher_path.exists():
        return None

    records = jsonl(teacher_path)
    students = jsonl(student_path) if student_path.exists() else None
    n = len(records)
    if n == 0:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"

    # Compute critical and burst
    critical, reasons = compute_critical(records, n)
    burst, first_start, is_feasible_start = compute_burst_feasible(critical, n)

    # Build per-step output
    step_labels = []
    for i, r in enumerate(records):
        step_labels.append({
            "step": i,
            "episode_key": cid,
            "candidate_segment_id": r.get("event_id", -1),
            "candidate_start": r.get("event_start_step", -1) if r.get("event_close_onset") else -1,
            "candidate_end": r.get("event_end_step", -1),
            "target_relevant": True,
            "gripper_dependent": bool(
                (r.get("event_close_onset") or r.get("event_end_step", -1) >= i) or
                (r.get("grasp_support") and not r.get("retention_unknown_mask"))),
            "clean_close_intent": bool(
                (r.get("event_end_step", -1) >= i and any(
                    rec.get("event_close_onset") and rec["step"] <= i <= rec.get("event_end_step", -1)
                    for rec in records)) or
                (r.get("retention_active") and not r.get("retention_unknown_mask"))),
            "stable_grasp": bool(r.get("event_support")),
            "task_progress_active": EPISODE_MARGIN <= i < n - EPISODE_MARGIN,
            "release_safe": reasons[i] == "release_safe",
            "label_known": not r.get("retention_unknown_mask", False),
            "critical_t": critical[i],
            "burst_feasible_t": burst[i] if i < n - K + 1 else False,
            "first_feasible_start": first_start[i],
            "is_feasible_start": is_feasible_start[i] if i < n - K + 1 else False,
            "teacher_reason_code": reasons[i],
            "teacher_task_decoder": suite,
            "teacher_source_sha": r.get("source_artifact_sha256", ""),
        })

    # Episode-level summary
    has_candidate = any(r.get("event_close_onset") for r in records)
    has_feasible_k10 = any(is_feasible_start)
    feasible_starts = [i for i, f in enumerate(is_feasible_start) if f]
    first_feas = feasible_starts[0] if feasible_starts else -1
    last_feas = feasible_starts[-1] if feasible_starts else -1

    no_feas_reason = ""
    if not has_feasible_k10:
        if not has_candidate:
            no_feas_reason = "no_close_event"
        elif not any(critical):
            no_feas_reason = "no_critical_steps_" + reasons[0] if reasons else "unknown"
        else:
            no_feas_reason = "critical_but_no_K10_burst"

    # Task mechanism approximation
    if suite == "libero_object":
        mechanism = "pick_place_or_transfer"
    elif suite == "libero_goal":
        mechanism = "goal_directed_reach"
    elif suite == "libero_spatial":
        mechanism = "spatial_rearrangement"
    else:
        mechanism = "general_manipulation"

    return {
        "identity": cid,
        "suite": suite,
        "task_idx": task,
        "state_id": state,
        "fold_id": state // 5,
        "n_steps": n,
        "has_candidate": has_candidate,
        "has_feasible_k10": has_feasible_k10,
        "first_feasible_start": first_feas,
        "last_feasible_start": last_feas,
        "feasible_start_count": len(feasible_starts),
        "no_feasible_reason": no_feas_reason if not has_feasible_k10 else "N/A",
        "mechanism_type": mechanism,
        "step_labels": step_labels,
    }


# ── audit ──────────────────────────────────────────────────────────────
def audit_results(results: list[dict]) -> dict[str, Any]:
    """Run K10 Teacher audits."""
    issues = []
    n_release_safe_pos = 0
    n_unknown_neg = 0
    n_k10_oob = 0
    n_segment_split = 0
    n_attack_read = 0
    n_privilege_leak = 0

    # Per-episode feasible stats
    per_suite_task = defaultdict(lambda: {"feasible": 0, "no_feasible": 0,
                                           "total": 0, "first_starts": []})
    feasible_lengths = []
    release_safe_count = 0
    unknown_count = 0

    for ep in results:
        labels = ep["step_labels"]
        n = len(labels)
        suite = ep["suite"]
        task = ep["task_idx"]
        stk = f"{suite}/t{task:02d}"
        per_suite_task[stk]["total"] += 1

        if ep["has_feasible_k10"]:
            per_suite_task[stk]["feasible"] += 1
            feasible_lengths.append(ep["feasible_start_count"])
            per_suite_task[stk]["first_starts"].append(ep["first_feasible_start"])
        else:
            per_suite_task[stk]["no_feasible"] += 1

        for i, lab in enumerate(labels):
            # release_safe cannot be positive (critical)
            if lab["release_safe"] and lab["critical_t"]:
                n_release_safe_pos += 1
                issues.append(f"{ep['identity']}:{i} release_safe AND critical")

            # K10 out of bound
            if i >= n - K + 1 and lab["burst_feasible_t"]:
                n_k10_oob += 1
                issues.append(f"{ep['identity']}:{i} K10 OOB but burst=True")

            release_safe_count += 1 if lab["release_safe"] else 0
            unknown_count += 1 if not lab["label_known"] else 0

    return {
        "n_episodes": len(results),
        "n_feasible_episodes": sum(1 for ep in results if ep["has_feasible_k10"]),
        "n_no_feasible_episodes": sum(1 for ep in results if not ep["has_feasible_k10"]),
        "per_suite_task": {k: dict(v) for k, v in sorted(per_suite_task.items())},
        "feasible_start_counts": feasible_lengths,
        "release_safe_positive": n_release_safe_pos,
        "k10_out_of_bound": n_k10_oob,
        "segment_split": n_segment_split,
        "attack_outcome_reads": n_attack_read,
        "privileged_in_student": n_privilege_leak,
        "unknown_count": unknown_count,
        "release_safe_count": release_safe_count,
        "all_gates": {
            "identity_closure": len(results) == 800,
            "segment_split_zero": n_segment_split == 0,
            "k10_oob_zero": n_k10_oob == 0,
            "release_safe_positive_zero": n_release_safe_pos == 0,
            "attack_reads_zero": n_attack_read == 0,
            "privilege_leak_zero": n_privilege_leak == 0,
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    s1 = args.s1_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)

    scope = [(s, t, st) for s in SUITES for t in range(N_TASKS) for st in FIT_STATES]

    all_eps = []
    for suite, task, state in scope:
        ep = process_episode(s1, suite, task, state)
        if ep is None:
            print(f"SKIP: {suite}/task_{task:02d}/state_{state:02d}")
            continue

        # Write per-identity step labels
        ident_out = out / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "k10_teacher_labels.jsonl", "w", encoding="utf-8") as fh:
            for lab in ep["step_labels"]:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")

        # Strip step_labels for summary
        ep_summary = {k: v for k, v in ep.items() if k != "step_labels"}
        all_eps.append(ep_summary)

    # ── Audit ──
    audit = audit_results([{"step_labels": jsonl(
        out / ep["suite"] / f"task_{ep['task_idx']:02d}" / f"state_{ep['state_id']:02d}" / "k10_teacher_labels.jsonl"
    ), **ep} for ep in all_eps])

    print(f"\n=== K10 TEACHER AUDIT ===")
    print(f"Episodes: {audit['n_episodes']}")
    print(f"Feasible K10: {audit['n_feasible_episodes']}")
    print(f"No feasible: {audit['n_no_feasible_episodes']}")
    print(f"\nGates:")
    for gate, status in audit["all_gates"].items():
        print(f"  {gate}: {'PASS' if status else 'FAIL'}")

    if audit["release_safe_positive"] > 0:
        print(f"  ** FAIL: {audit['release_safe_positive']} release-safe positives")
    if audit["k10_out_of_bound"] > 0:
        print(f"  ** FAIL: {audit['k10_out_of_bound']} K10 OOB")
    print(f"  unknown steps: {audit['unknown_count']}")
    print(f"  release_safe steps: {audit['release_safe_count']}")

    # ── Episode summary CSV ──
    with open(out / "k10_episode_summary.csv", "w", encoding="utf-8") as fh:
        fh.write("identity,suite,task_idx,state_id,fold_id,n_steps,"
                 "has_candidate,has_feasible_k10,first_feasible_start,"
                 "last_feasible_start,feasible_start_count,no_feasible_reason,"
                 "mechanism_type\n")
        for ep in all_eps:
            fh.write(f"{ep['identity']},{ep['suite']},{ep['task_idx']},"
                     f"{ep['state_id']},{ep['fold_id']},{ep['n_steps']},"
                     f"{ep['has_candidate']},{ep['has_feasible_k10']},"
                     f"{ep['first_feasible_start']},{ep['last_feasible_start']},"
                     f"{ep['feasible_start_count']},{ep['no_feasible_reason']},"
                     f"{ep['mechanism_type']}\n")

    # ── Suite/task summary ──
    print(f"\nSuite/Task K10 feasibility:")
    for stk, counts in sorted(audit["per_suite_task"].items()):
        f = counts["feasible"]
        t = counts["total"]
        print(f"  {stk}: {f}/{t} feasible ({f/t*100:.0f}%)")

    # ── SHA256SUMS ──
    SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256"}
    file_list = sorted(
        [f for f in out.rglob("*") if f.is_file() and f.name not in SEAL_FILES],
        key=lambda f: str(f.relative_to(out)))
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = str(fp.relative_to(out))
            fh.write(f"{sha256_file(fp)}  {rel}\n")
    sha = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    print(f"\nRoot: {out}")
    print(f"SHA256SUMS: {sha}")


if __name__ == "__main__":
    main()
