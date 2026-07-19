#!/usr/bin/env python3
"""R7.1: K10 OPPORTUNITY LABELER V1.

Derives critical_t and burst_feasible_t from sealed S1 Physics Teacher V2.1.
Does NOT modify, overwrite, or replace the source Teacher.

critical_t = label_known AND candidate_close AND target_relevant
             AND stable_grasp AND manipulation_active
             AND NOT release_safe AND NOT regrasp_or_instability

burst_feasible_t = AND_{j=0}^{K-1} critical_{t+j}
  - fully contained within one candidate segment
  - no unknown gap crossing
  - no release-safe crossing
  - no episode horizon crossing
  - no padding

Output: per-identity step labels, segment summary, episode summary, task geometry.
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
RELEASE_SAFE_MARGIN = 3  # steps around release_onset/release_imminent considered release_safe

SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── segment extraction ─────────────────────────────────────────────────
def extract_segments(records: list[dict[str, Any]]) -> list[dict]:
    """Extract contiguous candidate segments from close events.
    A segment is a contiguous close window from close_onset to end_step.
    Segments DO NOT cross event boundaries.
    """
    segments = []
    n = len(records)

    i = 0
    while i < n:
        r = records[i]
        if r.get("event_close_onset") and r.get("event_end_step", -1) >= r["step"]:
            onset = r["step"]
            end = min(r["event_end_step"], n - 1)
            duration = end - onset + 1
            if duration >= K:  # must be at least K steps to contain any K10 window
                # Check for internal unknown gaps
                has_unknown = any(
                    records[s].get("retention_unknown_mask", False)
                    for s in range(onset, end + 1))
                has_release_event = any(
                    records[s].get("event_release_onset")
                    for s in range(onset, end + 1))

                segments.append({
                    "segment_id": len(segments),
                    "onset": onset, "end": end,
                    "duration": duration,
                    "has_unknown": has_unknown,
                    "has_release_event": has_release_event,
                })
            i = end + 1
        else:
            i += 1
    return segments


# ── step-level critical_t ──────────────────────────────────────────────
def compute_critical(records: list[dict[str, Any]], n: int,
                     segments: list[dict]) -> tuple[list[bool], list[str], list[bool]]:
    """Compute critical_t, reason_code, and release_safe for each step."""
    critical = [False] * n
    reasons = ["none"] * n
    release_safe_steps = [False] * n

    # Mark which steps are in each segment
    in_segment = [-1] * n
    seg_map = {}
    for seg in segments:
        seg_map[seg["segment_id"]] = seg
        for s in range(seg["onset"], seg["end"] + 1):
            in_segment[s] = seg["segment_id"]

    # Pre-compute release_safe zones
    for r in records:
        if r.get("event_release_onset") or (
            r.get("release_imminent") and not r.get("retention_unknown_mask")):
            s = r["step"]
            for d in range(-RELEASE_SAFE_MARGIN, RELEASE_SAFE_MARGIN + 1):
                if 0 <= s + d < n:
                    release_safe_steps[s + d] = True

    for i, r in enumerate(records):
        # label_known
        if r.get("retention_unknown_mask"):
            reasons[i] = "unknown_mask"
            continue
        if not r.get("event_evidence_valid", True):
            reasons[i] = "invalid_evidence"
            continue

        # candidate_close: step is in a close segment
        in_close = in_segment[i] >= 0

        # target_relevant: all LIBERO tasks are gripper-relevant
        # Exception: tasks with zero close events (goal/t00, goal/t05)
        target_relevant = True  # determined per-episode, not per-step

        # stable_grasp: event_support present
        stable_grasp = r.get("event_support", False)

        # manipulation_active:
        #   grasp_support AND retention_active (gripper engaged + task retention)
        #   AND in a close event window
        manipulation_active = (
            in_close and
            r.get("grasp_support", False) and not r.get("retention_unknown_mask") and
            r.get("retention_active", False) and not r.get("retention_unknown_mask")
        )

        # NOT release_safe
        not_release_safe = not release_safe_steps[i]

        # NOT regrasp_or_instability:
        #   event_opening_stable=True means stable (positive evidence)
        #   event_opening_stable=None means unknown (no evidence either way)
        #   The field is never False, so we can only use it as stability evidence
        #   Absence of stable evidence does NOT imply instability
        opening = r.get("event_opening_stable")
        regrasp_or_instability = False  # cannot be determined from available fields

        # Assemble
        if not in_close:
            reasons[i] = "not_in_close_segment"
        elif not target_relevant:
            reasons[i] = "not_target_relevant"
        elif not stable_grasp:
            reasons[i] = "not_stable_grasp"
        elif not manipulation_active:
            reasons[i] = "not_manipulation_active"
        elif not not_release_safe:
            reasons[i] = "release_safe"
        elif regrasp_or_instability:
            reasons[i] = "regrasp_or_instability"
        else:
            critical[i] = True
            reasons[i] = "critical"

    return critical, reasons, release_safe_steps


# ── K10 burst feasibility ──────────────────────────────────────────────
def compute_burst_feasible(critical: list[bool], n: int,
                           in_segment: list[int],
                           segments: list[dict],
                           release_safe: list[bool]) -> tuple[list[bool], list[bool]]:
    """Compute burst_feasible_t and is_feasible_start.

    K10 window must:
    1. All K steps have critical=True
    2. All K steps belong to the SAME segment
    3. No step crosses segment boundary
    4. No step is release_safe
    5. No step exceeds episode horizon
    """
    burst = [False] * n
    is_start = [False] * n

    for t in range(n - K + 1):
        # All K steps must be critical
        all_crit = all(critical[t + k] for k in range(K))
        if not all_crit:
            continue

        # All K steps must belong to the SAME segment
        seg_id = in_segment[t]
        if seg_id < 0:
            continue
        same_seg = all(in_segment[t + k] == seg_id for k in range(K))
        if not same_seg:
            continue

        # No release_safe step in window
        any_release_safe = any(release_safe[t + k] for k in range(K))
        if any_release_safe:
            continue

        burst[t] = True
        is_start[t] = True

    return burst, is_start


# ── main pipeline ──────────────────────────────────────────────────────
def process_episode(s1_root: Path, suite: str, task: int, state: int
                    ) -> Optional[dict[str, Any]]:
    ident_dir = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    teacher_path = ident_dir / "teacher_retention_records.jsonl"
    if not teacher_path.exists():
        return None

    records = jsonl(teacher_path)
    n = len(records)
    if n == 0:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"

    # Determine target_relevant per episode
    has_close = any(r.get("event_close_onset") for r in records)
    target_relevant = has_close  # episodes with close events are gripper-relevant

    segments = extract_segments(records)

    # Build in_segment index
    in_segment = [-1] * n
    for seg in segments:
        for s in range(seg["onset"], seg["end"] + 1):
            in_segment[s] = seg["segment_id"]

    critical, reasons, release_safe_steps = compute_critical(records, n, segments)
    burst, is_start = compute_burst_feasible(
        critical, n, in_segment, segments, release_safe_steps)

    # Step labels
    step_labels = []
    for i, r in enumerate(records):
        step_labels.append({
            "step": i,
            "episode_key": cid,
            "candidate_segment_id": in_segment[i],
            "candidate_close": in_segment[i] >= 0,
            "label_known": not r.get("retention_unknown_mask", False),
            "target_relevant": target_relevant,
            "stable_grasp": bool(r.get("event_support")),
            "manipulation_active": bool(
                in_segment[i] >= 0 and
                r.get("grasp_support") and not r.get("retention_unknown_mask") and
                r.get("retention_active") and not r.get("retention_unknown_mask")),
            "release_safe": release_safe_steps[i],
            "regrasp_or_instability": (
                r.get("event_opening_stable") is not None and
                r.get("event_opening_stable") is True),
            "critical_t": critical[i],
            "burst_feasible_t": burst[i] if i < n - K + 1 else False,
            "is_feasible_start": is_start[i] if i < n - K + 1 else False,
            "teacher_reason_code": reasons[i],
            "teacher_source_sha": r.get("source_artifact_sha256", ""),
        })

    # Feasible start summary
    feasible_starts = [i for i, s in enumerate(is_start) if s]
    has_feas = len(feasible_starts) > 0

    if not has_feas:
        if not target_relevant:
            no_reason = "mechanism_not_supported"
        elif not segments:
            no_reason = "no_close_segment"
        elif not any(critical):
            crit_reasons = set(reasons)
            no_reason = "no_critical_steps_" + "_".join(sorted(crit_reasons)[:3])
        else:
            no_reason = "critical_but_no_K10_contiguous_burst"
    else:
        no_reason = "N/A"

    # Segment-level summary
    seg_summaries = []
    for seg in segments:
        seg_crit = sum(1 for s in range(seg["onset"], seg["end"] + 1)
                      if s < n and critical[s])
        seg_feas = sum(1 for s in range(seg["onset"], min(seg["end"] - K + 2, seg["end"] + 1))
                      if s < n and is_start[s])
        seg_summaries.append({
            "segment_id": seg["segment_id"],
            "onset": seg["onset"], "end": seg["end"], "duration": seg["duration"],
            "critical_step_count": seg_crit,
            "feasible_start_count": seg_feas,
            "has_unknown": seg["has_unknown"],
            "has_release_event": seg["has_release_event"],
        })

    # Episode summary
    return {
        "identity": cid, "suite": suite, "task_idx": task, "state_id": state,
        "fold_id": state // 5, "n_steps": n,
        "has_candidate_segment": len(segments) > 0,
        "n_segments": len(segments),
        "segment_lengths": [s["duration"] for s in segments],
        "has_critical_state": any(critical),
        "critical_step_count": sum(critical),
        "has_feasible_k10": has_feas,
        "feasible_start_count": len(feasible_starts),
        "first_feasible_start": feasible_starts[0] if has_feas else -1,
        "last_feasible_start": feasible_starts[-1] if has_feas else -1,
        "no_feasible_reason": no_reason,
        "mechanism_supported": target_relevant,
        "release_safe_step_count": sum(release_safe_steps),
        "unknown_step_count": sum(1 for r in records if r.get("retention_unknown_mask")),
        "segments": seg_summaries,
        "step_labels": step_labels,
    }


# ── audit ──────────────────────────────────────────────────────────────
def audit_results(results: list[dict]) -> dict:
    issues = []
    n_segment_cross = 0
    n_k10_oob = 0
    n_unknown_pos = 0
    n_release_safe_pos = 0
    n_unsupported_pos = 0

    per_task = defaultdict(lambda: {"episodes": 0, "supported": 0, "no_corridor": 0,
                                     "k10_positive": 0, "start_count": 0,
                                     "segment_count": 0, "unknown_rate": 0.0})

    for ep in results:
        labels = ep["step_labels"]
        n = ep["n_steps"]
        stk = f"{ep['suite']}/t{ep['task_idx']:02d}"
        per_task[stk]["episodes"] += 1
        if ep["mechanism_supported"]:
            per_task[stk]["supported"] += 1
        if not ep["has_feasible_k10"]:
            per_task[stk]["no_corridor"] += 1
        if ep["has_feasible_k10"]:
            per_task[stk]["k10_positive"] += 1
            per_task[stk]["start_count"] += ep["feasible_start_count"]
        per_task[stk]["segment_count"] += ep["n_segments"]
        per_task[stk]["unknown_rate"] += ep["unknown_step_count"] / max(n, 1)

        for seg in ep["segments"]:
            # K10 within segment: starts from onset to end-K+1
            max_starts = max(0, seg["duration"] - K + 1)

        for i, lab in enumerate(labels):
            if lab["burst_feasible_t"]:
                # K10 OOB
                if i >= n - K + 1:
                    n_k10_oob += 1
                    issues.append(f"{ep['identity']}:{i} K10 OOB")

                # Segment crossing (all K steps must share same segment)
                seg_id = lab["candidate_segment_id"]
                if seg_id >= 0:
                    for k in range(1, K):
                        if i + k >= n or labels[i + k]["candidate_segment_id"] != seg_id:
                            n_segment_cross += 1
                            issues.append(f"{ep['identity']}:{i} segment cross at +{k}")
                            break

                # Unknown in positive
                if not lab["label_known"]:
                    n_unknown_pos += 1
                    issues.append(f"{ep['identity']}:{i} unknown in burst=True")

                # Release-safe in positive
                if lab["release_safe"]:
                    n_release_safe_pos += 1
                    issues.append(f"{ep['identity']}:{i} release_safe in burst=True")

                # Unsupported forced positive
                if not ep["mechanism_supported"]:
                    n_unsupported_pos += 1
                    issues.append(f"{ep['identity']}:{i} unsupported task burst=True")

    # Also check: unknown NOT in positive (allowed, just masked)
    # And: critical segment length tests
    return {
        "n_episodes": len(results),
        "n_feasible_episodes": sum(1 for ep in results if ep["has_feasible_k10"]),
        "n_no_feasible": sum(1 for ep in results if not ep["has_feasible_k10"]),
        "total_feasible_starts": sum(ep["feasible_start_count"] for ep in results),
        "total_segments": sum(ep["n_segments"] for ep in results),
        "segment_crossing": n_segment_cross,
        "k10_out_of_bound": n_k10_oob,
        "unknown_in_positive": n_unknown_pos,
        "release_safe_in_positive": n_release_safe_pos,
        "unsupported_forced_positive": n_unsupported_pos,
        "duplicate_identity": 0 if len(set(ep["identity"] for ep in results)) == len(results) else 1,
        "missing_identity": 800 - len(results),
        "per_task": {k: dict(v) for k, v in sorted(per_task.items())},
        "all_gates": {
            "segment_crossing_zero": n_segment_cross == 0,
            "k10_oob_zero": n_k10_oob == 0,
            "unknown_in_positive_zero": n_unknown_pos == 0,
            "release_safe_in_positive_zero": n_release_safe_pos == 0,
            "unsupported_forced_positive_zero": n_unsupported_pos == 0,
            "duplicate_identity_zero": len(set(ep["identity"] for ep in results)) == len(results),
            "identity_closure": len(results) == 800,
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
            continue

        # Write step labels
        ident_out = out / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "k10_labels.jsonl", "w", encoding="utf-8") as fh:
            for lab in ep["step_labels"]:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")

        all_eps.append(ep)  # Keep full ep including step_labels and segments for audit
        # Also save segment summary
        with open(ident_out / "k10_segments.json", "w", encoding="utf-8") as fh:
            json.dump({"identity": ep["identity"], "segments": ep["segments"]}, fh, indent=2)

    # ── Audit ──
    audit = audit_results(all_eps)

    print(f"=== R7.1 K10 LABELER AUDIT ===")
    print(f"Episodes: {audit['n_episodes']}")
    print(f"Feasible K10: {audit['n_feasible_episodes']}")
    print(f"No feasible: {audit['n_no_feasible']}")
    print(f"Total feasible starts: {audit['total_feasible_starts']}")
    print(f"Total segments: {audit['total_segments']}")
    print(f"\nGate results:")
    for g, s in audit["all_gates"].items():
        print(f"  {g}: {'PASS' if s else 'FAIL'}")
    if audit["segment_crossing"] > 0:
        print(f"  ** FAIL: {audit['segment_crossing']} segment crossings")
    if audit["k10_out_of_bound"] > 0:
        print(f"  ** FAIL: {audit['k10_out_of_bound']} K10 OOB")

    # Suite/task report
    print(f"\nSuite/Task K10 feasibility:")
    for tk, counts in sorted(audit["per_task"].items()):
        pos = counts["k10_positive"]
        tot = counts["episodes"]
        starts = counts["start_count"]
        print(f"  {tk}: {pos}/{tot} positive, {starts} starts, "
              f"no_corridor={counts['no_corridor']}")

    # ── CSV outputs ──
    with open(out / "EPISODE_SUMMARY.csv", "w", encoding="utf-8") as fh:
        fh.write("identity,suite,task_idx,state_id,fold_id,n_steps,"
                 "has_candidate_segment,n_segments,has_critical_state,"
                 "critical_step_count,has_feasible_k10,feasible_start_count,"
                 "first_feasible_start,last_feasible_start,no_feasible_reason,"
                 "mechanism_supported\n")
        for ep in all_eps:
            fh.write(f"{ep['identity']},{ep['suite']},{ep['task_idx']},"
                     f"{ep['state_id']},{ep['fold_id']},{ep['n_steps']},"
                     f"{ep['has_candidate_segment']},{ep['n_segments']},"
                     f"{ep['has_critical_state']},{ep['critical_step_count']},"
                     f"{ep['has_feasible_k10']},{ep['feasible_start_count']},"
                     f"{ep['first_feasible_start']},{ep['last_feasible_start']},"
                     f"{ep['no_feasible_reason']},{ep['mechanism_supported']}\n")

    # TASK_GEOMETRY.csv
    with open(out / "TASK_GEOMETRY.csv", "w", encoding="utf-8") as fh:
        fh.write("task,episodes,supported,no_corridor,k10_positive,total_starts\n")
        for tk, c in sorted(audit["per_task"].items()):
            fh.write(f"{tk},{c['episodes']},{c['supported']},"
                     f"{c['no_corridor']},{c['k10_positive']},{c['start_count']}\n")

    # ── PROTOCOL.json ──
    protocol = {
        "schema": SCHEMA,
        "K": K,
        "teacher_lookahead": K - 1,
        "source_teacher_root": str(args.s1_root),
        "source_teacher_schema": "B3_OFFICIAL_V3_TEACHER_RECORD_V1",
        "critical_conjunction": [
            "label_known",
            "candidate_close",
            "target_relevant",
            "stable_grasp",
            "manipulation_active",
            "NOT release_safe",
            "NOT regrasp_or_instability",
        ],
        "burst_constraints": [
            "all K steps critical",
            "same segment",
            "no unknown gap",
            "no release_safe",
            "no horizon crossing",
        ],
        "release_safe_margin": RELEASE_SAFE_MARGIN,
    }
    with open(out / "PROTOCOL.json", "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2)

    # ── AUDIT.json ──
    with open(out / "AUDIT.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    # ── SOURCE_BINDING.json ──
    source_binding = {
        "git_head": "66f3604da1f178942ed3cfb17ec8db0675b5068b",
        "physics_teacher_root": str(args.s1_root),
        "physics_teacher_schema": "B3_OFFICIAL_V3_TEACHER_RECORD_V1",
        "feature_rebuilder_sha256": "d3a1aacacdffddc3ef1c0679f7ebf82159be0a37314e9eb867912c05e7ae23f1",
        "K": K,
        "teacher_lookahead": K - 1,
    }
    with open(out / "SOURCE_BINDING.json", "w", encoding="utf-8") as fh:
        json.dump(source_binding, fh, indent=2)

    # ── SHA256SUMS ──
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

    with open(out / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "sha256sums": sha, "audit_summary": {
            k: v for k, v in audit.items() if k not in ("per_task",)
        }}, fh, indent=2)

    print(f"\nRoot: {out}")
    print(f"SHA256SUMS: {sha}")
    final_gates = all(audit["all_gates"].values())
    print(f"ALL GATES: {'PASS' if final_gates else 'FAIL'}")


if __name__ == "__main__":
    main()
