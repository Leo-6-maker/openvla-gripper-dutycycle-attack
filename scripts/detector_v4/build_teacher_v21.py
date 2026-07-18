#!/usr/bin/env python3
"""Teacher V2.1: Phase-based close event segmentation.

Fixes the P0 label conflict in V2.0 where long close windows that eventually
release got both criticality=1 AND veto=1 at the same step.

V2.1 splits each close event into sequential phases:
  PRE_SUPPORT       — close onset to first event_support (or first valid step)
  VALID_RETENTION   — support + retention_active + grasp_support + T10, no release_imminent
  RELEASE_IMMINENT_TAIL — release_imminent becomes True within valid segment
  POST_RELEASE      — after event_release_onset
  UNSTABLE_TRANSITION   — short gaps between close events

Hard invariant: quality_valid AND veto_invalid = 0 (enforced at output).
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── constants ──────────────────────────────────────────────────────────
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))
MIN_PHASE_DURATION = 3
SUSTAIN_GAP = 5

PHASES = [
    "PRE_SUPPORT",
    "VALID_RETENTION",
    "RELEASE_IMMINENT_TAIL",
    "POST_RELEASE",
    "UNSTABLE_TRANSITION",
    "UNKNOWN",
]

SCHEMA = "DETECTOR_V4_TEACHER_V21_V1"

# ── helpers ────────────────────────────────────────────────────────────
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ── core: phase-based segmentation ────────────────────────────────────
def segment_close_event(records: list[dict[str, Any]],
                        onset: int, end: int) -> list[dict[str, Any]]:
    """Split one close event [onset, end] into sequential phases.

    Returns list of {phase, start_step, end_step, ...}.
    Steps are assigned to exactly one phase; unknown steps are masked.
    """
    phases = []
    cursor = onset

    # Helper: check step-level conditions
    def step_has_support(s):
        return bool(records[s].get("event_support"))

    def step_has_retention(s):
        r = records[s]
        return bool(r.get("retention_active") and not r.get("retention_unknown_mask"))

    def step_has_grasp(s):
        r = records[s]
        return bool(r.get("grasp_support") and not r.get("retention_unknown_mask"))

    def step_has_t10(s):
        r = records[s]
        return bool(r.get("retention_continuation_t10") and not r.get("retention_unknown_mask"))

    def step_release_imminent(s):
        r = records[s]
        return bool(r.get("release_imminent") and not r.get("retention_unknown_mask"))

    def step_has_release_event(s):
        return bool(records[s].get("event_release_onset"))

    def step_evidence_valid(s):
        return bool(records[s].get("event_evidence_valid", True))

    def step_unknown(s):
        return bool(records[s].get("retention_unknown_mask", False))

    # Phase 1: PRE_SUPPORT — from onset to first support or first retention+grasp
    first_valid = None
    for s in range(onset, end + 1):
        if step_evidence_valid(s) and not step_unknown(s):
            if step_has_support(s) or (step_has_retention(s) and step_has_grasp(s)):
                first_valid = s
                break
    if first_valid is not None and first_valid > onset:
        phases.append({"phase": "PRE_SUPPORT", "start_step": onset,
                       "end_step": first_valid - 1})
        cursor = first_valid
    elif first_valid is None:
        # No valid evidence in window — whole thing is PRE_SUPPORT or UNKNOWN
        phases.append({"phase": "PRE_SUPPORT", "start_step": onset, "end_step": end})
        return phases

    # Phase 2: Scan forward from cursor for VALID_RETENTION segments
    # A step is VALID_RETENTION if: support + retention + grasp + T10 + NOT release_imminent + evidence_valid
    # Build contiguous segments
    in_valid = False
    valid_start = cursor
    for s in range(cursor, end + 1):
        is_valid = (step_evidence_valid(s) and not step_unknown(s)
                   and step_has_support(s) and step_has_retention(s)
                   and step_has_grasp(s) and step_has_t10(s)
                   and not step_release_imminent(s))

        if is_valid and not in_valid:
            valid_start = s
            in_valid = True
        elif not is_valid and in_valid:
            # Close current valid segment
            seg_end = s - 1
            duration = seg_end - valid_start + 1
            if duration >= MIN_PHASE_DURATION:
                phases.append({"phase": "VALID_RETENTION",
                              "start_step": valid_start, "end_step": seg_end})
            else:
                # Too short — merge into UNSTABLE_TRANSITION
                phases.append({"phase": "UNSTABLE_TRANSITION",
                              "start_step": valid_start, "end_step": seg_end})
            in_valid = False

    if in_valid:
        seg_end = end
        duration = seg_end - valid_start + 1
        if duration >= MIN_PHASE_DURATION:
            phases.append({"phase": "VALID_RETENTION",
                          "start_step": valid_start, "end_step": seg_end})
        else:
            phases.append({"phase": "UNSTABLE_TRANSITION",
                          "start_step": valid_start, "end_step": seg_end})

    # Phase 3: RELEASE_IMMINENT_TAIL — after VALID_RETENTION ends,
    # if remaining steps in window have release_imminent, label them
    # Find gaps between phases and label them
    # Build a coverage map first
    covered = [False] * (end + 1)
    for ph in phases:
        for s in range(ph["start_step"], ph["end_step"] + 1):
            covered[s] = True

    # Fill gaps
    for s in range(onset, end + 1):
        if not covered[s] and step_evidence_valid(s) and not step_unknown(s):
            # Find contiguous unlabeled segment
            gap_start = s
            while s <= end and not covered[s]:
                s += 1
            gap_end = s - 1

            # Classify gap
            has_rel_imm = any(step_release_imminent(gs) for gs in range(gap_start, gap_end + 1))
            has_rel_evt = any(step_has_release_event(gs) for gs in range(gap_start, gap_end + 1))
            has_sup = any(step_has_support(gs) for gs in range(gap_start, gap_end + 1))

            if has_rel_evt:
                phase = "POST_RELEASE"
            elif has_rel_imm:
                phase = "RELEASE_IMMINENT_TAIL"
            elif has_sup:
                phase = "UNSTABLE_TRANSITION"
            else:
                phase = "UNSTABLE_TRANSITION"

            duration = gap_end - gap_start + 1
            if duration >= MIN_PHASE_DURATION or phase in ("POST_RELEASE", "RELEASE_IMMINENT_TAIL"):
                phases.append({"phase": phase, "start_step": gap_start, "end_step": gap_end})
            for gs in range(gap_start, gap_end + 1):
                covered[gs] = True

    # Sort and dedup
    phases.sort(key=lambda p: p["start_step"])
    return phases


def build_v21_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build step-level Teacher V2.1 labels for one episode."""
    n = len(records)

    # Extract all close events
    close_events = []
    i = 0
    while i < n:
        r = records[i]
        if r.get("event_close_onset") and r.get("event_end_step", -1) >= r["step"] + MIN_PHASE_DURATION:
            onset = r["step"]
            end = min(r["event_end_step"], n - 1)
            close_events.append({"onset": onset, "end": end, "event_id": len(close_events)})
            i = end + 1
        else:
            i += 1

    # Phase-segment each close event
    all_phases = []
    for ce in close_events:
        phases = segment_close_event(records, ce["onset"], ce["end"])
        for ph in phases:
            ph["event_id"] = ce["event_id"]
            ph["close_onset"] = ce["onset"]
            ph["close_end"] = ce["end"]
        all_phases.extend(phases)

    # Build step-level labels with hard invariant: quality_valid AND veto_invalid = 0
    labels = []
    for step in range(n):
        r = records[step]
        evidence_valid = r.get("event_evidence_valid", True) and not r.get("retention_unknown_mask", False)

        # Find which phase this step belongs to
        in_candidate_close = any(
            ce["onset"] <= step <= ce["end"] for ce in close_events
        )
        phase = "NO_CLOSE"
        for ph in all_phases:
            if ph["start_step"] <= step <= ph["end_step"]:
                phase = ph["phase"]
                break

        # Compute quality_valid and veto_invalid with mutual exclusion
        quality_valid = False
        veto_invalid = False
        release_imminent = False

        if evidence_valid:
            if phase == "VALID_RETENTION":
                quality_valid = True
                veto_invalid = False  # invariant
            elif phase in ("RELEASE_IMMINENT_TAIL", "POST_RELEASE", "UNSTABLE_TRANSITION"):
                quality_valid = False
                veto_invalid = True
            elif phase == "PRE_SUPPORT":
                # Pre-support: neither quality nor veto (transitional)
                quality_valid = False
                veto_invalid = False

            release_imminent = bool(r.get("release_imminent") and not r.get("retention_unknown_mask"))

        # HARD INVARIANT CHECK
        if quality_valid and veto_invalid:
            raise AssertionError(
                f"INVARIANT VIOLATION: quality_valid AND veto_invalid both True at step {step}"
            )

        # known_mask: True when we have valid evidence to supervise
        # UNKNOWN phases or evidence-invalid steps are masked
        known_mask = evidence_valid and phase != "UNKNOWN"

        labels.append({
            "step": step,
            "candidate_close": in_candidate_close,
            "phase": phase,
            "quality_valid": quality_valid,
            "veto_invalid": veto_invalid,
            "release_imminent": release_imminent,
            "known_mask": known_mask,
            "event_id": next((ph["event_id"] for ph in all_phases
                             if ph["start_step"] <= step <= ph["end_step"]), -1),
        })

    return labels, all_phases, close_events


# ── audit ──────────────────────────────────────────────────────────────
def audit_labels(labels: list[dict[str, Any]], identity: str
                 ) -> dict[str, Any]:
    """Run label conflict census and mutual-exclusion audit."""
    n_conflict = sum(1 for l in labels if l["quality_valid"] and l["veto_invalid"])
    n_quality = sum(1 for l in labels if l["quality_valid"])
    n_veto = sum(1 for l in labels if l["veto_invalid"])
    n_close = sum(1 for l in labels if l["candidate_close"])
    n_known = sum(1 for l in labels if l["known_mask"])
    n_release = sum(1 for l in labels if l["release_imminent"])

    phase_counts = defaultdict(int)
    for l in labels:
        phase_counts[l["phase"]] += 1

    return {
        "identity": identity,
        "n_steps": len(labels),
        "n_conflict": n_conflict,
        "n_quality_valid": n_quality,
        "n_veto_invalid": n_veto,
        "n_candidate_close": n_close,
        "n_known": n_known,
        "n_release_imminent": n_release,
        "phase_counts": dict(phase_counts),
        "invariant_pass": n_conflict == 0,
    }


# ── main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    s1 = args.s1_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)  # fail if exists = non-overwritable

    # Process all 800 FIT identities
    scope = [(s, t, st) for s in SUITES for t in range(N_TASKS) for st in FIT_STATES]

    all_audits = []
    all_phase_counts: dict[str, int] = defaultdict(int)
    per_fold: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_labels = 0
    total_conflicts = 0

    for suite, task, state in scope:
        ident_dir = s1 / suite / f"task_{task:02d}" / f"state_{state:02d}"
        teacher_path = ident_dir / "teacher_retention_records.jsonl"
        if not teacher_path.exists():
            continue

        records = jsonl(teacher_path)
        if not records:
            continue

        cid = f"{suite}/task_{task:02d}/state_{state:02d}"
        labels, phases, close_events = build_v21_labels(records)
        audit = audit_labels(labels, cid)
        all_audits.append(audit)
        total_labels += len(labels)
        total_conflicts += audit["n_conflict"]

        fold_id = state // 5
        for ph, count in audit["phase_counts"].items():
            all_phase_counts[ph] += count
            per_fold[fold_id][ph] += count

        # Write per-identity labels
        ident_out = out / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "teacher_v21_labels.jsonl", "w", encoding="utf-8") as fh:
            for label in labels:
                fh.write(json.dumps(label, ensure_ascii=False) + "\n")

        # Write phases manifest
        with open(ident_out / "close_phases.json", "w", encoding="utf-8") as fh:
            json.dump({
                "schema": SCHEMA,
                "identity": cid,
                "n_close_events": len(close_events),
                "close_events": close_events,
                "phases": phases,
            }, fh, indent=2, ensure_ascii=False)

        if total_conflicts > 0:
            print(f"CONFLICT: {cid} n_conflict={audit['n_conflict']}")

    # ── Global summary ──
    print(f"\n=== TEACHER V2.1 AUDIT ===")
    print(f"Episodes processed: {len(all_audits)}")
    print(f"Total steps: {total_labels}")
    print(f"TOTAL CONFLICTS (quality_valid AND veto_invalid): {total_conflicts}")
    print(f"INVARIANT PASS: {total_conflicts == 0}")

    print(f"\nPhase distribution (step-level):")
    for ph in PHASES:
        print(f"  {ph}: {all_phase_counts.get(ph, 0)}")

    print(f"\nPer-fold phase distribution:")
    for fid in sorted(per_fold):
        print(f"  Fold {fid}:")
        for ph in PHASES:
            c = per_fold[fid].get(ph, 0)
            if c > 0:
                print(f"    {ph}: {c}")

    # Quality/veto summary
    n_quality = sum(a["n_quality_valid"] for a in all_audits)
    n_veto = sum(a["n_veto_invalid"] for a in all_audits)
    n_close = sum(a["n_candidate_close"] for a in all_audits)
    print(f"\nLabel summary:")
    print(f"  quality_valid steps:  {n_quality}")
    print(f"  veto_invalid steps:   {n_veto}")
    print(f"  candidate_close steps: {n_close}")
    print(f"  quality+veto overlap:  {total_conflicts} (must be 0)")

    # ── Build SHA256SUMS ──
    files = sorted(out.rglob("*"))
    file_list = [f for f in files if f.is_file()]
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = fp.relative_to(out)
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel}\n")
    sha = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    # Manifest (written AFTER SHA256SUMS, so it references the SHA)
    manifest = {
        "schema": f"{SCHEMA}_MANIFEST",
        "s1_root_sha256": "15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf",
        "n_episodes": len(all_audits),
        "total_steps": total_labels,
        "total_conflicts": total_conflicts,
        "invariant_pass": total_conflicts == 0,
        "phase_counts": dict(all_phase_counts),
        "per_fold_counts": {str(k): dict(v) for k, v in per_fold.items()},
        "quality_valid_steps": n_quality,
        "veto_invalid_steps": n_veto,
        "sha256sums_sha256": sha,
    }
    with open(out / "teacher_v21_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Rebuild SHA256SUMS to include the manifest
    files2 = sorted(out.rglob("*"))
    file_list2 = [f for f in files2 if f.is_file()]
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list2:
            rel = fp.relative_to(out)
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel}\n")
    sha2 = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha2}  SHA256SUMS\n")

    print(f"\nRoot: {out}")
    print(f"SHA256SUMS: {sha2}")
    print(f"INVARIANT: {'PASS' if total_conflicts == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
