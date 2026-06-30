#!/usr/bin/env python3
"""P0-4: Build authoritative supervision from V2PrivilegedTeacher.

Replays privileged_step_records.jsonl through the frozen V2PrivilegedTeacher
to produce per-step phase/corridor/release labels for all CLEAN2000 episodes.

For Object500, cross-validates against FOLD00_teacher_labels_heldout.jsonl.

Usage:
  python build_supervision_auth.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --object_labels FOLD00_teacher_labels_heldout.jsonl \
    [--teacher_config FOLD01_teacher_config.json] \
    --output_dir /path/to/output
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, TeacherConfig
from gripper_attack.sc5mlp_v1 import SC5_PHASES, N_PHASES

PHASE_TO_IDX = {p: i for i, p in enumerate(SC5_PHASES)}


def parse_args():
    p = argparse.ArgumentParser(description="Build authoritative CLEAN2000 supervision")
    p.add_argument("--index", required=True)
    p.add_argument("--object_labels", default=None)
    p.add_argument("--teacher_config", default=None)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def load_teacher_config(path):
    """Load teacher config from FOLD01_teacher_config.json or similar."""
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        thresh = data.get("thresholds", {})
        cfg = TeacherConfig()
        for k, v in thresh.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg, data
    return TeacherConfig(), {}


def compute_corridor_active(phase, prev_phase, gripper_close, grasp_stable):
    """Derive SC5 corridor active from teacher phase transitions.

    The SC5 FSM corridor fires when the gripper transitions from open to closed
    and the teacher indicates grasp_close or stable_grasp phase.
    """
    # Corridor active: during grasp initiation and stable grasp
    if phase in ("grasp_close", "stable_grasp") and gripper_close:
        return 1
    return 0


def compute_release_safe(phase, near_target, obj_lifted):
    """Derive release_safe from teacher phase.

    Release is safe when the object is near the target and not falling.
    """
    return 1 if phase == "release_safe" else 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load index ──
    print("Loading index: {}".format(args.index))
    rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("  {} episodes".format(len(rows)))

    # ── Load teacher config ──
    cfg, cfg_raw = load_teacher_config(args.teacher_config)
    teacher = V2PrivilegedTeacher(cfg)
    print("Teacher: V2PrivilegedTeacher (grasp_close_sustain={}, eef_obj_dist_max={})".format(
        cfg.grasp_close_sustain, cfg.eef_obj_dist_max))

    # ── Load existing Object500 labels for cross-validation ──
    existing_labels = {}
    if args.object_labels and os.path.exists(args.object_labels):
        with open(args.object_labels) as f:
            for line in f:
                rec = json.loads(line.strip())
                key = (rec.get("task_idx", -1), rec.get("state_id", -1), rec.get("step_idx", -1))
                existing_labels[key] = rec
        print("  {} existing Object500 labels".format(len(existing_labels)))

    # ── Process each episode ──
    print("Generating authoritative teacher labels...")

    event_index = []     # per-episode event summary
    step_labels = []     # per-step training labels
    parity_results = []  # Object500 cross-validation
    stats = {"total": 0, "teacher_valid": 0, "teacher_invalid": 0,
             "phase_counts": {p: 0 for p in SC5_PHASES},
             "parity_mismatches": 0, "parity_compared": 0}

    for i, row in enumerate(rows):
        ek = row["episode_key"]
        ep_dir = row["source_root"]
        suite = row["suite"]
        task_id = row["task_id"]
        state_id = row["state_id"]
        source_fmt = row["source_format"]

        # Load privileged records
        priv_path = os.path.join(ep_dir, "privileged_step_records.jsonl")
        if not os.path.exists(priv_path):
            stats["teacher_invalid"] += 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False,
                "invalid_reason": "no_privileged_records",
                "mechanism_eligible": False,
            })
            continue

        with open(priv_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        # Check mechanism eligibility
        teacher_eligible = row.get("teacher_eligible", True)
        mechanism_eligible = row.get("mechanism_eligible", teacher_eligible)

        if not mechanism_eligible:
            stats["teacher_invalid"] += 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False,
                "invalid_reason": "mechanism_ineligible",
                "mechanism_eligible": False,
                "abstain_reason": row.get("abstain_reason", ""),
            })
            continue

        # Run authoritative teacher
        labels = teacher.label_trajectory(records)

        if not labels:
            stats["teacher_invalid"] += 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False,
                "invalid_reason": "teacher_produced_no_labels",
                "mechanism_eligible": True,
            })
            continue

        # ── Episode-level event ──
        phases_seen = set()
        anchor_step = -1
        release_step = -1
        grasp_step = -1
        for j, lbl in enumerate(labels):
            phase = lbl.get("phase", "abstain_unsupported")
            phases_seen.add(phase)
            if phase == "grasp_close" and grasp_step < 0:
                grasp_step = lbl.get("step_idx", j)
            if phase == "release_safe" and release_step < 0:
                release_step = lbl.get("step_idx", j)

        anchor_step = release_step if release_step >= 0 else grasp_step

        teacher_valid = "release_safe" in phases_seen or "stable_carry" in phases_seen
        if teacher_valid:
            stats["teacher_valid"] += 1
        else:
            stats["teacher_invalid"] += 1

        K = 10
        window_start = max(0, anchor_step - K) if anchor_step >= 0 else -1
        window_end = min(row["n_steps"] - 1, anchor_step + K) if anchor_step >= 0 else -1

        event_index.append({
            "episode_key": ek,
            "parent_key": row["parent_key"],
            "suite": suite, "task_id": task_id, "state_id": state_id,
            "teacher_valid": teacher_valid,
            "teacher_anchor_step": anchor_step,
            "teacher_window_start": window_start,
            "teacher_window_end": window_end,
            "mechanism_eligible": mechanism_eligible,
            "invalid_reason": "" if teacher_valid else "no_release_or_carry_phase",
        })

        # ── Per-step training labels ──
        prev_phase = "approach"
        prev_gripper_close = False
        for j, lbl in enumerate(labels):
            phase = lbl.get("phase", "abstain_unsupported")
            phase_idx = PHASE_TO_IDX.get(phase, 8)  # default to abstain
            gripper_close = lbl.get("gripper_close", False)
            grip_stable = lbl.get("close_consecutive", 0) >= cfg.grasp_close_sustain
            near_target = lbl.get("obj_target_dist", 999) < cfg.release_target_dist_max
            obj_lifted = lbl.get("obj_lifted", False)
            opening_ok = lbl.get("opening_proxy_ok", False)
            confidence = lbl.get("confidence", 0.0)

            step_idx = lbl.get("step_idx", j)
            policy_step_idx = lbl.get("policy_step_idx", j)

            corridor = compute_corridor_active(phase, prev_phase, gripper_close, grip_stable)
            release_s = compute_release_safe(phase, near_target, obj_lifted)

            step_labels.append({
                "episode_key": ek,
                "step": step_idx,
                "policy_step_idx": policy_step_idx,
                "teacher_phase_idx": phase_idx,
                "teacher_phase": phase,
                "teacher_sc5_corridor_active": corridor,
                "release_safe": release_s,
                "teacher_confidence": confidence,
                "gripper_close": gripper_close,
                "opening_proxy_ok": opening_ok,
                "obj_lifted": obj_lifted,
            })

            stats["phase_counts"][phase] = stats["phase_counts"].get(phase, 0) + 1
            prev_phase = phase
            prev_gripper_close = gripper_close

            # ── Object500 parity ──
            if source_fmt == "object500_v1" and existing_labels:
                key = (task_id, state_id, step_idx)
                if key in existing_labels:
                    existing = existing_labels[key]
                    ex_phase = existing.get("phase", "")
                    stats["parity_compared"] += 1
                    if ex_phase != phase:
                        stats["parity_mismatches"] += 1
                        if stats["parity_mismatches"] <= 10:
                            parity_results.append({
                                "episode_key": ek,
                                "step": step_idx,
                                "new_phase": phase,
                                "existing_phase": ex_phase,
                            })

        if (i + 1) % 500 == 0:
            print("  {} / {} ...".format(i + 1, len(rows)))

    # ── Summary ──
    print()
    n_total = len(rows)
    n_valid = stats["teacher_valid"]
    n_invalid = stats["teacher_invalid"]
    parity_rate = 1.0 - stats["parity_mismatches"] / max(1, stats["parity_compared"])

    print("=== Supervision Summary ===")
    print("  Total: {}".format(n_total))
    print("  Teacher valid:   {}".format(n_valid))
    print("  Teacher invalid: {}".format(n_invalid))
    print("  Phase distribution:")
    for phase in SC5_PHASES:
        print("    {}: {}".format(phase, stats["phase_counts"].get(phase, 0)))
    print("  Object parity: {:.4f} ({}/{})".format(
        parity_rate, stats["parity_compared"] - stats["parity_mismatches"],
        stats["parity_compared"]))
    print("  Parity mismatches: {}".format(stats["parity_mismatches"]))

    # ── Write outputs ──
    # Event index
    evt_path = os.path.join(args.output_dir, "TEACHER_EVENT_INDEX.jsonl")
    with open(evt_path, "w") as f:
        for e in event_index:
            f.write(json.dumps(e) + "\n")
    print("  {}".format(evt_path))

    # Step labels
    lbl_path = os.path.join(args.output_dir, "TEACHER_STEP_LABELS.csv")
    import csv
    step_cols = ["episode_key", "step", "policy_step_idx", "teacher_phase_idx",
                 "teacher_phase", "teacher_sc5_corridor_active", "release_safe",
                 "teacher_confidence", "gripper_close", "opening_proxy_ok", "obj_lifted"]
    with open(lbl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=step_cols)
        w.writeheader()
        w.writerows(step_labels)
    print("  {} ({} rows)".format(lbl_path, len(step_labels)))

    # Object parity
    parity_path = os.path.join(args.output_dir, "OBJECT500_TEACHER_PARITY.json")
    parity_passed = stats["parity_mismatches"] == 0
    with open(parity_path, "w") as f:
        json.dump({
            "gate": "OBJECT500_TEACHER_PARITY_V1",
            "passed": parity_passed,
            "total_compared": stats["parity_compared"],
            "mismatches": stats["parity_mismatches"],
            "parity_rate": float(parity_rate),
            "sample_mismatches": parity_results[:20],
        }, f, indent=2)
    print("  {}".format(parity_path))

    # Teacher provenance
    prov_path = os.path.join(args.output_dir, "TEACHER_PROVENANCE.json")
    with open(prov_path, "w") as f:
        teacher_sha = hashlib.sha256(
            open(os.path.join(REPO_ROOT, "src/gripper_attack/v2_privileged_teacher.py"), "rb").read()
        ).hexdigest() if os.path.exists(os.path.join(REPO_ROOT, "src/gripper_attack/v2_privileged_teacher.py")) else "UNKNOWN"
        json.dump({
            "gate": "TEACHER_PROVENANCE_V1",
            "teacher_class": "V2PrivilegedTeacher",
            "source_file": "src/gripper_attack/v2_privileged_teacher.py",
            "source_sha256": teacher_sha,
            "config_source": args.teacher_config or "defaults",
            "config_thresholds": {k: getattr(cfg, k) for k in sorted(dir(cfg)) if not k.startswith("_") and not callable(getattr(cfg, k))},
            "sc5_phases": SC5_PHASES,
        }, f, indent=2)
    print("  {}".format(prov_path))

    # Supervision envelope
    env_path = os.path.join(args.output_dir, "SUPERVISION_ENVELOPE.json")
    envelope = {
        "gate": "CLEAN2000_SUPERVISION_AUTH_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "teacher_valid": stats["teacher_valid"],
        "teacher_invalid": stats["teacher_invalid"],
        "object_parity_passed": parity_passed,
        "binds_to": "CLEAN2000_CANONICAL_V1",
        "status": "AUTHORITATIVE" if parity_passed else "PARITY_FAILED",
    }
    with open(env_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print("  {}".format(env_path))

    print()
    if not parity_passed:
        print("OBJECT PARITY FAILED — {} mismatches".format(stats["parity_mismatches"]))
        sys.exit(1)
    print("DONE.")


if __name__ == "__main__":
    main()
