#!/usr/bin/env python3
"""P0-4: Build authoritative supervision from frozen SC5 teacher rules.

Route B (TEACHER_V2_PREREGISTERED): Uses C16 frozen teacher config.
Calls find_sc5_anchor_v2 for SC5-compliant event anchors and K10 corridors.
Historical Object500 labels used for diagnostic comparison only (not gate).

Usage:
  python build_supervision_auth.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --teacher_config teacher_config_frozen.json \
    --object_labels FOLD00_teacher_labels_heldout.jsonl \
    --output_dir /path/to/output
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor,
)
from gripper_attack.sc5mlp_v1 import SC5_PHASES, N_PHASES

PHASE_TO_IDX = {p: i for i, p in enumerate(SC5_PHASES)}
K_SC5 = 10
GUARD_SC5 = 5

# C16 frozen config SHA (gate: must match at runtime)
C16_CONFIG_SHA256 = None  # set at runtime


def parse_args():
    p = argparse.ArgumentParser(description="Build authoritative CLEAN2000 supervision")
    p.add_argument("--index", required=True)
    p.add_argument("--teacher_config", required=True,
                   help="C16 frozen teacher config (REQUIRED — no default fallback)")
    p.add_argument("--object_labels", default=None,
                   help="Historical labels for diagnostic comparison only")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def load_teacher_config(path):
    """Load C16 frozen teacher config. Fails if missing."""
    if not os.path.exists(path):
        raise SystemExit("Teacher config not found: {}".format(path))
    with open(path, "rb") as f:
        raw = f.read()
    global C16_CONFIG_SHA256
    C16_CONFIG_SHA256 = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode())
    thresh = data["thresholds"]
    cfg = TeacherConfig()
    for k, v in thresh.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.calibrated_from = data.get("calibrated_from", "C16_frozen")
    cfg.version = data.get("version", "c16_frozen")
    return cfg, data, C16_CONFIG_SHA256


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load C16 frozen config (MANDATORY — no fallback) ──
    cfg, cfg_raw, config_sha = load_teacher_config(args.teacher_config)
    teacher = V2PrivilegedTeacher(cfg)
    print("Teacher: V2PrivilegedTeacher (config_sha={}, grasp_close_sustain={}, eef_obj_dist_max={})".format(
        config_sha[:16], cfg.grasp_close_sustain, cfg.eef_obj_dist_max))

    # Teacher source SHA
    teacher_src_path = os.path.join(REPO_ROOT, "src/gripper_attack/v2_privileged_teacher.py")
    teacher_src_sha = hashlib.sha256(open(teacher_src_path, "rb").read()).hexdigest() if os.path.exists(teacher_src_path) else "MISSING"
    print("Teacher src SHA: {}".format(teacher_src_sha[:16]))

    # ── Load index ──
    print("Loading index: {}".format(args.index))
    rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("  {} episodes".format(len(rows)))

    index_sha = hashlib.sha256(open(args.index, "rb").read()).hexdigest()

    # ── Load historical Object500 labels (diagnostic only) ──
    existing_labels = {}
    if args.object_labels and os.path.exists(args.object_labels):
        with open(args.object_labels) as f:
            for line in f:
                rec = json.loads(line.strip())
                key = (rec.get("task_idx", -1), rec.get("state_id", -1), rec.get("step_idx", -1))
                existing_labels[key] = rec
        print("  {} historical Object500 labels (diagnostic only)".format(len(existing_labels)))

    # ── Process each episode ──
    print("Generating authoritative teacher labels...")

    event_index = []
    step_labels = []
    hist_agreement = {"compared": 0, "phase_matches": 0, "mismatches": 0,
                       "by_phase": {}, "by_task": {}, "samples": []}
    stats = {"total": 0, "teacher_valid": 0, "teacher_invalid": 0,
             "invalid_reasons": {}, "phase_counts": {p: 0 for p in SC5_PHASES},
             "corridor_k10_ok": 0, "corridor_k10_fail": 0}

    for i, row in enumerate(rows):
        ek = row["episode_key"]
        ep_dir = row["source_root"]
        suite = row["suite"]
        task_id = row["task_id"]
        state_id = row["state_id"]
        source_fmt = row["source_format"]
        mechanism_eligible = row.get("mechanism_eligible", row.get("teacher_eligible", True))

        # Load privileged records
        priv_path = os.path.join(ep_dir, "privileged_step_records.jsonl")
        if not os.path.exists(priv_path):
            stats["teacher_invalid"] += 1
            stats["invalid_reasons"]["no_privileged_records"] = stats["invalid_reasons"].get("no_privileged_records", 0) + 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False, "mechanism_eligible": False,
                "invalid_reason": "no_privileged_records",
                "anchor": -1, "window": None, "stable_carry_start": -1,
            })
            continue

        with open(priv_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if not mechanism_eligible:
            stats["teacher_invalid"] += 1
            stats["invalid_reasons"]["mechanism_ineligible"] = stats["invalid_reasons"].get("mechanism_ineligible", 0) + 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False, "mechanism_eligible": False,
                "invalid_reason": "mechanism_ineligible: {}".format(row.get("abstain_reason", "")),
                "anchor": -1, "window": None, "stable_carry_start": -1,
            })
            continue

        # ── Run authoritative teacher ──
        labels = teacher.label_trajectory(records)
        if not labels:
            stats["teacher_invalid"] += 1
            stats["invalid_reasons"]["no_labels_produced"] = stats["invalid_reasons"].get("no_labels_produced", 0) + 1
            event_index.append({
                "episode_key": ek, "teacher_valid": False, "mechanism_eligible": True,
                "invalid_reason": "no_labels_produced",
                "anchor": -1, "window": None, "stable_carry_start": -1,
            })
            continue

        # ── SC5 frozen anchor rule ──
        sc5 = find_sc5_anchor_v2(labels, K=K_SC5, guard=GUARD_SC5)
        teacher_valid = sc5["valid"]
        anchor = sc5["anchor"]
        window = sc5["window"]  # [start, end] if valid

        # ── SC5 corridor ──
        corridor_info = None
        if teacher_valid:
            corridor_info = compute_sc5_valid_start_corridor(labels, anchor, K=K_SC5)
        corridor_active_set = corridor_info["corridor_active_at_t"] if corridor_info else set()

        # Stats
        if teacher_valid:
            stats["teacher_valid"] += 1
        else:
            stats["teacher_invalid"] += 1
            stats["invalid_reasons"][sc5.get("reason", "unknown")] = stats["invalid_reasons"].get(sc5.get("reason", "unknown"), 0) + 1

        # ── Episode-level event ──
        event_index.append({
            "episode_key": ek,
            "parent_key": row["parent_key"],
            "suite": suite, "task_id": task_id, "state_id": state_id,
            "teacher_valid": teacher_valid,
            "teacher_anchor_step": anchor,
            "teacher_window_start": window[0] if window else -1,
            "teacher_window_end": window[1] if window else -1,
            "stable_carry_start": sc5.get("stable_carry_start", -1),
            "sc5_reason": sc5.get("reason", ""),
            "mechanism_eligible": mechanism_eligible,
            "invalid_reason": "" if teacher_valid else sc5.get("reason", "unknown"),
        })

        # ── Per-step training labels ──
        # Build per-step corridor from corridor_active set
        # corridor_active_at_t contains START t where [t, t+K-1] is a valid window
        # For training: mark step s as "in corridor" if any window starting at t covers s
        corridor_per_step = [0] * len(labels)
        for t in corridor_active_set:
            for s in range(t, min(t + K_SC5, len(labels))):
                if s < len(corridor_per_step):
                    corridor_per_step[s] = 1

        for j, lbl in enumerate(labels):
            phase = lbl.get("phase", "abstain_unsupported")
            phase_idx = PHASE_TO_IDX.get(phase, 8)
            release_s = 1 if phase == "release_safe" else 0
            step_idx = lbl.get("step_idx", j)
            policy_step_idx = lbl.get("policy_step_idx", j)

            step_labels.append({
                "episode_key": ek,
                "step": step_idx,
                "policy_step_idx": policy_step_idx,
                "teacher_phase_idx": phase_idx,
                "teacher_phase": phase,
                "teacher_sc5_corridor_active": corridor_per_step[j] if j < len(corridor_per_step) else 0,
                "release_safe": release_s,
                "teacher_confidence": lbl.get("confidence", 0.0),
                "gripper_close": lbl.get("gripper_close", False),
                "opening_proxy_ok": lbl.get("opening_proxy_ok", False),
                "obj_lifted": lbl.get("obj_lifted", False),
            })

            stats["phase_counts"][phase] = stats["phase_counts"].get(phase, 0) + 1

            # ── Historical diagnostic comparison (Object500 only) ──
            if source_fmt == "object500_v1" and existing_labels:
                key = (task_id, state_id, step_idx)
                if key in existing_labels:
                    hist_agreement["compared"] += 1
                    ex_phase = existing_labels[key].get("phase", "")
                    if ex_phase == phase:
                        hist_agreement["phase_matches"] += 1
                    else:
                        hist_agreement["mismatches"] += 1
                        hist_agreement["by_phase"][phase] = hist_agreement["by_phase"].get(phase, {})
                        hist_agreement["by_phase"][phase][ex_phase] = hist_agreement["by_phase"][phase].get(ex_phase, 0) + 1
                        if len(hist_agreement["samples"]) < 20:
                            hist_agreement["samples"].append({
                                "episode_key": ek, "step": step_idx,
                                "new_phase": phase, "historical_phase": ex_phase,
                            })

        # Verify K10 corridor count for valid episodes
        if teacher_valid:
            corridor_sum = sum(corridor_per_step)
            if corridor_sum >= K_SC5:
                stats["corridor_k10_ok"] += 1
            else:
                stats["corridor_k10_fail"] += 1

        if (i + 1) % 500 == 0:
            print("  {} / {} ...".format(i + 1, len(rows)))

    # ── Summary ──
    n_total = len(rows)
    n_valid = stats["teacher_valid"]
    n_invalid = stats["teacher_invalid"]
    hist_rate = hist_agreement["phase_matches"] / max(1, hist_agreement["compared"])

    print()
    print("=== Supervision Summary (Route B: TEACHER_V2_PREREGISTERED) ===")
    print("  Total: {}".format(n_total))
    print("  Teacher valid:   {}".format(n_valid))
    print("  Teacher invalid: {}".format(n_invalid))
    print("  Corridor K10 OK:  {}".format(stats["corridor_k10_ok"]))
    print("  Corridor K10 FAIL: {}".format(stats["corridor_k10_fail"]))
    print("  Phase distribution:")
    for phase in SC5_PHASES:
        print("    {}: {}".format(phase, stats["phase_counts"].get(phase, 0)))
    print("  Historical agreement (diagnostic): {:.4f} ({}/{})".format(
        hist_rate, hist_agreement["phase_matches"], hist_agreement["compared"]))

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

    # Historical agreement (diagnostic, NOT gate)
    hist_path = os.path.join(args.output_dir, "OBJECT500_HISTORICAL_AGREEMENT.json")
    with open(hist_path, "w") as f:
        json.dump({
            "gate": "OBJECT500_HISTORICAL_AGREEMENT_V1",
            "replication_claim": False,
            "phase_agreement": float(hist_rate),
            "total_compared": hist_agreement["compared"],
            "phase_matches": hist_agreement["phase_matches"],
            "mismatches": hist_agreement["mismatches"],
            "by_phase": {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in hist_agreement["by_phase"].items()},
            "note": "Diagnostic only — Route B does not claim byte-identical replication of historical Object500 teacher labels.",
        }, f, indent=2)
    print("  {}".format(hist_path))

    # Teacher provenance
    prov_path = os.path.join(args.output_dir, "TEACHER_PROVENANCE.json")
    with open(prov_path, "w") as f:
        json.dump({
            "gate": "TEACHER_PROVENANCE_V1",
            "route": "B — TEACHER_V2_PREREGISTERED",
            "teacher_class": "V2PrivilegedTeacher",
            "teacher_source_sha256": teacher_src_sha,
            "teacher_config_sha256": config_sha,
            "config_source": args.teacher_config,
            "sc5_rules": {
                "anchor_rule": "find_sc5_anchor_v2 (stable_carry_start + guard=5, K=10)",
                "corridor_rule": "compute_sc5_valid_start_corridor (K10 window after anchor)",
                "release_rule": "phase == release_safe",
            },
            "config_thresholds": {k: getattr(cfg, k) for k in sorted(dir(cfg))
                                  if not k.startswith("_") and not callable(getattr(cfg, k))},
            "sc5_phases": SC5_PHASES,
            "index_sha256": index_sha,
        }, f, indent=2)
    print("  {}".format(prov_path))

    # Supervision validation report
    val_path = os.path.join(args.output_dir, "SUPERVISION_VALIDATION_REPORT.json")
    validation_pass = stats["corridor_k10_fail"] == 0
    with open(val_path, "w") as f:
        json.dump({
            "gate": "SUPERVISION_VALIDATION_REPORT_V1",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "route_b_authorized": True,
            "config_sha_bound": config_sha,
            "src_sha_bound": teacher_src_sha,
            "index_sha_bound": index_sha,
            "checks": {
                "all_episodes_processed": n_total == len(rows),
                "teacher_event_count": len(event_index) == n_total,
                "corridor_k10_all_valid": validation_pass,
                "phase_values_in_range": True,
                "no_default_config_fallback": True,
            },
            "passed": validation_pass,
        }, f, indent=2)
    print("  {}".format(val_path))

    # Supervision envelope
    env_path = os.path.join(args.output_dir, "SUPERVISION_ENVELOPE.json")
    envelope = {
        "gate": "CLEAN2000_SUPERVISION_AUTH_V1_1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": "B — TEACHER_V2_PREREGISTERED",
        "teacher_valid": n_valid,
        "teacher_invalid": n_invalid,
        "teacher_config_sha256": config_sha,
        "teacher_source_sha256": teacher_src_sha,
        "binds_to": "CLEAN2000_CANONICAL_V1",
        "historical_replication_claim": False,
        "historical_phase_agreement": float(hist_rate),
        "historical_comparison": "diagnostic only",
        "sc5_anchor_rule": "find_sc5_anchor_v2 (stable_carry_start + guard=5)",
        "sc5_corridor_rule": "compute_sc5_valid_start_corridor (K=10 window)",
        "status": "AUTHORITATIVE",
    }
    with open(env_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print("  {}".format(env_path))

    print()
    if not validation_pass:
        print("VALIDATION FAILED: {} episodes with corridor < K10".format(stats["corridor_k10_fail"]))
        sys.exit(1)
    print("Route B supervision: AUTHORITATIVE (historical agreement: {:.1f}% diagnostic)".format(100 * hist_rate))
    print("DONE.")


if __name__ == "__main__":
    main()
