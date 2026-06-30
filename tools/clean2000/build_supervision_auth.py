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

# Pre-registered C16 frozen config SHA (migration_audit/.../teacher_config_frozen.json)
C16_EXPECTED_CONFIG_SHA256 = "ebc1ccda21cdfeae0f70f90ef0e433be3474ef0baa9cf52f609d620f863ce87a"

# Expected threshold keys in C16 config
C16_EXPECTED_THRESHOLD_KEYS = frozenset([
    "grasp_close_sustain", "grasp_open_proxy_max", "eef_obj_dist_max",
    "eef_obj_dist_stable_var", "lift_z_threshold", "lift_sustain_steps",
    "carry_obj_z_var_max", "carry_window",
    "preplace_target_dist_min", "preplace_target_dist_max",
    "release_target_dist_max", "regrasp_eef_obj_dist_max", "stability_window",
])


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
    """Load C16 frozen teacher config. Verifies SHA, key set, no defaults."""
    if not os.path.exists(path):
        raise SystemExit("Teacher config not found: {}".format(path))
    with open(path, "rb") as f:
        raw = f.read()
    actual_sha = hashlib.sha256(raw).hexdigest()

    data = json.loads(raw.decode())
    thresh = data.get("thresholds", {})
    observed_keys = set(thresh.keys())

    # Verify exact SHA identity
    if actual_sha != C16_EXPECTED_CONFIG_SHA256:
        raise SystemExit(
            "C16 config SHA mismatch: expected {}, got {}".format(
                C16_EXPECTED_CONFIG_SHA256[:16], actual_sha[:16]))

    # Verify exact key set
    missing_keys = C16_EXPECTED_THRESHOLD_KEYS - observed_keys
    extra_keys = observed_keys - C16_EXPECTED_THRESHOLD_KEYS
    if missing_keys:
        raise SystemExit("C16 config missing thresholds: {}".format(sorted(missing_keys)))
    if extra_keys:
        raise SystemExit("C16 config has unknown thresholds: {}".format(sorted(extra_keys)))

    cfg = TeacherConfig()
    for k in C16_EXPECTED_THRESHOLD_KEYS:
        setattr(cfg, k, thresh[k])  # KeyError if missing — fail-closed
    cfg.calibrated_from = data.get("calibrated_from", "C16_frozen")
    cfg.version = data.get("version", "c16_frozen")
    return cfg, data, actual_sha


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
        # Frozen SC5 semantics:
        #   corridor_active_at_t = set of valid K10 START steps (not internal window steps)
        #   teacher_sc5_corridor_active = 1 iff this step is a valid start
        #   teacher_sc5_attack_window_active = 1 iff this step is within [anchor, anchor+K-1]

        window_start = window[0] if window else -1
        window_end = window[1] if window else -1

        for j, lbl in enumerate(labels):
            phase = lbl.get("phase", "abstain_unsupported")
            phase_idx = PHASE_TO_IDX.get(phase, 8)
            release_s = 1 if phase == "release_safe" else 0
            step_idx = lbl.get("step_idx", j)
            policy_step_idx = lbl.get("policy_step_idx", j)

            # corridor_active_at_t: valid K10 start at this step
            corridor_active = 1 if step_idx in corridor_active_set else 0
            # attack window: step is within the specific anchor's K10 window
            in_window = 1 if (teacher_valid and window_start <= step_idx <= window_end) else 0

            step_labels.append({
                "episode_key": ek,
                "step": step_idx,
                "policy_step_idx": policy_step_idx,
                "teacher_phase_idx": phase_idx,
                "teacher_phase": phase,
                "teacher_sc5_corridor_active": corridor_active,
                "teacher_sc5_attack_window_active": in_window,
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

        # Verify exact K10 attack window for valid episodes
        if teacher_valid:
            window_steps = sum(1 for lbl in labels
                              if window_start <= lbl.get("step_idx", 0) <= window_end)
            if window_steps == K_SC5:
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
                 "teacher_phase", "teacher_sc5_corridor_active",
                 "teacher_sc5_attack_window_active", "release_safe",
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

    # Real checks
    event_ek_set = set(e["episode_key"] for e in event_index)
    index_ek_set = set(r["episode_key"] for r in rows)
    event_keys_ok = event_ek_set == index_ek_set
    event_no_dupes = len(event_ek_set) == len(event_index)

    step_ek_set = set(s["episode_key"] for s in step_labels)
    step_dupes = len(step_labels) - len(set((s["episode_key"], s["step"]) for s in step_labels))
    step_no_dupes = step_dupes == 0
    step_keys_subset = step_ek_set <= index_ek_set  # teacher-invalid have no labels

    phase_ok = all(0 <= s["teacher_phase_idx"] < N_PHASES for s in step_labels)
    corridor_binary = all(s["teacher_sc5_corridor_active"] in (0, 1) for s in step_labels)
    window_binary = all(s["teacher_sc5_attack_window_active"] in (0, 1) for s in step_labels)
    release_binary = all(s["release_safe"] in (0, 1) for s in step_labels)

    # Step continuity: for each labeled episode, steps must be 0..n-1 contiguous
    from collections import defaultdict
    ep_steps = defaultdict(list)
    for s in step_labels:
        ep_steps[s["episode_key"]].append(s["step"])
    step_contiguous = True
    for ek, steps in ep_steps.items():
        steps_sorted = sorted(set(steps))
        if steps_sorted != list(range(len(steps_sorted))):
            step_contiguous = False
            break
        if len(steps) != len(steps_sorted):
            step_contiguous = False
            break

    all_checks = [
        ("event_keys_equal_index", event_keys_ok),
        ("event_no_duplicate_keys", event_no_dupes),
        ("step_keys_subset_of_index", step_keys_subset),
        ("step_no_duplicate_keys", step_no_dupes),
        ("step_contiguous_per_episode", step_contiguous),
        ("corridor_k10_all_valid", stats["corridor_k10_fail"] == 0),
        ("phase_values_in_range", phase_ok),
        ("corridor_binary", corridor_binary),
        ("window_binary", window_binary),
        ("release_binary", release_binary),
    ]
    validation_pass = all(p for _, p in all_checks)

    with open(val_path, "w") as f:
        json.dump({
            "gate": "SUPERVISION_VALIDATION_REPORT_V1",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "route_b_authorized": True,
            "config_sha_bound": config_sha,
            "src_sha_bound": teacher_src_sha,
            "index_sha_bound": index_sha,
            "checks": {name: passed for name, passed in all_checks},
            "passed": validation_pass,
        }, f, indent=2)
    print("  {}".format(val_path))

    # ── Final envelope (only written after all validation passes) ──
    if not validation_pass:
        failed_checks = [name for name, p in all_checks if not p]
        msg = "VALIDATION FAILED: {}".format(", ".join(failed_checks))
        print(msg)
        # Write FAILED envelope, not AUTHORITATIVE
        env_path = os.path.join(args.output_dir, "SUPERVISION_ENVELOPE.json")
        with open(env_path, "w") as f:
            json.dump({
                "gate": "CLEAN2000_SUPERVISION_AUTH_V1_2",
                "status": "FAILED",
                "failed_checks": failed_checks,
            }, f, indent=2)
        sys.exit(1)

    envelope = {
        "gate": "CLEAN2000_SUPERVISION_AUTH_V1_2",
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
        "sc5_corridor_start_rule": "compute_sc5_valid_start_corridor (step in corridor_active_at_t)",
        "sc5_attack_window_rule": "exact K10 [anchor, anchor+9]",
        "status": "AUTHORITATIVE",
    }
    env_path = os.path.join(args.output_dir, "SUPERVISION_ENVELOPE.json")
    with open(env_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print("  {}".format(env_path))

    print()
    print("Route B supervision: AUTHORITATIVE (historical agreement: {:.1f}% diagnostic)".format(100 * hist_rate))
    print("DONE.")


if __name__ == "__main__":
    main()
