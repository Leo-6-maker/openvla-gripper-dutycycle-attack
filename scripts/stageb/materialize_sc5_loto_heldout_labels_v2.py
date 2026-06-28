#!/usr/bin/env python3
"""
Phase B: Materialize held-out teacher labels V2.

Requires --test_open_event (verifies Phase B authorization + 27 checkpoint SHAs).
Requires --fold_manifest (verifies teacher_config SHA).
Restores TeacherConfig via DIRECT KEY ACCESS (no .get, no silent default).
Preserves original step_idx.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
from datetime import timezone, datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor
)

TASKS = ["alphabet_soup","cream_cheese","salad_dressing","bbq_sauce","ketchup",
         "tomato_sauce","butter","milk","chocolate_pudding","orange_juice"]
WAVE1_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave1_50_0280c85_20260627T175204Z"
WAVE2_ROOT = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave2_remaining_states_0280c85_20260627T183812Z"
K_SC5 = 10; GUARD_SC5 = 5

# TeacherConfig fields that are thresholds (all others are metadata)
THRESHOLD_FIELDS = [
    "grasp_close_sustain", "grasp_open_proxy_max", "eef_obj_dist_max",
    "eef_obj_dist_stable_var", "lift_z_threshold", "lift_sustain_steps",
    "carry_obj_z_var_max", "carry_window",
    "preplace_target_dist_min", "preplace_target_dist_max",
    "release_target_dist_max", "regrasp_eef_obj_dist_max", "stability_window",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--fold_manifest", required=True)
    ap.add_argument("--test_open_event", required=True,
                    help="LOTO_TEST_OPEN_EVENT_V1.json (Phase B authorization)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)

    fold_id = args.fold
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Verify test_open_event ──
    with open(args.test_open_event) as f:
        event = json.load(f)
    assert event["gate"] == "LOTO_TEST_OPEN_EVENT_V1", "Invalid open event"
    assert event.get("phase_b_authorized", False), "Phase B not authorized"
    assert fold_id in event.get("folds_authorized", []), \
        "Fold %s not authorized in open event" % fold_id

    # Verify protocol SHA consistency
    assert event.get("protocol_sha256") == manifest["protocol_sha256"], \
        "Event protocol SHA mismatch"

    # Verify 27 checkpoint SHAs frozen (Fold 01-09 × 3 seeds)
    checkpoints = event.get("checkpoints", [])
    expected_cks = {(f"{f:02d}", s) for f in range(1, 10) for s in (1, 2, 3)}
    observed_cks = {(c["fold"], int(c["seed"])) for c in checkpoints}
    assert observed_cks == expected_cks, \
        "Checkpoint set mismatch: missing=%s extra=%s" % (
            expected_cks - observed_cks, observed_cks - expected_cks)
    assert len(checkpoints) == 27
    for c in checkpoints:
        assert len(c["sha256"]) == 64, "Invalid SHA length for fold %s seed %s" % (c["fold"], c["seed"])

    with open(args.test_open_event, "rb") as f:
        event_sha = hashlib.sha256(f.read()).hexdigest()

    # ── 2. Load fold manifest, verify teacher_config SHA ──
    with open(args.fold_manifest) as f: manifest = json.load(f)
    assert manifest["fold"] == fold_id
    test_task = manifest["test_task"]
    tc_path = os.path.join(os.path.dirname(args.fold_manifest),
                           "FOLD%s_teacher_config.json" % fold_id)
    with open(tc_path) as f: tc = json.load(f)
    with open(tc_path, "rb") as f: actual_tc_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_tc_sha == manifest["teacher_config_sha256"], \
        "Teacher config SHA mismatch: actual=%s manifest=%s" % (actual_tc_sha[:16], manifest["teacher_config_sha256"][:16])

    # ── 3. Restore TeacherConfig — DIRECT INDEX, no .get ──
    th = tc["thresholds"]
    cfg = TeacherConfig()
    for name in THRESHOLD_FIELDS:
        setattr(cfg, name, th[name])  # KeyError if missing — no silent default
    cfg.calibrated_from = tc["calibrated_from"]
    cfg.version = tc.get("version", "loto_fold%s" % fold_id)

    teacher = V2PrivilegedTeacher(cfg)

    # ── 4. Generate heldout labels + anchors ──
    all_labels = []; episode_anchors = []

    for s in range(50):
        wave_root = WAVE1_ROOT if s <= 4 else WAVE2_ROOT
        priv_path = os.path.join(wave_root, "jobs", "task_%d_%s" % (test_task, TASKS[test_task]),
                                 "state_%d" % s, "attempt_1", "privileged_step_records.jsonl")
        with open(priv_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        labels = teacher.label_trajectory(records)
        ep_labels = []
        for idx, lab in enumerate(labels):
            if lab is None: continue
            # Preserve original step_idx
            orig_step = int(lab["step_idx"])
            raw_step = int(records[idx].get("step_idx", idx))
            assert orig_step == raw_step
            lab["task_idx"] = test_task; lab["task_name"] = TASKS[test_task]
            lab["state_id"] = s; lab["split"] = "held_out"
            all_labels.append(lab); ep_labels.append(lab)

        sc5 = find_sc5_anchor_v2(ep_labels, K=K_SC5, guard=GUARD_SC5)
        corridor_active = []
        if sc5["valid"]:
            corridor_active = sorted(compute_sc5_valid_start_corridor(
                ep_labels, sc5["anchor"], K=K_SC5)["corridor_active_at_t"])

        episode_anchors.append({
            "task_idx": test_task, "state_id": s,
            "sc5_anchor": sc5["anchor"], "sc5_valid": sc5["valid"],
            "sc5_reason": sc5.get("reason", ""),
            "stable_carry_start": sc5.get("stable_carry_start", -1),
            "corridor_active_at_t": corridor_active,
            "corridor_first": corridor_active[0] if corridor_active else -1,
            "corridor_last": corridor_active[-1] if corridor_active else -1,
        })

    # ── 5. Write outputs with provenance ──
    labels_path = out_dir / ("FOLD%s_teacher_labels_heldout.jsonl" % fold_id)
    with open(labels_path, "w") as f:
        for lab in all_labels:
            f.write(json.dumps(lab) + "\n")
    with open(labels_path, "rb") as f:
        labels_sha = hashlib.sha256(f.read()).hexdigest()

    anchors_path = out_dir / ("FOLD%s_heldout_episode_anchors.json" % fold_id)
    anchor_data = {
        "fold": fold_id, "test_task": test_task, "test_task_name": TASKS[test_task],
        "episodes": episode_anchors,
        "teacher_config_sha256": actual_tc_sha,
        "test_open_event_sha256": event_sha,
        "fold_manifest_sha256": hashlib.sha256(open(args.fold_manifest, "rb").read()).hexdigest(),
        "materializer_script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "heldout_labels_sha256": labels_sha,
        "creation_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(anchors_path, "w") as f: json.dump(anchor_data, f, indent=2)
    with open(anchors_path, "rb") as f:
        anchors_sha = hashlib.sha256(f.read()).hexdigest()

    pos_eps = sum(1 for a in episode_anchors if a["sc5_valid"])
    print("Fold %s heldout: %d rows, %d episodes, %d teacher-positive" % (
        fold_id, len(all_labels), len(episode_anchors), pos_eps))
    print("  Labels SHA: %s" % labels_sha[:16])
    print("  Anchors SHA: %s" % anchors_sha[:16])


if __name__ == "__main__":
    main()
