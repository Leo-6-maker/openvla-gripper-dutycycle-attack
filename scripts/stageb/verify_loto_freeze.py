#!/usr/bin/env python3
"""LOTO Global Freeze V1 Verifier — read-only, never opens held-out labels."""
import hashlib, json, os, sys
from datetime import timezone, datetime

FREEZE_PATH = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/LOTO_GLOBAL_FREEZE_V1.json"
EVID_BASE = "/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1"

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]

def main():
    with open(FREEZE_PATH) as f:
        freeze = json.load(f)
    with open(FREEZE_PATH, "rb") as f:
        freeze_sha = hashlib.sha256(f.read()).hexdigest()

    assert freeze["gate"] == "LOTO_30_CHECKPOINT_GLOBAL_FREEZE_V1"
    assert freeze["total_checkpoints"] == 30
    assert freeze["fold00_role"] == "PILOT_DIAGNOSTIC"
    assert freeze["post_freeze_retraining_allowed"] == False
    assert freeze["post_freeze_threshold_tuning_allowed"] == False
    rt = freeze["runtime"]
    assert rt["tau_corridor"] == 0.3
    assert rt["tau_release"] == 0.3
    assert rt["guard"] == 5
    assert rt["K"] == 10

    # ── 1. Check 30 files exist, SHA match ──
    found = 0; sha_match = 0; sha_mismatch = 0; missing = 0
    fold_seeds = set(); paths_seen = set()
    duplicate_fold_seed = 0; duplicate_paths = 0

    for cp in freeze["checkpoints"]:
        fold = cp["fold"]; seed = cp["seed"]
        key = (fold, seed)
        if key in fold_seeds:
            duplicate_fold_seed += 1
        fold_seeds.add(key)

        if fold == "00":
            path = os.path.join(EVID_BASE, "strict_fold_00_combined500",
                              "training_v2_corridor_fixed", f"seed_{seed}", "best_model.pt")
        else:
            path = os.path.join(EVID_BASE, f"fold_{fold}", "training_v3",
                              f"seed_{seed}", "best_model.pt")

        if path in paths_seen:
            duplicate_paths += 1
        paths_seen.add(path)

        if not os.path.exists(path):
            missing += 1
            print(f"MISSING: {path}")
            continue
        found += 1

        with open(path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        if actual_sha == cp["sha256"]:
            sha_match += 1
        else:
            sha_mismatch += 1
            print(f"SHA MISMATCH Fold{fold} seed{seed}: expected={cp['sha256'][:16]}... actual={actual_sha[:16]}...")

    # ── 2. Check checkpoint metadata ──
    import torch
    test_accessed_any = False; metadata_ok = 0; metadata_issues = []
    feature_schema_ok = 0; phase_schema_ok = 0
    for cp in freeze["checkpoints"]:
        fold = cp["fold"]; seed = cp["seed"]
        if fold == "00":
            path = os.path.join(EVID_BASE, "strict_fold_00_combined500",
                              "training_v2_corridor_fixed", f"seed_{seed}", "best_model.pt")
        else:
            path = os.path.join(EVID_BASE, f"fold_{fold}", "training_v3",
                              f"seed_{seed}", "best_model.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("test_accessed", False):
            test_accessed_any = True
            print(f"TEST_ACCESSED Fold{fold} seed{seed}")
        feat = ckpt.get("feature_names", [])
        phases = ckpt.get("phase_classes", [])
        if feat == SC5_FEATURES:
            feature_schema_ok += 1
        else:
            metadata_issues.append(f"Fold{fold} s{seed}: feature schema mismatch")
        if phases == SC5_PHASES:
            phase_schema_ok += 1
        else:
            metadata_issues.append(f"Fold{fold} s{seed}: phase schema mismatch")

    # ── 3. Check held-out open event does NOT exist ──
    open_event_path = os.path.join(EVID_BASE, "LOTO_TEST_OPEN_EVENT_V1.json")
    heldout_open_event_exists = os.path.exists(open_event_path)

    # ── 4. Assess ──
    all_pass = (
        found == 30 and sha_match == 30 and sha_mismatch == 0 and missing == 0
        and duplicate_fold_seed == 0 and duplicate_paths == 0
        and not test_accessed_any and not heldout_open_event_exists
        and feature_schema_ok == 30 and phase_schema_ok == 30
        and len(metadata_issues) == 0
    )

    report = {
        "gate": "LOTO_GLOBAL_FREEZE_V1_VERIFY",
        "verifier_script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "freeze_json_sha256": freeze_sha,
        "freeze_git_commit": "2a6a9c93013a07b924fe05a949b2f73bd51df773",
        "freeze_git_blob_sha": "c5ffccbad90d8a6b19a2c3b5d935bf238400c138",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": {
            "expected_checkpoints": 30,
            "found_checkpoints": found,
            "sha_match": sha_match,
            "sha_mismatch": sha_mismatch,
            "missing": missing,
            "duplicate_fold_seed": duplicate_fold_seed,
            "duplicate_paths": duplicate_paths,
            "runtime_match": True,
            "feature_schema_match": feature_schema_ok == 30,
            "phase_schema_match": phase_schema_ok == 30,
            "test_accessed_any": test_accessed_any,
            "heldout_open_event_exists": heldout_open_event_exists,
            "metadata_issues": metadata_issues,
        },
        "overall": "ALL_PASS" if all_pass else "FAIL",
    }

    out_path = os.path.join(EVID_BASE, "LOTO_GLOBAL_FREEZE_V1_VERIFY.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(out_path, "rb") as f:
        report_sha = hashlib.sha256(f.read()).hexdigest()

    print(json.dumps(report["results"], indent=2))
    print(f"\nOverall: {report['overall']}")
    print(f"Report: {out_path}")
    print(f"Report SHA256: {report_sha}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
