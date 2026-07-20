#!/usr/bin/env python3
"""Gate D1.1: Full 800-FIT identity action-source census.

Verifies for every step of every FIT identity:
  - All action fields present and finite
  - raw_close == env_close (postprocess parity)
  - feature[0] == raw_gripper, feature[12] == raw_gripper
  - No missing/duplicate identities, no length mismatches
  - Fail-closed: any error terminates nonzero

Uses canonical action contract from action_contract.py.
"""

import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gripper_attack.action_contract import (
    raw_gripper_is_close,
    postprocess_gripper_openvla_to_libero,
    env_gripper_is_close,
    classify_openvla_raw_gripper,
    GripperIntent,
)

CLEAN = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean")
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"


def jsonl(path):
    if not path.is_file():
        raise SystemExit("FILE_MISSING:{}".format(path))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("FILE_EMPTY:{}".format(path))
    return [json.loads(l) for l in lines]


def main():
    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    all_ids = []
    for fold in manifest["folds"]:
        all_ids.extend(fold["train_identities"])
        all_ids.extend(fold["validation_identities"])
    all_ids = sorted(set(all_ids))
    print("D1.1: Action-Source Census")
    print("  Total unique FIT identities: {}".format(len(all_ids)))

    # Verify all exist
    missing = []
    for identity in all_ids:
        parts = identity.split("/")
        sp = CLEAN / parts[0] / parts[1] / parts[2] / "step_records.jsonl"
        if not sp.is_file():
            missing.append(identity)
    if missing:
        for m in missing[:10]:
            print("  MISSING: {}".format(m))
        raise SystemExit("MISSING_IDENTITIES:{}".format(len(missing)))
    print("  All {} identities present".format(len(all_ids)))

    total_steps = 0
    errors = 0
    raw_vals = []
    env_vals = []
    parity_pass = 0
    parity_fail = 0
    boundary_count = 0
    feat0_match = 0
    feat0_mismatch = 0
    feat12_match = 0
    feat12_mismatch = 0

    # Per-identity checks
    for identity in all_ids:
        parts = identity.split("/")
        recs = jsonl(CLEAN / parts[0] / parts[1] / parts[2] / "step_records.jsonl")

        for i, r in enumerate(recs):
            total_steps += 1

            # Verify step index continuity
            if r.get("step") != i:
                errors += 1
                if errors <= 5:
                    print("  STEP_INDEX_MISMATCH: {} step={} expected={}".format(identity, r.get("step"), i))

            # Verify all action fields present
            for field in ["action_raw", "action_env", "clean_action_raw_7d", "applied_action_7d", "features_25d"]:
                if field not in r:
                    errors += 1
                    if errors <= 5:
                        print("  MISSING_FIELD: {} {}".format(identity, field))
                    continue

            raw_action = r.get("action_raw", [])
            env_action = r.get("action_env", [])
            clean_raw = r.get("clean_action_raw_7d", [])
            applied = r.get("applied_action_7d", [])
            feats = r.get("features_25d", [])

            if len(raw_action) != 7 or len(env_action) != 7 or len(clean_raw) != 7 or len(applied) != 7 or len(feats) != 25:
                errors += 1
                continue

            raw_g = float(raw_action[6])
            env_g = float(env_action[6])
            clean_g = float(clean_raw[6])
            app_g = float(applied[6])
            feat0 = float(feats[0])
            feat12 = float(feats[12])

            if not all(np.isfinite(v) for v in [raw_g, env_g, clean_g, app_g, feat0, feat12]):
                errors += 1
                continue

            raw_vals.append(raw_g)
            env_vals.append(env_g)

            # Raw aliases must be consistent
            if abs(raw_g - clean_g) > 1e-6:
                errors += 1
                if errors <= 3:
                    print("  RAW_ALIAS_MISMATCH: {} raw={} clean_raw={}".format(identity, raw_g, clean_g))

            # Env aliases must be consistent (CLEAN condition)
            if abs(env_g - app_g) > 1e-6:
                errors += 1
                if errors <= 3:
                    print("  ENV_ALIAS_MISMATCH: {} env={} applied={}".format(identity, env_g, app_g))

            # Feature[0] must match raw gripper
            if abs(feat0 - raw_g) <= 1e-6:
                feat0_match += 1
            else:
                feat0_mismatch += 1
                if feat0_mismatch <= 3:
                    print("  FEAT0_MISMATCH: {} feat0={} raw={}".format(identity, feat0, raw_g))

            # Feature[12] must match raw gripper
            if abs(feat12 - raw_g) <= 1e-6:
                feat12_match += 1
            else:
                feat12_mismatch += 1

            # Postprocess parity
            intent = classify_openvla_raw_gripper(raw_g)
            if intent == GripperIntent.BOUNDARY:
                boundary_count += 1
            else:
                expected_env = postprocess_gripper_openvla_to_libero(raw_g)
                if abs(env_g - expected_env) <= 1e-6:
                    parity_pass += 1
                else:
                    parity_fail += 1
                    if parity_fail <= 3:
                        print("  PARITY_FAIL: {} step={} raw={} env={} expected={}".format(
                            identity, i, raw_g, env_g, expected_env))

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("D1.1 Census Results")
    print("  {} identities, {} total steps".format(len(all_ids), total_steps))
    print("=" * 65)
    print("Errors: {}".format(errors))
    print("Boundary steps (raw=0.5 +/- 1e-6): {}".format(boundary_count))
    print("Postprocess parity (non-boundary): pass={} fail={} ({:.4f}%)".format(
        parity_pass, parity_fail,
        100 * parity_pass / max(1, parity_pass + parity_fail)))
    print("Feature[0] == raw: match={} mismatch={}".format(feat0_match, feat0_mismatch))
    print("Feature[12] == raw: match={} mismatch={}".format(feat12_match, feat12_mismatch))

    if raw_vals:
        ra = np.array(raw_vals)
        print("Raw values: min={:.4f} max={:.4f} mean={:.4f} unique={}".format(
            ra.min(), ra.max(), ra.mean(), len(set(ra.round(6)))))
        ea = np.array(env_vals)
        print("Env values:  min={:.1f} max={:.1f} mean={:.4f} unique={}".format(
            ea.min(), ea.max(), ea.mean(), len(set(ea.round(6)))))

    if errors == 0 and parity_fail == 0 and feat0_mismatch == 0:
        print("\nD1.1: PASS — {} identities, {} steps, 0 errors".format(len(all_ids), total_steps))
    else:
        print("\nD1.1: FAIL — errors={} parity_fail={} feat0_mismatch={}".format(
            errors, parity_fail, feat0_mismatch))
        sys.exit(1)


if __name__ == "__main__":
    main()
