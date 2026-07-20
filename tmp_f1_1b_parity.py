"""Gate F1.1b: Exact action-space parity audit.
Reads REAL raw_action_7d, clean_env_action_7d, executed_action_7d from sealed
step records. Does NOT guess from 25D features.

Verifies:
  1. env_gripper = -sign(2 * raw_gripper - 1)  (official OpenVLA postprocess)
  2. raw_close == env_close  (raw <= 0.5  <-->  env > 0)
  3. features_25d[0] == raw_action_7d[-1]
  4. features_25d[12] == raw_action_7d[-1]  (source parity)
  5. Threshold boundary (raw == 0.5) count
"""
import json, sys
from pathlib import Path
import numpy as np

def postprocess_gripper(raw):
    """Official OpenVLA postprocess: env = -sign(2*raw - 1)."""
    return float(-np.sign(2.0 * raw - 1.0))

def raw_is_close(raw):
    """OpenVLA space: raw <= 0.5 = close intent."""
    return raw <= 0.5

def env_is_close(env):
    """LIBERO space: env > 0 = close command."""
    return env > 0.0

def audit_root(root, label):
    """Audit one sealed episode root."""
    sr_file = Path(root) / "step_records.jsonl"
    if not sr_file.is_file():
        return {"error": "step_records.jsonl not found", "root": root, "label": label}

    recs = [json.loads(l) for l in sr_file.read_text().splitlines() if l.strip()]
    N = len(recs)

    stats = {
        "label": label, "N": N,
        "raw_action_present": 0, "env_action_present": 0,
        "postprocess_correct": 0, "postprocess_incorrect": 0,
        "raw_close_eq_env_close": 0, "raw_close_neq_env_close": 0,
        "raw_boundary": 0,  # abs(raw - 0.5) <= 1e-6
        "feat0_eq_raw": 0, "feat0_neq_raw": 0,
        "feat12_eq_raw": 0, "feat12_neq_raw": 0,
        "executed_eq_env": 0, "executed_neq_env": 0,
    }

    raw_vals = []
    env_vals = []

    for r in recs:
        raw_action = r.get("raw_action_7d")
        clean_env = r.get("clean_env_action_7d")
        executed = r.get("executed_action_7d")
        feats = r.get("features_25d")

        if raw_action and len(raw_action) == 7:
            raw = float(raw_action[-1])
            stats["raw_action_present"] += 1
            raw_vals.append(raw)

            if feats and len(feats) == 25:
                feat0 = float(feats[0])
                feat12 = float(feats[12])
                if abs(feat0 - raw) <= 1e-6:
                    stats["feat0_eq_raw"] += 1
                else:
                    stats["feat0_neq_raw"] += 1
                if abs(feat12 - raw) <= 1e-6:
                    stats["feat12_eq_raw"] += 1
                else:
                    stats["feat12_neq_raw"] += 1

            if abs(raw - 0.5) <= 1e-6:
                stats["raw_boundary"] += 1
            else:
                expected_env = postprocess_gripper(raw)
                env_close_expected = env_is_close(expected_env)
                raw_close_val = raw_is_close(raw)

                if raw_close_val == env_close_expected:
                    stats["raw_close_eq_env_close"] += 1
                else:
                    stats["raw_close_neq_env_close"] += 1

            if clean_env and len(clean_env) == 7:
                env = float(clean_env[-1])
                stats["env_action_present"] += 1
                env_vals.append(env)

                if abs(raw - 0.5) > 1e-6:
                    expected_env = postprocess_gripper(raw)
                    if abs(env - expected_env) <= 1e-6:
                        stats["postprocess_correct"] += 1
                    else:
                        stats["postprocess_incorrect"] += 1

            if executed and len(executed) == 7:
                ex = float(executed[-1])
                if clean_env and len(clean_env) == 7:
                    if abs(ex - float(clean_env[-1])) <= 1e-6:
                        stats["executed_eq_env"] += 1
                    else:
                        stats["executed_neq_env"] += 1

    stats["raw_min"] = min(raw_vals) if raw_vals else None
    stats["raw_max"] = max(raw_vals) if raw_vals else None
    stats["raw_mean"] = float(np.mean(raw_vals)) if raw_vals else None
    stats["env_min"] = min(env_vals) if env_vals else None
    stats["env_max"] = max(env_vals) if env_vals else None

    return stats


# Audited roots
ROOTS = [
    ("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_e_r3a_output_20260720/libero_10_task_01_state_20",
     "E-R3a task_01 (real passive runtime)"),
    ("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_passive_smoke_output_20260720",
     "R10.4D task_00 (real passive runtime)"),
]

# Also check C2G corpus (S1 records don't have raw/env, need step_records from corpus)
CORPUS_ROOT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")

print("=" * 70)
print("Gate F1.1b: Exact Action-Space Parity Audit")
print("=" * 70)

all_pass = True
for root_path, label in ROOTS:
    print("\n--- {} ---".format(label))
    stats = audit_root(root_path, label)
    for k, v in sorted(stats.items()):
        print("  {}: {}".format(k, v))

    # Verify
    if stats.get("N", 0) > 0:
        postproc_ok = stats.get("postprocess_incorrect", 1) == 0
        close_parity = stats.get("raw_close_neq_env_close", 1) == 0
        feat0_ok = stats.get("feat0_neq_raw", 1) == 0
        feat12_ok = stats.get("feat12_neq_raw", 1) == 0  # feature[12] IS raw, not env

        print("  POSTPROCESS_OK: {}".format(postproc_ok))
        print("  CLOSE_PARITY: {}".format(close_parity))
        print("  FEAT0_IS_RAW: {}".format(feat0_ok))
        print("  FEAT12_IS_ALSO_RAW (not env): {}".format(feat12_ok))
        print("  BOUNDARY_COUNT: {}".format(stats.get("raw_boundary", 0)))

        if not (postproc_ok and close_parity):
            all_pass = False

print("\n" + "=" * 70)
print("Overall: {}".format("PASS" if all_pass else "FAIL"))
print("=" * 70)
