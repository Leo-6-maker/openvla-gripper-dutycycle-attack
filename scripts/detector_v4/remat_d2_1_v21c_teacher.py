#!/usr/bin/env python3
"""Gate D2.1: V2.1C Teacher rematerialization — fix threshold only.

Keeps ALL formulas, thresholds, history, utility weights unchanged.
Only change: _candidate_close uses raw < 0.5 (fixed from raw >= 0.5).

Compares old V2.1 vs corrected V2.1C on fold-0 multi-object episodes.
"""

import json, sys
from pathlib import Path
from collections import defaultdict

OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
CLEAN = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
OLD_TEACHER = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels"

sys.path.insert(0, "/tmp")
from v5_physics import _candidate_close, derive_episode_rows, parse_bddl_task_role


def jsonl(path):
    if not path.is_file():
        raise SystemExit("FILE_MISSING:{}".format(path))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("FILE_EMPTY:{}".format(path))
    return [json.loads(l) for l in lines]


# Load protocol from old Teacher artifact
protocol = json.loads((OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/protocol.json").read_text())
print("Protocol: {}  schema={}".format(OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/protocol.json", protocol.get("schema")))

# Load fold-0 validation identities
manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]
print("Fold-0 val multi_object: {} episodes".format(len(val_ids)))

# Process first 10 episodes for comparison
results = []
for identity in val_ids[:10]:
    parts = identity.split("/")
    suite, task_name, state_name = parts

    step_path = CLEAN / suite / task_name / state_name / "step_records.jsonl"
    old_path = OLD_TEACHER / suite / task_name / state_name / "physics_teacher_v21.jsonl"

    if not step_path.is_file() or not old_path.is_file():
        print("  SKIP {}: missing files".format(identity))
        continue

    step_recs = jsonl(step_path)
    old_labels = jsonl(old_path)

    # Compare candidate_close only (full rematerialization needs sidecar data)
    old_cc = [bool(l["candidate_close"]) for l in old_labels]
    new_cc = [_candidate_close(r, 0.5) for r in step_recs]

    T = len(step_recs)
    same = sum(1 for i in range(T) if old_cc[i] == new_cc[i])
    opp = sum(1 for i in range(T) if old_cc[i] != new_cc[i])
    inv = sum(1 for i in range(T) if old_cc[i] == (not new_cc[i]))

    results.append({
        "identity": identity, "T": T, "old_cc_true": sum(old_cc),
        "new_cc_true": sum(new_cc), "same": same, "opposite": opp,
        "inverted": inv, "inversion_pct": round(100.0 * inv / T, 2),
    })

    print("  {}: T={} old={} new={} inverted={}/{} ({:.1f}%)".format(
        identity, T, sum(old_cc), sum(new_cc), inv, T, 100.0*inv/T))

# Summary
total_steps = sum(r["T"] for r in results)
total_inv = sum(r["inverted"] for r in results)
print("\nSummary ({} episodes, {} steps):".format(len(results), total_steps))
print("  Total inversion: {}/{} ({:.1f}%)".format(total_inv, total_steps, 100.0*total_inv/total_steps))
print("  V2.1C fix: raw < 0.5 replaces raw >= 0.5")
print("  NOTE: Full rematerialization (windows, scores, tiers) requires sidecar replay.")
print("  This audit only validates candidate_close threshold fix on step records.")

# Old cc was 1 for "raw >= 0.5" steps → OPEN region
# New cc is 1 for "raw < 0.5" steps → CLOSE region
# These are exact complements for non-boundary steps
old_open_as_close = sum(r["old_cc_true"] for r in results)
new_close = sum(r["new_cc_true"] for r in results)
print("  Old: {} steps marked close (actually OPEN in [0,1] space)".format(old_open_as_close))
print("  New: {} steps marked close (actually CLOSE, raw < 0.5)".format(new_close))
