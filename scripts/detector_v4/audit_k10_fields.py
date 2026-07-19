#!/usr/bin/env python3
"""R7.1-A: Field availability audit from sealed S1 Physics Teacher V2.1 records."""

import json, sys
from collections import defaultdict
from pathlib import Path

S1 = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_S1_FIT_V1_5e27d7c")
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FIT_STATES = list(range(0, 20))
N_TASKS = 10

# ── Field census ───────────────────────────────────────────────────────
def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

# Read all records and build field census
field_stats = defaultdict(lambda: {
    "dtype": None, "known_count": 0, "unknown_count": 0, "true_count": 0,
    "false_count": 0, "none_count": 0, "per_task_known": defaultdict(int),
    "per_task_total": defaultdict(int), "sample_values": [],
})

total_steps = 0
total_episodes = 0
per_task_episodes = defaultdict(int)
per_task_feasible_episodes = defaultdict(int)  # any close event

for suite in SUITES:
    for task in range(N_TASKS):
        for state in FIT_STATES:
            path = S1 / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_retention_records.jsonl"
            if not path.exists():
                continue
            total_episodes += 1
            task_key = f"{suite}/t{task:02d}"
            per_task_episodes[task_key] += 1
            records = jsonl(path)

            has_close = any(r.get("event_close_onset") for r in records)
            if has_close:
                per_task_feasible_episodes[task_key] += 1

            for r in records:
                total_steps += 1
                for k, v in r.items():
                    fs = field_stats[k]
                    if fs["dtype"] is None:
                        fs["dtype"] = type(v).__name__

                    if v is None:
                        fs["none_count"] += 1
                    elif isinstance(v, bool):
                        if v:
                            fs["true_count"] += 1
                        else:
                            fs["false_count"] += 1

                    # Known vs unknown
                    if k.endswith("_mask") and v is True:
                        fs["unknown_count"] += 1
                    elif not k.endswith("_mask"):
                        fs["known_count"] += 1

                    fs["per_task_total"][task_key] += 1
                    if v is not None and v is not False:
                        fs["per_task_known"][task_key] += 1

                    if len(fs["sample_values"]) < 5:
                        fs["sample_values"].append(v)

# ── Report ─────────────────────────────────────────────────────────────
print(f"=== R7.1-A FIELD CENSUS ===")
print(f"Episodes: {total_episodes}")
print(f"Total steps: {total_steps}")
print(f"\nFields found: {len(field_stats)}")
print(f"\n{'Field':<35} {'Dtype':<12} {'Known':>8} {'None':>8} {'True':>8} {'False':>8}")
print("-" * 85)

for field in sorted(field_stats):
    fs = field_stats[field]
    dtype = fs["dtype"] or "?"
    known = fs["known_count"]
    none_c = fs["none_count"]
    true_c = fs["true_count"]
    false_c = fs["false_count"]
    print(f"{field:<35} {dtype:<12} {known:>8} {none_c:>8} {true_c:>8} {false_c:>8}")

# ── Per-task coverage of key fields ──
print(f"\n=== PER-TASK CLOSE EVENT COVERAGE ===")
print(f"{'Task':<20} {'Episodes':>8} {'HasClose':>10} {'ClosePct':>8}")
for tk in sorted(per_task_episodes):
    n = per_task_episodes[tk]
    fc = per_task_feasible_episodes.get(tk, 0)
    print(f"{tk:<20} {n:>8} {fc:>10} {fc/n*100:>7.0f}%")

# ── Field summary for K10 labeler ──
print(f"\n=== K10 LABELER FIELD AVAILABILITY ===")
required = {
    "candidate_close": ["event_close_onset", "event_end_step"],
    "label_known": ["retention_unknown_mask", "event_evidence_valid"],
    "stable_grasp": ["event_support"],
    "contact_evidence": ["grasp_support"],
    "manipulation_active": ["retention_active"],
    "release_safe": ["release_imminent", "event_release_onset"],
    "segment_identity": ["event_id", "event_start_step", "event_end_step"],
}

for concept, fields in required.items():
    available = [f for f in fields if f in field_stats]
    missing = [f for f in fields if f not in field_stats]
    status = "FULL" if not missing else f"MISSING: {missing}"
    print(f"  {concept:<25}: {status}")
    for f in available:
        fs = field_stats[f]
        print(f"    {f}: dtype={fs['dtype']} known={fs['known_count']} none={fs['none_count']} true={fs['true_count']}")

# ── Schema version ──
print(f"\n=== SCHEMA ===")
sample_path = S1 / "libero_10" / "task_00" / "state_00" / "teacher_retention_records.jsonl"
sample = jsonl(sample_path)[0]
print(f"  schema: {sample.get('schema', 'UNKNOWN')}")
print(f"  feature_rebuilder_sha256: {sample.get('feature_rebuilder_sha256', 'UNKNOWN')}")
