"""Check Teacher V2.1 labels for B3 false-trigger episodes and run V2.0 conflict census."""
import json, sys
from collections import defaultdict

V21_ROOT = "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V4_TEACHER_V21_5e27d7c_20260718"
V20_ROOT = "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V4_CANDIDATE_WINDOWS_V1_5e27d7c"

FALSE_TRIGGERS = [
    "libero_goal/task_07/state_04",
    "libero_goal/task_09/state_03",
    "libero_object/task_05/state_04",
]

print("=" * 60)
print("V2.1 LABELS FOR B3 FALSE-TRIGGER EPISODES")
print("=" * 60)

for cid in FALSE_TRIGGERS:
    parts = cid.split("/")
    path = f"{V21_ROOT}/{parts[0]}/{parts[1]}/{parts[2]}/teacher_v21_labels.jsonl"
    labels = [json.loads(l) for l in open(path)]

    n_q = sum(1 for l in labels if l["quality_valid"])
    n_v = sum(1 for l in labels if l["veto_invalid"])
    n_c = sum(1 for l in labels if l["candidate_close"])
    n_k = sum(1 for l in labels if l["known_mask"])

    phases_seen = defaultdict(int)
    for l in labels:
        phases_seen[l["phase"]] += 1

    print(f"\n{cid}:")
    print(f"  steps={len(labels)} quality_valid={n_q} veto_invalid={n_v} candidate_close={n_c} known={n_k}")
    print(f"  phases: {dict(phases_seen)}")

    # Show step ranges for each phase
    current_phase = None
    phase_start = 0
    for i, l in enumerate(labels):
        if l["phase"] != current_phase:
            if current_phase is not None and current_phase != "NO_CLOSE":
                print(f"    {current_phase}: steps {phase_start}-{i-1} (duration={i-phase_start})")
            current_phase = l["phase"]
            phase_start = i
    if current_phase is not None and current_phase != "NO_CLOSE":
        print(f"    {current_phase}: steps {phase_start}-{len(labels)-1} (duration={len(labels)-phase_start})")

print("\n" + "=" * 60)
print("V2.0 CONFLICT CENSUS (criticality=1 AND veto=1)")
print("=" * 60)

# Sample 3 episodes from each fold to check V2.0 conflict rate
total_conflict = 0
total_crit_and_veto = 0
samples_checked = 0

for fold_id in range(4):
    states = list(range(fold_id * 5, (fold_id + 1) * 5))
    fold_conflict = 0
    fold_both = 0
    for suite in ["libero_10", "libero_goal"]:
        for task in range(3):
            for state in states[:2]:
                cid = f"{suite}/task_{task:02d}/state_{state:02d}"
                path = f"{V20_ROOT}/{suite}/task_{task:02d}/state_{state:02d}/teacher_v2_labels.jsonl"
                try:
                    labels = [json.loads(l) for l in open(path)]
                except FileNotFoundError:
                    continue
                samples_checked += 1
                for l in labels:
                    crit = l.get("critical_retention_window", False) or l.get("valid_retention", False)
                    veto = l.get("false_trigger_veto", False)
                    if crit and veto:
                        fold_conflict += 1
                    if crit:
                        fold_both += 1
    total_conflict += fold_conflict
    total_crit_and_veto += fold_both
    print(f"  Fold {fold_id}: {fold_conflict} conflict steps (out of {fold_both} crit-positive steps in sample)")

print(f"\n  Total V2.0 conflicts in sample: {total_conflict}")
print(f"  Episodes checked: {samples_checked}")
