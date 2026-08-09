"""G2: Build 32-episode canary manifest — diverse mechanisms across 4 suites."""
import json, os, sys

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
OUT_PATH = os.path.join(os.path.dirname(__file__), 'canary_32_manifest.json')

# Pre-selected episodes covering diverse mechanisms:
# Each suite: 8 episodes = 2 tasks × 4 states
# Mechanisms: stable-grasp positive, transport positive, known negative,
#             safe-release, instability, unsupported articulated unknown

CANARY_SELECTION = {
    "libero_10": [
        # task_00: alphabet_soup + tomato_sauce (pick-place, known positive from Pilot V3)
        {"task": "task_00", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 80.5% crit"},
        {"task": "task_00", "state": "state_10", "mechanism": "transport_positive", "note": "early state for transport phase"},
        {"task": "task_00", "state": "state_42", "mechanism": "transport_positive", "note": "late state"},
        {"task": "task_00", "state": "state_05", "mechanism": "known_negative", "note": "early state, likely pre-grasp"},
        # task_02: 2 objects, known negative from Pilot V3
        {"task": "task_02", "state": "state_35", "mechanism": "known_negative", "note": "Pilot V3: 9.0% crit, 91% known-neg"},
        {"task": "task_02", "state": "state_10", "mechanism": "known_negative", "note": "early state"},
        {"task": "task_02", "state": "state_05", "mechanism": "short_horizon_negative", "note": "early state, short K10"},
        {"task": "task_02", "state": "state_45", "mechanism": "safe_release", "note": "late state, likely release phase"},
    ],
    "libero_goal": [
        # task_01: put bowl on stove (pick-place, physical binding)
        {"task": "task_01", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 61.5% crit"},
        {"task": "task_01", "state": "state_10", "mechanism": "transport_positive", "note": "mid state"},
        # task_06: put cream cheese in bowl
        {"task": "task_06", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 67.7% crit"},
        {"task": "task_06", "state": "state_10", "mechanism": "transport_positive", "note": "mid state"},
        # task_00: open middle drawer (articulated, unsupported)
        {"task": "task_00", "state": "state_35", "mechanism": "unsupported_articulated_unknown", "note": "Pilot V3: 0% crit, 100% unknown"},
        {"task": "task_00", "state": "state_10", "mechanism": "unsupported_articulated_unknown", "note": "mid state"},
        # task_07: turn on stove (articulated, unsupported)
        {"task": "task_07", "state": "state_35", "mechanism": "unsupported_articulated_unknown", "note": "Pilot V3: 0% crit, 100% unknown"},
        {"task": "task_09", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 52.0% crit (wine bottle)"},
    ],
    "libero_object": [
        {"task": "task_05", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 31.8% crit"},
        {"task": "task_05", "state": "state_10", "mechanism": "transport_positive"},
        {"task": "task_00", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 51.2% crit"},
        {"task": "task_00", "state": "state_10", "mechanism": "known_negative", "note": "early state"},
        {"task": "task_06", "state": "state_35", "mechanism": "instability_contact_loss"},
        {"task": "task_06", "state": "state_10", "mechanism": "known_negative"},
        {"task": "task_02", "state": "state_35", "mechanism": "critical_cc_false"},
        {"task": "task_04", "state": "state_35", "mechanism": "safe_release"},
    ],
    "libero_spatial": [
        {"task": "task_06", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 54.1% crit"},
        {"task": "task_06", "state": "state_10", "mechanism": "transport_positive"},
        {"task": "task_00", "state": "state_35", "mechanism": "stable_grasp_positive", "note": "Pilot V3: 22.0% crit"},
        {"task": "task_00", "state": "state_10", "mechanism": "known_negative"},
        {"task": "task_03", "state": "state_35", "mechanism": "known_negative"},
        {"task": "task_04", "state": "state_35", "mechanism": "safe_release"},
        {"task": "task_01", "state": "state_35", "mechanism": "short_horizon_negative"},
        {"task": "task_02", "state": "state_35", "mechanism": "instability_contact_loss"},
    ],
}

manifest = {
    "manifest": "N5_CANARY_32_MANIFEST_V1",
    "frozen_at": None,
    "n_episodes": 32,
    "selection_principle": "8 per suite, covering: stable-grasp+, transport+, known-neg, safe-release, instability, unsupported-articulated-unknown, short-horizon-neg, critical&&cc=false",
    "episodes": [],
}

total = 0
for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
    for ep in CANARY_SELECTION[suite]:
        sidecar = os.path.join(CS200_ROOT, suite, ep["task"], ep["state"], "privileged_teacher_sidecar.jsonl")
        summary = os.path.join(CS200_ROOT, suite, ep["task"], ep["state"], "episode_summary.json")
        if not os.path.isfile(sidecar):
            print(f"WARNING: missing {suite}/{ep['task']}/{ep['state']}")
            continue
        ep["suite"] = suite
        manifest["episodes"].append(ep)
        total += 1

import time
manifest["frozen_at"] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
manifest["n_episodes"] = total

with open(OUT_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)

# Verify diversity
mechanisms = {}
for ep in manifest["episodes"]:
    m = ep["mechanism"]
    mechanisms[m] = mechanisms.get(m, 0) + 1

print(f"Canary manifest: {total} episodes at {OUT_PATH}")
print("Mechanism distribution:")
for m, n in sorted(mechanisms.items()):
    print(f"  {m}: {n}")
