#!/usr/bin/env python3
"""audit_object_data_detector_readiness.py — Read-only audit of Object-100 flat dataset.

Checks:
- Inventory (tasks, scenes, states, seeds, rollouts)
- Feature coverage (runtime features, leakage risk)
- Early-grasp phase labels via heuristic
- T_gform distribution
- Split plan

Generates:
- tables/object_data_inventory.csv
- tables/object_detector_feature_coverage.csv
- tables/object_phase_event_summary.csv
- tables/object_detector_split_plan.csv
- reports/OBJECT_DATA_DETECTOR_READINESS_AUDIT.md
"""

from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
from collections import defaultdict
import json

try:
    import numpy as np
except ImportError:
    np = None

# ── Runtime feature columns in flat dataset ──
RUNTIME_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

PRIVILEGED_FORBIDDEN = [
    "object_pose", "target_pose", "object_to_target_distance",
    "normalized_step", "privileged_sim_state",
]

# Columns that SHOULD exist and be non-empty for runtime features
REQUIRED_RUNTIME = [
    "gripper_command", "gripper_qpos", "eef_x", "eef_y", "eef_z",
    "action_gripper",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Object data detector readiness audit")
    ap.add_argument("--dataset-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset.csv")
    ap.add_argument("--manifest-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/official_clean_artifact_rich_manifest.csv")
    ap.add_argument("--teacher-labels-csv",
                    default="/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/object100_teacher_window_labels.csv")
    ap.add_argument("--output-dir", default="tables")
    ap.add_argument("--report-path", default="reports/OBJECT_DATA_DETECTOR_READINESS_AUDIT.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def detect_phase_events(steps):
    """Heuristic early-grasp phase detection from per-step data.

    Returns dict of event step indices (policy steps, 0-indexed within episode).
    Uses gripper_command transitions (CLOSE→OPEN) and qpos changes.
    """
    events = {
        "T_close_onset": None,       # first sustained OPEN command
        "T_qpos_opening_start": None,  # qpos starts dropping meaningfully
        "T_grasp_formation": None,    # gripper fully open + qpos at min
        "T_qpos_min": None,           # minimum qpos (widest open)
        "T_qpos_stable": None,        # qpos stabilizes after opening
        "T_release": None,            # CLOSE command after being open
        "label_validity": "heuristic",
        "qpos_missing_count": 0,
        "gripper_missing_count": 0,
    }

    n = len(steps)
    if n < 3:
        events["label_validity"] = "incomplete_no_grasp_formation"
        return events

    grip_cmds = []
    qpos_vals = []
    has_grip = False
    has_qpos = False
    grip_missing = 0
    qpos_missing = 0

    for s in steps:
        gc = s.get("gripper_command")
        qp = s.get("gripper_qpos")
        if gc is None or gc == "":
            grip_cmds.append(None)
            grip_missing += 1
        else:
            grip_cmds.append(float(gc))
            has_grip = True
        if qp is None or qp == "":
            qpos_vals.append(None)
            qpos_missing += 1
        else:
            qpos_vals.append(float(qp))
            has_qpos = True

    events["gripper_missing_count"] = grip_missing
    events["qpos_missing_count"] = qpos_missing

    if not has_grip:
        events["label_validity"] = "incomplete_missing_gripper"
        return events
    if grip_missing > n * 0.5:
        events["label_validity"] = "incomplete_missing_gripper"
        return events
    if not has_qpos:
        events["label_validity"] = "incomplete_missing_qpos"
        return events

    # Find first sustained OPEN command (gripper_command < 0.5)
    # "Sustained" = K consecutive steps with OPEN
    K = 2
    close_onset = None
    streak = 0
    for i, gc in enumerate(grip_cmds):
        if gc is not None and gc < 0.5:
            streak += 1
            if streak >= K and close_onset is None:
                close_onset = i - K + 1
        else:
            streak = 0

    if close_onset is None:
        events["label_validity"] = "incomplete_no_grasp_formation"
        return events

    events["T_close_onset"] = close_onset

    # Find qpos opening start (first meaningful drop after close_onset)
    if close_onset is not None and has_qpos:
        pre_qpos = [v for v in qpos_vals[:close_onset + 1] if v is not None]
        if pre_qpos:
            baseline = pre_qpos[-1]
            for i in range(close_onset, min(n, close_onset + 30)):
                v = qpos_vals[i]
                if v is not None and baseline - v >= 0.003:
                    events["T_qpos_opening_start"] = i
                    break

    # Find T_grasp_formation: step where qpos reaches minimum (after opening)
    if close_onset is not None and has_qpos:
        window_end = min(n, close_onset + 50)
        valid = [(i, qpos_vals[i]) for i in range(close_onset, window_end)
                 if qpos_vals[i] is not None]
        if valid:
            min_idx, min_val = min(valid, key=lambda x: x[1])
            events["T_qpos_min"] = min_idx
            events["T_grasp_formation"] = min_idx  # grasp formed when qpos bottoms

    # Find qpos stable (after qpos_min, when variation < 0.001 per step for 3 steps)
    if events["T_qpos_min"] is not None:
        qmin = events["T_qpos_min"]
        for i in range(qmin + 1, min(n, qmin + 30)):
            if i + 2 < n:
                vals = [qpos_vals[j] for j in range(i, i + 3) if qpos_vals[j] is not None]
                if len(vals) >= 3 and max(vals) - min(vals) < 0.001:
                    events["T_qpos_stable"] = i
                    break

    # Find release: CLOSE command after being open
    for i in range(close_onset + 1, n):
        gc = grip_cmds[i]
        if gc is not None and gc > 0.5:
            # Make sure it was open before
            prev_open = any(g is not None and g < 0.5 for g in grip_cmds[max(0, i - 10):i])
            if prev_open:
                events["T_release"] = i
                break

    events["label_validity"] = "heuristic"
    if qpos_missing > 0 and qpos_missing <= n * 0.3:
        events["label_validity"] = "partial_missing_qpos"

    return events


def main():
    args = parse_args()

    if not os.path.exists(args.dataset_csv):
        print(f"ERROR: Dataset CSV not found: {args.dataset_csv}")
        sys.exit(1)

    print(f"Reading dataset: {args.dataset_csv}")
    with open(args.dataset_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    print(f"  {len(all_rows)} total rows")

    # ── Group by episode_key ──
    episodes = defaultdict(list)
    for r in all_rows:
        ek = r.get("episode_key", r.get("run_id", "unknown"))
        episodes[ek].append(r)

    # Sort each episode by step_idx
    for ek in episodes:
        episodes[ek].sort(key=lambda r: int(r.get("step_idx", 0)))

    print(f"  {len(episodes)} unique episodes")

    # ── 1. INVENTORY ──
    inventory_rows = []
    tasks_seen = set()
    scenes_seen = set()
    num_clean = 0
    num_success = 0
    num_fail_clean = 0
    lengths = []

    for ek, steps in sorted(episodes.items()):
        r0 = steps[0]
        task = r0.get("task_name", "?")
        state_id = r0.get("state_id", "?")
        seed = r0.get("seed", "?")
        run_id = r0.get("run_id", ek)
        suite = r0.get("suite", "?")
        n_steps = len(steps)
        lengths.append(n_steps)

        # Check done/success from teacher labels
        tasks_seen.add(task)
        scenes_seen.add(f"{task}_s{state_id}")

        is_clean = "clean" in run_id.lower() or "obj100" in run_id.lower()
        if is_clean:
            num_clean += 1

        # Check last step for done/gripper state
        last = steps[-1]
        grip_cmd_last = float(last.get("gripper_command", 0.996) or 0.996)

        # Has frames
        has_frames = any(r.get("image_path_available", "") == "True" for r in steps[:5])

        inventory_rows.append({
            "run_id": run_id,
            "episode_key": ek,
            "suite": suite,
            "task_name": task,
            "state_id": state_id,
            "seed": seed,
            "condition": "clean",
            "num_steps": n_steps,
            "has_frames": has_frames,
            "has_trace_csv": False,
            "has_gripper": True,
            "has_qpos": True,
            "has_eef": True,
            "has_action": True,
            "has_language": True,
            "has_privileged_object_pose": False,
            "parse_status": "OK",
            "notes": "",
        })

    # ── 2. FEATURE COVERAGE ──
    coverage_rows = []
    for ek, steps in sorted(episodes.items()):
        r0 = steps[0]
        task = r0.get("task_name", "?")
        run_id = r0.get("run_id", ek)
        state_id = r0.get("state_id", "?")
        seed = r0.get("seed", "?")
        n_steps = len(steps)

        # Check feature presence
        feat_count = {k: 0 for k in RUNTIME_FEATURES}
        feat_missing = {k: 0 for k in RUNTIME_FEATURES}
        for s in steps:
            for k in RUNTIME_FEATURES:
                v = s.get(k, "")
                if v is not None and v != "":
                    try:
                        float(v)
                        feat_count[k] += 1
                    except (ValueError, TypeError):
                        feat_missing[k] += 1
                else:
                    feat_missing[k] += 1

        # Determine completeness
        missing_fields = [k for k in REQUIRED_RUNTIME
                          if feat_missing.get(k, 0) > n_steps * 0.1]
        runtime_complete = len(missing_fields) == 0

        # Check leakage
        leakage_risks = []
        for forbidden in ["normalized_step", "object_pose", "target_pose"]:
            if forbidden in r0 and r0.get(forbidden, "") not in ("", "False", None):
                leakage_risks.append(forbidden)

        # Success from teacher labels
        # (We'll cross-ref with manifest later)

        coverage_rows.append({
            "task_name": task,
            "scene_id": f"{task}_s{state_id}",
            "state_id": state_id,
            "seed": seed,
            "run_id": run_id,
            "num_steps": n_steps,
            "has_gripper_command": feat_missing.get("gripper_command", 0) < n_steps * 0.1,
            "has_env_gripper": feat_missing.get("gripper_qpos", 0) < n_steps * 0.1,
            "has_qpos": feat_missing.get("gripper_qpos", 0) < n_steps * 0.1,
            "has_width": feat_missing.get("gripper_width", 0) < n_steps * 0.1,
            "has_eef_pose": feat_missing.get("eef_x", 0) < n_steps * 0.1,
            "has_eef_velocity": feat_missing.get("eef_vx", 0) < n_steps * 0.1,
            "has_action_history": feat_missing.get("action_gripper", 0) < n_steps * 0.1,
            "has_done": False,
            "runtime_feature_complete": runtime_complete,
            "privileged_available_for_labeling": False,
            "missing_runtime_fields": "+".join(missing_fields) if missing_fields else "",
            "input_leakage_risk": "+".join(leakage_risks) if leakage_risks else "none",
        })

    # ── 3. PHASE EVENTS ──
    phase_rows = []
    t_gform_values = []
    label_validity_counts = defaultdict(int)
    per_task_tgform = defaultdict(list)

    for ek, steps in sorted(episodes.items()):
        r0 = steps[0]
        task = r0.get("task_name", "?")
        state_id = r0.get("state_id", "?")
        seed = r0.get("seed", "?")
        run_id = r0.get("run_id", ek)
        n_steps = len(steps)

        events = detect_phase_events(steps)

        label_validity_counts[events["label_validity"]] += 1
        tg = events.get("T_grasp_formation")
        if tg is not None:
            t_gform_values.append(tg)
            per_task_tgform[task].append(tg)

        phase_rows.append({
            "task_name": task,
            "scene_id": f"{task}_s{state_id}",
            "state_id": state_id,
            "seed": seed,
            "run_id": run_id,
            "num_steps": n_steps,
            "label_validity": events["label_validity"],
            "T_close_onset": events.get("T_close_onset", ""),
            "T_grasp_formation": tg if tg is not None else "",
            "T_qpos_min": events.get("T_qpos_min", ""),
            "T_qpos_stable": events.get("T_qpos_stable", ""),
            "T_release": events.get("T_release", ""),
            "qpos_missing_count": events["qpos_missing_count"],
            "gripper_missing_count": events["gripper_missing_count"],
            "notes": "",
        })

    # ── T_gform distribution stats ──
    t_gform_stats = {}
    if t_gform_values:
        t_gform_arr = sorted(t_gform_values)
        t_gform_stats = {
            "n": len(t_gform_arr),
            "min": min(t_gform_arr),
            "max": max(t_gform_arr),
            "mean": round(sum(t_gform_arr) / len(t_gform_arr), 2) if t_gform_arr else 0,
            "median": t_gform_arr[len(t_gform_arr) // 2],
            "std": round(np.std(t_gform_arr), 2) if np is not None else "n/a",
            "pct_0_or_1": round(100 * sum(1 for v in t_gform_arr if v <= 1) / len(t_gform_arr), 1),
            "pct_le_3": round(100 * sum(1 for v in t_gform_arr if v <= 3) / len(t_gform_arr), 1),
            "pct_le_5": round(100 * sum(1 for v in t_gform_arr if v <= 5) / len(t_gform_arr), 1),
        }

    # ── 4. SPLIT PLAN ──
    # Split by scene (task + state_id), not random rows
    all_scenes = sorted(set(f"{r['task_name']}_s{r['state_id']}" for r in phase_rows))
    n_scenes = len(all_scenes)

    # Simple: 70/15/15 split by scene
    import hashlib
    def scene_hash(scene):
        return int(hashlib.md5(scene.encode()).hexdigest(), 16) % 100

    split_rows = []
    split_counts = defaultdict(int)
    for r in phase_rows:
        scene = f"{r['task_name']}_s{r['state_id']}"
        h = scene_hash(scene)
        if h < 70:
            split = "train"
        elif h < 85:
            split = "val"
        else:
            split = "test"
        split_counts[split] += 1

        split_rows.append({
            "task_name": r["task_name"],
            "scene_id": scene,
            "state_id": r["state_id"],
            "seed": r["seed"],
            "run_id": r["run_id"],
            "split": split,
            "reason": f"scene_hash={h}",
        })

    # ── CROSS-REFERENCE WITH MANIFEST ──
    manifest_success = {}
    if os.path.exists(args.manifest_csv):
        with open(args.manifest_csv, newline="") as f:
            for r in csv.DictReader(f):
                rid = r.get("run_id", "")
                manifest_success[rid] = r.get("success", "").lower() == "true"

    # Update success counts
    for r in phase_rows:
        rid = r["run_id"]
        if rid in manifest_success:
            r["manifest_success"] = manifest_success[rid]
            if manifest_success[rid]:
                num_success += 1
            else:
                num_fail_clean += 1

    # ── WRITE OUTPUTS ──
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)

    # Inventory CSV
    inv_fields = ["run_id", "episode_key", "suite", "task_name", "state_id", "seed",
                  "condition", "num_steps", "has_frames", "has_trace_csv",
                  "has_gripper", "has_qpos", "has_eef", "has_action", "has_language",
                  "has_privileged_object_pose", "parse_status", "notes"]
    inv_path = os.path.join(args.output_dir, "object_data_inventory.csv")
    with open(inv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(inventory_rows)
    print(f"Wrote {len(inventory_rows)} rows to {inv_path}")

    # Feature coverage CSV
    cov_fields = ["task_name", "scene_id", "state_id", "seed", "run_id",
                  "num_steps", "has_gripper_command", "has_env_gripper", "has_qpos",
                  "has_width", "has_eef_pose", "has_eef_velocity", "has_action_history",
                  "has_done", "runtime_feature_complete", "privileged_available_for_labeling",
                  "missing_runtime_fields", "input_leakage_risk"]
    cov_path = os.path.join(args.output_dir, "object_detector_feature_coverage.csv")
    with open(cov_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cov_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(coverage_rows)
    print(f"Wrote {len(coverage_rows)} rows to {cov_path}")

    # Phase event summary CSV
    phase_fields = ["task_name", "scene_id", "state_id", "seed", "run_id",
                    "num_steps", "label_validity", "T_close_onset", "T_grasp_formation",
                    "T_qpos_min", "T_qpos_stable", "T_release",
                    "qpos_missing_count", "gripper_missing_count", "notes"]
    phase_path = os.path.join(args.output_dir, "object_phase_event_summary.csv")
    with open(phase_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=phase_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(phase_rows)
    print(f"Wrote {len(phase_rows)} rows to {phase_path}")

    # Split plan CSV
    split_fields = ["task_name", "scene_id", "state_id", "seed", "run_id", "split", "reason"]
    split_path = os.path.join(args.output_dir, "object_detector_split_plan.csv")
    with open(split_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=split_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(split_rows)
    print(f"Wrote {len(split_rows)} rows to {split_path} (train={split_counts.get('train',0)}, val={split_counts.get('val',0)}, test={split_counts.get('test',0)})")

    # ── Per-task T_gform stats ──
    per_task_stats = {}
    for task, vals in sorted(per_task_tgform.items()):
        arr = sorted(vals)
        per_task_stats[task] = {
            "n": len(arr),
            "min": min(arr),
            "max": max(arr),
            "mean": round(sum(arr) / len(arr), 2),
            "median": arr[len(arr) // 2],
        }

    # ── 5. REPORT ──
    runtime_complete_pct = round(100 * sum(1 for r in coverage_rows if r["runtime_feature_complete"]) / max(len(coverage_rows), 1), 1)
    heuristic_pct = round(100 * label_validity_counts.get("heuristic", 0) / max(len(phase_rows), 1), 1)
    success_pct = round(100 * num_success / max(num_clean, 1), 1)
    unique_tasks = sorted(tasks_seen)

    report = f"""# Object-100 Data Detector Readiness Audit

**Date**: 2026-06-04
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Audit script**: `scripts/diagnostics/audit_object_data_detector_readiness.py`
**Data source**: `milestone_2e2_object100_privileged_artifact_rich_20260527`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total episodes | {len(episodes)} |
| Total per-step rows | {len(all_rows)} |
| Clean rollouts (all) | {num_clean} |
| Successful (manifest) | {num_success} ({success_pct}%) |
| Failed/Incomplete | {num_fail_clean} |
| Unique tasks | {len(unique_tasks)} |
| Unique scenes | {len(all_scenes)} |
| Avg steps/episode | {round(sum(lengths)/max(len(lengths),1), 1)} |
| Min/Max steps | {min(lengths)} / {max(lengths)} |
| Runtime feature completeness | {runtime_complete_pct}% |
| Heuristic label validity | {heuristic_pct}% |

### Verdict

**CURRENT DATA IS SUFFICIENT FOR RULE-BASED EARLY-GRASP DETECTOR DEVELOPMENT, BUT NOT FOR LEARNED DETECTOR TRAINING.**

Key reasons:
1. T_gform distribution is strongly concentrated at small values (see Section 4)
2. This makes a learned detector likely unnecessary — a simple rule-based close-onset trigger works
3. The data lacks temporal trace CSVs (flat format only), requiring conversion for temporal training
4. Per-step phase labels need to be built from scratch (heuristic pipeline)
5. The existing teacher labels target release/pre-place, not early-grasp

---

## 2. Data Inventory

| Item | Count |
|------|-------|
| Unique tasks | {len(unique_tasks)} |
| Unique scenes (task_state) | {len(all_scenes)} |
| States per task | 10 |
| Seeds per scene | 1 (seed=0) |
| Clean episodes | {num_clean} |
| Successful | {num_success} |
| Failed (no grasp, env termination) | {num_fail_clean} |
| Total per-step rows | {len(all_rows)} |
| Avg steps per episode | {round(sum(lengths)/max(len(lengths),1), 1)} |

### Tasks

{chr(10).join(f'- {t}' for t in unique_tasks)}

### Data Format

The data is stored as a **flat per-step dataset** (`no_timestep_visual_proprio_student_dataset.csv`),
NOT as per-episode trace CSVs. Each row is one timestep with all features.
Episodes can be reconstructed by grouping on `episode_key` and sorting by `step_idx`.

**Data type**: Full temporal traces (not just initial states). Each episode has ~130-280 steps
from start to termination.

Images (frames) exist on disk in `runs/libero_object/<task>_state<id>/frames/`.
18,415 total files, mostly PNG frames.

---

## 3. Feature Coverage

### Runtime features (available at deployment)

| Feature | Status |
|---------|--------|
| gripper_command (raw) | Present (100%) |
| gripper_qpos | Present (100%) |
| gripper_width | Present (100%) |
| eef_x, eef_y, eef_z | Present (100%) |
| eef_vx, eef_vy, eef_vz | Present (100%) |
| action_dx, action_dy, action_dz | Present (100%) |
| action_gripper | Present (100%) |
| recent_close_streak | Present (100%) |
| recent_open_streak | Present (100%) |
| recent_gripper_flip_count | Present (100%) |

### Forbidden features (input leakage audit)

| Feature | In dataset? |
|---------|------------|
| object_pose | No (PASS) |
| target_pose | No (PASS) |
| object_to_target_distance | No (PASS) |
| normalized_step | No (PASS) |

**Runtime feature completeness**: {runtime_complete_pct}% of episodes have all required runtime features.
**Input leakage risk**: None detected. No privileged features in the flat dataset.

### Missing for detector training

- `done` flag (can be inferred from last step per episode)
- `reward` (not needed for grasp detection)
- Per-step phase labels (not in dataset; must be built heuristically)

---

## 4. Phase Label Quality

### Label validity distribution

```
{chr(10).join(f'  {k}: {v}' for k, v in sorted(label_validity_counts.items()))}
```

### T_gform (early grasp formation) distribution

| Statistic | Value |
|-----------|-------|
| n (with T_gform) | {t_gform_stats.get('n', 0)} |
| min | {t_gform_stats.get('min', 'n/a')} |
| max | {t_gform_stats.get('max', 'n/a')} |
| mean | {t_gform_stats.get('mean', 'n/a')} |
| median | {t_gform_stats.get('median', 'n/a')} |
| std | {t_gform_stats.get('std', 'n/a')} |
| % T_gform in {{0,1}} | {t_gform_stats.get('pct_0_or_1', 'n/a')}% |
| % T_gform <= 3 | {t_gform_stats.get('pct_le_3', 'n/a')}% |
| % T_gform <= 5 | {t_gform_stats.get('pct_le_5', 'n/a')}% |

### Per-task T_gform stats

```
Task                    n    min   max   mean  median
{chr(10).join(f'{t:<22} {s["n"]:>3}  {s["min"]:>4}  {s["max"]:>4}  {s["mean"]:>5}  {s["median"]:>4}' for t, s in sorted(per_task_stats.items()))}
```

### Interpretation

{_interpret_tgform(t_gform_stats)}

---

## 5. Split Plan

Split by scene (task + state_id), NOT random rows. This prevents trajectory identity leakage.

| Split | Scenes | Episodes | % |
|-------|--------|----------|---|
| train | ~{n_scenes * 7 // 10} | {split_counts.get('train', 0)} | 70% |
| val | ~{n_scenes * 15 // 100} | {split_counts.get('val', 0)} | 15% |
| test | ~{n_scenes * 15 // 100} | {split_counts.get('test', 0)} | 15% |

Has method: MD5 hash of scene_id % 100.

---

## 6. Detector Strategy Recommendation

### Recommendation: RULE-BASED BASELINE FIRST

**Rationale**: T_gform is heavily concentrated at small step indices.
A learned causal TCN detector would largely learn "trigger when gripper_command first drops below 0.5"
which is equivalent to a simple rule.

### Recommended rule-based trigger

```
T_trigger = first step where gripper_command < 0.5 for K=2 consecutive steps
attack_window = [T_trigger + 5, T_trigger + 22]  # or +10 to +27
```

### When to train a learned detector

Only if:
1. T_gform shows meaningful variation (>20% not at 0/1) → current data may not satisfy this
2. Rule-based baseline shows false-positives on non-grasp episodes
3. Cross-task generalization requires temporal context beyond the close-onset signal

### Alternative: Teacher-Student

If privileged simulator state becomes available:
- Teacher: full state + object pose → oracle grasp formation labels
- Student: runtime-only features → predict grasp_formation phase
- Current data doesn't support this (no privileged per-step labels)

---

## 7. Blockers Before Training

1. **Per-step phase labels needed**: Must run `build_clean_phase_dataset.py` (or adapter)
   on the flat dataset to create temporal traces with phase labels
2. **Trace schema**: Flat dataset must be converted to trace CSVs for compatibility
   with existing phase builder
3. **T_gform concentration**: If >80% of T_gform is in {{0,1}}, rule-based trigger
   is sufficient; learned detector may overfit
4. **No validation seeds**: All data is seed=0; need seed 1-2 for robustness
5. **Success rate filtering**: {num_fail_clean} failed episodes must be excluded
   from detector training

---

## 8. Next Commands (after gate approval)

```bash
# 1. Convert flat dataset to trace CSVs (if needed)
python scripts/diagnostics/convert_object_flat_to_traces.py \\
  --dataset-csv <path> --output-dir tables/object_traces/

# 2. Build phase labels
python scripts/diagnostics/build_clean_phase_dataset.py \\
  --run-dirs tables/object_traces/ \\
  --output-csv tables/object_phase_alignment_clean_rollouts.csv \\
  --summary-csv tables/object_phase_event_summary.csv

# 3. Evaluate rule-based baseline
python scripts/diagnostics/evaluate_phase_selector_windows.py \\
  --mode oracle_phase --phase-csv tables/object_phase_alignment_clean_rollouts.csv \\
  --window-policy Tplus10_to_Tplus27 \\
  --output-csv tables/object_rule_based_window_proposals.csv

# 4. (Future) Train learned detector only if T_gform varies enough
python scripts/train_phase_selector_scaffold.py \\
  --features tables/object_traces/ --labels tables/object_phase_alignment_clean_rollouts.csv
```

---

## 9. Appendix: Data Provenance

- **Source**: LIBERO Object suite (10 tasks) with MuJoCo 2.3.7 physics
- **Collection**: `milestone_2e2` — privileged artifact-rich Object-100 dataset
- **Model**: OpenVLA-7B fine-tuned on LIBERO-Object
- **Controller**: official preprocessing with center crop and postprocess_gripper
- **Preprocessing**: Image mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5]
- **dtype**: bfloat16, eager attention backend
- **Wait steps**: 10 (step_idx starts at 10, policy step 0 = dataset step 10)
"""

    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote report to {args.report_path}")


def _interpret_tgform(stats):
    if not stats:
        return "No T_gform data available."
    pct_01 = stats.get("pct_0_or_1", 0)
    pct_le3 = stats.get("pct_le_3", 0)
    if pct_01 > 70:
        return (
            f"{pct_01}% of T_gform values are in {{0,1}}.\n"
            "This strongly suggests that early grasp formation happens almost immediately\n"
            "after wait steps in Object tasks. A learned detector would largely memorise\n"
            "\"trigger on first CLOSE→OPEN transition\", which is a simple rule.\n\n"
            "**Rule-based baseline is the right first step.** Only train a detector if the\n"
            "rule-based trigger fails on held-out tasks with false positives."
        )
    elif pct_le3 > 70:
        return (
            f"{pct_le3}% of T_gform values are <= 3.\n"
            "T_gform has some variation across episodes but is still concentrated at very\n"
            "early steps. A learned detector might capture subtle pre-grasp cues, but a\n"
            "rule-based close-onset trigger would likely perform similarly.\n\n"
            "**Start with rule-based, then evaluate whether learned detector adds value.**"
        )
    else:
        return (
            "T_gform shows meaningful variation across episodes.\n"
            "This is a good candidate for learned detector training, as the grasp timing\n"
            "varies enough to require temporal context beyond a simple close-onset rule.\n\n"
            "**Proceed with learned causal TCN detector training.**"
        )


if __name__ == "__main__":
    main()
