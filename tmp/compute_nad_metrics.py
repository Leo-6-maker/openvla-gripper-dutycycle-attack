#!/usr/bin/env python3
"""Offline computation of NAD, qpos deltas, CQ proxies, latency from existing telemetry.
Does NOT require GPU or model loading — pure CSV/JSON analysis of all 213 runs.

Output:
  evidence/phase7_table1/OFFLINE_METRICS_SUMMARY.json
  evidence/phase7_table1/OFFLINE_METRICS_PER_RUN.csv
  evidence/phase7_table1/OFFLINE_METRICS_PER_CELL.csv
"""
import csv, json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ──
BASE_VIS = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/attack_benchmark")
BASE_SHUFFLED = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/shuffled")
BASE_UNTARGETED = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/supplement_7h/untargeted")
BASE_RANDOM = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/supplement_7h/random_time")
BASE_EARLY = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/supplement_7h/early_shift")
BASE_UNTARGETED_CANARY = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/untargeted_canary")

OUT_DIR = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LIBERO action space approximate bounds (empirical from model action_stats)
# q01 and q99 for each of 7 DoF, used for NAD normalization
ACTION_Q01 = np.array([-0.18, -0.20, -0.15, -0.35, -0.30, -0.25, -0.99], dtype=np.float32)
ACTION_Q99 = np.array([ 0.18,  0.20,  0.15,  0.35,  0.30,  0.25,  0.99], dtype=np.float32)
ACTION_RANGE = ACTION_Q99 - ACTION_Q01
# Clip to avoid division by zero
ACTION_RANGE = np.where(ACTION_RANGE < 0.01, 0.01, ACTION_RANGE)

TARGET_TOKEN = 31744

PARENTS = [
    ("alphabet_soup_s0", 0, 0, "supplementary"),
    ("cream_cheese_s0", 1, 0, "primary"),
    ("salad_dressing_s0", 2, 0, "primary"),
    ("bbq_sauce_s0", 3, 0, "primary"),
    ("ketchup_s0", 4, 0, "primary"),
    ("tomato_sauce_s0", 5, 0, "primary"),
    ("butter_s0", 6, 0, "primary"),
    ("butter_s2", 6, 2, "primary"),
    ("milk_s4", 7, 4, "primary"),
    ("chocolate_pudding_s2", 8, 2, "primary"),
    ("orange_juice_s0", 9, 0, "primary"),
]

SEEDS = [42, 123, 456]


def load_telemetry(path):
    tel = path / "step_telemetry.csv"
    if not tel.exists():
        return None
    rows = list(csv.DictReader(open(tel)))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def load_summary(path):
    s = path / "episode_summary.json"
    if not s.exists():
        return None
    return json.load(open(s))


def parse_action(raw_str):
    """Parse raw_action_7d or env_action_7d from JSON string."""
    try:
        arr = json.loads(raw_str)
        return np.array(arr, dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def compute_nad(clean_action, adv_action):
    """Normalized Action Discrepancy per DoF, then aggregated.
    NAD_i = |adv_i - clean_i| / range_i
    NAD_all = mean over all 7 DoF
    NAD_arm = mean over DoF 0-5
    NAD_gripper = NAD_6 (DoF index 6)
    """
    if clean_action is None or adv_action is None:
        return None, None, None
    diff = np.abs(adv_action - clean_action)
    nad_per_dof = diff / ACTION_RANGE
    nad_all = float(np.mean(nad_per_dof))
    nad_arm = float(np.mean(nad_per_dof[:6]))
    nad_gripper = float(nad_per_dof[6])
    return nad_all, nad_arm, nad_gripper


def analyze_run(run_dir):
    """Extract all offline metrics from one run."""
    tel = load_telemetry(run_dir)
    summary = load_summary(run_dir)
    if tel is None or summary is None:
        return None

    n_steps = len(tel)
    attack_frames = summary.get("attack_frames", 0)
    task_success = summary.get("task_success", False)
    emit_step = summary.get("mlp_emit_step", -1)
    if emit_step is None:
        emit_step = -1

    # Identify attack window
    atk_rows = [r for r in tel if r.get("attack_this") == "True"]
    n_atk = len(atk_rows)

    # ── NAD computation ──
    nad_all_vals = []
    nad_arm_vals = []
    nad_gripper_vals = []
    tok_open_count = 0
    env_open_count = 0
    model_ms_vals = []

    for r in atk_rows:
        # Parse clean action (raw_action_7d) and executed action (env_action_7d)
        clean_act = parse_action(r.get("raw_action_7d", "[]"))
        env_act = parse_action(r.get("env_action_7d", "[]"))

        if clean_act is not None and env_act is not None:
            na, nar, ng = compute_nad(clean_act, env_act)
            if na is not None:
                nad_all_vals.append(na)
                nad_arm_vals.append(nar)
                nad_gripper_vals.append(ng)

        # Token OPEN check
        adv_token = r.get("adv_token", "")
        if adv_token and str(adv_token).strip():
            try:
                if int(adv_token) == TARGET_TOKEN:
                    tok_open_count += 1
            except ValueError:
                pass

        # Env OPEN check
        try:
            if float(r.get("env_gripper", 1.0)) < 0:
                env_open_count += 1
        except (ValueError, TypeError):
            pass

        # Latency
        try:
            model_ms_vals.append(float(r.get("model_ms", 0)))
        except (ValueError, TypeError):
            pass

    # ── Aggregated NAD ──
    mean_nad_all = float(np.mean(nad_all_vals)) if nad_all_vals else None
    mean_nad_arm = float(np.mean(nad_arm_vals)) if nad_arm_vals else None
    mean_nad_gripper = float(np.mean(nad_gripper_vals)) if nad_gripper_vals else None

    # ── Token/Env OPEN duty ──
    token_open_duty = tok_open_count / n_atk if n_atk > 0 else 0.0
    env_open_duty = env_open_count / n_atk if n_atk > 0 else 0.0

    # ── qpos/width/object deltas ──
    # Compare attack window vs pre-attack baseline
    pre_atk_rows = [r for r in tel if r.get("attack_this") != "True"]
    atk_qpos = []
    atk_width = []
    atk_obj_z = []
    atk_eef_dist = []
    pre_qpos = []
    pre_width = []
    pre_obj_z = []

    for r in pre_atk_rows:
        try:
            pre_qpos.append(float(r.get("qpos_sum", "nan")))
            pre_width.append(float(r.get("f_gripper_opening_proxy", r.get("gripper_opening_proxy", "nan"))))
            pre_obj_z.append(float(r.get("obj_z", "nan")))
        except (ValueError, TypeError):
            pass

    for r in atk_rows:
        try:
            atk_qpos.append(float(r.get("qpos_sum", "nan")))
            atk_width.append(float(r.get("f_gripper_opening_proxy", r.get("gripper_opening_proxy", "nan"))))
            atk_obj_z.append(float(r.get("obj_z", "nan")))
            atk_eef_dist.append(float(r.get("eef_obj_dist", "nan")))
        except (ValueError, TypeError):
            pass

    pre_qpos = [x for x in pre_qpos if not np.isnan(x)]
    atk_qpos = [x for x in atk_qpos if not np.isnan(x)]
    pre_width = [x for x in pre_width if not np.isnan(x)]
    atk_width = [x for x in atk_width if not np.isnan(x)]
    pre_obj_z = [x for x in pre_obj_z if not np.isnan(x)]
    atk_obj_z = [x for x in atk_obj_z if not np.isnan(x)]
    atk_eef_dist = [x for x in atk_eef_dist if not np.isnan(x)]

    mean_pre_qpos = float(np.mean(pre_qpos)) if pre_qpos else None
    mean_atk_qpos = float(np.mean(atk_qpos)) if atk_qpos else None
    mean_pre_width = float(np.mean(pre_width)) if pre_width else None
    mean_atk_width = float(np.mean(atk_width)) if atk_width else None
    qpos_delta = (mean_atk_qpos - mean_pre_qpos) if (mean_pre_qpos is not None and mean_atk_qpos is not None) else None
    width_delta = (mean_atk_width - mean_pre_width) if (mean_pre_width is not None and mean_atk_width is not None) else None

    # Object z change (last atk obj_z vs last pre obj_z)
    obj_z_pre = pre_obj_z[-1] if pre_obj_z else None
    obj_z_atk_end = atk_obj_z[-1] if atk_obj_z else None
    obj_z_drop = (obj_z_atk_end - obj_z_pre) if (obj_z_pre is not None and obj_z_atk_end is not None) else None

    # EEF-object distance at end of attack
    eef_dist_atk = atk_eef_dist[-1] if atk_eef_dist else None

    # ── Latency ──
    mean_model_ms = float(np.mean(model_ms_vals)) if model_ms_vals else None
    total_model_ms = float(np.sum(model_ms_vals)) if model_ms_vals else None

    # ── CQ proxy: object dropped if obj_z decreases > 0.02 during attack ──
    obj_drop_proxy = (obj_z_drop is not None and obj_z_drop < -0.02)

    # ── Episode-level TASR ──
    episode_tasr = 1.0 if token_open_duty >= 0.8 else 0.0

    return {
        "n_steps": n_steps,
        "attack_frames": n_atk,
        "task_success": task_success,
        "emit_step": emit_step,
        "nad_all": mean_nad_all,
        "nad_arm": mean_nad_arm,
        "nad_gripper": mean_nad_gripper,
        "token_open_duty": token_open_duty,
        "env_open_duty": env_open_duty,
        "frame_tasr": token_open_duty,  # same as token_open_duty
        "episode_tasr": episode_tasr,
        "qpos_delta": qpos_delta,
        "width_delta": width_delta,
        "obj_z_drop": obj_z_drop,
        "eef_obj_dist_atk_end": eef_dist_atk,
        "obj_drop_proxy": obj_drop_proxy,
        "mean_model_ms": mean_model_ms,
        "total_model_ms": total_model_ms,
        "attack_duty_cycle": n_atk / n_steps if n_steps > 0 else 0.0,
        "pre_mean_qpos": mean_pre_qpos,
        "atk_mean_qpos": mean_atk_qpos,
        "pre_mean_width": mean_pre_width,
        "atk_mean_width": mean_atk_width,
    }


def get_run_dirs(base, cell, suffix, seeds):
    """Find all run directories for a given cell and condition."""
    dirs = {}
    for s in seeds:
        d = base / f"{cell}_{suffix}_s{s}"
        if d.exists() and (d / "episode_summary.json").exists():
            dirs[s] = d
    return dirs


def main():
    all_runs = []

    # ── Collect all runs ──
    conditions = [
        ("VIS", BASE_VIS, "vis"),
        ("RAND", BASE_VIS, "rand"),
        ("SHUFFLED", BASE_SHUFFLED, "shuffled"),
        ("UNTARGETED", BASE_UNTARGETED, "untargeted"),
        ("RANDOM_TIME", BASE_RANDOM, "random"),
        ("EARLY_SHIFT", BASE_EARLY, "early"),
    ]

    for cell, task, state, subset in PARENTS:
        for cond_name, base, suffix in conditions:
            dirs = get_run_dirs(base, cell, suffix, SEEDS)
            for seed, d in dirs.items():
                metrics = analyze_run(d)
                if metrics is None:
                    continue
                metrics["cell_id"] = cell
                metrics["task_idx"] = task
                metrics["state_id"] = state
                metrics["subset"] = subset
                metrics["condition"] = cond_name
                metrics["seed"] = seed
                all_runs.append(metrics)

        # Untargeted canaries (u1=butter, u2=salad, u3=tomato)
        canary_map = {
            "butter_s0": "u1",
            "salad_dressing_s0": "u2",
            "tomato_sauce_s0": "u3",
        }
        if cell in canary_map:
            d = BASE_UNTARGETED_CANARY / canary_map[cell]
            if d.exists():
                metrics = analyze_run(d)
                if metrics:
                    metrics["cell_id"] = cell
                    metrics["task_idx"] = task
                    metrics["state_id"] = state
                    metrics["subset"] = subset
                    metrics["condition"] = "UNTARGETED"
                    metrics["seed"] = 42
                    all_runs.append(metrics)

    # Early-shift & Random-time canaries
    for canary_dir, cell, cond_name in [
        (BASE_EARLY / "canary_e1_salad_s42", "salad_dressing_s0", "EARLY_SHIFT"),
        (BASE_EARLY / "canary_e2_tomato_s42", "tomato_sauce_s0", "EARLY_SHIFT"),
        (BASE_RANDOM / "canary_r1_salad_s42", "salad_dressing_s0", "RANDOM_TIME"),
        (BASE_RANDOM / "canary_r2_tomato_s42", "tomato_sauce_s0", "RANDOM_TIME"),
    ]:
        if canary_dir.exists():
            metrics = analyze_run(canary_dir)
            if metrics:
                for c, t, s, ss in PARENTS:
                    if c == cell:
                        metrics["cell_id"] = cell
                        metrics["task_idx"] = t
                        metrics["state_id"] = s
                        metrics["subset"] = ss
                        metrics["condition"] = cond_name
                        metrics["seed"] = 42
                        all_runs.append(metrics)
                        break

    print(f"Total runs analyzed: {len(all_runs)}")

    # ── Save per-run CSV ──
    run_fields = [
        "cell_id", "task_idx", "state_id", "subset", "condition", "seed",
        "n_steps", "attack_frames", "task_success", "emit_step",
        "nad_all", "nad_arm", "nad_gripper",
        "token_open_duty", "env_open_duty", "frame_tasr", "episode_tasr",
        "qpos_delta", "width_delta", "obj_z_drop", "eef_obj_dist_atk_end",
        "obj_drop_proxy", "mean_model_ms", "total_model_ms", "attack_duty_cycle",
    ]
    with open(OUT_DIR / "OFFLINE_METRICS_PER_RUN.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_runs)

    # ── Per-cell aggregation ──
    cell_agg = defaultdict(lambda: defaultdict(list))
    for r in all_runs:
        key = (r["cell_id"], r["condition"])
        for field in ["nad_all", "nad_arm", "nad_gripper", "token_open_duty",
                       "env_open_duty", "qpos_delta", "width_delta",
                       "obj_z_drop", "mean_model_ms", "attack_duty_cycle"]:
            val = r.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                cell_agg[key][field].append(val)
        cell_agg[key]["task_success"].append(1 if r["task_success"] else 0)
        cell_agg[key]["obj_drop_proxy"].append(1 if r["obj_drop_proxy"] else 0)
        cell_agg[key]["episode_tasr"].append(r["episode_tasr"])
        cell_agg[key]["n_runs"].append(1)

    cell_rows = []
    for (cell, cond), vals in sorted(cell_agg.items()):
        row = {"cell_id": cell, "condition": cond, "n_runs": len(vals["n_runs"])}
        for field in ["nad_all", "nad_arm", "nad_gripper", "token_open_duty",
                       "env_open_duty", "qpos_delta", "width_delta",
                       "obj_z_drop", "mean_model_ms", "attack_duty_cycle"]:
            if vals[field]:
                row[f"mean_{field}"] = round(float(np.mean(vals[field])), 4)
                row[f"std_{field}"] = round(float(np.std(vals[field])), 4)
            else:
                row[f"mean_{field}"] = None
                row[f"std_{field}"] = None
        row["fr"] = round(1.0 - float(np.mean(vals["task_success"])), 4)
        row["cqfr_proxy"] = round(float(np.mean(vals["obj_drop_proxy"])), 4)
        row["episode_tasr_mean"] = round(float(np.mean(vals["episode_tasr"])), 4)
        cell_rows.append(row)

    cell_fields = [
        "cell_id", "condition", "n_runs", "fr", "cqfr_proxy", "episode_tasr_mean",
        "mean_nad_all", "std_nad_all",
        "mean_nad_arm", "std_nad_arm",
        "mean_nad_gripper", "std_nad_gripper",
        "mean_token_open_duty", "mean_env_open_duty",
        "mean_qpos_delta", "mean_width_delta",
        "mean_obj_z_drop", "mean_model_ms", "mean_attack_duty_cycle",
    ]
    with open(OUT_DIR / "OFFLINE_METRICS_PER_CELL.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cell_rows)

    # ── Aggregate per condition (9-cell qualified) ──
    qualified_cells = {c for c, _, _, _ in PARENTS if c not in ("cream_cheese_s0", "chocolate_pudding_s2")}

    print("\n=== AGGREGATE METRICS (9-cell qualified) ===")
    print(f"{'Condition':<15} {'FR':>6} {'ΔFR':>6} {'NAD_all':>8} {'NAD_arm':>8} {'NAD_grip':>8} {'TASR_f':>8} {'TASR_e':>8} {'qposΔ':>8} {'widthΔ':>8} {'CQFR_p':>8} {'Lat(ms)':>8}")
    print("-" * 115)

    cond_order = ["VIS", "RAND", "SHUFFLED", "UNTARGETED", "RANDOM_TIME", "EARLY_SHIFT"]
    agg_summary = {}

    CLEAN_FR = 2.0 / 11.0  # 2 no-emit cells fail naturally out of 11

    for cond in cond_order:
        cond_runs = [r for r in all_runs if r["condition"] == cond and r["cell_id"] in qualified_cells]
        if not cond_runs:
            continue

        n = len(cond_runs)
        fr = sum(1 for r in cond_runs if not r["task_success"]) / n
        delta_fr = fr - CLEAN_FR

        nad_all = np.mean([r["nad_all"] for r in cond_runs if r.get("nad_all") is not None])
        nad_arm = np.mean([r["nad_arm"] for r in cond_runs if r.get("nad_arm") is not None])
        nad_grip = np.mean([r["nad_gripper"] for r in cond_runs if r.get("nad_gripper") is not None])
        tasr_f = np.mean([r["frame_tasr"] for r in cond_runs])
        tasr_e = np.mean([r["episode_tasr"] for r in cond_runs])

        qpos_deltas = [r["qpos_delta"] for r in cond_runs if r.get("qpos_delta") is not None]
        qpos_d = np.mean(qpos_deltas) if qpos_deltas else float("nan")

        width_deltas = [r["width_delta"] for r in cond_runs if r.get("width_delta") is not None]
        width_d = np.mean(width_deltas) if width_deltas else float("nan")

        cqfr_p = np.mean([r["obj_drop_proxy"] for r in cond_runs])

        lat_ms = np.mean([r["mean_model_ms"] for r in cond_runs if r.get("mean_model_ms") is not None])

        print(f"{cond:<15} {fr:>6.3f} {delta_fr:>6.3f} {nad_all:>8.4f} {nad_arm:>8.4f} {nad_grip:>8.4f} {tasr_f:>8.4f} {tasr_e:>8.3f} {qpos_d:>8.4f} {width_d:>8.4f} {cqfr_p:>8.4f} {lat_ms:>8.1f}")

        agg_summary[cond] = {
            "n_runs": n, "FR": round(fr, 4), "delta_FR": round(delta_fr, 4),
            "NAD_all": round(float(nad_all), 4), "NAD_arm": round(float(nad_arm), 4),
            "NAD_gripper": round(float(nad_grip), 4),
            "TASR_frame": round(float(tasr_f), 4), "TASR_episode": round(float(tasr_e), 4),
            "mean_qpos_delta": round(float(qpos_d), 4) if not np.isnan(qpos_d) else None,
            "mean_width_delta": round(float(width_d), 4) if not np.isnan(width_d) else None,
            "CQFR_proxy": round(float(cqfr_p), 4),
            "mean_latency_ms": round(float(lat_ms), 1),
        }

    # ── Save summary JSON ──
    summary = {
        "gate": "OFFLINE_METRICS_ANALYSIS",
        "total_runs_analyzed": len(all_runs),
        "clean_baseline_FR": CLEAN_FR,
        "action_range_q01": ACTION_Q01.tolist(),
        "action_range_q99": ACTION_Q99.tolist(),
        "conditions": agg_summary,
        "notes": {
            "NAD": "Normalized Action Discrepancy: |adv-clean|/action_range per DoF",
            "CQFR_proxy": "object z drop > 0.02 during attack window (proxy, needs video verification)",
            "TASR_frame": "fraction of attack frames with target OPEN token",
            "TASR_episode": "fraction of episodes with >=80% target OPEN token duty",
            "qpos_delta": "mean qpos during attack minus mean qpos pre-attack",
            "width_delta": "mean gripper width proxy during attack minus pre-attack",
        },
    }
    with open(OUT_DIR / "OFFLINE_METRICS_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to: {OUT_DIR}")
    print("  OFFLINE_METRICS_PER_RUN.csv")
    print("  OFFLINE_METRICS_PER_CELL.csv")
    print("  OFFLINE_METRICS_SUMMARY.json")


if __name__ == "__main__":
    main()
