# -*- coding: utf-8 -*-
"""Build failure-first VIS multi-phase result tables from rollout artifacts."""
import argparse
import csv
import json
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gripper_attack.gripper_semantics import raw_gripper_is_open


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def longest_open_streak(rows):
    best = 0
    cur = 0
    for row in rows:
        if raw_gripper_is_open(to_float(row.get("adv_grip"))):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def delta(rows, key):
    vals = [to_float(r.get(key)) for r in rows if r.get(key) not in (None, "")]
    return max(vals) - min(vals) if vals else 0.0


def load_summary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_summary_file"] = path.name
    return data


def trace_path_from_summary(summary, root):
    trace = Path(str(summary.get("trace_path", "")))
    return root / "runs" / trace.name


def phase_rows_from_trace(summary, root):
    trace_path = trace_path_from_summary(summary, root)
    if not trace_path.exists():
        return []
    rows = read_csv(trace_path)
    phases = sorted({r.get("phase_name", "") for r in rows if r.get("phase_name") and r.get("phase_name") != "none"})
    out = []
    for phase in phases:
        phase_rows = [r for r in rows if r.get("phase_name") == phase]
        attacked = [r for r in phase_rows if to_bool(r.get("pgd_applied")) or to_bool(r.get("random_applied"))]
        window_start = min([int(float(r["policy_step"])) for r in phase_rows]) if phase_rows else ""
        window_end = max([int(float(r["policy_step"])) for r in phase_rows]) if phase_rows else ""
        failure_phase = str(summary.get("failure_phase", ""))
        out.append({
            "run_id": summary.get("run_id", ""),
            "task": summary.get("task", ""),
            "condition": summary.get("condition", ""),
            "phase_name": phase,
            "window_start": window_start,
            "window_end": window_end,
            "attacked_frames": len(attacked),
            "OPEN_count": sum(1 for r in attacked if to_float(r.get("adv_grip")) > 0.5),
            "longest_OPEN_streak": longest_open_streak(attacked),
            "token_flips": sum(1 for r in attacked if to_bool(r.get("token_flip"))),
            "qpos_delta_pre": round(delta(phase_rows, "qpos_pre_step"), 6),
            "qpos_delta_post": round(delta(phase_rows, "qpos_post_step"), 6),
            "width_delta_post": round(delta(phase_rows, "width_post_step"), 6),
            "armL2": round(sum(to_float(r.get("arm_l2")) for r in attacked) / len(attacked), 6) if attacked else 0.0,
            "failure_after_phase": str(failure_phase == phase and not bool(summary.get("official_success"))),
        })
    return out


def xid_status(log_dir):
    monitor = log_dir / "xid_monitor.log"
    post = log_dir / "xid_postlaunch.txt"
    texts = []
    for path in (monitor, post):
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(texts)
    return "xid_seen" if "Xid" in text else "no_new_xid_seen_in_logs"


def build_tables(root):
    summaries = [load_summary(p) for p in sorted((root / "runs").glob("*_summary.json"))]
    xids = xid_status(root / "logs")
    rollout_rows = []
    phase_rows = []
    for summary in summaries:
        trace_path = trace_path_from_summary(summary, root)
        phases = phase_rows_from_trace(summary, root)
        phase_rows.extend(phases)
        rollout_rows.append({
            "run_id": summary.get("run_id", ""),
            "task": summary.get("task", ""),
            "condition": summary.get("condition", ""),
            "attack_type": summary.get("attack_type", ""),
            "schedule": summary.get("schedule", ""),
            "seed": summary.get("seed", ""),
            "state_id": summary.get("state_id", ""),
            "official_success": summary.get("official_success", ""),
            "cq_success": summary.get("cq_success", ""),
            "cq_failure": summary.get("cq_failure", ""),
            "manual_audit_needed": summary.get("manual_audit_needed", ""),
            "failure_phase": summary.get("failure_phase", ""),
            "epsilon_pixels": summary.get("epsilon_pixels", ""),
            "attack_steps": summary.get("attack_steps", ""),
            "step_size_pixels": summary.get("step_size_pixels", ""),
            "objective": summary.get("objective", ""),
            "total_attack_frames": summary.get("total_attack_steps", ""),
            "phase_count": len(phases),
            "runtime_s": summary.get("total_dt_s", ""),
            "gpu_pair": summary.get("gpu_pair", ""),
            "xid_status": xids,
            "trace_path": str(trace_path),
        })
    return rollout_rows, phase_rows


def build_random_controls(rollout_rows, phase_rows):
    by_task = {}
    for row in rollout_rows:
        by_task.setdefault(row["task"], []).append(row)
    phase_by_run = {}
    for row in phase_rows:
        phase_by_run.setdefault(row["run_id"], []).append(row)
    out = []
    for task, rows in sorted(by_task.items()):
        randoms = [r for r in rows if r["attack_type"] == "random_linf"]
        viss = [r for r in rows if r["attack_type"] == "vis_pgd" and r["schedule"] == "ultra_three_phase_d20_d20_d20"]
        for rr in randoms:
            vr = viss[0] if viss else {}
            r_phase = phase_by_run.get(rr["run_id"], [])
            v_phase = phase_by_run.get(vr.get("run_id", ""), [])
            r_q = max([to_float(p["qpos_delta_post"]) for p in r_phase], default=0.0)
            v_q = max([to_float(p["qpos_delta_post"]) for p in v_phase], default=0.0)
            r_w = max([to_float(p["width_delta_post"]) for p in r_phase], default=0.0)
            v_w = max([to_float(p["width_delta_post"]) for p in v_phase], default=0.0)
            r_open = sum(int(float(p["OPEN_count"])) for p in r_phase)
            v_open = sum(int(float(p["OPEN_count"])) for p in v_phase)
            reproduced = (str(rr.get("official_success")).lower() == "false") or (v_q > 0 and r_q >= 0.8 * v_q and r_open >= max(1, 0.8 * v_open))
            out.append({
                "task": task,
                "seed": rr.get("seed", ""),
                "state_id": rr.get("state_id", ""),
                "random_run_id": rr.get("run_id", ""),
                "random_condition": rr.get("condition", ""),
                "random_official_success": rr.get("official_success", ""),
                "random_qpos_delta_post": round(r_q, 6),
                "random_width_delta_post": round(r_w, 6),
                "random_open_count": r_open,
                "vis_comparator_run_id": vr.get("run_id", ""),
                "vis_condition": vr.get("condition", ""),
                "vis_official_success": vr.get("official_success", ""),
                "vis_qpos_delta_post": round(v_q, 6),
                "vis_width_delta_post": round(v_w, 6),
                "random_reproduced": str(reproduced),
                "notes": "requires manual qpos/CQ review",
            })
    return out


def build_claims(rollout_rows, random_rows):
    vis_failures = [r for r in rollout_rows if r["attack_type"] == "vis_pgd" and str(r["official_success"]).lower() == "false"]
    random_repro = [r for r in random_rows if str(r["random_reproduced"]).lower() == "true"]
    hard_fail = {r["task"] for r in vis_failures if r["task"] in {"ketchup", "tomato_sauce"}}
    cream_fail = any(r["task"] == "cream_cheese" for r in vis_failures)
    salad_fail = any(r["task"] == "salad_dressing" for r in vis_failures)
    return [
        {
            "claim": "VIS failure upper bound exists",
            "status": "pass" if vis_failures else "pending_or_fail",
            "evidence": ";".join(r["run_id"] for r in vis_failures) if vis_failures else "no VIS failures parsed",
            "notes": "requires qpos/CQ/manual confirmation",
        },
        {
            "claim": "Matched random does not reproduce",
            "status": "pass" if random_rows and not random_repro else "blocked" if random_repro else "pending",
            "evidence": ";".join(r["random_run_id"] for r in random_repro) if random_repro else "no random reproduction parsed",
            "notes": "random reproduction heuristic uses official failure or VIS-like qpos/open trace",
        },
        {
            "claim": "Strong PASS",
            "status": "pass" if cream_fail and hard_fail and random_rows and not random_repro else "pending_or_fail",
            "evidence": f"cream_fail={cream_fail}; hard_fail={','.join(sorted(hard_fail))}",
            "notes": "requires manual audit before final wording",
        },
        {
            "claim": "Medium PASS",
            "status": "pass" if (cream_fail or salad_fail) and not hard_fail else "pending_or_fail",
            "evidence": f"cream_fail={cream_fail}; salad_fail={salad_fail}; hard_fail={','.join(sorted(hard_fail))}",
            "notes": "easy-positive only if hard cases survive",
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("tables"), type=Path)
    args = parser.parse_args()
    rollout_rows, phase_rows = build_tables(args.root)
    random_rows = build_random_controls(rollout_rows, phase_rows)
    claim_rows = build_claims(rollout_rows, random_rows)
    write_csv(args.out_dir / "vis_failure_first_rollout_summary.csv", rollout_rows)
    write_csv(args.out_dir / "vis_failure_first_phasewise_metrics.csv", phase_rows)
    write_csv(args.out_dir / "vis_failure_first_random_controls.csv", random_rows)
    write_csv(args.out_dir / "vis_failure_first_claims.csv", claim_rows)
    print(f"wrote {len(rollout_rows)} rollout rows")
    print(f"wrote {len(phase_rows)} phase rows")
    print(f"wrote {len(random_rows)} random-control rows")


if __name__ == "__main__":
    main()
