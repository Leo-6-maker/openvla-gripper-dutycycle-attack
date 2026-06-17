#!/usr/bin/env python3
"""D3: Build H3 attack-window interval plans (CPU-only).

For each H2-passing parent, selects the strongest frame, generates
preregistered +-3 step plan, identifies existing/missing frames.
Never changes lambda/epsilon/seeds. Never searches outside +-3.
"""

import csv, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"

SELECTED_PARENTS = {
    "butter_s11": {
        "task": "butter", "state_id": 11, "timing_class": "exact",
        "teacher_ws": 58, "teacher_anchor": 60, "teacher_we": 68, "d5_emit": 60,
        "primary_frames": [58, 60], "diagnostic_frames": [68],
    },
    "tomato_sauce_s23": {
        "task": "tomato_sauce", "state_id": 23, "timing_class": "early",
        "teacher_ws": 139, "teacher_anchor": 141, "teacher_we": 149, "d5_emit": 69,
        "primary_frames": [139, 141], "diagnostic_frames": [69],
    },
    "salad_dressing_s11": {
        "task": "salad_dressing", "state_id": 11, "timing_class": "late",
        "teacher_ws": 57, "teacher_anchor": 59, "teacher_we": 67, "d5_emit": 128,
        "primary_frames": [57, 59], "diagnostic_frames": [67, 128],
    },
}


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def get_existing_frames(parent_dir: str) -> Set[int]:
    """Return set of steps for which .npy files exist."""
    frames = set()
    dp = os.path.join(FRAME_DIR, parent_dir)
    if not os.path.isdir(dp):
        return frames
    for fname in os.listdir(dp):
        if fname.endswith(".npy") and fname.startswith("frame_"):
            try:
                step = int(fname.replace("frame_", "").replace(".npy", ""))
                frames.add(step)
            except ValueError:
                pass
    return frames


class AttackWindowBuilder:
    def __init__(self):
        self.interval_plans = []
        self.warnings = []

    def build_interval(self, pid: str, best_step: int, existing_frames: Set[int]):
        """Generate preregistered +-3 interval around best_step."""
        sel = SELECTED_PARENTS[pid]
        task, sid = sel["task"], sel["state_id"]
        parent_dir = f"{task}_s{sid}_frame"

        interval_start = max(0, best_step - 3)
        interval_end = best_step + 3

        planned_steps = list(range(interval_start, interval_end + 1))
        existing = [s for s in planned_steps if s in existing_frames]
        missing = [s for s in planned_steps if s not in existing_frames]

        # Frame role for each step
        roles = {}
        for s in planned_steps:
            if s == sel["teacher_anchor"]: roles[s] = "teacher_anchor"
            elif s == sel["teacher_ws"]: roles[s] = "teacher_ws"
            elif s == sel["teacher_we"]: roles[s] = "teacher_we"
            elif s == sel["d5_emit"]: roles[s] = "d5_emit"
            else: roles[s] = "interval"

        plan = {
            "parent_id": pid, "task": task, "state_id": sid,
            "timing_class": sel["timing_class"],
            "best_step": best_step,
            "interval_start": interval_start, "interval_end": interval_end,
            "n_planned": len(planned_steps),
            "n_existing": len(existing), "n_missing": len(missing),
            "existing_steps": str(existing), "missing_steps": str(missing),
            "planned_steps": str(planned_steps),
            "teacher_ws": sel["teacher_ws"], "teacher_anchor": sel["teacher_anchor"],
            "teacher_we": sel["teacher_we"], "d5_emit": sel["d5_emit"],
            "best_in_teacher_window": str(sel["teacher_ws"] <= best_step < sel["teacher_we"]),
            "best_d5_relation": "emit" if best_step == sel["d5_emit"] else
                               ("pre" if best_step < sel["d5_emit"] else "post"),
        }

        # Collect SHAs for existing frames
        for s in existing:
            npy_path = os.path.join(FRAME_DIR, parent_dir, f"frame_{s:04d}.npy")
            pt_path = os.path.join(FRAME_DIR, parent_dir, f"processor_{s:04d}.pt")
            plan[f"frame_{s:04d}_npy_sha"] = sha256_file(npy_path)[:16]
            plan[f"frame_{s:04d}_pt_sha"] = sha256_file(pt_path)[:16]

        return plan

    def run(self, h2_frame_results_csv: Optional[str] = None):
        """Generate interval plans from H2 results.

        If h2_frame_results_csv is provided, read passed frames from it.
        Otherwise, generate the preregistered plan template.
        """
        print("=== D3: Attack-Window Interval Builder ===\n")

        if h2_frame_results_csv and os.path.isfile(h2_frame_results_csv):
            # Read H2 results, identify passing frames per parent
            frame_rows = list(csv.DictReader(open(h2_frame_results_csv)))
            parent_passing_frames = defaultdict(list)
            for r in frame_rows:
                if r.get("frame_result") == "FRAME_TWO_SEED_PASS":
                    pid = r["parent_id"]
                    step = int(r["step"])
                    parent_passing_frames[pid].append(step)
        else:
            self.warnings.append("No H2 results available — generating preregistered template only")
            parent_passing_frames = {}

        for pid, sel in SELECTED_PARENTS.items():
            task, sid = sel["task"], sel["state_id"]
            parent_dir = f"{task}_s{sid}_frame"
            existing = get_existing_frames(parent_dir)

            passing = parent_passing_frames.get(pid, [])
            if passing:
                # Choose strongest: prefer in-window > anchor > lowest step
                in_window = [s for s in passing if sel["teacher_ws"] <= s < sel["teacher_we"]]
                if in_window:
                    best_step = min(in_window, key=lambda s: abs(s - sel["teacher_anchor"]))
                else:
                    best_step = min(passing, key=lambda s: abs(s - sel["teacher_anchor"]))
                plan = self.build_interval(pid, best_step, existing)
                plan["h2_status"] = "PASS"
                plan["h2_passing_frames"] = str(passing)
                self.interval_plans.append(plan)
                print(f"  {pid}: best={best_step} interval=[{plan['interval_start']},{plan['interval_end']}] "
                      f"existing={plan['n_existing']} missing={plan['n_missing']}")
            else:
                # Preregistered template: choose anchor or best primary frame
                anchor = sel["teacher_anchor"]
                if anchor in existing:
                    plan = self.build_interval(pid, anchor, existing)
                    plan["h2_status"] = "AWAITING_H2"
                    plan["h2_passing_frames"] = ""
                    self.interval_plans.append(plan)
                    print(f"  {pid}: preregistered anchor={anchor} interval=[{plan['interval_start']},{plan['interval_end']}] "
                          f"(awaiting H2)")
                else:
                    # Use first available primary frame
                    for s in sel["primary_frames"]:
                        if s in existing:
                            plan = self.build_interval(pid, s, existing)
                            plan["h2_status"] = "AWAITING_H2"
                            plan["h2_passing_frames"] = ""
                            self.interval_plans.append(plan)
                            break
                    else:
                        self.warnings.append(f"{pid}: no primary frame available for interval template")

        self._write_outputs()
        print(f"\n  Plans: {len(self.interval_plans)} parents")
        for w in self.warnings:
            print(f"  WARNING: {w}")

    def _write_outputs(self):
        out_dir = REPO_ROOT / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = REPO_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if self.interval_plans:
            fields = list(self.interval_plans[0].keys())
            # Drop per-frame SHA columns from main table
            main_fields = [k for k in fields if not k.endswith("_sha")]
            sha_fields = [k for k in fields if k.endswith("_sha")]

            with open(out_dir / "l3_attack_window_intervals.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=main_fields, extrasaction="ignore")
                w.writeheader(); w.writerows(self.interval_plans)

            if sha_fields:
                sha_rows = []
                for plan in self.interval_plans:
                    row = {"parent_id": plan["parent_id"], "best_step": plan["best_step"]}
                    for k in sha_fields:
                        row[k] = plan.get(k, "")
                    sha_rows.append(row)
                with open(out_dir / "l3_attack_window_frame_evidence.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(sha_rows[0].keys()))
                    w.writeheader(); w.writerows(sha_rows)

        with open(reports_dir / "L3_ATTACK_WINDOW_INTERVAL_AUDIT.md", "w") as f:
            f.write("# L3 H3 Attack-Window Interval Audit\n\n")
            f.write(f"**Plans generated:** {len(self.interval_plans)}\n\n")
            for plan in self.interval_plans:
                f.write(f"## {plan['parent_id']} ({plan['timing_class']})\n\n")
                f.write(f"- Status: {plan['h2_status']}\n")
                f.write(f"- Best step: {plan['best_step']}\n")
                f.write(f"- Interval: [{plan['interval_start']}, {plan['interval_end']}]\n")
                f.write(f"- Existing frames: {plan['n_existing']}\n")
                f.write(f"- Missing frames: {plan['n_missing']}\n")
                f.write(f"- Teacher window: [{plan['teacher_ws']}, {plan['teacher_we']})\n")
                f.write(f"- D5 emit: {plan['d5_emit']}\n")
                if plan.get("h2_passing_frames"):
                    f.write(f"- H2 passing frames: {plan['h2_passing_frames']}\n")
                f.write("\n")
            if self.warnings:
                f.write("## Warnings\n\n")
                for w in self.warnings:
                    f.write(f"- {w}\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2-results", default="",
                    help="Path to l3_vis_independent_frame_results.csv from D2")
    args = ap.parse_args()

    builder = AttackWindowBuilder()
    builder.run(args.h2_results if args.h2_results else None)


if __name__ == "__main__":
    main()
