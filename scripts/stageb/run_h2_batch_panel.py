#!/usr/bin/env python3
"""H2 Batch Panel — Sequential M3 True-PGD Canary Runner for remaining 19 jobs.

Runs the remaining 19 H2 jobs after H1 (butter_s11 step60 seed81) completes.
Each job is a subprocess call to run_m3_step78_true_pgd_fixed_frame.py --mode canary.

GPU(1,5) are reserved for DeepSeek L3. This script uses a separate GPU set
(configurable via H2_CUDA_VISIBLE_DEVICES, default=0,2,3,4).

Usage:
    python scripts/stageb/run_h2_batch_panel.py [--dry-run] [--resume]

Exit code: 0 if all jobs pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ── Environment constants (Linux server paths) ───────────────────────────
MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
CONFIG_PATH = REPO_ROOT / "configs" / "m3_step78_true_pgd_31744.yaml"
CANONICAL_DIR = Path(
    "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages"
)
OUTPUT_ROOT = Path("/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1")
H2_INPUT_BASE = OUTPUT_ROOT / "h2_inputs"
H2_OUTPUT_BASE = OUTPUT_ROOT / "h2_outputs"
RUNNER = (
    REPO_ROOT / "scripts" / "stageb" / "run_m3_step78_true_pgd_fixed_frame.py"
)
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"

# GPU(1,5) reserved for DeepSeek L3 — use remaining GPUs.
# Override via H2_CUDA_VISIBLE_DEVICES env var.
AVAILABLE_GPUS = os.environ.get("H2_CUDA_VISIBLE_DEVICES", "0,2,3,4")

# ── Job definitions ──────────────────────────────────────────────────────
# H1 = butter_s11 step60 seed81 (already running / done).
# Remaining 19 jobs below.

H2_JOBS: List[Dict[str, Any]] = [
    # ── Eligible primary (3 frames, minus H1 seed81) ──
    {"parent": "butter_s11",        "step": 60,  "seeds": [82],          "group": "primary_eligible"},
    {"parent": "tomato_sauce_s23",  "step": 141, "seeds": [81, 82],     "group": "primary_eligible"},
    {"parent": "salad_dressing_s11","step": 59,  "seeds": [81, 82],     "group": "primary_eligible"},
    # ── Ineligible primary (ws already OPEN) ──
    {"parent": "butter_s11",        "step": 58,  "seeds": [81, 82],     "group": "primary_ineligible"},
    {"parent": "tomato_sauce_s23",  "step": 139, "seeds": [81, 82],     "group": "primary_ineligible"},
    {"parent": "salad_dressing_s11","step": 57,  "seeds": [81, 82],     "group": "primary_ineligible"},
    # ── Diagnostic frames ──
    {"parent": "butter_s11",        "step": 68,  "seeds": [81, 82],     "group": "diagnostic"},
    {"parent": "tomato_sauce_s23",  "step": 69,  "seeds": [81, 82],     "group": "diagnostic"},
    {"parent": "salad_dressing_s11","step": 67,  "seeds": [81, 82],     "group": "diagnostic"},
    {"parent": "salad_dressing_s11","step": 128, "seeds": [81, 82],     "group": "diagnostic"},
]

# Verify total count
assert sum(len(spec["seeds"]) for spec in H2_JOBS) == 19, (
    f"Expected 19 H2 jobs, got {sum(len(spec['seeds']) for spec in H2_JOBS)}"
)


# ── Helper functions ─────────────────────────────────────────────────────

def load_clean_generation(canonical_pkg_dir: Path) -> Dict[str, Any]:
    """Load the canonical clean_generation.json."""
    path = canonical_pkg_dir / "clean_generation.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical clean_generation.json: {path}")
    return json.loads(path.read_text("utf-8"))


def convert_to_step78_format(clean_gen: Dict[str, Any]) -> Dict[str, Any]:
    """Convert canonical clean_generation.json -> clean_generation_step78.json format.

    The runner expects:
      - instruction
      - clean_action
      - clean_exact_7_tokens  (canonical stores it as "exact_clean_7_tokens")
      - official: {...}       (metadata wrapper, runner re-decodes anyway)
    """
    return {
        "instruction": clean_gen["instruction"],
        "clean_action": clean_gen["clean_action"],
        "clean_exact_7_tokens": clean_gen["exact_clean_7_tokens"],
        "official": {
            "tokens": clean_gen.get("exact_clean_7_tokens", []),
            "arm_prefix": clean_gen.get("clean_arm_prefix", []),
            "gripper_token": clean_gen.get("clean_gripper_token", -1),
            "score_row_sha256": "",
            "score_invariant": clean_gen.get("official_score_invariant", {}),
            "target_stats": {},
            "score_audit": {},
        },
    }


def prepare_input_dir(parent: str, step: int, seed: int) -> Path:
    """Set up input directory with raw_agentview_step78.npy and
    clean_generation_step78.json from the canonical package."""
    canonical_pkg_dir = CANONICAL_DIR / f"{parent}_step{step:04d}"
    input_dir = H2_INPUT_BASE / f"{parent}_step{step:04d}_seed{seed}"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw_frame.npy -> raw_agentview_step78.npy
    src_raw = canonical_pkg_dir / "raw_frame.npy"
    dst_raw = input_dir / "raw_agentview_step78.npy"
    if not dst_raw.exists():
        shutil.copy2(str(src_raw), str(dst_raw))

    # Convert and write clean_generation_step78.json
    dst_gen = input_dir / "clean_generation_step78.json"
    if not dst_gen.exists():
        clean_gen = load_clean_generation(canonical_pkg_dir)
        step78_gen = convert_to_step78_format(clean_gen)
        dst_gen.write_text(json.dumps(step78_gen, indent=2), "utf-8")

    return input_dir


def run_one_job(
    parent: str,
    step: int,
    seed: int,
    group: str,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single canary job via subprocess. Returns result dict."""
    input_dir = prepare_input_dir(parent, step, seed)
    output_dir = H2_OUTPUT_BASE / f"{parent}_step{step:04d}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "parent": parent,
        "step": step,
        "seed": seed,
        "group": group,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "status": "PENDING",
        "return_code": -1,
        "duration_s": 0.0,
        "stdout_tail": "",
        "finished_at": "",
    }

    if dry_run:
        result["status"] = "DRY_RUN"
        print(f"  [DRY RUN] {parent} step{step} seed{seed} ({group})")
        return result

    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--config", str(CONFIG_PATH),
        "--mode", "canary",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--attack_seed", str(seed),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = AVAILABLE_GPUS

    label = f"{parent} step{step} seed{seed}"
    print(f"  [{label}] Starting...")
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minutes per job (should take ~45-60s)
            cwd=str(REPO_ROOT),
        )
        duration = time.time() - t0
        result["duration_s"] = round(duration, 1)
        result["return_code"] = proc.returncode
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")

        # Extract tail of stdout for diagnostic context
        stdout_lines = proc.stdout.strip().split("\n")
        stderr_lines = (
            proc.stderr.strip().split("\n") if proc.stderr.strip() else []
        )
        tail = stdout_lines[-min(len(stdout_lines), 10):]
        if stderr_lines:
            tail.append("--- stderr tail ---")
            tail.extend(stderr_lines[-min(len(stderr_lines), 5):])
        result["stdout_tail"] = "\n".join(tail)

        # Parse status from last stdout JSON line
        status = "FAIL"
        for line in reversed(stdout_lines):
            line = line.strip()
            if '"status": "CANARY_COMPLETE_UNCLASSIFIED"' in line:
                status = "PASS"
                break
            if '"status": "PREFLIGHT_MISMATCH"' in line:
                # Preflight failed — score path mismatch between surrogate and official.
                # This is a canary FAILURE (the surrogate is not reliable for this frame).
                status = "PREFLIGHT_MISMATCH"
                break
            if '"status": "PREFLIGHT_PASS"' in line:
                status = "PREFLIGHT_ONLY"
                break

        if proc.returncode != 0 and status in ("FAIL",):
            status = "CRASH"
        elif proc.returncode != 0:
            status = f"{status}_WITH_ERROR"

        result["status"] = status
        print(
            f"  [{label}] {status} ({result['duration_s']:.1f}s)"
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        result["duration_s"] = round(duration, 1)
        result["status"] = "TIMEOUT"
        result["return_code"] = -9
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"  [{label}] TIMEOUT ({result['duration_s']:.1f}s)")

    except Exception as e:
        duration = time.time() - t0
        result["duration_s"] = round(duration, 1)
        result["status"] = f"ERROR"
        result["stdout_tail"] = str(e)
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"  [{label}] ERROR: {e}")

    return result


def write_ledger(ledger: List[Dict[str, Any]], path: Path) -> None:
    """Write job ledger as CSV with summary fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parent", "step", "seed", "group", "status",
        "return_code", "duration_s", "input_dir", "output_dir",
        "finished_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ledger:
            writer.writerow(row)


def print_summary(ledger: List[Dict[str, Any]]) -> None:
    """Print a formatted summary of all job results."""
    total = len(ledger)
    passed = sum(1 for r in ledger if r["status"] == "PASS")
    skipped = sum(1 for r in ledger if r["status"] in ("SKIPPED", "DRY_RUN"))
    failed = [
        r for r in ledger
        if r["status"] not in ("PASS", "SKIPPED", "DRY_RUN")
    ]

    passed_msg = "PASS" if not failed else "FAIL"

    print()
    print("=" * 68)
    print(f"  H2 Batch Panel Summary  [{passed_msg}]")
    print("=" * 68)
    print(f"  Total jobs:  {total}")
    print(f"  Passed:      {passed}")
    print(f"  Skipped:     {skipped}")
    print(f"  Failed:      {len(failed)}")
    print()
    print(f"  Wall elapsed: completed jobs only (subprocess time)")

    # Group breakdown
    for group in ["primary_eligible", "primary_ineligible", "diagnostic"]:
        grp = [r for r in ledger if r["group"] == group]
        active = [r for r in grp if r["status"] not in ("SKIPPED", "DRY_RUN")]
        if active:
            g_pass = sum(1 for r in active if r["status"] == "PASS")
            g_fail = [r for r in active if r["status"] != "PASS"]
            print()
            print(f"  [{group}] {g_pass}/{len(active)} passed")
            if g_fail:
                for r in g_fail:
                    print(
                        f"    FAIL  {r['parent']} step{r['step']} "
                        f"seed{r['seed']}: {r['status']} "
                        f"({r['duration_s']:.1f}s)"
                    )
                    tail = r.get("stdout_tail", "")
                    if tail:
                        # Show last meaningful line
                        for line in tail.split("\n")[-3:]:
                            print(f"          | {line}")

    print()
    if failed:
        print("  All failed jobs:")
        for r in failed:
            print(
                f"    - {r['parent']} step{r['step']} seed{r['seed']}: "
                f"{r['status']} ({r['duration_s']:.1f}s)"
            )
        print()
    print("=" * 68)


def verify_canonical_packages() -> bool:
    """Verify all required canonical packages exist before starting."""
    seen = set()
    all_ok = True
    for spec in H2_JOBS:
        for seed in spec["seeds"]:
            key = (spec["parent"], spec["step"])
            if key in seen:
                continue
            seen.add(key)
            pkg_dir = CANONICAL_DIR / f"{spec['parent']}_step{spec['step']:04d}"
            raw = pkg_dir / "raw_frame.npy"
            gen = pkg_dir / "clean_generation.json"
            if not raw.exists():
                print(f"  MISSING: {raw}")
                all_ok = False
            if not gen.exists():
                print(f"  MISSING: {gen}")
                all_ok = False
    return all_ok


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="H2 Batch Panel — 19 sequential M3 true-PGD canary jobs"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job list and input setup without executing",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose output_dir already has condition_results.csv",
    )
    args = ap.parse_args()

    print("=" * 68)
    print("  H2 Batch Panel")
    print("=" * 68)
    print(f"  Runner:      {RUNNER}")
    print(f"  Config:      {CONFIG_PATH}")
    print(f"  Python:      {PYTHON}")
    print(f"  GPUs:        {AVAILABLE_GPUS}")
    print(f"  Canonical:   {CANONICAL_DIR}")
    print(f"  Input base:  {H2_INPUT_BASE}")
    print(f"  Output base: {H2_OUTPUT_BASE}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Resume:      {args.resume}")
    print()

    if not CANONICAL_DIR.exists():
        print(
            f"ERROR: Canonical packages directory not found:\n"
            f"  {CANONICAL_DIR}\n"
            f"Run build_h0_canonical_packages.py first."
        )
        return 1

    if not args.dry_run:
        print("Verifying canonical packages...")
        if not verify_canonical_packages():
            print("ERROR: Missing canonical packages. Aborting.")
            return 1
        print("  All canonical packages found.")

    # Expand H2_JOBS into individual (parent, step, seed) items
    all_jobs: List[Dict[str, Any]] = []
    for spec in H2_JOBS:
        for seed in spec["seeds"]:
            all_jobs.append({
                "parent": spec["parent"],
                "step": spec["step"],
                "seed": seed,
                "group": spec["group"],
            })

    print(f"\nTotal individual jobs: {len(all_jobs)}")
    print()

    # ── Sequential execution ──────────────────────────────────────────
    ledger: List[Dict[str, Any]] = []
    n_total = len(all_jobs)
    t_start = time.time()

    for i, job in enumerate(all_jobs, 1):
        output_dir = (
            H2_OUTPUT_BASE
            / f"{job['parent']}_step{job['step']:04d}_seed{job['seed']}"
        )

        # Resume: skip if output_dir already has results
        if args.resume and (
            output_dir / "m3_step78_condition_results.csv"
        ).exists():
            print(
                f"[{i:2d}/{n_total}] SKIP (exists): "
                f"{job['parent']} step{job['step']} seed{job['seed']}"
            )
            ledger.append({
                **job,
                "status": "SKIPPED",
                "return_code": 0,
                "duration_s": 0.0,
                "input_dir": str(
                    H2_INPUT_BASE
                    / f"{job['parent']}_step{job['step']:04d}_seed{job['seed']}"
                ),
                "output_dir": str(output_dir),
                "finished_at": "",
            })
            continue

        if not args.dry_run:
            print(
                f"[{i:2d}/{n_total}] {job['parent']} "
                f"step{job['step']} seed{job['seed']} "
                f"({job['group']})"
            )

        result = run_one_job(
            job["parent"],
            job["step"],
            job["seed"],
            job["group"],
            dry_run=args.dry_run,
        )
        ledger.append(result)

        # Brief cooldown between jobs for GPU memory cleanup
        if not args.dry_run and result["status"] not in ("DRY_RUN",):
            time.sleep(1)

    t_elapsed = time.time() - t_start

    # ── Write ledger ──────────────────────────────────────────────────
    ledger_path = H2_OUTPUT_BASE / "h2_batch_ledger.csv"
    write_ledger(ledger, ledger_path)
    print(f"\nLedger written: {ledger_path}")

    # Write a JSON summary next to the ledger
    summary_path = H2_OUTPUT_BASE / "h2_batch_summary.json"
    passed = sum(1 for r in ledger if r["status"] == "PASS")
    failed_count = sum(
        1 for r in ledger if r["status"] not in ("PASS", "SKIPPED", "DRY_RUN")
    )
    summary_path.write_text(
        json.dumps(
            {
                "pipeline": "H2_BATCH_PANEL",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "total_jobs": len(ledger),
                "passed": passed,
                "failed": failed_count,
                "skipped_or_dry": sum(
                    1 for r in ledger if r["status"] in ("SKIPPED", "DRY_RUN")
                ),
                "elapsed_seconds": round(t_elapsed, 1),
                "overall_status": "PASS" if failed_count == 0 else "FAIL",
                "results": [
                    {
                        "parent": r["parent"],
                        "step": r["step"],
                        "seed": r["seed"],
                        "group": r["group"],
                        "status": r["status"],
                        "duration_s": r["duration_s"],
                    }
                    for r in ledger
                ],
            },
            indent=2,
        ),
        "utf-8",
    )
    print(f"Summary written: {summary_path}")

    # ── Print summary ─────────────────────────────────────────────────
    print_summary(ledger)

    failed_count = sum(
        1 for r in ledger if r["status"] not in ("PASS", "SKIPPED", "DRY_RUN")
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
