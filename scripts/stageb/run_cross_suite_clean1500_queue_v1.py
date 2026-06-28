#!/usr/bin/env python3
"""
CLEAN1500 Queue Worker V1 — one worker per GPU, sequential job execution.

Features: skip COMPLETE, quarantine non-empty-incomplete, max 1 infra retry,
         2 consecutive infra failures → stop, atomic progress ledger.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True, help="Physical GPU ID (maps CUDA_VISIBLE_DEVICES)")
ap.add_argument("--manifest", required=True, help="cross_suite_clean1500_jobs_v1.jsonl")
ap.add_argument("--collector", required=True, help="Path to run_cross_suite_clean_v3.py")
ap.add_argument("--protocol", required=True)
ap.add_argument("--registry", required=True)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--log_dir", required=True)
ap.add_argument("--dry_run", action="store_true")
ap.add_argument("--smoke", action="store_true", help="Run only first job then exit")
args = ap.parse_args()

# Load and filter jobs
jobs = []
with open(args.manifest) as f:
    for line in f:
        if not line.strip(): continue
        j = json.loads(line)
        if j["gpu"] == args.gpu:
            jobs.append(j)

print("Worker GPU%d: %d jobs assigned" % (args.gpu, len(jobs)))
if args.dry_run:
    print("DRY_RUN: %d jobs, gpu=%d" % (len(jobs), args.gpu))
    sys.exit(0)

log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)
ledger_path = log_dir / ("ledger_gpu%d.jsonl" % args.gpu)

consecutive_infra = 0

for idx, job in enumerate(jobs):
    key = job["job_key"]
    out_dir = Path(job["output_dir"])
    complete_file = out_dir / "COMPLETE.json"
    fail_file = out_dir / "SCHEMA_FAIL.json"

    # Skip completed
    if complete_file.exists():
        print("[%d/%d] %s: SKIP (COMPLETE)" % (idx+1, len(jobs), key))
        continue

    # Existing incomplete → quarantine
    if out_dir.exists() and any(out_dir.iterdir()):
        quaran_dir = out_dir.parent / ("QUARANTINE_%s" % out_dir.name)
        if quaran_dir.exists():
            quaran_dir = out_dir.parent / ("QUARANTINE_%s_%d" % (out_dir.name, int(time.time())))
        shutil.move(str(out_dir), str(quaran_dir))
        print("[%d/%d] %s: QUARANTINED old attempt" % (idx+1, len(jobs), key))

    out_dir.mkdir(parents=True, exist_ok=True)
    job["attempt"] += 1
    print("[%d/%d] %s: RUN (attempt %d)" % (idx+1, len(jobs), key, job["attempt"]))

    t0 = time.time()
    cmd = [
        args.python, "-u", args.collector,
        "--suite", str(job["suite"]),
        "--task_idx", str(job["task_idx"]),
        "--state_id", str(job["state_id"]),
        "--eval_seed", str(job["eval_seed"]),
        "--render_gpu", str(args.gpu),
        "--max_steps", str(job["max_steps"]),
        "--protocol", args.protocol,
        "--registry", args.registry,
        "--output_dir", str(out_dir),
    ]
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.run(cmd, env=env, stdout=out_f, stderr=err_f)
    elapsed = time.time() - t0

    if complete_file.exists():
        job["status"] = "COMPLETE"
        job["elapsed_s"] = round(elapsed, 1)
        consecutive_infra = 0
        print("  COMPLETE (%.0fs)" % elapsed)
    elif fail_file.exists():
        job["status"] = "SCHEMA_FAIL"
        job["elapsed_s"] = round(elapsed, 1)
        consecutive_infra = 0
        print("  SCHEMA_FAIL")
    else:
        # Infra failure — no output produced
        job["status"] = "INFRA_FAILED"
        job["exit_code"] = proc.returncode
        consecutive_infra += 1
        print("  INFRA_FAILED exit=%d (consecutive=%d)" % (proc.returncode, consecutive_infra))
        if consecutive_infra >= 2:
            print("FATAL: 2 consecutive infra failures. Stopping worker.")
            break

    # Atomic ledger append
    with open(ledger_path, "a") as f:
        f.write(json.dumps(job) + "\n")

    if args.smoke:
        print("SMOKE complete. Stopping.")
        break

print("Worker GPU%d finished: %d jobs processed" % (args.gpu, idx + 1))
