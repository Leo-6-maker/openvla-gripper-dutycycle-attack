#!/usr/bin/env python3
"""
Phase 8 Controller V2 — Phase-gated queue manager with atomic claim verification.
Run: python scripts/phase8_controller_v2.py --manifest manifests/ALL_630_JOBS.jsonl --phase P1
"""
import argparse, json, os, sys, time, glob
from pathlib import Path
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="Path to ALL_630_JOBS.jsonl")
ap.add_argument("--phase", default="P1", choices=["P1", "P2", "P3", "P4"])
ap.add_argument("--evidence_root", default="evidence/phase8_cross_suite_v1")
ap.add_argument("--gpus", default="1,2,3,6", help="Comma-separated GPU IDs available")
ap.add_argument("--workers_per_gpu", type=int, default=1)
args = ap.parse_args()

EVIDENCE = Path(args.evidence_root)
QDIR = EVIDENCE / "queue_v2"
RUNS = EVIDENCE / "runs_v2"
QDIR.mkdir(parents=True, exist_ok=True)
for d in ["pending", "running", "done", "failed", "claims", "heartbeats", "logs"]:
    (QDIR / d).mkdir(exist_ok=True)

# Phase definitions
PHASE_DEFS = {
    "P1": {"condition": "P1", "n_expected": 21, "next": "P2"},
    "P2": {"condition": lambda j: j.get("phase", "") == "P2_CLEAN", "n_expected": 90},
    "P3": {"condition": lambda j: j.get("phase", "") == "P3_ATTACK", "n_expected": 360},
    "P4": {"condition": lambda j: j.get("phase", "") == "P4_ARMLOCK", "n_expected": 180},
}

def load_manifest():
    jobs = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs

def filter_phase(jobs, phase):
    if phase == "P1":
        return [j for j in jobs if j.get("phase", "").startswith("P1")]
    else:
        cond = PHASE_DEFS[phase]["condition"]
        return [j for j in jobs if cond(j)]

def seed_queue(jobs):
    seeded = 0
    for j in jobs:
        jid = j["job_id"]
        pending_path = QDIR / "pending" / f"{jid}.json"
        done_path = QDIR / "done" / f"{jid}.json"
        failed_path = QDIR / "failed" / f"{jid}.json"
        if pending_path.exists() or done_path.exists() or failed_path.exists():
            continue
        with open(pending_path, "w") as f:
            json.dump(j, f)
        seeded += 1
    return seeded

def check_phase_gate(phase):
    """Check if all jobs in current phase are done."""
    if phase == "P1":
        jobs = filter_phase(load_manifest(), "P1")
    else:
        jobs = filter_phase(load_manifest(), phase)

    done = 0
    failed_tech = 0
    for j in jobs:
        jid = j["job_id"]
        if (QDIR / "done" / f"{jid}.json").exists():
            done += 1
        elif (QDIR / "failed" / f"{jid}.json").exists():
            failed_tech += 1

    total = len(jobs)
    remaining = total - done - failed_tech
    return {
        "phase": phase, "total": total, "done": done,
        "failed_technical": failed_tech, "remaining": remaining,
        "gate_pass": remaining == 0 and done == total,
    }

def print_status():
    pending = len(list((QDIR / "pending").glob("*.json")))
    running = len(list((QDIR / "running").glob("*.json")))
    done = len(list((QDIR / "done").glob("*.json")))
    failed = len(list((QDIR / "failed").glob("*.json")))
    claims = len(list((QDIR / "claims").glob("*.lock")))

    print(f"\n{'='*50}")
    print(f"Queue V2 Status — {datetime.now(timezone.utc).isoformat()}")
    print(f"Pending: {pending} | Running: {running} | Done: {done} | Failed: {failed}")
    print(f"Active claims: {claims}")
    print(f"Output root: {RUNS}")

    for phase in ["P1", "P2", "P3", "P4"]:
        gate = check_phase_gate(phase)
        if gate["total"] > 0:
            status = "PASS" if gate["gate_pass"] else f"{gate['remaining']} remaining"
            print(f"  {phase}: {gate['done']}/{gate['total']} done, {gate['failed_technical']} tech_fail → {status}")

    # Check run dirs
    run_dirs = list(RUNS.glob("p8_*")) if RUNS.exists() else []
    done_dirs = [d for d in run_dirs if (d / ".done").exists()]
    print(f"Run dirs: {len(run_dirs)} total, {len(done_dirs)} .done")

    return 0

def cmd_status():
    print_status()
    return 0

def cmd_seed():
    all_jobs = load_manifest()
    print(f"Loaded {len(all_jobs)} jobs from manifest")
    phase_jobs = filter_phase(all_jobs, args.phase)
    print(f"Phase {args.phase}: {len(phase_jobs)} jobs")
    n = seed_queue(phase_jobs)
    print(f"Seeded {n} new jobs into pending/")
    print_status()
    return 0

def cmd_gate():
    gate = check_phase_gate(args.phase)
    print(json.dumps(gate, indent=2))
    return 0 if gate["gate_pass"] else 1

def cmd_clear_phase():
    """Clear pending/running for a phase to restart it."""
    phase_jobs = filter_phase(load_manifest(), args.phase)
    jids = {j["job_id"] for j in phase_jobs}
    cleared = 0
    for dname in ["pending", "running"]:
        for f in (QDIR / dname).glob("*.json"):
            jid = f.stem.rsplit(".", 1)[0] if "." in f.stem else f.stem
            if jid in jids:
                f.unlink()
                cleared += 1
    # Clear claims for these jobs
    for lockdir in (QDIR / "claims").glob("*.lock"):
        jid = lockdir.name.replace(".lock", "")
        if jid in jids:
            import shutil
            shutil.rmtree(lockdir, ignore_errors=True)
            cleared += 1
    print(f"Cleared {cleared} entries for phase {args.phase}")
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        sys.exit(cmd_status())
    elif len(sys.argv) > 1 and sys.argv[1] == "seed":
        sys.exit(cmd_seed())
    elif len(sys.argv) > 1 and sys.argv[1] == "gate":
        sys.exit(cmd_gate())
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        sys.exit(cmd_clear_phase())
    else:
        print("Usage: phase8_controller_v2.py [status|seed|gate|clear]")
        print_status()
        sys.exit(0)
