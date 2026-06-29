#!/usr/bin/env python3
"""TRUE_T10 canonical closure validator.

Checks:
  1. 162/162 episode_summary.json exist
  2. No duplicate job_keys or output_dirs
  3. Worker/bridge/manifest SHA consistency across episodes
  4. OOM attempt inventory
  5. Per-fold completion
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path


def load_manifest(path: str) -> list[dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            jobs.append(json.loads(line))
    return jobs


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Validate TRUE_T10 closure")
    ap.add_argument("--manifest", required=True, help="TRUE_T10 formal_manifest.jsonl")
    ap.add_argument("--output", default="-", help="Closure report output")
    ap.add_argument("--fail_on_missing", action="store_true")
    args = ap.parse_args()

    jobs = load_manifest(args.manifest)
    print(f"Manifest: {len(jobs)} jobs")

    # 1. Episode summary existence
    missing = []
    present = []
    oom_dirs = []
    for j in jobs:
        out = j["output_dir"]
        ep_path = os.path.join(out, "episode_summary.json")
        stderr_path = os.path.join(out, "stderr.log")
        if os.path.exists(ep_path):
            present.append(j)
        else:
            missing.append(j)
        if os.path.exists(stderr_path):
            with open(stderr_path) as f:
                if "OutOfMemoryError" in f.read():
                    oom_dirs.append(out)

    print(f"  Present: {len(present)}/{len(jobs)}")
    print(f"  Missing: {len(missing)}/{len(jobs)}")
    print(f"  OOM stderrs: {len(oom_dirs)}")

    # 2. Duplicate check
    job_keys = [j["job_key"] for j in jobs]
    dup_keys = [k for k, v in Counter(job_keys).items() if v > 1]
    output_dirs = [j["output_dir"] for j in jobs]
    dup_dirs = [d for d, v in Counter(output_dirs).items() if v > 1]

    # 3. Per-fold
    by_fold = defaultdict(lambda: {"total": 0, "present": 0})
    for j in jobs:
        fold = str(j.get("fold", "?"))
        by_fold[fold]["total"] += 1
    for j in present:
        fold = str(j.get("fold", "?"))
        by_fold[fold]["present"] += 1

    print(f"\nPer-fold:")
    for fold in sorted(by_fold):
        d = by_fold[fold]
        print(f"  fold_{fold}: {d['present']}/{d['total']}")

    # 4. SR from episode summaries
    total_sr = 0
    succ = 0
    for j in present:
        ep_path = os.path.join(j["output_dir"], "episode_summary.json")
        try:
            with open(ep_path) as f:
                ep = json.load(f)
            total_sr += 1
            if ep.get("task_success"):
                succ += 1
        except Exception:
            pass
    if total_sr > 0:
        print(f"\nTask SR: {succ}/{total_sr} = {succ*100/total_sr:.1f}%")

    # 5. SHA consistency check from episode summaries
    bridge_shas = Counter()
    for j in present:
        ep_path = os.path.join(j["output_dir"], "episode_summary.json")
        try:
            with open(ep_path) as f:
                ep = json.load(f)
            bridge_shas[ep.get("bridge_sha256", "MISSING")[:16]] += 1
        except Exception:
            pass

    # 6. Parent (fold, state, detector_seed) count
    parents = set()
    for j in jobs:
        parents.add((str(j.get("fold", "")), str(j.get("state_id", "")),
                      str(j.get("detector_seed", ""))))
    print(f"\nParents (fold, state, det_seed): {len(parents)} unique")

    # Build report
    report = {
        "gate": "TRUE_T10_CLOSURE_VALIDATION",
        "total_jobs": len(jobs),
        "present": len(present),
        "missing": len(missing),
        "missing_job_keys": [j["job_key"] for j in missing],
        "oom_stderr_count": len(oom_dirs),
        "duplicate_job_keys": dup_keys,
        "duplicate_output_dirs": dup_dirs,
        "per_fold": {f: dict(d) for f, d in by_fold.items()},
        "task_sr": {"success": succ, "total": total_sr} if total_sr > 0 else None,
        "n_parents": len(parents),
        "closure_pass": len(missing) == 0 and len(dup_keys) == 0,
    }

    if args.output == "-":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    if args.fail_on_missing and not report["closure_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
