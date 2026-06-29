#!/usr/bin/env python3
"""TRUE_T10 closure validator — strict fail-closed. All checks mandatory."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import Counter
from pathlib import Path

EXPECTED_JOBS = 162
EXPECTED_PARENTS = 54
EXPECTED_FOLDS = 9
EXPECTED_PER_FOLD = 18
EXPECTED_PERT_REPLICATES = 3


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Strict TRUE_T10 closure validation")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected_bridge_sha", help="Expected runtime bridge SHA-256")
    ap.add_argument("--expected_worker_sha", help="Expected runtime worker SHA-256")
    ap.add_argument("--output", default="-")
    ap.add_argument("--fail_on_error", action="store_true", default=True)
    args = ap.parse_args()

    jobs = []
    with open(args.manifest) as f:
        for line in f:
            jobs.append(json.loads(line))
    errors = []

    # 1. Exact counts
    if len(jobs) != EXPECTED_JOBS:
        errors.append(f"Job count: {len(jobs)} != {EXPECTED_JOBS}")

    job_keys = [j.get("job_key", "?") for j in jobs]
    dup_keys = [k for k, v in Counter(job_keys).items() if v > 1]
    if dup_keys:
        errors.append(f"Duplicate job_keys: {dup_keys}")

    output_dirs = [j.get("output_dir", "?") for j in jobs]
    dup_dirs = [d for d, v in Counter(output_dirs).items() if v > 1]
    if dup_dirs:
        errors.append(f"Duplicate output_dirs: {dup_dirs}")

    # 2. Parents
    parents = set()
    for j in jobs:
        parents.add((str(j.get("fold", "")), str(j.get("state_id", "")),
                      str(j.get("detector_seed", ""))))
    if len(parents) != EXPECTED_PARENTS:
        errors.append(f"Parent count: {len(parents)} != {EXPECTED_PARENTS}")

    pert_count = Counter()
    for j in jobs:
        pert_count[(str(j.get("fold", "")), str(j.get("state_id", "")),
                     str(j.get("detector_seed", "")))] += 1
    bad = [(k, v) for k, v in pert_count.items() if v != EXPECTED_PERT_REPLICATES]
    if bad:
        errors.append(f"Parents with wrong replicate count: {len(bad)}")

    # 3. Per-fold
    by_fold = Counter()
    for j in jobs:
        by_fold[str(j.get("fold", ""))] += 1
    for f in sorted(by_fold):
        if by_fold[f] != EXPECTED_PER_FOLD:
            errors.append(f"fold_{f}: {by_fold[f]} != {EXPECTED_PER_FOLD}")
    if len(by_fold) != EXPECTED_FOLDS:
        errors.append(f"Fold count: {len(by_fold)} != {EXPECTED_FOLDS}")

    # 4. Episode summary existence + parse
    missing, unparseable, oom = [], [], []
    succ, total_sr = 0, 0
    fold_present = Counter()
    for j in jobs:
        out = j.get("output_dir", "")
        ep_path = os.path.join(out, "episode_summary.json") if out else ""
        sf_path = os.path.join(out, "stderr.log") if out else ""
        if not ep_path or not Path(ep_path).exists():
            missing.append(j.get("job_key", "?"))
            continue
        try:
            d = json.loads(open(ep_path).read())
            total_sr += 1
            if d.get("task_success"):
                succ += 1
            fold_present[str(j.get("fold", ""))] += 1
        except Exception as e:
            unparseable.append(f"{j.get('job_key', '?')}: {e}")
        if sf_path and Path(sf_path).exists():
            if "OutOfMemoryError" in open(sf_path).read():
                oom.append(j.get("job_key", "?"))

    if missing:
        errors.append(f"Missing episode_summary: {len(missing)}")
    if unparseable:
        errors.append(f"Unparseable episode_summary: {len(unparseable)}")
    for f in sorted(by_fold):
        if fold_present.get(f, 0) != EXPECTED_PER_FOLD:
            errors.append(f"fold_{f} present: {fold_present.get(f, 0)} != {EXPECTED_PER_FOLD}")

    # 5. SHA verification
    if args.expected_bridge_sha:
        actual = sha256_file("/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py")
        if actual != args.expected_bridge_sha:
            errors.append(f"Bridge SHA: expected {args.expected_bridge_sha[:16]}... got {actual[:16]}...")
    if args.expected_worker_sha:
        actual = sha256_file("/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py")
        if actual != args.expected_worker_sha:
            errors.append(f"Worker SHA: expected {args.expected_worker_sha[:16]}... got {actual[:16]}...")

    report = {
        "gate": "TRUE_T10_CLOSURE_V2",
        "closure_pass": len(errors) == 0,
        "errors": errors,
        "total_jobs": len(jobs), "present_parsed": total_sr,
        "missing": len(missing), "unparseable": len(unparseable),
        "oom_stderr_count": len(oom),
        "duplicate_job_keys": dup_keys, "duplicate_output_dirs": dup_dirs,
        "n_parents": len(parents),
        "per_fold_present": dict(fold_present),
        "task_sr": {"success": succ, "total": total_sr} if total_sr > 0 else None,
    }

    if args.output == "-":
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    if errors:
        print(f"\nCLOSURE FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"  FAIL: {e}")
        if args.fail_on_error:
            sys.exit(1)
    else:
        print(f"\nCLOSURE PASS: {total_sr}/{len(jobs)} parsed, {succ}/{total_sr} success, {len(oom)} OOM")


if __name__ == "__main__":
    main()
