#!/usr/bin/env python3
"""Launch formal Factorized Student OOF training on GPU 1 and 3.

Workers per GPU: use --workers flag (default 6).
Fail-closed: any run failure → overall HOLD.
Existing outputs verified via seal before SKIP.
"""
import subprocess, sys, time, json, hashlib, argparse
from pathlib import Path
from collections import defaultdict

PY = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_factorized_oof.py"
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
S1 = OPS / "OFFICIAL_V3_S1_FIT_V1_d31187f"
TEACHER = OPS / "OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721"
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
POLICY_INTENT = OPS / "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_01"
AUTH_ROOT = OPS / "OFFICIAL_V3_FACTORIZED_STUDENT_OOF_AUTH"
F3_ROOT = OPS / "OFFICIAL_V3_F3_GEOMETRY_GATE_de07e1a_20260721"
DEFAULT_OUT_BASE = OPS / "OFFICIAL_V3_FACTORIZED_STUDENT_OOF"

def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): d.update(b)
    return d.hexdigest()

def verify_existing_output(out: Path) -> bool:
    """Verify sealed output is valid before skipping."""
    if not out.exists():
        return False
    sums = out / "SHA256SUMS"
    sidecar = out / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        return False
    expected = sidecar.read_text().strip()
    actual = f"{sha256_file(sums)}  SHA256SUMS"
    return expected == actual

SEEDS = [42, 123, 456]
FOLD_IDS = [0, 1, 2, 3]
GPUS = [1, 3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--source-commit", type=str, required=True)
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUT_BASE)
    args = ap.parse_args()

    WORKERS_PER_GPU = args.workers
    OUT_BASE = args.output_base

    jobs = []
    for mt in ["25D9D", "25D"]:
        for fold in FOLD_IDS:
            for seed in SEEDS:
                out = OUT_BASE / mt / f"fold{fold}_seed{seed}"
                jobs.append((mt, fold, seed, out))

    print(f"Total jobs: {len(jobs)} | GPUs: {GPUS} | workers/GPU: {WORKERS_PER_GPU}")

    def launch_worker(gpu, worker_id, job_list):
        env = {}
        env.update(__import__("os").environ)
        env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "PYTHONPATH": "/mnt/sdc/dty_user/openvla_attack/src",
                     "CUDA_VISIBLE_DEVICES": str(gpu)})
        LOG_DIR = Path("/mnt/sdc/dty_user/openvla_attack/logs/oof")
        results = []
        for mt, fold, seed, out in job_list:
            log_file = LOG_DIR / f"{mt}_fold{fold}_seed{seed}_gpu{gpu}_w{worker_id}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            if verify_existing_output(out):
                print(f"  [GPU{gpu} W{worker_id}] SKIP {mt} fold{fold} seed{seed} (sealed)")
                results.append((mt, fold, seed, True, "SKIP_SEALED"))
                continue

            extra_args = []
            if mt == "25D9D":
                extra_args = ["--policy-intent-root", str(POLICY_INTENT)]

            print(f"  [GPU{gpu} W{worker_id}] START {mt} fold{fold} seed{seed}")
            start = time.time()
            with open(log_file, "w") as lf:
                r = subprocess.run(
                    [PY, str(TRAIN_SCRIPT),
                     "--model-type", mt, "--fold-id", str(fold), "--seed", str(seed), "--gpu", "0",
                     "--output-root", str(out),
                     "--s1-root", str(S1), "--teacher-root", str(TEACHER), "--fold-root", str(FOLD_ROOT),
                     "--authorization-root", str(AUTH_ROOT), "--f3-root", str(F3_ROOT),
                     "--expected-source-commit", args.source_commit] + extra_args,
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - start
            success = r.returncode == 0
            if success:
                print(f"  [GPU{gpu} W{worker_id}] DONE  {mt} fold{fold} seed{seed} ({elapsed:.0f}s)")
            else:
                print(f"  [GPU{gpu} W{worker_id}] FAIL  {mt} fold{fold} seed{seed} (exit={r.returncode}) see {log_file}")
            results.append((mt, fold, seed, success, "OK" if success else f"FAIL_EXIT_{r.returncode}"))
        return results

    worker_assignments = defaultdict(list)
    for i, job in enumerate(jobs):
        gpu = GPUS[i % len(GPUS)]
        wid = (i // len(GPUS)) % WORKERS_PER_GPU
        worker_assignments[(gpu, wid)].append(job)

    print("Worker assignments:")
    for (gpu, w), wjobs in sorted(worker_assignments.items()):
        print(f"  GPU{gpu} W{w}: {len(wjobs)} jobs")

    import threading
    all_results = []
    threads = []
    def worker_thread(gpu, w, wjobs):
        results = launch_worker(gpu, w, wjobs)
        all_results.extend(results)

    for i, ((gpu, w), wjobs) in enumerate(sorted(worker_assignments.items())):
        t = threading.Thread(target=worker_thread, args=(gpu, w, wjobs))
        threads.append(t)
        t.start()
        time.sleep(3)

    for t in threads:
        t.join()

    failed = [(mt, f, s, r) for mt, f, s, ok, r in all_results if not ok]
    total = len(all_results)
    passed = total - len(failed)
    print(f"\n{'='*50}")
    print(f"OOF RESULTS: {passed}/{total} passed")
    if failed:
        for mt, f, s, r in failed:
            print(f"  FAIL {mt} fold{f} seed{s}: {r}")
        print("STATUS: HOLD")
        sys.exit(1)
    print("STATUS: ALL PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
