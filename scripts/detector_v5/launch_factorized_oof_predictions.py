#!/usr/bin/env python3
"""Launch held-out OOF predictions for all 24 checkpoints.

Parallel execution across GPUs. Fail-closed.
"""
import subprocess, sys, time, json, hashlib, argparse
from pathlib import Path
from collections import defaultdict

PY = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
PREDICT_SCRIPT = Path(__file__).resolve().parent / "predict_factorized_oof.py"
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
S1 = OPS / "OFFICIAL_V3_S1_FIT_V1_d31187f"
TEACHER = OPS / "OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721"
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
POLICY_INTENT = OPS / "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_01"
TRAINING_OUT = OPS / "OFFICIAL_V3_FACTORIZED_STUDENT_OOF_335048c_20260721"
REGISTRY = OPS / "OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"
DEFAULT_OUT_BASE = OPS / "OFFICIAL_V3_FACTORIZED_STUDENT_OOF_PREDICTIONS_V1_20260721"


def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): d.update(b)
    return d.hexdigest()


def verify_sealed_directory(root):
    s = root / "SHA256SUMS"; c = root / "SHA256SUMS.sha256"
    if not s.is_file() or not c.is_file():
        raise RuntimeError(f"SEAL MISSING: {root}")
    if c.read_text().strip() != f"{sha256_file(s)}  SHA256SUMS":
        raise RuntimeError(f"SEAL MISMATCH: {root}")
    for l in s.read_text().splitlines():
        d, _, n = l.partition("  "); t = root / n
        if not t.is_file() or sha256_file(t) != d:
            raise RuntimeError(f"FILE MISMATCH: {root}/{n}")
    return sha256_file(s)


SEEDS = [42, 123, 456]
FOLD_IDS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 3, 4, 5])
    ap.add_argument("--output-base", type=Path, default=DEFAULT_OUT_BASE)
    args = ap.parse_args()

    GPUS = args.gpus
    WORKERS_PER_GPU = args.workers
    OUT_BASE = args.output_base

    jobs = []
    for mt in ["25D9D", "25D"]:
        for fold in FOLD_IDS:
            for seed in SEEDS:
                ckpt_dir = TRAINING_OUT / mt / f"fold{fold}_seed{seed}"
                out_dir = OUT_BASE / f"predict_{mt}_fold{fold}_seed{seed}"
                jobs.append((mt, fold, seed, ckpt_dir, out_dir))

    print(f"Total jobs: {len(jobs)} | GPUs: {GPUS} | workers/GPU: {WORKERS_PER_GPU}")

    def launch_worker(gpu, worker_id, job_list):
        env = {}
        env.update(__import__("os").environ)
        env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                     "PYTHONPATH": "/mnt/sdc/dty_user/openvla_attack/src",
                     "CUDA_VISIBLE_DEVICES": str(gpu)})
        LOG_DIR = Path("/mnt/sdc/dty_user/openvla_attack/logs/oof_predict")
        results = []
        for mt, fold, seed, ckpt_dir, out_dir in job_list:
            log_file = LOG_DIR / f"predict_{mt}_fold{fold}_seed{seed}_gpu{gpu}_w{worker_id}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            if out_dir.exists():
                try:
                    verify_sealed_directory(out_dir)
                    print(f"  [GPU{gpu} W{worker_id}] SKIP {mt} fold{fold} seed{seed} (sealed)")
                    results.append((mt, fold, seed, True, "SKIP_SEALED"))
                    continue
                except Exception as e:
                    print(f"  [GPU{gpu} W{worker_id}] HOLD {mt} fold{fold} seed{seed}: existing invalid: {e}")
                    results.append((mt, fold, seed, False, str(e)[:80]))
                    continue

            extra_args = []
            if mt == "25D9D":
                extra_args = ["--policy-intent-root", str(POLICY_INTENT)]

            print(f"  [GPU{gpu} W{worker_id}] START {mt} fold{fold} seed{seed}")
            start = time.time()
            with open(log_file, "w") as lf:
                r = subprocess.run(
                    [PY, str(PREDICT_SCRIPT),
                     "--checkpoint-dir", str(ckpt_dir),
                     "--s1-root", str(S1), "--teacher-root", str(TEACHER),
                     "--fold-root", str(FOLD_ROOT), "--registry", str(REGISTRY),
                     "--output-root", str(out_dir), "--gpu", "0"] + extra_args,
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
    print(f"PREDICTION RESULTS: {passed}/{total} passed")
    if failed:
        for mt, f, s, r in failed:
            print(f"  FAIL {mt} fold{f} seed{s}: {r}")
        print("STATUS: HOLD")
        sys.exit(1)
    print("STATUS: ALL PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
