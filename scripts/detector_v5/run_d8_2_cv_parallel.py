"""D8 CV: Safe parallel launcher — single dispatch queue, one job per GPU at a time.

D8-3B protocol: CONFIGS must be ["B3"]. Any other configuration aborts.
Kill switch: touch STOP_D8_3B in output root to prevent new launches + drain running jobs.
"""
from __future__ import annotations

import argparse, json, os, signal, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FOLDS = [0, 1, 2, 3, 4]
GPU_IDS = [0, 1, 2, 3, 6, 7]

# ── D8-3B frozen matrix ──────────────────────────────────────────────
# Changing CONFIGS or FOLDS without gate re-approval invalidates D8-3B.
ALLOWED_D8_3B_CONFIGS = frozenset({"B3"})
ALLOWED_D8_3B_FOLDS = frozenset(FOLDS)


def _check_kill_switch(output_root: Path) -> bool:
    return (output_root / "STOP_D8_3B").exists()


def _build_job_list(configs: list[str], folds: list[int], gpu_ids: list[int], seeds: list[int]):
    """Flat list of (config, fold, seed) — GPU assigned at dispatch time."""
    jobs = []
    for seed in seeds:
        for config in configs:
            for fold in folds:
                jobs.append((config, fold, seed))
    return jobs


def launch_unit(config: str, fold: int, seed: int, gpu: int,
                cache_root: str, output_root: str, python: str, epochs: int) -> subprocess.Popen:
    unit_dir = Path(output_root) / f"seed{seed}" / f"{config}_fold{fold}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    log_path = unit_dir / "train.log"

    cmd = [
        python, "-u", str(ROOT / "scripts" / "detector_v5" / "run_d8_2_cv_unit.py"),
        "--cache-root", cache_root,
        "--config", config,
        "--fold", str(fold),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-dir", str(unit_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{ROOT / 'scripts' / 'detector_v5'}:{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    launched = datetime.now(timezone.utc)
    print(f"  [{config} fold {fold} seed {seed}] GPU {gpu} PID {proc.pid} @ {launched.isoformat(timespec='seconds')}")
    return proc, log_fh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--gpus", type=str, default=",".join(str(g) for g in GPU_IDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (legacy). Use --seeds for multi-seed D8-3B.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. 20260720,20260721,...")
    parser.add_argument("--configs", type=str, default="B3",
                        help="Comma-separated config list (default: B3 for D8-3B)")
    args = parser.parse_args()

    # ── Seed resolution ──
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [20260717]

    # ── Config resolution + D8-3B gate ──
    configs = [c.strip() for c in args.configs.split(",")]

    # Detect D8-3B multi-seed run
    is_d8_3b = len(seeds) > 1

    if is_d8_3b:
        invalid_configs = set(configs) - ALLOWED_D8_3B_CONFIGS
        if invalid_configs:
            raise RuntimeError(
                f"D8-3B requires B3-only execution. "
                f"Got configs={configs}, invalid={sorted(invalid_configs)}. "
                f"ABORTING."
            )
        print(f"D8-3B FROZEN MATRIX: configs={configs} folds={FOLDS} seeds={seeds} epochs={args.epochs}")
        print(f"Gate assertion: CONFIGS subset of {set(ALLOWED_D8_3B_CONFIGS)} — PASSED")

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    cache_root = str(args.cache_root.resolve())
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists — refusing to clobber")
    output_root.mkdir(parents=True)

    # ── Build job queue ──
    job_queue = _build_job_list(configs, FOLDS, gpu_ids, seeds)
    total_jobs = len(job_queue)
    print(f"\nJob queue: {total_jobs} jobs on {len(gpu_ids)} GPUs ({configs} x {len(FOLDS)} folds x {len(seeds)} seeds)")
    if is_d8_3b:
        print(f"Kill switch: touch {output_root / 'STOP_D8_3B'} to drain and stop")

    # ── Single-queue dispatch ──
    # One slot per GPU. A slot holds (proc, log_fh, config, fold, seed, gpu, started_at).
    slots: dict[int, tuple] = {}  # gpu -> slot
    results: dict[str, dict] = {}
    job_idx = 0
    manifest_entries: list[dict] = []

    while job_idx < total_jobs or slots:
        # Drain completed slots
        finished_gpus = []
        for gpu, (proc, fh, cfg, fld, s, _gpu, started) in list(slots.items()):
            ret = proc.poll()
            if ret is not None:
                fh.close()
                finished_at = datetime.now(timezone.utc)
                unit_dir = output_root / f"seed{s}" / f"{cfg}_fold{fld}"
                metrics_path = unit_dir / "metrics.json"
                status = "UNKNOWN"
                metrics = {}
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                    auroc = metrics.get("auroc", float("nan"))
                    bacc = metrics.get("balanced_accuracy", 0)
                    status = "OK" if bacc > 0 else "WARN"
                else:
                    status = f"EXIT_{ret}"
                key = f"seed{s}/{cfg}_fold{fld}"
                results[key] = {"config": cfg, "fold": fld, "seed": s, "gpu": gpu,
                                "exit_code": ret, "status": status, **metrics}
                elapsed = (finished_at - started).total_seconds()
                print(f"  [{cfg} fold {fld} seed {s}] GPU {gpu} {status} "
                      f"(exit {ret}, {elapsed:.0f}s, AUROC={metrics.get('auroc', float('nan')):.4f})")
                manifest_entries.append({
                    "config": cfg, "fold": fld, "seed": s, "gpu": gpu,
                    "pid": proc.pid, "exit_code": ret, "status": status,
                    "started_utc": started.isoformat(), "finished_utc": finished_at.isoformat(),
                    "elapsed_s": elapsed,
                })
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del slots[gpu]

        # Fill empty slots from queue
        while job_idx < total_jobs and len(slots) < len(gpu_ids):
            # Kill switch check
            if is_d8_3b and _check_kill_switch(output_root):
                print(f"\n*** KILL SWITCH ACTIVE — draining {len(slots)} running job(s), "
                      f"{total_jobs - job_idx} pending job(s) skipped ***")
                job_idx = total_jobs  # skip all pending
                break

            available_gpus = [g for g in gpu_ids if g not in slots]
            if not available_gpus:
                break

            gpu = available_gpus[0]
            config, fold, seed = job_queue[job_idx]
            proc, fh = launch_unit(config, fold, seed, gpu, cache_root, str(output_root),
                                   args.python, args.epochs)
            slots[gpu] = (proc, fh, config, fold, seed, gpu, datetime.now(timezone.utc))
            job_idx += 1
            time.sleep(0.5)  # brief stagger for CUDA init

        # Brief sleep to avoid busy-waiting
        if job_idx < total_jobs or slots:
            time.sleep(2)

    # ── Aggregate ──
    completed = [e for e in manifest_entries if e["status"] == "OK"]
    failed = [e for e in manifest_entries if e["status"] != "OK"]
    print(f"\n{'='*60}")
    print(f"COMPLETED: {len(completed)}/{len(manifest_entries)}  FAILED: {len(failed)}/{len(manifest_entries)}")
    if failed:
        for e in failed:
            print(f"  FAIL: seed{e['seed']}/{e['config']}_fold{e['fold']} GPU {e['gpu']} exit {e['exit_code']}")

    # Per-seed aggregate
    seed_summary = {}
    for seed in seeds:
        seed_results = [r for r in manifest_entries if r["seed"] == seed and r["status"] == "OK"]
        if seed_results:
            aurocs = [results[f"seed{seed}/{r['config']}_fold{r['fold']}"].get("auroc", float("nan"))
                      for r in seed_results]
            baccs = [results[f"seed{seed}/{r['config']}_fold{r['fold']}"].get("balanced_accuracy", 0)
                     for r in seed_results]
            seed_summary[str(seed)] = {
                "mean_auroc": float(np.nanmean(auroc_vals)) if (auroc_vals := [a for a in aurocs if not np.isnan(a)]) else float("nan"),
                "mean_bacc": float(np.mean(baccs)) if baccs else 0,
                "folds_completed": len(seed_results),
                "auroc_std_folds": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else float("nan"),
            }

    # D8-3B stability gate
    if is_d8_3b and seed_summary:
        seed_means = [s["mean_auroc"] for s in seed_summary.values()
                      if not np.isnan(s["mean_auroc"])]
        stability_std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
        stability_pass = stability_std <= 0.03 if not np.isnan(stability_std) else False
        print(f"\nD8-3B STABILITY GATE: AUROC std={stability_std:.5f} (n={len(seed_means)} seeds) "
              f"{'PASS' if stability_pass else 'FAIL'} (threshold <=0.03)")
        gate = {"std": stability_std, "n_seeds": len(seed_means), "pass": stability_pass,
                "seed_means": seed_means, "threshold": 0.03}
    else:
        gate = None

    # ── Write manifests ──
    summary = {
        "schema": "D8_CV_SAFE_SUMMARY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configs": configs, "folds": FOLDS, "seeds": seeds, "epochs": args.epochs,
        "gpus": gpu_ids, "dispatch": "single_queue_per_gpu",
        "total_jobs": total_jobs, "completed": len(completed), "failed": len(failed),
        "stability_gate": gate,
        "seed_summary": seed_summary,
    }
    (output_root / "D8_CV_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_root / "JOB_MANIFEST.json").write_text(json.dumps(manifest_entries, indent=2) + "\n")

    print(f"\nManifests written to {output_root}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
