#!/usr/bin/env python3
"""Launch the 12-split recommended V2B exact-W32 engineering sidecar.

No process-killing or restart behavior is present. Default concurrency is one so
this can run beside the saturated 864-job grid without destabilizing it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts/detector_v5"
DEFAULT_SPLITS = Path(
    "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/"
    "c2g_cs200_official_v3_20260716/ops/"
    "OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721"
)


def run_command(command: list[str], log_path: Path, timeout: int, env: dict) -> tuple[bool, float]:
    started = time.time()
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=False,
            )
            return result.returncode == 0, time.time() - started
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT\n")
            return False, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--inner-cv-splits-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--reference-authorization-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    output_base = args.output_base.resolve()
    if "RECOMMENDED" not in output_base.name.upper():
        raise SystemExit("output-base name must contain RECOMMENDED")
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir = output_base / "_logs"
    log_dir.mkdir(exist_ok=True)

    if args.workers < 1 or args.workers > len(args.gpus):
        raise SystemExit("workers must be between 1 and number of supplied GPUs")

    jobs = []
    for outer in range(4):
        for inner in range(3):
            label = f"V2B_EXACT_W32_H64_D0.1_WD1e-4_o{outer}_i{inner}_s42"
            jobs.append({"label": label, "outer": outer, "inner": inner})

    base_env = os.environ.copy()
    base_env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    def process(job: dict, gpu: int) -> tuple[str, bool, str]:
        label = job["label"]
        train_dir = output_base / label
        pred_dir = output_base / f"predict_{label}"
        eval_file = output_base / f"eval_{label}.json"
        audit_file = output_base / f"audit_{label}.json"

        if audit_file.is_file():
            try:
                audit = json.loads(audit_file.read_text())
                if audit.get("status") == "PASS":
                    return label, True, "SKIP_COMPLETE"
            except Exception:
                pass

        train_cmd = [
            str(args.python),
            str(SCRIPTS / "train_factorized_v2_recommended_canary.py"),
            "--outer-fold", str(job["outer"]),
            "--inner-fold", str(job["inner"]),
            "--seed", "42",
            "--gpu", str(gpu),
            "--output-root", str(train_dir),
            "--inner-cv-splits-root", str(args.inner_cv_splits_root.resolve()),
        ]
        if args.reference_authorization_root is not None:
            train_cmd.extend([
                "--reference-authorization-root",
                str(args.reference_authorization_root.resolve()),
            ])
        ok, elapsed = run_command(
            train_cmd, log_dir / f"{label}.train.log", args.timeout, base_env
        )
        if not ok:
            return label, False, f"TRAIN_FAIL_{elapsed:.0f}s"

        predict_cmd = [
            str(args.python),
            str(SCRIPTS / "predict_factorized_v2_recommended_canary.py"),
            "--checkpoint-dir", str(train_dir),
            "--inner-cv-splits-root", str(args.inner_cv_splits_root.resolve()),
            "--output-root", str(pred_dir),
            "--gpu", str(gpu),
        ]
        ok, _ = run_command(
            predict_cmd, log_dir / f"{label}.predict.log", args.timeout, base_env
        )
        if not ok:
            return label, False, "PREDICT_FAIL"

        eval_cmd = [
            str(args.python),
            str(SCRIPTS / "evaluate_factorized_v2_inner_cv.py"),
            "--predictions-base", str(output_base),
            "--prediction-dir", str(pred_dir),
            "--output", str(eval_file),
            "--mode", "single",
        ]
        ok, _ = run_command(
            eval_cmd, log_dir / f"{label}.evaluate.log", args.timeout, base_env
        )
        if not ok:
            return label, False, "EVAL_FAIL"

        audit_cmd = [
            str(args.python),
            str(SCRIPTS / "audit_factorized_v2_inner_cv_predictions.py"),
            "--prediction-dir", str(pred_dir),
            "--inner-cv-splits-root", str(args.inner_cv_splits_root.resolve()),
            "--output", str(audit_file),
        ]
        ok, _ = run_command(
            audit_cmd, log_dir / f"{label}.audit.log", args.timeout, base_env
        )
        if not ok:
            return label, False, "AUDIT_FAIL"
        return label, True, f"COMPLETE_{elapsed:.0f}s"

    assignments = [(job, args.gpus[i % len(args.gpus)]) for i, job in enumerate(jobs)]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, job, gpu): job["label"]
            for job, gpu in assignments
        }
        for future in as_completed(futures):
            label, ok, message = future.result()
            results.append({"label": label, "ok": ok, "message": message})
            print(f"{'PASS' if ok else 'FAIL'} {label}: {message}", flush=True)

    summary = {
        "status": "PASS" if all(item["ok"] for item in results) else "HOLD",
        "formal_selection_eligible": False,
        "config": {
            "candidate": "V2B_RECOMMENDED_EXACT",
            "context_steps": 32,
            "hidden_dim": 64,
            "dropout": 0.1,
            "weight_decay": 1e-4,
            "seed": 42,
        },
        "results": sorted(results, key=lambda item: item["label"]),
    }
    (output_base / "sidecar_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
