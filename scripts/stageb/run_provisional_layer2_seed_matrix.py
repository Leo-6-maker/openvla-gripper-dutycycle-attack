#!/usr/bin/env python3
"""Run the provisional Layer2 seed matrix.

Engineering-only wrapper around ``train_provisional_layer2.run_one``. It
creates exactly 24 terminal runs:

- M1 in-domain Spatial/Goal/LIBERO-10 for seeds 1..3.
- M2 leave-one-suite-out Spatial/Goal/LIBERO-10 for seeds 1..5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.train_provisional_layer2 import (  # noqa: E402
    PROVISIONAL_SENTINEL,
    SUITES,
    read_csv,
    run_one,
    sha256_file,
    write_csv,
    write_json,
)


def run_seed_matrix(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PROVISIONAL_SENTINEL).write_text(
        "Provisional Layer2 seed matrix output. Not final paper evidence.\n",
        encoding="utf-8",
    )
    dataset_path = Path(args.dataset)
    dataset_rows = read_csv(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    summaries: list[dict[str, Any]] = []

    for suite in SUITES:
        for seed in args.m1_seeds:
            summaries.append(
                run_one(
                    name=f"M1_in_domain_{suite}_seed{seed}",
                    dataset_rows=dataset_rows,
                    train_suites={suite},
                    val_suites={suite},
                    test_suites={suite},
                    output_dir=output_dir,
                    seed=seed,
                    device=args.device,
                    epochs=args.epochs,
                    dataset_sha=dataset_sha,
                )
            )

    for heldout in SUITES:
        source = set(SUITES) - {heldout}
        for seed in args.m2_seeds:
            summaries.append(
                run_one(
                    name=f"M2_leave_one_suite_out_test_{heldout}_seed{seed}",
                    dataset_rows=dataset_rows,
                    train_suites=source,
                    val_suites=source,
                    test_suites={heldout},
                    output_dir=output_dir,
                    seed=seed,
                    device=args.device,
                    epochs=args.epochs,
                    dataset_sha=dataset_sha,
                )
            )

    metrics_rows = [
        {
            "run_name": r["run_name"],
            "run_status": r.get("run_status", ""),
            "skip_reason": r.get("skip_reason", ""),
            "n_train_rows": r.get("n_train_rows", ""),
            "n_val_rows": r.get("n_val_rows", ""),
            "n_test_rows": r.get("n_test_rows", ""),
            "checkpoint_sha256": r.get("checkpoint_sha256", ""),
            "tau_corridor": r.get("selected_threshold", {}).get("tau_corridor", ""),
            "tau_release": r.get("selected_threshold", {}).get("tau_release", ""),
            "val_event_f1": r.get("selected_threshold", {}).get("event_f1", ""),
            "val_false_trigger_episode_rate": r.get("selected_threshold", {}).get("false_trigger_episode_rate", ""),
            "val_median_latency": r.get("selected_threshold", {}).get("median_latency", ""),
            "test_event_f1": r.get("test_metrics", {}).get("event_f1", ""),
            "test_event_precision": r.get("test_metrics", {}).get("event_precision", ""),
            "test_event_recall": r.get("test_metrics", {}).get("event_recall", ""),
            "test_false_trigger_episode_rate": r.get("test_metrics", {}).get("false_trigger_episode_rate", ""),
            "test_no_emit_rate": r.get("test_metrics", {}).get("no_emit_rate", ""),
            "test_frame_auroc": r.get("test_metrics", {}).get("frame_auroc", ""),
            "test_frame_auprc": r.get("test_metrics", {}).get("frame_auprc", ""),
        }
        for r in summaries
    ]
    write_csv(output_dir / "provisional_layer2_seed_matrix_metrics.csv", metrics_rows)

    deployment = []
    for heldout in SUITES:
        candidates = [
            r
            for r in summaries
            if r["run_name"].startswith(f"M2_leave_one_suite_out_test_{heldout}_")
            and r.get("run_status") == "COMPLETED"
        ]
        candidates.sort(
            key=lambda r: (
                -float(r.get("selected_threshold", {}).get("event_f1", -1.0)),
                float(r.get("selected_threshold", {}).get("false_trigger_episode_rate", 1e9)),
                float(r.get("selected_threshold", {}).get("median_latency", 1e9)),
                int(str(r["run_name"]).rsplit("seed", 1)[-1]),
            )
        )
        best = candidates[0] if candidates else {}
        deployment.append(
            {
                "heldout_suite": heldout,
                "selected_run": best.get("run_name", ""),
                "checkpoint_path": best.get("checkpoint_path", ""),
                "checkpoint_sha256": best.get("checkpoint_sha256", ""),
                "val_event_f1": best.get("selected_threshold", {}).get("event_f1", ""),
                "val_false_trigger_episode_rate": best.get("selected_threshold", {}).get("false_trigger_episode_rate", ""),
                "val_median_latency": best.get("selected_threshold", {}).get("median_latency", ""),
            }
        )
    write_csv(output_dir / "m2_deployment_checkpoint_selection.csv", deployment)

    report = {
        "provisional_engineering_only": True,
        "official_h2_status": "NOT_GRANTED",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "device": args.device,
        "epochs": args.epochs,
        "m1_seeds": args.m1_seeds,
        "m2_seeds": args.m2_seeds,
        "planned_runs": 24,
        "completed_runs": sum(1 for row in summaries if row.get("run_status") == "COMPLETED"),
        "skipped_runs": sum(1 for row in summaries if row.get("run_status") == "SKIPPED_NO_SUPERVISED_ROWS"),
        "runs": summaries,
        "deployment_checkpoints": deployment,
    }
    write_json(output_dir / "provisional_layer2_seed_matrix_summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--m1-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--m2-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return ap.parse_args()


def main() -> None:
    run_seed_matrix(parse_args())


if __name__ == "__main__":
    main()
