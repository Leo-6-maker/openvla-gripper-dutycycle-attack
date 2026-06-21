#!/usr/bin/env python3
"""Build the frozen job manifest for provisional two-suite Layer3 smoke."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITIONS = ("CLEAN", "VIS", "RAND", "SHUFFLED")
PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"
ROOT = "/data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621"

SUITE_CONFIG = {
    "libero_spatial": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial",
        "unnorm_key": "libero_spatial",
        "detector_path": "/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v3/M2_leave_one_suite_out_test_libero_spatial/model.pt",
        "assigned_worker": "PAIR_A",
        "cuda_visible_devices": "1,3",
        "render_gpu": "3",
    },
    "libero_goal": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal",
        "unnorm_key": "libero_goal",
        "detector_path": "/data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v3/M2_leave_one_suite_out_test_libero_goal/model.pt",
        "assigned_worker": "PAIR_B",
        "cuda_visible_devices": "5,4",
        "render_gpu": "4",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_key(key: str) -> str:
    return key.replace("|", "_").replace("/", "_")


def build_jobs(parents: list[dict[str, str]], *, attack_seed: int, output_root: str = ROOT) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for parent in parents:
        suite = parent["suite"]
        cfg = SUITE_CONFIG[suite]
        parent_key = safe_key(parent["canonical_episode_key"])
        for condition in CONDITIONS:
            job_id = f"{parent_key}_{condition.lower()}"
            jobs.append(
                {
                    "job_id": job_id,
                    "parent_key": parent["canonical_episode_key"],
                    "review_id": parent["review_id"],
                    "suite": suite,
                    "task_idx": parent["task_idx"],
                    "state_id": parent["state_id"],
                    "teacher_anchor": parent["teacher_anchor"],
                    "teacher_window_start": parent["teacher_window_start"],
                    "teacher_window_end": parent["teacher_window_end"],
                    "condition": condition,
                    "attack_seed": str(attack_seed),
                    "model_path": cfg["model_path"],
                    "unnorm_key": cfg["unnorm_key"],
                    "detector_path": cfg["detector_path"],
                    "expected_detector_sha256": parent["detector_checkpoint_sha256"],
                    "assigned_worker": cfg["assigned_worker"],
                    "cuda_visible_devices": cfg["cuda_visible_devices"],
                    "render_gpu": cfg["render_gpu"],
                    "output_dir": f"{output_root}/{cfg['assigned_worker']}/{job_id}",
                    "status": "PLANNED",
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--attack-seed", type=int, default=81)
    ap.add_argument("--output-root", default=ROOT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / PROVISIONAL_SENTINEL).write_text("Provisional two-suite Layer3 job plan. Not final paper evidence.\n", encoding="utf-8")
    parents = read_csv(Path(args.parent_manifest))
    jobs = build_jobs(parents, attack_seed=args.attack_seed, output_root=args.output_root)
    write_csv(out / "provisional_layer3_two_suite_job_manifest.csv", jobs)
    write_json(
        out / "provisional_layer3_two_suite_job_manifest_audit.json",
        {
            "status": "PASS" if len(jobs) == 16 else "FAIL",
            "provisional_engineering_only": True,
            "parent_count": len(parents),
            "condition_count": len(CONDITIONS),
            "planned_job_count": len(jobs),
            "attack_seed": args.attack_seed,
            "output_root": args.output_root,
            "conditions": list(CONDITIONS),
            "pair_a": "CUDA_VISIBLE_DEVICES=1,3 render_gpu=3",
            "pair_b": "CUDA_VISIBLE_DEVICES=5,4 render_gpu=4",
        },
    )


if __name__ == "__main__":
    main()

