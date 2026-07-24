#!/usr/bin/env python3
"""C2e3: 25D-only baseline freezing and limitation package.

Freezes C2e2D-R GRU as reproducible baseline, documents all 25D model
comparisons, and records known limitations (FP borderline, L10 weak,
C2f blocked by missing observation artifacts).

No training. No OpenVLA/LIBERO/env.step.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, shutil, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: r.get(k,"") for k in fields})

def read_report_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}

# All 25D model results (from completed experiments)
ALL_25D_MODELS = {
    "Pooling MLP": {
        "source": "d4c2e2_temporal_pooling_mlp_v2",
        "recall": 0.812, "fp_rate": 0.407, "f1": 0.618,
        "l10_recall": 0.557, "l10_fp": 0.450,
        "object_fp": 1.000, "spatial_fp": 0.583,
        "status": "HOLD", "gate": "FP>30%",
    },
    "GRU (C2e2D report)": {
        "source": "d4c2e2d_sequence_model",
        "recall": 0.751, "fp_rate": 0.298, "f1": 0.639,
        "l10_recall": 0.468, "l10_fp": 0.388,
        "object_fp": 0.125, "spatial_fp": 0.306,
        "status": "PASS_BUT_ARTIFACT_INCONSISTENT", "gate": "worker/main mismatch",
    },
    "GRU artifact-frozen (C2e2D-R)": {
        "source": "d4c2e2d_r_rerun",
        "recall": 0.756, "fp_rate": 0.318, "f1": 0.631,
        "l10_recall": 0.456, "l10_fp": 0.374,
        "object_fp": 0.125, "spatial_fp": 0.417,
        "status": "BORDERLINE_BASELINE", "gate": "FP=31.8%>30%",
    },
    "Causal TCN": {
        "source": "d4c2e2h_causal_tcn",
        "recall": 0.802, "fp_rate": 0.386, "f1": 0.622,
        "l10_recall": 0.544, "l10_fp": 0.467,
        "object_fp": 0.250, "spatial_fp": 0.389,
        "status": "HOLD", "gate": "FP>30%",
    },
    "FP-aware GRU": {
        "source": "d4c2e2k_fp_aware_gru",
        "recall": 0.772, "fp_rate": 0.323, "f1": 0.637,
        "l10_recall": 0.494, "l10_fp": 0.394,
        "object_fp": 0.125, "spatial_fp": 0.278,
        "status": "HOLD", "gate": "FP>30%",
    },
    "Multi-window GRU": {
        "source": "d4c2e2j_multiwindow_gru",
        "recall": 0.787, "fp_rate": 0.326, "f1": 0.644,
        "l10_recall": 0.481, "l10_fp": 0.381,
        "object_fp": 0.250, "spatial_fp": 0.472,
        "status": "HOLD", "gate": "FP>30%",
    },
}

KNOWN_LIMITATIONS = [
    {"limitation": "STRICT_FP_GATE_MISSED",
     "detail": "Artifact-consistent GRU baseline FP=31.8%, exceeds 30% gate by 1.8pp"},
    {"limitation": "LIBERO_10_PRIMARY_RECALL_WEAK",
     "detail": "All 25D models have L10 recall <55%; GRU baseline 45.6%"},
    {"limitation": "C2F_BLOCKED_NO_OBSERVATION_ARTIFACTS",
     "detail": "Current clean rollout artifacts contain no RGB frames, camera images, or task language. C2f0 HOLD."},
    {"limitation": "ENSEMBLE_NO_IMPROVEMENT",
     "detail": "GRU/TCN score fusion cannot separate TCN recall gain from FP noise (C2e2I)"},
    {"limitation": "TASK_CONDITIONING_NOT_PRIORITY",
     "detail": "L10 errors dispersed across 79 tasks; task-specific calibration insufficient (C2e2F)"},
]

C2E_PIPELINE_EVIDENCE = [
    {"finding": "L10 errors dispersed across 79 tasks", "source": "C2e2F"},
    {"finding": "GRU/TCN ensemble cannot fuse recall+FP", "source": "C2e2I"},
    {"finding": "Multi-window GRU: best recall, Spatial FP degrades", "source": "C2e2J"},
    {"finding": "FP-aware loss helps Spatial but not L10", "source": "C2e2K"},
    {"finding": "TCN proves longer temporal context helps L10 recall", "source": "C2e2H"},
    {"finding": "Task errors not concentrated in few tasks", "source": "C2e2F"},
    {"finding": "No RGB/language in clean rollout artifacts", "source": "C2f0"},
    {"finding": "All 25D models converge to FP floor ~31-39%", "source": "C2e2B-H-K-J"},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2e2d-r-root", required=True)
    ap.add_argument("--c2e1-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)

    c2e2dr = Path(args.c2e2d_r_root); c2e1 = Path(args.c2e1_root)

    # Copy frozen model
    src_model = c2e2dr / "c2e2k_selected_model.pt"
    dst_model = out / "c2e3_selected_baseline_model.pt"
    shutil.copy2(src_model, dst_model)

    # Copy config
    src_cfg = c2e2dr / "c2e2k_selected_model_config.json"
    dst_cfg = out / "c2e3_selected_baseline_config.json"
    shutil.copy2(src_cfg, dst_cfg)

    # Copy normalization stats
    norm_path = c2e1 / "c2e1_w16_normalization_stats_train_only.json"
    dst_norm = out / "c2e3_normalization_stats_train_only.json"
    shutil.copy2(norm_path, dst_norm)

    # Copy predictions and metrics from C2e2D-R
    for fn in ["c2e2k_selected_test_predictions.csv", "c2e2k_selected_test_metrics_by_suite.csv"]:
        sp = c2e2dr / fn
        if sp.exists():
            shutil.copy2(sp, out / fn.replace("c2e2k_", "c2e3_"))

    # Copy all config metrics
    sp = c2e2dr / "c2e2k_all_config_metrics.csv"
    if sp.exists():
        shutil.copy2(sp, out / "c2e3_rerun_config_metrics.csv")

    # All 25D model comparison table
    comp_rows = []
    for name, m in ALL_25D_MODELS.items():
        comp_rows.append({"model": name, **m})
    write_csv(out / "c2e3_all_25d_model_comparison.csv", comp_rows,
              ["model", "source", "recall", "fp_rate", "f1", "l10_recall", "l10_fp",
               "object_fp", "spatial_fp", "status", "gate"])

    # Limitations
    write_csv(out / "c2e3_limitation_summary.csv", KNOWN_LIMITATIONS,
              ["limitation", "detail"])

    # Evidence summary
    write_csv(out / "c2e3_pipeline_evidence_summary.csv", C2E_PIPELINE_EVIDENCE,
              ["finding", "source"])

    # C2f0 blocker summary
    write_json(out / "c2e3_c2f0_observation_blocker_summary.json", {
        "status": "C2F0_HOLD_NO_OBSERVATION_ARTIFACTS",
        "finding": "Current clean rollout temporal CSVs contain only 52 proprio/action/derived columns. No RGB frames, camera images, or task language text found in evidence directories.",
        "temporal_csv_columns": 52,
        "image_files_found": 0,
        "language_fields_found": 0,
        "privileged_files_present": "privileged_step_records.jsonl (contains object_pose, target_distance — FORBIDDEN for detector)",
        "recommendation": "Re-collect clean rollouts with RGB frame saving enabled, or use existing observation-rich rollout artifacts from D5/D6 period if available",
    })

    # Violations
    violations = [
        {"violation": "STRICT_FP_GATE_MISSED", "detail": "overall_fp=0.318 > 0.300", "severity": "WARNING"},
        {"violation": "L10_RECALL_WEAK", "detail": "libero_10_recall=0.456 < 0.500", "severity": "WARNING"},
        {"violation": "C2F_BLOCKED", "detail": "no observation features in artifacts", "severity": "INFO"},
    ]
    write_csv(out / "c2e3_violations.csv", violations, ["violation", "detail", "severity"])

    # Checksums
    checks = []
    for fn in sorted(out.glob("*")):
        if fn.is_file() and fn.name != "checksum_report.json":
            checks.append({"path": fn.name, "sha256": sha256_file(str(fn)), "bytes": fn.stat().st_size})
    write_json(out / "checksum_report.json", {"files": checks})
    with open(out / "SHA256SUMS", "w") as f:
        for c in checks:
            f.write(f"{c['sha256']}  {c['path']}\n")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(str(out/'SHA256SUMS'))}  SHA256SUMS\n")

    # Main report
    report = {
        "gate": "C2E3_25D_BASELINE_FREEZING_AND_LIMITATION_PACKAGE",
        "status": "PASS_BASELINE_PACKAGE_WITH_KNOWN_FP_LIMITATION",
        "reason": "Artifact-consistent GRU baseline frozen; FP=31.8% borderline vs 30% gate; all limitations documented",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "selected_baseline": {
            "model": "GRU", "window": 16, "hidden": 128, "lr": 0.001, "dropout": 0.1, "seed": 2,
            "recall": 0.756, "fp_rate": 0.318, "f1": 0.631,
            "tau_emit": 0.33, "tau_suppress": 0.67,
            "artifact_consistency": "PASS",
        },
        "model_comparison": {name: {"recall": m["recall"], "fp_rate": m["fp_rate"]} for name, m in ALL_25D_MODELS.items()},
        "limitations": [l["limitation"] for l in KNOWN_LIMITATIONS],
        "narrative": (
            "25D proprio/action temporal detector is reproducible and informative, "
            "but reaches a false-positive floor on LIBERO-10. "
            "Artifact-consistent GRU baseline achieves recall 75.6% with FP 31.8% "
            "(borderline vs 30% gate). All 25D models converge to FP 31-39%. "
            "GRU/TCN ensemble, FP-aware loss, and multi-window GRU do not resolve "
            "the L10 FP limitation. C2f observation features blocked by missing RGB "
            "in current clean rollout artifacts. This motivates observation-enhanced detectors."
        ),
        "next_steps": [
            "C2f-data-collection: re-collect clean rollouts with RGB frames and task language",
            "C2f1: frozen visual/language embedding materialization",
            "C2f2: 25D temporal + observation detector training",
            "D6C-v3: dense replay audit with observation-enhanced detector (only after C2f2 PASS)",
        ],
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED", "device": "cpu",
            "OpenVLA_model": "NOT_LOADED", "LIBERO_runtime": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED", "attack": "NOT_PERFORMED",
        },
    }
    write_json(out / "c2e3_25d_baseline_package_report.json", report)

    print(json.dumps({"status": report["status"], "baseline_recall": 0.756, "baseline_fp": 0.318,
        "limitations": len(KNOWN_LIMITATIONS), "frozen_model": str(dst_model)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
