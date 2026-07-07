#!/usr/bin/env python3
"""Prepare a machine-readable C4 execution plan for Codex/server runs.

This tool only writes a plan. It does not train detectors, run OpenVLA/LIBERO,
perform rollouts, attacks, exact-prefix replay, victim inference, or use GPU.
The generated commands are intentionally explicit so Codex can execute the next
server steps without re-designing the protocol.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

DEFAULTS = {
    "dataset_root": "/mnt/sdc/dty_user/openvla_attack_evidence/detector_dataset/formal_detector_dataset_d03560a",
    "feature_csv": "/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1/CLEAN2000_FEATURES_25D_ALL_STEPS.csv",
    "c4_1_root": "/mnt/sdc/dty_user/openvla_attack_evidence/detector_training/c4_balanced_parent_random_d03560a",
    "c4_2_root": "/mnt/sdc/dty_user/openvla_attack_evidence/detector_training/c4_2_bundle_audit_d03560a",
    "c4_3_split_root": "/mnt/sdc/dty_user/openvla_attack_evidence/detector_dataset/c4_scientific_splits_d03560a",
    "c4_freeze_root": "/mnt/sdc/dty_user/openvla_attack_evidence/detector_training/c4_3c_detector_freeze_d03560a",
    "dataset_csv_sha256": "f7808c4ef2a74887689804758c131a19a7fecbbc0e5400bcc3322d08c796010a",
    "split_csv_sha256": "df23607b3791e414d0e07900508c095bda6a190e8f6500502b056f0988e02673",
    "state_index_sha256": "e4fafbb01e70418ec04b7dc19294b1f6b9c0b52ecc0d8aaa5b56997c3ba53691",
    "checkpoint_sha256": "5747a9c967b5b08f0e4b8fc8ba0cbf47c13533ffb5e347c38470e84efe17d79b",
    "threshold": "0.95",
    "seed": "2026070401",
    "val_ratio": "0.15",
}

NON_ACTIONS = [
    "OpenVLA",
    "LIBERO",
    "rollout",
    "attack",
    "exact_prefix_replay",
    "victim_inference",
    "paper_main_table",
]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def command(parts: list[str]) -> str:
    return " ".join(q(p) for p in parts)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    dataset_csv = dataset_root / "detector_dataset_manifest_v1.csv"
    feature_csv = Path(args.feature_csv)
    c4_2_root = Path(args.c4_2_root)
    c4_3_split_root = Path(args.c4_3_split_root)
    freeze_root = Path(args.c4_freeze_root)
    object_split = c4_3_split_root / "object_task_heldout_with_val_v1.csv"
    suite_split = c4_3_split_root / "suite_loso_with_val_v1.csv"
    object_split_report = c4_3_split_root / "object_task_heldout_validator_report.json"
    suite_split_report = c4_3_split_root / "suite_loso_validator_report.json"
    c4_2_validator_report = c4_2_root / "validator_report.json"
    freeze_validator_report = freeze_root / "freeze_validator_report.json"

    validate_c4_2 = command([
        "python",
        "tools/multisuite_detector/validate_c4_bundle_audit_v1.py",
        "--audit-root", str(c4_2_root),
        "--expected-checkpoint-sha256", args.checkpoint_sha256,
        "--expected-dataset-csv-sha256", args.dataset_csv_sha256,
        "--expected-split-csv-sha256", args.split_csv_sha256,
        "--expected-state-index-sha256", args.state_index_sha256,
        "--expected-threshold", args.threshold,
        "--output-json", str(c4_2_validator_report),
    ])

    build_object = command([
        "python",
        "tools/multisuite_detector/build_c4_scientific_splits_v1.py",
        "build-object-task-heldout",
        "--dataset-csv", str(dataset_csv),
        "--output", str(object_split),
        "--seed", args.seed,
        "--val-ratio", args.val_ratio,
    ])
    validate_object = command([
        "python",
        "tools/multisuite_detector/build_c4_scientific_splits_v1.py",
        "validate",
        "--dataset-csv", str(dataset_csv),
        "--split-csv", str(object_split),
        "--output-json", str(object_split_report),
    ])
    build_suite = command([
        "python",
        "tools/multisuite_detector/build_c4_scientific_splits_v1.py",
        "build-suite-loso",
        "--dataset-csv", str(dataset_csv),
        "--output", str(suite_split),
        "--seed", args.seed,
        "--val-ratio", args.val_ratio,
    ])
    validate_suite = command([
        "python",
        "tools/multisuite_detector/build_c4_scientific_splits_v1.py",
        "validate",
        "--dataset-csv", str(dataset_csv),
        "--split-csv", str(suite_split),
        "--output-json", str(suite_split_report),
    ])

    freeze_validate = command([
        "python",
        "tools/multisuite_detector/validate_c4_detector_freeze_v1.py",
        "--freeze-root", str(freeze_root),
        "--expected-checkpoint-sha256", args.checkpoint_sha256,
        "--expected-dataset-csv-sha256", args.dataset_csv_sha256,
        "--expected-split-csv-sha256", "<SCIENTIFIC_SPLIT_SHA256_FROM_FREEZE_BUNDLE>",
        "--expected-state-index-sha256", args.state_index_sha256,
        "--expected-threshold", "<VALIDATION_SELECTED_THRESHOLD_FROM_FREEZE_BUNDLE>",
        "--output-json", str(freeze_validator_report),
    ])

    plan = {
        "schema_version": "c4_codex_execution_plan_v1",
        "status": "PLAN_ONLY",
        "purpose": "Direct-use C4 server execution checklist for Codex after quota recovery.",
        "repo_requirements": {
            "branch": "plan/codex-gated-experiment-v1",
            "minimum_head": "b99b16e36a6c8cfa24052400837104cbe90b225e",
            "required_ci": "cpu-stageb success",
        },
        "identities": {
            "dataset_root": str(dataset_root),
            "dataset_csv": str(dataset_csv),
            "feature_csv": str(feature_csv),
            "c4_1_root": args.c4_1_root,
            "dataset_csv_sha256": args.dataset_csv_sha256,
            "parent_random_split_csv_sha256": args.split_csv_sha256,
            "state_index_sha256": args.state_index_sha256,
            "c4_1_checkpoint_sha256": args.checkpoint_sha256,
            "c4_1_threshold": args.threshold,
        },
        "global_non_actions": {name: "NOT_PERFORMED" for name in NON_ACTIONS},
        "steps": [
            {
                "id": "C4_2_BUNDLE_AUDIT_VALIDATE",
                "description": "Validate the C4-2 bundle-audit evidence once the server audit files exist.",
                "output_root": str(c4_2_root),
                "commands": [f"mkdir -p {q(c4_2_root)}", validate_c4_2],
                "pass_status": "C4_2_DETECTOR_BUNDLE_AUDIT = PASS",
            },
            {
                "id": "C4_3A_OBJECT_TASK_HELDOUT_SPLIT_BUILD",
                "description": "Build and validate Object task-held-out split with validation folds.",
                "output_root": str(c4_3_split_root),
                "commands": [f"mkdir -p {q(c4_3_split_root)}", build_object, validate_object],
                "pass_status": "C4_3A_OBJECT_TASK_HELDOUT_SPLIT = PASS",
            },
            {
                "id": "C4_3B_SUITE_LOSO_SPLIT_BUILD",
                "description": "Build and validate suite LOSO split with validation folds.",
                "output_root": str(c4_3_split_root),
                "commands": [f"mkdir -p {q(c4_3_split_root)}", build_suite, validate_suite],
                "pass_status": "C4_3B_SUITE_LOSO_SPLIT = PASS",
            },
            {
                "id": "C4_3C_DETECTOR_FREEZE_VALIDATE",
                "description": "Validate a scientific detector freeze bundle before C5 replay. Fill placeholders from the freeze bundle.",
                "output_root": str(freeze_root),
                "commands": [freeze_validate],
                "pass_status": "C4_3C_DETECTOR_FREEZE = PASS",
                "blocking_note": "Do not run C5 replay until this validator passes without --allow-parent-random-candidate.",
            },
        ],
        "server_execution_order": [
            "C4_2_BUNDLE_AUDIT_VALIDATE",
            "C4_3A_OBJECT_TASK_HELDOUT_SPLIT_BUILD",
            "C4_3B_SUITE_LOSO_SPLIT_BUILD",
            "scientific detector training on generated splits",
            "C4_3C_DETECTOR_FREEZE_VALIDATE",
            "C5 detector-only exact-prefix replay",
        ],
        "final_guardrail": "This plan does not authorize OpenVLA, LIBERO, rollout, attack, exact-prefix replay, victim inference, or paper main-table claims.",
    }
    return plan


def write_bash(plan: dict[str, Any], output_bash: str | Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by prepare_c4_codex_execution_plan_v1.py",
        "# This script only runs CPU-side validators/split builders listed in the plan.",
        "",
    ]
    for step in plan["steps"]:
        lines.append(f"echo '### {step['id']}'")
        for cmd in step["commands"]:
            lines.append(cmd)
        lines.append("")
    out = Path(output_bash)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULTS["dataset_root"])
    parser.add_argument("--feature-csv", default=DEFAULTS["feature_csv"])
    parser.add_argument("--c4-1-root", default=DEFAULTS["c4_1_root"])
    parser.add_argument("--c4-2-root", default=DEFAULTS["c4_2_root"])
    parser.add_argument("--c4-3-split-root", default=DEFAULTS["c4_3_split_root"])
    parser.add_argument("--c4-freeze-root", default=DEFAULTS["c4_freeze_root"])
    parser.add_argument("--dataset-csv-sha256", default=DEFAULTS["dataset_csv_sha256"])
    parser.add_argument("--split-csv-sha256", default=DEFAULTS["split_csv_sha256"])
    parser.add_argument("--state-index-sha256", default=DEFAULTS["state_index_sha256"])
    parser.add_argument("--checkpoint-sha256", default=DEFAULTS["checkpoint_sha256"])
    parser.add_argument("--threshold", default=DEFAULTS["threshold"])
    parser.add_argument("--seed", default=DEFAULTS["seed"])
    parser.add_argument("--val-ratio", default=DEFAULTS["val_ratio"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-bash")
    args = parser.parse_args(argv)
    plan = build_plan(args)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_bash:
        write_bash(plan, args.output_bash)
    print(json.dumps({"status": "PASS", "output_json": str(out), "step_count": len(plan["steps"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
