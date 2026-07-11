#!/usr/bin/env python3
"""Run R5 multisuite materialization only after strict R4 provenance validation.

This wrapper is intentionally narrow. It verifies the frozen clean collection,
the mandatory dual-head R4 HOLD-to-PASS lineage, and the four-suite model bytes
before delegating to the existing suite-isolated materializer. It then emits a
sidecar report that binds the resulting dataset to the collection head, audit
head, model manifests, and materializer code bytes.

Use ``--dry-run`` for a CPU/read-only command preview. No OpenVLA model is
loaded and no dataset is written in dry-run mode.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.build_c2g_suite_model_map import sha256_file  # noqa: E402
from scripts.stageb.verify_c2g_suite_model_map_strict import (  # noqa: E402
    verify as verify_models,
)
from tools.multisuite_detector.bind_c2g_r4_dual_head_provenance import (  # noqa: E402
    verify_binding,
)

BASE_MATERIALIZER = (
    REPO
    / "tools"
    / "multisuite_detector"
    / "materialize_c2g_multisuite_dataset.py"
)
BOUND_MATERIALIZER = Path(__file__).resolve()
REPORT_NAME = "c2g_r5_bound_materialization_report.json"
SCHEMA = "c2g.r5.bound_materialization.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R5_BOUND_MATERIALIZATION"
DEFAULT_MIN_FREE_BYTES = 15 * 1024**3


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _assert_external_empty_output(output_dir: Path, input_root: Path) -> None:
    output_dir = output_dir.resolve()
    input_root = input_root.resolve()
    repo = REPO.resolve()
    if output_dir == repo or repo in output_dir.parents:
        raise ValueError("materialization output must be outside the repository")
    if (
        output_dir == input_root
        or input_root in output_dir.parents
        or output_dir in input_root.parents
    ):
        raise ValueError(
            "materialization output must be disjoint from the frozen collection"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("materialization output directory must be empty")


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(
                f"no existing filesystem anchor for output path: {path}"
            )
        candidate = parent
    return candidate


def build_materializer_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(BASE_MATERIALIZER),
        "--input-root",
        str(args.input_root.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--suite-model-map",
        str(args.suite_model_map.resolve()),
        "--window",
        str(args.window),
        "--burst-length",
        str(args.burst_length),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--embedding-dim",
        str(args.embedding_dim),
        "--split-mode",
        args.split_mode,
        "--held-out-task",
        args.held_out_task,
        "--held-out-suite",
        args.held_out_suite,
        "--val-fraction",
        str(args.val_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--positive-weight",
        str(args.positive_weight),
        "--max-episodes-per-suite",
        str(args.max_episodes_per_suite),
        "--git-commit",
        args.audit_head,
    ]


def build_bound_invocation(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BOUND_MATERIALIZER),
        "--input-root",
        str(args.input_root.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--r4-provenance-binding",
        str(args.r4_provenance_binding.resolve()),
        "--audit-head",
        args.audit_head,
        "--suite-model-map",
        str(args.suite_model_map.resolve()),
        "--suite-model-report",
        str(args.suite_model_report.resolve()),
        "--goal-model-manifest",
        str(args.goal_model_manifest.resolve()),
        "--model-verification-report",
        str(args.model_verification_report.resolve()),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--embedding-dim",
        str(args.embedding_dim),
        "--window",
        str(args.window),
        "--burst-length",
        str(args.burst_length),
        "--split-mode",
        args.split_mode,
        "--held-out-task",
        args.held_out_task,
        "--held-out-suite",
        args.held_out_suite,
        "--val-fraction",
        str(args.val_fraction),
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--positive-weight",
        str(args.positive_weight),
        "--max-episodes-per-suite",
        str(args.max_episodes_per_suite),
        "--min-free-bytes",
        str(args.min_free_bytes),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    _assert_external_empty_output(output_dir, input_root)
    if args.window <= 0 or args.burst_length <= 0:
        raise ValueError("window and burst_length must be positive")
    if args.max_episodes_per_suite < 0:
        raise ValueError("max_episodes_per_suite must be non-negative")
    if args.min_free_bytes < 0:
        raise ValueError("min_free_bytes must be non-negative")

    r4 = verify_binding(
        args.r4_provenance_binding.resolve(),
        collection_root=input_root,
        expected_audit_head=args.audit_head,
    )
    model_verification = verify_models(
        args.suite_model_map.resolve(),
        args.suite_model_report.resolve(),
        args.goal_model_manifest.resolve(),
    )
    recorded_verification = _read_json(args.model_verification_report.resolve())
    if (
        recorded_verification.get("status")
        != "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION"
    ):
        raise ValueError("recorded model verification report is not PASS")
    if recorded_verification.get("frozen_report_sha256") != sha256_file(
        args.suite_model_report.resolve()
    ):
        raise ValueError("recorded model verification binds another suite report")
    if recorded_verification.get("suite_models") != model_verification.get(
        "suite_models"
    ):
        raise ValueError("recorded model verification differs from current model bytes")

    output_filesystem_anchor = _nearest_existing_parent(output_dir.parent)
    free_bytes = shutil.disk_usage(output_filesystem_anchor).free
    if free_bytes < args.min_free_bytes:
        raise ValueError(
            f"insufficient free space: {free_bytes} < required {args.min_free_bytes}"
        )

    materializer_command = build_materializer_command(args)
    bound_invocation = build_bound_invocation(args)
    return {
        "gate": "C2G_R5_BOUND_MATERIALIZATION_PREFLIGHT",
        "status": "PASS_C2G_R5_BOUND_MATERIALIZATION_PREFLIGHT",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "output_filesystem_anchor": str(output_filesystem_anchor),
        "collection_head": r4["collection_head"],
        "audit_head": r4["audit_head"],
        "r4_provenance_binding": str(args.r4_provenance_binding.resolve()),
        "r4_provenance_binding_sha256": sha256_file(
            args.r4_provenance_binding.resolve()
        ),
        "suite_model_map": str(args.suite_model_map.resolve()),
        "suite_model_map_sha256": sha256_file(args.suite_model_map.resolve()),
        "suite_model_report": str(args.suite_model_report.resolve()),
        "suite_model_report_sha256": sha256_file(
            args.suite_model_report.resolve()
        ),
        "goal_model_manifest": str(args.goal_model_manifest.resolve()),
        "goal_model_manifest_sha256": sha256_file(
            args.goal_model_manifest.resolve()
        ),
        "model_verification_report": str(
            args.model_verification_report.resolve()
        ),
        "model_verification_report_sha256": sha256_file(
            args.model_verification_report.resolve()
        ),
        # ``command`` is the complete provenance-bound invocation suitable for
        # review/replay. ``materializer_command`` is the internal delegated
        # command executed only after this wrapper's preflight passes.
        "command": bound_invocation,
        "materializer_command": materializer_command,
        "free_bytes": int(free_bytes),
        "minimum_free_bytes": int(args.min_free_bytes),
        "dry_run": bool(args.dry_run),
        "openvla_models_loaded": 0,
        "libero_environments_created": 0,
        "clean_rollouts_launched": 0,
        "attacks_launched": 0,
        "training_epochs": 0,
    }


def _expected_combined_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir = args.output_dir.resolve()
    dataset = output_dir / (
        f"c2g_clean_window_w{args.window:02d}_{args.backend}_{args.split_mode}.npz"
    )
    report = output_dir / "c2g_multisuite_materialization_report.json"
    return dataset, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = preflight(args)
    if args.dry_run:
        return {
            **preflight_report,
            "status": "PASS_C2G_R5_BOUND_MATERIALIZATION_DRY_RUN",
        }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        preflight_report["materializer_command"],
        cwd=REPO,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"base materializer failed with exit code {completed.returncode}"
        )

    dataset_path, base_report_path = _expected_combined_paths(args)
    if not dataset_path.is_file() or not base_report_path.is_file():
        raise FileNotFoundError("combined materialization outputs are missing")
    base_report = _read_json(base_report_path)
    if base_report.get("status") != "PASS_C2G_MULTISUITE_DATASET_MATERIALIZED":
        raise ValueError("base multisuite materialization report is not PASS")
    boundaries = base_report.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("base materialization report lacks boundaries")
    if boundaries.get("clean_only") is not True:
        raise ValueError("base materializer did not attest clean-only inputs")
    if boundaries.get("attack_outcomes_read") is not False:
        raise ValueError("base materializer records attacked-outcome access")
    if boundaries.get("suite_task_identity_used_as_model_feature") is not False:
        raise ValueError("base materializer records suite/task shortcut features")

    post_r4 = verify_binding(
        args.r4_provenance_binding.resolve(),
        collection_root=args.input_root.resolve(),
        expected_audit_head=args.audit_head,
    )
    report = {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "collection_head": post_r4["collection_head"],
        "audit_head": post_r4["audit_head"],
        "r4_provenance_binding": str(args.r4_provenance_binding.resolve()),
        "r4_provenance_binding_sha256": sha256_file(
            args.r4_provenance_binding.resolve()
        ),
        "base_materializer_path": str(BASE_MATERIALIZER),
        "base_materializer_sha256": sha256_file(BASE_MATERIALIZER),
        "bound_wrapper_path": str(BOUND_MATERIALIZER),
        "bound_wrapper_sha256": sha256_file(BOUND_MATERIALIZER),
        "combined_dataset": str(dataset_path),
        "combined_dataset_sha256": sha256_file(dataset_path),
        "base_report": str(base_report_path),
        "base_report_sha256": sha256_file(base_report_path),
        "combined_samples": int(base_report.get("combined_samples", -1)),
        "split_counts": base_report.get("split_counts"),
        "per_suite": base_report.get("per_suite"),
        "command": preflight_report["command"],
        "materializer_command": preflight_report["materializer_command"],
        "output_filesystem_anchor": preflight_report[
            "output_filesystem_anchor"
        ],
        "boundaries": {
            "clean_only": True,
            "attack_outcomes_read": False,
            "counterfactual_read": False,
            "suite_task_identity_used_as_model_feature": False,
            "libero_rollouts_launched": 0,
            "attacks_launched": 0,
            "training_epochs": 0,
        },
    }
    report_path = output_dir / REPORT_NAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r4-provenance-binding", type=Path, required=True)
    parser.add_argument("--audit-head", required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("stats", "clip", "openvla_siglip"),
        default="openvla_siglip",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument(
        "--split-mode",
        choices=("within_task", "leave_one_task_out", "leave_one_suite_out"),
        default="within_task",
    )
    parser.add_argument("--held-out-task", default="")
    parser.add_argument("--held-out-suite", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--max-episodes-per-suite", type=int, default=0)
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_BYTES,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "HOLD_C2G_R5_BOUND_MATERIALIZATION",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
