"""Materialize R9P OGS-1500 episodes into per-episode full-sequence NPZ files.

Reads R8Z derived episode data (step_records_prefix.jsonl, teacher_v2_labels.jsonl,
derived_episode_metadata.json) and writes per-episode NPZ files containing full
variable-length sequences. Supports --smoke mode for 24-episode validation before
full 900-episode materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
    R9P_MODEL_TARGET_MAP,
    TARGET_SUITES,
)
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

SCHEMA = "c2g.r9p.materialization.2026-07-12.v1"
SMOKE_SALT = "C2G_R9P_SMOKE"
SMOKE_PER_SUITE = 8

GATE_PASS_SMOKE = "PASS_C2G_R9P_MATERIALIZATION_SMOKE"
GATE_PASS_FULL = "PASS_C2G_R9P_TRAINONLY_MATERIALIZATION"

FORBIDDEN_NPZ_KEYS = frozenset({
    "object_pose", "target_pose", "object_target_distance",
    "contact_pairs", "teacher_phase", "teacher_reason_code",
    "resolved_target_objects", "resolved_target_manipulable_entities",
    "attack_outcome", "post_intervention_state",
    "clean_final_success", "late_success_in_extended_source",
    "uses_privileged_sim_state", "uses_attack_outcome",
    "uses_future_student_input",
})

# Allowlist: only these step fields are projected into student NPZ
STUDENT_ALLOWLIST = frozenset({
    "step", "features_25d", "clean_policy_intent_9d",
    "clean_policy_features", "policy_intent",
})


def _bucket_rank(key: str, salt: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{salt}|{key}".encode()).digest()[:4], "big"
    )


def select_smoke_episodes(manifest_rows: list[dict], seed: int = 42) -> list[dict]:
    selected = []
    for suite in TARGET_SUITES:
        suite_rows = [r for r in manifest_rows if r["suite"] == suite]
        ranked = sorted(suite_rows, key=lambda r: _bucket_rank(r["parent_key"], SMOKE_SALT))
        selected.extend(ranked[:SMOKE_PER_SUITE])
    return selected


def _read_teacher_labels(label_path: Path) -> list[dict]:
    return read_jsonl(label_path)


def _labels_to_targets(labels: list[dict]) -> dict[str, np.ndarray]:
    T = len(labels)
    targets: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for head_name in R9P_HEAD_NAMES:
        targets[head_name] = np.zeros(T, dtype=np.float32)
        masks[head_name] = np.zeros(T, dtype=bool)

    for i, row in enumerate(labels):
        known = bool(row.get("label_known_mask", False))
        if head_name := "grounding_confidence":
            gc = float(row.get("grounding_confidence", 0.0))
            targets["grounding_confidence"][i] = gc
            masks["grounding_confidence"][i] = True
        for head_name in R9P_HEAD_NAMES:
            if head_name == "grounding_confidence":
                continue
            teacher_name = R9P_MODEL_TARGET_MAP[head_name]
            value = row.get(teacher_name)
            if known and value is not None:
                targets[head_name][i] = float(bool(value))
                masks[head_name][i] = True
            # unknown: stays 0.0 / False
    return {"targets": targets, "masks": masks}


def _validate_npz_keys(keys: set) -> list[str]:
    violations = sorted(FORBIDDEN_NPZ_KEYS & keys)
    return violations


def materialize_episode(
    episode_dir: Path,
    metadata: dict,
) -> dict[str, Any]:
    steps_path = episode_dir / "step_records_prefix.jsonl"
    labels_path = episode_dir / "teacher_v2_labels.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"missing step_records_prefix.jsonl: {steps_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"missing teacher_v2_labels.jsonl: {labels_path}")

    steps = read_jsonl(steps_path)
    labels = _read_teacher_labels(labels_path)

    T_steps = len(steps)
    T_labels = len(labels)
    if T_steps != T_labels:
        raise ValueError(
            f"step/label mismatch: {T_steps} steps vs {T_labels} labels in {episode_dir}"
        )

    features_25d = np.zeros((T_steps, 25), dtype=np.float32)
    features_9d = np.zeros((T_steps, 9), dtype=np.float32)
    known_mask = np.zeros(T_steps, dtype=bool)

    for i, (step, label) in enumerate(zip(steps, labels)):
        if int(step.get("step", -1)) != i:
            raise ValueError(f"step discontinuity at index {i}: got step={step.get('step')}")

        # Project only allowlist fields from the (potentially Teacher-rich) source step
        projected = {k: step[k] for k in STUDENT_ALLOWLIST if k in step}

        f25 = np.asarray(projected.get("features_25d", []), dtype=np.float32)
        if f25.shape != (25,):
            raise ValueError(f"features_25d shape {f25.shape} at step {i}")
        if not np.isfinite(f25).all():
            raise ValueError(f"non-finite features_25d at step {i}")
        features_25d[i] = f25

        # 9D policy features: fail-closed if missing
        f9_raw = projected.get("clean_policy_intent_9d")
        if f9_raw is None:
            f9_raw = projected.get("clean_policy_features")
        if f9_raw is None:
            f9_raw = projected.get("policy_intent")
        if f9_raw is None:
            raise ValueError(
                f"clean_policy_intent_9d missing at step {i} — "
                f"Model B requires 9D features. Check that source step_records_prefix "
                f"contains clean_policy_intent_9d."
            )
        f9 = np.asarray(f9_raw, dtype=np.float32)
        if f9.shape != (9,):
            raise ValueError(f"clean_policy_intent_9d shape {f9.shape} at step {i}, expected (9,)")
        if not np.isfinite(f9).all():
            raise ValueError(f"non-finite clean_policy_intent_9d at step {i}")
        features_9d[i] = f9

        known_mask[i] = bool(label.get("label_known_mask", False))

        # Validate grounding_confidence is present and valid
        gc = label.get("grounding_confidence")
        if gc is None:
            raise ValueError(f"grounding_confidence missing in teacher label at step {i}")
        gc = float(gc)
        if not np.isfinite(gc) or not 0.0 <= gc <= 1.0:
            raise ValueError(f"grounding_confidence={gc} out of [0,1] at step {i}")

    label_data = _labels_to_targets(labels)

    # Verify projected NPZ has no forbidden keys
    for i, step in enumerate(steps):
        projected_keys = set(STUDENT_ALLOWLIST & set(step.keys()))
        bad = sorted(FORBIDDEN_NPZ_KEYS & projected_keys)
        if bad:
            raise ValueError(f"forbidden keys in allowlist projection at step {i}: {bad}")

    valid_mask = np.ones(T_steps, dtype=bool)

    episode_fully_known_negative = bool(
        known_mask.all() and not label_data["targets"]["critical_window"].any()
    )

    return {
        "features_25d": features_25d,
        "features_9d": features_9d,
        "targets": label_data["targets"],
        "masks": label_data["masks"],
        "valid_mask": valid_mask,
        "known_mask": known_mask,
        "step": np.arange(T_steps, dtype=np.int64),
        "n_steps": T_steps,
        "n_known": int(known_mask.sum()),
        "n_positive": int(label_data["targets"]["critical_window"].sum()),
        "episode_fully_known_negative": episode_fully_known_negative,
        "has_start": bool(label_data["targets"]["window_start"].any()),
        "has_burst_feasible": bool(label_data["targets"]["burst_feasible"].any()),
    }


def write_episode_npz(data: dict, npz_path: Path) -> str:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        features_25d=data["features_25d"],
        features_9d=data["features_9d"],
        **{f"y_{h}": data["targets"][h] for h in R9P_HEAD_NAMES},
        **{f"m_{h}": data["masks"][h] for h in R9P_HEAD_NAMES},
        valid_mask=data["valid_mask"],
        known_mask=data["known_mask"],
        step=data["step"],
    )
    return sha256_file(npz_path)


def compute_normalization(index_rows: list[dict], episodes_dir: Path) -> dict:
    fit_rows = [r for r in index_rows if r["preview_split"] == "FIT"]
    all_25d = []
    all_9d = []
    for row in fit_rows:
        npz = np.load(episodes_dir / row["npz_path"])
        all_25d.append(npz["features_25d"])
        all_9d.append(npz["features_9d"])
    stacked_25d = np.concatenate(all_25d, axis=0)
    stacked_9d = np.concatenate(all_9d, axis=0)
    mean_25d = stacked_25d.mean(axis=0).tolist()
    std_25d = stacked_25d.std(axis=0).tolist()
    mean_9d = stacked_9d.mean(axis=0).tolist()
    std_9d = stacked_9d.std(axis=0).tolist()
    for i, s in enumerate(std_25d):
        if s < 1e-12:
            raise ValueError(f"features_25d dimension {i} has zero std — degenerate")
    for i, s in enumerate(std_9d):
        if s < 1e-12:
            raise ValueError(f"features_9d dimension {i} has zero std — degenerate")
    return {
        "schema": SCHEMA,
        "proprio_mean": mean_25d,
        "proprio_std": std_25d,
        "policy_intent_mean": mean_9d,
        "policy_intent_std": std_9d,
        "fit_episode_count": len(fit_rows),
        "fit_step_count": int(stacked_25d.shape[0]),
    }


def run_materialization(
    plan_root: Path,
    output_root: Path,
    *,
    smoke: bool = False,
    smoke_seed: int = 42,
    suite_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    manifest_path = plan_root / "r9p_preview_episode_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"plan manifest not found: {manifest_path}")
    manifest_rows = read_jsonl(manifest_path)

    if smoke:
        selected = select_smoke_episodes(manifest_rows, seed=smoke_seed)
    else:
        selected = manifest_rows

    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    episodes_dir = output_root / "episodes"
    errors = []
    index_rows = []

    # Write smoke selection manifest for audit closure
    if smoke:
        smoke_manifest = [
            {"parent_key": r["parent_key"], "suite": r["suite"],
             "task_index": r["task_index"], "preview_split": r["preview_split"]}
            for r in selected
        ]
        write_jsonl(output_root / "smoke_selection_manifest.jsonl", smoke_manifest)

    for row in selected:
        parent_key = row["parent_key"]
        suite = row["suite"]
        try:
            if suite_roots is None:
                raise ValueError("suite_roots required for materialization")
            suite_root = suite_roots.get(suite)
            if suite_root is None:
                raise ValueError(f"no suite root for {suite}")
            episode_dir = suite_root / "episodes" / suite / parent_key
            if not episode_dir.is_dir():
                # Try with derived_episode_metadata path
                meta_path_rel = row.get("metadata_path", "")
                if meta_path_rel:
                    meta_file = suite_root / meta_path_rel
                    episode_dir = meta_file.parent
            if not episode_dir.is_dir():
                raise FileNotFoundError(f"episode dir not found for {parent_key}")

            meta = read_json(episode_dir / "derived_episode_metadata.json")
            data = materialize_episode(episode_dir, meta)

            npz_rel = f"episodes/{parent_key}.npz"
            npz_path = output_root / npz_rel
            npz_sha = write_episode_npz(data, npz_path)

            index_rows.append({
                "suite": suite,
                "task_index": row["task_index"],
                "state_id": row["state_id"],
                "parent_key": parent_key,
                "cohort": row["cohort"],
                "preview_split": row["preview_split"],
                "npz_path": npz_rel,
                "npz_sha256": npz_sha,
                "n_steps": data["n_steps"],
                "n_known": data["n_known"],
                "n_positive": data["n_positive"],
                "episode_fully_known_negative": data["episode_fully_known_negative"],
                "has_start": data["has_start"],
                "has_burst_feasible": data["has_burst_feasible"],
            })
        except Exception as exc:
            errors.append({"parent_key": parent_key, "suite": suite, "error": str(exc)})

    index_path = output_root / "dataset_index.jsonl"
    write_jsonl(index_path, index_rows)

    status = GATE_PASS_SMOKE if smoke else GATE_PASS_FULL
    if errors:
        status = f"HOLD_{status}"

    report = {
        "schema": SCHEMA,
        "status": status,
        "smoke": smoke,
        "total_episodes": len(selected),
        "materialized": len(index_rows),
        "errors": len(errors),
        "error_details": errors[:20] if errors else [],
    }

    if not smoke and not errors:
        norm = compute_normalization(index_rows, output_root)
        write_json(output_root / "normalization.json", norm)
        report["normalization"] = {
            "fit_episode_count": norm["fit_episode_count"],
            "fit_step_count": norm["fit_step_count"],
        }

    report_path = output_root / ("smoke_report.json" if smoke else "materialization_report.json")
    write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize R9P OGS-1500 episodes")
    parser.add_argument("--plan-root", required=True, type=Path, help="R9P plan output root")
    parser.add_argument("--spatial-root", type=Path, help="R8Z spatial suite root")
    parser.add_argument("--object-root", type=Path, help="R8Z object suite root")
    parser.add_argument("--goal-root", type=Path, help="R8Z goal suite root")
    parser.add_argument("--output-root", required=True, type=Path, help="Materialization output root")
    parser.add_argument("--smoke", action="store_true", help="Run 24-episode smoke only")
    parser.add_argument("--smoke-seed", type=int, default=42, help="Smoke selection seed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite_roots = {}
    if args.spatial_root:
        suite_roots["libero_spatial"] = args.spatial_root
    if args.object_root:
        suite_roots["libero_object"] = args.object_root
    if args.goal_root:
        suite_roots["libero_goal"] = args.goal_root

    report = run_materialization(
        plan_root=args.plan_root,
        output_root=args.output_root,
        smoke=args.smoke,
        smoke_seed=args.smoke_seed,
        suite_roots=suite_roots if suite_roots else None,
    )
    status = report["status"]
    print(f"Materialization: {status}")
    print(f"  Episodes: {report['materialized']}/{report['total_episodes']}")
    if report["errors"]:
        print(f"  Errors: {report['errors']}")
    return 0 if "PASS" in status else 1


if __name__ == "__main__":
    sys.exit(main())
