#!/usr/bin/env python3
"""Independent reducer for C3R clean renderer qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


VISUAL_METRICS = tuple(f"{stage}_{metric}" for stage in ("r1", "r2", "r4") for metric in ("diff_count", "max_abs", "mae"))
POLICY_COLUMNS = (
    "tokens_exact",
    "raw_action_exact",
    "env_action_exact",
    "gripper_semantic_exact",
)
STATE_COLUMNS = (
    "pre_qpos_exact",
    "pre_qvel_exact",
    "pre_flat_sim_exact",
    "pre_student_exact",
    "pre_feature_history_exact",
    "post_student_exact",
    "post_feature_history_exact",
    "post_qpos_exact",
    "post_qvel_exact",
    "post_flat_sim_exact",
)
EXACT_COLUMNS = POLICY_COLUMNS + STATE_COLUMNS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_seal(root: Path) -> list[str]:
    manifest = root / "recursive_sha256_manifest.csv"
    if not manifest.is_file():
        return ["MISSING_MANIFEST"]
    failures = []
    for row in read_csv(manifest):
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"MISSING:{row['path']}")
        elif int(row["size"]) != path.stat().st_size:
            failures.append(f"SIZE:{row['path']}")
        elif row["sha256"] != sha256_file(path):
            failures.append(f"SHA:{row['path']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-process-dir", required=True)
    parser.add_argument("--fresh-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-workers", type=int, default=10)
    args = parser.parse_args()

    same = Path(args.same_process_dir)
    fresh_dirs = sorted(path for path in Path(args.fresh_root).glob("worker_*") if path.is_dir())
    if len(fresh_dirs) != int(args.expected_workers):
        raise SystemExit(f"expected {args.expected_workers} fresh workers, found {len(fresh_dirs)}")

    calibration = read_csv(same / "c3r_prefix_replay_calibration.csv")
    static_same = read_csv(same / "c3r_static_render_same_process.csv")
    validation: list[dict[str, str]] = []
    static_fresh: list[dict[str, str]] = []
    seal_failures = {str(same): audit_seal(same)}
    for directory in fresh_dirs:
        validation.extend(read_csv(directory / "c3r_prefix_replay_validation.csv"))
        static_fresh.extend(read_csv(directory / "c3r_static_render_fresh_process.csv"))
        seal_failures[str(directory)] = audit_seal(directory)

    bounds = {metric: max(float(row[metric]) for row in calibration) for metric in VISUAL_METRICS}
    exact_failures = [
        {"cohort": row["cohort"], "worker": row["repetition"], "step": row["step"], "field": field}
        for row in calibration + validation
        for field in EXACT_COLUMNS
        if not is_true(row[field])
    ]
    bound_failures = [
        {
            "worker": row["repetition"],
            "step": row["step"],
            "field": metric,
            "value": row[metric],
            "bound": bounds[metric],
        }
        for row in validation
        for metric in VISUAL_METRICS
        if float(row[metric]) > bounds[metric]
    ]
    seal_failure_count = sum(len(items) for items in seal_failures.values())
    heldout_workers = {int(row["repetition"]) for row in validation}
    policy_failure_count = sum(1 for row in exact_failures if row["field"] in POLICY_COLUMNS)
    state_failure_count = sum(1 for row in exact_failures if row["field"] in STATE_COLUMNS)
    policy_equivalence = (
        len(heldout_workers) == int(args.expected_workers)
        and not exact_failures
        and not bound_failures
        and seal_failure_count == 0
    )
    strict_rgb_exact = all(int(row["r1_diff_count"]) == 0 for row in calibration + validation)
    all_rows = static_same + static_fresh + calibration + validation
    first_divergent_stage = next(
        (
            stage.upper()
            for stage in ("r0", "r1", "r2", "r3", "r4")
            if any(int(float(row[f"{stage}_diff_count"])) != 0 for row in all_rows)
        ),
        "NONE",
    )
    result = {
        "stage": "C3R_RENDERER_DETERMINISM_QUALIFICATION",
        "strict_rgb_byte_exact_parity": "PASS" if strict_rgb_exact else "FAIL",
        "nonvisual_state_exactness": "PASS" if state_failure_count == 0 else "FAIL",
        "policy_output_exactness": "PASS" if policy_failure_count == 0 else "FAIL",
        "render_equivalence": "PASS_UNDER_PREREGISTERED_BOUND" if policy_equivalence else "FAIL",
        "route_decision": (
            "C3_RENDER_TOLERANT_POLICY_EQUIVALENCE"
            if policy_equivalence
            else "CANONICAL_BOUNDARY_OBSERVATION_REQUIRED"
        ),
        "calibration_rows": len(calibration),
        "validation_rows": len(validation),
        "heldout_workers": len(heldout_workers),
        "first_divergent_stage": first_divergent_stage,
        "visual_bounds": bounds,
        "exact_failure_count": len(exact_failures),
        "policy_failure_count": policy_failure_count,
        "state_failure_count": state_failure_count,
        "bound_failure_count": len(bound_failures),
        "seal_failure_count": seal_failure_count,
        "attack_paths_run": False,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "c3r_static_render_same_process.csv", static_same)
    write_csv(output / "c3r_static_render_fresh_process.csv", static_fresh)
    write_csv(output / "c3r_prefix_replay_calibration.csv", calibration)
    write_csv(output / "c3r_prefix_replay_validation.csv", validation)
    write_csv(output / "c3r_policy_output_exactness.csv", exact_failures or [{"status": "PASS"}])
    write_csv(output / "c3r_visual_bound_failures.csv", bound_failures or [{"status": "PASS"}])
    (output / "c3r_renderer_determinism_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "c3r_seal_audit.json").write_text(
        json.dumps(seal_failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
