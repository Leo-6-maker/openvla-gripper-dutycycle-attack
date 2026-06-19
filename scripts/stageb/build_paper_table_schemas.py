#!/usr/bin/env python3
"""Write frozen paper-table schemas before final results are available."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TABLES = {
    "table1_end_to_end_attack_results_schema.csv": [
        "cohort", "suite", "task_idx", "state_id", "seed", "condition",
        "success", "failure_taxonomy", "target_token_rate", "qpos_open_response",
        "arm_selectivity", "control_matched", "claim_allowed",
    ],
    "table2_detector_localization_transfer_schema.csv": [
        "suite", "task_idx", "state_id", "clean_success", "mlp_emit",
        "mlp_emit_step", "teacher_valid", "teacher_anchor_step", "timing_class",
        "invalid_feature_steps", "eligible_denominator",
    ],
    "table3_visual_open_qpos_failure_mechanism_schema.csv": [
        "parent", "seed", "frame_or_window", "condition", "official_token",
        "env_gripper", "qpos_delta", "width_delta", "object_contact_proxy",
        "task_outcome", "mechanism_claim_level",
    ],
    "table4_timing_payload_ablations_schema.csv": [
        "ablation", "suite", "parent", "seed", "trigger_policy", "payload",
        "epsilon", "steps", "selected_window", "primary_metric", "control_metric",
        "pass_fail",
    ],
    "table5_online_latency_schema.csv": [
        "component", "hardware_pair", "suite", "median_ms", "p95_ms",
        "max_ms", "num_samples", "online_budget_ms", "meets_budget",
    ],
}

CLAIM_MATRIX = [
    {
        "table": "Table 1",
        "allowed_only_after": "VIS/RAND/control attack runs with physical telemetry",
        "forbidden_before": "clean-only census",
    },
    {
        "table": "Table 2",
        "allowed_only_after": "clean detector transfer audit with explicit denominator",
        "forbidden_before": "Teacher labels for cross-suite timing unless privileged_valid",
    },
    {
        "table": "Table 3",
        "allowed_only_after": "visual-to-token-to-qpos evidence chain",
        "forbidden_before": "token-only fixed-frame result",
    },
    {
        "table": "Table 4",
        "allowed_only_after": "preregistered ablation matrix",
        "forbidden_before": "ad hoc window or lambda tuning",
    },
    {
        "table": "Table 5",
        "allowed_only_after": "online measured component timings",
        "forbidden_before": "offline batch timings",
    },
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    for filename, fields in TABLES.items():
        write_csv(out / filename, fields)
    write_csv(out / "paper_table_claim_matrix.csv", list(CLAIM_MATRIX[0]), CLAIM_MATRIX)
    (out / "paper_table_schema_manifest.json").write_text(json.dumps({
        "tables": TABLES,
        "claim_matrix": CLAIM_MATRIX,
        "note": "Schemas are intentionally empty until denominators are finalized.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PAPER_TABLE_SCHEMAS_DONE", "table_count": len(TABLES)}, sort_keys=True))


if __name__ == "__main__":
    main()
