"""Build the outcome-blind M3.5 intervention-corridor probe plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from gripper_attack.stage_v_m3_5_phase_classifier import PHASES  # noqa: E402


SCHEMA = "STAGE_V_M3_5_PROBE_PLAN_V2"
VERSION = "V2"
PROBE_COUNT = 24
H_PHYS = 10
DOSE_STEPS = {"T3": 3, "T5": 5, "T10": 10}
MIN_REMAINING_STEPS = max(DOSE_STEPS.values()) + H_PHYS
DEFAULT_SELECTION_VERSION = "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1"


class ProbePlanError(ValueError):
    """Raised when a clean trajectory cannot produce the frozen 24 probes."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value if isinstance(value, list) else value.get("steps") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ProbePlanError("CLEAN_TRAJECTORY_ROWS_REQUIRED")
    return [dict(row) for row in rows]


def _finite_vector(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _corridor_candidate(row: Mapping[str, Any], *, observed_remaining_horizon: int) -> dict[str, Any] | None:
    try:
        declared_remaining = int(row.get("remaining_horizon"))
    except (TypeError, ValueError):
        return None
    phase = row.get("clean_only_phase_label")
    distance = row.get("object_eef_distance_m")
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return None
    if not (
        row.get("clean_record_valid") is True
        and row.get("clean_terminal") is not True
        and row.get("phase_eligible") is True
        and phase in PHASES
        and row.get("contact_telemetry_valid") is True
        and row.get("object_gripper_contact") is True
        and isinstance(row.get("object_support_contact"), bool)
        and isinstance(row.get("object_identity"), str)
        and bool(row.get("object_identity"))
        and _finite_vector(row.get("object_position"))
        and _finite_vector(row.get("eef_position"))
        and math.isfinite(distance)
        and declared_remaining >= MIN_REMAINING_STEPS
        and int(observed_remaining_horizon) >= MIN_REMAINING_STEPS
    ):
        return None
    return {
        "step": int(row["step"]),
        "phase_label": str(phase),
        "object_identity": str(row["object_identity"]),
        "object_gripper_contact": True,
        "object_support_contact": bool(row["object_support_contact"]),
        "object_eef_distance_m": distance,
        "remaining_horizon": min(declared_remaining, int(observed_remaining_horizon)),
        "declared_remaining_horizon": declared_remaining,
        "observed_remaining_horizon": int(observed_remaining_horizon),
        "state_sha256": row.get("state_sha256"),
        "policy_input_sha256": row.get("policy_input_sha256"),
        "policy_rgb_224_sha256": row.get("policy_rgb_224_sha256"),
        "eligibility": {
            "clean_record_valid": True,
            "clean_terminal": False,
            "phase_eligible": True,
            "contact_telemetry_valid": True,
            "object_identity_bound": True,
            "object_gripper_contact": True,
            "intentional_post_release": False,
            "remaining_horizon_gte": MIN_REMAINING_STEPS,
            "outcomes_read": False,
        },
    }


def _quantile_indices(candidate_count: int) -> list[int]:
    if candidate_count < PROBE_COUNT:
        raise ProbePlanError(f"PROBE_PLAN_INSUFFICIENT_CORRIDOR:{candidate_count}/{PROBE_COUNT}")
    denominator = PROBE_COUNT - 1
    indices = [((ordinal * (candidate_count - 1)) + denominator // 2) // denominator for ordinal in range(PROBE_COUNT)]
    if len(set(indices)) != PROBE_COUNT:
        raise ProbePlanError(f"PROBE_PLAN_QUANTILE_DEDUP_INSUFFICIENT:{len(set(indices))}/{PROBE_COUNT}")
    return indices


def select_probe_steps(
    rows: Sequence[Mapping[str, Any]], parent_key: str, *, selection_version: str = DEFAULT_SELECTION_VERSION,
) -> dict[str, Any]:
    if not parent_key:
        raise ProbePlanError("PARENT_KEY_REQUIRED")
    seen: set[int] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step in seen:
            raise ProbePlanError("CLEAN_TRAJECTORY_STEP_INVALID_OR_DUPLICATED")
        seen.add(step)
        normalized_rows.append(dict(row))
    normalized_rows.sort(key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in normalized_rows] != list(range(len(normalized_rows))):
        raise ProbePlanError("CLEAN_TRAJECTORY_STEPS_NOT_CONTIGUOUS_FROM_ZERO")
    candidates = [
        candidate for index, row in enumerate(normalized_rows)
        if (candidate := _corridor_candidate(row, observed_remaining_horizon=len(normalized_rows) - index)) is not None
    ]
    indices = _quantile_indices(len(candidates))
    selected = [
        {
            **candidates[index],
            "probe_id": f"Q{ordinal:02d}",
            "quantile_ordinal": ordinal,
            "quantile_index": index,
            "quantile_fraction": f"{ordinal}/{PROBE_COUNT - 1}",
        }
        for ordinal, index in enumerate(indices)
    ]
    if len({int(item["step"]) for item in selected}) != PROBE_COUNT:
        raise ProbePlanError("PROBE_PLAN_STEP_DUPLICATION")
    phase_distribution = {phase: sum(item["phase_label"] == phase for item in selected) for phase in PHASES}
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "FROZEN_BEFORE_COUNTERFACTUAL_OUTCOMES",
        "canonical_parent_key": parent_key,
        "selection_algorithm": "sort eligible corridor by timestep; for q=0..23 choose round_half_up(q*(N-1)/23); preserve order; deterministic dedup; fail if fewer than 24 unique",
        "selection_version": selection_version,
        "source_inputs": ["clean trajectory rows", "clean-only contact and object telemetry", "remaining horizon"],
        "forbidden_inputs": ["OPEN outcome", "CONTROL outcome", "V_phys", "V_task", "Teacher prediction", "Student prediction"],
        "outcomes_read": False,
        "probe_count": PROBE_COUNT,
        "corridor_candidate_count": len(candidates),
        "corridor_first_step": int(candidates[0]["step"]),
        "corridor_last_step": int(candidates[-1]["step"]),
        "corridor_rule": "clean valid nonterminal state with bound finite object/eef telemetry, explicit gripper-object contact, registered clean phase, and remaining_horizon >= T10 + H_phys",
        "intentional_post_release_exclusion": "object_gripper_contact must be true at the probe",
        "selected_phase_distribution_descriptive_only": phase_distribution,
        "dose_steps": dict(DOSE_STEPS),
        "H_phys": H_PHYS,
        "minimum_remaining_steps": MIN_REMAINING_STEPS,
        "trajectory_sha256": _canonical_sha(normalized_rows),
        "probe_steps": selected,
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-version", default=DEFAULT_SELECTION_VERSION)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ProbePlanError(f"REFUSE_OVERWRITE:{output}")
    plan = select_probe_steps(_load_rows(args.trajectory.resolve()), args.parent_key, selection_version=args.selection_version)
    _write_json(output, plan)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "probe_count": plan["probe_count"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
