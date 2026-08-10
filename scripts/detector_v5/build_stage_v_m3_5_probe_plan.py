"""Build an outcome-blind, phase-stratified M3.5 probe plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from gripper_attack.stage_v_m3_5_phase_classifier import (  # noqa: E402
    MIN_REMAINING_STEPS,
    PHASES,
)


SCHEMA = "STAGE_V_M3_5_PROBE_PLAN_V1"
DEFAULT_SALT = "STAGE_V_M3_5_PROBE_SELECTION_V1_1"


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


def _rank(salt: str, parent_key: str, phase: str, step: int) -> str:
    return hashlib.sha256(f"{salt}::{parent_key}::{phase}::{step}".encode("utf-8")).hexdigest()


def select_probe_steps(
    rows: Sequence[Mapping[str, Any]], parent_key: str, *, salt: str = DEFAULT_SALT,
) -> dict[str, Any]:
    if not parent_key:
        raise ProbePlanError("PARENT_KEY_REQUIRED")
    candidates: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    seen: set[int] = set()
    normalized_rows = []
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step in seen:
            raise ProbePlanError("CLEAN_TRAJECTORY_STEP_INVALID_OR_DUPLICATED")
        seen.add(step)
        normalized_rows.append(dict(row))
        phase = row.get("clean_only_phase_label")
        eligible = row.get("phase_eligible") is True and row.get("clean_record_valid") is True
        try:
            remaining = int(row.get("remaining_horizon"))
        except (TypeError, ValueError):
            remaining = -1
        if eligible and phase in candidates and remaining >= MIN_REMAINING_STEPS:
            candidates[str(phase)].append({
                "step": step,
                "phase_label": str(phase),
                "remaining_horizon": remaining,
                "eligibility": {
                    "clean_record_valid": True,
                    "phase_eligible": True,
                    "remaining_horizon_gte": MIN_REMAINING_STEPS,
                    "outcomes_read": False,
                },
            })
    if any(len(candidates[phase]) < 6 for phase in PHASES):
        counts = {phase: len(candidates[phase]) for phase in PHASES}
        raise ProbePlanError(f"PROBE_PLAN_INSUFFICIENT_PHASE_COVERAGE:{counts}")
    selected: list[dict[str, Any]] = []
    for phase in PHASES:
        ranked = sorted(
            (_rank(salt, parent_key, phase, int(item["step"])), item)
            for item in candidates[phase]
        )
        for rank, item in ranked[:6]:
            selected.append({**item, "selection_rank_sha256": rank})
    selected.sort(key=lambda item: int(item["step"]))
    if len({int(item["step"]) for item in selected}) != 24:
        raise ProbePlanError("PROBE_PLAN_STEP_DUPLICATION")
    return {
        "schema": SCHEMA,
        "version": "V1",
        "status": "FROZEN_BEFORE_COUNTERFACTUAL_OUTCOMES",
        "canonical_parent_key": parent_key,
        "selection_algorithm": "sha256(salt + '::' + parent_key + '::' + phase + '::' + decimal_step), ascending rank",
        "selection_salt": salt,
        "source_inputs": ["clean trajectory rows", "clean-only phase labels", "remaining horizon"],
        "forbidden_inputs": ["OPEN outcome", "CONTROL outcome", "V_phys", "V_task", "Teacher prediction", "Student prediction"],
        "outcomes_read": False,
        "probe_count": 24,
        "probes_per_phase": {phase: 6 for phase in PHASES},
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
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ProbePlanError(f"REFUSE_OVERWRITE:{output}")
    plan = select_probe_steps(_load_rows(args.trajectory.resolve()), args.parent_key, salt=args.salt)
    _write_json(output, plan)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "probe_count": plan["probe_count"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
