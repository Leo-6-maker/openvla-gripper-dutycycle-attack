"""Closed-world manifest contract for the Detector-v2 matched-load 2x2 matrix."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CORE_CONDITIONS = (
    "CLEAN",
    "DET_GRIPPER_VIS_PGD",
    "DET_RANDOM_VIS_ATTACK",
    "RANDTIME_GRIPPER_VIS_PGD",
    "RANDTIME_RANDOM_VIS_ATTACK",
)
ATTACK_CONDITIONS = frozenset(CORE_CONDITIONS[1:])
DETECTOR_TIMING_CONDITIONS = frozenset({"DET_GRIPPER_VIS_PGD", "DET_RANDOM_VIS_ATTACK"})
RANDOM_TIMING_CONDITIONS = frozenset({"RANDTIME_GRIPPER_VIS_PGD", "RANDTIME_RANDOM_VIS_ATTACK"})
GRIPPER_OBJECTIVE_CONDITIONS = frozenset({"DET_GRIPPER_VIS_PGD", "RANDTIME_GRIPPER_VIS_PGD"})
CONTROL_OBJECTIVE_CONDITIONS = frozenset({"DET_RANDOM_VIS_ATTACK", "RANDTIME_RANDOM_VIS_ATTACK"})


@dataclass(frozen=True)
class AttackLoadSpec:
    burst_length: int
    epsilon: float
    step_size: float
    pgd_steps: int
    projection: str
    cast_policy: str
    preprocessing: str
    image_height: int
    image_width: int
    random_start_policy: str
    temporal_init_policy: str
    num_loss_forwards_per_frame: int
    num_backwards_per_frame: int
    num_adv_decodes_per_frame: int

    def validate(self) -> None:
        if self.burst_length <= 0 or self.pgd_steps <= 0:
            raise ValueError("burst_length and pgd_steps must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        for name in (
            "projection",
            "cast_policy",
            "preprocessing",
            "random_start_policy",
            "temporal_init_policy",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        # The mature target-token route reports K optimization forwards plus a
        # final audited forward. Other frozen routes may report K. The exact
        # route-reported count is frozen here and compared at runtime.
        if type(self.num_loss_forwards_per_frame) is not int or self.num_loss_forwards_per_frame < self.pgd_steps:
            raise ValueError("num_loss_forwards_per_frame must be an integer >= pgd_steps")
        if type(self.num_backwards_per_frame) is not int or self.num_backwards_per_frame < self.pgd_steps:
            raise ValueError("num_backwards_per_frame must be an integer >= pgd_steps")
        if self.num_adv_decodes_per_frame != 1:
            raise ValueError("num_adv_decodes_per_frame must be exactly one")

    def fingerprint(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


REQUIRED_JOB_FIELDS = (
    "condition",
    "parent_key",
    "suite",
    "task_index",
    "state_id",
    "eval_seed",
    "clean_parent_sha256",
    "initial_state_sha256",
    "detector_checkpoint_sha256",
    "detector_config_sha256",
    "timing_source",
    "objective_family",
    "objective_seed",
    "attack_enabled",
    "expected_attacked_frames",
    "load_spec",
)


def _require_sha(value: Any, field: str) -> str:
    value = str(value)
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a full lowercase SHA256")
    return value


def _load_spec(value: Any) -> AttackLoadSpec:
    if isinstance(value, AttackLoadSpec):
        value.validate()
        return value
    if not isinstance(value, Mapping):
        raise ValueError("load_spec must be an AttackLoadSpec or mapping")
    missing = [field for field in AttackLoadSpec.__dataclass_fields__ if field not in value]
    if missing:
        raise ValueError("load_spec missing fields: " + ", ".join(missing))
    spec = AttackLoadSpec(**{field: value[field] for field in AttackLoadSpec.__dataclass_fields__})
    spec.validate()
    return spec


def validate_matched_load_job(row: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_JOB_FIELDS if field not in row]
    if missing:
        raise ValueError("missing matched-load job fields: " + ", ".join(missing))
    condition = str(row["condition"])
    if condition not in CORE_CONDITIONS:
        raise ValueError("unknown core condition")
    if not str(row["parent_key"]).strip() or not str(row["suite"]).strip():
        raise ValueError("parent_key and suite are required")
    for field in ("task_index", "state_id", "eval_seed", "objective_seed"):
        if type(row[field]) is not int or int(row[field]) < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    for field in (
        "clean_parent_sha256",
        "initial_state_sha256",
        "detector_checkpoint_sha256",
        "detector_config_sha256",
    ):
        _require_sha(row[field], field)
    if type(row["attack_enabled"]) is not bool:
        raise ValueError("attack_enabled must be boolean")
    if type(row["expected_attacked_frames"]) is not int or row["expected_attacked_frames"] < 0:
        raise ValueError("expected_attacked_frames must be a non-negative integer")

    spec = _load_spec(row["load_spec"])
    timing = str(row["timing_source"])
    objective = str(row["objective_family"])
    if condition == "CLEAN":
        if row["attack_enabled"] or row["expected_attacked_frames"] != 0:
            raise ValueError("CLEAN cannot execute attacked frames")
        if timing != "NONE" or objective != "NONE":
            raise ValueError("CLEAN timing/objective must be NONE")
        return

    if not row["attack_enabled"]:
        raise ValueError("attack conditions must set attack_enabled=true")
    if row["expected_attacked_frames"] != spec.burst_length:
        raise ValueError("expected_attacked_frames must equal the frozen burst length")
    if condition in DETECTOR_TIMING_CONDITIONS and timing != "DETECTOR":
        raise ValueError("DET conditions require timing_source=DETECTOR")
    if condition in RANDOM_TIMING_CONDITIONS and timing != "RANDOM_TIME_MATCHED":
        raise ValueError("RANDTIME conditions require timing_source=RANDOM_TIME_MATCHED")
    if condition in GRIPPER_OBJECTIVE_CONDITIONS and objective != "GRIPPER_TARGETED_VIS_PGD":
        raise ValueError("gripper conditions require the gripper-targeted objective")
    if condition in CONTROL_OBJECTIVE_CONDITIONS and objective not in {
        "RANDOM_DIRECTION_PGD_LOOP",
        "NONGRIPPER_VIS_PGD",
        "SHUFFLED_GRIPPER_GRADIENT",
    }:
        raise ValueError("control condition requires a compute-matched random/non-gripper objective")


def _identity_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["parent_key"]),
        str(row["suite"]),
        int(row["task_index"]),
        int(row["state_id"]),
        int(row["eval_seed"]),
        str(row["clean_parent_sha256"]),
        str(row["initial_state_sha256"]),
        str(row["detector_checkpoint_sha256"]),
        str(row["detector_config_sha256"]),
    )


def validate_core_2x2_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_conditions: Sequence[str] = CORE_CONDITIONS,
) -> Dict[str, Any]:
    """Validate closed-world parent groups and exact attack-load matching."""

    if not rows:
        raise ValueError("matched-load manifest cannot be empty")
    required = tuple(required_conditions)
    if len(set(required)) != len(required):
        raise ValueError("required_conditions contains duplicates")
    if set(required) != set(CORE_CONDITIONS):
        raise ValueError("primary manifest must contain the frozen five core conditions")

    by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_matched_load_job(row)
        by_parent[str(row["parent_key"])].append(row)

    parent_summaries: list[dict[str, Any]] = []
    for parent, group in sorted(by_parent.items()):
        conditions = [str(row["condition"]) for row in group]
        duplicates = sorted(condition for condition in set(conditions) if conditions.count(condition) > 1)
        missing = sorted(set(required) - set(conditions))
        unexpected = sorted(set(conditions) - set(required))
        if duplicates or missing or unexpected:
            raise ValueError(
                f"parent {parent} condition closure failed duplicates={duplicates} "
                f"missing={missing} unexpected={unexpected}"
            )
        identities = {_identity_tuple(row) for row in group}
        if len(identities) != 1:
            raise ValueError(f"parent {parent} identity/provenance differs across conditions")

        attack_rows = [row for row in group if row["condition"] in ATTACK_CONDITIONS]
        load_fingerprints = {_load_spec(row["load_spec"]).fingerprint() for row in attack_rows}
        if len(load_fingerprints) != 1:
            raise ValueError(f"parent {parent} attack load is not exactly matched")
        control_families = {
            str(row["objective_family"])
            for row in attack_rows
            if row["condition"] in CONTROL_OBJECTIVE_CONDITIONS
        }
        if len(control_families) != 1:
            raise ValueError(f"parent {parent} must use one frozen control objective family")

        objective_counts = Counter(str(row["objective_family"]) for row in attack_rows)
        objective_seeds: dict[str, set[int]] = defaultdict(set)
        for row in attack_rows:
            objective_seeds[str(row["objective_family"])].add(int(row["objective_seed"]))
        if any(count != 2 for count in objective_counts.values()) or len(objective_counts) != 2:
            raise ValueError(
                f"parent {parent} must contain exactly two timing rows per objective family: "
                f"{dict(objective_counts)}"
            )
        unpaired = {
            objective: sorted(seeds)
            for objective, seeds in objective_seeds.items()
            if len(seeds) != 1
        }
        if unpaired:
            raise ValueError(
                f"parent {parent} objective seeds are not paired across timing conditions: {unpaired}"
            )

        detector_starts = {
            row.get("planned_start_step")
            for row in attack_rows
            if row["condition"] in DETECTOR_TIMING_CONDITIONS
        }
        random_starts = {
            row.get("planned_start_step")
            for row in attack_rows
            if row["condition"] in RANDOM_TIMING_CONDITIONS
        }
        if None in detector_starts or len(detector_starts) != 1:
            raise ValueError(f"parent {parent} DET conditions require one identical planned_start_step")
        if None in random_starts or len(random_starts) != 1:
            raise ValueError(f"parent {parent} RANDTIME conditions require one identical planned_start_step")
        if detector_starts == random_starts:
            raise ValueError(f"parent {parent} random-time start must differ from detector timing")

        parent_summaries.append(
            {
                "parent_key": parent,
                "condition_count": len(group),
                "load_spec_sha256": next(iter(load_fingerprints)),
                "detector_start_step": next(iter(detector_starts)),
                "random_time_start_step": next(iter(random_starts)),
                "control_objective_family": next(iter(control_families)),
                "objective_seeds": {
                    objective: next(iter(seeds))
                    for objective, seeds in sorted(objective_seeds.items())
                },
            }
        )

    canonical = [dict(row, load_spec=asdict(_load_spec(row["load_spec"]))) for row in rows]
    manifest_sha256 = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "PASS_CORE_2X2_MATCHED_LOAD",
        "parent_count": len(by_parent),
        "job_count": len(rows),
        "conditions": list(required),
        "manifest_sha256": manifest_sha256,
        "parents": parent_summaries,
    }


def deterministic_objective_seed(parent_key: str, condition: str, master_seed: int) -> int:
    """Stable per-parent/objective seed without Python's randomized hash."""

    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    material = f"{master_seed}|{parent_key}|{condition}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFFFFFF
