"""Episode-level legacy-label and replay-feasibility checks for R8S."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.multisuite_detector.c2g_r8s_common import LEGACY, SUITES, read_jsonl

ACTION_KEYS = ("clean_action", "policy_action", "raw_action", "action_7d", "action_vector", "action")
META_KEYS = {
    "libero": ("libero_commit", "libero_git_commit", "libero_version", "benchmark_commit"),
    "runtime": ("robosuite_version", "mujoco_version", "python_version"),
    "controller": ("controller_config", "controller_configs", "controller", "controller_type", "control_freq"),
    "action_semantics": ("action_space", "action_semantics", "action_normalization", "action_normalization_stats", "unnorm_key", "control_mode"),
    "bddl": ("bddl_file", "bddl_path", "task_bddl", "problem_file"),
    "seed": ("seed", "env_seed", "episode_seed", "eval_seed"),
    "max_steps": ("max_steps", "max_episode_steps", "horizon"),
}


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def finite_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        output = [float(item) for item in value]
    except Exception:
        return None
    return output if output and all(math.isfinite(item) for item in output) else None


def nested_present(metadata: Mapping[str, Any], keys: Sequence[str]) -> bool:
    if any(metadata.get(key) not in (None, "", [], {}) for key in keys):
        return True
    for parent in ("runtime_versions", "versions", "environment", "controller_metadata", "provenance"):
        child = metadata.get(parent)
        if isinstance(child, Mapping) and any(
            child.get(key) not in (None, "", [], {}) for key in keys
        ):
            return True
    return False


def row_action(row: Mapping[str, Any]) -> tuple[list[float] | None, str]:
    for key in ACTION_KEYS:
        value = finite_vector(row.get(key))
        if value is not None:
            return value, key
    return None, ""


def partial_action_4d(row: Mapping[str, Any]) -> bool:
    values = [row.get(key) for key in ("action_dx", "action_dy", "action_dz", "action_gripper")]
    if all(value is not None for value in values):
        try:
            return all(math.isfinite(float(value)) for value in values)
        except Exception:
            return False
    features = finite_vector(row.get("features_25d"))
    return bool(features is not None and len(features) == 25)


def audit_episode(source: Mapping[str, str]) -> dict[str, Any]:
    output = {
        key: source.get(key, "")
        for key in ("suite", "parent_key", "cohort", "split", "metadata_path", "step_records_path")
    }
    output.update(
        task_index=int(source.get("task_index", -1)),
        state_id=int(source.get("state_id", -1)),
        episode_read_ok=False,
        read_error="",
    )
    try:
        metadata = json.loads(Path(output["metadata_path"]).read_text(encoding="utf-8"))
        steps = read_jsonl(Path(output["step_records_path"]))
        indices = [int(row.get("step", index)) for index, row in enumerate(steps)]
        contiguous = indices == list(range(len(indices)))
        present = {
            field: any(field in row and row[field] is not None for row in steps)
            for field in LEGACY
        }
        primary_comparable = primary_disagree = 0
        release_comparable = release_disagree = 0
        dimensions: list[int] = []
        action_keys = Counter()
        full_action_7d = bool(steps)
        partial_action = bool(steps)
        for row in steps:
            role = str(row.get("teacher_event_role", ""))
            phase = str(row.get("teacher_phase", ""))
            primary = optional_bool(row.get("teacher_primary_attackable"))
            release = optional_bool(row.get("teacher_release_safe"))
            if primary is not None and role:
                primary_comparable += 1
                primary_disagree += int(primary != (role == "primary_attackable"))
            if release is not None and phase:
                release_comparable += 1
                release_disagree += int(release != (phase == "release_safe"))
            action, key = row_action(row)
            if action is None:
                full_action_7d = False
            else:
                dimensions.append(len(action))
                action_keys[key] += 1
                full_action_7d = full_action_7d and len(action) == 7
            partial_action = partial_action and partial_action_4d(row)
        identity = (
            output["suite"] in SUITES
            and output["task_index"] >= 0
            and output["state_id"] >= 0
        )
        bound = {
            name: nested_present(metadata, fields)
            for name, fields in META_KEYS.items()
        }
        checks = (
            (full_action_7d, "FULL_ACTION_7D_MISSING"),
            (contiguous, "STEP_DISCONTINUITY"),
            (identity, "OFFICIAL_INIT_STATE_REFERENCE_MISSING"),
            (bound["libero"], "LIBERO_VERSION_UNBOUND"),
            (bound["runtime"], "RUNTIME_VERSIONS_UNBOUND"),
            (bound["controller"], "CONTROLLER_CONFIG_UNBOUND"),
            (bound["action_semantics"], "ACTION_SEMANTICS_UNBOUND"),
            (bound["bddl"], "TASK_BDDL_UNBOUND"),
            (bound["seed"], "REPLAY_SEED_UNBOUND"),
            (bound["max_steps"], "MAX_STEPS_UNBOUND"),
        )
        blockers = [reason for passed, reason in checks if not passed]
        output.update(
            episode_read_ok=True,
            step_count=len(steps),
            step_contiguous=contiguous,
            legacy_any_present=any(present.values()),
            **{f"{field}_present": value for field, value in present.items()},
            primary_comparable_steps=primary_comparable,
            primary_disagreement_steps=primary_disagree,
            release_comparable_steps=release_comparable,
            release_disagreement_steps=release_disagree,
            full_action_7d_complete=full_action_7d,
            partial_action_4d_complete=partial_action,
            action_vector_key=action_keys.most_common(1)[0][0] if action_keys else "",
            action_dimension_min=min(dimensions) if dimensions else None,
            action_dimension_max=max(dimensions) if dimensions else None,
            official_init_state_reference_present=identity,
            libero_version_bound=bound["libero"],
            runtime_versions_bound=bound["runtime"],
            controller_config_bound=bound["controller"],
            action_semantics_bound=bound["action_semantics"],
            task_bddl_bound=bound["bddl"],
            seed_bound=bound["seed"],
            max_steps_bound=bound["max_steps"],
            strict_replay_candidate=bool(full_action_7d and contiguous and identity),
            strict_replay_ready=not blockers,
            replay_blockers=";".join(blockers),
            legacy_auxiliary_eligible=bool(any(present.values()) and contiguous),
            current_teacher_v2_exact_supervision_eligible=False,
        )
    except Exception as exc:
        output["read_error"] = f"{type(exc).__name__}: {exc}"
    return output
