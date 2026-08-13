"""Dependency-injected zero-treatment compatibility auditor.

The caller supplies the official environment and primary-input capture.  This
module only replays frozen clean actions, restores captured state, and emits a
fail-closed receipt.  It deliberately has no intervention or label-producer
dependency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_causal_observation_snapshot import (
    assert_primary_observation_exact,
    capture_runtime_state,
    capture_simulator_state,
    load_snapshot,
    restore_rng_state,
    restore_runtime_state,
)

from scripts.detector_v5.stage_v_runtime_diff import diff as runtime_diff


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
PRIMARY_FIELDS = (
    "raw_observation", "canonical_policy_rgb_224", "processed_image", "input_ids",
    "pixel_values", "attention_mask", "prompt", "decode_config",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(snapshot_root: Path, manifest: Mapping[str, Any], payload: Mapping[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    binding = manifest.get("binding") if isinstance(manifest.get("binding"), Mapping) else {}
    probe = payload.get("probe") if isinstance(payload.get("probe"), Mapping) else {}
    result = {
        "parent_key": binding.get("parent_key"),
        "probe_id": binding.get("probe_id", probe.get("probe_id")),
        "branch_id": None,
        "snapshot_manifest_sha256": _sha(snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json"),
        "snapshot_source_commit": binding.get("source_commit"),
        "snapshot_source_tree": binding.get("source_tree"),
        "current_runtime_commit": None,
        "current_runtime_tree": None,
        "runtime_worktree": None,
        "exact_plan_manifest_sha256": None,
        "runtime_provenance_receipt_sha256": None,
        "closure_report_sha256": None,
    }
    result.update(dict(extra or {}))
    return result


def audit_probe(
    snapshot_root: Path,
    env: Any,
    clean_prefix_actions: Sequence[Any],
    *,
    actual_primary_input: Mapping[str, Any] | None = None,
    capture_primary_input: Callable[[Any, Any], Mapping[str, Any]] | None = None,
    capture_simulator: Callable[[Any], Mapping[str, Any]] | None = None,
    capture_runtime: Callable[[Any], Mapping[str, Any]] | None = None,
    restore_runtime: Callable[[Any, Mapping[str, Any]], None] | None = None,
    restore_rng: Callable[[Mapping[str, Any]], None] | None = None,
    close_env: Callable[[Any], None] | None = None,
    model: Any | None = None,
    adapter: Any | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_root = Path(snapshot_root).resolve()
    loaded: Mapping[str, Any] | None = None
    payload: Mapping[str, Any] = {}
    receipt_context: dict[str, Any] = {}
    errors: list[str] = []
    diffs: list[dict[str, Any]] = []
    replayed = 0
    current_observation: Any = None
    observation_hashes: Mapping[str, Any] | None = None
    try:
        loaded = load_snapshot(snapshot_root, materialize_torch=True)
        payload = loaded["payload"]
        receipt_context = _context(snapshot_root, loaded["manifest"], payload, context)
        observation_hashes = assert_primary_observation_exact(payload)
        probe = payload.get("probe") if isinstance(payload.get("probe"), Mapping) else {}
        step = int(probe.get("step", -1))
        if step < 0:
            raise ValueError("PROBE_STEP_INVALID")
        episode_rng = payload.get("episode_start_rng_state")
        expected_runtime = payload.get("controller_and_wrapper_runtime_state")
        required_rng = payload.get("required_rng_state")
        expected_simulator = payload.get("full_simulator_state")
        if not isinstance(episode_rng, Mapping) or not isinstance(expected_runtime, Mapping) or not isinstance(required_rng, Mapping) or not isinstance(expected_simulator, Mapping):
            raise ValueError("SNAPSHOT_RUNTIME_FIELDS_MISSING")
        (restore_rng or restore_rng_state)(episode_rng)
        for index in range(step):
            if index >= len(clean_prefix_actions):
                raise ValueError(f"CLEAN_PREFIX_MISSING:{index}")
            current_observation, _reward, done, _info = env.step(clean_prefix_actions[index])
            replayed += 1
            if bool(done):
                raise ValueError(f"CLEAN_PREFIX_TERMINATED:{index}")
        sim_capture = capture_simulator or capture_simulator_state
        runtime_capture = capture_runtime or (lambda target: capture_runtime_state(target, model=model, adapter=adapter))
        simulator_actual = sim_capture(env)
        (restore_runtime or (lambda target, state: restore_runtime_state(target, state, model=model, adapter=adapter)))(env, expected_runtime)
        (restore_rng or restore_rng_state)(required_rng)
        runtime_actual = runtime_capture(env)
        diffs.extend(runtime_diff(expected_simulator, simulator_actual, context={**receipt_context, "branch_id": "ZERO_TREATMENT_SIMULATOR"}))
        diffs.extend(runtime_diff(expected_runtime, runtime_actual, context={**receipt_context, "branch_id": "ZERO_TREATMENT_RUNTIME"}))
        if capture_primary_input is not None:
            actual_primary_input = capture_primary_input(env, current_observation)
        if not isinstance(actual_primary_input, Mapping):
            errors.append("PRIMARY_INPUT_CAPTURE_MISSING")
        else:
            expected_primary = {field: payload.get(field) for field in PRIMARY_FIELDS}
            actual_primary = {field: actual_primary_input.get(field) for field in PRIMARY_FIELDS}
            diffs.extend(runtime_diff(expected_primary, actual_primary, context={**receipt_context, "branch_id": "ZERO_TREATMENT_PRIMARY_INPUT"}))
    except Exception as exc:
        errors.append(f"AUDIT_ERROR:{type(exc).__name__}:{exc}")
    finally:
        try:
            (close_env or (lambda target: target.close()))(env)
        except Exception as exc:
            errors.append(f"ENV_CLOSE_ERROR:{type(exc).__name__}:{exc}")
    return {
        "schema": "STAGE_V_M4_ZERO_TREATMENT_AUDIT_RECEIPT_V1",
        "status": "PASS_ZERO_TREATMENT_COMPATIBILITY" if not errors and not diffs else "HOLD_ZERO_TREATMENT_COMPATIBILITY",
        "snapshot_root": str(snapshot_root),
        **receipt_context,
        "observation_hashes": dict(observation_hashes or {}),
        "clean_prefix_replay_steps": replayed,
        "post_snapshot_primary_window_steps": 0,
        "treatment_steps": 0,
        "forced_open_steps": 0,
        "label_records": 0,
        "v_phys_generated": False,
        "intervention_executed": False,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
        "runtime_diff_count": len(diffs),
        "runtime_diffs": diffs,
        "errors": sorted(set(errors)),
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(receipt), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
