"""V23 Teacher replay runner for the frozen DEV pilot.

This is intentionally separate from every V22 adapter.  It consumes only
physical sidecar rows plus an independently sealed geometry case stream; raw
policy/action/outcome fields are never used to form a head input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from n5.phase2_labels.c3_t0_semantic_contract import (
    ContractError,
    FALSE,
    TRUE,
    UNKNOWN,
    apply_persistence,
    evaluate_heads,
)
from n5.phase3_student.c3_g_predicate_evaluator import evaluate_case, load_contract


RUNNER_SCHEMA = "C3_T1_V23_TEACHER_RUNNER_V1"
FORBIDDEN_FIELDS = frozenset({
    "task_success", "task_terminal", "terminal", "terminal_state", "reward",
    "outcome", "attack", "future", "action", "action_raw",
    "clean_action_raw_7d", "applied_action_7d", "policy_action", "command",
    "close_intent",
})


class RunnerHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, width: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != width:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _reject_forbidden(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ContractError(f"forbidden physical input: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _tri(value: Any) -> str:
    if value is True or value == TRUE:
        return TRUE
    if value is False or value == FALSE:
        return FALSE
    return UNKNOWN


def _contact_state(row: Mapping[str, Any], object_ids: set[str]) -> str:
    pairs = row.get("mujoco_contact_pairs")
    if not isinstance(pairs, list):
        return UNKNOWN
    found = False
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            return UNKNOWN
        endpoints = {str(item) for item in pair}
        object_hit = any(any(endpoint == name or endpoint.startswith(name + "_") for name in object_ids) for endpoint in endpoints)
        gripper_hit = any("gripper0" in endpoint or "finger1" in endpoint or "finger2" in endpoint for endpoint in endpoints)
        if object_hit and gripper_hit:
            found = True
    return TRUE if found else FALSE


def _qpos_scalar(row: Mapping[str, Any]) -> float | None:
    qpos = _finite_vector(row.get("robot0_gripper_qpos"), 2)
    if qpos is None:
        return None
    return sum(abs(value) for value in qpos) / 2.0


def _qpos_open(row: Mapping[str, Any], threshold: float) -> bool | None:
    value = _qpos_scalar(row)
    return None if value is None else value > threshold


def _comotion(previous: Mapping[str, Any] | None, current: Mapping[str, Any], geometry: Mapping[str, Any] | None) -> str:
    if previous is None or geometry is None:
        return UNKNOWN
    object_now = _finite_vector(geometry.get("object_position"), 3)
    object_prev = _finite_vector(geometry.get("previous_object_position"), 3)
    eef_now = _finite_vector(current.get("robot0_eef_pos"), 3)
    eef_prev = _finite_vector(previous.get("robot0_eef_pos"), 3)
    if None in (object_now, object_prev, eef_now, eef_prev):
        return UNKNOWN
    object_delta = [object_now[i] - object_prev[i] for i in range(3)]
    eef_delta = [eef_now[i] - eef_prev[i] for i in range(3)]
    object_norm = math.sqrt(sum(item * item for item in object_delta))
    eef_norm = math.sqrt(sum(item * item for item in eef_delta))
    if object_norm == 0.0 or eef_norm == 0.0:
        return FALSE
    cosine = sum(object_delta[i] * eef_delta[i] for i in range(3)) / (object_norm * eef_norm)
    return TRUE if cosine >= 0.0 else FALSE


def _placement(geometry: Mapping[str, Any] | None, predicate_contract: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if geometry is None:
        return UNKNOWN, {"reason": "MISSING_GEOMETRY_CASE"}
    result = evaluate_case(geometry, predicate_contract)
    value = _tri(result.get("value"))
    return value, {
        "predicate": result.get("predicate"),
        "reason": result.get("reason"),
        "raw_measurements": result.get("raw_measurements", {}),
        "relative_position": result.get("relative_position"),
    }


def evaluate_step(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    geometry: Mapping[str, Any] | None,
    geometry_contract: Mapping[str, Any],
    step: int,
    step_count: int,
    object_ids: set[str],
    qpos_close_threshold: float,
) -> dict[str, Any]:
    """Evaluate one prefix using only current/past physical observations."""
    # Reject forbidden fields before projecting the physical allowlist.  A
    # silent projection would make injected outcome/action data look harmless
    # instead of proving that it never entered the runner boundary.
    _reject_forbidden(current)
    physical_keys = {
        key: current[key]
        for key in ("mujoco_contact_pairs", "robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat")
        if key in current
    }
    _reject_forbidden(physical_keys)
    contact_now = _contact_state(current, object_ids)
    contact_prev = _contact_state(previous, object_ids) if previous is not None else UNKNOWN
    qpos = _qpos_scalar(current)
    close = None if qpos is None else qpos <= qpos_close_threshold
    open_now = _qpos_open(current, qpos_close_threshold)
    open_prev = _qpos_open(previous, qpos_close_threshold) if previous is not None else None
    release = UNKNOWN if open_now is None or open_prev is None else (TRUE if not open_prev and open_now else FALSE)
    stable = UNKNOWN if contact_now == UNKNOWN or close is None else (TRUE if contact_now == TRUE and close else FALSE)
    transport = _comotion(previous, current, geometry)
    placement, geometry_record = _placement(geometry, geometry_contract)
    stability = UNKNOWN if previous is None else (TRUE if stable == TRUE and contact_now == contact_prev else FALSE)
    contact_loss = UNKNOWN if previous is None or contact_prev == UNKNOWN else (TRUE if contact_prev == TRUE and contact_now == FALSE else FALSE)
    regrasp = UNKNOWN if previous is None or contact_prev == UNKNOWN else (TRUE if contact_prev == FALSE and contact_now == TRUE and close is True else FALSE)
    slip = UNKNOWN if previous is None or contact_prev == UNKNOWN else (TRUE if contact_prev == TRUE and contact_now == FALSE else FALSE)
    physical_known = contact_now != UNKNOWN and qpos is not None
    record = {
        "physical_known": physical_known,
        "stable_grasp": stable,
        "transport_or_manipulation": transport,
        "placement": placement,
        "release": release,
        "stability": stability,
        "protocol_steps_remaining": step_count - step,
        "observed_future_steps_available": step_count - step - 1,
        "slip": slip,
        "regrasp": regrasp,
        "contact_loss": contact_loss,
        "gripper_qpos": qpos,
        "qpos_close_threshold": qpos_close_threshold,
    }
    heads = evaluate_heads(record)
    return {
        "step": step,
        "protocol_steps_remaining": record["protocol_steps_remaining"],
        "observed_future_steps_available": record["observed_future_steps_available"],
        "geometry": geometry_record,
        "geometry_case_present": geometry is not None,
        "physical_components": {
            "contact": contact_now,
            "stable_grasp": stable,
            "transport_or_manipulation": transport,
            "slip": slip,
            "regrasp": regrasp,
            "contact_loss": contact_loss,
        },
        "heads": heads,
    }


def run_episode(sidecar_rows: Sequence[Mapping[str, Any]], geometry_cases: Mapping[int, Mapping[str, Any]],
                episode_id: str, geometry_contract: Mapping[str, Any], object_ids: set[str],
                qpos_close_threshold: float = 0.2) -> dict[str, Any]:
    if not sidecar_rows:
        raise RunnerHold("empty physical sidecar")
    for index, row in enumerate(sidecar_rows):
        if row.get("step") != index:
            raise RunnerHold("sidecar step stream is not contiguous")
    outputs = [
        evaluate_step(
            sidecar_rows[index - 1] if index else None,
            sidecar_rows[index],
            geometry_cases.get(index),
            geometry_contract,
            index,
            len(sidecar_rows),
            object_ids,
            qpos_close_threshold,
        )
        for index in range(len(sidecar_rows))
    ]
    for item in outputs:
        if item["heads"]["safe_release"]["value"] == TRUE and item["heads"]["k10_feasible"]["value"] == TRUE:
            raise ContractError("safe_release TRUE with k10 TRUE")
    return {
        "schema": RUNNER_SCHEMA,
        "episode_id": episode_id,
        "step_count": len(outputs),
        "steps": outputs,
        "unknown_to_false": sum(
            1 for item in outputs for head in item["heads"].values()
            if head["value"] == FALSE and not head["mask"]
        ),
        "forbidden_reads": 0,
    }


def _strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_source_file(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RunnerHold(f"source file is not regular: {path}")
    if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
        raise RunnerHold(f"source file SHA/size mismatch: {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise RunnerHold(f"non-object JSONL row: {path}")
                rows.append(row)
    return rows


def _verify_root_seal(root: Path) -> None:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or sums.is_symlink() or sidecar.is_symlink() or not sums.is_file() or not sidecar.is_file():
        raise RunnerHold(f"sealed root is incomplete: {root}")
    sidecar_text = sidecar.read_text(encoding="utf-8").strip().split()
    if len(sidecar_text) != 2 or sidecar_text[1] != "SHA256SUMS" or sidecar_text[0] != sha256_file(sums):
        raise RunnerHold(f"SHA256SUMS sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RunnerHold(f"malformed SHA256SUMS entry: {root}")
        name = parts[1].strip()
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise RunnerHold(f"unsafe checksum path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != parts[0]:
            raise RunnerHold(f"sealed file mismatch: {path}")
        listed.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if listed != actual:
        raise RunnerHold(f"sealed file closure mismatch: {root}")


def _pilot_episode_ids(manifest: Mapping[str, Any]) -> set[str]:
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 40:
        raise RunnerHold("pilot records are not a closed 40-episode set")
    ids = {record.get("episode_id") for record in records if isinstance(record, Mapping)}
    if len(ids) != 40 or any(not isinstance(item, str) for item in ids):
        raise RunnerHold("pilot episode identities are not unique")
    return ids


def _verify_registry_binding(registry_root: Path) -> None:
    _verify_root_seal(registry_root)
    summary = registry_root / "run_A" / "ENTITY_REGISTRY_V2_SUMMARY.json"
    if not summary.is_file():
        raise RunnerHold("C1-V2 registry summary is missing")
    data = _strict_json(summary)
    if data.get("version") != "C1-V2" or data.get("n_tasks") != 40 or data.get("n_ok") != 40:
        raise RunnerHold("C1-V2 registry is not a closed 40-task PASS")
    task_rows = data.get("per_task")
    if not isinstance(task_rows, list) or len(task_rows) != 40 or any(row.get("status") != "OK" for row in task_rows):
        raise RunnerHold("C1-V2 per-task registry closure failed")


def _verify_geometry_binding(geometry_root: Path, geometry_allowlist: Path,
                             pilot_ids: set[str]) -> None:
    allowlist = _strict_json(geometry_allowlist)
    allowed = allowlist.get("allowed_episode_geometry_roots")
    expected_path = str(geometry_root.resolve())
    if not isinstance(allowed, list) or not any(
        isinstance(item, Mapping) and str(item.get("path", "")).rstrip("/") == expected_path.rstrip("/")
        for item in allowed
    ):
        raise RunnerHold("geometry root is not explicitly allowlisted")
    _verify_root_seal(geometry_root)
    manifest_path = geometry_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RunnerHold("geometry dataset manifest is missing")
    data = _strict_json(manifest_path)
    geometry_ids = {
        row.get("episode_id") for row in data.get("episodes", [])
        if isinstance(row, Mapping)
    }
    if geometry_ids != pilot_ids:
        raise RunnerHold("geometry root does not bind exactly the frozen pilot identities")
    if data.get("clean2000_payload_read") is True or data.get("protected_payload_read") is True:
        raise RunnerHold("geometry root declares protected/Clean2000 payload access")


def verify_required_contract_files(registry_root: Path, geometry_root: Path, geometry_allowlist: Path,
                                   predicate_contract: Path, semantic_contract: Path) -> dict[str, str]:
    paths = (registry_root, geometry_root, geometry_allowlist, predicate_contract, semantic_contract)
    if not all(path.exists() for path in paths):
        raise RunnerHold("required C1/C3 geometry contract root is not mounted")
    if not geometry_root.is_dir() or not geometry_allowlist.is_file():
        raise RunnerHold("geometry root/allowlist is not consumable")
    return {str(path): sha256_file(path) if path.is_file() else "DIRECTORY_SEAL_REQUIRED" for path in paths}


def preflight(manifest_path: Path, registry_root: Path, geometry_root: Path,
              geometry_allowlist: Path, predicate_contract: Path, semantic_contract: Path) -> dict[str, Any]:
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != "V23_DEV_PILOT_V1" or manifest.get("status") != "FROZEN_INPUT_BYTES_ONLY":
        raise RunnerHold("pilot input manifest is not frozen")
    if manifest.get("episode_count") != 40 or len(manifest.get("records", [])) != 40:
        raise RunnerHold("pilot manifest does not contain 40 episodes")
    pilot_ids = _pilot_episode_ids(manifest)
    bindings = verify_required_contract_files(registry_root, geometry_root, geometry_allowlist, predicate_contract, semantic_contract)
    _verify_registry_binding(registry_root)
    _verify_geometry_binding(geometry_root, geometry_allowlist, pilot_ids)
    contract = load_contract(predicate_contract)
    return {
        "schema": RUNNER_SCHEMA,
        "status": "PREFLIGHT_PASS_NO_EPISODE_EXECUTION",
        "manifest_sha256": sha256_file(manifest_path),
        "contract_bindings": bindings,
        "predicate_contract_schema": contract.get("schema"),
        "protected_payload_read": False,
        "model_inference": False,
        "rollout": False,
        "attack": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--geometry-allowlist", required=True, type=Path)
    parser.add_argument("--predicate-contract", required=True, type=Path)
    parser.add_argument("--semantic-contract", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = preflight(
            args.manifest, args.registry_root, args.geometry_root,
            args.geometry_allowlist, args.predicate_contract, args.semantic_contract,
        )
    except (RunnerHold, ContractError) as exc:
        print(json.dumps({
            "schema": RUNNER_SCHEMA,
            "status": "PREFLIGHT_HOLD",
            "reason": str(exc),
            "protected_payload_read": False,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        }, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
