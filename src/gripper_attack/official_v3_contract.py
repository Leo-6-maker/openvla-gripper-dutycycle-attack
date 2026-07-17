"""Fail-closed source contract for the Official V3 CLEAN corpus.

This module only audits CLEAN source evidence.  It never opens Teacher labels
and it never authorizes training or attack.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractViolation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SPLITS = {
    "FIT_TRAIN": range(0, 20),
    "FIT_DEV": range(20, 24),
    "CAL": range(24, 27),
    "CHECK": range(27, 30),
    "FINAL_EVAL_CANDIDATE": range(30, 50),
}
PROVENANCE_CLASSES = {"A_CURRENT_HEAD_CLEAN_START_VERIFIED", "B_PREVIOUS_HEAD_EQUIVALENT", "C_START_RECORD_MISSING", "D_DIRTY_START_QUARANTINE"}
PASS_STATUSES = {"PASS_FORMAL_CANDIDATE", "PASS_DATA_CONTRACT_PROVENANCE_HOLD"}
EMBEDDED_WORKER_MANIFEST_FILES = {"worker_start_manifest.json", "worker_start_manifest.json.sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def canonical_key(suite: str, task_idx: int, state_id: int) -> str:
    return f"{suite}/task_{int(task_idx):02d}/state_{int(state_id):02d}"


def expected_split(state_id: int) -> str:
    for split, states in SPLITS.items():
        if int(state_id) in states:
            return split
    raise ContractViolation("PROTOCOL", f"state outside official range: {state_id}")


def resolve_split(source_split: str, state_id: int, contract: dict[str, Any]) -> tuple[str, str]:
    """Map the collector's raw split to the frozen experiment split.

    V3 artifacts use the collector labels ``FIT``, ``CAL``, ``CHECK`` and
    ``FINAL_EVAL_CANDIDATE``.  The formal registry uses the finer-grained
    ``FIT_TRAIN``/``FIT_DEV`` labels.  Synthetic fixtures may already carry a
    formal label, so an exact formal label is accepted only when it agrees with
    the state-derived split.
    """
    raw = str(source_split)
    state = int(state_id)
    formal = expected_split(state)
    mapping = contract.get("source_split_mapping", {})
    if raw == formal:
        return raw, "FORMAL_SPLIT_EXACT"
    rules = mapping.get(raw)
    if not isinstance(rules, list):
        raise ContractViolation("IDENTITY", f"unknown source split: {raw!r}")
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        state_range = rule.get("state_range")
        if not isinstance(state_range, list) or len(state_range) != 2:
            continue
        if int(state_range[0]) <= state <= int(state_range[1]) and rule.get("formal_split") == formal:
            return formal, str(rule.get("rule", "CONFIGURED_SOURCE_SPLIT_MAPPING"))
    raise ContractViolation("IDENTITY", f"source split/state mismatch: split={raw!r}, state={state}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractViolation("PROTOCOL", f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractViolation("PROTOCOL", f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractViolation("PROTOCOL", f"cannot read stream: {path}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ContractViolation("PROTOCOL", f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ContractViolation("PROTOCOL", f"JSON object required at {path}:{line_no}")
        rows.append(value)
    return rows


def load_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("schema") != "OFFICIAL_V3_SOURCE_CONTRACT_V1":
        raise ContractViolation("PROTOCOL", "unexpected Official V3 source contract schema")
    if contract.get("status") != "PREPARATION_ONLY":
        raise ContractViolation("PROTOCOL", "source contract must remain preparation-only")
    if contract.get("num_tasks_per_suite") != 10 or contract.get("num_trials_per_task") != 50:
        raise ContractViolation("PROTOCOL", "official task/trial counts are not frozen")
    if contract.get("num_steps_wait") != 10 or contract.get("action_dimension") != 7:
        raise ContractViolation("PROTOCOL", "official wait/action contract is not frozen")
    if contract.get("official_horizons") != HORIZONS:
        raise ContractViolation("PROTOCOL", "official horizon map mismatch")
    names = contract.get("feature_names_25d")
    intent_names = contract.get("policy_intent_feature_names_9d")
    if not isinstance(names, list) or len(names) != 25 or not all(isinstance(v, str) for v in names):
        raise ContractViolation("PROTOCOL", "25D feature order is not frozen")
    if not isinstance(intent_names, list) or len(intent_names) != 9 or not all(isinstance(v, str) for v in intent_names):
        raise ContractViolation("PROTOCOL", "9D feature order is not frozen")
    if contract.get("feature_order_sha256") != json_sha(names) or contract.get("policy_intent_order_sha256") != json_sha(intent_names):
        raise ContractViolation("PROTOCOL", "feature order SHA mismatch")
    if contract.get("formal_training_authorized") is not False or contract.get("formal_attack_authorized") is not False:
        raise ContractViolation("PROTOCOL", "PR-A cannot authorize training or attack")
    mapping = contract.get("source_split_mapping")
    if not isinstance(mapping, dict):
        raise ContractViolation("PROTOCOL", "source split mapping is not frozen")
    for raw_split, rules in mapping.items():
        if not isinstance(raw_split, str) or not isinstance(rules, list) or not rules:
            raise ContractViolation("PROTOCOL", f"invalid source split mapping: {raw_split!r}")
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("formal_split"), str)
                or rule.get("formal_split") not in SPLITS
                or not isinstance(rule.get("state_range"), list)
                or len(rule["state_range"]) != 2
            ):
                raise ContractViolation("PROTOCOL", f"invalid source split mapping rule: {raw_split!r}")
    return contract


def load_external_manifest_registry(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Load a separately sealed worker-start registry.

    The registry is intentionally outside the artifact root.  It binds an
    exact canonical identity and artifact recursive SHA to an immutable
    worker-start manifest and its sidecar; it never edits or augments the
    original artifact.
    """
    path = path.resolve()
    if not path.is_file():
        raise ContractViolation("PROVENANCE", f"external manifest registry is missing: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(path)}  {path.name}":
        raise ContractViolation("PROVENANCE", "external manifest registry SHA sidecar is invalid")
    payload = load_json(path)
    if payload.get("schema") != "OFFICIAL_V3_EXTERNAL_WORKER_MANIFEST_REGISTRY_V1":
        raise ContractViolation("PROVENANCE", "unexpected external manifest registry schema")
    if payload.get("status") != "SEALED":
        raise ContractViolation("PROVENANCE", "external manifest registry is not sealed")
    if payload.get("formal_training_authorized") is not False or payload.get("formal_attack_authorized") is not False:
        raise ContractViolation("PROVENANCE", "external registry contains an authorization flag")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ContractViolation("PROVENANCE", "external manifest registry entries must be a list")
    result: dict[str, dict[str, Any]] = {}
    required = {
        "canonical_parent_key", "artifact_recursive_sha256", "worker_start_manifest_path",
        "worker_start_manifest_sha256", "worker_start_manifest_sidecar_sha256", "provenance_class",
    }
    for entry in entries:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ContractViolation("PROVENANCE", "external registry entry is incomplete")
        key = str(entry["canonical_parent_key"])
        if key in result:
            raise ContractViolation("PROVENANCE", f"duplicate external registry identity: {key}")
        try:
            parts = key.split("/")
            if len(parts) != 3 or parts[0] not in SUITES or canonical_key(parts[0], int(parts[1].split("_")[1]), int(parts[2].split("_")[1])) != key:
                raise ValueError
            state = int(parts[2].split("_")[1])
            if not 0 <= state < 50:
                raise ValueError
        except (IndexError, TypeError, ValueError):
            raise ContractViolation("PROVENANCE", f"invalid external registry identity: {key}")
        for name in ("artifact_recursive_sha256", "worker_start_manifest_sha256", "worker_start_manifest_sidecar_sha256"):
            if not isinstance(entry[name], str) or len(entry[name]) != 64:
                raise ContractViolation("PROVENANCE", f"invalid external registry SHA: {key}:{name}")
        if entry["provenance_class"] not in PROVENANCE_CLASSES:
            raise ContractViolation("PROVENANCE", f"invalid external provenance class: {key}")
        manifest_path = Path(str(entry["worker_start_manifest_path"])).resolve()
        sidecar_path = Path(str(entry.get("worker_start_manifest_sidecar_path", f"{manifest_path}.sha256"))).resolve()
        if not manifest_path.is_file() or not sidecar_path.is_file():
            raise ContractViolation("PROVENANCE", f"external worker manifest files are missing: {key}")
        if sha256_file(manifest_path) != entry["worker_start_manifest_sha256"]:
            raise ContractViolation("PROVENANCE", f"external worker manifest SHA mismatch: {key}")
        if sha256_file(sidecar_path) != entry["worker_start_manifest_sidecar_sha256"]:
            raise ContractViolation("PROVENANCE", f"external worker manifest sidecar SHA mismatch: {key}")
        if sidecar_path.read_text(encoding="utf-8").strip() != f"{entry['worker_start_manifest_sha256']}  {manifest_path.name}":
            raise ContractViolation("PROVENANCE", f"external worker manifest sidecar content mismatch: {key}")
        result[key] = {**entry, "worker_start_manifest_path": str(manifest_path), "worker_start_manifest_sidecar_path": str(sidecar_path)}
    return result, sha256_file(path)


def _finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in value)
    )


def _step_number(row: dict[str, Any], fallback: int) -> int:
    value = row.get("step", row.get("step_idx", fallback))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation("PROTOCOL", f"invalid step number: {value!r}") from exc


def _identity_matches(row: dict[str, Any], meta: dict[str, Any]) -> bool:
    for name in ("suite", "task_idx", "state_id", "canonical_parent_key"):
        if name not in row:
            continue
        left, right = row[name], meta.get(name)
        if name in {"task_idx", "state_id"}:
            try:
                left, right = int(left), int(right)
            except (TypeError, ValueError):
                return False
        if left != right:
            return False
    return True


def _vector(row: dict[str, Any], names: Iterable[str], length: int) -> list[float] | None:
    for name in names:
        if name in row:
            value = row[name]
            return [float(item) for item in value] if _finite_vector(value, length) else None
    return None


def _verify_checksum(root: Path, required_files: set[str]) -> str:
    manifest_path = root / "artifact_sha256.json"
    if not manifest_path.is_file():
        raise ContractViolation("CHECKSUM", "artifact_sha256.json is missing")
    payload = load_json(manifest_path)
    rows = payload.get("files")
    if not isinstance(rows, list) or payload.get("recursive_sha256") != json_sha(rows):
        raise ContractViolation("CHECKSUM", "artifact recursive checksum is invalid")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ContractViolation("CHECKSUM", "invalid artifact checksum row")
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() == "artifact_sha256.json":
            raise ContractViolation("CHECKSUM", f"unsafe checksum path: {row.get('path')}")
        name = rel.as_posix()
        if name in seen:
            raise ContractViolation("CHECKSUM", f"duplicate checksum path: {name}")
        seen.add(name)
        path = root / rel
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ContractViolation("CHECKSUM", f"checksum mismatch: {name}")
        if "size" in row and int(path.stat().st_size) != int(row["size"]):
            raise ContractViolation("CHECKSUM", f"size mismatch: {name}")
    required = set(required_files) - {"artifact_sha256.json"}
    if not required.issubset(seen):
        raise ContractViolation("CHECKSUM", f"required files omitted: {sorted(required - seen)}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"}
    if actual != seen:
        raise ContractViolation("CHECKSUM", f"checksum coverage mismatch: extra={sorted(actual - seen)} missing={sorted(seen - actual)}")
    return str(payload["recursive_sha256"])


def _check_provenance_class(provenance_class: Any, equivalence_status: str) -> None:
    if provenance_class not in PROVENANCE_CLASSES:
        raise ContractViolation("PROVENANCE", "unknown worker-start provenance class")
    if provenance_class == "B_PREVIOUS_HEAD_EQUIVALENT" and equivalence_status != "PASS":
        raise ContractViolation("PROVENANCE", "old-head equivalence decision is not PASS")
    if provenance_class in {"C_START_RECORD_MISSING", "D_DIRTY_START_QUARANTINE"}:
        raise ContractViolation("PROVENANCE", f"provenance class is quarantined: {provenance_class}")


def _verify_worker_manifest(
    root: Path,
    meta: dict[str, Any],
    contract: dict[str, Any],
    equivalence_status: str,
    *,
    artifact_recursive_sha256: str,
    external_registry: Mapping[str, dict[str, Any]] | None = None,
    external_registry_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = root / "worker_start_manifest.json"
    sidecar = root / "worker_start_manifest.json.sha256"
    local_manifest_present = manifest_path.is_file() or sidecar.is_file()
    if local_manifest_present:
        if not manifest_path.is_file() or not sidecar.is_file():
            raise ContractViolation("PROVENANCE", "embedded worker-start manifest pair is incomplete")
        expected_sidecar = f"{sha256_file(manifest_path)}  worker_start_manifest.json"
        if sidecar.read_text(encoding="utf-8").strip() != expected_sidecar:
            raise ContractViolation("PROVENANCE", "embedded worker-start manifest SHA mismatch")
        worker = load_json(manifest_path)
        required = set(contract.get("required_provenance_fields", []))
        missing = sorted(name for name in required if worker.get(name) in (None, ""))
        if missing:
            raise ContractViolation("PROVENANCE", f"worker-start fields missing: {missing}")
        if worker.get("worktree_clean") is not True:
            raise ContractViolation("PROVENANCE", "worker-start worktree was not clean")
        provenance_class = worker.get("provenance_class")
        _check_provenance_class(provenance_class, equivalence_status)
        comparisons = {
            "collector_head": "worker_start_git_head",
            "worker_script_sha256": "worker_start_script_sha256",
            "adapter_sha256": "worker_start_adapter_sha256",
            "protocol_sha256": "worker_start_protocol_sha256",
            "model_tree_sha256": "worker_start_model_tree_sha256",
            "processor_tree_sha256": "worker_start_processor_tokenizer_sha256",
        }
        for manifest_name, metadata_name in comparisons.items():
            if metadata_name in meta and meta[metadata_name] != worker.get(manifest_name):
                raise ContractViolation("PROVENANCE", f"artifact/worker-start mismatch: {manifest_name}")
        return {
            **worker,
            "provenance_class": provenance_class,
            "provenance_binding_mode": "LOCAL_EMBEDDED_MANIFEST",
            "worker_start_manifest_sha256": sha256_file(manifest_path),
            "worker_start_manifest_sidecar_sha256": sha256_file(sidecar),
            "external_manifest_registry_sha256": None,
        }

    if external_registry is None:
        raise ContractViolation("PROVENANCE", "worker-start manifest is absent and no external sealed registry was supplied")
    key = str(meta.get("canonical_parent_key", ""))
    entry = external_registry.get(key)
    if entry is None:
        raise ContractViolation("PROVENANCE", f"external worker-start registry has no exact identity: {key}")
    if entry.get("artifact_recursive_sha256") != artifact_recursive_sha256:
        raise ContractViolation("PROVENANCE", f"external worker-start registry artifact SHA mismatch: {key}")
    external_path = Path(str(entry["worker_start_manifest_path"]))
    external_sidecar = Path(str(entry["worker_start_manifest_sidecar_path"]))
    try:
        external_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ContractViolation("PROVENANCE", f"external manifest is inside artifact root: {key}")
    if sha256_file(external_path) != entry["worker_start_manifest_sha256"] or sha256_file(external_sidecar) != entry["worker_start_manifest_sidecar_sha256"]:
        raise ContractViolation("PROVENANCE", f"external worker-start registry file SHA mismatch: {key}")
    worker = load_json(external_path)
    if worker.get("schema") != "OFFICIAL_V3_WORKER_START_CONTRACT_V1":
        raise ContractViolation("PROVENANCE", f"external worker-start schema mismatch: {key}")
    required_external = set(contract.get("external_manifest_required_fields", []))
    missing = sorted(name for name in required_external if worker.get(name) in (None, ""))
    if missing:
        raise ContractViolation("PROVENANCE", f"external worker-start fields missing: {missing}")
    if worker.get("collector_worktree_clean") is not True:
        raise ContractViolation("PROVENANCE", "external worker-start worktree was not clean")
    provenance_class = entry.get("provenance_class")
    _check_provenance_class(provenance_class, equivalence_status)
    comparisons = {
        "worker_id": "collector_worker_id",
        "gpu_id": "collector_gpu",
        "pid": "collector_pid",
        "collector_head": "collector_git_head",
        "worker_script_sha256": "collector_script_sha256",
        "model_tree_sha256": "checkpoint_tree_sha256",
        "processor_tree_sha256": "processor_tokenizer_sha256",
        "collector_worktree_clean": "collector_worktree_clean",
    }
    for manifest_name, metadata_name in comparisons.items():
        if metadata_name not in meta:
            raise ContractViolation("PROVENANCE", f"artifact provenance field missing: {metadata_name}")
        left, right = meta[metadata_name], worker.get(manifest_name)
        if manifest_name in {"gpu_id", "pid"}:
            try:
                left, right = int(left), int(right)
            except (TypeError, ValueError):
                pass
        if left != right:
            raise ContractViolation("PROVENANCE", f"artifact/external worker-start mismatch: {manifest_name}")
    return {
        **worker,
        "adapter_sha256": worker["adapter_script_sha256"],
        "protocol_sha256": worker["protocol_config_sha256"],
        "provenance_class": provenance_class,
        "provenance_binding_mode": "EXTERNAL_SEALED_MANIFEST_REGISTRY",
        "worker_start_manifest_sha256": entry["worker_start_manifest_sha256"],
        "worker_start_manifest_sidecar_sha256": entry["worker_start_manifest_sidecar_sha256"],
        "external_manifest_registry_sha256": external_registry_sha256,
    }


def _verify_streams(root: Path, meta: dict[str, Any], contract: dict[str, Any], summary: dict[str, Any]) -> int:
    streams = {name: load_jsonl(root / name) for name in contract["stream_files"]}
    lengths = {name: len(rows) for name, rows in streams.items()}
    if len(set(lengths.values())) != 1 or not lengths["step_records.jsonl"]:
        raise ContractViolation("PROTOCOL", f"stream length mismatch: {lengths}")
    count = lengths["step_records.jsonl"]
    summary_count = summary.get("step_count", summary.get("steps"))
    if int(summary_count) != count:
        raise ContractViolation("PROTOCOL", "summary step count mismatch")
    names = contract["feature_names_25d"]
    intent_names = contract["policy_intent_feature_names_9d"]
    for index, (step, policy, sidecar) in enumerate(zip(streams["step_records.jsonl"], streams["policy_intent_records.jsonl"], streams["privileged_teacher_sidecar.jsonl"])):
        for row in (step, policy, sidecar):
            if _step_number(row, index) != index or not _identity_matches(row, meta):
                raise ContractViolation("PROTOCOL", f"step identity/index mismatch at {index}")
        features = _vector(step, ("features_25d",), 25)
        intent = _vector(policy, ("clean_policy_intent_9d",), 9)
        if features is None or intent is None:
            raise ContractViolation("PROTOCOL", f"student vector invalid at {index}")
        if step.get("feature_names_25d", names) != names or policy.get("policy_intent_feature_names_9d", intent_names) != intent_names:
            raise ContractViolation("PROTOCOL", f"feature order mismatch at {index}")
        raw = _vector(step, ("clean_action_raw_7d", "action_raw_7d", "action_raw"), 7)
        env = _vector(step, ("applied_action_7d", "action_env", "env_action_7d"), 7)
        if raw is None or env is None:
            raise ContractViolation("PROTOCOL", f"7D action missing at {index}")
        for row in (step, policy):
            if row.get("generation_passes_per_step") != 1 or row.get("single_generation_parity_pass") is not True:
                raise ContractViolation("GENERATION", f"measured generation count is not one at {index}")
            if row.get("score_adapter_parity_pass") is not True:
                raise ContractViolation("GENERATION", f"score/action parity missing at {index}")
        tokens = step.get("action_token_ids")
        scores = step.get("score_head_summary")
        if not isinstance(tokens, list) or len(tokens) != contract["action_dimension"] or not isinstance(scores, list) or len(scores) != contract["action_dimension"]:
            raise ContractViolation("GENERATION", f"token/score telemetry invalid at {index}")
        if policy.get("action_token_ids") != tokens:
            raise ContractViolation("GENERATION", f"policy/action token mismatch at {index}")
        eef = sidecar.get("robot0_eef_pos")
        qpos = sidecar.get("robot0_gripper_qpos")
        if not _finite_vector(eef, 3) or not _finite_vector(qpos, 2):
            raise ContractViolation("PROTOCOL", f"robot sidecar invalid at {index}")
        if "robot0_eef_pos" in step and _finite_vector(step["robot0_eef_pos"], 3):
            if max(abs(float(a) - float(b)) for a, b in zip(step["robot0_eef_pos"], eef)) > float(contract["parity_tolerances"]["eef_m"]):
                raise ContractViolation("PROTOCOL", f"EEF parity mismatch at {index}")
        if "robot0_gripper_qpos" in step and _finite_vector(step["robot0_gripper_qpos"], 2):
            if max(abs(float(a) - float(b)) for a, b in zip(step["robot0_gripper_qpos"], qpos)) > float(contract["parity_tolerances"]["qpos"]):
                raise ContractViolation("PROTOCOL", f"qpos parity mismatch at {index}")
    return count


def _verify_25d_streams(root: Path, meta: dict[str, Any], contract: dict[str, Any], summary: dict[str, Any]) -> int:
    """Verify the robot/action evidence needed by the 25D-only S1 path.

    This deliberately does not open policy_intent_records.jsonl.  The 9D
    stream is an optional, physically separate ablation export; it is not a
    prerequisite for the robot-centric 25D Teacher.
    """
    step_rows = load_jsonl(root / "step_records.jsonl")
    sidecar_rows = load_jsonl(root / "privileged_teacher_sidecar.jsonl")
    if not step_rows or len(step_rows) != len(sidecar_rows):
        raise ContractViolation("PROTOCOL", "25D source stream length mismatch")
    count = len(step_rows)
    summary_count = summary.get("step_count", summary.get("steps"))
    if int(summary_count) != count:
        raise ContractViolation("PROTOCOL", "summary step count mismatch")
    names = contract["feature_names_25d"]
    for index, (step, sidecar) in enumerate(zip(step_rows, sidecar_rows)):
        for row in (step, sidecar):
            if _step_number(row, index) != index or not _identity_matches(row, meta):
                raise ContractViolation("PROTOCOL", f"25D step identity/index mismatch at {index}")
        features = _vector(step, ("features_25d",), 25)
        raw = _vector(step, ("clean_action_raw_7d", "action_raw_7d", "action_raw"), 7)
        env = _vector(step, ("applied_action_7d", "action_env", "env_action_7d"), 7)
        if features is None or raw is None or env is None:
            raise ContractViolation("PROTOCOL", f"25D source vector/action invalid at {index}")
        if step.get("feature_names_25d", names) != names:
            raise ContractViolation("PROTOCOL", f"25D feature order mismatch at {index}")
        if step.get("generation_passes_per_step") != 1 or step.get("single_generation_parity_pass") is not True:
            raise ContractViolation("GENERATION", f"measured generation count is not one at {index}")
        if step.get("score_adapter_parity_pass") is not True:
            raise ContractViolation("GENERATION", f"25D execution telemetry parity missing at {index}")
        tokens = step.get("action_token_ids")
        scores = step.get("score_head_summary")
        if not isinstance(tokens, list) or len(tokens) != contract["action_dimension"] or not isinstance(scores, list) or len(scores) != contract["action_dimension"]:
            raise ContractViolation("GENERATION", f"25D token/score telemetry invalid at {index}")
        eef = sidecar.get("robot0_eef_pos")
        qpos = sidecar.get("robot0_gripper_qpos")
        if not _finite_vector(eef, 3) or not _finite_vector(qpos, 2):
            raise ContractViolation("PROTOCOL", f"robot sidecar invalid at {index}")
        if "robot0_eef_pos" in step and _finite_vector(step["robot0_eef_pos"], 3):
            if max(abs(float(a) - float(b)) for a, b in zip(step["robot0_eef_pos"], eef)) > float(contract["parity_tolerances"]["eef_m"]):
                raise ContractViolation("PROTOCOL", f"EEF parity mismatch at {index}")
        if "robot0_gripper_qpos" in step and _finite_vector(step["robot0_gripper_qpos"], 2):
            if max(abs(float(a) - float(b)) for a, b in zip(step["robot0_gripper_qpos"], qpos)) > float(contract["parity_tolerances"]["qpos"]):
                raise ContractViolation("PROTOCOL", f"qpos parity mismatch at {index}")
    return count


def verify_artifact(
    artifact_root: Path,
    contract: dict[str, Any],
    *,
    equivalence_status: str = "HOLD",
    mode: str = "full",
    external_registry: Mapping[str, dict[str, Any]] | None = None,
    external_registry_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in {"full", "25d"}:
        raise ContractViolation("PROTOCOL", f"unknown artifact audit mode: {mode}")
    root = artifact_root.resolve()
    required_files = set(contract["required_files"])
    required_files -= EMBEDDED_WORKER_MANIFEST_FILES
    if mode == "25d":
        required_files.discard("policy_intent_records.jsonl")
    missing = sorted(name for name in required_files if not (root / name).is_file())
    if missing:
        raise ContractViolation("PROTOCOL", f"required artifact files missing: {missing}")
    meta = load_json(root / "episode_metadata.json")
    runtime = load_json(root / "runtime_audit.json")
    summary = load_json(root / "episode_summary.json")
    condition = load_json(root / "condition_config.json")
    attack = load_json(root / "attack_config.json")
    if meta.get("schema") not in contract["source_artifact_schemas"] or meta.get("condition") != "CLEAN":
        raise ContractViolation("PROTOCOL", "artifact is not an allowed Official CLEAN source")
    if meta.get("runtime_valid") is not True or runtime.get("runtime_valid") is not True:
        raise ContractViolation("RUNTIME", "runtime_valid is not true")
    suite, task, state = meta.get("suite"), meta.get("task_idx"), meta.get("state_id")
    if suite not in SUITES or not isinstance(task, int) or not 0 <= task < 10 or not isinstance(state, int) or not 0 <= state < 50:
        raise ContractViolation("PROTOCOL", "invalid suite/task/state identity")
    key = canonical_key(suite, task, state)
    source_split_raw = meta.get("split")
    formal_split, split_mapping_rule = resolve_split(source_split_raw, state, contract)
    if meta.get("canonical_parent_key") != key:
        raise ContractViolation("IDENTITY", "canonical identity or split mismatch")
    if meta.get("official_horizon") != HORIZONS[suite] or runtime.get("official_horizon") != HORIZONS[suite] or meta.get("num_steps_wait") != 10:
        raise ContractViolation("PROTOCOL", "official horizon/wait mismatch")
    if attack.get("attack_enabled") is not False or condition.get("condition") != "CLEAN":
        raise ContractViolation("PROTOCOL", "CLEAN artifact has attack enabled")
    if not isinstance(meta.get("success"), bool) or meta.get("env_success") != meta.get("success"):
        raise ContractViolation("PROTOCOL", "task success semantics are inconsistent")
    if meta.get("official_execution_adapter") != "OfficialOpenVLAActionAdapter.predict_action":
        raise ContractViolation("PROTOCOL", "official action adapter mismatch")
    if meta.get("generation_passes_per_step") != 1 or runtime.get("generation_passes_per_step") != 1:
        raise ContractViolation("GENERATION", "episode generation contract is not one")
    if meta.get("feature_names_25d") != contract["feature_names_25d"] or (
        mode == "full" and meta.get("policy_intent_feature_names_9d") != contract["policy_intent_feature_names_9d"]
    ):
        raise ContractViolation("PROTOCOL", "metadata feature order mismatch")
    for name in ("initial_state_sha256", "model_tree_sha256", "processor_tokenizer_sha256", "protocol_sha256"):
        if not isinstance(meta.get(name), str) or len(meta[name]) != 64:
            raise ContractViolation("PROVENANCE", f"metadata SHA missing: {name}")
    if (root / "teacher_retention_records.jsonl").exists() or (root / "retention_events.json").exists():
        raise ContractViolation("PROTOCOL", "Teacher/event output is present in source artifact")
    recursive_sha = _verify_checksum(root, required_files)
    step_count = _verify_streams(root, meta, contract, summary) if mode == "full" else _verify_25d_streams(root, meta, contract, summary)
    worker = _verify_worker_manifest(
        root,
        meta,
        contract,
        equivalence_status,
        artifact_recursive_sha256=recursive_sha,
        external_registry=external_registry,
        external_registry_sha256=external_registry_sha256,
    )
    formal = worker["provenance_class"] == "A_CURRENT_HEAD_CLEAN_START_VERIFIED" or (
        worker["provenance_class"] == "B_PREVIOUS_HEAD_EQUIVALENT" and equivalence_status == "PASS"
    )
    binding_sha = json_sha({
        "canonical_parent_key": key,
        "artifact_recursive_sha256": recursive_sha,
        "provenance_class": worker["provenance_class"],
        "provenance_binding_mode": worker["provenance_binding_mode"],
        "worker_start_manifest_sha256": worker["worker_start_manifest_sha256"],
        "worker_start_manifest_sidecar_sha256": worker["worker_start_manifest_sidecar_sha256"],
        "external_manifest_registry_sha256": worker["external_manifest_registry_sha256"],
    })
    return {
        "schema": "OFFICIAL_V3_ARTIFACT_AUDIT_V1",
        "status": "PASS_FORMAL_CANDIDATE" if formal else "PASS_DATA_CONTRACT_PROVENANCE_HOLD",
        "canonical_parent_key": key,
        "suite": suite,
        "task_idx": task,
        "state_id": state,
        "split": formal_split,
        "source_split_raw": source_split_raw,
        "split_mapping_rule": split_mapping_rule,
        "task_success": bool(meta["success"]),
        "artifact_root": str(root),
        "artifact_recursive_sha256": recursive_sha,
        "step_count": step_count,
        "provenance_class": worker["provenance_class"],
        "worker_id": worker["slot_id"],
        "gpu_id": worker["gpu_id"],
        "worker_start_manifest_sha256": worker["worker_start_manifest_sha256"],
        "worker_start_manifest_sidecar_sha256": worker["worker_start_manifest_sidecar_sha256"],
        "provenance_binding_mode": worker["provenance_binding_mode"],
        "external_manifest_registry_sha256": worker["external_manifest_registry_sha256"],
        "provenance_binding_sha256": binding_sha,
        "collector_head": worker["collector_head"],
        "worker_script_sha256": worker["worker_script_sha256"],
        "adapter_sha256": worker["adapter_sha256"],
        "protocol_sha256": worker["protocol_sha256"],
        "model_tree_sha256": worker["model_tree_sha256"],
        "processor_tree_sha256": worker["processor_tree_sha256"],
        "formal_eligible": formal,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "audit_mode": mode,
    }


def audit_artifact(
    artifact_root: Path,
    contract: dict[str, Any],
    *,
    equivalence_status: str = "HOLD",
    mode: str = "full",
    external_registry: Mapping[str, dict[str, Any]] | None = None,
    external_registry_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        return verify_artifact(
            artifact_root,
            contract,
            equivalence_status=equivalence_status,
            mode=mode,
            external_registry=external_registry,
            external_registry_sha256=external_registry_sha256,
        )
    except ContractViolation as exc:
        status = {
            "CHECKSUM": "HOLD_CHECKSUM",
            "GENERATION": "HOLD_GENERATION",
            "IDENTITY": "HOLD_DUPLICATE_OR_IDENTITY",
            "PROVENANCE": "HOLD_PROVENANCE",
            "RUNTIME": "HOLD_RUNTIME",
        }.get(exc.code, "HOLD_PROTOCOL")
        return {
            "schema": "OFFICIAL_V3_ARTIFACT_AUDIT_V1",
            "status": status,
            "artifact_root": str(artifact_root.resolve()),
            "formal_eligible": False,
            "error_code": exc.code,
            "error": str(exc),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
            "audit_mode": mode,
        }


__all__ = [
    "ContractViolation", "SUITES", "HORIZONS", "SPLITS", "PROVENANCE_CLASSES", "PASS_STATUSES",
    "audit_artifact", "canonical_key", "expected_split", "resolve_split", "json_sha", "load_contract",
    "load_external_manifest_registry", "sha256_file",
]
