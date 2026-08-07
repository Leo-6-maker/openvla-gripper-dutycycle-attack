"""Materialize one immutable Stage V R2 plan from real receipts."""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

try:
    from ..detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file
except ImportError:  # direct server execution
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file


REGISTRY_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_REGISTRY_V2"
PLAN_SCHEMA = "STAGE_V_R2_ORCHESTRATOR_PLAN_V2"
STAGE_SPEC_SCHEMA = "STAGE_V_R2_STAGE_SPEC_V1"


def _write_sha(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fd, temp_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=str(sidecar.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(f"{digest}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, sidecar)
        if os.name != "nt":
            directory_fd = os.open(sidecar.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return digest


def _json_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: Path, label: str) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}_INVALID:{path}")
    return value


def _binding(name: str, path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name}_MISSING:{path}")
    return {"name": name, "path": str(path), "sha256": sha256_file(path)}


def _latest_registry(state_root: Path) -> tuple[Path | None, Mapping[str, Any] | None]:
    candidates = sorted(state_root.glob("PLAN_REGISTRY_V*.json"))
    if not candidates:
        return None, None
    rows: list[tuple[int, Path, Mapping[str, Any]]] = []
    for path in candidates:
        value = _load(path, "REGISTRY")
        try:
            version = int(value["version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"REGISTRY_VERSION_INVALID:{path}") from exc
        rows.append((version, path, value))
    rows.sort(key=lambda item: item[0])
    return rows[-1][1], rows[-1][2]


def append_registry(
    *,
    state_root: Path,
    source_commit: str,
    source_tree: str,
    stage: str,
    plan_path: Path,
    upstream_receipts: list[Mapping[str, Any]],
) -> Path:
    """Append exactly one registry version; never rewrite an older version."""
    state_root = state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    previous_path, previous = _latest_registry(state_root)
    previous_version = int(previous["version"]) if previous else 0
    version = previous_version + 1
    if previous and stage in {str(item.get("stage")) for item in previous.get("plans", []) if isinstance(item, Mapping)}:
        raise ValueError(f"STAGE_ALREADY_REGISTERED:{stage}")
    plan_path = plan_path.resolve()
    plan_sha = sha256_file(plan_path)
    plans = [dict(item) for item in (previous.get("plans", []) if previous else []) if isinstance(item, Mapping)]
    plans.append({"stage": stage, "path": str(plan_path), "sha256": plan_sha})
    registry = {
        "schema": REGISTRY_SCHEMA,
        "version": version,
        "previous_registry_path": str(previous_path.resolve()) if previous_path else None,
        "previous_registry_sha256": sha256_file(previous_path) if previous_path else None,
        "newly_added_stage": stage,
        "new_plan_path": str(plan_path),
        "new_plan_sha256": plan_sha,
        "plans": plans,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "upstream_receipts": [dict(item) for item in upstream_receipts],
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    path = state_root / f"PLAN_REGISTRY_V{version:04d}.json"
    if path.exists():
        raise ValueError(f"REGISTRY_VERSION_ALREADY_EXISTS:{path}")
    atomic_write_json(path, registry)
    _write_sha(path)
    return path


def _candidate_rows(candidate_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = candidate_manifest.get("selected_parents")
    if not isinstance(rows, list):
        rows = candidate_manifest.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("CANDIDATE_ROWS_MISSING")
    result = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(result) != len(rows) or len({str(row.get("canonical_parent_key")) for row in result}) != len(result):
        raise ValueError("CANDIDATE_PARENT_IDENTITY_INVALID")
    return result


def build_c0_plan(
    *,
    repo_root: Path,
    state_root: Path,
    qualification_root: Path,
    candidate_manifest: Path,
    science_provenance: Path,
    source_commit: str,
    source_tree: str,
    python_executable: str,
    external_pid: int,
    allow_gpu5: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    """Build the C0 plan and its fresh diagnostic parent manifest."""
    qualification_root = qualification_root.resolve()
    q2_mode = (qualification_root / "Q2_CONTROL_QUALIFICATION_REPORT.json").is_file()
    report_path = qualification_root / ("Q2_CONTROL_QUALIFICATION_REPORT.json" if q2_mode else "CONTROL_QUALIFICATION_REPORT.json")
    audit_path = qualification_root / ("Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json" if q2_mode else "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json")
    formal_path = qualification_root / ("Q2_PARENT_MANIFEST_A.json" if q2_mode else "STAGE_V_R2_PARENT_MANIFEST_A.json")
    for path in (report_path, audit_path, formal_path, candidate_manifest, science_provenance):
        if not path.is_file():
            raise ValueError(f"C0_INPUT_MISSING:{path}")
    report = _load(report_path, "QUALIFICATION_REPORT")
    audit = _load(audit_path, "QUALIFICATION_AUDIT")
    formal = _load(formal_path, "QUALIFICATION_MANIFEST")
    candidate = _load(candidate_manifest.resolve(), "CANDIDATE_MANIFEST")
    if report.get("status") != "PASS" or audit.get("verdict") != "PASS":
        raise ValueError("QUALIFICATION_NOT_PASS")
    if report.get("source_commit") != formal.get("source_commit") or report.get("source_tree") != formal.get("source_tree"):
        raise ValueError("QUALIFICATION_SOURCE_MISMATCH")
    if formal.get("status") not in {"PASS", "FROZEN"} or int(formal.get("selected_count", -1)) != 40:
        raise ValueError("QUALIFICATION_MANIFEST_NOT_40")
    candidate_boundary_fail = (
        candidate.get("old_artifacts_reused") is not False or candidate.get("source_artifacts_modified") is not False
    ) if not q2_mode else (
        candidate.get("schema") != "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1"
        or any(candidate.get("gates", {}).get(field, 1) != 0 for field in ("eval160_reads", "protected_eval_reads", "attack_rollouts"))
    )
    if candidate_boundary_fail:
        raise ValueError("CANDIDATE_OLD_ARTIFACT_BOUNDARY_FAIL")
    boundary_fields = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts") if q2_mode else ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")
    if any(int(report.get(field, -1)) != 0 for field in boundary_fields):
        raise ValueError("QUALIFICATION_BOUNDARY_FAIL")
    candidates = _candidate_rows(candidate)
    formal_rows = formal.get("selected_parents")
    if not isinstance(formal_rows, list) or len(formal_rows) != 40:
        raise ValueError("FORMAL_PARENT_ROWS_INVALID")
    formal_keys = {str(row.get("canonical_parent_key")) for row in formal_rows if isinstance(row, Mapping)}
    if len(formal_keys) != 40:
        raise ValueError("FORMAL_PARENT_KEYS_INVALID")
    def diagnostic_rank(row: Mapping[str, Any]) -> tuple[str, str]:
        rank = str(row.get("qualification_rank_sha256", ""))
        if q2_mode and not rank:
            rank = hashlib.sha256(f"STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807::{row.get('canonical_parent_key')}".encode()).hexdigest()
        return rank, str(row.get("canonical_parent_key", ""))

    diagnostic = sorted(
        (row for row in candidates if str(row.get("canonical_parent_key")) not in formal_keys),
        key=diagnostic_rank,
    )[:8]
    if len(diagnostic) != 8:
        raise ValueError("C0_DIAGNOSTIC_POOL_UNDERFLOW")
    diagnostic_keys = [str(row["canonical_parent_key"]) for row in diagnostic]
    if len(set(diagnostic_keys)) != 8 or set(diagnostic_keys) & formal_keys:
        raise ValueError("C0_DIAGNOSTIC_IDENTITY_FAIL")

    root = state_root.resolve()
    manifest_path = root / "C0_DIAGNOSTIC_PARENT_MANIFEST.json"
    manifest = {
        "schema": "STAGE_V_R2_DYNAMIC8_CONTROL_CANARY_PARENT_MANIFEST_V1",
        "status": "FROZEN",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "qualification_source_commit": formal.get("source_commit"),
        "qualification_source_tree": formal.get("source_tree"),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "formal_parent_manifest_sha256": sha256_file(formal_path),
        "selection_rule": "hash-order candidates excluding frozen formal 40; no vulnerability outcome input",
        "diagnostic_only": True,
        "old_artifacts_reused": False,
        "source_artifact_read": False,
        "selected_count": 8,
        "selected_parents": diagnostic,
        "parents": diagnostic,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
    }
    if manifest_path.exists():
        if sha256_file(manifest_path) != _json_sha(manifest):
            raise ValueError("C0_DIAGNOSTIC_MANIFEST_ALREADY_EXISTS")
    else:
        atomic_write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    runner = repo_root.resolve() / "scripts/detector_v5/run_stage_v_dynamic_control_canary.py"
    auditor = repo_root.resolve() / "scripts/detector_v5/audit_stage_v_dynamic_queue.py"
    for path in (runner, auditor):
        if not path.is_file():
            raise ValueError(f"C0_TOOL_MISSING:{path}")
    config_path = root / "C0_CONFIG.json"
    config = {
        "schema": "STAGE_V_R2_C0_CONFIG_V1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "diagnostic_parent_manifest": str(manifest_path),
        "diagnostic_parent_manifest_sha256": manifest_sha,
        "qualification_report": _binding("qualification_report", report_path),
        "qualification_audit": _binding("qualification_audit", audit_path),
        "boundaries": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0},
        "resource_policy": {
            "resource_kind": "GPU", "required_gpu_count": 8, "strict_gpu_count": True,
            "excluded_gpus": [] if allow_gpu5 else [5], "gpu5_authorized": bool(allow_gpu5),
        },
    }
    if not config_path.exists():
        atomic_write_json(config_path, config)
    elif sha256_file(config_path) != _json_sha(config):
        raise ValueError("C0_CONFIG_ALREADY_EXISTS")

    state_root.mkdir(parents=True, exist_ok=True)
    queue_db = root / "C0_CONTROL_CANARY.sqlite"
    preflight = root / "C0_PREFLIGHT.json"
    project_lock = root.parent / ".stage_v_r2_c0.lock"
    plan_path = root / "plans/C0_PLAN_V0001.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    receipts = [_binding(name, path) for name, path in (
        ("qualification_report", report_path),
        ("qualification_audit", audit_path),
        ("formal_parent_manifest", formal_path),
        ("candidate_manifest", candidate_manifest),
        ("science_provenance", science_provenance),
    )]
    plan = {
        "schema": PLAN_SCHEMA,
        "stage": "C0",
        "plan_version": 1,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "auditor_path": str(auditor),
        "auditor_sha256": sha256_file(auditor),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cwd": str(repo_root.resolve()),
        "python_executable": python_executable,
        "input_receipts": receipts,
        "parent_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "output_root_template": str(root.parent / "DYNAMIC8_CONTROL_CANARY_{commit8}_{utc}"),
        "command_template": [
            python_executable, str(runner), "--run-root", "{output_root}", "--repo-root", str(repo_root.resolve()),
            "--parent-manifest", "{parent_manifest}", "--queue-db", str(queue_db), "--run-id", "stage-v-r2-c0-{source_commit}",
            "--source-commit", "{source_commit}", "--source-tree", "{source_tree}", "--preflight-file", str(preflight),
            "--lock-path", str(project_lock), "--approved-gpus", "{approved_gpus}", "--external-pid", str(external_pid),
            "--canary-peak-mib", "0",
        ],
        "audit_command_template": [
            python_executable, str(auditor), "--run-root", "{output_root}", "--parent-manifest", "{parent_manifest}",
            "--queue-db", str(queue_db), "--run-id", "stage-v-r2-c0-{source_commit}", "--expected-parent-count", "8",
            "--expected-source-commit", "{source_commit}", "--expected-source-tree", "{source_tree}",
        ],
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        "resource_policy": {
            "resource_kind": "GPU", "required_gpu_count": 8, "minimum_gpu_count": 8, "maximum_gpu_count": 8,
            "strict_gpu_count": True, "excluded_gpus": [] if allow_gpu5 else [5],
            "gpu5_authorized": bool(allow_gpu5), "protected_pids": [external_pid], "canary_peak_mib": 0,
        },
        "gpu_policy": {
            "required_count": 8, "excluded_gpus": [] if allow_gpu5 else [5],
            "gpu5_authorized": bool(allow_gpu5), "protected_pids": [external_pid], "canary_peak_mib": 0,
        },
        "completion_receipts": ["DYNAMIC8_CONTROL_CANARY_REPORT.json", "DYNAMIC8_CONTROL_CANARY_AUDIT.json"],
        "lock_path": str(project_lock),
        "forbidden_boundary_contract": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0},
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    if plan_path.exists():
        if sha256_file(plan_path) != _json_sha(plan):
            raise ValueError("C0_PLAN_ALREADY_EXISTS")
    else:
        atomic_write_json(plan_path, plan)
        _write_sha(plan_path)
    return plan, plan_path, manifest_path


def build_stage_plan_from_spec(
    *,
    stage: str,
    spec_path: Path,
    state_root: Path,
    source_commit: str,
    source_tree: str,
) -> tuple[dict[str, Any], Path]:
    """Materialize a later-stage plan from a SHA-bound deployment spec."""
    spec_path = spec_path.resolve()
    spec = _load(spec_path, f"{stage}_SPEC")
    if spec.get("schema") != STAGE_SPEC_SCHEMA or spec.get("stage") != stage:
        raise ValueError(f"{stage}_SPEC_SCHEMA_INVALID")
    if spec.get("source_commit") != source_commit or spec.get("source_tree") != source_tree:
        raise ValueError(f"{stage}_SPEC_SOURCE_MISMATCH")
    for field in ("runner_path", "auditor_path", "config_path"):
        value = spec.get(field)
        expected = spec.get(f"{field}_sha256")
        path = Path(str(value)).resolve() if value else Path("")
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"{stage}_{field.upper()}_BINDING_INVALID")
    parent = spec.get("parent_manifest")
    if not isinstance(parent, Mapping):
        raise ValueError(f"{stage}_PARENT_MANIFEST_MISSING")
    parent_path = Path(str(parent.get("path", ""))).resolve()
    if not parent_path.is_file() or sha256_file(parent_path) != parent.get("sha256"):
        raise ValueError(f"{stage}_PARENT_MANIFEST_BINDING_INVALID")
    receipts = spec.get("input_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError(f"{stage}_INPUT_RECEIPTS_MISSING")
    normalized_receipts: list[dict[str, Any]] = [{
        "name": "stage_spec", "path": str(spec_path), "sha256": sha256_file(spec_path),
    }]
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            raise ValueError(f"{stage}_INPUT_RECEIPT_INVALID:{index}")
        receipt_path = Path(str(receipt.get("path", ""))).resolve()
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt.get("sha256"):
            raise ValueError(f"{stage}_INPUT_RECEIPT_BINDING_INVALID:{index}")
        normalized_receipts.append(dict(receipt))
    for field in ("command_template", "audit_command_template", "completion_receipts"):
        if not isinstance(spec.get(field), list) or not spec[field]:
            raise ValueError(f"{stage}_{field.upper()}_MISSING")
        if field != "completion_receipts" and not all(isinstance(item, str) and item for item in spec[field]):
            raise ValueError(f"{stage}_{field.upper()}_INVALID")
    cwd = Path(str(spec.get("cwd", ""))).resolve()
    output_template = str(spec.get("output_root_template", ""))
    if not cwd.is_dir() or not output_template:
        raise ValueError(f"{stage}_PATH_BINDING_INVALID")
    policy = dict(spec.get("resource_policy") or {})
    policy.setdefault("resource_kind", "CPU_ONLY")
    policy.setdefault("required_gpu_count", 0)
    policy.setdefault("minimum_gpu_count", policy["required_gpu_count"])
    policy.setdefault("maximum_gpu_count", policy["required_gpu_count"])
    policy.setdefault("strict_gpu_count", bool(policy["required_gpu_count"]))
    policy.setdefault("excluded_gpus", [])
    policy.setdefault("protected_pids", [])
    policy.setdefault("canary_peak_mib", 0)
    plan = {
        "schema": PLAN_SCHEMA,
        "stage": stage,
        "plan_version": 1,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runner_path": str(Path(str(spec["runner_path"])).resolve()),
        "runner_sha256": spec["runner_path_sha256"],
        "auditor_path": str(Path(str(spec["auditor_path"])).resolve()),
        "auditor_sha256": spec["auditor_path_sha256"],
        "config_path": str(Path(str(spec["config_path"])).resolve()),
        "config_sha256": spec["config_path_sha256"],
        "cwd": str(cwd),
        "python_executable": spec.get("python_executable"),
        "input_receipts": normalized_receipts,
        "parent_manifest": {"path": str(parent_path), "sha256": parent["sha256"]},
        "output_root_template": output_template,
        "command_template": list(spec["command_template"]),
        "audit_command_template": list(spec["audit_command_template"]),
        "completion_receipts": list(spec["completion_receipts"]),
        "decision_receipt_names": list(spec.get("decision_receipt_names", [])),
        "env": dict(spec.get("env") or {}),
        "resource_policy": policy,
        "gpu_policy": dict(spec.get("gpu_policy") or policy),
        "lock_path": str(Path(str(spec.get("lock_path", state_root / f".{stage.lower()}.lock"))).resolve()),
        "forbidden_boundary_contract": dict(spec.get("forbidden_boundary_contract") or {
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
        }),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "created_utc": spec.get("created_utc") or _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }
    plan_dir = state_root.resolve() / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{stage}_PLAN_V0001.json"
    if plan_path.exists():
        existing = _load(plan_path, f"{stage}_PLAN")
        if _json_sha(existing) != _json_sha(plan):
            raise ValueError(f"{stage}_PLAN_ALREADY_EXISTS_DIFFERENT")
    else:
        atomic_write_json(plan_path, plan)
        _write_sha(plan_path)
    return plan, plan_path


if __name__ == "__main__":
    raise SystemExit("Use run_stage_v_r2_plan_controller.py")
