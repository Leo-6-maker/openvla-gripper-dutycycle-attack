"""Freeze the downstream FIT670 -> Teacher permission transition.

This is metadata-only.  It never opens an episode payload and never runs a
model.  The original collection transition remains immutable and keeps its
teacher_labels_authorized=false permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
PERMISSIONS = {
    "fit_episode_read": True,
    "teacher_label_generation": True,
    "student_dataset_generation": False,
    "student_training": False,
    "detector_load": False,
    "rollout": False,
    "shadow": False,
    "attack": False,
    "protected_payload_read": False,
    "CAL_READ": False,
    "CHECK_READ": False,
    "G10_READ": False,
    "T2R_D_READ": False,
}


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(value: Any, field: str, length: int = 64) -> str:
    value = str(value)
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{field} is not a lowercase SHA")
    return value


def _safe_root(path: Path, *, must_exist: bool = True) -> Path:
    resolved = path.resolve()
    if any(part.lower() in FORBIDDEN_PARTS for part in resolved.parts):
        raise ValueError(f"forbidden-looking path: {resolved}")
    if must_exist and (not resolved.is_dir() or resolved.is_symlink()):
        raise ValueError(f"missing or symlinked root: {resolved}")
    return resolved


def _write_seal(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    if not files:
        raise ValueError("cannot seal empty transition")
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _binding_digest(bindings: Mapping[str, Any]) -> str:
    rows = []
    for identity in sorted(bindings):
        row = bindings[identity]
        if not isinstance(row, Mapping) or row.get("episode_id", identity) != identity:
            raise ValueError(f"episode binding identity mismatch: {identity}")
        required = ("suite", "task_id", "task_name", "state_id", "seed", "episode_id", "initial_state_sha256", "relative_path", "episode_sha256", "episode_sha256sums_sha256", "worker_id", "shard_id", "worker_result_target", "worker_result_steps", "worker_result_source_sha256", "worker_result_episode_sha256sums_sha256", "worker_result_initial_state_sha256", "worker_result_binding_mode", "worker_manifest_sha256", "worker_seal_sha256sums_sha256", "collection_source_commit", "collection_source_tree", "collector_script_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256", "allowlist_sha256", "c1_canonical_digest", "schema")
        if any(key not in row or row[key] in (None, "") for key in required):
            raise ValueError(f"episode binding incomplete: {identity}")
        rows.append({key: row[key] for key in required})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity_digest(bindings: Mapping[str, Any]) -> str:
    rows = []
    for identity in sorted(bindings):
        row = bindings[identity]
        rows.append({"episode_id": row["episode_id"], "suite": row["suite"], "task_id": row["task_id"], "state_id": row["state_id"], "collection_seed": row["seed"], "initial_state_sha256": row["initial_state_sha256"]})
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _episode_seal_digest(bindings: Mapping[str, Any]) -> str:
    values = {identity: bindings[identity]["episode_sha256sums_sha256"] for identity in sorted(bindings)}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(parent_transition: Path, input_audit_root: Path, teacher_contract: Path, teacher_runner: Path, protocol: Path, formal_root: Path, output_root: Path, *, expected_parent_seal: str, expected_audit_seal: str, runner_commit: str, runner_tree: str, environment_fingerprint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    parent_transition = parent_transition.resolve()
    input_audit_root = _safe_root(input_audit_root)
    formal_root = _safe_root(formal_root)
    teacher_contract = teacher_contract.resolve()
    teacher_runner = teacher_runner.resolve()
    protocol = protocol.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root = output_root.resolve()
    if output_root.parent != formal_root.parent:
        raise ValueError("FIT_TO_TEACHER output must be a new sibling of the formal input root")
    try:
        parent_seal = verify_seal(parent_transition.parent)
    except ValueError as exc:
        raise ValueError("parent transition seal mismatch") from exc
    if parent_seal["sha256sums_sha256"] != expected_parent_seal:
        raise ValueError("parent transition seal mismatch")
    parent = _read_json(parent_transition)
    if parent.get("schema") != "FIT670_INFERENCE_TRANSITION_V2" or parent.get("collection_mode") != "formal" or parent.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise ValueError("parent transition schema mismatch")
    if parent.get("teacher_labels_authorized") is not False or parent.get("student_training_authorized") is not False or parent.get("attack_authorized") is not False:
        raise ValueError("parent transition historical permissions are not closed")
    if parent.get("protected_payload_read") is not False or parent.get("protected_overlap_verified") != 0:
        raise ValueError("parent transition protected boundary is not closed")
    if parent.get("max_episodes") != 670 or parent.get("n_shards") != 8 or parent.get("identity_set_frozen") is not True:
        raise ValueError("parent transition cardinality is not frozen")
    authorized = parent.get("authorized_identities")
    if not ((isinstance(authorized, int) and not isinstance(authorized, bool) and authorized == 670) or (isinstance(authorized, list) and len(authorized) == 670 and len(set(map(str, authorized))) == 670)):
        raise ValueError("parent transition authorized identity binding is not exact")

    audit_seal = verify_seal(input_audit_root)
    if audit_seal["sha256sums_sha256"] != expected_audit_seal:
        raise ValueError("T0-A audit seal mismatch")
    audit_manifest = _read_json(input_audit_root / "FORMAL_INPUT_MANIFEST.json")
    if audit_manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or audit_manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE":
        raise ValueError("T0-A audit is not consumable")
    if audit_manifest.get("episode_count") != 670 or audit_manifest.get("protected_reads") != 0 or audit_manifest.get("teacher_labels_generated") is not False:
        raise ValueError("T0-A audit cardinality/permission mismatch")
    if audit_manifest.get("labels_generated") is not False or audit_manifest.get("student_started") is not False or audit_manifest.get("source_staging_residue") != []:
        raise ValueError("T0-A audit downstream/staging boundary is not closed")
    gate = audit_manifest.get("gate")
    expected_zero_gate = {"duplicate", "missing", "extra", "unallowlisted", "bad_episode_seal", "bad_worker_seal", "schema_error", "empty_entity_records", "identity_binding_error", "source_binding_error", "nonfinite", "staging_residue", "protected_reads"}
    if not isinstance(gate, Mapping) or any(gate.get(key) != 0 for key in expected_zero_gate):
        raise ValueError("T0-A gate is not closed")
    if Path(str(audit_manifest.get("formal_root", ""))).resolve() != formal_root:
        raise ValueError("T0-A formal root mismatch")
    identity_digest = _sha(audit_manifest.get("identity_set_digest"), "identity_set_digest")
    episode_digest = _sha(audit_manifest.get("finalization", {}).get("episode_seal_digest"), "episode_seal_digest")
    if audit_manifest.get("transition_manifest_sha256") != sha256_file(parent_transition):
        raise ValueError("T0-A does not bind the requested parent transition")
    if audit_manifest.get("collection_source_commit") != parent.get("collection_source_commit") or audit_manifest.get("collection_source_tree") != parent.get("collection_source_tree"):
        raise ValueError("source lineage mismatch")
    bindings = audit_manifest.get("episode_bindings")
    if not isinstance(bindings, dict) or len(bindings) != 670 or len(set(bindings)) != 670:
        raise ValueError("T0-A episode bindings are not exact 670")
    binding_digest = _sha(audit_manifest.get("episode_binding_digest"), "episode_binding_digest")
    if _binding_digest(bindings) != binding_digest:
        raise ValueError("T0-A episode binding digest does not match its contents")
    if _identity_digest(bindings) != identity_digest or _episode_seal_digest(bindings) != episode_digest:
        raise ValueError("T0-A identity/episode seal digest does not recompute from bindings")

    if not teacher_contract.is_file() or teacher_contract.is_symlink() or not teacher_runner.is_file() or teacher_runner.is_symlink() or not protocol.is_file() or protocol.is_symlink():
        raise ValueError("teacher contract/runner source missing or symlinked")
    runner_commit = _sha(runner_commit, "runner_commit", 40)
    runner_tree = _sha(runner_tree, "runner_tree", 40)
    created_at = datetime.now(timezone.utc)
    parent_created_at = parent.get("created_at") or parent.get("generated_at")
    if parent_created_at:
        try:
            parent_time = datetime.fromisoformat(str(parent_created_at).replace("Z", "+00:00"))
            if created_at < parent_time:
                raise ValueError("downstream transition chronology precedes parent")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("parent chronology is malformed") from exc
    report = {
        "schema": "FIT_TO_TEACHER_TRANSITION_V1",
        "status": "PASS_FIT_TO_TEACHER_AUTHORIZATION",
        "created_at": created_at.isoformat(),
        "formal_root": str(formal_root),
        "output_root": str(output_root.resolve()),
        "parent_transition_manifest_sha256": sha256_file(parent_transition),
        "parent_transition_sha256sums_sha256": parent_seal["sha256sums_sha256"],
        "input_audit_manifest_sha256": sha256_file(input_audit_root / "FORMAL_INPUT_MANIFEST.json"),
        "input_audit_seal_sha256sums_sha256": audit_seal["sha256sums_sha256"],
        "identity_count": 670,
        "identity_set_digest": identity_digest,
        "episode_seal_digest": episode_digest,
        "episode_binding_digest": binding_digest,
        "collection_source_commit": parent.get("collection_source_commit"),
        "collection_source_tree": parent.get("collection_source_tree"),
        "teacher_contract_sha256": sha256_file(teacher_contract),
        "teacher_contract_path": str(teacher_contract),
        "teacher_runner_sha256": sha256_file(teacher_runner),
        "teacher_runner_path": str(teacher_runner),
        "protocol_sha256": sha256_file(protocol),
        "protocol_path": str(protocol),
        "runner_source_commit": runner_commit,
        "runner_source_tree": runner_tree,
        "environment_fingerprint": dict(environment_fingerprint or {}),
        "permissions": PERMISSIONS,
        "labels_generated": False,
        "protected_reads": 0,
        "training_authorized": False,
        "student_training_authorized": False,
        "rollout_authorized": False,
        "attack_authorized": False,
        "parent_chronology": {"parent_created_at": parent_created_at, "child_created_at": created_at.isoformat()},
    }
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists() or output_root.exists():
        raise FileExistsError("transition output already exists")
    staging.mkdir(parents=True)
    try:
        (staging / "FIT_TO_TEACHER_TRANSITION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "PERMISSION_MATRIX.json").write_text(json.dumps(PERMISSIONS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "PARENT_BINDING.json").write_text(json.dumps({"parent_transition": report["parent_transition_manifest_sha256"], "input_audit": report["input_audit_manifest_sha256"], "identity_set_digest": identity_digest, "episode_seal_digest": episode_digest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        raise
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-transition", type=Path, required=True)
    parser.add_argument("--input-audit-root", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--teacher-runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-parent-seal", required=True)
    parser.add_argument("--expected-audit-seal", required=True)
    parser.add_argument("--runner-commit", required=True)
    parser.add_argument("--runner-tree", required=True)
    parser.add_argument("--environment-fingerprint-json", type=Path)
    args = parser.parse_args()
    fingerprint = _read_json(args.environment_fingerprint_json) if args.environment_fingerprint_json else {}
    print(json.dumps(build(args.parent_transition, args.input_audit_root, args.teacher_contract, args.teacher_runner, args.protocol, args.formal_root, args.output_root, expected_parent_seal=args.expected_parent_seal, expected_audit_seal=args.expected_audit_seal, runner_commit=args.runner_commit, runner_tree=args.runner_tree, environment_fingerprint=fingerprint), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
