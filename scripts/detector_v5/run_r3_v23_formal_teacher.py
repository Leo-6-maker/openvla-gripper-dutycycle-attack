"""Materialize V23 five-head labels from the frozen FIT670 formal root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_r3_contact_input import sha256_file, verify_seal
from audit_r3_formal_input import BINDING_FIELDS, _canonical_digest
from build_r3_teacher_pilot_manifest import validate_task_groups
from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_teacher import HEADS, canonicalize_fit670_episode, derive_episode_labels, validate_contact_row


EXPECTED_STATUS = "PASS_CONSUMABLE_FINAL"
FINALIZATION_FILES = (
    "GLOBAL_MANIFEST.json",
    "PROGRESS_RECONCILIATION.json",
    "IDENTITY_CLOSURE.json",
    "PER_SHARD_CLOSURE.json",
    "WORKER_RUNTIME_STATE.json",
    "STAGING_AUDIT.json",
)
FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
FORBIDDEN_FIELDS = {"task_success", "terminal", "reward", "outcome", "attack_result", "future_frame", "future_label"}


def _write_seal(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _episode_label_name(identity: str) -> str:
    return identity.replace("/", "__")


def _load_sealed_episode_labels(root: Path, expected_identity: str, expected_binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seal = verify_seal(root)
    manifest_path = root / "EPISODE_TEACHER_MANIFEST.json"
    records_path = root / "teacher_records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise ValueError(f"sealed episode label bundle incomplete: {expected_identity}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("episode_id") != expected_identity or manifest.get("status") != "PASS_EPISODE_TEACHER_LABELS":
        raise ValueError(f"sealed episode label identity/status mismatch: {expected_identity}")
    for field, expected in expected_binding.items():
        if manifest.get(field) != expected:
            raise ValueError(f"sealed episode label provenance mismatch: {expected_identity}.{field}")
    if manifest.get("unknown_to_negative") is not False or manifest.get("future_fields_used") is not False or manifest.get("outcome_fields_used") is not False:
        raise ValueError(f"sealed episode label permission mismatch: {expected_identity}")
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != manifest.get("step_count"):
        raise ValueError(f"sealed episode label count mismatch: {expected_identity}")
    for expected_step, row in enumerate(rows):
        _reject_forbidden_fields(row, f"{expected_identity}[{expected_step}]")
        if row.get("episode_id") != expected_identity or row.get("step") != expected_step:
            raise ValueError(f"sealed episode label step binding mismatch: {expected_identity}")
    return rows, manifest


def _safe_identity_path(root: Path, identity: str) -> Path:
    parts = identity.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe formal identity: {identity!r}")
    relative = Path("episodes", *parts, "episode.json")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"formal episode escapes or is missing: {identity}")
    return relative


def _reject_forbidden_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden future/outcome field: {path}.{key}")
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _require_pilot_selection(selection_manifest_path: Path | None, selection_digest: str | None) -> None:
    if selection_manifest_path is None or selection_digest is None:
        raise ValueError("T1 selected pilot manifest is required; full formal path is disabled")


def _verify_transition(path: Path, formal_root: Path, expected_digest: str) -> dict[str, Any]:
    manifest_path = path.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("formal transition manifest missing or symlinked")
    transition_root = manifest_path.parent
    seal = verify_seal(transition_root)
    if seal["sha256sums_sha256"] != expected_digest:
        raise ValueError("formal transition seal mismatch")
    transition = json.loads(manifest_path.read_text(encoding="utf-8"))
    if transition.get("schema") != "FIT670_INFERENCE_TRANSITION_V2" or transition.get("collection_mode") != "formal":
        raise ValueError("formal transition schema/mode mismatch")
    if transition.get("teacher_labels_authorized") is not False:
        raise ValueError("historical collection transition must keep teacher_labels_authorized=false")
    if transition.get("student_training_authorized") is not False or transition.get("attack_authorized") is not False:
        raise ValueError("historical transition grants a forbidden permission")
    if transition.get("max_episodes") != 670 or transition.get("n_shards") != 8 or transition.get("identity_set_frozen") is not True:
        raise ValueError("formal transition cardinality is not frozen")
    authorized = transition.get("authorized_identities")
    if not ((isinstance(authorized, int) and not isinstance(authorized, bool) and authorized == 670) or (isinstance(authorized, list) and len(authorized) == 670 and len(set(map(str, authorized))) == 670)):
        raise ValueError("formal transition authorized identity binding is not exact")
    if transition.get("protected_overlap_verified") != 0 or transition.get("protected_payload_read") is not False:
        raise ValueError("formal transition protected boundary is not closed")
    if str(formal_root) not in {str(Path(item).resolve()) for item in transition.get("allowed_output_roots", [])}:
        raise ValueError("formal root is not transition-allowlisted")
    allowlist_path = Path(str(transition.get("identity_allowlist_path", ""))).resolve()
    if any(part.lower() in FORBIDDEN_PARTS for part in allowlist_path.parts):
        raise ValueError("formal identity allowlist is on a forbidden path")
    if sha256_file(allowlist_path) != transition.get("identity_allowlist_file_sha256"):
        raise ValueError("formal identity allowlist SHA mismatch")
    allowlist_root_seal = verify_seal(allowlist_path.parent)
    if allowlist_root_seal["sha256sums_sha256"] != transition.get("identity_allowlist_root_sha256sums_sha256"):
        raise ValueError("formal identity allowlist root seal mismatch")
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    entries = allowlist.get("identities")
    if allowlist.get("schema") != "FIT670_IDENTITY_ALLOWLIST_V1" or not isinstance(entries, list):
        raise ValueError("formal identity allowlist schema mismatch")
    ids = [str(item["episode_id"]) for item in entries]
    if len(ids) != 670 or len(set(ids)) != 670:
        raise ValueError("formal identity allowlist is not unique 670")
    if allowlist.get("protected_overlap") != 0:
        raise ValueError("formal identity allowlist has protected overlap")
    return {
        "manifest": transition,
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256sums_sha256": seal["sha256sums_sha256"],
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": transition["identity_allowlist_file_sha256"],
        "allowlist_root_sha256sums_sha256": allowlist_root_seal["sha256sums_sha256"],
        "allowlist": allowlist,
        "identities": ids,
    }


def _verify_input_audit(audit_root: Path, formal_root: Path, finalization_root: Path, transition_path: Path, transition: Mapping[str, Any], expected_digest: str) -> dict[str, Any]:
    root = audit_root.resolve()
    if root.is_symlink() or not root.is_dir() or any(part.lower() in FORBIDDEN_PARTS for part in root.parts):
        raise ValueError("T0-A audit root is missing, symlinked, or forbidden-looking")
    seal = verify_seal(root)
    manifest_path = root / "FORMAL_INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("T0-A manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE":
        raise ValueError("T0-A audit is not consumable")
    if manifest.get("episode_count") != 670 or manifest.get("episode_list_nonempty") is not True or manifest.get("protected_reads") != 0:
        raise ValueError("T0-A audit cardinality/boundary is not closed")
    if manifest.get("teacher_labels_generated") is not False or manifest.get("labels_generated") is not False or manifest.get("student_started") is not False or manifest.get("attack_authorized") is not False:
        raise ValueError("T0-A audit already contains downstream execution")
    if Path(str(manifest.get("formal_root", ""))).resolve() != formal_root.resolve():
        raise ValueError("T0-A formal root mismatch")
    finalization = manifest.get("finalization")
    if not isinstance(finalization, Mapping) or Path(str(finalization.get("root", ""))).resolve() != finalization_root.resolve():
        raise ValueError("T0-A finalization root mismatch")
    if finalization.get("episode_seal_digest") != expected_digest:
        raise ValueError("T0-A episode digest mismatch")
    if finalization.get("global_manifest_sha256") != sha256_file(finalization_root / "GLOBAL_MANIFEST.json"):
        raise ValueError("T0-A global manifest binding mismatch")
    if finalization.get("sha256sums_sha256") != sha256_file(finalization_root / "SHA256SUMS"):
        raise ValueError("T0-A finalization seal binding mismatch")
    auxiliary = finalization.get("auxiliary_sha256")
    if not isinstance(auxiliary, Mapping) or not auxiliary:
        raise ValueError("T0-A auxiliary finalization binding missing")
    for name, digest in auxiliary.items():
        if sha256_file(finalization_root / str(name)) != str(digest):
            raise ValueError(f"T0-A auxiliary finalization mismatch: {name}")
    if manifest.get("transition_manifest_sha256") != sha256_file(transition_path):
        raise ValueError("T0-A transition manifest binding mismatch")
    bindings = manifest.get("episode_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 670 or set(bindings) != set(finalization.get("episode_seals", {})):
        raise ValueError("T0-A episode binding closure is incomplete")
    return {"root": str(root), "seal": seal, "manifest": manifest, "manifest_sha256": sha256_file(manifest_path), "seal_sha256sums_sha256": seal["sha256sums_sha256"]}


def _verify_selection_manifest(path: Path, audit: Mapping[str, Any], formal_root: Path, expected_seal: str | None) -> dict[str, Any]:
    manifest_path = path.resolve()
    if any(part.lower() in FORBIDDEN_PARTS for part in manifest_path.parts):
        raise ValueError("pilot selection path is forbidden-looking")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("pilot selection manifest missing or symlinked")
    seal = verify_seal(manifest_path.parent)
    if expected_seal is None or seal["sha256sums_sha256"] != expected_seal:
        raise ValueError("pilot selection seal mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "V5_R3_TEACHER_PILOT_MANIFEST_V1" or manifest.get("status") != "PASS_FROZEN_TEACHER_PILOT_INPUT":
        raise ValueError("pilot selection manifest is not frozen")
    if manifest.get("formal_root") != str(formal_root) or manifest.get("input_audit_manifest_sha256") != str(audit["manifest_sha256"]):
        raise ValueError("pilot selection source binding mismatch")
    if manifest.get("input_audit_seal_sha256sums_sha256") != str(audit["seal_sha256sums_sha256"]):
        raise ValueError("pilot selection audit seal mismatch")
    if manifest.get("protected_reads") != 0 or manifest.get("teacher_labels_generated") is not False or manifest.get("labels_generated") is not False or manifest.get("student_started") is not False or manifest.get("attack_authorized") is not False:
        raise ValueError("pilot selection permission boundary is not closed")
    selected = manifest.get("selected_bindings")
    audit_bindings = audit["manifest"].get("episode_bindings")
    if not isinstance(selected, list) or len(selected) != 40 or len({row.get("episode_id") for row in selected if isinstance(row, Mapping)}) != 40 or not isinstance(audit_bindings, Mapping):
        raise ValueError("pilot selection identity closure is not exact 40")
    for row in selected:
        identity = row.get("episode_id")
        if identity not in audit_bindings or any(row.get(key) != audit_bindings[identity].get(key) for key in BINDING_FIELDS):
            raise ValueError(f"pilot selection binding mismatch: {identity}")
    if manifest.get("selected_identity_digest") != _canonical_digest(selected):
        raise ValueError("pilot selection digest mismatch")
    try:
        validate_task_groups(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError("pilot selection is not exactly 4 suites x 10 tasks") from exc
    for field, expected in {
        "episode_binding_digest": audit["manifest"].get("episode_binding_digest"),
        "identity_set_digest": audit["manifest"].get("identity_set_digest"),
        "episode_seal_digest": audit["manifest"].get("finalization", {}).get("episode_seal_digest"),
        "transition_manifest_sha256": audit["manifest"].get("transition_manifest_sha256"),
        "transition_sha256sums_sha256": audit["manifest"].get("transition_sha256sums_sha256"),
        "allowlist_sha256": audit["manifest"].get("allowlist_sha256"),
        "allowlist_root_sha256sums_sha256": audit["manifest"].get("allowlist_root_sha256sums_sha256"),
        "shard_plan_sha256": audit["manifest"].get("shard_plan_sha256"),
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"pilot selection source closure mismatch: {field}")
    return {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path), "seal_sha256sums_sha256": seal["sha256sums_sha256"], "identities": [row["episode_id"] for row in selected]}


def _validate_teacher_label_root(output_root: Path, declared_transition_root: Path, transition_manifest_root: Path) -> None:
    declared_transition_root = declared_transition_root.resolve()
    transition_manifest_root = transition_manifest_root.resolve()
    output_root = output_root.resolve()
    if declared_transition_root != transition_manifest_root:
        raise ValueError("FIT_TO_TEACHER transition output root mismatch")
    if output_root == declared_transition_root or output_root.parent != declared_transition_root.parent:
        raise ValueError("Teacher label output root must be a new sibling of the sealed transition root")


def _verify_fit_to_teacher_transition(path: Path, original: Mapping[str, Any], audit: Mapping[str, Any], formal_root: Path, output_root: Path, teacher_contract_path: Path, teacher_runner_path: Path, protocol_path: Path, expected_seal: str) -> dict[str, Any]:
    manifest_path = path.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("FIT_TO_TEACHER transition missing or symlinked")
    seal = verify_seal(manifest_path.parent)
    if seal["sha256sums_sha256"] != expected_seal:
        raise ValueError("FIT_TO_TEACHER transition seal mismatch")
    transition = json.loads(manifest_path.read_text(encoding="utf-8"))
    if transition.get("schema") != "FIT_TO_TEACHER_TRANSITION_V1" or transition.get("status") != "PASS_FIT_TO_TEACHER_AUTHORIZATION":
        raise ValueError("FIT_TO_TEACHER transition is not frozen/authorized")
    if transition.get("parent_transition_manifest_sha256") != sha256_file(Path(str(original.get("manifest_path", "")))):
        raise ValueError("FIT_TO_TEACHER parent transition mismatch")
    if transition.get("input_audit_manifest_sha256") != str(audit.get("manifest_sha256")):
        raise ValueError("FIT_TO_TEACHER input audit mismatch")
    if transition.get("input_audit_seal_sha256sums_sha256") != str(audit.get("seal_sha256sums_sha256")):
        raise ValueError("FIT_TO_TEACHER input audit seal mismatch")
    if transition.get("identity_set_digest") != audit["manifest"].get("identity_set_digest") or transition.get("identity_count") != 670 or transition.get("episode_seal_digest") != audit["manifest"].get("finalization", {}).get("episode_seal_digest") or transition.get("episode_binding_digest") != audit["manifest"].get("episode_binding_digest"):
        raise ValueError("FIT_TO_TEACHER identity binding mismatch")
    if Path(str(transition.get("formal_root", ""))).resolve() != formal_root.resolve():
        raise ValueError("FIT_TO_TEACHER formal root mismatch")
    _validate_teacher_label_root(output_root, Path(str(transition.get("output_root", ""))), manifest_path.parent)
    permissions = transition.get("permissions")
    expected_permissions = {
        "fit_episode_read": True, "teacher_label_generation": True,
        "student_dataset_generation": False, "student_training": False,
        "detector_load": False, "rollout": False, "shadow": False,
        "attack": False, "protected_payload_read": False,
        "CAL_READ": False, "CHECK_READ": False, "G10_READ": False, "T2R_D_READ": False,
    }
    if permissions != expected_permissions:
        raise ValueError("FIT_TO_TEACHER permissions mismatch")
    for label, file_path, field in (("teacher contract", teacher_contract_path, "teacher_contract_sha256"), ("teacher runner", teacher_runner_path, "teacher_runner_sha256")):
        if file_path.is_symlink() or not file_path.is_file() or transition.get(field) != sha256_file(file_path):
            raise ValueError(f"FIT_TO_TEACHER {label} binding mismatch")
    if transition.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("FIT_TO_TEACHER protocol binding mismatch")
    if transition.get("protected_reads") != 0 or transition.get("labels_generated") is not False:
        raise ValueError("FIT_TO_TEACHER execution boundary is not closed")
    return {"manifest": transition, "manifest_path": str(manifest_path), "seal": seal, "manifest_sha256": sha256_file(manifest_path), "seal_sha256sums_sha256": seal["sha256sums_sha256"]}


def _verify_finalization(finalization_root: Path, transition: Mapping[str, Any], expected_digest: str) -> dict[str, Any]:
    root = finalization_root.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("symlink in formal finalization root")
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError("formal finalization sidecar mismatch")
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in listed or name != "GLOBAL_MANIFEST.json":
            raise ValueError("formal finalization seal is not the expected global-manifest seal")
        listed[name] = digest
    global_path = root / "GLOBAL_MANIFEST.json"
    if listed.get("GLOBAL_MANIFEST.json") != sha256_file(global_path):
        raise ValueError("formal global manifest SHA mismatch")
    global_manifest = json.loads(global_path.read_text(encoding="utf-8"))
    episode_seals = global_manifest.get("episode_seals")
    if not isinstance(episode_seals, dict) or len(episode_seals) != 670:
        raise ValueError("formal global episode seal map is incomplete")
    map_digest = hashlib.sha256(json.dumps(episode_seals, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if global_manifest.get("episode_seal_digest") != expected_digest or map_digest != expected_digest:
        raise ValueError("formal episode seal digest mismatch")
    if global_manifest.get("collection_source_commit") != transition["collection_source_commit"] or global_manifest.get("collection_source_tree") != transition["collection_source_tree"]:
        raise ValueError("formal global source lineage mismatch")
    auxiliary = {}
    for name in FINALIZATION_FILES[1:]:
        path = root / name
        if not path.is_file():
            raise ValueError(f"formal finalization file missing: {name}")
        auxiliary[name] = sha256_file(path)
    progress = json.loads((root / "PROGRESS_RECONCILIATION.json").read_text(encoding="utf-8"))
    if not (
        progress.get("verdict") == "PASS"
        and progress.get("allowlist_count") == 670
        and progress.get("published_count") == 670
        and progress.get("duplicates") == 0
        and progress.get("bad_seals") == 0
        and progress.get("unallowlisted") == 0
        and progress.get("staging_residue") == 0
        and not progress.get("missing")
        and not progress.get("extra")
    ):
        raise ValueError("formal progress reconciliation is not PASS")
    identity_closure = json.loads((root / "IDENTITY_CLOSURE.json").read_text(encoding="utf-8"))
    if len(identity_closure.get("allowlist_ids", [])) != 670 or set(identity_closure["allowlist_ids"]) != set(episode_seals):
        raise ValueError("formal identity closure mismatch")
    per_shard = json.loads((root / "PER_SHARD_CLOSURE.json").read_text(encoding="utf-8"))
    if len(per_shard) != 8 or not all(item.get("match") is True and item.get("missing") == [] and item.get("extra") == [] for item in per_shard):
        raise ValueError("formal per-shard closure is not PASS")
    worker_state = json.loads((root / "WORKER_RUNTIME_STATE.json").read_text(encoding="utf-8"))
    workers = worker_state.get("workers", {})
    if len(workers) != 8 or not all(item.get("manifest_present") is True and item.get("n_fail") == 0 and item.get("n_skipped") == 0 for item in workers.values()):
        raise ValueError("formal worker runtime state is not PASS")
    staging = json.loads((root / "STAGING_AUDIT.json").read_text(encoding="utf-8"))
    if staging.get("verdict") != "PASS" or staging.get("staging_residue_count") != 0 or staging.get("staging_residue") != []:
        raise ValueError("formal staging audit is not PASS")
    return {
        "root": str(root),
        "global_manifest_sha256": sha256_file(global_path),
        "sha256sums_sha256": sha256_file(sums),
        "episode_seal_digest": expected_digest,
        "episode_seals": episode_seals,
        "auxiliary_sha256": auxiliary,
        "seal_scope": "GLOBAL_MANIFEST_ONLY_WITH_AUXILIARY_FINALIZATION_AUDIT",
    }


def _load_formal(formal_root: Path, finalization_root: Path, transition_path: Path, input_audit_root: Path, fit_to_teacher_transition_path: Path, teacher_contract_path: Path, teacher_runner_path: Path, protocol_path: Path, selection_manifest_path: Path | None, selection_digest: str | None, *, transition_digest: str, episode_digest: str, fit_to_teacher_transition_digest: str, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_pilot_selection(selection_manifest_path, selection_digest)
    root = formal_root.resolve()
    paths = [root, finalization_root.resolve(), transition_path.resolve(), input_audit_root.resolve(), fit_to_teacher_transition_path.resolve(), teacher_contract_path.resolve(), teacher_runner_path.resolve()]
    if selection_manifest_path is not None:
        paths.append(selection_manifest_path.resolve())
    if any(part.lower() in FORBIDDEN_PARTS for path in paths for part in path.parts):
        raise ValueError("formal input path is forbidden-looking")
    transition = _verify_transition(transition_path, root, transition_digest)
    audit = _verify_input_audit(input_audit_root, root, finalization_root, transition_path, transition["manifest"], episode_digest)
    selection = _verify_selection_manifest(selection_manifest_path, audit, root, selection_digest)
    downstream = _verify_fit_to_teacher_transition(fit_to_teacher_transition_path, {**transition, "manifest_path": str(transition_path.resolve())}, audit, root, output_root, teacher_contract_path, teacher_runner_path, protocol_path, fit_to_teacher_transition_digest)
    finalization = audit["manifest"]["finalization"]
    bindings = audit["manifest"]["episode_bindings"]
    authorized = transition["manifest"].get("authorized_identities")
    authorized_ok = (isinstance(authorized, int) and not isinstance(authorized, bool) and authorized == len(bindings)) or (isinstance(authorized, list) and len(authorized) == len(bindings) and set(map(str, authorized)) == set(bindings))
    if not authorized_ok or set(bindings) != set(finalization["episode_seals"]):
        raise ValueError("formal transition/audit identity mismatch")
    loaded = []
    identities = selection["identities"]
    for identity in identities:
        relative = _safe_identity_path(root, identity)
        episode_path = root / relative
        episode_seal = verify_seal(episode_path.parent)
        binding_row = bindings[identity]
        if binding_row.get("relative_path") != relative.as_posix() or episode_seal["sha256sums_sha256"] != binding_row.get("episode_sha256sums_sha256") or episode_seal["sha256sums_sha256"] != finalization["episode_seals"][identity]:
            raise ValueError(f"formal episode seal mismatch: {identity}")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        _reject_forbidden_fields(episode, identity)
        if episode.get("schema") != "FIT670_EPISODE_V2" or episode.get("episode_id") != identity:
            raise ValueError(f"formal episode schema/identity mismatch: {identity}")
        if episode.get("attack_enabled") is not False or episode.get("detector_loaded") is not False or episode.get("teacher_labels_generated") is not False:
            raise ValueError(f"formal episode authorization mismatch: {identity}")
        generation_passes = episode.get("generation_passes_per_step")
        expected_steps = episode.get("n_steps")
        if (episode.get("forward_before_capture") is not True or not isinstance(generation_passes, list) or not isinstance(expected_steps, int) or expected_steps <= 0 or len(generation_passes) != expected_steps or not generation_passes or any(isinstance(value, bool) or not isinstance(value, int) or value != 1 for value in generation_passes)):
            raise ValueError(f"formal episode causal capture mismatch: {identity}")
        provenance = episode.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("collector_commit") != transition["manifest"]["collection_source_commit"] or provenance.get("collector_tree") != transition["manifest"]["collection_source_tree"]:
            raise ValueError(f"formal episode source lineage mismatch: {identity}")
        expected_identity = f"{episode.get('suite')}/task_{int(episode.get('task_id')):02d}/state_{int(episode.get('state_id')):02d}"
        if expected_identity != identity or binding_row.get("suite") != episode.get("suite") or binding_row.get("task_id") != episode.get("task_id") or binding_row.get("state_id") != episode.get("state_id") or binding_row.get("seed") != episode.get("collection_seed") or binding_row.get("initial_state_sha256") != episode.get("initial_state_sha256"):
            raise ValueError(f"formal episode identity binding mismatch: {identity}")
        if binding_row.get("episode_sha256") != sha256_file(episode_path) or binding_row.get("collection_source_commit") != provenance.get("collector_commit") or binding_row.get("collection_source_tree") != provenance.get("collector_tree"):
            raise ValueError(f"formal episode source binding mismatch: {identity}")
        rows = canonicalize_fit670_episode(episode)
        if len(rows) != int(episode.get("step_count")) or len(rows) != int(episode.get("n_steps")):
            raise ValueError(f"formal episode step count mismatch: {identity}")
        for step, row in enumerate(rows):
            validate_contact_row(row, expected_step=step)
        loaded.append({
            "manifest": {
                "episode_id": identity,
                "suite": episode["suite"],
                "task_id": episode["task_id"],
                "state_id": episode["state_id"],
                "seed": episode["collection_seed"],
                "relative_path": relative.as_posix(),
                "step_count": len(rows),
                "source_sha256": sha256_file(episode_path),
                "episode_sha256sums_sha256": episode_seal["sha256sums_sha256"],
                "source_commit": provenance["collector_commit"],
                "source_tree": provenance["collector_tree"],
                "collector_script_sha256": provenance["collector_script_sha256"],
                "worker_id": binding_row.get("worker_id"),
                "shard_id": binding_row.get("shard_id"),
            },
            "rows": rows,
        })
    return {
        "schema": "FIT670_V2_FORMAL_CONSUMABLE_INPUT_V1",
        "status": EXPECTED_STATUS,
        "formal_root": str(root),
        "finalization": finalization,
        "input_audit": audit,
        "fit_to_teacher_transition": downstream,
        "transition": transition,
        "selection": selection,
        "identity_count": len(loaded),
        "step_count": sum(len(item["rows"]) for item in loaded),
        "protected_reads": 0,
        "teacher_labels_generated": False,
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
        "attack_authorized": False,
    }, loaded


def run(formal_root: Path, finalization_root: Path, transition_path: Path, input_audit_root: Path, fit_to_teacher_transition_path: Path, teacher_contract_path: Path, teacher_runner_path: Path, protocol_path: Path, output_root: Path, *, transition_digest: str, episode_digest: str, fit_to_teacher_transition_digest: str, selection_manifest_path: Path | None = None, selection_digest: str | None = None, resume: bool = False) -> dict[str, Any]:
    _require_pilot_selection(selection_manifest_path, selection_digest)
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("schema", "").startswith("V5_TEACHER_STUDENT_R3_DEV_PROTOCOL"):
        raise ValueError("unexpected R3 protocol")
    binding, episodes = _load_formal(formal_root, finalization_root, transition_path, input_audit_root, fit_to_teacher_transition_path, teacher_contract_path, teacher_runner_path, protocol_path, selection_manifest_path, selection_digest, transition_digest=transition_digest, episode_digest=episode_digest, fit_to_teacher_transition_digest=fit_to_teacher_transition_digest, output_root=output_root)
    staging = output_root.with_name(f".{output_root.name}.staging")
    if output_root.exists():
        raise FileExistsError(output_root)
    if staging.exists() and not resume:
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=True)
    downstream_manifest = binding["fit_to_teacher_transition"]["manifest"]
    resume_binding = {
        "schema": "V5_R3_TEACHER_RESUME_MANIFEST_V1",
        "input_audit_manifest_sha256": binding["input_audit"]["manifest_sha256"],
        "input_audit_seal_sha256sums_sha256": binding["input_audit"]["seal_sha256sums_sha256"],
        "fit_to_teacher_transition_manifest_sha256": binding["fit_to_teacher_transition"]["manifest_sha256"],
        "fit_to_teacher_transition_seal_sha256sums_sha256": binding["fit_to_teacher_transition"]["seal_sha256sums_sha256"],
        "teacher_contract_sha256": downstream_manifest["teacher_contract_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "identity_set_digest": binding["transition"]["manifest"].get("identity_set_digest"),
        "identity_count": len(episodes),
        "selection_manifest_sha256": binding["selection"]["manifest_sha256"] if binding["selection"] else None,
        "selection_seal_sha256sums_sha256": binding["selection"]["seal_sha256sums_sha256"] if binding["selection"] else None,
    }
    resume_manifest_path = staging / "RESUME_MANIFEST.json"
    if resume_manifest_path.exists():
        if json.loads(resume_manifest_path.read_text(encoding="utf-8")) != resume_binding:
            raise ValueError("resume manifest provenance mismatch")
    else:
        resume_manifest_path.write_text(json.dumps(resume_binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    episodes_root = staging / "episodes"
    if episodes_root.exists():
        expected_names = {_episode_label_name(str(item["manifest"]["episode_id"])) for item in episodes}
        actual_names = {path.name for path in episodes_root.iterdir()}
        if not actual_names.issubset(expected_names):
            raise ValueError(f"resume staging contains extra episode labels: {sorted(actual_names - expected_names)}")
    label_count = 0
    try:
        records = []
        source_manifest = []
        for episode in episodes:
            identity = str(episode["manifest"]["episode_id"])
            downstream_manifest = binding["fit_to_teacher_transition"]["manifest"]
            label_binding = {
                "source_episode_sha256": episode["manifest"]["source_sha256"],
                "source_episode_sha256sums_sha256": episode["manifest"]["episode_sha256sums_sha256"],
                "teacher_contract_sha256": downstream_manifest["teacher_contract_sha256"],
                "protocol_sha256": sha256_file(protocol_path),
                "input_audit_manifest_sha256": binding["input_audit"]["manifest_sha256"],
                "input_audit_seal_sha256sums_sha256": binding["input_audit"]["seal_sha256sums_sha256"],
                "fit_to_teacher_transition_manifest_sha256": binding["fit_to_teacher_transition"]["manifest_sha256"],
                "fit_to_teacher_transition_seal_sha256sums_sha256": binding["fit_to_teacher_transition"]["seal_sha256sums_sha256"],
            }
            target = staging / "episodes" / _episode_label_name(identity)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                labels, episode_report = _load_sealed_episode_labels(target, identity, label_binding)
            else:
                labels = derive_episode_labels(episode["rows"], protocol)
                episode_staging = staging / f".episode.{_episode_label_name(identity)}.{os.getpid()}"
                if episode_staging.exists():
                    raise FileExistsError(episode_staging)
                episode_staging.mkdir(parents=True)
                (episode_staging / "teacher_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
                episode_report = {
                    "schema": "V5_R3_V23_TEACHER_EPISODE_V1",
                    "status": "PASS_EPISODE_TEACHER_LABELS",
                    "episode_id": identity,
                    "step_count": len(labels),
                    "source_episode_sha256": episode["manifest"]["source_sha256"],
                    "source_episode_sha256sums_sha256": episode["manifest"]["episode_sha256sums_sha256"],
                    "teacher_contract_sha256": binding["fit_to_teacher_transition"]["manifest"].get("teacher_contract_sha256"),
                    "unknown_to_negative": False,
                    "future_fields_used": False,
                    "outcome_fields_used": False,
                    **label_binding,
                }
                (episode_staging / "EPISODE_TEACHER_MANIFEST.json").write_text(json.dumps(episode_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                _write_seal(episode_staging)
                rename_noreplace(episode_staging, target)
            records.extend(labels)
            source_manifest.append({**episode["manifest"], **label_binding, "label_root": target.relative_to(staging).as_posix(), "label_seal_sha256sums_sha256": verify_seal(target)["sha256sums_sha256"]})
            label_count += len(labels)
        report = {
            "schema": "V5_R3_V23_TEACHER_FORMAL_V1",
            "status": "DEVELOPMENT_NONCONSUMABLE",
            "input_status": EXPECTED_STATUS,
            "input_binding": binding,
            "protocol_sha256": sha256_file(protocol_path),
            "identity_count": len(episodes),
            "step_count": label_count,
            "heads": list(HEADS),
            "unknown_to_negative": False,
            "future_fields_used": False,
            "outcome_fields_used": False,
            "protected_reads": 0,
            "teacher_labels_generated": True,
            "formal_training_authorized": False,
            "formal_inference_authorized": False,
            "attack_authorized": False,
            "authorization_boundary": "FIT_TO_TEACHER_TRANSITION_V1 only; no Student, rollout, shadow, protected or attack permission",
        }
        (staging / "teacher_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "teacher_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        (staging / "source_episode_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in source_manifest), encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
        report["sha256sums_sha256"] = digest
        return report
    except Exception:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_FORMAL_TEACHER_FAILURE_V1", "input_status": EXPECTED_STATUS}, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--finalization-root", type=Path, required=True)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--fit-to-teacher-transition", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--teacher-runner", type=Path, default=Path(__file__))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "R3_DEV_PROTOCOL.json")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selection-sha256sums-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-sha256sums-sha256", required=True)
    parser.add_argument("--episode-seal-digest", required=True)
    parser.add_argument("--fit-to-teacher-transition-sha256sums-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.selection_manifest is None) != (args.selection_sha256sums_sha256 is None):
        parser.error("--selection-manifest and --selection-sha256sums-sha256 must be provided together")
    print(json.dumps(run(args.formal_root.resolve(), args.finalization_root.resolve(), args.transition.resolve(), args.input_audit.resolve(), args.fit_to_teacher_transition.resolve(), args.teacher_contract.resolve(), args.teacher_runner.resolve(), args.protocol.resolve(), args.output_root.resolve(), transition_digest=args.transition_sha256sums_sha256, episode_digest=args.episode_seal_digest, fit_to_teacher_transition_digest=args.fit_to_teacher_transition_sha256sums_sha256, selection_manifest_path=args.selection_manifest.resolve() if args.selection_manifest else None, selection_digest=args.selection_sha256sums_sha256, resume=args.resume), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
