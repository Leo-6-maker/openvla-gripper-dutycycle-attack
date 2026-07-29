"""Freeze one deterministic FIT-only identity per suite/task for the R3 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from audit_r3_formal_input import BINDING_FIELDS, _canonical_digest, _write_seal
from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


EXPECTED_TASK_COUNT = 40
EXPECTED_IDENTITY_COUNT = 40
EXPECTED_SUITE_COUNT = 4
EXPECTED_TASK_IDS = set(range(10))


def validate_task_groups(rows: list[Mapping[str, Any]]) -> None:
    groups = {(str(row.get("suite")), int(row.get("task_id"))) for row in rows}
    suites = {suite for suite, _ in groups}
    if len(groups) != EXPECTED_TASK_COUNT or len(suites) != EXPECTED_SUITE_COUNT:
        raise ValueError("pilot task closure is not 4 suites x 10 tasks")
    if any({task for suite, task in groups if suite == name} != EXPECTED_TASK_IDS for name in suites):
        raise ValueError("pilot task ids are not exactly 0..9 in every suite")


def select_task_balanced_bindings(bindings: Mapping[str, Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for identity, row in bindings.items():
        if not isinstance(row, Mapping) or row.get("episode_id") != identity:
            raise ValueError(f"pilot binding identity mismatch: {identity}")
        try:
            group = (str(row["suite"]), int(row["task_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"pilot binding task key missing: {identity}") from exc
        groups[group].append(dict(row))
    if any(not rows for rows in groups.values()):
        raise ValueError(f"pilot task closure is not {EXPECTED_TASK_COUNT} nonempty groups")
    selected = []
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda row: (hashlib.sha256(f"{seed}||{row['episode_id']}".encode()).hexdigest(), row["episode_id"]))
        selected.append(rows[0])
    if len(selected) != EXPECTED_IDENTITY_COUNT or len({row["episode_id"] for row in selected}) != EXPECTED_IDENTITY_COUNT:
        raise ValueError("pilot selection is not unique 40 identities")
    validate_task_groups(selected)
    return selected


def _validate_audit_closure(manifest: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]]) -> None:
    gate = manifest.get("gate")
    required_gate = {
        "duplicate", "missing", "extra", "unallowlisted", "bad_episode_seal", "bad_worker_seal",
        "schema_error", "empty_entity_records", "identity_binding_error", "source_binding_error",
        "nonfinite", "staging_residue", "protected_reads",
    }
    if not isinstance(gate, Mapping) or set(gate) != required_gate or any(gate[key] != 0 for key in required_gate):
        raise ValueError("T0-A gate closure is incomplete")
    if manifest.get("source_staging_residue") != []:
        raise ValueError("T0-A source staging is not empty")
    required_source_fields = (
        "identity_set_digest", "episode_binding_digest", "collection_source_commit", "collection_source_tree",
        "transition_manifest_sha256", "transition_sha256sums_sha256", "allowlist_sha256",
        "allowlist_root_sha256sums_sha256", "shard_plan_sha256", "worker_closure", "finalization",
    )
    if any(not manifest.get(field) for field in required_source_fields):
        raise ValueError("T0-A source/allowlist/shard closure is incomplete")
    rows = []
    for identity in sorted(bindings):
        row = bindings[identity]
        if not isinstance(row, Mapping) or row.get("episode_id") != identity:
            raise ValueError(f"T0-A binding identity mismatch: {identity}")
        if any(field not in row for field in BINDING_FIELDS):
            raise ValueError(f"T0-A binding field missing: {identity}")
        rows.append(dict(row))
    if _canonical_digest(rows) != manifest.get("episode_binding_digest"):
        raise ValueError("T0-A binding digest mismatch")
    identity_rows = [
        {"episode_id": row["episode_id"], "suite": row["suite"], "task_id": row["task_id"],
         "state_id": row["state_id"], "collection_seed": row["seed"],
         "initial_state_sha256": row["initial_state_sha256"]}
        for row in rows
    ]
    identity_digest = hashlib.sha256(json.dumps(identity_rows, sort_keys=True).encode()).hexdigest()
    if identity_digest != manifest.get("identity_set_digest"):
        raise ValueError("T0-A identity digest mismatch")
    finalization = manifest.get("finalization")
    if not isinstance(finalization, Mapping) or not isinstance(finalization.get("episode_seals"), Mapping):
        raise ValueError("T0-A finalization closure is incomplete")
    if set(finalization["episode_seals"]) != set(bindings):
        raise ValueError("T0-A finalization identity closure mismatch")
    episode_digest = hashlib.sha256(json.dumps(dict(sorted(finalization["episode_seals"].items())), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if episode_digest != finalization.get("episode_seal_digest"):
        raise ValueError("T0-A episode seal digest mismatch")
    workers = manifest.get("worker_closure")
    if not isinstance(workers, list) or len(workers) != 8 or sum(int(item.get("count", -1)) for item in workers) != EXPECTED_IDENTITY_COUNT:
        raise ValueError("T0-A worker closure is incomplete")


def build(audit_root: Path, output_root: Path, *, seed: int, expected_audit_seal: str) -> dict[str, Any]:
    audit_root = audit_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() or output_root.parent != audit_root.parent:
        raise FileExistsError("pilot output must be a new sibling of the audit root")
    if audit_root.is_symlink() or not audit_root.is_dir() or any(path.is_symlink() for path in audit_root.rglob("*")):
        raise ValueError("T0-A audit root is missing, symlinked, or not a closed regular-file root")
    if any(part.lower() in {"cal", "check", "g10", "t2r-d", "protected", "attack"} for part in audit_root.parts):
        raise ValueError("T0-A audit root is forbidden-looking")
    audit_seal = verify_seal(audit_root)
    if audit_seal["sha256sums_sha256"] != expected_audit_seal:
        raise ValueError("T0-A audit seal mismatch")
    manifest_path = audit_root / "FORMAL_INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE":
        raise ValueError("T0-A audit is not consumable")
    if (
        manifest.get("episode_count") != 670
        or manifest.get("protected_reads") != 0
        or manifest.get("teacher_labels_generated") is not False
        or manifest.get("labels_generated") is not False
        or manifest.get("student_started") is not False
        or manifest.get("attack_authorized") is not False
    ):
        raise ValueError("T0-A boundary/cardinality is not closed")
    bindings = manifest.get("episode_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 670 or not bindings:
        raise ValueError("T0-A bindings are incomplete")
    _validate_audit_closure(manifest, bindings)
    selected = select_task_balanced_bindings(bindings, seed)
    selected_digest = _canonical_digest(selected)
    report = {
        "schema": "V5_R3_TEACHER_PILOT_MANIFEST_V1",
        "status": "PASS_FROZEN_TEACHER_PILOT_INPUT",
        "selection_algorithm": "sha256(seed || episode_id) ascending within sorted (suite, task_id) groups",
        "selection_seed": seed,
        "task_count": EXPECTED_TASK_COUNT,
        "identity_count": EXPECTED_IDENTITY_COUNT,
        "selected_identity_digest": selected_digest,
        "selected_bindings": selected,
        "formal_root": manifest["formal_root"],
        "input_audit_root": str(audit_root),
        "input_audit_manifest_sha256": sha256_file(manifest_path),
        "input_audit_seal_sha256sums_sha256": audit_seal["sha256sums_sha256"],
        "identity_set_digest": manifest["identity_set_digest"],
        "episode_seal_digest": manifest["finalization"]["episode_seal_digest"],
        "episode_binding_digest": manifest["episode_binding_digest"],
        "transition_manifest_sha256": manifest["transition_manifest_sha256"],
        "transition_sha256sums_sha256": manifest["transition_sha256sums_sha256"],
        "allowlist_sha256": manifest["allowlist_sha256"],
        "allowlist_root_sha256sums_sha256": manifest["allowlist_root_sha256sums_sha256"],
        "shard_plan_sha256": manifest["shard_plan_sha256"],
        "worker_closure_digest": hashlib.sha256(json.dumps(manifest["worker_closure"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "collection_source_commit": manifest["collection_source_commit"],
        "collection_source_tree": manifest["collection_source_tree"],
        "protected_reads": 0,
        "teacher_labels_generated": False,
        "labels_generated": False,
        "student_started": False,
        "training_authorized": False,
        "attack_authorized": False,
    }
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "PILOT_MANIFEST.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "SELECTION_ROWS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    (staging / "SOURCE_BINDING.json").write_text(json.dumps({"audit_manifest_sha256": report["input_audit_manifest_sha256"], "audit_seal_sha256sums_sha256": report["input_audit_seal_sha256sums_sha256"], "formal_root": report["formal_root"], "payload_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = sha256_file(output_root / "SHA256SUMS")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--expected-audit-seal", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.audit_root, args.output_root, seed=args.seed, expected_audit_seal=args.expected_audit_seal), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
