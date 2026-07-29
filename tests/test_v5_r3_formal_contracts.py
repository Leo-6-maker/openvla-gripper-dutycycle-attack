from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from build_r3_fit_to_teacher_transition import PERMISSIONS, build  # noqa: E402
from build_r3_teacher_pilot_manifest import select_task_balanced_bindings, validate_task_groups  # noqa: E402
from audit_r3_formal_input import BINDING_FIELDS, _canonical_digest, _validate_episode_relations  # noqa: E402
import run_r3_v23_formal_teacher as formal_runner  # noqa: E402


GIT_SHA = "a" * 40
TREE_SHA = "b" * 40
DIGEST = "c" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    digest = _sha(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _fixture(tmp_path: Path):
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent_path = parent_root / "FIT670_INFERENCE_TRANSITION.json"
    parent = {
        "schema": "FIT670_INFERENCE_TRANSITION_V2",
        "collection_mode": "formal",
        "status": "FROZEN_BEFORE_EXECUTION",
        "max_episodes": 670,
        "n_shards": 8,
        "identity_set_frozen": True,
        "authorized_identities": 670,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "attack_authorized": False,
        "protected_payload_read": False,
        "protected_overlap_verified": 0,
        "collection_source_commit": GIT_SHA,
        "collection_source_tree": TREE_SHA,
        "identity_set_digest": DIGEST,
        "collection_source_commit": GIT_SHA,
        "collection_source_tree": TREE_SHA,
        "created_at": "2026-07-29T00:00:00+00:00",
    }
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    parent_seal = _seal(parent_root)

    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    finalization_root = tmp_path / "finalization"
    finalization_root.mkdir()
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    bindings = {}
    for i in range(670):
        identity = f"libero_10/task_{i // 10:02d}/state_{i % 10:02d}"
        bindings[identity] = {
            "suite": "libero_10", "task_id": i // 10, "task_name": "task", "state_id": i % 10,
            "seed": 20260717, "episode_id": identity, "initial_state_sha256": DIGEST,
            "relative_path": f"episodes/{identity}/episode.json", "episode_sha256": DIGEST,
            "episode_sha256sums_sha256": DIGEST, "worker_id": "gpu_0", "shard_id": 0,
            "worker_result_target": f"/tmp/formal/{identity}", "worker_result_steps": 1,
            "worker_result_source_sha256": DIGEST,
            "worker_result_episode_sha256sums_sha256": DIGEST,
            "worker_result_initial_state_sha256": DIGEST,
            "worker_result_binding_mode": "RESULT_TARGET_STEPS_JOINED_TO_SEALED_EPISODE",
            "worker_manifest_sha256": DIGEST, "worker_seal_sha256sums_sha256": DIGEST,
            "collection_source_commit": GIT_SHA, "collection_source_tree": TREE_SHA,
            "collector_script_sha256": DIGEST, "transition_manifest_sha256": DIGEST,
            "transition_sha256sums_sha256": DIGEST, "allowlist_sha256": DIGEST,
            "c1_canonical_digest": DIGEST, "schema": "FIT670_EPISODE_V2",
        }
    binding_rows = [{key: bindings[identity][key] for key in ("suite", "task_id", "task_name", "state_id", "seed", "episode_id", "initial_state_sha256", "relative_path", "episode_sha256", "episode_sha256sums_sha256", "worker_id", "shard_id", "worker_result_target", "worker_result_steps", "worker_result_source_sha256", "worker_result_episode_sha256sums_sha256", "worker_result_initial_state_sha256", "worker_result_binding_mode", "worker_manifest_sha256", "worker_seal_sha256sums_sha256", "collection_source_commit", "collection_source_tree", "collector_script_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256", "allowlist_sha256", "c1_canonical_digest", "schema")} for identity in sorted(bindings)]
    binding_digest = hashlib.sha256(json.dumps(binding_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity_rows = [{"episode_id": bindings[identity]["episode_id"], "suite": bindings[identity]["suite"], "task_id": bindings[identity]["task_id"], "state_id": bindings[identity]["state_id"], "collection_seed": bindings[identity]["seed"], "initial_state_sha256": bindings[identity]["initial_state_sha256"]} for identity in sorted(bindings)]
    identity_digest = hashlib.sha256(json.dumps(identity_rows, sort_keys=True).encode()).hexdigest()
    episode_digest = hashlib.sha256(json.dumps({identity: bindings[identity]["episode_sha256sums_sha256"] for identity in sorted(bindings)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit_manifest = {
        "schema": "V5_R3_FORMAL_INPUT_AUDIT_V1",
        "status": "PASS_FORMAL_INPUT_CONSUMABLE",
        "formal_root": str(formal_root),
        "episode_count": 670,
        "protected_reads": 0,
        "teacher_labels_generated": False,
        "labels_generated": False,
        "student_started": False,
        "attack_authorized": False,
        "source_staging_residue": [],
        "identity_set_digest": identity_digest,
        "collection_source_commit": GIT_SHA,
        "collection_source_tree": TREE_SHA,
        "transition_manifest_sha256": _sha(parent_path),
        "episode_binding_digest": binding_digest,
        "episode_bindings": bindings,
        "finalization": {
            "root": str(finalization_root),
            "episode_seal_digest": episode_digest,
        },
        "gate": {key: 0 for key in ("duplicate", "missing", "extra", "unallowlisted", "bad_episode_seal", "bad_worker_seal", "schema_error", "empty_entity_records", "identity_binding_error", "source_binding_error", "nonfinite", "staging_residue", "protected_reads")},
    }
    (audit_root / "FORMAL_INPUT_MANIFEST.json").write_text(json.dumps(audit_manifest), encoding="utf-8")
    audit_seal = _seal(audit_root)
    contract = tmp_path / "teacher_contract.json"
    contract.write_text(json.dumps({"schema": "V23_TEACHER_CONTRACT_V1"}), encoding="utf-8")
    runner = tmp_path / "teacher_runner.py"
    runner.write_text("# frozen runner\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    return parent_path, parent_seal, audit_root, audit_seal, contract, runner, protocol, formal_root, tmp_path / "teacher_transition"


def test_fit_to_teacher_positive_path_binds_exact_permissions(tmp_path):
    parent, parent_seal, audit, audit_seal, contract, runner, protocol, formal, output = _fixture(tmp_path)
    report = build(parent, audit, contract, runner, protocol, formal, output, expected_parent_seal=parent_seal, expected_audit_seal=audit_seal, runner_commit=GIT_SHA, runner_tree=TREE_SHA)
    assert report["status"] == "PASS_FIT_TO_TEACHER_AUTHORIZATION"
    saved = json.loads((output / "FIT_TO_TEACHER_TRANSITION.json").read_text())
    assert saved["permissions"] == PERMISSIONS
    assert saved["labels_generated"] is False
    assert saved["attack_authorized"] is False


def test_fit_to_teacher_rejects_tampered_parent_seal(tmp_path):
    parent, parent_seal, audit, audit_seal, contract, runner, protocol, formal, output = _fixture(tmp_path)
    parent.write_text(parent.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent transition seal mismatch"):
        build(parent, audit, contract, runner, protocol, formal, output, expected_parent_seal=parent_seal, expected_audit_seal=audit_seal, runner_commit=GIT_SHA, runner_tree=TREE_SHA)


def test_fit_to_teacher_nonoverwrite(tmp_path):
    parent, parent_seal, audit, audit_seal, contract, runner, protocol, formal, output = _fixture(tmp_path)
    build(parent, audit, contract, runner, protocol, formal, output, expected_parent_seal=parent_seal, expected_audit_seal=audit_seal, runner_commit=GIT_SHA, runner_tree=TREE_SHA)
    with pytest.raises(FileExistsError):
        build(parent, audit, contract, runner, protocol, formal, output, expected_parent_seal=parent_seal, expected_audit_seal=audit_seal, runner_commit=GIT_SHA, runner_tree=TREE_SHA)


def test_formal_runner_has_nonempty_generation_contract():
    source = (ROOT / "scripts" / "detector_v5" / "run_r3_v23_formal_teacher.py").read_text(encoding="utf-8")
    assert "not generation_passes" in source
    assert "len(generation_passes) != expected_steps" in source
    assert "teacher_labels_authorized" in source


def test_formal_runner_rejects_future_outcome_fields():
    with pytest.raises(ValueError, match="forbidden future/outcome field"):
        formal_runner._reject_forbidden_fields({"steps": [{"outcome": "success"}]}, "episode")


def test_formal_runner_resume_label_requires_provenance(tmp_path):
    root = tmp_path / "episode_label"
    root.mkdir()
    binding = {
        "source_episode_sha256": DIGEST,
        "source_episode_sha256sums_sha256": DIGEST,
        "teacher_contract_sha256": DIGEST,
        "protocol_sha256": DIGEST,
        "input_audit_manifest_sha256": DIGEST,
        "input_audit_seal_sha256sums_sha256": DIGEST,
        "fit_to_teacher_transition_manifest_sha256": DIGEST,
        "fit_to_teacher_transition_seal_sha256sums_sha256": DIGEST,
    }
    manifest = {"schema": "V5_R3_V23_TEACHER_EPISODE_V1", "status": "PASS_EPISODE_TEACHER_LABELS", "episode_id": "e", "step_count": 1, "unknown_to_negative": False, "future_fields_used": False, "outcome_fields_used": False, **binding}
    (root / "EPISODE_TEACHER_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "teacher_records.jsonl").write_text(json.dumps({"episode_id": "e", "step": 0}) + "\n", encoding="utf-8")
    _seal(root)
    rows, _ = formal_runner._load_sealed_episode_labels(root, "e", binding)
    assert rows == [{"episode_id": "e", "step": 0}]
    with pytest.raises(ValueError, match="provenance mismatch"):
        formal_runner._load_sealed_episode_labels(root, "e", {**binding, "protocol_sha256": "d" * 64})


def test_empty_relations_are_only_valid_for_explicit_not_applicable_geometry():
    assert _validate_episode_relations({"relations": [], "geometry_status": "NOT_APPLICABLE"}, "e") == (0, "NOT_APPLICABLE")
    with pytest.raises(ValueError, match="empty relation records"):
        _validate_episode_relations({"relations": [], "geometry_status": "OK"}, "e")


def test_t0_binding_digest_uses_the_shared_required_field_contract():
    row = {key: (str(i) if key != "worker_result_steps" else 1) for i, key in enumerate(BINDING_FIELDS)}
    expected = hashlib.sha256(json.dumps([row], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert _canonical_digest([row]) == expected


def test_pilot_selection_is_exactly_one_identity_per_task_and_deterministic():
    bindings = {
        f"suite_{suite}/task_{task:02d}/state_{state:02d}": {
            "episode_id": f"suite_{suite}/task_{task:02d}/state_{state:02d}",
            "suite": f"suite_{suite}",
            "task_id": task,
        }
        for suite in range(4)
        for task in range(10)
        for state in range(2)
    }
    first = select_task_balanced_bindings(bindings, 20260717)
    second = select_task_balanced_bindings(bindings, 20260717)
    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert len(first) == 40
    assert len({(row["suite"], row["task_id"]) for row in first}) == 40


def test_pilot_selection_rejects_incomplete_or_misnumbered_task_grid():
    rows = [{"suite": "suite_0", "task_id": 0, "episode_id": "e"}]
    with pytest.raises(ValueError, match="4 suites x 10 tasks"):
        validate_task_groups(rows)


def test_t1_runner_rejects_unselected_full_formal_path():
    with pytest.raises(ValueError, match="selected pilot manifest is required"):
        formal_runner._require_pilot_selection(None, None)
    with pytest.raises(ValueError, match="selected pilot manifest is required"):
        formal_runner._require_pilot_selection(Path("selection.json"), None)
