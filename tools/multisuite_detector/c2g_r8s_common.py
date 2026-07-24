"""Shared constants and deterministic writers for R8S."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "c2g.r8s.semantic_replay.2026-07-11.v1"
ATTACK_EVAL = "ATTACK_EVAL_PREREGISTERED"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
GO_REPLAY = "GO_DETERMINISTIC_REPLAY_CANARY"
GO_PARTIAL = "GO_PARTIAL_DETERMINISTIC_REPLAY_CANARY"
GO_AUX = "GO_AUXILIARY_LEGACY_SUPERVISION_ONLY"
HOLD_INPUT = "HOLD_R8R_INPUT_INTEGRITY"
HOLD_NONE = "HOLD_NO_DEFENSIBLE_SEMANTIC_OR_REPLAY_SALVAGE"
LEGACY = (
    "teacher_phase", "teacher_event_role", "teacher_primary_attackable",
    "teacher_release_safe", "teacher_hazard", "teacher_grounded_object",
    "teacher_target_match", "teacher_reason_code",
    "teacher_used_absolute_z_fallback", "corridor_label", "release_label",
)
MAPPINGS = (
    ("teacher_primary_attackable OR event_role=primary_attackable", "y_gripper_critical_window", "PARTIAL_PROXY", "auxiliary supervision", "not exact Teacher-v2 critical-window ground truth"),
    ("teacher_release_safe OR phase=release_safe", "y_release_safe", "PARTIAL_PROXY", "auxiliary release supervision", "not exact Teacher-v2 release-safe ground truth"),
    ("teacher_phase", "teacher_phase", "AUXILIARY_ONLY", "phase pretraining / legacy replication", "not current ontology equivalence"),
    ("teacher_event_role", "target relevance / grounding", "AUXILIARY_ONLY", "event-role auxiliary head", "not resolved target identity"),
    ("teacher_hazard OR corridor_label", "gripper dependency / criticality", "PARTIAL_PROXY", "legacy hazard auxiliary target", "not mechanism-grounded Teacher-v2 dependency"),
)
EP_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "metadata_path", "step_records_path", "episode_read_ok", "read_error",
    "step_count", "step_contiguous", "legacy_any_present", *[f"{x}_present" for x in LEGACY],
    "primary_comparable_steps", "primary_disagreement_steps",
    "release_comparable_steps", "release_disagreement_steps",
    "full_action_7d_complete", "partial_action_4d_complete", "action_vector_key",
    "action_dimension_min", "action_dimension_max", "official_init_state_reference_present",
    "libero_version_bound", "runtime_versions_bound", "controller_config_bound",
    "action_semantics_bound", "task_bddl_bound", "seed_bound", "max_steps_bound",
    "strict_replay_candidate", "strict_replay_ready", "replay_blockers",
    "legacy_auxiliary_eligible", "current_teacher_v2_exact_supervision_eligible",
)
FIELD_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split", *LEGACY,
    "full_action_7d_complete", "partial_action_4d_complete",
    "official_init_state_reference_present", "libero_version_bound",
    "runtime_versions_bound", "controller_config_bound", "action_semantics_bound",
    "task_bddl_bound", "seed_bound", "max_steps_bound", "strict_replay_ready",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_r8r(root: Path, expected_report_sha: str):
    names = (
        "clean2000_r7_reuse_audit_report.json",
        "clean2000_r7_episode_ledger.csv",
        "SHA256SUMS",
        "SHA256SUMS.sha256",
    )
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    report_path = root / names[0]
    if sha256_file(report_path) != expected_report_sha:
        raise ValueError("R8R report hash mismatch")
    declared = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"R8R SHA256SUMS mismatch for {name}")
        declared[name] = digest
    digest, name = (root / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()
    if name != "SHA256SUMS" or sha256_file(root / name) != digest:
        raise ValueError("R8R SHA256SUMS self binding mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("final_decision") != "HOLD_TEACHER_V2_RAW_EVIDENCE":
        raise ValueError("R8S requires HOLD_TEACHER_V2_RAW_EVIDENCE input")
    with (root / names[1]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(report.get("canonical_registered_identities", -1)):
        raise ValueError("R8R episode ledger cardinality mismatch")
    return report, rows, declared
