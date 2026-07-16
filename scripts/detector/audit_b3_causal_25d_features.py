#!/usr/bin/env python3
"""Fail-closed, read-only audit of B3 multi-event 25D reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_PATH.parent))

from audit_b3_legacy_generation_evidence import _checksum_ok  # noqa: E402
from gripper_attack.b3_causal_25d import (  # noqa: E402
    ACTION_PARITY_TOLERANCE,
    B3Causal25DMultieventV1,
    FEATURE_NAMES,
    LEGACY_SOURCE_FEATURE_NAMES_25D,
    LEGACY_SOURCE_FEATURE_ORDER_SHA256,
    SCHEMA,
    SOURCE_SCHEMA,
    STUDENT_FORBIDDEN_FEATURE_NAMES,
    ROBOT_EEF_PARITY_TOLERANCE,
    ROBOT_QPOS_PARITY_TOLERANCE,
    serialize_student_25d,
)


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EXPECTED_METADATA_SCHEMA = "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2"
TIME_INDEX = FEATURE_NAMES.index("time_since_close")
EEF_Z_DELTA_INDEX = FEATURE_NAMES.index("eef_z_delta_since_close")
FLIP_INDEX = FEATURE_NAMES.index("recent_gripper_flip_count")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row: {path}")
                rows.append(value)
    return rows


def git_provenance(repo: Path, expected_head: str) -> dict[str, object]:
    repo = repo.resolve()
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    try:
        relative_script = SCRIPT_PATH.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError(f"audit script is outside runner repository: {SCRIPT_PATH}") from exc
    tracked = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{relative_script}"],
        capture_output=True,
        text=True,
    ).returncode == 0
    return {
        "expected_head": expected_head,
        "actual_head": actual,
        "worktree_clean": not bool(status),
        "script_tracked_at_head": tracked,
        "provenance_pass": actual == expected_head and not status and tracked,
        "script_relative_path": relative_script,
    }


def expected_split(state_id: int) -> str:
    if state_id <= 23:
        return "FIT"
    if state_id <= 26:
        return "CAL"
    if state_id <= 29:
        return "CHECK"
    return "FINAL_EVAL_CANDIDATE"


def load_canonical_manifest(path: Path, global_audit_path: Path, state_max: int) -> tuple[dict[str, dict], str, str]:
    manifest_sha = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != manifest_sha:
        raise ValueError("canonical manifest SHA sidecar mismatch")
    global_audit = load_json(global_audit_path)
    if global_audit.get("status") != "OFFICIAL_CLEAN_QUEUE_FROZEN" or global_audit.get("rows") != 2000:
        raise ValueError("global CLEAN queue audit is not frozen at 2000 rows")
    if global_audit.get("manifest_sha256") != manifest_sha:
        raise ValueError("global CLEAN queue audit is not bound to canonical manifest")
    expected: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("canonical_parent_key", "")
            if not key or key in expected:
                raise ValueError(f"duplicate or missing canonical key in manifest: {key!r}")
            state_id = int(row["state_id"])
            if state_id <= state_max:
                if row.get("split") != expected_split(state_id):
                    raise ValueError(f"canonical manifest split mismatch: {key}")
                expected[key] = row
    expected_count = 40 * (state_max + 1)
    if len(expected) != expected_count:
        raise ValueError(f"canonical manifest subset count {len(expected)} != {expected_count}")
    return expected, manifest_sha, sha256_file(global_audit_path)


def source_read_scope_sha(artifact: Path) -> str:
    payload = load_json(artifact / "artifact_sha256.json")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("artifact checksum files list missing")
    entries = []
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe artifact checksum path")
        path = artifact / relative
        if not path.is_file():
            raise ValueError(f"missing artifact checksum target: {relative}")
        entries.append({"path": relative.as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest_path = artifact / "artifact_sha256.json"
    entries.append({"path": manifest_path.name, "sha256": sha256_file(manifest_path), "size": manifest_path.stat().st_size})
    return json_sha(sorted(entries, key=lambda item: item["path"]))


def metadata_errors(metadata: dict, expected: dict, key: str) -> list[str]:
    errors = []
    if metadata.get("canonical_parent_key") != key:
        errors.append("IDENTITY_CANONICAL_KEY")
    if metadata.get("suite") != expected.get("suite"):
        errors.append("IDENTITY_SUITE")
    if int(metadata.get("task_idx", -1)) != int(expected.get("task_idx", -2)):
        errors.append("IDENTITY_TASK")
    if int(metadata.get("state_id", -1)) != int(expected.get("state_id", -2)):
        errors.append("IDENTITY_STATE")
    if metadata.get("condition") != "CLEAN":
        errors.append("CONDITION_NOT_CLEAN")
    if metadata.get("runtime_valid") is not True:
        errors.append("RUNTIME_INVALID")
    if metadata.get("schema") != EXPECTED_METADATA_SCHEMA:
        errors.append("METADATA_SCHEMA")
    if metadata.get("split") != expected_split(int(expected["state_id"])):
        errors.append("SPLIT_MISMATCH")
    if str(metadata.get("official_horizon")) != str(expected.get("official_horizon")):
        errors.append("HORIZON_MISMATCH")
    if metadata.get("initial_state_sha256") != expected.get("initial_state_sha256"):
        errors.append("INITIAL_STATE_MISMATCH")
    if metadata.get("feature_names_25d") != list(LEGACY_SOURCE_FEATURE_NAMES_25D):
        errors.append("FEATURE_ORDER_METADATA")
    return errors


def semantic_invariants(rebuilt: dict) -> dict[str, object]:
    rows = rebuilt["rows"]
    events = rebuilt["events"]
    violations: list[str] = []
    onset_rows = [row for row in rows if row.get("close_onset")]
    reset_rows = [row for row in rows if row.get("event_local_state_reset")]
    if [event["event_id"] for event in events] != list(range(len(events))):
        violations.append("EVENT_IDS_NOT_CONTIGUOUS")
    if len(onset_rows) != len(events):
        violations.append("ONSET_EVENT_COUNT_MISMATCH")
    released_events = sum(event.get("release_step") is not None for event in events)
    if len(reset_rows) != released_events:
        violations.append("RESET_RELEASE_COUNT_MISMATCH")
    event_by_id = {event["event_id"]: event for event in events}
    for row in rows:
        if row.get("event_id") != row.get("event_ordinal"):
            violations.append(f"EVENT_ORDINAL_MISMATCH:{row.get('step')}")
        if bool(row.get("release_onset")) != bool(row.get("event_local_state_reset")):
            violations.append(f"RELEASE_RESET_MISMATCH:{row.get('step')}")
        if row.get("close_onset"):
            event = event_by_id.get(row.get("event_id"))
            if event is None or event.get("start_step") != row.get("step"):
                violations.append(f"ONSET_EVENT_BINDING:{row.get('step')}")
        if row.get("release_onset"):
            released_id = row.get("released_event_id")
            event = event_by_id.get(released_id)
            if event is None or event.get("release_step") != row.get("step"):
                violations.append(f"RELEASE_EVENT_BINDING:{row.get('step')}")
        if row.get("event_active"):
            event = event_by_id.get(row.get("event_id"))
            if event is None or not event.get("start_step") <= row.get("step") <= event.get("end_step"):
                violations.append(f"ACTIVE_EVENT_BINDING:{row.get('step')}")
    for row in onset_rows:
        vector = row.get("features_25d") or []
        if len(vector) != 25 or vector[TIME_INDEX] != 0.0 or vector[EEF_Z_DELTA_INDEX] != 0.0:
            violations.append(f"ONSET_LOCAL_RESET:{row.get('step')}")
    for row in reset_rows:
        vector = row.get("features_25d") or []
        if row.get("event_active") or len(vector) != 25 or vector[TIME_INDEX] != -1.0:
            violations.append(f"RELEASE_LOCAL_RESET:{row.get('step')}")
    for row in rows:
        vector = row.get("features_25d")
        if row.get("valid") and (not isinstance(vector, list) or not 0.0 <= vector[FLIP_INDEX] <= 16.0):
            violations.append(f"FLIP_RANGE:{row.get('step')}")
    previous_end = -1
    for event in events:
        if event["start_step"] <= previous_end:
            violations.append(f"EVENT_INTERVAL_OVERLAP:{event['event_id']}")
        if event["end_step"] < event["start_step"]:
            violations.append(f"EVENT_INTERVAL_INVALID:{event['event_id']}")
        previous_end = event["end_step"]
    return {
        "command_event_count": len(events),
        "completed_release_count": released_events,
        "later_command_event_count": max(0, len(events) - 1),
        "later_event_rows": sum(row.get("event_id", -1) >= 1 for row in rows),
        "onset_count": len(onset_rows),
        "reset_count": len(reset_rows),
        "violations": violations,
    }


def robot_parity_summary(records: list[dict]) -> dict[str, object]:
    """Report the measured robot/action alias errors checked by the builder."""
    max_eef = max_qpos = max_opening = max_action = 0.0
    for row in records:
        features = row["features_25d"]
        eef = row["robot0_eef_pos"]
        qpos = row["robot0_gripper_qpos"]
        max_eef = max(max_eef, *(abs(float(features[3 + i]) - float(eef[i])) for i in range(3)))
        qpos_sum = float(qpos[0]) + float(qpos[1])
        opening = abs(float(qpos[0])) + abs(float(qpos[1]))
        max_qpos = max(max_qpos, abs(float(features[1]) - qpos_sum))
        max_opening = max(max_opening, abs(float(features[2]) - opening))

        raw = row["action_raw"]
        env = row["action_env"]
        max_action = max(max_action, *(abs(float(raw[i]) - float(env[i])) for i in range(6)))
        max_action = max(max_action, abs(float(features[0]) - float(raw[-1])))
        max_action = max(max_action, abs(float(features[12]) - float(raw[-1])))
        for i, feature_index in enumerate((9, 10, 11)):
            max_action = max(max_action, abs(float(features[feature_index]) - float(raw[i])))
    return {
        "max_eef_abs_error": max_eef,
        "max_qpos_abs_error": max_qpos,
        "max_opening_abs_error": max_opening,
        "max_action_abs_error": max_action,
        "robot_alias_parity_pass": (
            max_eef <= ROBOT_EEF_PARITY_TOLERANCE
            and max_qpos <= ROBOT_QPOS_PARITY_TOLERANCE
            and max_opening <= ROBOT_QPOS_PARITY_TOLERANCE
            and max_action <= ACTION_PARITY_TOLERANCE
        ),
    }


def audit_artifact(artifact: Path, source_root: Path, expected: dict) -> dict:
    key = expected["canonical_parent_key"]
    result: dict[str, object] = {
        "canonical_parent_key": key,
        "suite": expected.get("suite"),
        "task_idx": int(expected.get("task_idx", -1)),
        "state_id": int(expected.get("state_id", -1)),
        "source_artifact_relative": str(artifact.relative_to(source_root)),
        "status": "HOLD",
    }
    before_scope = after_scope = None
    try:
        metadata = load_json(artifact / "episode_metadata.json")
        result["metadata_identity_contract_pass"] = not metadata_errors(metadata, expected, key)
        result["metadata_errors"] = metadata_errors(metadata, expected, key)
        result["checksum_closed"] = _checksum_ok(artifact)
        before_scope = source_read_scope_sha(artifact)
        steps = jsonl(artifact / "step_records.jsonl")
        sidecar = jsonl(artifact / "privileged_teacher_sidecar.jsonl")
        if len(steps) != len(sidecar):
            raise ValueError("step/sidecar length mismatch")
        records = []
        for step, privileged in zip(steps, sidecar):
            if step.get("step") != privileged.get("step"):
                raise ValueError("step/sidecar identity mismatch")
            if not isinstance(step.get("features_25d"), list) or len(step["features_25d"]) != 25:
                raise ValueError("missing or invalid sealed source features_25d")
            record = dict(step)
            for name in ("robot0_eef_pos", "robot0_gripper_qpos"):
                if name not in privileged:
                    raise ValueError(f"missing sidecar robot field: {name}")
                record[name] = privileged[name]
            record["feature_names_25d"] = metadata["feature_names_25d"]
            record["feature_order_sha256"] = LEGACY_SOURCE_FEATURE_ORDER_SHA256
            records.append(record)
        rebuilt = B3Causal25DMultieventV1().rebuild(records)
        valid_rows = [row for row in rebuilt["rows"] if row["valid"]]
        student_pass = True
        for row in valid_rows:
            student = {name: row[name] for name in ("schema", "source_schema", "valid", "features_25d")}
            serialize_student_25d(student)
        invariants = semantic_invariants(rebuilt)
        parity = robot_parity_summary(records)
        after_scope = source_read_scope_sha(artifact)
        feature_order_bound = bool(
            result["metadata_identity_contract_pass"]
            and all(row.get("feature_order_bound") for row in valid_rows)
        )
        feature_valid_pass = len(valid_rows) == len(rebuilt["rows"])
        source_scope_unchanged = before_scope == after_scope
        result.update({
            "feature_order_bound": feature_order_bound,
            **parity,
            "feature_valid_pass": feature_valid_pass,
            "student_projection_pass": student_pass,
            "step_count": len(rebuilt["rows"]),
            "valid_feature_rows": len(valid_rows),
            "invalid_feature_rows": len(rebuilt["rows"]) - len(valid_rows),
            "source_read_scope_before": before_scope,
            "source_read_scope_after": after_scope,
            "source_read_scope_unchanged": source_scope_unchanged,
            "feature_order_sha256": LEGACY_SOURCE_FEATURE_ORDER_SHA256,
            **{key: value for key, value in invariants.items() if key != "violations"},
            "semantic_invariant_violations": invariants["violations"],
            "status": "PASS" if (
                result["metadata_identity_contract_pass"]
                and result["checksum_closed"]
                and feature_order_bound
                and parity["robot_alias_parity_pass"]
                and feature_valid_pass
                and result["student_projection_pass"]
                and source_scope_unchanged
                and not invariants["violations"]
            ) else "HOLD",
        })
    except Exception as exc:  # report every identity, fail closed later
        result.update({"error": f"{type(exc).__name__}: {exc}", "status": "HOLD"})
        if before_scope is not None:
            result["source_read_scope_before"] = before_scope
        if after_scope is not None:
            result["source_read_scope_after"] = after_scope
            result["source_read_scope_unchanged"] = before_scope == after_scope
    return result


def run(
    source_root: Path,
    output_root: Path,
    state_max: int,
    expected_runner_head: str,
    runner_repo: Path,
    feature_config: Path,
    canonical_manifest: Path,
    canonical_global_audit: Path,
) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    config = load_json(feature_config)
    if (
        config.get("schema") != SCHEMA
        or config.get("status") != "FEATURE_CONTRACT_ONLY"
        or config.get("formal_training_authorized") is not False
        or config.get("attack_authorized") is not False
        or config.get("feature_names") != list(FEATURE_NAMES)
        or config.get("legacy_source_feature_names") != list(LEGACY_SOURCE_FEATURE_NAMES_25D)
        or config.get("legacy_source_feature_order_sha256") != LEGACY_SOURCE_FEATURE_ORDER_SHA256
        or config.get("required_measured_action_fields") != ["action_raw", "action_env"]
        or config.get("action_parity_tolerance") != ACTION_PARITY_TOLERANCE
        or config.get("robot_qpos_parity_tolerance") != ROBOT_QPOS_PARITY_TOLERANCE
        or config.get("robot_eef_parity_tolerance") != ROBOT_EEF_PARITY_TOLERANCE
    ):
        raise ValueError("feature config does not bind the B3 schema and order")
    provenance = git_provenance(runner_repo, expected_runner_head)
    expected, manifest_sha, global_audit_sha = load_canonical_manifest(canonical_manifest, canonical_global_audit, state_max)
    records = []
    for key in sorted(expected):
        artifact = source_root / key
        records.append(audit_artifact(artifact, source_root, expected[key]))

    output_root.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for row in records for key in row})
    census_path = output_root / "B3_CAUSAL_25D_FEATURE_AUDIT_V2.csv"
    with census_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    by_suite = {}
    by_task = {}
    for suite in SUITES:
        group = [row for row in records if row.get("suite") == suite]
        by_suite[suite] = {
            "identities": len(group),
            "pass": sum(row.get("status") == "PASS" for row in group),
            "hold": sum(row.get("status") != "PASS" for row in group),
            "command_event_count": sum(int(row.get("command_event_count", 0)) for row in group),
            "completed_release_count": sum(int(row.get("completed_release_count", 0)) for row in group),
            "later_command_event_count": sum(int(row.get("later_command_event_count", 0)) for row in group),
            "later_event_rows": sum(int(row.get("later_event_rows", 0)) for row in group),
        }
        for task in range(10):
            task_group = [row for row in group if int(row.get("task_idx", -1)) == task]
            name = f"{suite}/task_{task:02d}"
            by_task[name] = {
                "episodes": len(task_group),
                "episodes_with_later_command_events": sum(int(row.get("later_command_event_count", 0)) > 0 for row in task_group),
                "later_command_event_count": sum(int(row.get("later_command_event_count", 0)) for row in task_group),
                "later_event_rows": sum(int(row.get("later_event_rows", 0)) for row in task_group),
                "reset_violation_count": sum(len(row.get("semantic_invariant_violations", [])) for row in task_group),
            }

    l10_rows = [row for row in records if row.get("suite") == "libero_10"]
    l10_by_task = {}
    for task in range(10):
        task_group = [row for row in l10_rows if int(row.get("task_idx", -1)) == task]
        l10_by_task[f"task_{task:02d}"] = {
            "episodes": len(task_group),
            "command_event_count": sum(int(row.get("command_event_count", 0)) for row in task_group),
            "completed_release_count": sum(int(row.get("completed_release_count", 0)) for row in task_group),
            "later_command_event_count": sum(int(row.get("later_command_event_count", 0)) for row in task_group),
            "episodes_with_later_command_events": sum(
                int(row.get("later_command_event_count", 0)) > 0 for row in task_group
            ),
        }

    summary = {
        "schema": "B3_CAUSAL_25D_FEATURE_AUDIT_V2",
        "source_root": str(source_root),
        "state_range": [0, state_max],
        "identity_count": len(records),
        "unique_identity_count": len({row.get("canonical_parent_key") for row in records}),
        "checksum_closed": sum(row.get("checksum_closed") is True for row in records),
        "identity_contract_pass": sum(row.get("metadata_identity_contract_pass") is True for row in records),
        "feature_order_bound": sum(row.get("feature_order_bound") is True for row in records),
        "robot_alias_parity_pass": sum(row.get("robot_alias_parity_pass") is True for row in records),
        "feature_valid_pass": sum(row.get("feature_valid_pass") is True for row in records),
        "student_projection_pass": sum(row.get("student_projection_pass") is True for row in records),
        "source_read_scope_unchanged": all(row.get("source_read_scope_unchanged") is True for row in records),
        "semantic_invariant_violations": sum(len(row.get("semantic_invariant_violations", [])) for row in records),
        "forbidden_student_field_count": sum(name in STUDENT_FORBIDDEN_FEATURE_NAMES for name in FEATURE_NAMES),
        "by_suite": by_suite,
        "by_task": by_task,
        "l10_command_event_breakdown": l10_by_task,
        "canonical_manifest_sha256": manifest_sha,
        "canonical_global_audit_sha256": global_audit_sha,
        "feature_config_sha256": sha256_file(feature_config),
        "feature_order_sha256": LEGACY_SOURCE_FEATURE_ORDER_SHA256,
        "action_parity_tolerance": ACTION_PARITY_TOLERANCE,
        "robot_qpos_parity_tolerance": ROBOT_QPOS_PARITY_TOLERANCE,
        "robot_eef_parity_tolerance": ROBOT_EEF_PARITY_TOLERANCE,
        "builder_sha256": sha256_file(REPO_ROOT / "src" / "gripper_attack" / "b3_causal_25d.py"),
        "audit_script_sha256": sha256_file(SCRIPT_PATH),
        "runner_provenance": provenance,
        "source_teacher_labels_read": False,
        "source_object_state_used": False,
        "source_contact_pairs_used": False,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary["status"] = "PASS" if (
        summary["identity_count"] == 40 * (state_max + 1)
        and summary["unique_identity_count"] == summary["identity_count"]
        and summary["checksum_closed"] == summary["identity_count"]
        and summary["identity_contract_pass"] == summary["identity_count"]
        and summary["feature_order_bound"] == summary["identity_count"]
        and summary["robot_alias_parity_pass"] == summary["identity_count"]
        and summary["feature_valid_pass"] == summary["identity_count"]
        and summary["student_projection_pass"] == summary["identity_count"]
        and summary["source_read_scope_unchanged"]
        and summary["semantic_invariant_violations"] == 0
        and summary["forbidden_student_field_count"] == 0
        and provenance["provenance_pass"]
    ) else "HOLD"
    summary.pop("summary_value", None)
    summary_path = output_root / "B3_CAUSAL_25D_FEATURE_AUDIT_SUMMARY_V2.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            sums.append(f"{sha256_file(path)}  {path.name}\n")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("".join(sums), encoding="utf-8")
    (output_root / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(sums_path)}  {sums_path.name}\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-max", type=int, choices=range(50), default=19)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-global-audit", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.source_root, args.output_root, args.state_max,
        args.expected_runner_head, args.runner_repo, args.feature_config,
        args.canonical_manifest, args.canonical_global_audit,
    )
    print(json.dumps({"status": summary["status"], "identity_count": summary["identity_count"], "checksum_closed": summary["checksum_closed"]}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
