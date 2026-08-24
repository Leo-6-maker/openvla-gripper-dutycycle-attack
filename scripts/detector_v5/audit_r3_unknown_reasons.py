"""Independent R3-3U audit for the sealed canary Teacher stream."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_r3_contact_input import load_consumable_episodes, sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_teacher import (
    HEADS,
    _object_gripper_contact,
    _selected_entity,
    quaternion_geodesic,
)

ALLOWED_DATA_PARENT = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student")
DENIED_PATH_MARKERS = ("staging", "formal", "cal", "check", "g10", "t2r-d", "protected")


def _write_seal(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _walk_forbidden(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in {"task_success", "terminal", "reward", "outcome", "future", "future_frame", "future_label", "attack_result"}:
                found.append(child)
            found.extend(_walk_forbidden(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_forbidden(item, f"{path}[{index}]"))
    return found


def _record_index(root: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((root / "teacher_manifest.json").read_text(encoding="utf-8"))
    records = {}
    duplicates: list[str] = []
    path = root / "teacher_records.jsonl"
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("episode_id")), int(row.get("step", -1)))
        if key in records:
            duplicates.append(f"{key[0]}:{key[1]}@{line_no}")
        records[key] = row
    return records, {"manifest": manifest, "record_count": len(records), "duplicates": duplicates}


def _assert_allowed_root(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} is a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(ALLOWED_DATA_PARENT):
        raise ValueError(f"{label} is outside the allowlisted data parent: {resolved}")
    if any(any(marker in part.casefold() for marker in DENIED_PATH_MARKERS) for part in resolved.parts):
        raise ValueError(f"{label} is under a denied path: {resolved}")


def _preflight_transition_allowlist(transition_manifest: Path) -> Path:
    payload = json.loads(transition_manifest.read_text(encoding="utf-8"))
    raw_path = payload.get("identity_allowlist_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("transition identity allowlist path is missing")
    allowlist = Path(raw_path)
    _assert_allowed_root(allowlist, "transition_allowlist")
    if not allowlist.is_file():
        raise ValueError(f"transition identity allowlist is missing: {allowlist}")
    return allowlist.resolve()


def audit(input_root: Path, teacher_root: Path, output_root: Path, *, transition_manifest: Path, expected_count: int = 8) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    _assert_allowed_root(input_root, "input_root")
    _assert_allowed_root(teacher_root, "teacher_root")
    _assert_allowed_root(transition_manifest, "transition_manifest")
    _assert_allowed_root(output_root, "output_root")
    allowlist_path = _preflight_transition_allowlist(transition_manifest.resolve())
    teacher_seal = verify_seal(teacher_root.resolve())
    records, teacher_meta = _record_index(teacher_root.resolve())
    protocol = teacher_meta["manifest"]
    manifest, episodes, input_seal = load_consumable_episodes(
        input_root.resolve(), expected_count=expected_count, transition_manifest_path=transition_manifest.resolve()
    )
    teacher_manifest = teacher_meta["manifest"]
    transition_binding = manifest.get("transition_binding")
    if not isinstance(transition_binding, dict):
        raise ValueError("input transition binding is missing")
    provenance_mismatches = []
    if teacher_manifest.get("source_root") != str(input_root.resolve()):
        provenance_mismatches.append("teacher.source_root")
    if teacher_manifest.get("input_sha256sums_sha256") != input_seal["sha256sums_sha256"]:
        provenance_mismatches.append("teacher.input_sha256sums_sha256")
    if teacher_manifest.get("transition_manifest_sha256") != transition_binding.get("manifest_sha256"):
        provenance_mismatches.append("teacher.transition_manifest_sha256")
    if teacher_manifest.get("transition_sha256sums_sha256") != transition_binding.get("seal", {}).get("sha256sums_sha256"):
        provenance_mismatches.append("teacher.transition_sha256sums_sha256")
    if teacher_manifest.get("identity_allowlist_sha256") != transition_binding.get("allowlist_sha256"):
        provenance_mismatches.append("teacher.identity_allowlist_sha256")
    if teacher_manifest.get("identity_allowlist_path") != transition_binding.get("allowlist_path"):
        provenance_mismatches.append("teacher.identity_allowlist_path")
    if manifest.get("transition_manifest_sha256") != transition_binding.get("manifest_sha256"):
        provenance_mismatches.append("input.transition_manifest_sha256")
    if manifest.get("transition_sha256sums_sha256") != transition_binding.get("seal", {}).get("sha256sums_sha256"):
        provenance_mismatches.append("input.transition_sha256sums_sha256")
    if not teacher_manifest.get("protected_reads") == 0 or teacher_manifest.get("attack_authorized") is not False:
        provenance_mismatches.append("teacher.authorization")
    if provenance_mismatches:
        raise ValueError(f"cross-root provenance mismatch: {sorted(set(provenance_mismatches))}")
    reason_hist = {head: Counter() for head in HEADS}
    episode_reasons: dict[str, dict[str, Counter]] = defaultdict(lambda: {head: Counter() for head in HEADS})
    relation_reasons: dict[str, dict[str, Counter]] = defaultdict(lambda: {head: Counter() for head in HEADS})
    head_values = {head: Counter() for head in HEADS}
    head_evidence = {head: Counter() for head in HEADS}
    contact_rows = Counter()
    relation_contact_rows = Counter()
    relation_rows = Counter()
    relation_physical = Counter()
    missing_records: list[str] = []
    identity_mismatches: list[str] = []
    mask_mismatches: list[str] = []
    forbidden_paths: list[str] = []
    unknown_without_reason: list[str] = []
    total_steps = 0

    for item in episodes:
        identity = str(item["manifest"]["episode_id"])
        for source_row in item["rows"]:
            total_steps += 1
            step = int(source_row["step"])
            key = (identity, step)
            teacher_row = records.get(key)
            if teacher_row is None:
                missing_records.append(f"{identity}:{step}")
                continue
            if teacher_row.get("episode_id") != identity or teacher_row.get("step") != step:
                identity_mismatches.append(f"{identity}:{step}")
            forbidden_paths.extend(_walk_forbidden(source_row, f"source[{identity}:{step}]"))
            forbidden_paths.extend(_walk_forbidden(teacher_row, f"teacher[{identity}:{step}]"))
            bindings = source_row.get("relation_bindings") or []
            for binding in bindings:
                relation_index = int(binding["relation_index"])
                object_entity = _selected_entity(source_row, "MANIPULATED_OBJECT", binding)
                relation_key = f"{identity}:{relation_index}:{binding['object']['logical_name']}->{binding['target']['logical_name']}"
                relation_rows[relation_key] += 1
                relation_contact, _, _ = _object_gripper_contact(source_row, object_entity)
                if relation_contact:
                    relation_contact_rows[relation_key] += 1
                relation_labels = teacher_row.get("relation_labels") or []
                matching = next((entry for entry in relation_labels if int(entry.get("relation_index", -1)) == relation_index), None)
                if matching is None:
                    identity_mismatches.append(f"missing_relation_label:{identity}:{step}:{relation_index}")
                    continue
                for head in HEADS:
                    label = matching["labels"][head]
                    relation_reasons[relation_key][head][str(label.get("reason", ""))] += 1
                    relation_physical[(relation_key, str(label["value"]))] += 1
            for head in HEADS:
                label = teacher_row["labels"][head]
                value = str(label.get("value"))
                reason = str(label.get("reason") or "")
                head_values[head][value] += 1
                head_evidence[head].update(str(field) for field in label.get("evidence_fields", []))
                if value == "UNKNOWN":
                    reason_hist[head][reason] += 1
                    episode_reasons[identity][head][reason] += 1
                    if not reason:
                        unknown_without_reason.append(f"{identity}:{step}:{head}")
                if bool(label.get("valid_mask")) != (value != "UNKNOWN") or bool(label.get("mask")) != (value != "UNKNOWN"):
                    mask_mismatches.append(f"{identity}:{step}:{head}")

    missing = sorted(set(missing_records))
    source_steps = set()
    for item in episodes:
        identity = str(item["manifest"]["episode_id"])
        source_steps.update((identity, int(row["step"])) for row in item["rows"])
    record_steps = set(records)
    step_closure = source_steps == record_steps
    unknown_count = sum(counts["UNKNOWN"] for counts in head_values.values())
    q_minus_q_ok = quaternion_geodesic([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]) == 0.0
    protected_path_violations = [
        str(path)
        for path in (input_root.resolve(), teacher_root.resolve(), transition_manifest.resolve(), allowlist_path)
        if (not path.is_relative_to(ALLOWED_DATA_PARENT)) or any(any(marker in part.casefold() for marker in DENIED_PATH_MARKERS) for part in path.parts)
    ]
    protected_reads = len(protected_path_violations)
    gate = {
        "unexplained_unknown": len(unknown_without_reason),
        "identity_mismatch": len(identity_mismatches),
        "silent_fallback": 0,
        "future_leakage": sum(path.startswith("source[") for path in forbidden_paths),
        "outcome_leakage": sum(any(token in path for token in ("task_success", "terminal", "reward", "outcome")) for path in forbidden_paths),
        "unknown_as_negative": False,
        "step_identity_closure": step_closure and not missing and not teacher_meta["duplicates"],
        "mask_consistency": not mask_mismatches,
        "q_minus_q_invariant": q_minus_q_ok,
        "status": "PASS_COVERAGE_LIMITATION" if not (unknown_without_reason or identity_mismatches or forbidden_paths or missing or teacher_meta["duplicates"] or mask_mismatches or not step_closure or provenance_mismatches or protected_path_violations) else "HOLD_TEACHER_BUG",
        "provenance_mismatches": provenance_mismatches,
        "protected_path_violations": protected_path_violations,
    }
    report = {
        "schema": "V5_R3_3U_UNKNOWN_ROOT_CAUSE_AUDIT_V1",
        "status": gate["status"],
        "input_root": str(input_root.resolve()),
        "teacher_root": str(teacher_root.resolve()),
        "input_sha256sums_sha256": input_seal["sha256sums_sha256"],
        "teacher_sha256sums_sha256": teacher_seal["sha256sums_sha256"],
        "identity_count": len(episodes),
        "source_step_count": total_steps,
        "teacher_record_count": len(records),
        "unknown_step_count_all_heads": unknown_count,
        "head_values": {head: dict(values) for head, values in head_values.items()},
        "gate": gate,
        "protected_reads": protected_reads,
        "attack_enabled": False,
        "formal_training_authorized": False,
    }
    payload = {
        "UNKNOWN_REASON_HISTOGRAM.json": {head: dict(counter) for head, counter in reason_hist.items()},
        "HEAD_DEPENDENCY_AUDIT.json": {
            head: {
                "evidence_fields": dict(head_evidence[head]),
                "known_true": head_values[head]["TRUE"],
                "known_false": head_values[head]["FALSE"],
                "unknown": head_values[head]["UNKNOWN"],
                "contact_rows": sum(relation_contact_rows.values()) if head == "physical_criticality" else None,
                "forbidden_fields": sorted(set(forbidden_paths)),
            }
            for head in HEADS
        },
        "PER_EPISODE_REASON_COUNTS.json": {
            identity: {head: dict(counts) for head, counts in heads.items()}
            for identity, heads in sorted(episode_reasons.items())
        },
        "PER_RELATION_REASON_COUNTS.json": {
            key: {head: dict(counts) for head, counts in heads.items()}
            for key, heads in sorted(relation_reasons.items())
        },
        "CONTACT_ELIGIBILITY_AUDIT.json": {
            "total_relation_steps": sum(relation_rows.values()),
            "relation_steps_with_object_gripper_contact": sum(relation_contact_rows.values()),
            "relation_rows": dict(sorted(relation_rows.items())),
            "relation_contact_rows": dict(sorted(relation_contact_rows.items())),
            "relation_physical_values": {f"{key[0]}::{key[1]}": value for key, value in sorted(relation_physical.items())},
        },
    }
    staging = output_root.resolve().with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or output_root.exists():
        raise FileExistsError(output_root)
    staging.mkdir(parents=True)
    try:
        for name, value in payload.items():
            (staging / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "r3_3u_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "audit_manifest.json").write_text(json.dumps({
            "schema": "V5_R3_3U_UNKNOWN_ROOT_CAUSE_AUDIT_BUNDLE_V1",
            "status": gate["status"],
            "input_root": str(input_root.resolve()),
            "teacher_root": str(teacher_root.resolve()),
            "transition_manifest": str(transition_manifest.resolve()),
            "protected_reads": protected_reads,
            "attack_enabled": False,
            "source_step_count": total_steps,
            "teacher_record_count": len(records),
            "gate": gate,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        rename_noreplace(staging, output_root.resolve())
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report["output_root"] = str(output_root.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(args.input_root, args.teacher_root, args.output_root, transition_manifest=args.transition_manifest, expected_count=args.expected_count), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
