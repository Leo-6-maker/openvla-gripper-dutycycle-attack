#!/usr/bin/env python3
"""Build and verify the R4 dual-head provenance binding for C2g Detector-v2.

The clean collection can legitimately have been produced by an older, frozen
collection head while Teacher-v2 labels are re-audited by a newer audit head.
This tool binds those two heads without mutating the clean collection.  It also
verifies the two R4 scientific audits, the collection artifact manifest, the
label-builder bytes, and (when supplied) the narrowly pre-registered one-shot
start drift from the previous HOLD reports.

No OpenVLA model, LIBERO environment, GPU, dataset materialization, training,
attack, or attacked outcome is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.bind_c2g_collection_model_provenance import (  # noqa: E402
    collection_artifact_rows,
    verify_collection_artifact_manifest,
)
from scripts.stageb.build_c2g_suite_model_map import sha256_file  # noqa: E402

SCHEMA = "c2g.r4.dual_head_provenance.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING"
CANONICAL_PASS = "PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT"
GOAL_EVENT_PASS = "PASS_C2G_GOAL_EVENT_TRACKING_AUDIT"

CANONICAL_INVARIANT_FIELDS = (
    "label_row_count",
    "known_row_count",
    "unknown_row_count",
    "critical_positive_row_count",
    "known_negative_row_count",
    "release_safe_row_count",
    "distractor_row_count",
)
GOAL_EVENT_INVARIANT_FIELDS = (
    "active_target_known_rows",
    "contacted_unresolved_rows",
    "active_progress_unresolved_rows",
    "known_teacher_rows",
    "critical_positive_rows",
    "burst_feasible_rows",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolved_report_input(report: Mapping[str, Any], key: str) -> Path:
    value = str(report.get(key, "")).strip()
    if not value:
        raise ValueError(f"report missing {key}")
    return Path(value).resolve()


def _source_manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = sorted(
        (
            {
                "path": str(row["path"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in rows
        ),
        key=lambda row: row["path"],
    )
    return hashlib.sha256(
        "".join(
            f"{row['path']}|{row['bytes']}|{row['sha256']}\n" for row in canonical
        ).encode("utf-8")
    ).hexdigest()


def _collection_head(collection_root: Path) -> str:
    metadata_paths = sorted(collection_root.rglob("episode_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata under {collection_root}")
    heads: set[str] = set()
    for path in metadata_paths:
        metadata = _read_json(path)
        head = str(metadata.get("git_commit", "")).strip()
        if not head:
            raise ValueError(f"collection metadata lacks git_commit: {path}")
        heads.add(head)
    if len(heads) != 1:
        raise ValueError(f"collection contains multiple git heads: {sorted(heads)}")
    return next(iter(heads))


def _assert_clean_audits(
    collection_root: Path,
    canonical_path: Path,
    goal_event_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = _read_json(canonical_path)
    goal_event = _read_json(goal_event_path)
    if canonical.get("status") != CANONICAL_PASS:
        raise ValueError("canonical R4 audit is not PASS")
    if goal_event.get("status") != GOAL_EVENT_PASS:
        raise ValueError("goal-event R4 audit is not PASS")
    if _resolved_report_input(canonical, "input_root") != collection_root:
        raise ValueError("canonical audit binds another collection root")
    if _resolved_report_input(goal_event, "input_root") != collection_root:
        raise ValueError("goal-event audit binds another collection root")
    if int(canonical.get("read_error_count", -1)) != 0:
        raise ValueError("canonical audit contains read errors")
    if int(canonical.get("violation_count", -1)) != 0:
        raise ValueError("canonical audit contains violations")
    if int(goal_event.get("violation_count", -1)) != 0:
        raise ValueError("goal-event audit contains violations")
    if canonical.get("uses_attack_outcome") is not False:
        raise ValueError("canonical audit does not prove clean-only operation")
    if goal_event.get("uses_attack_outcomes") is not False:
        raise ValueError("goal-event audit does not prove clean-only operation")
    if int(canonical.get("datasets_materialized", -1)) != 0:
        raise ValueError("canonical audit unexpectedly materialized a dataset")
    if int(canonical.get("detectors_trained", -1)) != 0:
        raise ValueError("canonical audit unexpectedly trained a detector")
    if int(goal_event.get("openvla_model_loads", -1)) != 0:
        raise ValueError("goal-event audit unexpectedly loaded OpenVLA")
    if int(goal_event.get("libero_environments_created", -1)) != 0:
        raise ValueError("goal-event audit unexpectedly created LIBERO environments")
    if int(goal_event.get("attacks_launched", -1)) != 0:
        raise ValueError("goal-event audit unexpectedly launched attacks")

    known = int(canonical.get("known_row_count", -1))
    unknown = int(canonical.get("unknown_row_count", -1))
    total = int(canonical.get("label_row_count", -1))
    positives = int(canonical.get("critical_positive_row_count", -1))
    negatives = int(canonical.get("known_negative_row_count", -1))
    if min(known, unknown, total, positives, negatives) < 0:
        raise ValueError("canonical audit has invalid label counts")
    if known + unknown != total:
        raise ValueError("canonical audit known/unknown counts do not close")
    if positives + negatives != known:
        raise ValueError("canonical audit positive/negative counts do not close")
    return canonical, goal_event


def _goal_totals(report: Mapping[str, Any]) -> Mapping[str, Any]:
    totals = report.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("goal-event report lacks totals")
    return totals


def _compare_one_shot_hold_drift(
    previous_canonical: Mapping[str, Any],
    previous_goal_event: Mapping[str, Any],
    canonical: Mapping[str, Any],
    goal_event: Mapping[str, Any],
) -> dict[str, Any]:
    if not str(previous_canonical.get("status", "")).startswith("HOLD_"):
        raise ValueError("previous canonical report is not a HOLD")
    if not str(previous_goal_event.get("status", "")).startswith("HOLD_"):
        raise ValueError("previous goal-event report is not a HOLD")

    for field in CANONICAL_INVARIANT_FIELDS:
        if previous_canonical.get(field) != canonical.get(field):
            raise ValueError(f"unexpected canonical label drift: {field}")
    old_start = int(previous_canonical.get("attack_start_row_count", -1))
    new_start = int(canonical.get("attack_start_row_count", -1))
    if old_start != new_start + 1:
        raise ValueError("canonical attack-start count is not the expected one-shot 2_to_1 drift")
    old_reasons = previous_canonical.get("reason_code_counts", {})
    new_reasons = canonical.get("reason_code_counts", {})
    if not isinstance(old_reasons, Mapping) or not isinstance(new_reasons, Mapping):
        raise ValueError("canonical reason-code counts are missing")
    if int(old_reasons.get("TARGET_CRITICAL_WINDOW_START", 0)) != int(
        new_reasons.get("TARGET_CRITICAL_WINDOW_START", 0)
    ) + 1:
        raise ValueError("TARGET_CRITICAL_WINDOW_START count did not decrease by exactly one")

    old_totals = _goal_totals(previous_goal_event)
    new_totals = _goal_totals(goal_event)
    for field in GOAL_EVENT_INVARIANT_FIELDS:
        if old_totals.get(field) != new_totals.get(field):
            raise ValueError(f"unexpected goal-event label drift: {field}")
    old_goal_start = int(old_totals.get("attack_start_rows", -1))
    new_goal_start = int(new_totals.get("attack_start_rows", -1))
    if old_goal_start != new_goal_start + 1:
        raise ValueError("goal-event attack-start count is not the expected one-shot drift")

    old_violations = previous_goal_event.get("violations", [])
    if not isinstance(old_violations, list):
        raise ValueError("previous goal-event violations must be a list")
    multiple = [
        row
        for row in old_violations
        if isinstance(row, Mapping) and row.get("reason") == "MULTIPLE_ATTACK_START_ROWS"
    ]
    if len(multiple) != 1 or int(multiple[0].get("count", -1)) != 2:
        raise ValueError("previous HOLD does not contain the single expected multiple-start violation")

    return {
        "libero_10_attack_start_rows": "2_to_1",
        "multiple_attack_start_violations": "1_to_0",
        "target_critical_window_start_reason_count": "minus_1",
    }


def build_binding(
    *,
    collection_root: Path,
    collection_report_path: Path,
    collection_binding_report_path: Path,
    canonical_audit_path: Path,
    goal_event_audit_path: Path,
    label_builder_path: Path,
    collection_head: str,
    audit_head: str,
    previous_canonical_hold_path: Path | None = None,
    previous_goal_event_hold_path: Path | None = None,
    previous_hold_binding_path: Path | None = None,
) -> dict[str, Any]:
    collection_root = collection_root.resolve()
    paths = [
        collection_report_path,
        collection_binding_report_path,
        canonical_audit_path,
        goal_event_audit_path,
        label_builder_path,
    ]
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if not collection_head.strip() or not audit_head.strip():
        raise ValueError("collection_head and audit_head are required")

    before_rows = collection_artifact_rows(collection_root)
    before_sha = _source_manifest_digest(before_rows)
    artifact_manifest = verify_collection_artifact_manifest(collection_root)
    collection_report = _read_json(collection_report_path)
    binding_report = _read_json(collection_binding_report_path)
    actual_collection_head = _collection_head(collection_root)
    if actual_collection_head != collection_head:
        raise ValueError("declared collection_head differs from episode metadata")
    if str(collection_report.get("git_commit", "")) != collection_head:
        raise ValueError("collection report git_commit differs from collection_head")
    if Path(str(binding_report.get("collection_root", ""))).resolve() != collection_root:
        raise ValueError("collection model-binding report points to another collection")
    if binding_report.get("status") != "PASS_C2G_CLEAN_COLLECTION_MODEL_BINDING":
        raise ValueError("collection model-binding report is not PASS")
    if binding_report.get("artifact_manifest") != artifact_manifest:
        raise ValueError("collection model-binding report does not match current artifact manifest")

    canonical, goal_event = _assert_clean_audits(
        collection_root,
        canonical_audit_path.resolve(),
        goal_event_audit_path.resolve(),
    )

    previous: dict[str, Any] = {}
    expected_only_change: dict[str, Any] = {}
    supplied_holds = (
        previous_canonical_hold_path,
        previous_goal_event_hold_path,
        previous_hold_binding_path,
    )
    if any(path is not None for path in supplied_holds):
        if not all(path is not None for path in supplied_holds):
            raise ValueError("all three previous HOLD artifacts must be supplied together")
        assert previous_canonical_hold_path is not None
        assert previous_goal_event_hold_path is not None
        assert previous_hold_binding_path is not None
        for path in supplied_holds:
            assert path is not None
            if not path.resolve().is_file():
                raise FileNotFoundError(path)
        previous_canonical = _read_json(previous_canonical_hold_path)
        previous_goal_event = _read_json(previous_goal_event_hold_path)
        expected_only_change = _compare_one_shot_hold_drift(
            previous_canonical,
            previous_goal_event,
            canonical,
            goal_event,
        )
        previous = {
            "previous_canonical_hold_path": str(previous_canonical_hold_path.resolve()),
            "previous_canonical_hold_sha256": sha256_file(previous_canonical_hold_path.resolve()),
            "previous_goal_event_hold_path": str(previous_goal_event_hold_path.resolve()),
            "previous_goal_event_hold_sha256": sha256_file(previous_goal_event_hold_path.resolve()),
            "previous_hold_binding_path": str(previous_hold_binding_path.resolve()),
            "previous_hold_binding_sha256": sha256_file(previous_hold_binding_path.resolve()),
        }

    after_rows = collection_artifact_rows(collection_root)
    after_sha = _source_manifest_digest(after_rows)
    if before_rows != after_rows or before_sha != after_sha:
        raise ValueError("clean collection changed while building R4 provenance")

    return {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "collection_head": collection_head,
        "audit_head": audit_head,
        "collection_root": str(collection_root),
        "collection_report_path": str(collection_report_path.resolve()),
        "collection_report_sha256": sha256_file(collection_report_path.resolve()),
        "collection_input_manifest_path": artifact_manifest["path"],
        "collection_input_manifest_sha256": artifact_manifest["sha256"],
        "collection_model_binding_report_path": str(collection_binding_report_path.resolve()),
        "collection_model_binding_report_sha256": sha256_file(
            collection_binding_report_path.resolve()
        ),
        "label_builder_path": str(label_builder_path.resolve()),
        "label_builder_sha256": sha256_file(label_builder_path.resolve()),
        "canonical_audit_path": str(canonical_audit_path.resolve()),
        "canonical_audit_sha256": sha256_file(canonical_audit_path.resolve()),
        "goal_event_audit_path": str(goal_event_audit_path.resolve()),
        "goal_event_audit_sha256": sha256_file(goal_event_audit_path.resolve()),
        "canonical_status": canonical["status"],
        "goal_event_status": goal_event["status"],
        "source_collection_file_count": len(before_rows),
        "source_collection_manifest_before_sha256": before_sha,
        "source_collection_manifest_after_sha256": after_sha,
        "source_collection_unchanged": True,
        "unknown_to_negative_count": 0,
        "uses_attack_outcomes": False,
        "openvla_model_loads": 0,
        "libero_environments_created": 0,
        "clean_rollouts_launched": 0,
        "attacked_rollouts_launched": 0,
        "training_epochs": 0,
        "datasets_materialized": 0,
        "expected_only_change": expected_only_change,
        "unexpected_label_drift": False,
        **previous,
    }


def verify_binding(
    binding_path: Path,
    *,
    collection_root: Path,
    expected_audit_head: str | None = None,
) -> dict[str, Any]:
    binding_path = binding_path.resolve()
    report = _read_json(binding_path)
    if report.get("schema") != SCHEMA or report.get("status") != PASS_STATUS:
        raise ValueError("R4 provenance binding schema/status mismatch")
    collection_root = collection_root.resolve()
    if Path(str(report.get("collection_root", ""))).resolve() != collection_root:
        raise ValueError("R4 provenance binding points to another collection")
    if expected_audit_head is not None and report.get("audit_head") != expected_audit_head:
        raise ValueError("R4 provenance binding was produced by another audit head")
    if report.get("collection_head") != _collection_head(collection_root):
        raise ValueError("R4 provenance collection head no longer matches metadata")

    current_rows = collection_artifact_rows(collection_root)
    current_sha = _source_manifest_digest(current_rows)
    if int(report.get("source_collection_file_count", -1)) != len(current_rows):
        raise ValueError("R4 provenance source file count changed")
    if report.get("source_collection_manifest_before_sha256") != current_sha:
        raise ValueError("R4 provenance source manifest changed")
    if report.get("source_collection_manifest_after_sha256") != current_sha:
        raise ValueError("R4 provenance after-manifest changed")
    if report.get("source_collection_unchanged") is not True:
        raise ValueError("R4 provenance does not attest an unchanged collection")

    file_bindings = (
        ("collection_report_path", "collection_report_sha256"),
        ("collection_model_binding_report_path", "collection_model_binding_report_sha256"),
        ("label_builder_path", "label_builder_sha256"),
        ("canonical_audit_path", "canonical_audit_sha256"),
        ("goal_event_audit_path", "goal_event_audit_sha256"),
    )
    for path_key, hash_key in file_bindings:
        path = Path(str(report.get(path_key, ""))).resolve()
        if not path.is_file() or sha256_file(path) != report.get(hash_key):
            raise ValueError(f"R4 provenance file binding changed: {path_key}")
    manifest_path = Path(str(report.get("collection_input_manifest_path", ""))).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != report.get(
        "collection_input_manifest_sha256"
    ):
        raise ValueError("R4 provenance collection input manifest changed")

    canonical, goal_event = _assert_clean_audits(
        collection_root,
        Path(str(report["canonical_audit_path"])),
        Path(str(report["goal_event_audit_path"])),
    )
    if report.get("canonical_status") != canonical.get("status"):
        raise ValueError("R4 provenance canonical status changed")
    if report.get("goal_event_status") != goal_event.get("status"):
        raise ValueError("R4 provenance goal-event status changed")
    if int(report.get("unknown_to_negative_count", -1)) != 0:
        raise ValueError("R4 provenance records unknown-to-negative conversion")
    if report.get("uses_attack_outcomes") is not False:
        raise ValueError("R4 provenance records attacked-outcome use")
    for key in (
        "openvla_model_loads",
        "libero_environments_created",
        "clean_rollouts_launched",
        "attacked_rollouts_launched",
        "training_epochs",
        "datasets_materialized",
    ):
        if int(report.get(key, -1)) != 0:
            raise ValueError(f"R4 provenance boundary is not zero: {key}")
    if report.get("unexpected_label_drift") is not False:
        raise ValueError("R4 provenance records unexpected label drift")
    return report


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--collection-root", type=Path, required=True)
    build.add_argument("--collection-report", type=Path, required=True)
    build.add_argument("--collection-binding-report", type=Path, required=True)
    build.add_argument("--canonical-audit", type=Path, required=True)
    build.add_argument("--goal-event-audit", type=Path, required=True)
    build.add_argument("--label-builder", type=Path, required=True)
    build.add_argument("--collection-head", required=True)
    build.add_argument("--audit-head", required=True)
    build.add_argument("--previous-canonical-hold", type=Path)
    build.add_argument("--previous-goal-event-hold", type=Path)
    build.add_argument("--previous-hold-binding", type=Path)
    build.add_argument("--output-report", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--binding", type=Path, required=True)
    verify.add_argument("--collection-root", type=Path, required=True)
    verify.add_argument("--expected-audit-head")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            report = build_binding(
                collection_root=args.collection_root,
                collection_report_path=args.collection_report,
                collection_binding_report_path=args.collection_binding_report,
                canonical_audit_path=args.canonical_audit,
                goal_event_audit_path=args.goal_event_audit,
                label_builder_path=args.label_builder,
                collection_head=args.collection_head,
                audit_head=args.audit_head,
                previous_canonical_hold_path=args.previous_canonical_hold,
                previous_goal_event_hold_path=args.previous_goal_event_hold,
                previous_hold_binding_path=args.previous_hold_binding,
            )
            _write_json(args.output_report.resolve(), report)
        else:
            report = verify_binding(
                args.binding,
                collection_root=args.collection_root,
                expected_audit_head=args.expected_audit_head,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "HOLD_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
