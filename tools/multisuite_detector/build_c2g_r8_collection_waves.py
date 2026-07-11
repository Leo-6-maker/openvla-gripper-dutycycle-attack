#!/usr/bin/env python3
"""Build provenance-bound, leakage-safe R8 clean-collection waves.

R7 freezes the complete 2,000-parent scientific registry. R8 does not redraw any
state identity. It subtracts the exact R7 reusable source inventory, emits a
small cross-suite detector canary, emits the complete remaining detector corpus,
and keeps preregistered attack-evaluation parents in a separate never-train wave.

This program is planning-only. It does not load OpenVLA, create LIBERO
environments, launch rollouts, materialize embeddings, train, calibrate, or read
attack outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.audit_c2g_clean_source_inventory import (  # noqa: E402
    PASS_STATUS as SOURCE_PASS_STATUS,
    SCHEMA as SOURCE_SCHEMA,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (  # noqa: E402
    ATTACK_EVAL,
    COHORTS,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    PASS_STATUS as PLAN_PASS_STATUS,
    SCHEMA as PLAN_SCHEMA,
    SUITES,
)

SCHEMA = "c2g.r8.collection_wave_plan.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R8_COLLECTION_WAVE_PLAN"
DETECTOR_CANARY = "detector_canary"
DETECTOR_FULL = "detector_full"
ATTACK_EVAL_WAVE = "attack_eval"
WAVES = (DETECTOR_CANARY, DETECTOR_FULL, ATTACK_EVAL_WAVE)
DETECTOR_COHORTS = (DETECTOR_TRAIN, DETECTOR_VAL, DETECTOR_TEST)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _assert_hash(path: Path, expected: str, *, name: str) -> str:
    expected = _require_sha256(expected, name=name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _assert_new_external_output(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.exists():
        raise FileExistsError(resolved)
    if _is_within(resolved, REPO):
        raise ValueError("R8 runtime plan output must be outside the repository")
    return resolved


def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_index"]), int(row["state_id"])


def _validate_registry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("R7 registry is empty")
    output: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int]] = set()
    parent_keys: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        suite, task_index, state_id = identity(row)
        cohort = str(row.get("cohort", ""))
        split = str(row.get("split", ""))
        parent_key = str(row.get("parent_key", ""))
        if suite not in SUITES or task_index < 0 or state_id < 0:
            raise ValueError(f"invalid registry identity at row {index}")
        if cohort not in COHORTS:
            raise ValueError(f"invalid registry cohort at row {index}")
        expected_split = {
            DETECTOR_TRAIN: "train",
            DETECTOR_VAL: "val",
            DETECTOR_TEST: "test",
            ATTACK_EVAL: "attack_eval",
        }[cohort]
        if split != expected_split:
            raise ValueError(f"registry cohort/split mismatch at row {index}")
        if not parent_key:
            raise ValueError(f"registry parent_key is empty at row {index}")
        key = (suite, task_index, state_id)
        if key in identities:
            raise ValueError(f"duplicate registry identity: {key}")
        if parent_key in parent_keys:
            raise ValueError(f"duplicate registry parent_key: {parent_key}")
        identities.add(key)
        parent_keys.add(parent_key)
        row["registry_index"] = index
        output.append(row)
    if {row["suite"] for row in output} != set(SUITES):
        raise ValueError("R7 registry does not contain all four suites")
    return output


def load_bound_inputs(
    *,
    registry_path: Path,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    source_audit_report_path: Path,
    expected_source_audit_report_sha256: str,
    reusable_manifest_path: Path,
    expected_reusable_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    registry_path = registry_path.resolve()
    plan_report_path = plan_report_path.resolve()
    source_audit_report_path = source_audit_report_path.resolve()
    reusable_manifest_path = reusable_manifest_path.resolve()
    for path in (
        registry_path,
        plan_report_path,
        source_audit_report_path,
        reusable_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    _assert_hash(plan_report_path, expected_plan_report_sha256, name="R7 corpus plan report")
    _assert_hash(source_audit_report_path, expected_source_audit_report_sha256, name="R7 source audit report")
    _assert_hash(reusable_manifest_path, expected_reusable_manifest_sha256, name="R7 reusable manifest")

    plan = read_json(plan_report_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_PASS_STATUS:
        raise ValueError("R7 corpus plan report is not accepted")
    if Path(str(plan.get("registry", ""))).resolve() != registry_path:
        raise ValueError("R7 plan report binds another registry path")
    if str(plan.get("registry_sha256", "")) != sha256_file(registry_path):
        raise ValueError("R7 registry bytes differ from plan report")

    source = read_json(source_audit_report_path)
    if source.get("schema") != SOURCE_SCHEMA or source.get("status") != SOURCE_PASS_STATUS:
        raise ValueError("R7 source audit report is not accepted")
    if Path(str(source.get("plan_report", ""))).resolve() != plan_report_path:
        raise ValueError("R7 source audit binds another plan report")
    if str(source.get("plan_report_sha256", "")) != sha256_file(plan_report_path):
        raise ValueError("R7 source audit plan hash mismatch")
    if Path(str(source.get("registry", ""))).resolve() != registry_path:
        raise ValueError("R7 source audit binds another registry")
    if str(source.get("registry_sha256", "")) != sha256_file(registry_path):
        raise ValueError("R7 source audit registry hash mismatch")
    if Path(str(source.get("reusable_manifest", ""))).resolve() != reusable_manifest_path:
        raise ValueError("R7 source audit binds another reusable manifest")
    if str(source.get("reusable_manifest_sha256", "")) != sha256_file(reusable_manifest_path):
        raise ValueError("R7 reusable manifest hash differs from source audit")
    if not str(source.get("training_authorization", "")).startswith("HOLD_"):
        raise ValueError("R7 source audit unexpectedly authorizes training")

    registry = _validate_registry(read_jsonl(registry_path))
    reusable = read_jsonl(reusable_manifest_path)
    registry_lookup = {identity(row): row for row in registry}
    reusable_identities: set[tuple[str, int, int]] = set()
    for index, row in enumerate(reusable):
        key = identity(row)
        if key not in registry_lookup:
            raise ValueError(f"reusable row {index} is not in the frozen registry: {key}")
        if key in reusable_identities:
            raise ValueError(f"duplicate reusable identity: {key}")
        reusable_identities.add(key)
        frozen = registry_lookup[key]
        recorded_parent_key = str(row.get("registry_parent_key") or row.get("parent_key") or "")
        if recorded_parent_key != str(frozen.get("parent_key", "")):
            raise ValueError(f"reusable row {index} changes frozen parent_key")
        for field in ("cohort", "split"):
            if str(row.get(field, "")) != str(frozen.get(field, "")):
                raise ValueError(f"reusable row {index} changes frozen {field}")
    if int(source.get("registered_reusable_episode_count", -1)) != len(reusable):
        raise ValueError("R7 reusable count differs from reusable manifest")
    return registry, reusable, plan, source


def _select_canary(
    missing: Sequence[Mapping[str, Any]],
    *,
    tasks_per_suite: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tasks_per_suite <= 0:
        raise ValueError("canary tasks per suite must be positive")
    by_suite_task_cohort: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in missing:
        row = dict(raw)
        if row["cohort"] in DETECTOR_COHORTS:
            by_suite_task_cohort[(row["suite"], int(row["task_index"]), row["cohort"])].append(row)
    selected: list[dict[str, Any]] = []
    suite_details: dict[str, Any] = {}
    for suite in SUITES:
        tasks = sorted(
            {
                int(row["task_index"])
                for row in missing
                if row["suite"] == suite and row["cohort"] in DETECTOR_COHORTS
            }
        )
        eligible_tasks = [
            task
            for task in tasks
            if all(by_suite_task_cohort.get((suite, task, cohort)) for cohort in DETECTOR_COHORTS)
        ]
        chosen = eligible_tasks[:tasks_per_suite]
        for task in chosen:
            for cohort in DETECTOR_COHORTS:
                candidates = sorted(
                    by_suite_task_cohort[(suite, task, cohort)],
                    key=lambda row: int(row["registry_index"]),
                )
                selected.append(dict(candidates[0]))
        suite_details[suite] = {
            "eligible_task_count": len(eligible_tasks),
            "selected_task_indices": chosen,
            "requested_task_count": tasks_per_suite,
            "task_shortfall": max(0, tasks_per_suite - len(chosen)),
            "selected_episode_count": len(chosen) * len(DETECTOR_COHORTS),
        }
    selected.sort(key=lambda row: int(row["registry_index"]))
    return selected, suite_details


def _write_wave(
    output_dir: Path,
    wave: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_size: int,
) -> dict[str, Any]:
    if wave not in WAVES:
        raise ValueError(f"unknown wave: {wave}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    wave_rows = [dict(row) for row in rows]
    manifest_path = output_dir / f"c2g_r8_{wave}.jsonl"
    write_jsonl(manifest_path, wave_rows)
    shard_records: list[dict[str, Any]] = []
    by_suite = {suite: [row for row in wave_rows if row["suite"] == suite] for suite in SUITES}
    for suite in SUITES:
        local = by_suite[suite]
        for shard_index, start in enumerate(range(0, len(local), shard_size)):
            shard_rows = local[start : start + shard_size]
            shard_path = output_dir / "shards" / wave / suite / f"shard_{shard_index:03d}.jsonl"
            write_jsonl(shard_path, shard_rows)
            shard_records.append(
                {
                    "wave": wave,
                    "suite": suite,
                    "shard_id": f"{wave}__{suite}__{shard_index:03d}",
                    "manifest": str(shard_path.resolve()),
                    "manifest_sha256": sha256_file(shard_path),
                    "episode_count": len(shard_rows),
                    "cohort_counts": dict(sorted(Counter(row["cohort"] for row in shard_rows).items())),
                }
            )
    shard_index_path = output_dir / f"c2g_r8_{wave}_shards.jsonl"
    write_jsonl(shard_index_path, shard_records)
    return {
        "wave": wave,
        "episode_count": len(wave_rows),
        "suite_count": len({row["suite"] for row in wave_rows}),
        "task_count": len({(row["suite"], row["task_index"]) for row in wave_rows}),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in wave_rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in wave_rows).items())),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "shard_size": shard_size,
        "shard_count": len(shard_records),
        "shard_index": str(shard_index_path.resolve()),
        "shard_index_sha256": sha256_file(shard_index_path),
        "shards": shard_records,
    }


def build_collection_waves(
    *,
    registry_path: Path,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    source_audit_report_path: Path,
    expected_source_audit_report_sha256: str,
    reusable_manifest_path: Path,
    expected_reusable_manifest_sha256: str,
    output_dir: Path,
    expected_git_commit: str,
    canary_tasks_per_suite: int = 2,
    canary_shard_size: int = 64,
    detector_full_shard_size: int = 10,
    attack_eval_shard_size: int = 10,
) -> dict[str, Any]:
    output_dir = _assert_new_external_output(output_dir)
    registry, reusable, plan, source = load_bound_inputs(
        registry_path=registry_path,
        plan_report_path=plan_report_path,
        expected_plan_report_sha256=expected_plan_report_sha256,
        source_audit_report_path=source_audit_report_path,
        expected_source_audit_report_sha256=expected_source_audit_report_sha256,
        reusable_manifest_path=reusable_manifest_path,
        expected_reusable_manifest_sha256=expected_reusable_manifest_sha256,
    )
    expected_git_commit = str(expected_git_commit).strip()
    if len(expected_git_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_git_commit):
        raise ValueError("expected_git_commit must be a lowercase 40-character commit SHA")

    reusable_identities = {identity(row) for row in reusable}
    missing = [row for row in registry if identity(row) not in reusable_identities]
    detector_missing = [row for row in missing if row["cohort"] != ATTACK_EVAL]
    attack_missing = [row for row in missing if row["cohort"] == ATTACK_EVAL]
    canary, canary_selection = _select_canary(detector_missing, tasks_per_suite=canary_tasks_per_suite)
    if any(row["cohort"] == ATTACK_EVAL for row in canary + detector_missing):
        raise AssertionError("detector waves must exclude attack-evaluation parents")
    if {identity(row) for row in detector_missing} & reusable_identities:
        raise AssertionError("detector missing wave contains reusable identity")
    if {identity(row) for row in attack_missing} & reusable_identities:
        raise AssertionError("attack-eval missing wave contains reusable identity")

    output_dir.mkdir(parents=True)
    waves = {
        DETECTOR_CANARY: _write_wave(output_dir, DETECTOR_CANARY, canary, shard_size=canary_shard_size),
        DETECTOR_FULL: _write_wave(output_dir, DETECTOR_FULL, detector_missing, shard_size=detector_full_shard_size),
        ATTACK_EVAL_WAVE: _write_wave(output_dir, ATTACK_EVAL_WAVE, attack_missing, shard_size=attack_eval_shard_size),
    }
    report_path = output_dir / "c2g_r8_collection_wave_plan.json"
    report = {
        "schema": SCHEMA,
        "gate": "C2G_R8_COLLECTION_WAVE_PLAN",
        "status": PASS_STATUS,
        "expected_git_commit": expected_git_commit,
        "r7_plan_report": str(plan_report_path.resolve()),
        "r7_plan_report_sha256": sha256_file(plan_report_path.resolve()),
        "r7_registry": str(registry_path.resolve()),
        "r7_registry_sha256": sha256_file(registry_path.resolve()),
        "r7_source_audit_report": str(source_audit_report_path.resolve()),
        "r7_source_audit_report_sha256": sha256_file(source_audit_report_path.resolve()),
        "r7_reusable_manifest": str(reusable_manifest_path.resolve()),
        "r7_reusable_manifest_sha256": sha256_file(reusable_manifest_path.resolve()),
        "registered_parent_count": len(registry),
        "reusable_parent_count": len(reusable),
        "missing_parent_count": len(missing),
        "detector_missing_parent_count": len(detector_missing),
        "attack_eval_missing_parent_count": len(attack_missing),
        "canary_tasks_per_suite": canary_tasks_per_suite,
        "canary_selection": canary_selection,
        "waves": waves,
        "collection_authorization": "HOLD_PENDING_EXPLICIT_R8_CANARY_AUTHORIZATION",
        "detector_full_authorization": "HOLD_PENDING_R8_CANARY_AUDIT",
        "attack_eval_authorization": "HOLD_UNTIL_DETECTOR_FROZEN",
        "training_authorization": "HOLD_PENDING_FULL_CORPUS_MATERIALIZATION_AND_AUDIT",
        "next_stage": "STOP_FOR_R8_COLLECTION_WAVE_PLAN_REVIEW",
        "boundaries": {
            "registry_redrawn": False,
            "reusable_assets_modified": False,
            "attack_outcomes_read": False,
            "openvla_models_loaded": 0,
            "libero_environments_created": 0,
            "clean_rollouts_launched": 0,
            "attacked_rollouts_launched": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
        },
    }
    write_json(report_path, report)
    return {**report, "report": str(report_path.resolve()), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--source-audit-report", type=Path, required=True)
    parser.add_argument("--expected-source-audit-report-sha256", required=True)
    parser.add_argument("--reusable-manifest", type=Path, required=True)
    parser.add_argument("--expected-reusable-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--canary-tasks-per-suite", type=int, default=2)
    parser.add_argument("--canary-shard-size", type=int, default=64)
    parser.add_argument("--detector-full-shard-size", type=int, default=10)
    parser.add_argument("--attack-eval-shard-size", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_collection_waves(
        registry_path=args.registry,
        plan_report_path=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        source_audit_report_path=args.source_audit_report,
        expected_source_audit_report_sha256=args.expected_source_audit_report_sha256,
        reusable_manifest_path=args.reusable_manifest,
        expected_reusable_manifest_sha256=args.expected_reusable_manifest_sha256,
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        canary_tasks_per_suite=args.canary_tasks_per_suite,
        canary_shard_size=args.canary_shard_size,
        detector_full_shard_size=args.detector_full_shard_size,
        attack_eval_shard_size=args.attack_eval_shard_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
