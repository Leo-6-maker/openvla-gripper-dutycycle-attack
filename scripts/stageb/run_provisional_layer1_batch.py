#!/usr/bin/env python3
"""Run provisional full-batch Layer 1 labels for the engineering bypass.

This script is intentionally CPU-only. It does not load OpenVLA, does not read
Layer 2 detector telemetry as Teacher input, and does not launch LIBERO. It
turns frozen CLEAN ledgers into full resolver manifests, runs the existing
Layer 1 resolver, and writes an audit suitable for the provisional Layer2
engineering path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.cross_suite_layer1_resolver import (  # noqa: E402
    classify_mechanism,
    canonical_key,
    load_ontology,
    run_resolver,
    stable_hash,
    write_csv,
    write_json,
)

PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"
EXPECTED_COMPONENT_WORKTREE_SHA256 = {
    "ontology": "89bd296b15525c48a4fbd3be84eb4a8c0b269ca11170cfe901b449cc1bb77359",
    "physics_config": "1f5e0dbeb0e227d2c6708a310ef171863c216bf84a5cddda62af992766700059",
    "teacher_schema": "ac2ffb8b064a502f8b5cab7b0c4183914701c7bf21443dd01189e023403ebca8",
    "timing_contract": "18b21f9e032fdba410e291f67edba60e649b2c6c4ae3d2f33df08a7c31f6ef60",
}
EXPECTED_COMPONENT_GIT_BLOB_SHA256 = {
    "ontology": "70f4c03860d617b5fc64e61dbe9e287dae67c1178e57634d5be9d4d2bde99462",
    "physics_config": "1f5e0dbeb0e227d2c6708a310ef171863c216bf84a5cddda62af992766700059",
    "teacher_schema": "c5324d46e50b4a84ba23415c55e75c07ef09acbe0bdd8ae2a15f4f425a480a6e",
    "timing_contract": "18b21f9e032fdba410e291f67edba60e649b2c6c4ae3d2f33df08a7c31f6ef60",
}
COMPONENT_PATHS = {
    "ontology": REPO / "configs" / "cross_suite_task_ontology_v1.yaml",
    "physics_config": REPO / "configs" / "cross_suite_teacher_physics_v1.yaml",
    "teacher_schema": REPO / "docs" / "schemas" / "cross_suite_teacher_label_schema_v1.md",
    "timing_contract": REPO / "reports" / "layer1_h2_20260620" / "timing_alignment_contract_20260620.md",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_dirty_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "True"
    if text.lower() in {"false", "0", "no"}:
        return "False"
    return text


def _episode_path(row: dict[str, Any]) -> str:
    for key in ("episode_path", "primary_output_dir", "output_dir"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _source_sha(row: dict[str, Any]) -> str:
    for key in ("artifact_recursive_sha256", "source_episode_sha", "episode_sha256"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _is_complete_clean(row: dict[str, Any], *, ledger_kind: str) -> tuple[bool, str]:
    if str(row.get("condition", "")).strip() != "CLEAN":
        return False, "condition_not_clean"
    if _as_int(row.get("invalid_feature_steps", 0), default=0) != 0:
        return False, "invalid_feature_steps_nonzero"
    if ledger_kind == "clean300":
        if str(row.get("status", "")).strip() != "COMPLETE_VALID":
            return False, "status_not_complete_valid"
        if str(row.get("clean_only_contract", "True")).strip() not in {"True", "true", "1"}:
            return False, "clean_only_contract_false"
    elif ledger_kind == "train300":
        if str(row.get("primary_status", "")).strip() != "COMPLETE":
            return False, "primary_status_not_complete"
    else:
        return False, f"unknown_ledger_kind:{ledger_kind}"
    if not _episode_path(row):
        return False, "missing_episode_path"
    return True, ""


def build_full_manifest(
    *,
    ledger_rows: list[dict[str, str]],
    ontology: dict[tuple[str, int], Any],
    ledger_kind: str,
    split_name: str,
    state_min: int,
    state_max: int,
    expected_count: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_keys: list[str] = []
    for raw in ledger_rows:
        state = _as_int(raw.get("state_id"), default=-1)
        if state < state_min or state > state_max:
            continue
        ok, reason = _is_complete_clean(raw, ledger_kind=ledger_kind)
        ck = str(raw.get("canonical_key") or canonical_key(raw))
        if not ok:
            rejected.append({"canonical_key": ck, "reason": reason})
            continue
        suite = str(raw.get("suite", "")).strip()
        task_idx = _as_int(raw.get("task_idx"), default=-1)
        task = ontology.get((suite, task_idx))
        if task is None:
            rejected.append({"canonical_key": ck, "reason": "ontology_missing"})
            continue
        if ck in seen:
            duplicate_keys.append(ck)
            rejected.append({"canonical_key": ck, "reason": "duplicate_key"})
            continue
        seen.add(ck)
        episode_path = _episode_path(raw)
        source_sha = _source_sha(raw)
        selected.append(
            {
                "canonical_key": ck,
                "suite": suite,
                "task_idx": task_idx,
                "state_id": state,
                "eval_seed": _as_int(raw.get("eval_seed"), default=0),
                "condition": "CLEAN",
                "episode_path": episode_path,
                "source_episode_sha": source_sha,
                "artifact_recursive_sha256": source_sha,
                "task_success": _as_bool_text(raw.get("task_success", "")),
                "n_steps": _as_int(raw.get("n_steps"), default=0),
                "mechanism_type": task.mechanism_type,
                "teacher_applicable": task.teacher_applicable,
                "mechanism_group": classify_mechanism(task),
                "split_name": split_name,
                "ledger_kind": ledger_kind,
                "selection_hash": stable_hash([split_name, suite, task_idx, state, raw.get("eval_seed", 0), source_sha]),
            }
        )
    selected.sort(key=lambda row: (row["suite"], int(row["task_idx"]), int(row["state_id"]), int(row["eval_seed"])))
    status = "PASS" if len(selected) == expected_count and not duplicate_keys else "FAIL"
    return {
        "manifest_type": "provisional_full_layer1_manifest_v1",
        "provisional_engineering_only": True,
        "split_name": split_name,
        "ledger_kind": ledger_kind,
        "state_min": state_min,
        "state_max": state_max,
        "expected_count": expected_count,
        "selected_count": len(selected),
        "status": status,
        "duplicate_keys": duplicate_keys,
        "rejected_count": len(rejected),
        "selected": selected,
        "rejected": rejected,
        "claim_boundary": {
            "official_h2_status": "NOT_GRANTED",
            "human_review_status": "DEFERRED_NONBLOCKING_FOR_ENGINEERING",
            "paper_evidence": "FORBIDDEN",
        },
    }


def component_manifest() -> dict[str, Any]:
    rows = {}
    mismatches = {}
    for name, path in COMPONENT_PATHS.items():
        rel = path.relative_to(REPO).as_posix()
        worktree_digest = sha256_file(path)
        blob_bytes = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO)
        blob_digest = hashlib.sha256(blob_bytes).hexdigest()
        worktree_match = worktree_digest == EXPECTED_COMPONENT_WORKTREE_SHA256[name]
        blob_match = blob_digest == EXPECTED_COMPONENT_GIT_BLOB_SHA256[name]
        rows[name] = {
            "path": rel,
            "worktree_sha256": worktree_digest,
            "expected_handoff_worktree_sha256": EXPECTED_COMPONENT_WORKTREE_SHA256[name],
            "git_blob_sha256": blob_digest,
            "expected_git_blob_sha256": EXPECTED_COMPONENT_GIT_BLOB_SHA256[name],
            "accepted_by": "handoff_worktree_sha256" if worktree_match else ("git_blob_sha256" if blob_match else "NONE"),
            "line_ending_note": "handoff SHA came from the Windows working tree for some text files; git_blob_sha256 is the cross-platform content identity",
        }
        if not (worktree_match or blob_match):
            mismatches[name] = {
                "actual_worktree": worktree_digest,
                "expected_handoff_worktree": EXPECTED_COMPONENT_WORKTREE_SHA256[name],
                "actual_git_blob": blob_digest,
                "expected_git_blob": EXPECTED_COMPONENT_GIT_BLOB_SHA256[name],
            }
    return {"components": rows, "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


def summarize_labels(resolver_dir: Path) -> dict[str, Any]:
    episode_rows = read_csv_rows(resolver_dir / "teacher_episode_labels_v1.csv")
    event_rows = read_csv_rows(resolver_dir / "teacher_event_labels_v1.csv")
    return {
        "episode_count": len(episode_rows),
        "event_count": len(event_rows),
        "teacher_status_counts": dict(Counter(row.get("teacher_status", "") for row in episode_rows)),
        "mechanism_counts": dict(Counter(row.get("mechanism_type", "") for row in episode_rows)),
        "suite_counts": dict(Counter(row.get("suite", "") for row in episode_rows)),
        "eligible_event_count": sum(1 for row in episode_rows if row.get("teacher_status") == "ELIGIBLE_EVENT"),
        "ignore_mask_status_count": sum(
            1
            for row in episode_rows
            if row.get("teacher_status")
            in {
                "TARGET_BINDING_AMBIGUOUS",
                "TARGET_BINDING_FAILED",
                "OBJECT_BINDING_AMBIGUOUS",
                "RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM",
            }
        ),
    }


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PROVISIONAL_SENTINEL).write_text(
        "This output is provisional engineering evidence only. It is not final paper evidence.\n",
        encoding="utf-8",
    )
    components = component_manifest()
    write_json(output_dir / "component_manifest.json", components)
    if components["status"] != "PASS":
        raise SystemExit(f"component SHA mismatch: {components['mismatches']}")

    ontology = load_ontology(Path(args.ontology))
    train_rows = read_csv_rows(Path(args.train_ledger))
    clean_rows = read_csv_rows(Path(args.clean300_ledger))
    specs = [
        ("train300_train_s10_17", "train300", train_rows, 10, 17, 240),
        ("train300_val_s18_19", "train300", train_rows, 18, 19, 60),
        ("clean300_test_s0_9", "clean300", clean_rows, 0, 9, 300),
    ]
    manifest_summaries: list[dict[str, Any]] = []
    resolver_summaries: dict[str, Any] = {}
    for split_name, ledger_kind, rows, state_min, state_max, expected in specs:
        manifest = build_full_manifest(
            ledger_rows=rows,
            ontology=ontology,
            ledger_kind=ledger_kind,
            split_name=split_name,
            state_min=state_min,
            state_max=state_max,
            expected_count=expected,
        )
        manifest_path = output_dir / "manifests" / f"{split_name}_manifest.json"
        write_json(manifest_path, manifest)
        write_csv(output_dir / "manifests" / f"{split_name}_manifest.csv", manifest["selected"])
        manifest_summaries.append(
            {
                "split_name": split_name,
                "ledger_kind": ledger_kind,
                "expected_count": expected,
                "selected_count": manifest["selected_count"],
                "status": manifest["status"],
                "duplicate_count": len(manifest["duplicate_keys"]),
                "rejected_count": manifest["rejected_count"],
            }
        )
        if manifest["status"] != "PASS":
            raise SystemExit(f"manifest failed for {split_name}: {manifest_summaries[-1]}")
        resolver_dir = output_dir / "resolver_outputs" / split_name
        sidecar = run_resolver(
            manifest_path,
            Path(args.ontology),
            resolver_dir,
            teacher_run_id=f"{args.teacher_run_prefix}_{split_name}",
        )
        label_summary = summarize_labels(resolver_dir)
        resolver_summaries[split_name] = {"sidecar": sidecar, "labels": label_summary}
        if sidecar["failure_count"] or sidecar["validation_error_count"]:
            raise SystemExit(f"resolver failed for {split_name}: {sidecar}")

    audit = {
        "audit_type": "provisional_layer1_full_batch_v1",
        "provisional_engineering_only": True,
        "official_h2_status": "NOT_GRANTED",
        "human_review_status": "DEFERRED_NONBLOCKING_FOR_ENGINEERING",
        "paper_claims": "FORBIDDEN",
        "git_commit": git_commit(),
        "git_dirty_status": git_dirty_status(),
        "train_ledger": str(Path(args.train_ledger)),
        "clean300_ledger": str(Path(args.clean300_ledger)),
        "component_manifest": components,
        "manifest_summaries": manifest_summaries,
        "resolver_summaries": resolver_summaries,
    }
    write_json(output_dir / "provisional_layer1_batch_audit.json", audit)
    write_csv(output_dir / "provisional_layer1_manifest_summary.csv", manifest_summaries)
    return audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-ledger", required=True)
    ap.add_argument("--clean300-ledger", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ontology", default=str(REPO / "configs" / "cross_suite_task_ontology_v1.yaml"))
    ap.add_argument("--teacher-run-prefix", default="provisional_layer1_6eb8863_20260621")
    return ap.parse_args()


def main() -> None:
    run_batch(parse_args())


if __name__ == "__main__":
    main()
