#!/usr/bin/env python3
"""Classify legacy generation evidence without reading Teacher labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
STREAMS = ("episode_metadata.json", "runtime_audit.json", "step_records.jsonl", "policy_intent_records.jsonl")
REQUIRED = {
    "episode_metadata.json", "episode_summary.json", "runtime_audit.json",
    "condition_config.json", "attack_config.json", "step_records.jsonl",
    "policy_intent_records.jsonl", "privileged_teacher_sidecar.jsonl", "artifact_sha256.json",
}
PREDICT_WITH_SCORES = "OfficialOpenVLAActionAdapter.predict_action_with_scores"
SCORE_SAME_INPUTS = "OfficialOpenVLAScoreAdapter.generate_same_inputs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {path.name}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row: {path.name}")
            rows.append(value)
    return rows


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def classify_generation_state(values: list[Any], expected_count: int) -> str:
    """Classify explicit values and missing required stream fields separately."""
    if any(value == 0 for value in values):
        return "FIELD_EXPLICIT_0"
    if any(_numeric(value) and float(value) > 1 for value in values):
        return "FIELD_EXPLICIT_GT1"
    if any(not _numeric(value) or float(value) != 1 for value in values):
        return "FIELD_INCONSISTENT_ACROSS_STREAMS"
    if len(values) != expected_count:
        return "FIELD_MISSING"
    return "FIELD_PRESENT_VALUE_1"


def _finite_vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_numeric(item) for item in value)


def _generation_values(meta: dict[str, Any], runtime: dict[str, Any], steps: list[dict[str, Any]], intents: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    stream_rows = {
        "episode_metadata.json": [meta] if "generation_passes_per_step" in meta else [],
        "runtime_audit.json": [runtime] if "generation_passes_per_step" in runtime else [],
        "step_records.jsonl": [row for row in steps if "generation_passes_per_step" in row],
        "policy_intent_records.jsonl": [row for row in intents if "generation_passes_per_step" in row],
    }
    stream_values = {name: [row["generation_passes_per_step"] for row in rows] for name, rows in stream_rows.items()}
    expected_count = 2 + len(steps) + len(intents)
    explicit_values = [value for values in stream_values.values() for value in values]
    metadata_state = classify_generation_state(stream_values["episode_metadata.json"], 1)
    cross_stream_state = classify_generation_state(explicit_values, expected_count)
    return {
        "generation_metadata_state": metadata_state,
        "generation_cross_stream_state": cross_stream_state,
        "generation_present_streams": [name for name, values in stream_values.items() if values],
        "generation_missing_streams": [name for name, values in stream_values.items() if not values],
        "generation_explicit_values": sorted({str(value) for value in explicit_values}),
    }, cross_stream_state


def _checksum_ok(root: Path) -> bool:
    try:
        payload = load_json(root / "artifact_sha256.json")
        rows = payload.get("files")
        if not isinstance(rows, list) or payload.get("recursive_sha256") != json_sha(rows):
            return False
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                return False
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
                return False
            seen.add(relative.as_posix())
            path = root / relative
            if not path.is_file() or sha256_file(path) != row.get("sha256"):
                return False
            if "size" in row and int(path.stat().st_size) != int(row["size"]):
                return False
        return REQUIRED - {"artifact_sha256.json"} <= seen
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _step_contract(steps: list[dict[str, Any]], intents: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    if not steps or len(steps) != len(intents) or summary.get("steps") != len(steps):
        errors.append("STEP_POLICY_SUMMARY_LENGTH_MISMATCH")
    for index, (step, intent) in enumerate(zip(steps, intents)):
        if step.get("action_token_ids") != intent.get("action_token_ids"):
            errors.append(f"TOKEN_MISMATCH:{index}")
        if not isinstance(step.get("action_token_ids"), list) or len(step["action_token_ids"]) != 7:
            errors.append(f"ACTION_TOKEN_COUNT:{index}")
        if not isinstance(step.get("score_head_summary"), list) or len(step["score_head_summary"]) != 7:
            errors.append(f"SCORE_COUNT:{index}")
        if step.get("score_adapter_parity_pass") is not True or intent.get("score_adapter_parity_pass") is not True:
            errors.append(f"SCORE_ACTION_PARITY:{index}")
        error = step.get("score_adapter_action_max_abs_error")
        if not _numeric(error) or float(error) > 1e-6:
            errors.append(f"SCORE_ACTION_ERROR:{index}")
        if step.get("official_execution") is not True:
            errors.append(f"OFFICIAL_EXECUTION:{index}")
        if not _finite_vector(step.get("features_25d"), 25):
            errors.append(f"FEATURES_25D:{index}")
        if not _finite_vector(step.get("clean_policy_intent_9d"), 9):
            errors.append(f"INTENT_9D:{index}")
        if not _finite_vector(step.get("clean_action_raw_7d"), 7) or not _finite_vector(step.get("applied_action_7d"), 7):
            errors.append(f"ACTION_7D:{index}")
    return not errors, errors[:20]


def _provenance(meta: dict[str, Any]) -> dict[str, Any]:
    keywords = ("collector", "runner", "adapter", "model", "processor", "checkpoint", "timestamp", "started", "finished", "created", "batch")
    return {key: value for key, value in meta.items() if any(word in key.lower() for word in keywords) and isinstance(value, (str, int, float, bool))}


def audit_episode(root: Path, expected_suite: str, expected_task: int, expected_state: int) -> dict[str, Any]:
    key = f"{expected_suite}/task_{expected_task:02d}/state_{expected_state:02d}"
    result: dict[str, Any] = {
        "suite": expected_suite, "task_idx": expected_task, "state_id": expected_state,
        "canonical_parent_key": key, "artifact_root": str(root), "status": "HOLD",
    }
    try:
        meta = load_json(root / "episode_metadata.json")
        runtime = load_json(root / "runtime_audit.json")
        summary = load_json(root / "episode_summary.json")
        steps = load_jsonl(root / "step_records.jsonl")
        intents = load_jsonl(root / "policy_intent_records.jsonl")
        generation, cross_state = _generation_values(meta, runtime, steps, intents)
        step_pass, step_errors = _step_contract(steps, intents, summary)
        checksum_pass = _checksum_ok(root)
        adapter = meta.get("score_adapter", "")
        score_alignment = "PASS" if step_pass else "HOLD"
        if adapter == PREDICT_WITH_SCORES and step_pass and checksum_pass:
            legacy_class = "LEGACY_CAPTURED_EXECUTION_SCORE_ALIGNED"
        elif adapter == SCORE_SAME_INPUTS and step_pass and checksum_pass:
            legacy_class = "LEGACY_DETERMINISTIC_SEPARATE_SCORE_ALIGNED_REQUIRES_INPUT_PROOF"
        else:
            legacy_class = "POLICY_INTENT_ALIGNMENT_UNPROVEN"
        b3_25d = bool(meta.get("runtime_valid") is True and runtime.get("runtime_valid") is True and step_pass and checksum_pass and (root / "privileged_teacher_sidecar.jsonl").is_file())
        b3_25d9d = bool(b3_25d and legacy_class == "LEGACY_CAPTURED_EXECUTION_SCORE_ALIGNED")
        result.update({
            "task_language": meta.get("task_language", ""),
            "success": meta.get("success"), "runtime_valid": meta.get("runtime_valid"),
            "official_horizon": meta.get("official_horizon"),
            "score_adapter": adapter, "official_execution_adapter": meta.get("official_execution_adapter", ""),
            "step_count": len(steps), "summary_steps": summary.get("steps"),
            "checksum_pass": checksum_pass, "step_contract_pass": step_pass,
            "step_contract_errors": json.dumps(step_errors, separators=(",", ":")),
            "legacy_classification": legacy_class,
            "b3_25d_eligible": b3_25d, "b3_25d9d_eligible": b3_25d9d,
            "generation_metadata_state": generation["generation_metadata_state"],
            "generation_cross_stream_state": generation["generation_cross_stream_state"],
            "generation_present_streams": json.dumps(generation["generation_present_streams"], separators=(",", ":")),
            "generation_missing_streams": json.dumps(generation["generation_missing_streams"], separators=(",", ":")),
            "generation_explicit_values": json.dumps(generation["generation_explicit_values"], separators=(",", ":")),
            "generation_measured_marker": any("generation" in key.lower() and "measur" in key.lower() for key in meta),
            "provenance_fields": json.dumps(_provenance(meta), sort_keys=True, separators=(",", ":")),
            "status": "PASS" if b3_25d else "HOLD",
        })
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result.update({"status": "HOLD", "error": f"{type(exc).__name__}:{exc}"})
        return result


def run(source_root: Path, output_root: Path, state_max: int, runner_git_head: str) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    rows = []
    for suite in SUITES:
        for task in range(10):
            for state in range(state_max + 1):
                root = source_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
                rows.append(audit_episode(root, suite, task, state))
    output_root.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for row in rows for key in row})
    with (output_root / "B3_LEGACY_GENERATION_EVIDENCE_CENSUS_V1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows)
    gen_counts = Counter(row.get("generation_cross_stream_state", "ERROR") for row in rows)
    metadata_counts = Counter(row.get("generation_metadata_state", "ERROR") for row in rows)
    class_counts = Counter(row.get("legacy_classification", "ERROR") for row in rows)
    def group_counts(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "identities": len(group),
            "generation_metadata": dict(sorted(Counter(row.get("generation_metadata_state", "ERROR") for row in group).items())),
            "legacy_classification": dict(sorted(Counter(row.get("legacy_classification", "ERROR") for row in group).items())),
            "b3_25d_eligible": sum(row.get("b3_25d_eligible") is True for row in group),
            "b3_25d9d_eligible": sum(row.get("b3_25d9d_eligible") is True for row in group),
        }
    by_suite = {suite: group_counts([row for row in rows if row["suite"] == suite]) for suite in SUITES}
    by_state_bucket = {}
    for name, lo, hi in (("FIT", 0, 19), ("FIT_DEV", 20, 23), ("CAL", 24, 26), ("CHECK", 27, 29)):
        group = [row for row in rows if lo <= int(row["state_id"]) <= hi]
        if group:
            by_state_bucket[name] = group_counts(group)
    summary = {
        "schema": "B3_LEGACY_GENERATION_EVIDENCE_CENSUS_V1",
        "source_root": str(source_root.resolve()), "state_range": [0, state_max],
        "runner_git_head": runner_git_head,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "identity_count": len(rows), "unique_identity_count": len({row["canonical_parent_key"] for row in rows}),
        "status": "PASS" if len(rows) == 40 * (state_max + 1) and all(row["status"] == "PASS" for row in rows) else "HOLD",
        "generation_cross_stream_counts": dict(sorted(gen_counts.items())),
        "generation_metadata_counts": dict(sorted(metadata_counts.items())),
        "legacy_classification_counts": dict(sorted(class_counts.items())),
        "by_suite": by_suite,
        "by_state_bucket": by_state_bucket,
        "b3_25d_eligible_count": sum(row.get("b3_25d_eligible") is True for row in rows),
        "b3_25d9d_eligible_count": sum(row.get("b3_25d9d_eligible") is True for row in rows),
        "score_action_alignment_pass_count": sum(row.get("step_contract_pass") is True for row in rows),
        "checksum_pass_count": sum(row.get("checksum_pass") is True for row in rows),
        "teacher_labels_read": False, "teacher_files_opened": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_root / "B3_LEGACY_GENERATION_EVIDENCE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            sums.append(f"{sha256_file(path)}  {path.name}\n")
    (output_root / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-max", type=int, choices=range(0, 50), default=19)
    parser.add_argument("--runner-git-head", required=True)
    args = parser.parse_args()
    summary = run(args.source_root.resolve(), args.output_root.resolve(), args.state_max, args.runner_git_head)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
