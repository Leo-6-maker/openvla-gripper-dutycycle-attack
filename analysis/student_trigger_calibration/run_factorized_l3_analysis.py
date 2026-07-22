#!/usr/bin/env python3
"""Replay the real Factorized V2 adapter and compute L3 timing metrics.

Both diagnostic and authoritative modes require one schema-valid V3
calibration-and-threshold contract for every split. Diagnostic mode permits
``l3_evaluation_eligible=false`` but never substitutes an empty/default
contract. Authoritative mode additionally requires all 12 contracts to be
independently calibrated and L3 eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from fit_factorized_calibrators import load_json, verify_sealed_directory  # noqa: E402
from validate_factorized_codex_handoff import (  # noqa: E402
    validate_handoff_execution,
    validate_handoff_static,
)

EXPECTED_SPLITS = frozenset(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


def validate_episode_step_sequence(rows: list[dict[str, Any]], step_field: str) -> int:
    steps = sorted(row[step_field] for row in rows)
    if not steps:
        raise SystemExit("EMPTY_EPISODE")
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise SystemExit("STEP_INVALID")
    if steps[0] != 0:
        raise SystemExit(f"FIRST_STEP_NOT_ZERO:{steps[0]}")
    for expected, observed in enumerate(steps):
        if observed != expected:
            raise SystemExit(f"STEP_GAP:expected={expected}:observed={observed}")
    return len(steps)


def classify_episode(offline_rows: list[dict[str, Any]], step_field: str) -> str:
    length = len(offline_rows)
    if length < 10:
        return "unknown"
    eligible = offline_rows[: length - 10 + 1]
    for row in eligible:
        if not isinstance(row.get("strict_k10_known_mask"), bool):
            raise SystemExit("STRICT_K10_KNOWN_MASK_INVALID")
        if not isinstance(row.get("strict_k10_feasible"), bool):
            raise SystemExit("STRICT_K10_FEASIBLE_INVALID")
    known_all = all(row["strict_k10_known_mask"] for row in eligible)
    has_positive = any(
        row["strict_k10_feasible"] and row["strict_k10_known_mask"]
        for row in eligible
    )
    if has_positive:
        return "positive"
    if known_all:
        return "negative"
    return "unknown"


def is_valid_start(row: Mapping[str, Any]) -> bool:
    return (
        row.get("strict_k10_feasible") is True
        and row.get("strict_k10_known_mask") is True
    )


def compute_timing(
    offline_rows: list[dict[str, Any]],
    emit_step: int,
    step_field: str,
) -> tuple[int | None, int | None, float | None]:
    valid = sorted(row[step_field] for row in offline_rows if is_valid_start(row))
    if not valid:
        return None, None, None
    regions: list[tuple[int, int]] = []
    region_start = valid[0]
    previous = valid[0]
    for step in valid[1:]:
        if step == previous + 1:
            previous = step
        else:
            regions.append((region_start, previous))
            region_start = step
            previous = step
    regions.append((region_start, previous))
    for region_start, region_end in regions:
        if region_start <= emit_step <= region_end:
            region_length = region_end - region_start + 1
            offset = emit_step - region_start
            relative = offset / max(1, region_length - 1)
            return offset, region_length, relative
    return None, None, None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"JSONL_MISSING:{path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSONL_PARSE_ERROR:{path}:{line_number}:{exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit(f"JSONL_OBJECT_REQUIRED:{path}:{line_number}")
        rows.append(value)
    return rows


def exact_join(
    runtime_file: Path,
    offline_file: Path,
    episode_field: str,
    step_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_rows = _read_jsonl(runtime_file)
    offline_rows = _read_jsonl(offline_file)
    runtime_keys: set[tuple[Any, Any]] = set()
    offline_keys: set[tuple[Any, Any]] = set()

    for row in runtime_rows:
        if episode_field not in row or step_field not in row:
            raise SystemExit("RUNTIME_JOIN_FIELD_MISSING")
        key = (row[episode_field], row[step_field])
        if key in runtime_keys:
            raise SystemExit(f"DUP_RT:{key}")
        runtime_keys.add(key)

    for row in offline_rows:
        if episode_field not in row or step_field not in row:
            raise SystemExit("OFFLINE_JOIN_FIELD_MISSING")
        key = (row[episode_field], row[step_field])
        if key in offline_keys:
            raise SystemExit(f"DUP_OL:{key}")
        offline_keys.add(key)

    if runtime_keys != offline_keys:
        raise SystemExit(
            f"JOIN_MISMATCH:"
            f"runtime_only={len(runtime_keys - offline_keys)}:"
            f"offline_only={len(offline_keys - runtime_keys)}"
        )
    return runtime_rows, offline_rows


def compute_l3_metrics(
    offline_episodes: Mapping[str, list[dict[str, Any]]],
    scheduler_results: Mapping[str, Mapping[str, Any]],
    step_field: str,
) -> dict[str, Any]:
    counts = {
        "negative_episodes": 0,
        "positive_episodes": 0,
        "unknown_episodes": 0,
        "negative_episode_emits": 0,
        "positive_on_corridor_emits": 0,
        "positive_off_corridor_emits": 0,
        "positive_abstentions": 0,
        "unknown_episode_emits": 0,
    }
    offsets: list[int] = []
    per_episode: list[dict[str, Any]] = []

    if set(offline_episodes) != set(scheduler_results):
        raise SystemExit(
            f"EP_CLOSURE:"
            f"offline_only={len(set(offline_episodes) - set(scheduler_results))}:"
            f"scheduler_only={len(set(scheduler_results) - set(offline_episodes))}"
        )

    for episode in sorted(offline_episodes):
        offline_rows = offline_episodes[episode]
        result = scheduler_results[episode]
        emitted = result.get("emitted")
        emit_step = result.get("emit_step")
        if not isinstance(emitted, bool):
            raise SystemExit(f"SCHEDULER_EMITTED_INVALID:{episode}")
        if isinstance(emit_step, bool) or not isinstance(emit_step, int):
            raise SystemExit(f"SCHEDULER_EMIT_STEP_INVALID:{episode}")

        classification = classify_episode(offline_rows, step_field)
        timing_offset = None
        region_length = None
        relative_position = None
        on_corridor = False

        if classification == "unknown":
            counts["unknown_episodes"] += 1
            if emitted:
                counts["unknown_episode_emits"] += 1
        elif classification == "negative":
            counts["negative_episodes"] += 1
            if emitted:
                counts["negative_episode_emits"] += 1
        else:
            counts["positive_episodes"] += 1
            if emitted:
                row = next(
                    (value for value in offline_rows if value[step_field] == emit_step),
                    None,
                )
                if row is None:
                    raise SystemExit(f"EMIT_STEP_NOT_FOUND:{episode}:{emit_step}")
                on_corridor = is_valid_start(row)
                if on_corridor:
                    counts["positive_on_corridor_emits"] += 1
                    timing_offset, region_length, relative_position = compute_timing(
                        offline_rows,
                        emit_step,
                        step_field,
                    )
                    if timing_offset is not None:
                        offsets.append(timing_offset)
                else:
                    counts["positive_off_corridor_emits"] += 1
            else:
                counts["positive_abstentions"] += 1

        per_episode.append(
            {
                "episode_key": episode,
                "classification": classification,
                "scheduler_emitted": emitted,
                "emit_step": emit_step,
                "on_corridor": on_corridor,
                "timing_offset": timing_offset,
                "region_length": region_length,
                "relative_region_position": relative_position,
            }
        )

    total_emitted_all = (
        counts["negative_episode_emits"]
        + counts["positive_on_corridor_emits"]
        + counts["positive_off_corridor_emits"]
        + counts["unknown_episode_emits"]
    )
    total_emitted_verified = (
        counts["negative_episode_emits"]
        + counts["positive_on_corridor_emits"]
        + counts["positive_off_corridor_emits"]
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    return {
        **counts,
        "total_emitted_all": total_emitted_all,
        "total_emitted_verified": total_emitted_verified,
        "negative_episode_false_start_rate": ratio(
            counts["negative_episode_emits"],
            counts["negative_episodes"],
        ),
        "valid_opportunity_recall": ratio(
            counts["positive_on_corridor_emits"],
            counts["positive_episodes"],
        ),
        "all_emit_precision": ratio(
            counts["positive_on_corridor_emits"],
            total_emitted_all,
        ),
        "verified_emit_precision": ratio(
            counts["positive_on_corridor_emits"],
            total_emitted_verified,
        ),
        "invalid_emit_fraction": ratio(
            counts["negative_episode_emits"]
            + counts["positive_off_corridor_emits"],
            total_emitted_verified,
        ),
        "abstention_rate": ratio(
            counts["positive_abstentions"],
            counts["positive_episodes"],
        ),
        "unknown_emit_rate": ratio(
            counts["unknown_episode_emits"],
            counts["unknown_episodes"],
        ),
        "unverifiable_emit_fraction": ratio(
            counts["unknown_episode_emits"],
            total_emitted_all,
        ),
        "median_timing_offset": (
            float(statistics.median(offsets)) if offsets else None
        ),
        "unknown_fraction": ratio(
            counts["unknown_episodes"],
            counts["positive_episodes"]
            + counts["negative_episodes"]
            + counts["unknown_episodes"],
        ),
        "per_episode": per_episode,
    }


def _load_adapter(handoff: Mapping[str, Any]):
    adapter_relative = handoff["runtime_adapter"]["source"]["path"]
    adapter_class = handoff["runtime_adapter"]["class"]
    if not adapter_relative.startswith("src/") or not adapter_relative.endswith(".py"):
        raise SystemExit(f"ADAPTER_PATH_UNEXPECTED:{adapter_relative}")
    module_name = adapter_relative[len("src/") : -len(".py")].replace("/", ".")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, adapter_class)
    except Exception as exc:
        raise SystemExit(f"SCHEDULER_ADAPTER_IMPORT_FAILED:{exc}") from exc


def _group(rows: list[dict[str, Any]], episode_field: str, step_field: str):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode = row.get(episode_field)
        if not isinstance(episode, str) or not episode:
            raise SystemExit("EPISODE_ID_INVALID")
        grouped[episode].append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda value: value[step_field])
        validate_episode_step_sequence(episode_rows, step_field)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-handoff", type=Path, required=True)
    parser.add_argument("--runtime-bundle-root", type=Path, required=True)
    parser.add_argument("--offline-eval-bundle-root", type=Path, required=True)
    parser.add_argument("--calibration-contract-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blocker-receipt-root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("authoritative", "diagnostic"),
        default="diagnostic",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS:{output_root}")

    from load_factorized_handoff import load_handoff_file

    handoff_path = args.codex_handoff.resolve()
    handoff = load_handoff_file(handoff_path, ROOT)
    validator = (
        validate_handoff_execution
        if args.mode == "authoritative"
        else validate_handoff_static
    )
    ok, errors = validator(handoff)
    if not ok:
        raise SystemExit(
            f"HANDOFF_VALIDATION_FAILED:{len(errors)}:{'|'.join(errors)}"
        )

    runtime_description = handoff["runtime_bundle"]
    episode_field = runtime_description["episode_field"]
    step_field = runtime_description["step_field"]
    runtime_filename = runtime_description["data_filename"]
    offline_filename = handoff["offline_bundles"]["evaluation"]["data_filename"]
    AdapterClass = _load_adapter(handoff)

    runtime_root = args.runtime_bundle_root.resolve()
    offline_root = args.offline_eval_bundle_root.resolve()
    calibration_root = args.calibration_contract_root.resolve()
    for root, label in (
        (runtime_root, "RUNTIME"),
        (offline_root, "OFFLINE"),
        (calibration_root, "CALIBRATION"),
    ):
        if not root.is_dir():
            raise SystemExit(f"{label}_ROOT_MISSING:{root}")

    runtime_splits = {path.name for path in runtime_root.iterdir() if path.is_dir()}
    offline_splits = {path.name for path in offline_root.iterdir() if path.is_dir()}
    calibration_splits = {
        path.name for path in calibration_root.iterdir() if path.is_dir()
    }
    if runtime_splits != offline_splits or runtime_splits != calibration_splits:
        raise SystemExit("RUNTIME_OFFLINE_CALIBRATION_SPLIT_MISMATCH")
    if runtime_splits != EXPECTED_SPLITS:
        raise SystemExit(
            f"EXACT_12_SPLIT_CLOSURE_REQUIRED:"
            f"missing={sorted(EXPECTED_SPLITS - runtime_splits)}:"
            f"extra={sorted(runtime_splits - EXPECTED_SPLITS)}"
        )

    contracts: dict[str, dict[str, Any]] = {}
    for split in sorted(EXPECTED_SPLITS):
        verify_sealed_directory(runtime_root / split)
        verify_sealed_directory(offline_root / split)
        verify_sealed_directory(calibration_root / split)
        contract_path = (
            calibration_root
            / split
            / "calibration_and_threshold_contract.json"
        )
        contract = load_json(contract_path)
        if (
            contract.get("schema")
            != "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3"
        ):
            raise SystemExit(f"CAL_WRONG_SCHEMA:{split}")
        if contract.get("split") != split:
            raise SystemExit(f"CAL_SPLIT_MISMATCH:{split}")
        if args.mode == "authoritative":
            if (
                contract.get("status") != "AUTHORITATIVE"
                or contract.get("calibration_fit_authoritative") is not True
                or contract.get("threshold_selection_authoritative") is not True
                or contract.get("l3_evaluation_eligible") is not True
            ):
                raise SystemExit(f"CAL_NOT_AUTHORITATIVE:{split}")
            for head in ("grasp", "manipulation", "release"):
                value = contract.get(head)
                if (
                    not isinstance(value, dict)
                    or value.get("provenance_class") != "INDEPENDENT_CALIBRATION"
                    or value.get("fit_data_valid") is not True
                ):
                    raise SystemExit(f"CAL_NOT_INDEPENDENT:{split}:{head}")
        contracts[split] = contract

    structure = load_json(ROOT / handoff["structural_config"]["path"])
    all_metrics: dict[str, dict[str, Any]] = {}
    all_contracts_eligible = True

    for split in sorted(EXPECTED_SPLITS):
        runtime_file = runtime_root / split / runtime_filename
        offline_file = offline_root / split / offline_filename
        runtime_rows, offline_rows = exact_join(
            runtime_file,
            offline_file,
            episode_field,
            step_field,
        )
        runtime_episodes = _group(runtime_rows, episode_field, step_field)
        offline_episodes = _group(offline_rows, episode_field, step_field)

        adapter = AdapterClass(
            structure=structure,
            calibration_contract=contracts[split],
            require_l3_eligible=args.mode == "authoritative",
        )
        all_contracts_eligible = (
            all_contracts_eligible and adapter.l3_evaluation_eligible
        )

        scheduler_results: dict[str, dict[str, Any]] = {}
        for episode in sorted(runtime_episodes):
            result = adapter.run_episode(runtime_episodes[episode])
            required = {
                "per_step_trace",
                "ever_emitted",
                "first_emit_step",
                "first_emit_trace",
                "final_state",
                "reason_histogram",
                "l3_evaluation_eligible",
                "diagnostic_only",
            }
            if not required.issubset(result):
                raise SystemExit(
                    f"ADAPTER_RESULT_FIELDS_MISSING:"
                    f"{sorted(required - set(result))}"
                )
            scheduler_results[episode] = {
                "emitted": result["ever_emitted"],
                "emit_step": (
                    result["first_emit_step"]
                    if result["first_emit_step"] is not None
                    else -1
                ),
                "final_state": result["final_state"],
            }

        all_metrics[split] = compute_l3_metrics(
            offline_episodes,
            scheduler_results,
            step_field,
        )

    false_rates = {
        split: metrics["negative_episode_false_start_rate"]
        for split, metrics in all_metrics.items()
    }
    defined = {split: value for split, value in false_rates.items() if value is not None}
    worst = max(defined.values()) if defined else None
    undefined = sorted(split for split, value in false_rates.items() if value is None)

    authoritative_l3 = (
        args.mode == "authoritative"
        and all_contracts_eligible
        and not undefined
    )
    if args.mode == "authoritative" and not authoritative_l3:
        raise SystemExit("AUTHORITATIVE_L3_CLOSURE_FAILED")

    staging = output_root.with_name(
        f".{output_root.name}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir(parents=True)
    manifest = {
        "analysis": "FACTORIZED_V2_L3_ANALYSIS_V2",
        "authoritative_l3": authoritative_l3,
        "runner_status": (
            "AUTHORITATIVE_L3_COMPLETE"
            if authoritative_l3
            else "NON_AUTHORITATIVE_DIAGNOSTIC_COMPLETE"
        ),
        "mode": args.mode,
        "codex_handoff_sha256": sha256_file(handoff_path),
        "all_contracts_l3_eligible": all_contracts_eligible,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (staging / "l3_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "per_split_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "summary.json").write_text(
        json.dumps(
            {
                "n_splits": len(all_metrics),
                "worst_false_start": worst,
                "undefined_splits": undefined,
                "n_undefined": len(undefined),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    files = sorted(path for path in staging.iterdir() if path.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    (staging / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n",
        encoding="utf-8",
    )
    os.replace(staging, output_root)
    print(f"Sealed:{output_root}:authoritative_l3={authoritative_l3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
