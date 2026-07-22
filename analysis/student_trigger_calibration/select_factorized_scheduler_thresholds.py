#!/usr/bin/env python3
"""Select Factorized V2 scheduler thresholds on independent policy-selection data.

The selector replays the real Codex ``FactorizedV2SchedulerAdapter`` over all
12 split-scoped runtime streams and joins them exactly to offline strict-K10
labels. It performs one global joint grid search over grasp, manipulation and
release-veto thresholds.

Selection rule (pre-registered):
1. every split must have a defined negative-episode false-start rate;
2. worst-split false-start rate must be <= ``--max-false-start``;
3. maximize aggregate valid-opportunity recall;
4. maximize aggregate all-emit precision;
5. minimize median on-corridor timing offset;
6. deterministic conservative threshold tie-break.

The policy-selection identities must be pairwise disjoint from calibration-fit
and checkpoint-training identities. Heldout L3 evaluation data is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter  # noqa: E402
from fit_factorized_calibrators import (  # noqa: E402
    HEADS,
    load_json,
    sha256_file,
    verify_sealed_directory,
)
from run_factorized_l3_analysis import (  # noqa: E402
    compute_l3_metrics,
    exact_join,
    validate_episode_step_sequence,
)

EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class ThresholdSelectionError(ValueError):
    pass


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ThresholdSelectionError(f"{label}_SHA_INVALID")
    return value.lower()


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise ThresholdSelectionError(f"{label}_COMMIT_INVALID")
    return value.lower()


def _identity_set(manifest: Mapping[str, Any], keys: Sequence[str], label: str) -> set[str]:
    values = None
    for key in keys:
        if key in manifest:
            values = manifest[key]
            break
    if not isinstance(values, list) or not values:
        raise ThresholdSelectionError(f"{label}_IDENTITIES_MISSING")
    identities: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ThresholdSelectionError(f"{label}_IDENTITY_INVALID")
        if value in identities:
            raise ThresholdSelectionError(f"{label}_IDENTITY_DUPLICATE:{value}")
        identities.add(value)
    return identities


def _parse_grid(text: str, label: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ThresholdSelectionError(f"{label}_GRID_INVALID:{item}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ThresholdSelectionError(f"{label}_GRID_OUT_OF_RANGE:{value}")
        values.append(value)
    result = tuple(sorted(set(values)))
    if not result:
        raise ThresholdSelectionError(f"{label}_GRID_EMPTY")
    return result


def _require_split_directories(root: Path, label: str) -> None:
    if not root.is_dir():
        raise ThresholdSelectionError(f"{label}_ROOT_MISSING:{root}")
    found = {path.name for path in root.iterdir() if path.is_dir()}
    expected = set(EXPECTED_SPLITS)
    if found != expected:
        raise ThresholdSelectionError(
            f"{label}_SPLIT_CLOSURE_FAIL:"
            f"missing={sorted(expected - found)}:extra={sorted(found - expected)}"
        )


def _group(rows: list[dict[str, Any]], episode_field: str, step_field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode = row.get(episode_field)
        if not isinstance(episode, str) or not episode:
            raise ThresholdSelectionError("EPISODE_ID_INVALID")
        grouped[episode].append(row)
    for episode, episode_rows in grouped.items():
        episode_rows.sort(key=lambda value: value[step_field])
        validate_episode_step_sequence(episode_rows, step_field)
    return dict(grouped)


def _binding(rows: list[dict[str, Any]], split: str) -> dict[str, str]:
    required = (
        "checkpoint_sha256",
        "source_commit",
        "feature_order_sha256",
        "split",
        "scheduler_source_sha256",
        "structural_config_sha256",
    )
    if not rows:
        raise ThresholdSelectionError(f"RUNTIME_ROWS_EMPTY:{split}")
    values: dict[str, set[str]] = {field: set() for field in required}
    for row in rows:
        for field in required:
            value = row.get(field)
            if value in (None, ""):
                raise ThresholdSelectionError(f"RUNTIME_BINDING_MISSING:{split}:{field}")
            values[field].add(str(value).lower())
    if any(len(entries) != 1 for entries in values.values()):
        bad = {field: sorted(entries) for field, entries in values.items() if len(entries) != 1}
        raise ThresholdSelectionError(f"RUNTIME_BINDING_NONUNIFORM:{split}:{bad}")
    result = {field: next(iter(entries)) for field, entries in values.items()}
    if result["split"] != split:
        raise ThresholdSelectionError(f"RUNTIME_SPLIT_MISMATCH:{split}:{result['split']}")
    _sha(result["checkpoint_sha256"], f"{split}.checkpoint")
    _commit(result["source_commit"], f"{split}.source")
    _sha(result["feature_order_sha256"], f"{split}.feature")
    _sha(result["scheduler_source_sha256"], f"{split}.scheduler")
    _sha(result["structural_config_sha256"], f"{split}.structure")
    return result


def _calibrators(contract: Mapping[str, Any], split: str) -> dict[str, dict[str, Any]]:
    if contract.get("schema") != "FACTORIZED_V2_CALIBRATION_CONTRACT_V2":
        raise ThresholdSelectionError(f"CALIBRATION_SCHEMA_INVALID:{split}")
    if contract.get("split") != split:
        raise ThresholdSelectionError(f"CALIBRATION_SPLIT_MISMATCH:{split}")
    if contract.get("provenance") != "INDEPENDENT_CALIBRATION":
        raise ThresholdSelectionError(f"CALIBRATION_NOT_INDEPENDENT:{split}")
    if contract.get("authoritative") is not True or contract.get("all_heads_valid") is not True:
        raise ThresholdSelectionError(f"CALIBRATION_NOT_AUTHORITATIVE:{split}")
    values = contract.get("calibrators")
    if not isinstance(values, list):
        raise ThresholdSelectionError(f"CALIBRATORS_MISSING:{split}")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or value.get("head") not in HEADS:
            raise ThresholdSelectionError(f"CALIBRATOR_INVALID:{split}")
        if value["head"] in result:
            raise ThresholdSelectionError(f"CALIBRATOR_DUPLICATE:{split}:{value['head']}")
        if value.get("method_valid") is not True:
            raise ThresholdSelectionError(f"CALIBRATOR_METHOD_INVALID:{split}:{value['head']}")
        result[value["head"]] = value
    if set(result) != set(HEADS):
        raise ThresholdSelectionError(f"CALIBRATOR_HEAD_CLOSURE:{split}")
    return result


def provisional_v3_contract(
    *,
    split: str,
    calibration_contract: Mapping[str, Any],
    thresholds: Mapping[str, float],
    binding: Mapping[str, str],
    fit_manifest_sha256: str,
    policy_manifest_sha256: str,
) -> dict[str, Any]:
    heads = _calibrators(calibration_contract, split)
    output: dict[str, Any] = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
        "status": "DIAGNOSTIC",
        "split": split,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "scheduler_source_sha256": binding["scheduler_source_sha256"],
        "structural_config_sha256": binding["structural_config_sha256"],
        "student_source_commit": binding["source_commit"],
        "feature_order_sha256": binding["feature_order_sha256"],
        "calibration_fit_authoritative": True,
        "threshold_selection_authoritative": False,
        "l3_evaluation_eligible": False,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
    }
    for head in HEADS:
        value = heads[head]
        output[head] = {
            "method": value["method"],
            "a": float(value["a"]),
            "b": float(value["b"]),
            "threshold": float(thresholds[head]),
            "transform": "probability=sigmoid(a*raw_logit+b)",
            "method_valid": True,
            "transform_valid": True,
            "fit_data_valid": True,
            "provenance_class": "INDEPENDENT_CALIBRATION",
            "fit_manifest_sha256": fit_manifest_sha256,
            "policy_selection_manifest_sha256": policy_manifest_sha256,
        }
    return output


def load_split_payload(
    *,
    split: str,
    runtime_root: Path | None,
    evaluation_root: Path | None,
    calibration_contract_root: Path,
    policy_manifest_root: Path | None,
    fit_manifest_root: Path,
    checkpoint_manifest_root: Path,
    structure: Mapping[str, Any],
    policy_selection_bundle_root: Path | None = None,
) -> dict[str, Any]:
    calibration_split = calibration_contract_root / split
    verify_sealed_directory(calibration_split)

    if policy_selection_bundle_root is not None:
        policy_split = policy_selection_bundle_root / split
        verify_sealed_directory(policy_split)
        runtime_file = policy_split / "policy_selection_runtime_records.jsonl"
        evaluation_file = policy_split / "policy_selection_evaluation_labels.jsonl"
        policy_manifest_path = policy_split / "manifest.json"
    else:
        if runtime_root is None or evaluation_root is None or policy_manifest_root is None:
            raise ThresholdSelectionError(f"POLICY_SELECTION_BUNDLE_REQUIRED:{split}")
        runtime_split = runtime_root / split
        evaluation_split = evaluation_root / split
        verify_sealed_directory(runtime_split)
        verify_sealed_directory(evaluation_split)
        runtime_file = runtime_split / "runtime_scheduler_inputs.jsonl"
        evaluation_file = evaluation_split / "evaluation_records.jsonl"
        policy_manifest_path = policy_manifest_root / split / "manifest.json"
    if not runtime_file.is_file() or not evaluation_file.is_file():
        raise ThresholdSelectionError(f"POLICY_STREAM_MISSING:{split}")

    runtime_rows, evaluation_rows = exact_join(
        runtime_file,
        evaluation_file,
        "episode",
        "step",
    )
    runtime_episodes = _group(runtime_rows, "episode", "step")
    evaluation_episodes = _group(evaluation_rows, "episode", "step")
    if set(runtime_episodes) != set(evaluation_episodes):
        raise ThresholdSelectionError(f"POLICY_EPISODE_CLOSURE_FAIL:{split}")

    fit_manifest_path = fit_manifest_root / split / "manifest.json"
    checkpoint_manifest_path = checkpoint_manifest_root / split / "manifest.json"
    policy_manifest = load_json(policy_manifest_path)
    fit_manifest = load_json(fit_manifest_path)
    checkpoint_manifest = load_json(checkpoint_manifest_path)

    policy_ids = _identity_set(
        policy_manifest,
        ("policy_selection_identities", "identities"),
        f"{split}.POLICY",
    )
    fit_ids = _identity_set(
        fit_manifest,
        ("fit_identities", "identities"),
        f"{split}.FIT",
    )
    training_ids = _identity_set(
        checkpoint_manifest,
        ("training_identities", "train_identities", "identities"),
        f"{split}.TRAINING",
    )
    if policy_ids & fit_ids:
        raise ThresholdSelectionError(f"POLICY_FIT_LEAKAGE:{split}:{sorted(policy_ids & fit_ids)}")
    if policy_ids & training_ids:
        raise ThresholdSelectionError(
            f"POLICY_TRAINING_LEAKAGE:{split}:{sorted(policy_ids & training_ids)}"
        )
    if set(runtime_episodes) != policy_ids:
        raise ThresholdSelectionError(
            f"POLICY_IDENTITY_CLOSURE_FAIL:{split}:"
            f"missing={sorted(policy_ids - set(runtime_episodes))}:"
            f"extra={sorted(set(runtime_episodes) - policy_ids)}"
        )

    binding = _binding(runtime_rows, split)
    for field in (
        "checkpoint_sha256",
        "source_commit",
        "feature_order_sha256",
        "scheduler_source_sha256",
        "structural_config_sha256",
    ):
        if str(policy_manifest.get(field, "")).lower() != binding[field]:
            raise ThresholdSelectionError(f"POLICY_MANIFEST_RUNTIME_MISMATCH:{split}:{field}")

    calibration_path = calibration_split / "calibration_contract.json"
    calibration_contract = load_json(calibration_path)
    if str(calibration_contract.get("checkpoint_sha256", "")).lower() != binding["checkpoint_sha256"]:
        raise ThresholdSelectionError(f"CALIBRATION_RUNTIME_CHECKPOINT_MISMATCH:{split}")
    if str(calibration_contract.get("student_source_commit", "")).lower() != binding["source_commit"]:
        raise ThresholdSelectionError(f"CALIBRATION_RUNTIME_SOURCE_MISMATCH:{split}")
    if str(checkpoint_manifest.get("checkpoint_sha256", "")).lower() != binding["checkpoint_sha256"]:
        raise ThresholdSelectionError(f"CHECKPOINT_MANIFEST_RUNTIME_MISMATCH:{split}")

    _calibrators(calibration_contract, split)
    return {
        "split": split,
        "runtime_episodes": runtime_episodes,
        "evaluation_episodes": evaluation_episodes,
        "calibration_contract": calibration_contract,
        "calibration_contract_sha256": sha256_file(calibration_path),
        "fit_manifest_sha256": sha256_file(fit_manifest_path),
        "policy_manifest_sha256": sha256_file(policy_manifest_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "binding": binding,
        "structure": dict(structure),
    }


def evaluate_candidate(
    payloads: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    per_split: dict[str, dict[str, Any]] = {}
    all_offsets: list[float] = []
    totals = {
        "negative_episodes": 0,
        "positive_episodes": 0,
        "unknown_episodes": 0,
        "negative_episode_emits": 0,
        "positive_on_corridor_emits": 0,
        "positive_off_corridor_emits": 0,
        "positive_abstentions": 0,
        "unknown_episode_emits": 0,
        "total_emitted_all": 0,
        "total_emitted_verified": 0,
    }

    for payload in payloads:
        split = str(payload["split"])
        contract = provisional_v3_contract(
            split=split,
            calibration_contract=payload["calibration_contract"],
            thresholds=thresholds,
            binding=payload["binding"],
            fit_manifest_sha256=payload["fit_manifest_sha256"],
            policy_manifest_sha256=payload["policy_manifest_sha256"],
        )
        adapter = FactorizedV2SchedulerAdapter(
            structure=payload["structure"],
            calibration_contract=contract,
            require_l3_eligible=False,
        )
        scheduler_results: dict[str, dict[str, Any]] = {}
        for episode, rows in sorted(payload["runtime_episodes"].items()):
            result = adapter.run_episode(rows)
            scheduler_results[episode] = {
                "emitted": result["ever_emitted"],
                "emit_step": result["first_emit_step"] if result["first_emit_step"] is not None else -1,
                "final_state": result["final_state"],
            }

        metrics = compute_l3_metrics(
            payload["evaluation_episodes"],
            scheduler_results,
            "step",
        )
        per_split[split] = {
            key: value
            for key, value in metrics.items()
            if key != "per_episode"
        }
        for row in metrics["per_episode"]:
            if row.get("timing_offset") is not None:
                all_offsets.append(float(row["timing_offset"]))
        for key in totals:
            totals[key] += int(metrics[key])

    false_rates = {
        split: metrics["negative_episode_false_start_rate"]
        for split, metrics in per_split.items()
    }
    defined = [value for value in false_rates.values() if value is not None]
    all_defined = len(defined) == len(per_split)
    worst_false_start = max(defined) if defined else None

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    aggregate = {
        **totals,
        "negative_episode_false_start_rate": ratio(
            totals["negative_episode_emits"],
            totals["negative_episodes"],
        ),
        "valid_opportunity_recall": ratio(
            totals["positive_on_corridor_emits"],
            totals["positive_episodes"],
        ),
        "all_emit_precision": ratio(
            totals["positive_on_corridor_emits"],
            totals["total_emitted_all"],
        ),
        "verified_emit_precision": ratio(
            totals["positive_on_corridor_emits"],
            totals["total_emitted_verified"],
        ),
        "median_timing_offset": (
            float(statistics.median(all_offsets)) if all_offsets else None
        ),
    }
    return {
        "thresholds": dict(thresholds),
        "all_split_false_start_defined": all_defined,
        "worst_split_negative_false_start_rate": worst_false_start,
        "per_split": per_split,
        "aggregate": aggregate,
    }


def candidate_selection_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    aggregate = result["aggregate"]
    recall = aggregate["valid_opportunity_recall"]
    all_precision = aggregate["all_emit_precision"]
    timing = aggregate["median_timing_offset"]
    thresholds = result["thresholds"]
    return (
        -1.0 if recall is None else float(recall),
        -1.0 if all_precision is None else float(all_precision),
        float("-inf") if timing is None else -float(timing),
        float(thresholds["grasp"]),
        float(thresholds["manipulation"]),
        -float(thresholds["release"]),
    )


def select_thresholds(
    payloads: Sequence[Mapping[str, Any]],
    *,
    grasp_grid: Sequence[float],
    manipulation_grid: Sequence[float],
    release_grid: Sequence[float],
    max_false_start: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for grasp, manipulation, release in itertools.product(
        grasp_grid,
        manipulation_grid,
        release_grid,
    ):
        result = evaluate_candidate(
            payloads,
            {"grasp": grasp, "manipulation": manipulation, "release": release},
        )
        result["constraint_pass"] = (
            result["all_split_false_start_defined"]
            and result["worst_split_negative_false_start_rate"] is not None
            and result["worst_split_negative_false_start_rate"] <= max_false_start
        )
        results.append(result)
        if result["constraint_pass"] and (
            best is None or candidate_selection_key(result) > candidate_selection_key(best)
        ):
            best = result
    return best, results


def _aggregate_manifest_sha(entries: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(sorted(entries.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _seal_output(output_root: Path, files: Mapping[str, Any]) -> None:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, value in files.items():
        (staging / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    data_files = sorted(path for path in staging.iterdir() if path.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in data_files),
        encoding="utf-8",
    )
    (staging / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n",
        encoding="utf-8",
    )
    os.replace(staging, output_root)


def blocker(output_root: Path, reason: str, details: Mapping[str, Any]) -> int:
    receipt = {
        "schema": "FACTORIZED_V2_THRESHOLD_SELECTION_BLOCKER_RECEIPT_V1",
        "status": reason,
        "details": dict(details),
        "authoritative_l3": False,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _seal_output(output_root, {"BLOCKER_RECEIPT.json": receipt})
    print(f"BLOCKER:{reason}:{output_root}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-bundle-root", type=Path)
    parser.add_argument("--offline-eval-bundle-root", type=Path)
    parser.add_argument("--calibration-contract-root", type=Path, required=True)
    parser.add_argument("--policy-selection-manifest-root", type=Path)
    parser.add_argument("--policy-selection-bundle-root", type=Path)
    parser.add_argument("--calibration-fit-manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    parser.add_argument(
        "--structure-config",
        type=Path,
        default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grasp-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--manipulation-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--release-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--max-false-start", type=float, default=0.10)
    args = parser.parse_args()

    if args.policy_selection_bundle_root is None and (
        args.runtime_bundle_root is None
        or args.offline_eval_bundle_root is None
        or args.policy_selection_manifest_root is None
    ):
        raise SystemExit(
            "POLICY_SELECTION_INPUT_REQUIRED: provide --policy-selection-bundle-root "
            "or the legacy runtime/evaluation/manifest roots"
        )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS:{output_root}")
    if not math.isfinite(args.max_false_start) or not 0.0 <= args.max_false_start <= 1.0:
        return blocker(output_root, "BLOCKED_FALSE_START_BOUND_INVALID", {})

    try:
        grasp_grid = _parse_grid(args.grasp_grid, "GRASP")
        manipulation_grid = _parse_grid(args.manipulation_grid, "MANIPULATION")
        release_grid = _parse_grid(args.release_grid, "RELEASE")
        combinations = len(grasp_grid) * len(manipulation_grid) * len(release_grid)
        if combinations > 4096:
            raise ThresholdSelectionError(f"GRID_TOO_LARGE:{combinations}")

        structure_path = args.structure_config.resolve()
        structure = load_json(structure_path)
        structural_sha = sha256_file(structure_path)
        scheduler_path = ROOT / "src/gripper_attack/factorized_scheduler.py"
        scheduler_sha = sha256_file(scheduler_path)

        roots = [
            (args.calibration_contract_root.resolve(), "CALIBRATION"),
            (args.calibration_fit_manifest_root.resolve(), "FIT_MANIFEST"),
            (args.checkpoint_manifest_root.resolve(), "CHECKPOINT_MANIFEST"),
        ]
        if args.policy_selection_bundle_root is not None:
            roots.append((args.policy_selection_bundle_root.resolve(), "POLICY_SELECTION_BUNDLE"))
        else:
            roots.extend([
                (args.runtime_bundle_root.resolve(), "RUNTIME"),
                (args.offline_eval_bundle_root.resolve(), "EVALUATION"),
                (args.policy_selection_manifest_root.resolve(), "POLICY_MANIFEST"),
            ])
        for root, label in roots:
            _require_split_directories(root, label)

        payloads = [
            load_split_payload(
                split=split,
                runtime_root=args.runtime_bundle_root.resolve() if args.runtime_bundle_root is not None else None,
                evaluation_root=args.offline_eval_bundle_root.resolve() if args.offline_eval_bundle_root is not None else None,
                calibration_contract_root=args.calibration_contract_root.resolve(),
                policy_manifest_root=args.policy_selection_manifest_root.resolve() if args.policy_selection_manifest_root is not None else None,
                fit_manifest_root=args.calibration_fit_manifest_root.resolve(),
                checkpoint_manifest_root=args.checkpoint_manifest_root.resolve(),
                structure=structure,
                policy_selection_bundle_root=args.policy_selection_bundle_root.resolve() if args.policy_selection_bundle_root is not None else None,
            )
            for split in EXPECTED_SPLITS
        ]
        for payload in payloads:
            binding = payload["binding"]
            if binding["structural_config_sha256"] != structural_sha:
                raise ThresholdSelectionError(
                    f"STRUCTURAL_CONFIG_BINDING_MISMATCH:{payload['split']}"
                )
            if binding["scheduler_source_sha256"] != scheduler_sha:
                raise ThresholdSelectionError(
                    f"SCHEDULER_SOURCE_BINDING_MISMATCH:{payload['split']}"
                )

        best, all_results = select_thresholds(
            payloads,
            grasp_grid=grasp_grid,
            manipulation_grid=manipulation_grid,
            release_grid=release_grid,
            max_false_start=args.max_false_start,
        )
    except (SystemExit, ThresholdSelectionError, ValueError) as exc:
        return blocker(output_root, "BLOCKED_INPUT_OR_PROVENANCE_INVALID", {"reason": str(exc)})

    policy_manifest_digests = {
        payload["split"]: payload["policy_manifest_sha256"]
        for payload in payloads
    }
    fit_manifest_digests = {
        payload["split"]: payload["fit_manifest_sha256"]
        for payload in payloads
    }
    checkpoint_manifest_digests = {
        payload["split"]: payload["checkpoint_manifest_sha256"]
        for payload in payloads
    }
    if best is None:
        return blocker(
            output_root,
            "HOLD_NO_FEASIBLE_THRESHOLD_COMBINATION",
            {
                "grid_combinations": len(all_results),
                "max_false_start": args.max_false_start,
                "best_observed_worst_false_start": min(
                    (
                        result["worst_split_negative_false_start_rate"]
                        for result in all_results
                        if result["worst_split_negative_false_start_rate"] is not None
                    ),
                    default=None,
                ),
            },
        )

    calibration_digests = {
        payload["split"]: payload["calibration_contract_sha256"]
        for payload in payloads
    }
    checkpoint_by_split = {
        payload["split"]: payload["binding"]["checkpoint_sha256"]
        for payload in payloads
    }
    source_by_split = {
        payload["split"]: payload["binding"]["source_commit"]
        for payload in payloads
    }
    feature_by_split = {
        payload["split"]: payload["binding"]["feature_order_sha256"]
        for payload in payloads
    }

    contract = {
        "schema": "FACTORIZED_V2_THRESHOLD_SELECTION_CONTRACT_V2",
        "status": "COMPLETE",
        "provenance": "INDEPENDENT_POLICY_SELECTION",
        "expected_splits": list(EXPECTED_SPLITS),
        "selected_thresholds": best["thresholds"],
        "selection_rule": {
            "max_worst_split_negative_false_start_rate": args.max_false_start,
            "objective_order": [
                "maximize aggregate valid_opportunity_recall",
                "maximize aggregate all_emit_precision",
                "minimize median on-corridor timing offset",
                "prefer higher grasp threshold",
                "prefer higher manipulation threshold",
                "prefer lower release-veto threshold",
            ],
            "grid": {
                "grasp": list(grasp_grid),
                "manipulation": list(manipulation_grid),
                "release": list(release_grid),
            },
            "grid_combinations": len(all_results),
        },
        "selected_metrics": {
            "worst_split_negative_false_start_rate": best[
                "worst_split_negative_false_start_rate"
            ],
            "aggregate": best["aggregate"],
            "per_split": best["per_split"],
        },
        "policy_selection_manifest_sha256": _aggregate_manifest_sha(
            policy_manifest_digests
        ),
        "policy_selection_manifest_sha256_by_split": policy_manifest_digests,
        "calibration_fit_manifest_sha256_by_split": fit_manifest_digests,
        "checkpoint_manifest_sha256_by_split": checkpoint_manifest_digests,
        "calibration_contract_sha256_by_split": calibration_digests,
        "checkpoint_sha256_by_split": checkpoint_by_split,
        "student_source_commit_by_split": source_by_split,
        "feature_order_sha256_by_split": feature_by_split,
        "scheduler_source_sha256": scheduler_sha,
        "structural_config_sha256": structural_sha,
        "formal_selection_eligible": True,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    audit = {
        "schema": "FACTORIZED_V2_THRESHOLD_GRID_AUDIT_V1",
        "candidate_count": len(all_results),
        "feasible_candidate_count": sum(
            1 for result in all_results if result["constraint_pass"]
        ),
        "selected_thresholds": best["thresholds"],
        "candidates": [
            {
                "thresholds": result["thresholds"],
                "constraint_pass": result["constraint_pass"],
                "worst_split_negative_false_start_rate": result[
                    "worst_split_negative_false_start_rate"
                ],
                "aggregate": result["aggregate"],
            }
            for result in all_results
        ],
    }
    _seal_output(
        output_root,
        {
            "threshold_contract.json": contract,
            "threshold_grid_audit.json": audit,
        },
    )
    print(f"Threshold contract sealed:{output_root}:status=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
