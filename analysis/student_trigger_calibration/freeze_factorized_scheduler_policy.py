#!/usr/bin/env python3
"""Freeze joint scheduler thresholds from P Student predictions and P Teacher/K10 labels.

Reads only policy-selection identities (P). Never reads H, A, or modifies
calibration. Joint grid search with pre-registered selection rule.

Produces a sealed FACTORIZED_SCHEDULER_FREEZE_V1.json.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, math, os, statistics, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from fit_factorized_calibrators import HEADS, load_json, sha256_file, verify_sealed_directory
from run_factorized_l3_analysis import (
    compute_l3_metrics, exact_join, validate_episode_step_sequence, classify_episode,
    is_valid_start,
)
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter

FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
SELF_SHA = None


def _identity_set(manifest: dict[str, Any], role: str, split_key: str) -> set[str]:
    if "identities" in manifest:
        return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    if split_key in splits:
        sd = splits[split_key]
        if isinstance(sd, list): return set(sd)
        if isinstance(sd, dict): return set(sd.get(role, []))
    if role in manifest:
        rd = manifest[role]
        if isinstance(rd, list): return set(rd)
    return set()


def _parse_grid(text: str, label: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item: continue
        try:
            v = float(item)
        except ValueError:
            raise SystemExit(f"{label}_GRID_INVALID: {item}")
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise SystemExit(f"{label}_GRID_RANGE: {v}")
        values.append(v)
    result = tuple(sorted(set(values)))
    if not result:
        raise SystemExit(f"{label}_GRID_EMPTY")
    return result


def _group(rows: list[dict[str, Any]], ep_field: str, step_field: str) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[r[ep_field]].append(r)
    for ep, ep_rows in g.items():
        ep_rows.sort(key=lambda r2: r2[step_field])
        validate_episode_step_sequence(ep_rows, step_field)
    return dict(g)


def evaluate_candidate(
    payloads: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    structure: Mapping[str, Any],
) -> dict[str, Any]:
    per_split: dict[str, dict[str, Any]] = {}
    all_offsets: list[float] = []
    totals = {"negative_episodes": 0, "positive_episodes": 0, "unknown_episodes": 0,
              "negative_episode_emits": 0, "positive_on_corridor_emits": 0,
              "positive_off_corridor_emits": 0, "positive_abstentions": 0,
              "unknown_episode_emits": 0, "total_emitted_all": 0, "total_emitted_verified": 0}

    for payload in payloads:
        split = str(payload["split"])
        contract = {
            "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
            "status": "DIAGNOSTIC", "split": split,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "scheduler_source_sha256": payload["scheduler_source_sha256"],
            "structural_config_sha256": sha256_file(payload["structure_path"]),
            "student_source_commit": payload["student_source_commit"],
            "feature_order_sha256": payload["feature_order_sha256"],
            "calibration_fit_authoritative": True,
            "threshold_selection_authoritative": False,
            "l3_evaluation_eligible": False,
            "training_authorized": False, "full_fit_authorized": False,
            "attack_authorized": False,
        }
        for head in HEADS:
            hd = payload["calibrators"][head]
            contract[head] = {
                "method": hd["method"], "a": float(hd["a"]), "b": float(hd["b"]),
                "threshold": float(thresholds[head]),
                "transform": "probability=sigmoid(a*raw_logit+b)",
                "method_valid": True, "transform_valid": True,
                "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION",
                "fit_manifest_sha256": payload["fit_manifest_sha256"],
                "policy_selection_manifest_sha256": payload["policy_manifest_sha256"],
            }

        adapter = FactorizedV2SchedulerAdapter(structure=structure, calibration_contract=contract, require_l3_eligible=False)
        scheduler_results: dict[str, dict[str, Any]] = {}
        for episode, rows in sorted(payload["runtime_episodes"].items()):
            result = adapter.run_episode(rows)
            scheduler_results[episode] = {
                "emitted": result["ever_emitted"],
                "emit_step": result["first_emit_step"] if result["first_emit_step"] is not None else -1,
                "final_state": result["final_state"],
            }
        metrics = compute_l3_metrics(payload["evaluation_episodes"], scheduler_results, "step")
        per_split[split] = {k: v for k, v in metrics.items() if k != "per_episode"}
        for row in metrics["per_episode"]:
            if row.get("timing_offset") is not None:
                all_offsets.append(float(row["timing_offset"]))
        for key in totals:
            totals[key] += int(metrics[key])

    false_rates = {split: m["negative_episode_false_start_rate"] for split, m in per_split.items()}
    defined = [v for v in false_rates.values() if v is not None]
    worst = max(defined) if defined else None

    def ratio(n: int, d: int) -> float | None:
        return n / d if d > 0 else None

    aggregate = {**totals,
        "negative_episode_false_start_rate": ratio(totals["negative_episode_emits"], totals["negative_episodes"]),
        "valid_opportunity_recall": ratio(totals["positive_on_corridor_emits"], totals["positive_episodes"]),
        "all_emit_precision": ratio(totals["positive_on_corridor_emits"], totals["total_emitted_all"]),
        "verified_emit_precision": ratio(totals["positive_on_corridor_emits"], totals["total_emitted_verified"]),
        "median_timing_offset": float(statistics.median(all_offsets)) if all_offsets else None,
    }
    return {"thresholds": dict(thresholds), "all_split_false_start_defined": len(defined) == len(per_split),
            "worst_split_negative_false_start_rate": worst, "per_split": per_split, "aggregate": aggregate}


def selection_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    a = result["aggregate"]
    t = result["thresholds"]
    return (
        -1.0 if a["valid_opportunity_recall"] is None else float(a["valid_opportunity_recall"]),
        -1.0 if a["all_emit_precision"] is None else float(a["all_emit_precision"]),
        float("-inf") if a["median_timing_offset"] is None else -float(a["median_timing_offset"]),
        float(t["grasp"]), float(t["manipulation"]), -float(t["release"]),
    )


def _seal_output(output_root: Path, files: dict[str, str]) -> str:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    data = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in data))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, output_root)
    return seal


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--policy-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--policy-teacher-bundle-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--phase-b-receipt", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-receipt", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-receipt", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--grasp-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--manipulation-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--release-grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--max-false-start", type=float, default=0.10)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    if not math.isfinite(args.max_false_start) or not 0.0 <= args.max_false_start <= 1.0:
        raise SystemExit("MAX_FALSE_START_INVALID")

    # Load receipts
    phase_b = load_json(args.phase_b_receipt)
    if not phase_b.get("cp_inference_authorized"):
        raise SystemExit("PHASE_B_CP_NOT_AUTHORIZED")
    cp_val = load_json(args.cp_prediction_validation_receipt)
    if not cp_val.get("cp_predictions_ready"):
        raise SystemExit("CP_PREDICTIONS_NOT_READY")

    # Load calibrator freeze
    freeze_root = args.calibrator_freeze_root.resolve()
    verify_sealed_directory(freeze_root)
    freeze_path = freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"
    freeze_contract = load_json(freeze_path)

    # Load calibrator freeze validation
    cf_validation = load_json(args.calibrator_freeze_validation_receipt)
    if cf_validation.get("status") != "PASS":
        raise SystemExit("CALIBRATOR_FREEZE_VALIDATION_NOT_PASS")

    # Load P manifest and verify seals
    pol_manifest = load_json(args.policy_selection_manifest)
    p_pred_root = args.policy_prediction_bundle_root.resolve()
    p_teacher_root = args.policy_teacher_bundle_root.resolve()
    verify_sealed_directory(p_pred_root)
    verify_sealed_directory(p_teacher_root)

    # Grids
    grasp_grid = _parse_grid(args.grasp_grid, "GRASP")
    manipulation_grid = _parse_grid(args.manipulation_grid, "MANIPULATION")
    release_grid = _parse_grid(args.release_grid, "RELEASE")

    # Structure config
    structure_path = args.structure_config.resolve()
    structure = load_json(structure_path)
    structural_sha = sha256_file(structure_path)
    scheduler_path = ROOT / "src/gripper_attack/factorized_scheduler.py"
    scheduler_source_sha = sha256_file(scheduler_path)
    adapter_path = ROOT / "src/gripper_attack/factorized_scheduler_adapter.py"
    adapter_source_sha = sha256_file(adapter_path)

    expected = [s.strip() for s in args.expected_splits.split(",")]

    # Build payloads
    payloads: list[dict[str, Any]] = []
    fit_manifest_shas: dict[str, str] = {}
    pol_manifest_shas: dict[str, str] = {}
    for sk in expected:
        p_ids = _identity_set(pol_manifest, "policy_selection", sk)

        # Load predictions
        pred_path = p_pred_root / sk / "predictions.jsonl"
        if not pred_path.is_file():
            raise SystemExit(f"P_PRED_MISSING: {sk}")
        pred_rows: list[dict[str, Any]] = []
        for ln, line in enumerate(pred_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            pred_rows.append(json.loads(line))

        pred_ids = {r["canonical_parent_key"] for r in pred_rows}
        if pred_ids != p_ids:
            raise SystemExit(f"P_ID_CLOSURE: {sk}")

        # Load Teacher labels
        teacher_path = p_teacher_root / sk / "factorized_teacher_v1.jsonl"
        if not teacher_path.is_file():
            raise SystemExit(f"P_TEACHER_MISSING: {sk}")
        teacher_rows: list[dict[str, Any]] = []
        for ln, line in enumerate(teacher_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip(): continue
            teacher_rows.append(json.loads(line))

        # Build runtime and evaluation episodes
        runtime_episodes: dict[str, list[dict[str, Any]]] = {}
        evaluation_episodes: dict[str, list[dict[str, Any]]] = {}
        pred_by_key = {(r["canonical_parent_key"], r["step"]): r for r in pred_rows}
        for t_row in teacher_rows:
            ep = t_row["canonical_parent_key"]
            step = t_row["step"]
            key = (ep, step)
            if key not in pred_by_key:
                raise SystemExit(f"P_JOIN_FAIL: {sk} {key}")
            p_row = pred_by_key[key]
            runtime_row = {
                "episode": ep, "step": step, "split": sk,
                "checkpoint_sha256": p_row.get("checkpoint_sha256", ""),
                "source_commit": p_row.get("checkpoint_source_commit", ""),
                "feature_order_sha256": p_row.get("feature_order_sha256", ""),
                "scheduler_source_sha256": scheduler_source_sha,
                "structural_config_sha256": structural_sha,
                "candidate_close": False, "action_known": True,
                "student_valid": True, "route_supported": True,
                "grasp_logit": p_row["grasp_logit"],
                "manipulation_logit": p_row["manipulation_logit"],
                "release_logit": p_row["release_logit"],
            }
            runtime_episodes.setdefault(ep, []).append(runtime_row)
            eval_row = {
                "step_index": step, "canonical_parent_key": ep, "step": step,
                "strict_k10_feasible": t_row.get("strict_k10_feasible", False),
                "strict_k10_known_mask": t_row.get("strict_k10_known_mask", False),
            }
            evaluation_episodes.setdefault(ep, []).append(eval_row)

        for ep_rows in runtime_episodes.values():
            ep_rows.sort(key=lambda r: r["step"])
        for ep_rows in evaluation_episodes.values():
            ep_rows.sort(key=lambda r: r["step"])

        checkpoint_sha = pred_rows[0].get("checkpoint_sha256", "")

        # Get calibrator params from freeze contract
        freeze_split = freeze_contract["per_split"].get(sk)
        if not freeze_split:
            raise SystemExit(f"FREEZE_SPLIT_MISSING: {sk}")
        calibrators = {head: freeze_split[head] for head in HEADS}

        payloads.append({
            "split": sk,
            "runtime_episodes": runtime_episodes,
            "evaluation_episodes": evaluation_episodes,
            "calibrators": calibrators,
            "checkpoint_sha256": checkpoint_sha,
            "scheduler_source_sha256": scheduler_source_sha,
            "student_source_commit": pred_rows[0].get("checkpoint_source_commit", ""),
            "feature_order_sha256": pred_rows[0].get("feature_order_sha256", ""),
            "fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest) if hasattr(args, 'calibrator_fit_manifest') else "",
            "policy_manifest_sha256": sha256_file(args.policy_selection_manifest),
            "structure_path": structure_path,
        })
        pol_manifest_shas[sk] = sha256_file(args.policy_selection_manifest)

    # Grid search
    best = None
    all_results: list[dict[str, Any]] = []
    for g, m, r in itertools.product(grasp_grid, manipulation_grid, release_grid):
        result = evaluate_candidate(payloads, {"grasp": g, "manipulation": m, "release": r}, structure)
        result["constraint_pass"] = (
            result["all_split_false_start_defined"]
            and result["worst_split_negative_false_start_rate"] is not None
            and result["worst_split_negative_false_start_rate"] <= args.max_false_start
        )
        all_results.append(result)
        if result["constraint_pass"] and (best is None or selection_key(result) > selection_key(best)):
            best = result

    # Build freeze contract
    if best is None:
        freeze_scheduler = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "HOLD_NO_FEASIBLE_THRESHOLD",
            "grid_combinations": len(all_results),
            "max_false_start": args.max_false_start,
            "attack_authorized": False,
            "heldout_l3_authorized": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    else:
        freeze_scheduler = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "COMPLETE",
            "selected_thresholds": best["thresholds"],
            "selection_rule": {
                "constraint": f"worst-split false-start <= {args.max_false_start}",
                "objective": ["max recall", "max precision", "min timing offset",
                              "higher grasp", "higher manipulation", "lower release"],
                "grid": {"grasp": list(grasp_grid), "manipulation": list(manipulation_grid),
                         "release": list(release_grid)},
            },
            "selected_metrics": best["aggregate"],
            "worst_split_false_start": best["worst_split_negative_false_start_rate"],
            "per_split": best["per_split"],
            "bindings": {
                "calibrator_freeze_sha256": sha256_file(freeze_path),
                "calibrator_freeze_validation_sha256": sha256_file(args.calibrator_freeze_validation_receipt),
                "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
                "policy_prediction_bundle_sha256": sha256_file(p_pred_root / "SHA256SUMS"),
                "policy_teacher_bundle_sha256": sha256_file(p_teacher_root / "SHA256SUMS"),
                "runtime_adapter_source_sha256": adapter_source_sha,
                "scheduler_source_sha256": scheduler_source_sha,
                "structural_config_sha256": structural_sha,
                "freeze_code_sha256": SELF_SHA,
            },
            "attack_authorized": False,
            "heldout_l3_authorized": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Build search ledger
    ledger_lines: list[str] = []
    for candidate in all_results:
        ledger_lines.append(json.dumps({
            "thresholds": candidate["thresholds"],
            "constraint_pass": candidate["constraint_pass"],
            "worst_false_start": candidate.get("worst_split_negative_false_start_rate"),
            "recall": candidate["aggregate"].get("valid_opportunity_recall"),
            "precision": candidate["aggregate"].get("all_emit_precision"),
        }) + "\n")

    # Build policy metrics
    policy_metrics = {
        "schema": "FACTORIZED_POLICY_SELECTION_METRICS_V1",
        "best_thresholds": best["thresholds"] if best else None,
        "per_split": best["per_split"] if best else {},
        "aggregate": best["aggregate"] if best else {},
    }

    files = {
        "FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(freeze_scheduler, indent=2, sort_keys=True) + "\n",
        "FACTORIZED_POLICY_SELECTION_METRICS_V1.json": json.dumps(policy_metrics, indent=2, sort_keys=True) + "\n",
        "FACTORIZED_THRESHOLD_SEARCH_LEDGER_V1.jsonl": "".join(ledger_lines),
    }
    _seal_output(out_root, files)
    print(f"Scheduler Freeze: {out_root} status={freeze_scheduler['status']}")
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
