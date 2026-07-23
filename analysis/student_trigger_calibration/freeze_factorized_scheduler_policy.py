#!/usr/bin/env python3
"""Freeze joint scheduler thresholds (P0-1, P0-2, P0-3, P0-6).

Consumes REAL runtime bundles (--policy-runtime-bundle-root), not hardcoded values.
Bans zero-recall policy freeze. Uses actual calibrator fit manifest SHA.
Consumes sealed receipt roots.
"""
from __future__ import annotations

import argparse, itertools, json, math, os, statistics, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, HEADS, sha256_file, load_strict_json, load_strict_jsonl,
    verify_bundle_seal, seal_output_dir, extract_manifest_identities,
    verify_identity_closure, verify_step_closure, exact_three_way_join,
    consume_sealed_receipt, verify_runtime_source_files,
)
from run_factorized_l3_analysis import (
    compute_l3_metrics, validate_episode_step_sequence, classify_episode, is_valid_start,
)
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter

SELF_SHA = None


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
            "training_authorized": False, "full_fit_authorized": False, "attack_authorized": False,
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

    recall = ratio(totals["positive_on_corridor_emits"], totals["positive_episodes"])
    aggregate = {**totals,
        "negative_episode_false_start_rate": ratio(totals["negative_episode_emits"], totals["negative_episodes"]),
        "valid_opportunity_recall": recall,
        "all_emit_precision": ratio(totals["positive_on_corridor_emits"], totals["total_emitted_all"]),
        "verified_emit_precision": ratio(totals["positive_on_corridor_emits"], totals["total_emitted_verified"]),
        "median_timing_offset": float(statistics.median(all_offsets)) if all_offsets else None,
    }
    return {"thresholds": dict(thresholds), "all_split_false_start_defined": len(defined) == len(per_split),
            "worst_split_negative_false_start_rate": worst, "per_split": per_split, "aggregate": aggregate,
            "recall": recall}


def selection_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    a = result["aggregate"]
    t = result["thresholds"]
    return (
        -1.0 if a["valid_opportunity_recall"] is None else float(a["valid_opportunity_recall"]),
        -1.0 if a["all_emit_precision"] is None else float(a["all_emit_precision"]),
        float("-inf") if a["median_timing_offset"] is None else -float(a["median_timing_offset"]),
        float(t["grasp"]), float(t["manipulation"]), -float(t["release"]),
    )


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--policy-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--policy-teacher-bundle-root", type=Path, required=True)
    # P0-1: REAL runtime bundle
    ap.add_argument("--policy-runtime-bundle-root", type=Path, required=True)
    # P0-6: sealed receipt roots
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--phase-b-validation-root", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-root", type=Path, required=True)
    # P0-3: actual calibrator fit manifest
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--runtime-source-root", type=Path, default=None)
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

    # P0-6: Consume sealed receipts
    phase_b, _ = consume_sealed_receipt(args.phase_b_validation_root,
        "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "cp_inference_authorized", True, "PHASE_B")
    cp_val, _ = consume_sealed_receipt(args.cp_prediction_validation_root,
        "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1", "cp_predictions_ready", True, "CP_VAL")
    cf_val, _ = consume_sealed_receipt(args.calibrator_freeze_validation_root,
        "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1", "status", "PASS", "CAL_FREEZE_VAL")

    # Load calibrator freeze
    cf_root = args.calibrator_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    freeze_contract = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")

    # P0-3: Actual calibrator fit manifest SHA
    fit_manifest_sha = sha256_file(args.calibrator_fit_manifest)

    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    p_pred_root = args.policy_prediction_bundle_root.resolve()
    p_teacher_root = args.policy_teacher_bundle_root.resolve()
    p_rt_root = args.policy_runtime_bundle_root.resolve()
    verify_bundle_seal(p_pred_root, "P_PRED")
    verify_bundle_seal(p_teacher_root, "P_TEACHER")
    verify_bundle_seal(p_rt_root, "P_RUNTIME")

    grasp_grid = _parse_grid(args.grasp_grid, "GRASP")
    manipulation_grid = _parse_grid(args.manipulation_grid, "MANIPULATION")
    release_grid = _parse_grid(args.release_grid, "RELEASE")

    structure_path = args.structure_config.resolve()
    structure = load_strict_json(structure_path, "STRUCTURE")
    structural_sha = sha256_file(structure_path)

    # P0-5: Actual runtime source SHAs
    runtime_src_root = args.runtime_source_root.resolve() if args.runtime_source_root else ROOT
    rt_sources = verify_runtime_source_files(runtime_src_root)

    expected = [s.strip() for s in args.expected_splits.split(",")]

    payloads: list[dict[str, Any]] = []
    for sk in expected:
        p_ids = extract_manifest_identities(pol_manifest, "policy_selection", sk)

        pred_rows = load_strict_jsonl(p_pred_root / sk / "predictions.jsonl", f"P_PRED_{sk}")
        teacher_rows = load_strict_jsonl(p_teacher_root / sk / "factorized_teacher_v1.jsonl", f"P_TEACHER_{sk}")
        # P0-1: Load REAL runtime rows
        runtime_rows = load_strict_jsonl(p_rt_root / sk / "runtime_scheduler_inputs.jsonl", f"P_RUNTIME_{sk}")

        pred_ids = {r["canonical_parent_key"] for r in pred_rows}
        verify_identity_closure(pred_ids, p_ids, "POLICY", sk)

        # P0-1: Exact 3-way join (pred, teacher, runtime) — NO hardcoded values
        pred_by_key, teacher_by_key, rt_by_key = exact_three_way_join(
            pred_rows, teacher_rows, runtime_rows, f"P_{sk}")

        # Build runtime episodes from REAL runtime rows
        runtime_episodes: dict[str, list[dict[str, Any]]] = {}
        evaluation_episodes: dict[str, list[dict[str, Any]]] = {}
        for (ep, step), rt_row in rt_by_key.items():
            p_row = pred_by_key[(ep, step)]
            t_row = teacher_by_key[(ep, step)]
            runtime_episodes.setdefault(ep, []).append(rt_row)
            evaluation_episodes.setdefault(ep, []).append({
                "step_index": step, "canonical_parent_key": ep, "step": step,
                "strict_k10_feasible": t_row.get("strict_k10_feasible", False),
                "strict_k10_known_mask": t_row.get("strict_k10_known_mask", False),
            })

        for ep, ep_rows in runtime_episodes.items():
            ep_rows.sort(key=lambda r: r["step"])
        for ep, ep_rows in evaluation_episodes.items():
            ep_rows.sort(key=lambda r: r["step"])

        checkpoint_sha = pred_rows[0].get("checkpoint_sha256", "")
        freeze_split = freeze_contract["per_split"].get(sk)
        if not freeze_split:
            raise SystemExit(f"FREEZE_SPLIT_MISSING: {sk}")
        calibrators = {head: freeze_split[head] for head in HEADS}

        payloads.append({
            "split": sk, "runtime_episodes": runtime_episodes,
            "evaluation_episodes": evaluation_episodes, "calibrators": calibrators,
            "checkpoint_sha256": checkpoint_sha,
            "scheduler_source_sha256": rt_sources["scheduler_source_sha256"],
            "student_source_commit": pred_rows[0].get("checkpoint_source_commit", ""),
            "feature_order_sha256": pred_rows[0].get("feature_order_sha256", ""),
            "fit_manifest_sha256": fit_manifest_sha,
            "policy_manifest_sha256": sha256_file(args.policy_selection_manifest),
            "structure_path": structure_path,
        })

    # Grid search
    best = None
    all_results: list[dict[str, Any]] = []
    for g, m, r in itertools.product(grasp_grid, manipulation_grid, release_grid):
        result = evaluate_candidate(payloads, {"grasp": g, "manipulation": m, "release": r}, structure)
        # P0-2: Additional constraints beyond false-start
        recall_val = result["aggregate"].get("valid_opportunity_recall")
        positive_eps = result["aggregate"].get("positive_episodes", 0)
        on_corridor = result["aggregate"].get("positive_on_corridor_emits", 0)
        result["constraint_pass"] = (
            result["all_split_false_start_defined"]
            and result["worst_split_negative_false_start_rate"] is not None
            and result["worst_split_negative_false_start_rate"] <= args.max_false_start
            and positive_eps > 0                                   # P0-2: positive denominator > 0
            and recall_val is not None and recall_val > 0.0        # P0-2: recall > 0
            and on_corridor > 0                                     # P0-2: at least one verified on-corridor emit
        )
        all_results.append(result)
        if result["constraint_pass"] and (best is None or selection_key(result) > selection_key(best)):
            best = result

    if best is None:
        best_recall = max(
            (r["aggregate"].get("valid_opportunity_recall") or 0.0 for r in all_results), default=0.0
        )
        freeze_scheduler = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "HOLD_NO_FEASIBLE_THRESHOLD",
            "grid_combinations": len(all_results),
            "max_false_start": args.max_false_start,
            "best_observed_recall": best_recall,
            "attack_authorized": False, "heldout_l3_authorized": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    else:
        freeze_scheduler = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "COMPLETE",
            "selected_thresholds": best["thresholds"],
            "selection_rule": {
                "constraint": f"worst-split false-start <= {args.max_false_start} AND recall > 0 AND positive_eps > 0",
                "objective": ["max recall", "max precision", "min timing offset",
                              "higher grasp", "higher manipulation", "lower release"],
                "grid": {"grasp": list(grasp_grid), "manipulation": list(manipulation_grid), "release": list(release_grid)},
            },
            "selected_metrics": best["aggregate"],
            "worst_split_false_start": best["worst_split_negative_false_start_rate"],
            "per_split": best["per_split"],
            "bindings": {
                "calibrator_freeze_sha256": sha256_file(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"),
                "calibrator_fit_manifest_sha256": fit_manifest_sha,
                "phase_b_validation_seal_sha256": sha256_file(args.phase_b_validation_root / "SHA256SUMS"),
                "cp_prediction_validation_seal_sha256": sha256_file(args.cp_prediction_validation_root / "SHA256SUMS"),
                "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
                "policy_prediction_bundle_sha256": sha256_file(p_pred_root / "SHA256SUMS"),
                "policy_teacher_bundle_sha256": sha256_file(p_teacher_root / "SHA256SUMS"),
                "policy_runtime_bundle_sha256": sha256_file(p_rt_root / "SHA256SUMS"),
                "runtime_adapter_source_sha256": rt_sources["runtime_adapter_source_sha256"],
                "scheduler_source_sha256": rt_sources["scheduler_source_sha256"],
                "structural_config_sha256": structural_sha,
                "freeze_code_sha256": SELF_SHA,
            },
            "attack_authorized": False, "heldout_l3_authorized": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    ledger_lines = [
        json.dumps({"thresholds": c["thresholds"], "constraint_pass": c["constraint_pass"],
                     "worst_false_start": c.get("worst_split_negative_false_start_rate"),
                     "recall": c["aggregate"].get("valid_opportunity_recall"),
                     "precision": c["aggregate"].get("all_emit_precision")}) + "\n"
        for c in all_results
    ]
    policy_metrics = {
        "schema": "FACTORIZED_POLICY_SELECTION_METRICS_V1",
        "best_thresholds": best["thresholds"] if best else None,
        "per_split": best["per_split"] if best else {},
        "aggregate": best["aggregate"] if best else {},
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in [
        ("FACTORIZED_SCHEDULER_FREEZE_V1.json", json.dumps(freeze_scheduler, indent=2, sort_keys=True) + "\n"),
        ("FACTORIZED_POLICY_SELECTION_METRICS_V1.json", json.dumps(policy_metrics, indent=2, sort_keys=True) + "\n"),
        ("FACTORIZED_THRESHOLD_SEARCH_LEDGER_V1.jsonl", "".join(ledger_lines)),
    ]:
        (staging / name).write_text(content, encoding="utf-8")
    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"Scheduler Freeze: {out_root} status={freeze_scheduler['status']}")
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
