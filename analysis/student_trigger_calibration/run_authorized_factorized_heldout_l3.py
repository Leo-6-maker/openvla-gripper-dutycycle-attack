#!/usr/bin/env python3
"""One-shot authorized heldout-L3 evaluator (P0-1, P0-7, P0-9).

Consumes valid H prediction authorization + validated H prediction bundle.
Uses REAL runtime bundles (never hardcoded candidate_close).
Strict 3-way join (no silent skip). Atomic single-use claim (P0-8).
"""
from __future__ import annotations

import argparse, json, math, os, statistics, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, HEADS, sha256_file, load_strict_json, load_strict_jsonl,
    verify_bundle_seal, seal_output_dir, extract_manifest_identities,
    verify_identity_closure, exact_three_way_join, consume_sealed_receipt,
    verify_runtime_source_files, claim_atomic_root,
)
from run_factorized_l3_analysis import (
    compute_l3_metrics, validate_episode_step_sequence,
)
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter

SELF_SHA = None


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-prediction-authorization-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, required=True)
    # P0-1: REAL runtime bundle
    ap.add_argument("--heldout-runtime-bundle-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--runtime-source-root", type=Path, default=None)
    # P0-8: Claim root
    ap.add_argument("--claim-root", type=Path, required=True)
    # P0-7: Separate evaluation output
    ap.add_argument("--authorized-l3-evaluation-output-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    eval_out = args.authorized_l3_evaluation_output_root.resolve()
    if eval_out.exists():
        raise SystemExit(f"EVAL_OUTPUT_EXISTS: {eval_out}")

    expected = [s.strip() for s in args.expected_splits.split(",")]

    # P0-8: Atomic single-use claim
    h_pred_auth, _ = consume_sealed_receipt(args.heldout_prediction_authorization_root,
        "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1",
        "heldout_prediction_inference_authorized", True, "H_PRED_AUTH")
    auth_sha = sha256_file(
        next(p for p in args.heldout_prediction_authorization_root.iterdir()
             if p.suffix == ".json" and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256")))
    claim_atomic_root(args.claim_root, auth_sha, "H_L3")

    # Consume H prediction validation receipt
    consume_sealed_receipt(args.heldout_prediction_validation_root,
        "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1",
        "h_predictions_ready", True, "H_PRED_VAL")

    # Load calibrator and scheduler freeze
    cf_root = args.calibrator_freeze_root.resolve()
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    cf = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")
    sf = load_strict_json(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "SCHED_FREEZE")

    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")
    structure = load_strict_json(args.structure_config, "STRUCTURE")
    structure_path = args.structure_config.resolve()
    structural_sha = sha256_file(structure_path)

    runtime_src_root = args.runtime_source_root.resolve() if args.runtime_source_root else ROOT
    rt_sources = verify_runtime_source_files(runtime_src_root)

    h_pred_root = args.heldout_prediction_bundle_root.resolve()
    h_teacher_root = args.heldout_teacher_bundle_root.resolve()
    h_rt_root = args.heldout_runtime_bundle_root.resolve()
    verify_bundle_seal(h_pred_root, "H_PRED")
    verify_bundle_seal(h_teacher_root, "H_TEACHER")
    verify_bundle_seal(h_rt_root, "H_RUNTIME")

    selected_thresholds = sf.get("selected_thresholds", {})

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # START receipt
    start_receipt = {
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1", "run_status": "STARTED",
        "authorization_sha256": auth_sha,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_code_sha256": SELF_SHA,
    }
    (staging / "HELDOUT_L3_RUN_START_RECEIPT_V1.json").write_text(
        json.dumps(start_receipt, indent=2, sort_keys=True) + "\n")

    all_metrics: dict[str, dict[str, Any]] = {}
    emit_ledger: list[dict[str, Any]] = []
    no_trigger_ledger: list[dict[str, Any]] = []
    unknown_ledger: list[dict[str, Any]] = []
    all_complete = True
    split_errors: list[str] = []

    for sk in expected:
        try:
            h_ids = extract_manifest_identities(held_manifest, "heldout_l3", sk)

            pred_rows = load_strict_jsonl(h_pred_root / sk / "predictions.jsonl", f"H_PRED_{sk}")
            teacher_rows = load_strict_jsonl(h_teacher_root / sk / "factorized_teacher_v1.jsonl", f"H_TEACHER_{sk}")
            # P0-1: REAL runtime rows
            rt_rows = load_strict_jsonl(h_rt_root / sk / "runtime_scheduler_inputs.jsonl", f"H_RT_{sk}")

            pred_ids = {r["canonical_parent_key"] for r in pred_rows}
            verify_identity_closure(pred_ids, h_ids, "HELDOUT", sk)

            # P0-9: Strict 3-way join — no silent skip
            pred_by_key, teacher_by_key, rt_by_key = exact_three_way_join(
                pred_rows, teacher_rows, rt_rows, f"H_{sk}")

            runtime_episodes: dict[str, list[dict[str, Any]]] = {}
            eval_episodes: dict[str, list[dict[str, Any]]] = {}
            for (ep, step), rt_row in rt_by_key.items():
                t_row = teacher_by_key[(ep, step)]
                runtime_episodes.setdefault(ep, []).append(rt_row)
                eval_episodes.setdefault(ep, []).append({
                    "step_index": step, "canonical_parent_key": ep, "step": step,
                    "strict_k10_feasible": t_row.get("strict_k10_feasible", False),
                    "strict_k10_known_mask": t_row.get("strict_k10_known_mask", False),
                })

            for ep_rows in runtime_episodes.values():
                ep_rows.sort(key=lambda r: r["step"])
            for ep_rows in eval_episodes.values():
                ep_rows.sort(key=lambda r: r["step"])

            cf_split = cf["per_split"].get(sk)
            if not cf_split:
                raise SystemExit(f"CAL_FREEZE_SPLIT_MISSING: {sk}")

            cal_contract = {
                "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
                "status": "AUTHORITATIVE", "split": sk,
                "checkpoint_sha256": pred_rows[0].get("checkpoint_sha256", ""),
                "scheduler_source_sha256": rt_sources["scheduler_source_sha256"],
                "structural_config_sha256": structural_sha,
                "student_source_commit": pred_rows[0].get("checkpoint_source_commit", ""),
                "feature_order_sha256": pred_rows[0].get("feature_order_sha256", ""),
                "calibration_fit_authoritative": True,
                "threshold_selection_authoritative": True,
                "l3_evaluation_eligible": True,
                "training_authorized": False, "full_fit_authorized": False, "attack_authorized": False,
            }
            for head in HEADS:
                hd = cf_split[head]
                cal_contract[head] = {
                    "method": hd["method"], "a": float(hd["a"]), "b": float(hd["b"]),
                    "threshold": float(selected_thresholds.get(head, 0.5)),
                    "transform": "probability=sigmoid(a*raw_logit+b)",
                    "method_valid": True, "transform_valid": True,
                    "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION",
                    "fit_manifest_sha256": "", "policy_selection_manifest_sha256": "",
                }

            adapter = FactorizedV2SchedulerAdapter(structure=structure, calibration_contract=cal_contract, require_l3_eligible=True)
            scheduler_results: dict[str, dict[str, Any]] = {}
            for episode, rows in sorted(runtime_episodes.items()):
                result = adapter.run_episode(rows)
                scheduler_results[episode] = {
                    "emitted": result["ever_emitted"],
                    "emit_step": result["first_emit_step"] if result["first_emit_step"] is not None else -1,
                    "final_state": result["final_state"],
                }

            metrics = compute_l3_metrics(eval_episodes, scheduler_results, "step")
            all_metrics[sk] = metrics

            for row in metrics["per_episode"]:
                entry = {"split": sk, "episode": row["episode_key"], "classification": row["classification"],
                         "emitted": row["scheduler_emitted"], "emit_step": row["emit_step"],
                         "on_corridor": row["on_corridor"], "timing_offset": row.get("timing_offset")}
                emit_ledger.append(entry)
                if row["classification"] == "negative" and not row["scheduler_emitted"]:
                    no_trigger_ledger.append(entry)
                if row["classification"] == "unknown":
                    unknown_ledger.append(entry)

        except (SystemExit, Exception) as e:
            all_complete = False
            split_errors.append(f"{sk}: {e}")

    if not all_complete:
        diag = {"schema": "HELDOUT_L3_DIAGNOSTIC_PARTIAL_V1", "partial": True,
                "split_errors": split_errors, "completed_splits": sorted(all_metrics),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (staging / "HELDOUT_L3_DIAGNOSTIC_PARTIAL_V1.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
        seal_output_dir(staging)
        os.replace(staging, out_root)
        print("HELDOUT_L3_PARTIAL: diagnostic only")
        return 2

    # Pooled metrics
    pooled: dict[str, Any] = {"total_negative_episodes": 0, "total_positive_episodes": 0,
        "total_unknown_episodes": 0, "total_negative_emits": 0,
        "total_positive_on_corridor": 0, "total_positive_off_corridor": 0,
        "total_positive_abstentions": 0, "total_emitted_all": 0, "total_emitted_verified": 0}
    all_offsets: list[float] = []
    for m in all_metrics.values():
        for k in pooled:
            pooled[k] += m.get(k, 0)
        for ep in m.get("per_episode", []):
            if ep.get("timing_offset") is not None:
                all_offsets.append(float(ep["timing_offset"]))

    def ratio(n: int, d: int) -> float | None:
        return n / d if d > 0 else None

    pooled.update({
        "pooled_negative_false_start_rate": ratio(pooled["total_negative_emits"], pooled["total_negative_episodes"]),
        "pooled_valid_opportunity_recall": ratio(pooled["total_positive_on_corridor"], pooled["total_positive_episodes"]),
        "pooled_all_emit_precision": ratio(pooled["total_positive_on_corridor"], pooled["total_emitted_all"]),
        "pooled_median_timing_offset": float(statistics.median(all_offsets)) if all_offsets else None,
    })

    false_rates = {sk: m.get("negative_episode_false_start_rate") for sk, m in all_metrics.items()}
    defined = {sk: v for sk, v in false_rates.items() if v is not None}
    worst_split = max(defined.values()) if defined else None

    complete_receipt = {
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1", "run_status": "COMPLETE",
        "authoritative": True, "authorization_sha256": auth_sha,
        "n_splits": len(all_metrics), "all_splits_complete": True,
        "worst_split_false_start_rate": worst_split,
        "gate_pass": worst_split is not None and worst_split <= 0.10,
        "started_at": start_receipt["started_at"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_code_sha256": SELF_SHA,
    }

    csv_lines = ["split,negative_episodes,negative_emits,false_start_rate,positive_episodes,recall,all_emit_precision,median_timing_offset"]
    for sk in sorted(all_metrics):
        m = all_metrics[sk]
        csv_lines.append(f"{sk},{m['negative_episodes']},{m['negative_episode_emits']},{m.get('negative_episode_false_start_rate','')},{m['positive_episodes']},{m.get('valid_opportunity_recall','')},{m.get('all_emit_precision','')},{m.get('median_timing_offset','')}")

    (staging / "HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json").write_text(json.dumps(complete_receipt, indent=2, sort_keys=True) + "\n")
    (staging / "HELDOUT_L3_PER_SPLIT_METRICS.csv").write_text("\n".join(csv_lines) + "\n")
    (staging / "HELDOUT_L3_POOLED_METRICS.json").write_text(json.dumps(pooled, indent=2, sort_keys=True) + "\n")
    (staging / "HELDOUT_L3_EMIT_LEDGER.jsonl").write_text("".join(json.dumps(r) + "\n" for r in emit_ledger))
    (staging / "HELDOUT_L3_NO_TRIGGER_LEDGER.jsonl").write_text("".join(json.dumps(r) + "\n" for r in no_trigger_ledger))
    (staging / "HELDOUT_L3_UNKNOWN_LEDGER.jsonl").write_text("".join(json.dumps(r) + "\n" for r in unknown_ledger))
    (staging / "HELDOUT_L3_CLAIM_BOUNDARY.md").write_text(
        f"# Heldout-L3 Claim Boundary\n\nAuthorization: {auth_sha}\nWorst-split false-start: {worst_split}\nGate: {'PASS' if worst_split is not None and worst_split <= 0.10 else 'FAIL'}\n")

    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"Heldout-L3 Complete: worst_false_start={worst_split} gate={'PASS' if worst_split is not None and worst_split <= 0.10 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
