#!/usr/bin/env python3
"""One-shot authorized heldout-L3 evaluator (A1, A2, A3, A4, A7).

A1: Consumes L3 evaluation authorization (FACTORIZED_HELDOUT_L3_EVALUATION_AUTHORIZATION_RECEIPT_V1),
    not just H prediction authorization.
A2: No default threshold — missing/malformed threshold → non-zero exit.
A3: fit_manifest_sha256 / policy_selection_manifest_sha256 from freeze contracts, never "".
A4: Cross-receipt binding: output root must match authorization exactly.
A7: build_adapter_row(pred_row, rt_row) — prediction provides logits, runtime provides close/known/valid.
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
    verify_runtime_source_files, claim_atomic_root, verify_receipt_binding,
    is_64char_hex,
)
from run_factorized_l3_analysis import compute_l3_metrics
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter

SELF_SHA = None


# ── A7: Frozen runtime/prediction field merge ──────────────────────────────

PREDICTION_LOGIT_FIELDS = frozenset({
    "grasp_logit", "manipulation_logit", "release_logit",
    "grasp_probability", "manipulation_probability", "release_probability",
})
PREDICTION_BINDING_FIELDS = frozenset({
    "checkpoint_sha256", "checkpoint_source_commit",
    "feature_order_sha256", "normalization_sha256", "runtime_source_sha256",
    "source_artifact_recursive_sha256", "source_episode_step_count",
})
RUNTIME_ONLY_FIELDS = frozenset({
    "candidate_close", "action_known", "student_valid", "route_supported",
    "episode", "step", "split",
    "scheduler_source_sha256", "structural_config_sha256",
})


def build_adapter_row(pred_row: dict[str, Any], rt_row: dict[str, Any]) -> dict[str, Any]:
    """A7: Merge prediction logits + bindings with runtime close/known/valid.

    Runtime must NOT contain Student score fields (grasp_logit, etc.).
    Prediction provides logits and binding; runtime provides close/known/valid.
    """
    # Reject duplicate score fields in runtime
    forbidden = PREDICTION_LOGIT_FIELDS & set(rt_row)
    if forbidden:
        raise SystemExit(f"RUNTIME_FORBIDDEN_SCORE_FIELDS: {sorted(forbidden)}")

    row = {}
    # From prediction: logits, probabilities, bindings
    for fld in PREDICTION_LOGIT_FIELDS | PREDICTION_BINDING_FIELDS:
        if fld in pred_row:
            row[fld] = pred_row[fld]

    # From runtime: close, known, valid, supported, structural
    for fld in RUNTIME_ONLY_FIELDS:
        if fld in rt_row:
            row[fld] = rt_row[fld]

    # Validate critical fields
    if "candidate_close" not in row or not isinstance(row["candidate_close"], bool):
        raise SystemExit("ADAPTER_ROW: candidate_close must be strict bool from runtime")
    if "grasp_logit" not in row:
        raise SystemExit("ADAPTER_ROW: grasp_logit missing from prediction")

    return row


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    # A1: L3 evaluation authorization (NOT just H prediction auth)
    ap.add_argument("--heldout-l3-evaluation-authorization-root", type=Path, required=True)
    # H prediction auth + validation (proves predictions were authorized)
    ap.add_argument("--heldout-prediction-authorization-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-validation-root", type=Path, required=True)
    # Bundles
    ap.add_argument("--heldout-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-runtime-bundle-root", type=Path, required=True)
    # Freeze contracts
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    # Manifests
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    # Config
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--runtime-source-root", type=Path, default=None)
    # A1: Output root from authorization only — no separate --authorized-l3-evaluation-output-root
    ap.add_argument("--output-root", type=Path, required=True)
    # A8: Claim root for single-use
    ap.add_argument("--claim-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)
    if len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS:
        raise SystemExit("SPLIT_ENFORCEMENT: requires exactly 12 unique frozen splits")

    # ═══ A1: Consume L3 evaluation authorization ═══════════════════════
    l3_auth, l3_auth_seal = consume_sealed_receipt(
        args.heldout_l3_evaluation_authorization_root,
        "FACTORIZED_HELDOUT_L3_EVALUATION_AUTHORIZATION_RECEIPT_V1",
        "heldout_l3_evaluation_authorized", True, "L3_AUTH",
    )
    # A1: Output root MUST match authorization
    authorized_eval_root = l3_auth.get("authorized_l3_evaluation_output_root", "")
    if not authorized_eval_root:
        raise SystemExit("L3_AUTH: authorized_l3_evaluation_output_root missing")
    if Path(authorized_eval_root).resolve() != out_root:
        raise SystemExit(
            f"L3_AUTH_OUTPUT_MISMATCH: authorized={authorized_eval_root} provided={out_root}"
        )

    # A4: Bind L3 auth to manifests
    verify_receipt_binding(l3_auth, "authorized_h_manifest_sha256",
                           sha256_file(args.heldout_l3_manifest), "L3_AUTH_H_MANIFEST")
    verify_receipt_binding(l3_auth, "authorized_calibrator_freeze_sha256",
                           sha256_file(args.calibrator_freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"),
                           "L3_AUTH_CAL_FREEZE")
    verify_receipt_binding(l3_auth, "authorized_scheduler_freeze_sha256",
                           sha256_file(args.scheduler_freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"),
                           "L3_AUTH_SCHED_FREEZE")

    # A8: Single-use claim keyed by L3 evaluation authorization SHA
    l3_auth_json = next(p for p in args.heldout_l3_evaluation_authorization_root.iterdir()
                        if p.suffix == ".json" and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    l3_auth_sha = sha256_file(l3_auth_json)
    claim_atomic_root(args.claim_root, l3_auth_sha, "H_L3_EVAL")

    # Consume H prediction auth + validation (proves predictions exist)
    consume_sealed_receipt(args.heldout_prediction_authorization_root,
        "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1",
        "heldout_prediction_inference_authorized", True, "H_PRED_AUTH")
    consume_sealed_receipt(args.heldout_prediction_validation_root,
        "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1",
        "h_predictions_ready", True, "H_PRED_VAL")

    # Load freeze contracts
    cf_root = args.calibrator_freeze_root.resolve()
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    cf = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")
    sf = load_strict_json(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "SCHED_FREEZE")

    # A3: Extract fit_manifest_sha256 from freeze contracts, not ""
    fit_manifest_sha = cf.get("freeze_bindings", {}).get("calibrator_fit_manifest_sha256", "")
    policy_manifest_sha = sf.get("bindings", {}).get("policy_selection_manifest_sha256", "")
    if not is_64char_hex(fit_manifest_sha):
        raise SystemExit(f"CAL_FREEZE_MISSING_FIT_MANIFEST_SHA: {fit_manifest_sha[:40]}")
    if not is_64char_hex(policy_manifest_sha):
        raise SystemExit(f"SCHED_FREEZE_MISSING_POLICY_MANIFEST_SHA: {policy_manifest_sha[:40]}")

    # A4: Verify manifests match freeze contract SHAs
    verify_receipt_binding(
        {"fit_manifest_sha": fit_manifest_sha}, "fit_manifest_sha",
        sha256_file(args.calibrator_fit_manifest), "FIT_MANIFEST")
    verify_receipt_binding(
        {"policy_manifest_sha": policy_manifest_sha}, "policy_manifest_sha",
        sha256_file(args.policy_selection_manifest), "POL_MANIFEST")

    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")
    structure = load_strict_json(args.structure_config, "STRUCTURE")
    structure_path = args.structure_config.resolve()

    runtime_src_root = args.runtime_source_root.resolve() if args.runtime_source_root else ROOT
    rt_sources = verify_runtime_source_files(runtime_src_root)

    h_pred_root = args.heldout_prediction_bundle_root.resolve()
    h_teacher_root = args.heldout_teacher_bundle_root.resolve()
    h_rt_root = args.heldout_runtime_bundle_root.resolve()
    verify_bundle_seal(h_pred_root, "H_PRED")
    verify_bundle_seal(h_teacher_root, "H_TEACHER")
    verify_bundle_seal(h_rt_root, "H_RUNTIME")

    # A2: Strict threshold validation — no default 0.5
    selected_thresholds = sf.get("selected_thresholds", {})
    for head in HEADS:
        t_val = selected_thresholds.get(head)
        if t_val is None:
            raise SystemExit(f"THRESHOLD_MISSING_FROM_SCHED_FREEZE: {head}")
        if isinstance(t_val, bool) or not isinstance(t_val, (int, float)):
            raise SystemExit(f"THRESHOLD_TYPE_INVALID: {head} value={t_val!r}")
        f_val = float(t_val)
        if not math.isfinite(f_val) or not 0.0 <= f_val <= 1.0:
            raise SystemExit(f"THRESHOLD_RANGE_INVALID: {head} value={f_val}")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # START receipt
    start_receipt = {
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1", "run_status": "STARTED",
        "l3_evaluation_authorization_sha256": l3_auth_sha,
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
            rt_rows = load_strict_jsonl(h_rt_root / sk / "runtime_scheduler_inputs.jsonl", f"H_RT_{sk}")

            pred_ids = {r["canonical_parent_key"] for r in pred_rows}
            verify_identity_closure(pred_ids, h_ids, "HELDOUT", sk)

            pred_by_key, teacher_by_key, rt_by_key = exact_three_way_join(
                pred_rows, teacher_rows, rt_rows, f"H_{sk}")

            # A7: Build adapter rows from prediction + runtime merge
            runtime_episodes: dict[str, list[dict[str, Any]]] = {}
            eval_episodes: dict[str, list[dict[str, Any]]] = {}
            for (ep, step), rt_row in rt_by_key.items():
                p_row = pred_by_key[(ep, step)]
                t_row = teacher_by_key[(ep, step)]
                adapter_row = build_adapter_row(p_row, rt_row)
                runtime_episodes.setdefault(ep, []).append(adapter_row)
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

            # A2+A3: No default threshold, no empty manifest SHA
            cal_contract = {
                "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
                "status": "AUTHORITATIVE", "split": sk,
                "checkpoint_sha256": pred_rows[0].get("checkpoint_sha256", ""),
                "scheduler_source_sha256": rt_sources["scheduler_source_sha256"],
                "structural_config_sha256": sha256_file(structure_path),
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
                    "threshold": float(selected_thresholds[head]),  # A2: must exist
                    "transform": "probability=sigmoid(a*raw_logit+b)",
                    "method_valid": True, "transform_valid": True,
                    "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION",
                    "fit_manifest_sha256": fit_manifest_sha,          # A3: from freeze
                    "policy_selection_manifest_sha256": policy_manifest_sha,  # A3: from freeze
                }

            adapter = FactorizedV2SchedulerAdapter(
                structure=structure, calibration_contract=cal_contract, require_l3_eligible=True)
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
                entry = {"split": sk, "episode": row["episode_key"],
                         "classification": row["classification"],
                         "emitted": row["scheduler_emitted"], "emit_step": row["emit_step"],
                         "on_corridor": row["on_corridor"],
                         "timing_offset": row.get("timing_offset")}
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
                "split_errors": split_errors, "completed_splits": sorted(all_metrics)}
        (staging / "HELDOUT_L3_DIAGNOSTIC_PARTIAL_V1.json").write_text(
            json.dumps(diag, indent=2, sort_keys=True) + "\n")
        seal_output_dir(staging)
        os.replace(staging, out_root)
        return 2

    # Pooled metrics
    pooled: dict[str, Any] = {"total_negative_episodes": 0, "total_positive_episodes": 0,
        "total_unknown_episodes": 0, "total_negative_emits": 0,
        "total_positive_on_corridor": 0, "total_positive_off_corridor": 0,
        "total_positive_abstentions": 0, "total_emitted_all": 0, "total_emitted_verified": 0}
    all_offsets: list[float] = []
    for m in all_metrics.values():
        for k in pooled: pooled[k] += m.get(k, 0)
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
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1", "run_status": "COMPLETE", "authoritative": True,
        "l3_evaluation_authorization_sha256": l3_auth_sha,
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
        f"# Heldout-L3 Claim Boundary\n\nL3 Auth: {l3_auth_sha}\n"
        f"Worst-split false-start: {worst_split}\n"
        f"Gate: {'PASS' if worst_split is not None and worst_split <= 0.10 else 'FAIL'}\n")

    seal_output_dir(staging)
    os.replace(staging, out_root)
    print(f"Heldout-L3 Complete: worst_false_start={worst_split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
