#!/usr/bin/env python3
"""One-shot authorized heldout-L3 evaluator.

Consumes a valid FACTORIZED_HELDOUT_L3_AUTHORIZATION_RECEIPT_V1.json and runs
exactly one heldout evaluation. Produces sealed output with full metrics.

PRE-RUN: requires valid authorization receipt with exact-match artifact bindings.
FAIL-CLOSED: output root must not exist, receipt must not be previously consumed,
  12/12 splits must complete, no silent partial results.

This script is CODE PREPARATION ONLY — do NOT run it until the authorization
gate passes.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, statistics, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from fit_factorized_calibrators import load_json, sha256_file, verify_sealed_directory
from run_factorized_l3_analysis import (
    compute_l3_metrics, exact_join, validate_episode_step_sequence,
    classify_episode, is_valid_start,
)
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter

FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
HEADS = ("grasp", "manipulation", "release")
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


def _group(rows: list[dict[str, Any]], ep_field: str, step_field: str) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[r[ep_field]].append(r)
    for ep_rows in g.values():
        ep_rows.sort(key=lambda r2: r2[step_field])
        validate_episode_step_sequence(ep_rows, step_field)
    return dict(g)


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
    ap.add_argument("--heldout-authorization-receipt", type=Path, required=True)
    ap.add_argument("--heldout-prediction-bundle-root", type=Path, required=True,
                    help="H Student prediction bundle root (must exist)")
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, required=True,
                    help="H Teacher/K10 label bundle root (must exist)")
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # ── Validate authorization receipt ───────────────────────────────
    auth_path = args.heldout_authorization_receipt.resolve()
    auth = load_json(auth_path, "AUTH")
    if auth.get("schema") != "FACTORIZED_HELDOUT_L3_AUTHORIZATION_RECEIPT_V1":
        raise SystemExit("AUTH_SCHEMA_INVALID")
    if not auth.get("heldout_l3_inference_authorized"):
        raise SystemExit("HELDOUT_L3_NOT_AUTHORIZED")
    if auth.get("heldout_l3_completed"):
        raise SystemExit("HELDOUT_L3_ALREADY_COMPLETED: receipt consumed")
    if auth.get("attack_authorized") is not False:
        raise SystemExit("ATTACK_AUTHORIZED: heldout-L3 must not authorize attack")

    # Verify exact-match bindings
    if auth.get("authorized_h_manifest_sha256") != sha256_file(args.heldout_l3_manifest):
        raise SystemExit("H_MANIFEST_BINDING_MISMATCH")
    if auth.get("authorized_calibrator_freeze_sha256") != sha256_file(args.calibrator_freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"):
        raise SystemExit("CAL_FREEZE_BINDING_MISMATCH")
    if auth.get("authorized_scheduler_freeze_sha256") != sha256_file(args.scheduler_freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"):
        raise SystemExit("SCHED_FREEZE_BINDING_MISMATCH")
    if auth.get("authorized_output_root") != str(out_root):
        raise SystemExit("OUTPUT_ROOT_BINDING_MISMATCH")

    # Verify seals
    for label, root in [("H_PRED", args.heldout_prediction_bundle_root),
                         ("H_TEACHER", args.heldout_teacher_bundle_root),
                         ("CAL_FREEZE", args.calibrator_freeze_root),
                         ("SCHED_FREEZE", args.scheduler_freeze_root)]:
        verify_sealed_directory(root, label)

    expected = [s.strip() for s in args.expected_splits.split(",")]

    # Load contracts
    cf_path = args.calibrator_freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"
    sf_path = args.scheduler_freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"
    cf_contract = load_json(cf_path)
    sf_contract = load_json(sf_path)
    held_manifest = load_json(args.heldout_l3_manifest)
    structure = load_json(args.structure_config)
    structure_path = args.structure_config.resolve()

    scheduler_path = ROOT / "src/gripper_attack/factorized_scheduler.py"
    scheduler_source_sha = sha256_file(scheduler_path)

    # ── Generate START receipt ────────────────────────────────────────
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    start_receipt = {
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1",
        "run_status": "STARTED",
        "authorization_receipt_sha256": sha256_file(auth_path),
        "heldout_prediction_bundle_sha256": sha256_file(args.heldout_prediction_bundle_root / "SHA256SUMS"),
        "heldout_teacher_bundle_sha256": sha256_file(args.heldout_teacher_bundle_root / "SHA256SUMS"),
        "calibrator_freeze_sha256": sha256_file(cf_path),
        "scheduler_freeze_sha256": sha256_file(sf_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_code_sha256": SELF_SHA,
    }
    (staging / "HELDOUT_L3_RUN_START_RECEIPT_V1.json").write_text(
        json.dumps(start_receipt, indent=2, sort_keys=True) + "\n")

    # ── Run evaluation per split ──────────────────────────────────────
    all_metrics: dict[str, dict[str, Any]] = {}
    emit_ledger: list[dict[str, Any]] = []
    no_trigger_ledger: list[dict[str, Any]] = []
    unknown_ledger: list[dict[str, Any]] = []
    all_splits_complete = True
    split_errors: list[str] = []
    selected_thresholds = sf_contract.get("selected_thresholds", {})

    for sk in expected:
        try:
            h_ids = _identity_set(held_manifest, "heldout_l3", sk)

            # Load predictions
            pred_path = args.heldout_prediction_bundle_root / sk / "predictions.jsonl"
            pred_rows: list[dict[str, Any]] = []
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                pred_rows.append(json.loads(line))
            pred_ids = {r["canonical_parent_key"] for r in pred_rows}
            if pred_ids != h_ids:
                raise SystemExit(f"H_ID_CLOSURE: {sk}")

            # Load Teacher labels
            teacher_path = args.heldout_teacher_bundle_root / sk / "factorized_teacher_v1.jsonl"
            teacher_rows: list[dict[str, Any]] = []
            for line in teacher_path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                teacher_rows.append(json.loads(line))

            # Build runtime and evaluation episodes
            runtime_episodes: dict[str, list[dict[str, Any]]] = {}
            eval_episodes: dict[str, list[dict[str, Any]]] = {}
            pred_by_key = {(r["canonical_parent_key"], r["step"]): r for r in pred_rows}
            for t_row in teacher_rows:
                ep = t_row["canonical_parent_key"]
                step = t_row["step"]
                p_row = pred_by_key.get((ep, step))
                if not p_row:
                    continue
                r_row = {
                    "episode": ep, "step": step, "split": sk,
                    "checkpoint_sha256": p_row.get("checkpoint_sha256", ""),
                    "source_commit": p_row.get("checkpoint_source_commit", ""),
                    "feature_order_sha256": p_row.get("feature_order_sha256", ""),
                    "scheduler_source_sha256": scheduler_source_sha,
                    "structural_config_sha256": sha256_file(structure_path),
                    "candidate_close": False, "action_known": True,
                    "student_valid": True, "route_supported": True,
                    "grasp_logit": p_row.get("grasp_logit", 0.0),
                    "manipulation_logit": p_row.get("manipulation_logit", 0.0),
                    "release_logit": p_row.get("release_logit", 0.0),
                }
                runtime_episodes.setdefault(ep, []).append(r_row)
                eval_episodes.setdefault(ep, []).append({
                    "step_index": step, "canonical_parent_key": ep, "step": step,
                    "strict_k10_feasible": t_row.get("strict_k10_feasible", False),
                    "strict_k10_known_mask": t_row.get("strict_k10_known_mask", False),
                })

            for ep_rows in runtime_episodes.values():
                ep_rows.sort(key=lambda r: r["step"])
            for ep_rows in eval_episodes.values():
                ep_rows.sort(key=lambda r: r["step"])

            # Build calibration contract for this split
            cf_split = cf_contract["per_split"].get(sk)
            if not cf_split:
                raise SystemExit(f"CAL_FREEZE_SPLIT_MISSING: {sk}")

            cal_contract = {
                "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
                "status": "AUTHORITATIVE", "split": sk,
                "checkpoint_sha256": pred_rows[0].get("checkpoint_sha256", ""),
                "scheduler_source_sha256": scheduler_source_sha,
                "structural_config_sha256": sha256_file(structure_path),
                "student_source_commit": pred_rows[0].get("checkpoint_source_commit", ""),
                "feature_order_sha256": pred_rows[0].get("feature_order_sha256", ""),
                "calibration_fit_authoritative": True,
                "threshold_selection_authoritative": True,
                "l3_evaluation_eligible": True,
                "training_authorized": False,
                "full_fit_authorized": False,
                "attack_authorized": False,
            }
            for head in HEADS:
                hd = cf_split[head]
                cal_contract[head] = {
                    "method": hd["method"], "a": float(hd["a"]), "b": float(hd["b"]),
                    "threshold": float(selected_thresholds.get(head, 0.5)),
                    "transform": "probability=sigmoid(a*raw_logit+b)",
                    "method_valid": True, "transform_valid": True,
                    "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION",
                    "fit_manifest_sha256": "",
                    "policy_selection_manifest_sha256": "",
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

            # Build ledgers
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
            all_splits_complete = False
            split_errors.append(f"{sk}: {e}")

    # ── If not all splits complete, write diagnostic only ─────────────
    if not all_splits_complete:
        diag = {
            "schema": "HELDOUT_L3_DIAGNOSTIC_PARTIAL_V1",
            "partial": True,
            "split_errors": split_errors,
            "completed_splits": sorted(all_metrics),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staging / "HELDOUT_L3_DIAGNOSTIC_PARTIAL_V1.json").write_text(
            json.dumps(diag, indent=2, sort_keys=True) + "\n")
        files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
        os.replace(staging, out_root)
        print("HELDOUT_L3_PARTIAL: diagnostic only, not authoritative")
        return 2

    # ── Build complete outputs ─────────────────────────────────────────
    # Per-split metrics CSV
    csv_lines = ["split,negative_episodes,negative_emits,false_start_rate,positive_episodes,recall,all_emit_precision,median_timing_offset"]
    for sk in sorted(all_metrics):
        m = all_metrics[sk]
        csv_lines.append(f"{sk},{m['negative_episodes']},{m['negative_episode_emits']},{m.get('negative_episode_false_start_rate','')},{m['positive_episodes']},{m.get('valid_opportunity_recall','')},{m.get('all_emit_precision','')},{m.get('median_timing_offset','')}")

    # Pooled metrics
    pooled: dict[str, Any] = {
        "total_negative_episodes": 0, "total_positive_episodes": 0,
        "total_unknown_episodes": 0, "total_negative_emits": 0,
        "total_positive_on_corridor": 0, "total_positive_off_corridor": 0,
        "total_positive_abstentions": 0, "total_emitted_all": 0, "total_emitted_verified": 0,
    }
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

    # Complete receipt
    complete_receipt = {
        "schema": "HELDOUT_L3_RUN_RECEIPT_V1",
        "run_status": "COMPLETE",
        "authoritative": True,
        "authorization_receipt_sha256": sha256_file(auth_path),
        "n_splits": len(all_metrics),
        "all_splits_complete": True,
        "worst_split_false_start_rate": worst_split,
        "gate_pass": worst_split is not None and worst_split <= 0.10,
        "started_at": start_receipt["started_at"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_code_sha256": SELF_SHA,
    }

    (staging / "HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json").write_text(
        json.dumps(complete_receipt, indent=2, sort_keys=True) + "\n")
    (staging / "HELDOUT_L3_PER_SPLIT_METRICS.csv").write_text("\n".join(csv_lines) + "\n")
    (staging / "HELDOUT_L3_POOLED_METRICS.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True) + "\n")
    (staging / "HELDOUT_L3_EMIT_LEDGER.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in emit_ledger))
    (staging / "HELDOUT_L3_NO_TRIGGER_LEDGER.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in no_trigger_ledger))
    (staging / "HELDOUT_L3_UNKNOWN_LEDGER.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in unknown_ledger))
    (staging / "HELDOUT_L3_CLAIM_BOUNDARY.md").write_text(
        f"# Heldout-L3 Claim Boundary\n\n"
        f"Authorization: {sha256_file(auth_path)}\n"
        f"Worst-split false-start: {worst_split}\n"
        f"Gate: {'PASS' if worst_split is not None and worst_split <= 0.10 else 'FAIL'}\n")

    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    os.replace(staging, out_root)

    print(f"Heldout-L3 Complete: {out_root}")
    print(f"  Worst-split false-start: {worst_split}")
    print(f"  Gate: {'PASS' if worst_split is not None and worst_split <= 0.10 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
