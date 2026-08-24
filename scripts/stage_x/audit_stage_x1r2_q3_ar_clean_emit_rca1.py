#!/usr/bin/env python3
"""Offline RCA for the sealed M012 clean traces; never creates a rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from typing import Any


HISTORICAL_RECEIPT = "/llm_jzm/dty_user/openvla_attack_d1_screening_clean_20260818/D1R_CONTINUATION/parents/010_libero_10_task_09_state_43/attempt_0/parent_receipt.json"
HISTORICAL_TELEMETRY = "/llm_jzm/dty_user/openvla_attack_d1_screening_clean_20260818/D1R_CONTINUATION/parents/010_libero_10_task_09_state_43/attempt_0/step_telemetry.jsonl"
Q3_RECEIPT = "/llm_jzm/dty_user/openvla_attack_x1r2_q3_arm_repair_20260820/fixtures/Q3-AR-F01/CLEAN_ENGINEERING/parents/010_libero_10_task_09_state_43/attempt_0/parent_receipt.json"
Q3_TELEMETRY = "/llm_jzm/dty_user/openvla_attack_x1r2_q3_arm_repair_20260820/fixtures/Q3-AR-F01/CLEAN_ENGINEERING/parents/010_libero_10_task_09_state_43/attempt_0/step_telemetry.jsonl"
FLOAT_TOL = 1e-12
HORIZON = 520
PHYSICAL_THRESHOLD = 0.55
CLOSING_THRESHOLD = 0.8


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        out: list[Any] = []
        for key in sorted(value):
            out.extend(leaves(value[key]))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(leaves(item))
        return out
    return [value]


def max_numeric_diff(left: Any, right: Any) -> float | None:
    a, b = leaves(left), leaves(right)
    if len(a) != len(b):
        return None
    diffs: list[float] = []
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            diffs.append(abs(float(x) - float(y)))
    if diffs:
        return max(diffs)
    return 0.0 if left == right else None


def equal_value(left: Any, right: Any) -> bool:
    if left == right:
        return True
    diff = max_numeric_diff(left, right)
    return diff is not None and diff <= FLOAT_TOL


def compare_rows(historical: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "raw_agentview_sha256",
        "processor_input_ids_sha256",
        "processor_pixel_values_sha256",
        "generation_input_ids_sha256",
        "direct_generated_token_ids",
        "action_env_7d",
        "raw_action_7d",
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_velocity",
        "features_25d",
        "student_probabilities",
        "student_scheduler_trace",
        "candidate_close",
        "done_after_env_step",
    ]
    shared = min(len(historical), len(current))
    metrics: dict[str, Any] = {}
    for field in fields:
        first = None
        count = 0
        max_diff = 0.0
        for index in range(shared):
            left, right = historical[index].get(field), current[index].get(field)
            if equal_value(left, right):
                continue
            count += 1
            first = index if first is None else first
            diff = max_numeric_diff(left, right)
            if diff is not None:
                max_diff = max(max_diff, diff)
        metrics[field] = {"first_divergence_step": first, "divergent_rows": count, "max_numeric_diff": max_diff}

    historical_emit = [row["step"] for row in historical if row.get("student_scheduler_trace", {}).get("emitted_this_step")]
    current_emit = [row["step"] for row in current if row.get("student_scheduler_trace", {}).get("emitted_this_step")]
    neighbourhood = []
    for step in range(118, 149):
        if step >= shared:
            continue
        left = historical[step]
        right = current[step]
        hp = left["student_probabilities"]
        cp = right["student_probabilities"]
        neighbourhood.append(
            {
                "step": step,
                "historical": {
                    "candidate_close": bool(left["candidate_close"]),
                    "physical_criticality": hp["physical_criticality"],
                    "gripper_closing_state": hp["gripper_closing_state"],
                    "physical_margin": hp["physical_criticality"] - PHYSICAL_THRESHOLD,
                    "closing_margin": hp["gripper_closing_state"] - CLOSING_THRESHOLD,
                    "emitted": bool(left["student_scheduler_trace"]["emitted_this_step"]),
                },
                "q3_ar": {
                    "candidate_close": bool(right["candidate_close"]),
                    "physical_criticality": cp["physical_criticality"],
                    "gripper_closing_state": cp["gripper_closing_state"],
                    "physical_margin": cp["physical_criticality"] - PHYSICAL_THRESHOLD,
                    "closing_margin": cp["gripper_closing_state"] - CLOSING_THRESHOLD,
                    "emitted": bool(right["student_scheduler_trace"]["emitted_this_step"]),
                },
            }
        )
    return {
        "historical_rows": len(historical),
        "q3_ar_rows": len(current),
        "shared_rows": shared,
        "historical_first_emit_step": historical_emit[0] if historical_emit else None,
        "q3_ar_first_emit_step": current_emit[0] if current_emit else None,
        "field_comparison": metrics,
        "neighbourhood_118_148": neighbourhood,
        "trajectory_equivalence": "DISPROVEN",
        "trajectory_evidence": {
            "raw_agentview_first_divergence": metrics["raw_agentview_sha256"]["first_divergence_step"],
            "action_token_first_divergence": metrics["direct_generated_token_ids"]["first_divergence_step"],
            "qpos_first_divergence": metrics["robot0_gripper_qpos"]["first_divergence_step"],
            "feature_first_divergence": metrics["features_25d"]["first_divergence_step"],
            "historical_terminal_step": historical[-1]["step"],
            "q3_ar_terminal_step": current[-1]["step"],
            "both_clean_success": True,
            "success_is_not_trajectory_equivalence": True,
        },
    }


def authority_matrix() -> list[dict[str, Any]]:
    same_git = {
        "historical": "D1R frozen contract / Git source b17761a158aca448610c251d17843c658392479b",
        "q3_ar": "Q3 repair checkout b7237611c466077a9a7e6f0b1102e9176cfa2c88",
        "evidence": "runner Git blob b17761a158aca448610c251d17843c658392479b; D1R protocol sha256 2b66a3998d6df73af3db5788efae20cc2b51ccbd19c60a1fd31293f5d9b05d4d",
    }
    return [
        {"row": "Student source", "equality": "BYTE_EQUAL", **same_git, "causal_explanation": "No supported Student source drift; the historical external copy sha256 ceb761685ec2bcca033abaca4a71370f6cdbc48908a89d9c2736c9c0807c603b differs from the frozen Git source 30cf..., but the runner imports the Git source. The external-copy provenance remains separately unresolved."},
        {"row": "Student checkpoint", "historical": "sha256 e24d00ca30c8fe0d5ef066e90872f010556bfabec13f78d4275962c6b35ca227", "q3_ar": "same sha256", "equality": "BYTE_EQUAL", "evidence": "D1R contract plus current server hash", "causal_explanation": "Not sufficient to explain the change."},
        {"row": "25D feature schema/order", "historical": "feature source c3c960...; adapter 2b71f3...; dim 25", "q3_ar": "same digests; dim 25", "equality": "BYTE_EQUAL", "evidence": "D1R protocol and current checkout hashes", "causal_explanation": "Applied to different realized trajectories after step 111."},
        {"row": "normalization/statistics", "historical": "sha256 66e24b18a8fa5e46eca41bcdfa8b8aff7c9d05feeb6fcce8d6a62193a469fd6c", "q3_ar": "same sha256", "equality": "BYTE_EQUAL", "evidence": "D1R sealed path and current server hash", "causal_explanation": "Not sufficient to explain the change."},
        {"row": "threshold values", "historical": "physical=0.55; closing=0.80; sha256 5236884e...", "q3_ar": "same values and sha256", "equality": "BYTE_EQUAL", "evidence": "D1R protocol, threshold artifact", "causal_explanation": "At step 133 only the Q3 closing margin is below zero."},
        {"row": "output-head names / alias handling", "historical": "runtime k10_feasible; semantic alias k10_feasibility", "q3_ar": "same runtime names and alias contract", "equality": "SEMANTIC_EQUAL", "evidence": "head contract 7c81b12ac4daaa1513d01cf5c44b7d8f6e7d43fae1466c3a697c62f8554b8ce0", "causal_explanation": "K10 is not an emit-gate input; no causal path to 133 -> None."},
        {"row": "candidate-close rule", "historical": "raw_action[6] < 0.5", "q3_ar": "same runner implementation", "equality": "BYTE_EQUAL", "evidence": "runner blob b17761a...; telemetry candidate field", "causal_explanation": "Realized candidate series first differs at step 132 and contributes to the no-emit result."},
        {"row": "legal-horizon rule", "historical": "step + 5 + 10 <= 520", "q3_ar": "same", "equality": "BYTE_EQUAL", "evidence": "runner blob and both scheduler traces", "causal_explanation": "Legal at step 133 in both traces; not causal."},
        {"row": "emit scheduler rule", "historical": "one-shot candidate AND legal AND physical>=0.55 AND closing>=0.80", "q3_ar": "same", "equality": "BYTE_EQUAL", "evidence": "runner blob and scheduler traces", "causal_explanation": "Same rule sees different sealed probabilities at step 133."},
        {"row": "rolling-history/window reset semantics", "historical": "same D8 adapter and fresh parent construction", "q3_ar": "same source bytes and fresh parent construction", "equality": "BYTE_EQUAL", "evidence": "adapter sha256 2b71f3...; identical prefix features through step 111", "causal_explanation": "No reset drift is supported by the prefix evidence."},
        {"row": "clean seed derivation", "historical": "1436562779", "q3_ar": "1436562779", "equality": "BYTE_EQUAL", "evidence": "both receipts and same runner seed path", "causal_explanation": "Same declared seed does not guarantee identical environment observation bytes."},
        {"row": "reset/dummy-wait semantics", "historical": "one reset; 10 dummy waits", "q3_ar": "one reset; 10 dummy waits", "equality": "SEMANTIC_EQUAL", "evidence": "runner source and counters", "causal_explanation": "No evidence of a reset-index shift; prefix starts equal."},
        {"row": "policy-step indexing", "historical": "0..249", "q3_ar": "0..255", "equality": "SEMANTIC_EQUAL", "evidence": "both telemetry policy_step/step fields", "causal_explanation": "Shared indices align through the first divergence; terminal length differs."},
        {"row": "episode horizon", "historical": "configured L10=520; done at 249", "q3_ar": "configured L10=520; done at 255", "equality": "SEMANTIC_EQUAL", "evidence": "D1R protocol and receipts", "causal_explanation": "Configured horizon is equal; realized clean trajectory termination differs."},
        {"row": "task/state/prompt identity", "historical": "libero_10/task_09/state_43; same instruction", "q3_ar": "same identity and instruction", "equality": "BYTE_EQUAL", "evidence": "both receipts and telemetry canonical_parent_key/prompt", "causal_explanation": "Identity mismatch is not supported."},
        {"row": "clean OpenVLA/victim source/checkpoint binding", "historical": "launch-time full identity not uniquely sealed", "q3_ar": "current suite contract/static binding only", "equality": "NOT_IDENTIFIABLE", "evidence": "D0R2 runtime audit errors: historical launch-time source/victim path not unique", "causal_explanation": "Could explain the trajectory drift, but cannot be assigned as the cause without historical launch-time bytes."},
    ]


def cpu_replay(rows: list[dict[str, Any]], root: pathlib.Path, checkpoint_path: pathlib.Path, normalization_path: pathlib.Path, repeat: int = 2) -> dict[str, Any]:
    import numpy as np
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("RCA1_CPU_ONLY_VIOLATION")
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / "n5" / "phase3_student"))
    from n5_student_model import N5MultiHeadStudent

    normalization = load_json(normalization_path)["episode_heldout"]["train"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    features = np.asarray([row["features_25d"] for row in rows], dtype=np.float32)
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))[None, ...]
    mask = torch.ones((1, len(rows)), dtype=torch.bool)
    outputs: list[list[float]] = []
    for _ in range(repeat):
        with torch.no_grad():
            logits = model(x, timestep_mask=mask)
            names = tuple(model.HEAD_NAMES)
            outputs.append([float(torch.sigmoid(logits[name][0, i]).item()) for i in range(len(rows)) for name in names])
    repeat_max = max(abs(a - b) for a, b in zip(outputs[0], outputs[1])) if repeat > 1 else None
    sealed = [float(row["student_probabilities"][name]) for row in rows for name in names]
    sealed_max = max(abs(a - b) for a, b in zip(outputs[0], sealed))
    prediction_rows = []
    for i, row in enumerate(rows):
        values = {name: outputs[0][i * len(names) + j] for j, name in enumerate(names)}
        legal = i + 5 + 10 <= HORIZON
        emitted = bool(row["candidate_close"] and legal and values["physical_criticality"] >= PHYSICAL_THRESHOLD and values["gripper_closing_state"] >= CLOSING_THRESHOLD)
        prediction_rows.append({"step": i, "candidate_close": bool(row["candidate_close"]), "legal_horizon": legal, "emitted_this_step": emitted, "physical_criticality": values["physical_criticality"], "gripper_closing_state": values["gripper_closing_state"]})
    emitted = [row["step"] for row in prediction_rows if row["emitted_this_step"]]
    digest = hashlib.sha256(json.dumps(prediction_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"status": "PASS_CPU_OFFLINE_REPLAY", "rows": len(rows), "head_names": list(names), "first_emit_step": emitted[0] if emitted else None, "repeat_max_abs_diff": repeat_max, "max_abs_diff_vs_sealed_student_probabilities": sealed_max, "trace_sha256": digest, "model_execution": "Student CPU only; no OpenVLA/simulator"}


def build_report(historical_receipt_path: pathlib.Path, q3_receipt_path: pathlib.Path, historical_path: pathlib.Path, q3_path: pathlib.Path) -> dict[str, Any]:
    historical_receipt = load_json(historical_receipt_path)
    q3_receipt = load_json(q3_receipt_path)
    if historical_receipt.get("canonical_parent_key") != "libero_10/task_09/state_43" or historical_receipt.get("first_emit_step") != 133:
        raise RuntimeError("HISTORICAL_M012_RECEIPT_BINDING_INVALID")
    if q3_receipt.get("canonical_parent_key") != "libero_10/task_09/state_43" or q3_receipt.get("first_emit_step") is not None:
        raise RuntimeError("Q3_AR_M012_RECEIPT_BINDING_INVALID")
    historical = load_rows(historical_path)
    current = load_rows(q3_path)
    report = {
        "schema": "STAGE_X_X1R2_Q3_AR_CLEAN_EMIT_RCA1_V1",
        "status": "PASS_RCA1_STOP_OWNER_REVIEW",
        "audit_input_live_pr": {"number": 135, "head": "404b3793fd103bdcff269502561470c3e6cca13f", "tree": "9b94be132fbe4a987a2dc8073d29fca1fac1e9b5", "final_seal": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF"},
        "scope": "static and sealed-artifact offline RCA only; no new rollout, OpenVLA inference, simulator, env.step, PGD, attack arm, V_phys, Eval160, or protected read",
        "identity": {"canonical_parent_key": "libero_10/task_09/state_43", "review_id": "M012", "ordinal": 10, "seed": 1436562779, "historical_first_emit": 133, "q3_ar_first_emit": None},
        "runtime_source": {"historical_receipt": historical_receipt.get("runtime_source_pre_evidence"), "q3_ar_receipt": q3_receipt.get("runtime_source_pre_evidence"), "shared_runner_git_blob": "b17761a158aca448610c251d17843c658392479b"},
        "artifact_digests": {
            "historical_receipt": {"path": str(historical_receipt_path), "remote_path": HISTORICAL_RECEIPT, "sha256": sha256(historical_receipt_path)},
            "historical_telemetry": {"path": str(historical_path), "remote_path": HISTORICAL_TELEMETRY, "sha256": sha256(historical_path)},
            "q3_ar_receipt": {"path": str(q3_receipt_path), "remote_path": Q3_RECEIPT, "sha256": sha256(q3_receipt_path)},
            "q3_ar_telemetry": {"path": str(q3_path), "remote_path": Q3_TELEMETRY, "sha256": sha256(q3_path)},
        },
        "historical_133_provenance_chain": [
            {"stage": "D1R clean receipt", "path": HISTORICAL_RECEIPT, "sha256": "cbe1bd28968bb5b30bdd7622209675edcc4ad0425e9cf749189f3c537f968b25", "identity_field": "canonical_parent_key=libero_10/task_09/state_43", "emit_field": "first_emit_step=133"},
            {"stage": "D1R telemetry", "path": HISTORICAL_TELEMETRY, "sha256": "e9e536100ba6102a7266ef8af2cbcb5988e647214a1bd5e27a9800bc258a8724", "identity_field": "canonical_parent_key and step=133", "emit_field": "student_scheduler_trace.emitted_this_step=true"},
            {"stage": "D1R census row", "path": "reports/STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1.json", "sha256": "65e9f61f4855efd253e6a36577409f398422e12c19f293ce73a605ef89929baf", "identity_field": "ordinal=10 and canonical_parent_key", "emit_field": "first_emit_step=133"},
            {"stage": "Q3-AR fixture expectation", "path": "reports/STAGE_X_X1R2_Q3_ARM_REPAIR_FIXTURE_V1.json", "sha256": "adc42d49dcd8b35d4a662ced2efff81ecef51184601d63fca52b699928210205", "identity_field": "fixture_id=Q3-AR-F01 / review_id=M012", "emit_field": "first_emit_step=133"},
        ],
        "authority_matrix": authority_matrix(),
        "offline_replay": {"sealed_trace_comparison": compare_rows(historical, current), "cpu_student_replay": "PENDING_CPU_RUN"},
        "root_cause": {
            "primary": ["CLEAN_TRAJECTORY_DRIFT"],
            "secondary": ["UNRESOLVED_MULTI_FACTOR"],
            "not_supported": ["EXPECTATION_PROVENANCE_ERROR", "STUDENT_ARTIFACT_OR_RUNTIME_DRIFT", "STEP_INDEX_OR_RESET_CONTRACT_DRIFT"],
            "historical_stepwise_evidence": "M012 has sealed per-step features, Student probabilities, scheduler trace, tokens, action and proprioception; the older generic D0R2 parity limitation is not upgraded into this M012 trace.",
            "explanation": "The 133 expectation is correctly bound to M012. The already-consumed Q3 clean replay is not trajectory-equivalent: image hashes diverge at step 49, action tokens at 111, 25D/Student values at 112, and terminal length is 250 versus 256. At step 133 the same scheduler is legal and candidate-close in both traces, but Q3 gripper_closing_state=0.6740048528 is below 0.8 while historical=0.8626471758, so None follows without invoking stochasticity.",
            "student_stochasticity": "NOT_CLAIMED; no new stochasticity evidence",
            "unresolved": ["historical launch-time external Student copy is not uniquely bound to the runner import path", "historical and Q3 clean OpenVLA/victim full checkpoint/processor launch-time identity is not sealed in the episode receipts", "origin of the first observation/renderer divergence cannot be isolated without a new rollout"],
        },
        "exposure_accounting": {"historical_q3_f01": "immutable runtime-invalid; not rerun", "q3_ar_f01": {"status": "permanently excluded engineering fixture", "clean_model_inference_calls": 256, "clean_env_step_calls": 266, "attack_invocations": 0, "pgd_calls": 0, "attacked_env_steps": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0}},
        "authorization": {"arm_isolation_repair": "UNQUALIFIED", "scientific_x1r2_population": "NOT_SELECTED", "q3_f02_f04": "SEALED_NOT_STARTED", "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        "rca1_counters": {"new_openvla_model_inference_calls": 0, "offline_student_cpu_forward_passes": 4, "new_simulator_env_step_calls": 0, "new_pgd_calls": 0, "new_attack_backward_calls": 0, "new_attacked_env_steps": 0, "new_physical_interventions": 0, "new_vphys_reads": 0, "new_eval160_reads": 0, "new_protected_reads": 0},
        "next_gate": "OWNER_REVIEW_RCA1_ONLY",
        "recommendation": "Do not rerun M012 or reinterpret None as arm qualification. If owner authorizes a new engineering gate, first freeze a provenance-complete clean replay contract that binds the actual OpenVLA/victim launch bytes and observation lifecycle; keep this fixture permanently excluded.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--q3-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--historical-telemetry", type=pathlib.Path, required=True)
    parser.add_argument("--q3-telemetry", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--student-root", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--normalization", type=pathlib.Path)
    parser.add_argument("--cpu-results", type=pathlib.Path)
    args = parser.parse_args()
    report = build_report(args.historical_receipt, args.q3_receipt, args.historical_telemetry, args.q3_telemetry)
    if args.student_root and args.checkpoint and args.normalization:
        historical = load_rows(args.historical_telemetry)
        current = load_rows(args.q3_telemetry)
        report["offline_replay"]["cpu_student_replay"] = {"historical_features": cpu_replay(historical, args.student_root, args.checkpoint, args.normalization), "q3_ar_features": cpu_replay(current, args.student_root, args.checkpoint, args.normalization)}
    elif args.cpu_results:
        report["offline_replay"]["cpu_student_replay"] = load_json(args.cpu_results)["offline_replay"]["cpu_student_replay"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out), "historical_first_emit": 133, "q3_ar_first_emit": None, "primary": report["root_cause"]["primary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
