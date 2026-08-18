"""CPU/static T1-D0R2 authority audit.

The only model forwards allowed here are CPU forwards over four already-sealed
T1-C replay inputs.  No OpenVLA, simulator, fresh parent, or attack path is
loaded or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gripper_attack.stage_x_x1r_d1_clean_runtime_contract import (  # noqa: E402
    H_PHYS,
    HORIZONS,
    NUM_STEPS_WAIT,
    SUITES,
    T5_STEPS,
    configured_episode_length,
    legal_horizon,
)
from scripts.stage_x.audit_stage_x1r_t1_detector_authority import (  # noqa: E402
    clean_shadow_rows,
    schedule,
    sha256 as detector_sha256,
    student_prediction_at_prefix,
    student_predictions,
)
from scripts.stage_x.audit_stage_x1r_t1d0r1_preclean_integrity import (  # noqa: E402
    audit as audit_d0r1,
    derive_population,
    identity_digest,
    load_d0r1_sources,
    load_json as load_d0r1_json,
    load_d0_selected_parent_keys,
    seed_for_parent,
)


REPLAY_ATOL = 1e-6
MODEL_SOURCE = "n5/phase3_student/n5_student_model.py"
FEATURE_SOURCE = "src/gripper_attack/v5_r3_features.py"
ADAPTER_SOURCE = "src/gripper_attack/d8_streaming_features_v3.py"
FORBIDDEN_COUNTERS = (
    "openvla_weight_loads",
    "openvla_model_inference_calls",
    "prospective_parent_student_forward_calls",
    "prospective_parent_clean_rollouts",
    "env_reset_for_prospective_parent",
    "env_step_calls",
    "pgd_calls",
    "attack_backward_calls",
    "adversarial_images",
    "physical_interventions",
    "vphys_reads",
    "attack_outcome_reads",
    "eval160_reads",
    "protected_reads",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, *, normalize_text: bool = False) -> str:
    data = path.read_bytes()
    if normalize_text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_blob(repo: Path, ref: str, relative: str) -> dict[str, Any]:
    spec = f"{ref}:{relative}"
    payload = subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", spec])
    return {
        "ref": ref,
        "path": relative,
        "git_blob_sha": git_text(repo, "rev-parse", spec),
        "byte_size": len(payload),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def counters(historical_forward_calls: int = 0) -> dict[str, int]:
    return {
        "historical_replay_student_forward_calls": int(historical_forward_calls),
        **{name: 0 for name in FORBIDDEN_COUNTERS},
    }


def source_forensic(config: Mapping[str, Any], receipt: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    spec = config["student"]
    sealed = receipt.get("sealed_detector", {})
    receipt_sha = sealed.get("student_model_source_sha256")
    current_path = Path(spec["current_server_file_path"])
    current_sha = sha256_file(current_path) if current_path.is_file() else None
    git_row = git_blob(repo, "HEAD", MODEL_SOURCE)
    declarations = {
        "t1_handoff_declaration": spec["historical_t1_handoff_source_sha256"],
        "historical_t1_receipt_value": receipt_sha,
        "current_server_file": current_sha,
        "prospective_git_source_raw": git_row["raw_sha256"],
    }
    errors: list[str] = []
    if receipt_sha != spec["historical_receipt_source_sha256"]:
        errors.append("HISTORICAL_RECEIPT_SOURCE_SHA_MISMATCH")
    if current_sha != spec["current_server_file_sha256"]:
        errors.append("CURRENT_SERVER_SOURCE_SHA_MISMATCH")
    if git_row["raw_sha256"] != spec["prospective_source_raw_sha256"]:
        errors.append("PROSPECTIVE_GIT_SOURCE_RAW_SHA_MISMATCH")
    return {
        "schema": "STAGE_X_X1R_T1D0R2_STUDENT_SOURCE_FORENSIC_V1",
        "status": "PASS_FORENSIC_CLASSIFICATION" if not errors else "HOLD_SOURCE_BINDING",
        "declarations": declarations,
        "source_records": {
            "prospective_git_source": git_row,
            "current_server_file": {
                "path": str(current_path),
                "raw_sha256": current_sha,
                "byte_size": current_path.stat().st_size if current_path.is_file() else None,
            },
            "historical_t1_receipt": {
                "path": str(config["historical_t1_receipt"]["path"]),
                "json_key": "sealed_detector.student_model_source_sha256",
                "value": receipt_sha,
                "source_commit": receipt.get("source", {}).get("head"),
                "source_tree": receipt.get("source", {}).get("tree"),
            },
        },
        "classification": {
            "historical_training_source": "NOT_IDENTIFIABLE",
            "reason": "T1 handoff, receipt runtime identity, and current server file are not a single proven launch-time training source.",
            "prospective_inference_implementation": "TRACKED_PR127_SOURCE_30CF",
            "historical_receipt_runtime_identity_matches_prospective_git_bytes": receipt_sha == git_row["raw_sha256"],
            "current_server_file_matches_prospective_git_bytes": current_sha == git_row["raw_sha256"],
        },
        "errors": errors,
    }


def _has_per_step_reference(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("per_step_probabilities", "probability_trace", "sealed_trace_path", "expected_probabilities")):
            return True
        return any(_has_per_step_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_per_step_reference(item) for item in value)
    return False


def replay_parity(config: Mapping[str, Any], receipt: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    sealed = receipt.get("sealed_detector", {})
    checkpoint_path = Path(str(sealed.get("checkpoint", config["student"]["checkpoint_path"])))
    norm_info = sealed.get("normalization", {})
    thresholds_info = sealed.get("thresholds", {})
    normalization_path = Path(str(norm_info.get("path", "")))
    thresholds_path = Path(str(thresholds_info.get("path", "")))
    errors: list[str] = []
    for path in (checkpoint_path, normalization_path, thresholds_path):
        if not path.is_file():
            errors.append(f"MISSING_STUDENT_RUNTIME_FILE:{path}")
    if sha256_file(checkpoint_path) != config["student"]["checkpoint_sha256"]:
        errors.append("CHECKPOINT_SHA_MISMATCH")
    if sha256_file(repo / FEATURE_SOURCE, normalize_text=True) != config["student"]["feature_source_sha256"]:
        errors.append("FEATURE_SOURCE_SHA_MISMATCH")
    if sha256_file(repo / ADAPTER_SOURCE, normalize_text=True) != config["student"]["adapter_sha256"]:
        errors.append("ADAPTER_SHA_MISMATCH")
    if normalization_path.is_file() and sha256_file(normalization_path) != config["student"]["normalization_sha256"]:
        errors.append("NORMALIZATION_SHA_MISMATCH")
    if thresholds_path.is_file() and sha256_file(thresholds_path) != config["student"]["thresholds_sha256"]:
        errors.append("THRESHOLDS_SHA_MISMATCH")
    if errors:
        return {
            "schema": "STAGE_X_X1R_T1D0R2_STUDENT_REPLAY_PARITY_V1",
            "status": "HOLD_STUDENT_RUNTIME_BINDING",
            "sealed_per_step_reference_available": False,
            "errors": errors,
            "replays": {},
            "counters": counters(),
        }

    normalization = load_json(normalization_path)
    norm = normalization.get("episode_heldout", {}).get("train", {})
    mean = np.asarray(norm.get("mean", []), dtype=np.float32)
    std = np.asarray(norm.get("std", []), dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        errors.append("NORMALIZATION_SCHEMA_INVALID")
    thresholds = load_json(thresholds_path)
    physical_threshold = float(thresholds.get("physical_criticality", {}).get("threshold", float("nan")))
    closing_threshold = float(thresholds.get("gripper_closing_state", {}).get("threshold", float("nan")))
    if (physical_threshold, closing_threshold) != (0.55, 0.8):
        errors.append("FROZEN_THRESHOLD_BINDING_MISMATCH")

    replay_summaries = receipt.get("clean_shadow", {})
    reference_available = _has_per_step_reference(replay_summaries)
    if not reference_available:
        errors.append("STAGE_X_X1R_T1D0R2_HOLD_STUDENT_REPLAY_PARITY:MISSING_SEALED_PER_STEP_REFERENCE")

    sys.path.insert(0, str(repo / "n5/phase3_student"))
    from n5_student_model import N5MultiHeadStudent  # type: ignore  # noqa: PLC0415

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    except Exception as exc:
        errors.append(f"STUDENT_CHECKPOINT_STRICT_LOAD_FAILED:{type(exc).__name__}:{exc}")
        model = None

    active_heads = list(checkpoint.get("active_heads", []))
    if active_heads != config["student"]["active_heads"]:
        errors.append("ACTIVE_HEADS_MISMATCH")

    replay_rows: dict[str, Any] = {}
    forward_calls = 0
    replay_root = Path(config["historical_t1_receipt"]["replay_root"])
    for suite in SUITES:
        summary = replay_summaries.get(suite)
        if not isinstance(summary, dict):
            errors.append(f"MISSING_T1_C_REPLAY_SUMMARY:{suite}")
            continue
        path = Path(str(summary.get("replay_path", "")))
        if not path.is_file():
            errors.append(f"MISSING_T1_C_REPLAY_INPUT:{suite}:{path}")
            continue
        if path.parent.parent.parent != replay_root and not str(path).startswith(str(replay_root)):
            errors.append(f"REPLAY_OUTSIDE_SEALED_ROOT:{suite}")
        actual_sha = sha256_file(path)
        if actual_sha != summary.get("replay_sha256"):
            errors.append(f"REPLAY_SHA_MISMATCH:{suite}")
        replay = load_json(path)
        if replay.get("outcomes_read") is not False or replay.get("v_phys_read") is not False or replay.get("intervention_executed") is not False:
            errors.append(f"REPLAY_NOT_CLEAN_ONLY:{suite}")
        if any(int(v) != 0 for v in replay.get("protected_counters", {}).values()):
            errors.append(f"REPLAY_PROTECTED_COUNTER_NONZERO:{suite}")
        if replay.get("canonical_parent_key") in {str(row.get("canonical_parent_key")) for row in load_jsonl(repo / "reports/STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json")}:
            errors.append(f"REPLAY_FRESH_PARENT_OVERLAP:{suite}")
        if model is None or mean.shape != (25,):
            continue
        try:
            features, close_flags = clean_shadow_rows(path)
            first = student_predictions(model, features, mean, std)
            forward_calls += 1
            second = student_predictions(model, features, mean, std)
            forward_calls += 1
            first_schedule = schedule(first, close_flags, t5=T5_STEPS, h_phys=H_PHYS, physical_threshold=physical_threshold, closing_threshold=closing_threshold)
            second_schedule = schedule(second, close_flags, t5=T5_STEPS, h_phys=H_PHYS, physical_threshold=physical_threshold, closing_threshold=closing_threshold)
            first_array = np.asarray([[row["physical_criticality"], row["gripper_closing_state"]] for row in first])
            second_array = np.asarray([[row["physical_criticality"], row["gripper_closing_state"]] for row in second])
            repeat_diff = float(np.max(np.abs(first_array - second_array), initial=0.0))
            prefix_indices = list(summary.get("prefix_parity_indices", []))
            prefix_diffs: list[float] = []
            for index in prefix_indices:
                prefix = student_prediction_at_prefix(model, features, mean, std, int(index) + 1)
                forward_calls += 1
                full = first[int(index)]
                prefix_diffs.append(max(abs(prefix[name] - full[name]) for name in ("physical_criticality", "gripper_closing_state")))
            prefix_diff = float(max(prefix_diffs, default=0.0))
            if repeat_diff > REPLAY_ATOL or prefix_diff > REPLAY_ATOL:
                errors.append(f"REPLAY_FUNCTIONAL_DETERMINISM_FAIL:{suite}")
            expected_emit = summary.get("first_emit_step")
            if first_schedule["first_emit_step"] != expected_emit or bool(first_schedule["first_emit_step"] is None) != bool(summary.get("no_emit_retained")):
                errors.append(f"REPLAY_SUMMARY_EMIT_MISMATCH:{suite}")
            trace = [{"step": i, **row} for i, row in enumerate(first)]
            replay_rows[suite] = {
                "replay_path": str(path),
                "replay_sha256": actual_sha,
                "canonical_parent_key": replay.get("canonical_parent_key"),
                "row_count": len(features),
                "feature_dim": 25,
                "strict_checkpoint_load": True,
                "active_heads": active_heads,
                "first_emit_step": first_schedule["first_emit_step"],
                "sealed_summary_first_emit_step": expected_emit,
                "summary_emit_match": first_schedule["first_emit_step"] == expected_emit,
                "repeat_max_probability_abs_diff": repeat_diff,
                "prefix_parity_indices": prefix_indices,
                "prefix_max_probability_abs_diff": prefix_diff,
                "trace_sha256": canonical_sha(trace),
                "sealed_per_step_reference_available": False,
                "per_step_max_abs_diff": None,
                "per_step_reference_note": "T1-C receipt contains summary-only fields and no sealed per-step probability arrays.",
            }
        except Exception as exc:
            errors.append(f"REPLAY_FAILED:{suite}:{type(exc).__name__}:{exc}")

    return {
        "schema": "STAGE_X_X1R_T1D0R2_STUDENT_REPLAY_PARITY_V1",
        "status": "PASS_FUNCTIONAL_REPLAY_SUMMARY_ONLY" if not errors else "HOLD_STUDENT_REPLAY_PARITY",
        "prospective_implementation": {
            "source_path": MODEL_SOURCE,
            "git_source": git_blob(repo, "HEAD", MODEL_SOURCE),
            "checkpoint_sha256": config["student"]["checkpoint_sha256"],
            "input_dim": 25,
            "hidden": 64,
            "short_rf": 32,
            "long_rf": 128,
            "dropout_eval": 0.0,
        },
        "sealed_per_step_reference_available": reference_available,
        "sealed_per_step_reference_required": True,
        "replays": replay_rows,
        "historical_replay_student_forward_calls": forward_calls,
        "counters": counters(forward_calls),
        "errors": errors,
    }


def source_file_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    row = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        row.update({"raw_sha256": sha256_file(path), "byte_size": path.stat().st_size})
    else:
        row.update({"raw_sha256": None, "byte_size": None})
    if expected is not None:
        row["expected_raw_sha256"] = expected
        row["match"] = row["raw_sha256"] == expected
    return row


def success_horizon(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["success_horizon_lineage"]
    upstream = Path(spec["upstream_repo"])
    commit = spec["upstream_commit"]
    tree = git_text(upstream, "rev-parse", f"{commit}^{{tree}}")
    evaluator = git_blob(upstream, commit, spec["evaluator_path"])
    text = subprocess.check_output(["git", "-C", str(upstream), "cat-file", "blob", f"{commit}:{spec['evaluator_path']}"], text=True)
    errors: list[str] = []
    if tree != spec["upstream_tree"]:
        errors.append("UPSTREAM_TREE_MISMATCH")
    if evaluator["raw_sha256"] != spec["evaluator_raw_sha256"] or evaluator["git_blob_sha"] != spec["evaluator_git_blob_sha"]:
        errors.append("EVALUATOR_SOURCE_HASH_MISMATCH")
    required_text = (
        "num_steps_wait: int = 10",
        "while t < max_steps + cfg.num_steps_wait",
        "env.step(get_libero_dummy_action(cfg.model_family))",
        "env.step(action.tolist())",
        "if done:",
        "task_successes += 1",
    )
    missing = [fragment for fragment in required_text if fragment not in text]
    errors.extend(f"EVALUATOR_TEXT_MISSING:{fragment}" for fragment in missing)
    horizon_fragments = {"libero_spatial": "max_steps = 220", "libero_object": "max_steps = 280", "libero_goal": "max_steps = 300", "libero_10": "max_steps = 520"}
    for suite, fragment in horizon_fragments.items():
        if fragment not in text:
            errors.append(f"HORIZON_LITERAL_MISSING:{suite}")

    libero_root = Path(spec["libero_source_root"])
    env_wrapper = source_file_record(libero_root / spec["env_wrapper_path"], spec["env_wrapper_raw_sha256"])
    bddl = source_file_record(libero_root / spec["bddl_domain_path"], spec["bddl_domain_raw_sha256"])
    if not env_wrapper.get("match") or not bddl.get("match"):
        errors.append("LIBERO_SOURCE_HASH_MISMATCH")
    wrapper_text = (libero_root / spec["env_wrapper_path"]).read_text(encoding="utf-8") if env_wrapper["exists"] else ""
    bddl_text = (libero_root / spec["bddl_domain_path"]).read_text(encoding="utf-8") if bddl["exists"] else ""
    for fragment, source in (("def step(self, action):", wrapper_text), ("return self.env.step(action)", wrapper_text), ("def _check_success(self):", bddl_text), ("done = self._check_success()", bddl_text)):
        if fragment not in source:
            errors.append(f"LIBERO_SUCCESS_LINEAGE_TEXT_MISSING:{fragment}")

    horizon_rows = [
        {"suite": suite, "task_index": task, "configured_episode_length": configured_episode_length(suite, task), "policy_decision_indices": [0, HORIZONS[suite] - 1]}
        for suite in SUITES
        for task in range(10)
    ]
    boundary = {"t_emit": 0, "episode_length_14": legal_horizon(0, 14), "episode_length_15": legal_horizon(0, 15), "t_emit_5_length_19": legal_horizon(5, 19), "t_emit_5_length_20": legal_horizon(5, 20)}
    if boundary != {"t_emit": 0, "episode_length_14": False, "episode_length_15": True, "t_emit_5_length_19": False, "t_emit_5_length_20": True}:
        errors.append("LEGAL_HORIZON_OFF_BY_ONE")

    return {
        "schema": "STAGE_X_X1R_T1D0R2_SUCCESS_HORIZON_AUTHORITY_V1",
        "status": "PASS_SUCCESS_AND_HORIZON_BINDING" if not errors else "HOLD_SUCCESS_HORIZON_AUTHORITY",
        "success_authority": {
            "status": "BOUND",
            "semantic_origin": "OPENVLA_LIBERO_CANONICAL_ENV_STEP_DONE",
            "canonical_evaluator": evaluator,
            "call_expression": "obs, reward, done, info = env.step(action.tolist()); if done: task_successes += 1",
            "transition_position": "after env.step transition",
            "benchmark_native_predicate": "BDDL task _check_success() invoked by bddl_base_domain.step",
            "wrapper_delegate": "ControlEnv.step -> self.env.step(action)",
            "first_success": "first post-step done=True terminates the task loop",
            "step_index": "t=0 at first policy decision after dummy wait; wait steps are outside policy index",
            "suite_coverage": list(SUITES),
            "comparison_candidates": [
                {"name": "project_SC5_clean_runner", "status": "COMPARISON_ONLY", "reason": "three-suite local runner with 400-step override semantics"},
                {"name": "ControlEnv.check_success", "status": "EQUIVALENT_DELEGATE_NOT_CALLSITE", "source": source_file_record(libero_root / spec["env_wrapper_path"])},
            ],
        },
        "horizon_authority": {
            "status": "BOUND",
            "semantic_origin": "OPENVLA_LIBERO_CANONICAL_POLICY_DECISION_HORIZON",
            "configured_episode_length_definition": "hard maximum number of policy-decision actions after the canonical ten dummy wait steps",
            "num_steps_wait": NUM_STEPS_WAIT,
            "dummy_wait_counts_toward_policy_horizon": False,
            "reset_counts_toward_policy_horizon": False,
            "rows": horizon_rows,
            "source": evaluator,
        },
        "source_records": {"upstream_evaluator": evaluator, "libero_env_wrapper": env_wrapper, "libero_bddl_domain": bddl},
        "legal_horizon_boundary_tests": boundary,
        "errors": errors,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parent_seed_invariance(config: Mapping[str, Any], repo: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    protocol = load_d0r1_json(repo / config["d0r1_protocol"])
    g10, exclusions, _, _, physical_canonical = load_d0r1_sources(protocol)
    derived = derive_population(g10, exclusions, str(protocol["selection"]["selection_salt"]))
    d0r1_receipt, _, _ = audit_d0r1(protocol, repo)
    ledger = load_jsonl(repo / "reports/STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json")
    ledger = sorted(ledger, key=lambda row: int(row["ordinal"]))
    derived_rows = derived["parent_rows"]
    namespace = str(protocol["seed_authority"]["namespace"])
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if len(g10) != 1200 or len(derived["exclusion_union"]) != 990 or len(derived["fresh_rows"]) != 210 or len(derived["design_rows"]) != 40 or len(derived_rows) != 39:
        errors.append("D0R1_POPULATION_COUNTS_DRIFT")
    selected = [row["canonical_parent_key"] for row in derived_rows]
    if selected != load_d0_selected_parent_keys(repo):
        errors.append("D0R1_SELECTION_DRIFT")
    if len(ledger) != len(derived_rows):
        errors.append("D0R1_LEDGER_ROW_COUNT_DRIFT")
    for expected, actual in zip(derived_rows, ledger):
        key = expected["canonical_parent_key"]
        seed = seed_for_parent(namespace, key)
        row = {"ordinal": expected.get("ordinal"), "canonical_parent_key": key, "expected_clean_seed": seed, "ledger_clean_seed": actual.get("clean_seed"), "seed_match": seed == actual.get("clean_seed"), "ledger_key_match": key == actual.get("canonical_parent_key")}
        rows.append(row)
        if not row["seed_match"] or not row["ledger_key_match"]:
            errors.append(f"D0R1_SEED_OR_KEY_DRIFT:{key}")
    if d0r1_receipt.get("alias_invariance", {}).get("status") != "PASS":
        errors.append("D0R1_ALIAS_INVARIANCE_NOT_PASS")
    if protocol["selection"].get("replacement") is not False and protocol["selection"].get("zero_replacement") is not True:
        errors.append("D0R1_REPLACEMENT_POLICY_DRIFT")
    return {
        "schema": "STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1",
        "status": "PASS_D0R1_INVARIANTS" if not errors else "HOLD_PARENT_SEED_DRIFT",
        "selection_salt": protocol["selection"]["selection_salt"],
        "namespace": namespace,
        "population": {"g10": len(g10), "exclusion_union": len(derived["exclusion_union"]), "fresh": len(derived["fresh_rows"]), "nominal_cells": len(derived["design_rows"]), "executable_parents": len(derived_rows), "missing_cell": [row["design_cell"] for row in derived["design_rows"] if not row["selected"]], "suite_counts": {suite: sum(row["suite"] == suite for row in derived_rows) for suite in SUITES}, "replacement": False},
        "selection_digest": identity_digest(selected),
        "rows": rows,
        "errors": errors,
    }


def runtime_audit(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    current = {"head": git_text(repo, "rev-parse", "HEAD"), "tree": git_text(repo, "rev-parse", "HEAD^{tree}"), "status_porcelain": git_text(repo, "status", "--porcelain")}
    receipt_path = Path(config["historical_t1_receipt"]["path"])
    receipt = load_json(receipt_path)
    forensic = source_forensic(config, receipt, repo)
    replay = replay_parity(config, receipt, repo)
    success = success_horizon(config)
    parent = parent_seed_invariance(config, repo, receipt)
    errors = list(forensic.get("errors", [])) + list(replay.get("errors", [])) + list(success.get("errors", [])) + list(parent.get("errors", []))
    if current["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN_BEFORE_EVIDENCE")
    if any(name.startswith("STAGE_X_X1R_T1D0R2_HOLD_STUDENT_REPLAY_PARITY") for name in replay.get("errors", [])) or not replay.get("sealed_per_step_reference_available", False):
        status = "STAGE_X_X1R_T1D0R2_HOLD_STUDENT_REPLAY_PARITY"
    elif any("SUCCESS" in error for error in success.get("errors", [])):
        status = "STAGE_X_X1R_T1D0R2_HOLD_SUCCESS_EVALUATOR_AUTHORITY"
    elif any("HORIZON" in error for error in success.get("errors", [])):
        status = "STAGE_X_X1R_T1D0R2_HOLD_EPISODE_HORIZON_AUTHORITY"
    elif parent.get("status") != "PASS_D0R1_INVARIANTS":
        status = "STAGE_X_X1R_T1D0R2_HOLD_PARENT_SEED_DRIFT"
    elif forensic.get("status") != "PASS_FORENSIC_CLASSIFICATION":
        status = "STAGE_X_X1R_T1D0R2_HOLD_STUDENT_SOURCE_AUTHORITY"
    elif current["status_porcelain"]:
        status = "STAGE_X_X1R_T1D0R2_HOLD_SOURCE_BINDING"
    else:
        status = "STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_PASS"
    return {
        "schema": "STAGE_X_X1R_T1D0R2_RUNTIME_AUTHORITY_AUDIT_V1",
        "status": status,
        "runtime_source_pre_evidence": current,
        "student_source_forensic": forensic,
        "student_replay_parity": replay,
        "success_horizon_authority": success,
        "parent_seed_invariance": parent,
        "errors": errors,
        "counters": replay.get("counters", counters()),
        "authorization": config["authorization"],
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "counters_zero_except_allowed_historical_replay": True},
    }


def write_evidence(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    receipt = load_json(Path(config["historical_t1_receipt"]["path"]))
    runtime = runtime_audit(config, repo)
    forensic = runtime["student_source_forensic"]
    replay = runtime["student_replay_parity"]
    success = runtime["success_horizon_authority"]
    parent = runtime["parent_seed_invariance"]
    report_paths = {
        "student_source": repo / "reports/STAGE_X_X1R_T1D0R2_STUDENT_SOURCE_FORENSIC_V1.json",
        "student_replay": repo / "reports/STAGE_X_X1R_T1D0R2_STUDENT_REPLAY_PARITY_V1.json",
        "success_horizon": repo / "reports/STAGE_X_X1R_T1D0R2_SUCCESS_HORIZON_AUTHORITY_V1.json",
        "parent_seed": repo / "reports/STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1.json",
        "runtime": repo / "reports/STAGE_X_X1R_T1D0R2_RUNTIME_AUTHORITY_AUDIT_V1.json",
    }
    write_json(report_paths["student_source"], forensic)
    write_json(report_paths["student_replay"], replay)
    write_json(report_paths["success_horizon"], success)
    write_json(report_paths["parent_seed"], parent)
    write_json(report_paths["runtime"], runtime)

    handoff = repo / "docs/handoffs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_HANDOFF_20260818.md"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(
        f"""# STAGE X X1R T1-D0R2 — Clean Runtime Authority Closure\n\n"
        f"Status: `{runtime['status']}`\n\n"
        f"Runtime source before evidence: `{runtime['runtime_source_pre_evidence']['head']}` / `{runtime['runtime_source_pre_evidence']['tree']}`.\n\n"
        "This is a static/CPU historical-replay authority audit only. It did not load OpenVLA, use a GPU, reset a simulator, call an environment step, materialize a fresh parent, run PGD, read V_phys, read Eval160, or read protected evaluation.\n\n"
        "The historical Student training source remains `NOT_IDENTIFIABLE`: T1 handoff, T1 receipt runtime identity, and current server file are distinct provenance statements. The prospective implementation is the tracked PR127 source bound by raw bytes and Git blob.\n\n"
        f"Student replay parity: `{replay['status']}`; sealed per-step reference available: `{replay.get('sealed_per_step_reference_available')}`. The T1-C receipt is summary-only, so deterministic repeat/prefix checks are diagnostic and cannot be promoted to historical per-step parity.\n\n"
        f"Success/horizon authority: `{success['status']}`. The canonical evaluator is the immutable upstream OpenVLA LIBERO evaluator; `done` is consumed after `env.step`, and LIBERO's domain step derives it from `_check_success()`. Policy horizons are 520/300/280/220 for L10/goal/object/spatial; the ten dummy wait steps are outside the policy-decision horizon.\n\n"
        f"D0R1 population/seed invariance: `{parent['status']}`; 1200 G10 -> 990 exclusion union -> 210 fresh -> 40 nominal cells -> 39 executable parents; missing cell `libero_goal/task_01`; replacement false.\n\n"
        "Authorization remains closed: `openvla_model_inference_authorized=false`, `clean_parent_materialization_authorized=false`, `env_step_authorized=false`, `pgd_authorized=false`, `physical_intervention_authorized=false`, `attack_outcome_authorized=false`, `protected_authorized=false`. Next gate remains `CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED`.\n\n"
        "Frozen scientific claim: `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`.\n""",
        encoding="utf-8",
    )
    tracked = [
        "configs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_V1.json",
        "src/gripper_attack/stage_x_x1r_d1_clean_runtime_contract.py",
        "scripts/stage_x/audit_stage_x1r_t1d0r2_clean_runtime_authority.py",
        "tests/stage_x/test_stage_x1r_t1d0r2_clean_runtime_authority.py",
        "docs/handoffs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_HANDOFF_20260818.md",
        *[str(path.relative_to(repo)).replace("\\", "/") for path in report_paths.values()],
    ]
    seal = {
        "schema": "STAGE_X_X1R_T1D0R2_ROOT_SEAL_V1",
        "status": runtime["status"],
        "reviewed_source": config["reviewed_source"],
        "runtime_source_pre_evidence": runtime["runtime_source_pre_evidence"],
        "artifact_sha256": {path: sha256_file(repo / path) for path in tracked},
        "student_source_forensic": forensic,
        "student_replay_parity": replay,
        "success_horizon_authority": success,
        "parent_seed_invariance": parent,
        "runtime_authority_audit": runtime,
        "authorization": config["authorization"],
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "counters": runtime["counters"]},
        "final_branch_seal": {"commit": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF", "tree": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF"},
    }
    seal_path = repo / "reports/STAGE_X_X1R_T1D0R2_ROOT_SEAL.json"
    seal_sha_path = repo / "reports/STAGE_X_X1R_T1D0R2_ROOT_SEAL.sha256"
    sums_path = repo / "reports/STAGE_X_X1R_T1D0R2_SHA256SUMS.txt"
    write_json(seal_path, seal)
    sums = {path: sha256_file(repo / path) for path in tracked + ["reports/STAGE_X_X1R_T1D0R2_ROOT_SEAL.json"]}
    sums_path.write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(sums.items())), encoding="utf-8")
    seal_sha_path.write_text(f"{sha256_file(seal_path)}  {seal_path.name}\n", encoding="utf-8")
    return {"status": runtime["status"], "runtime": runtime, "reports": {key: str(path) for key, path in report_paths.items()}, "seal": str(seal_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_V1.json")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    config = load_json(args.config.resolve())
    if repo != ROOT.resolve():
        raise SystemExit("repo-root must be the reviewed worktree")
    result = write_evidence(config, repo) if args.write_evidence else {"status": runtime_audit(config, repo)["status"]}
    print(json.dumps({"status": result["status"], "reports": result.get("reports", {}), "seal": result.get("seal")}, sort_keys=True))
    return 0 if result["status"] == "STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
