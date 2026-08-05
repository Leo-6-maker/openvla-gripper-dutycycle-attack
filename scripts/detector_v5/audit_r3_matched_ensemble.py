#!/usr/bin/env python3
"""Audit the R3-A ensemble and run the frozen R2 clean transfer Gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "detector_v5", ROOT / "scripts" / "fec"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_d8_3b_run import verify_sha256_seal
from audit_r3_contact_input import sha256_file, verify_seal
from d8_train_core import apply_normalization, create_model
from run_detector_clean_freeze import cache_effective_rows, load_cache, load_clean_event_groups, load_oof, seal_directory, sha256_json, utc_now
from run_detector_stage2_r2 import detailed_candidate_metrics, build_aggregate_rows
from audit_stage2_r2_discrepancy import correlation, episode_rows, rank_values, score_summary, suite_summary

SEEDS = tuple(range(20260720, 20260730))
STAGE1_COMMIT = "990befe126bcce4bcc95c965f3677eda32a2e8e9"
STAGE1_TREE = "268b97da25c398120fd42ccec050f945b5a59756"
CACHE_A_SEAL = "929a0a666a867c93094b13752f4c2f848640bbedb2dadc9a20d834f3ee8b6814"
R2_ROOT_SEAL = "f36ac1e18fca516fb8fae82e1290d034848ad572f6fad6908beb2d40b9bc9277"
R2_SCHEDULER = {"threshold": 0.43356089500710393, "persistence": 5, "hysteresis": 1.0, "cooldown": 0}
CORE_BLOB = "bd4c505ada3696913b061f3132b7ea67622b3cad"
FEATURE_SCHEMA_BLOB = "3f6c62dd7b263d4d1faf42e6c6eae5e7d52196ab"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(metrics.get(key) is not None and math.isfinite(float(metrics[key])) for key in ("false_onset_episode_rate", "negative_active_step_rate", "active_overlap_event_recall", "median_first_activation_delay"))


def load_member(path: Path, expected: dict[str, Any], norm_sha: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "D8_R3A_FULLDATA_CHECKPOINT_V1":
        raise RuntimeError(f"unexpected checkpoint schema: {path.name}")
    if int(checkpoint.get("seed")) != int(expected["seed"]):
        raise RuntimeError(f"seed binding mismatch: {path.name}")
    for key, value in (("config", "B3"), ("epochs", 100), ("architecture", "25->32->16->1"), ("optimizer", "Adam"), ("learning_rate", 1e-3), ("weight_normalization", "mean_to_one"), ("cache_a_seal", CACHE_A_SEAL), ("d8_train_core_blob", CORE_BLOB), ("feature_schema_blob", FEATURE_SCHEMA_BLOB)):
        actual = checkpoint.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=0.0):
                raise RuntimeError(f"checkpoint {key} mismatch: {path.name}")
        elif actual != value:
            raise RuntimeError(f"checkpoint {key} mismatch: {path.name}")
    if checkpoint.get("r3_source_commit") != expected["r3_source_commit"] or checkpoint.get("r3_source_tree") != expected["r3_source_tree"]:
        raise RuntimeError(f"checkpoint source mismatch: {path.name}")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("schema") != "D8_NORMALIZATION_V2" or normalization.get("feature_dim") != 25:
        raise RuntimeError(f"checkpoint normalization mismatch: {path.name}")
    if checkpoint.get("normalization_sha256") != norm_sha or sha256_json(normalization) != norm_sha:
        raise RuntimeError(f"normalization SHA mismatch: {path.name}")
    model = create_model(int(expected["seed"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError(f"non-finite model state: {path.name}")
    return model, checkpoint


def compute_scores(members: list[dict[str, Any]], effective: list[dict[str, Any]], norm_sha: str, expected: dict[str, Any]) -> np.ndarray:
    batch_size = 8192
    features = np.asarray([row["features_25d_raw"] for row in effective], dtype=np.float32)
    if features.shape != (len(effective), 25) or not np.isfinite(features).all():
        raise RuntimeError("effective Cache A feature matrix is not finite 25D")
    outputs: list[np.ndarray] = []
    for member in members:
        model, checkpoint = load_member(Path(member["path"]), member, norm_sha)
        normalization = checkpoint["normalization"]
        values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.from_numpy(features[start : start + batch_size])
                values.append(model(apply_normalization(batch, normalization)).numpy())
        logits = np.concatenate(values).astype(np.float64, copy=False)
        if logits.shape != (len(effective),) or not np.isfinite(logits).all():
            raise RuntimeError(f"member logits malformed/non-finite: {member['seed']}")
        outputs.append(logits)
    result = np.stack(outputs, axis=1)
    if result.shape != (len(effective), len(SEEDS)) or not np.isfinite(result).all():
        raise RuntimeError("ensemble score matrix closure/finite check failed")
    return result


def save_npz(path: Path, keys: list[str], scores: np.ndarray, mean_scores: np.ndarray, targets: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, identity=np.asarray(keys), scores=scores.astype(np.float32), mean_score=mean_scores.astype(np.float32), target=targets.astype(np.float32))
    temporary.replace(path)


def update_goal(goal_root: Path, status: str, decision: str, audit_root: Path, audit_sha: str, stage3_authorized: bool) -> None:
    status_path = goal_root / "GOAL_STATUS_R3.json"
    value = read_json(status_path) if status_path.exists() else {"schema": "TEACHER_STUDENT_DETECTOR_GOAL_STATUS_R3_V1"}
    value.update({"status": status, "r3a": status, "stage3a_authorized": stage3_authorized, "updated_utc": utc_now(), "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0, "thresholds_changed": False, "attack_informed_tuning": False})
    if status.endswith("FAIL"):
        value["r3b"] = "AUTHORIZED_PENDING"
    atomic_json(status_path, value)
    ledger_path = goal_root / "DECISION_LEDGER_R3.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"schema": "TEACHER_STUDENT_DECISION_LEDGER_R3_V1", "decisions": []}
    ledger.setdefault("decisions", []).append({"decision_id": "R3A-TRANSFER", "stage": "R3-A clean transfer", "input_artifacts": [str(audit_root)], "preregistered_rule": "frozen R2 scheduler; false onset <=0.10 and negative active <=0.05", "observed_result": status, "decision": decision, "next_stage": "Stage3A shadow" if stage3_authorized else "R3-B clean holdout", "thresholds_changed": False, "attack_informed_tuning": False, "updated_utc": utc_now()})
    atomic_json(ledger_path, ledger)
    progress = f"# Teacher–Student Detector Goal R3\n\n- Stage T: PASS\n- R3-A: {status}\n- R3-B: {'pending fallback' if not stage3_authorized else 'not executed'}\n- Stage3A / Eval160 / attack: {'authorized only after canary' if stage3_authorized else 'not authorized'}\n\nBoundary counters: `Eval160=0`, `protected_eval=0`, `attack_rollouts=0`.\n"
    atomic_bytes(goal_root / "NIGHTLY_PROGRESS_R3.md", progress.encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--stage-t-audit", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-ensemble-source-commit", required=True)
    parser.add_argument("--expected-ensemble-source-tree", required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    report: dict[str, Any] = {"schema": "R3A_MATCHED_ENSEMBLE_TRANSFER_AUDIT_V1", "status": "RUNNING", "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0}
    try:
        repo = args.repo_root.resolve(strict=True)
        commit, tree = git_value(repo, "rev-parse", "HEAD"), git_value(repo, "rev-parse", "HEAD^{tree}")
        if commit != args.expected_source_commit or tree != args.expected_source_tree:
            raise RuntimeError("R3 source commit/tree mismatch")
        stage_t = read_json(args.stage_t_audit.resolve(strict=True))
        if stage_t.get("status") != "PASS":
            raise RuntimeError("Stage T is not PASS")
        ensemble_root = args.ensemble_root.resolve(strict=True)
        ensemble_seal = verify_sha256_seal(ensemble_root)
        manifest_path = ensemble_root / "R3A_ENSEMBLE_MANIFEST.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS_10_OF_10_COMPLETED" or sorted(manifest.get("seeds", [])) != list(SEEDS) or len(manifest.get("members", [])) != 10:
            raise RuntimeError("R3-A ensemble manifest closure failed")
        if manifest.get("provenance", {}).get("r3_source_commit") != args.expected_ensemble_source_commit or manifest.get("provenance", {}).get("r3_source_tree") != args.expected_ensemble_source_tree:
            raise RuntimeError("R3-A manifest source binding mismatch")
        plan_path = args.goal_root.resolve(strict=True) / "R3A_MATCHED_ENSEMBLE_PLAN.json"
        plan = read_json(plan_path)
        if plan.get("status") != "PREREGISTERED" or plan.get("r3_source", {}).get("commit") != args.expected_ensemble_source_commit or plan.get("r3_source", {}).get("tree") != args.expected_ensemble_source_tree or plan.get("r2_scheduler", {}).get("scheduler") != R2_SCHEDULER:
            raise RuntimeError("R3-A plan/scheduler binding failed")
        if manifest.get("plan_sha256") != sha256_file(plan_path):
            raise RuntimeError("R3-A manifest/plan hash mismatch")
        members: list[dict[str, Any]] = []
        norm_sha = str(manifest.get("normalization_sha256"))
        for member in manifest["members"]:
            checkpoint_path = ensemble_root / "R3A_CHECKPOINTS" / str(member["checkpoint"])
            checkpoint_sha = sha256_file(checkpoint_path)
            if checkpoint_sha != member.get("checkpoint_sha256"):
                raise RuntimeError(f"checkpoint manifest SHA mismatch: {checkpoint_path.name}")
            sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256").read_text(encoding="utf-8").strip()
            if sidecar != f"{checkpoint_sha}  {checkpoint_path.name}":
                raise RuntimeError(f"checkpoint sidecar mismatch: {checkpoint_path.name}")
            members.append({"seed": int(member["seed"]), "path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha, "normalization_sha256": norm_sha, "r3_source_commit": args.expected_ensemble_source_commit, "r3_source_tree": args.expected_ensemble_source_tree})
        rows, cache_manifest, cache_seal = load_cache(args.cache_a.resolve(strict=True), CACHE_A_SEAL)
        effective = sorted(cache_effective_rows(rows), key=lambda row: (str(row["episode_id"]), int(row["step"])))
        keys = [f"{row['episode_id']}::{int(row['step'])}" for row in effective]
        targets = np.asarray([float(row["physical_target"]) for row in effective], dtype=np.float64)
        scores = compute_scores(members, effective, norm_sha, {"r3_source_commit": args.expected_ensemble_source_commit, "r3_source_tree": args.expected_ensemble_source_tree})
        scores_again = compute_scores(members, effective, norm_sha, {"r3_source_commit": args.expected_ensemble_source_commit, "r3_source_tree": args.expected_ensemble_source_tree})
        score_deterministic = bool(np.array_equal(scores, scores_again))
        mean_scores = np.mean(scores, axis=1, dtype=np.float64)
        if not score_deterministic or not np.isfinite(mean_scores).all():
            raise RuntimeError("R3-A score determinism/finite Gate failed")
        oof_scores, oof_meta = load_oof(args.formal_root.resolve(strict=True), rows, STAGE1_COMMIT, STAGE1_TREE)
        oof_rows = build_aggregate_rows(rows, oof_scores)
        oof_rows = sorted(oof_rows, key=lambda row: (str(row["episode_id"]), int(row["step"])))
        oof_mean = np.asarray([float(row["score"]) for row in oof_rows], dtype=np.float64)
        if [f"{row['episode_id']}::{int(row['step'])}" for row in oof_rows] != keys:
            raise RuntimeError("OOF/R3 identity order mismatch")
        ensemble_rows = [dict(row, target=float(row["physical_target"]), score=float(score)) for row, score in zip(effective, mean_scores)]
        event_groups, event_binding = load_clean_event_groups(args.sidecar_root.resolve(strict=True), args.teacher_root.resolve(strict=True), rows)
        r2_root = args.r2_root.resolve(strict=True)
        if verify_sha256_seal(r2_root)["sha256sums_sha256"].lower() != R2_ROOT_SEAL:
            raise RuntimeError("R2 root seal mismatch")
        r2_receipt_path = r2_root / "DETECTOR_FREEZE_RECEIPT_R2.json"
        r2_receipt = read_json(r2_receipt_path)
        candidate = {key: r2_receipt["scheduler"][key] for key in R2_SCHEDULER}
        if any(float(candidate[key]) != float(R2_SCHEDULER[key]) for key in R2_SCHEDULER):
            raise RuntimeError("R2 scheduler changed")
        oof_metrics, oof_traces = detailed_candidate_metrics(oof_rows, event_groups, candidate)
        ensemble_metrics, ensemble_traces = detailed_candidate_metrics(ensemble_rows, event_groups, candidate)
        _, ensemble_traces_again = detailed_candidate_metrics(ensemble_rows, event_groups, candidate)
        scheduler_deterministic = json.dumps(ensemble_traces, sort_keys=True, separators=(",", ":")) == json.dumps(ensemble_traces_again, sort_keys=True, separators=(",", ":"))
        suites = sorted({str(row["episode_id"]).split("/", 1)[0] for row in effective})
        suite_coverage = bool(suites) and all(any(event_groups.get(str(row["episode_id"]), []) for row in effective if str(row["episode_id"]).startswith(f"{suite}/")) for suite in suites)
        gate = {"false_onset_episode_rate": float(ensemble_metrics["false_onset_episode_rate"]) <= 0.10, "negative_active_step_rate": float(ensemble_metrics["negative_active_step_rate"]) <= 0.05, "all_scores_finite": bool(np.isfinite(scores).all()), "all_suites_covered": suite_coverage, "identity_closure": len(keys) == len(set(keys)) == scores.shape[0], "score_deterministic": score_deterministic, "scheduler_deterministic": scheduler_deterministic, "finite_metrics": finite_metrics(ensemble_metrics)}
        gate_pass = all(gate.values())
        discrepancy_episodes = episode_rows(oof_traces, ensemble_traces, event_groups, oof_rows, ensemble_rows, float(candidate["threshold"]))
        score_discrepancy = {"pearson": correlation(oof_mean, mean_scores), "spearman": correlation(rank_values(oof_mean), rank_values(mean_scores)), "step_count": len(keys), "oof_mean_logit": {"mean": float(np.mean(oof_mean)), "std_ddof1": float(np.std(oof_mean, ddof=1))}, "r3a_mean_logit": {"mean": float(np.mean(mean_scores)), "std_ddof1": float(np.std(mean_scores, ddof=1))}}
        discrepancy = {"schema": "R3A_TRANSFER_DISCREPANCY_VS_OOF_V1", "scheduler": candidate, "episode_count": len(discrepancy_episodes), "category_counts": dict(sorted(Counter(row["category"] for row in discrepancy_episodes).items())), "oof_false_onset_episode_count": sum(row["oof"]["false_onset"] for row in discrepancy_episodes), "r3a_false_onset_episode_count": sum(row["final"]["false_onset"] for row in discrepancy_episodes), "new_r3a_false_onset_episode_count": sum(row["category"] == "OOF_TRUE_FINAL_FALSE" for row in discrepancy_episodes), "oof_only_false_onset_episode_count": sum(row["category"] == "OOF_FALSE_FINAL_TRUE" for row in discrepancy_episodes), "score_correlation": score_discrepancy, "suite_summary": suite_summary(discrepancy_episodes), "episodes": discrepancy_episodes}
        distribution = {"schema": "R3A_SCORE_DISTRIBUTION_V1", "scheduler": candidate, "oof": score_summary(oof_mean, targets, float(candidate["threshold"])), "r3a_mean": score_summary(mean_scores, targets, float(candidate["threshold"])), "members": {str(seed): score_summary(scores[:, index], targets, float(candidate["threshold"])) for index, seed in enumerate(SEEDS)}, "identity_digest": sha256_json(keys), "ensemble_score_sha256": hashlib.sha256(mean_scores.astype(np.float32).tobytes()).hexdigest(), "member_score_sha256": {str(seed): hashlib.sha256(scores[:, index].astype(np.float32).tobytes()).hexdigest() for index, seed in enumerate(SEEDS)}}
        output_root.mkdir(parents=True)
        save_npz(output_root / "R3A_CLEAN_SCORES.npz", keys, scores, mean_scores, targets)
        atomic_json(output_root / "R3A_SCORE_DISTRIBUTION.json", distribution)
        atomic_json(output_root / "R3A_TRANSFER_DISCREPANCY_VS_OOF.json", discrepancy)
        atomic_json(output_root / "R3A_TRANSFER_AUDIT.json", {"schema": "R3A_MATCHED_ENSEMBLE_TRANSFER_AUDIT_V1", "status": "R3A_MATCHED_ENSEMBLE_TRANSFER_PASS" if gate_pass else "R3A_MATCHED_ENSEMBLE_TRANSFER_FAIL", "ensemble_root": str(ensemble_root), "ensemble_root_seal": ensemble_seal, "ensemble_manifest": str(manifest_path), "ensemble_manifest_sha256": sha256_file(manifest_path), "r2_root": str(r2_root), "r2_root_seal": R2_ROOT_SEAL, "r2_scheduler_receipt_sha256": sha256_file(r2_receipt_path), "scheduler": candidate, "oof_metrics": oof_metrics, "r3a_metrics": ensemble_metrics, "gate": gate, "event_binding": event_binding, "cache_manifest_schema": cache_manifest.get("schema"), "cache_seal": cache_seal, "oof_meta": {key: value for key, value in oof_meta.items() if key != "formal_audit"}, "score_deterministic": score_deterministic, "scheduler_deterministic": scheduler_deterministic, "producer": {"audit_commit": commit, "audit_tree": tree, "ensemble_commit": args.expected_ensemble_source_commit, "ensemble_tree": args.expected_ensemble_source_tree, "d8_train_core_blob": CORE_BLOB, "feature_schema_blob": FEATURE_SCHEMA_BLOB}, "thresholds_changed": False, "attack_informed_tuning": False, "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0})
        seal = seal_directory(output_root)
        atomic_json(output_root / "R3A_TRANSFER_RECEIPT.json", {"schema": "R3A_TRANSFER_RECEIPT_V1", "status": "PASS" if gate_pass else "FAIL", "audit": "R3A_TRANSFER_AUDIT.json", "audit_sha256": sha256_file(output_root / "R3A_TRANSFER_AUDIT.json"), "payload_seal_before_receipt": seal, "stage3a_authorized": gate_pass, "active_guard_authorized": False, "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0})
        # Re-seal after the receipt is added.
        seal = seal_directory(output_root)
        audit_sha = sha256_file(output_root / "R3A_TRANSFER_AUDIT.json")
        update_goal(args.goal_root.resolve(strict=True), "R3A_MATCHED_ENSEMBLE_TRANSFER_PASS" if gate_pass else "R3A_MATCHED_ENSEMBLE_TRANSFER_FAIL", "proceed to Stage3A A0 canary" if gate_pass else "enter preregistered R3-B", output_root, audit_sha, gate_pass)
        report["status"] = "R3A_MATCHED_ENSEMBLE_TRANSFER_PASS" if gate_pass else "R3A_MATCHED_ENSEMBLE_TRANSFER_FAIL"
        report.update({"output_root": str(output_root), "root_seal": seal, "gate": gate, "metrics": ensemble_metrics, "oof_metrics": oof_metrics, "r3a_transfer_audit_sha256": audit_sha, "stage3a_authorized": gate_pass})
        print(json.dumps({"status": report["status"], "output_root": str(output_root), "seal": seal}, sort_keys=True))
        return 0 if gate_pass else 20
    except Exception as exc:
        report.update({"status": "R3A_ENGINEERING_FAIL", "error": f"{type(exc).__name__}: {exc}"})
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / "R3A_TRANSFER_AUDIT.json", report)
        print(json.dumps({"status": report["status"], "output_root": str(output_root)}, sort_keys=True))
        return 21


if __name__ == "__main__":
    raise SystemExit(main())
