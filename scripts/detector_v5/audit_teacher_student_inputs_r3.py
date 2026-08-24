#!/usr/bin/env python3
"""Fail-closed, read-only audit of the frozen Teacher/Student inputs for R3."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_d8_3b_run import audit_run, verify_sha256_seal
from audit_r3_contact_input import verify_seal
from run_detector_clean_freeze import load_cache, load_oof


STAGE1_COMMIT = "990befe126bcce4bcc95c965f3677eda32a2e8e9"
STAGE1_TREE = "268b97da25c398120fd42ccec050f945b5a59756"
STAGE2_COMMIT = "f14415b104501df2a5cf7b35e8965966866eea9e"
H1_COMMIT = "9dd324ad70a9be17548f72437da8454356abfd28"
H1_TREE = "0333510e291f8ec0c5b8738136019f30c5de17aa"
SOURCE_SNAPSHOT = "99648bdee45cde6411159f6d6586b8b7e46b626ea000f07a6cff0b38251efdbd"
CORE_BLOB = "bd4c505ada3696913b061f3132b7ea67622b3cad"
FEATURE_SCHEMA_BLOB = "3f6c62dd7b263d4d1faf42e6c6eae5e7d52196ab"
CACHE_A_SEAL = "929a0a666a867c93094b13752f4c2f848640bbedb2dadc9a20d834f3ee8b6814"
CACHE_B_SEAL = "4138127d5ba8f20f9735e09decb15d6448bb3c34dbdbd73beea92e5f169e7ae0"
COMPARATOR_SEAL = "5e63d0db69a3d5912c91ced40500e09f3d7dfdb4623ec097ecc4bc71cede7037"
P5_SEAL = "f4ea991c9f62de64eeb4a093a9fd956cd13017974d63c0391c3ac759a4aae817"
H1_REVIEW_SEAL = "5b11b527bb4d8eac9f10c014efe8e557dc65adcf5eda04936600e603d62de5c3"
TEACHER_SEAL = "eeedf304fa18feb277cf976dcb866a4e5b0a4a68161887e32fbb549150a29db3"
SIDECAR_SEAL = "3554d7c199fb9915992bdc83f5cb8b0cfb784479cad4fe901ad3b15ca9996e1d"
STAGE2_ROOT_SEAL = "f36ac1e18fca516fb8fae82e1290d034848ad572f6fad6908beb2d40b9bc9277"
CHECKPOINT_SHA = "ce7f03088d84a796d38fbdc107cea7f21bdb4808e35f7dc754e1b52e48bce1d4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--comparator-root", type=Path, required=True)
    parser.add_argument("--h1-review-root", type=Path, required=True)
    parser.add_argument("--p5-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema": "D8_TEACHER_STUDENT_INPUT_AUDIT_R3_V1",
        "status": "RUNNING",
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "stage1_retrained": False,
        "stage1_artifact_modified": False,
        "stage2_r2_artifact_modified": False,
        "thresholds_changed": False,
        "attack_informed_tuning": False,
    }
    checks: dict[str, bool] = {}
    try:
        repo = args.repo_root.resolve(strict=True)
        formal_root = args.formal_root.resolve(strict=True)
        stage2_root = args.stage2_root.resolve(strict=True)
        cache_a = args.cache_a.resolve(strict=True)
        cache_b = args.cache_b.resolve(strict=True)
        teacher_root = args.teacher_root.resolve(strict=True)
        sidecar_root = args.sidecar_root.resolve(strict=True)
        comparator_root = args.comparator_root.resolve(strict=True)
        review_root = args.h1_review_root.resolve(strict=True)
        p5_root = args.p5_root.resolve(strict=True)
        snapshot_path = args.source_snapshot.resolve(strict=True)

        head = git_value(repo, "rev-parse", "HEAD")
        tree = git_value(repo, "rev-parse", "HEAD^{tree}")
        checks["r3_base_commit_is_ancestor"] = git_value(repo, "merge-base", "402ba4e1fb46c1d639e98696607c9647adc38bb8", "HEAD") == "402ba4e1fb46c1d639e98696607c9647adc38bb8"
        checks["r3_core_blob"] = git_value(repo, "rev-parse", "HEAD:scripts/detector_v5/d8_train_core.py") == CORE_BLOB
        checks["r3_feature_schema_blob"] = git_value(repo, "rev-parse", "HEAD:configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json") == FEATURE_SCHEMA_BLOB
        schema = read_json(repo / "configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json")
        checks["feature_schema_25d_causal"] = (
            schema.get("dimensions") == 25
            and len(schema.get("features", [])) == 25
            and schema.get("causal_only") is True
            and schema.get("future_fields") == 0
            and schema.get("teacher_label_fields") == 0
        )

        formal_seal = verify_sha256_seal(formal_root)
        formal_seal_sidecar_file_sha = sha256_file(formal_root / "SHA256SUMS.sha256")
        formal_audit = audit_run(formal_root, write_artifacts=False)
        formal_receipt = read_json(formal_root / "EXECUTION_RECEIPT.json")
        formal_manifest = read_json(formal_root / "JOB_MANIFEST.json")
        formal_provenance = formal_receipt.get("provenance", {})
        formal_jobs = formal_manifest.get("jobs", [])
        checks["stage1_formal_root_seal"] = formal_seal["sha256sums_sha256"] == "7b2a5f7a3be45d57219d434ef2137bb4bc679492f9093baa5733557d8f6712c6"
        checks["stage1_root_seal_sidecar_hash"] = formal_seal_sidecar_file_sha == "4e4f6a5042e458e53d09bf2140f45e724d13c4c58f1bcc9eb19c3448b01a4f81"
        checks["stage1_auditor_pass"] = formal_audit.get("verdict") == "PASS" and formal_audit.get("launcher_verdict") == "PASS"
        checks["stage1_50_job_closure"] = len(formal_jobs) == 50 and all(job.get("status") == "COMPLETED" for job in formal_jobs)
        checks["stage1_10_seed_5_fold_closure"] = sorted({int(job.get("seed")) for job in formal_jobs}) == list(range(20260720, 20260730)) and sorted({int(job.get("fold")) for job in formal_jobs}) == [0, 1, 2, 3, 4]
        checks["stage1_source_binding"] = formal_provenance.get("source_commit") == STAGE1_COMMIT and formal_provenance.get("source_tree") == STAGE1_TREE
        checks["stage1_boundary_zero"] = all(formal_receipt.get(key) == 0 for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts"))

        cache_rows, cache_manifest, cache_seal = load_cache(cache_a, CACHE_A_SEAL)
        checks["cache_a_seal"] = cache_seal["sha256sums_sha256"] == CACHE_A_SEAL
        checks["cache_a_feature_dim"] = cache_manifest.get("feature_dim") == 25
        checks["oof_prediction_identity_closure"] = False
        _, oof_meta = load_oof(formal_root, cache_rows, STAGE1_COMMIT, STAGE1_TREE)
        checks["oof_prediction_identity_closure"] = oof_meta.get("prediction_file_count") == 50 and oof_meta.get("effective_identity_count") == sum(bool(row.get("effective_mask")) for row in cache_rows)

        sealed_inputs = {
            "cache_b": (cache_b, CACHE_B_SEAL),
            "teacher": (teacher_root, TEACHER_SEAL),
            "sidecar": (sidecar_root, SIDECAR_SEAL),
            "comparator": (comparator_root, COMPARATOR_SEAL),
            "h1_review": (review_root, H1_REVIEW_SEAL),
            "p5": (p5_root, P5_SEAL),
        }
        sealed_results: dict[str, Any] = {}
        for name, (path, expected) in sealed_inputs.items():
            seal = verify_seal(path)
            sealed_results[name] = {"root": str(path), "seal": seal["sha256sums_sha256"], "expected": expected}
            checks[f"{name}_seal"] = seal["sha256sums_sha256"].lower() == expected.lower()

        snapshot = read_json(snapshot_path)
        checks["h1_source_snapshot"] = snapshot.get("source_snapshot_sha256") == SOURCE_SNAPSHOT and snapshot.get("executable_source_commit") == H1_COMMIT and snapshot.get("executable_source_tree") == H1_TREE

        stage2_seal = verify_sha256_seal(stage2_root)
        stage2_receipt = read_json(stage2_root / "DETECTOR_FREEZE_RECEIPT_R2.json")
        stage2_checkpoint = stage2_root / "FINAL_DETECTOR_CHECKPOINT.pt"
        checkpoint = torch.load(str(stage2_checkpoint), map_location="cpu", weights_only=False)
        stage2_source_tree = stage2_receipt.get("source_tree")
        checks["stage2_root_seal"] = stage2_seal["sha256sums_sha256"] == STAGE2_ROOT_SEAL
        checks["stage2_source_binding"] = stage2_receipt.get("source_commit") == STAGE2_COMMIT and isinstance(stage2_source_tree, str) and len(stage2_source_tree) == 40
        checks["stage2_checkpoint_binding"] = sha256_file(stage2_checkpoint) == CHECKPOINT_SHA and stage2_receipt.get("checkpoint_sha256") == CHECKPOINT_SHA
        checks["stage2_checkpoint_25d"] = checkpoint.get("schema") == "D8_3B_CHECKPOINT_V2" and checkpoint.get("normalization", {}).get("schema") == "D8_NORMALIZATION_V2" and checkpoint.get("normalization", {}).get("feature_dim") == 25
        checks["stage2_boundary_zero"] = all(stage2_receipt.get(key) == 0 for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts"))

        report.update({
            "r3_producer": {"commit": head, "tree": tree, "d8_train_core_blob": CORE_BLOB, "feature_schema_blob": FEATURE_SCHEMA_BLOB},
            "oof_producer": {"commit": STAGE1_COMMIT, "tree": STAGE1_TREE, "formal_root": str(formal_root), "formal_root_seal": formal_seal["sha256sums_sha256"], "d8_train_core_blob": git_value(repo, "rev-parse", f"{STAGE1_COMMIT}:scripts/detector_v5/d8_train_core.py"), "feature_schema_blob": git_value(repo, "rev-parse", f"{STAGE1_COMMIT}:configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"), "runtime_train_core_sha256": formal_provenance.get("train_core_sha256"), "runtime_feature_schema_sha256": formal_provenance.get("feature_schema_sha256")},
            "stage2_r2_producer": {"commit": stage2_receipt.get("source_commit"), "tree": stage2_source_tree, "root": str(stage2_root), "root_seal": stage2_seal["sha256sums_sha256"], "d8_train_core_blob": git_value(repo, "rev-parse", f"{STAGE2_COMMIT}:scripts/detector_v5/d8_train_core.py"), "feature_schema_blob": git_value(repo, "rev-parse", f"{STAGE2_COMMIT}:configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"), "checkpoint_sha256": CHECKPOINT_SHA},
            "h1_lineage": {"commit": H1_COMMIT, "tree": H1_TREE, "source_snapshot_sha256": SOURCE_SNAPSHOT, "cache_a": CACHE_A_SEAL, "cache_b": CACHE_B_SEAL, "comparator": COMPARATOR_SEAL, "p5": P5_SEAL, "h1_review": H1_REVIEW_SEAL, "teacher": TEACHER_SEAL, "sidecar": SIDECAR_SEAL},
            "sealed_inputs": sealed_results,
            "stage1": {"formal_audit": formal_audit, "formal_receipt": {"schema": formal_receipt.get("schema"), "verdict": formal_receipt.get("verdict"), "counters": {key: formal_receipt.get(key) for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts")}}, "formal_root_seal": formal_seal, "formal_root_seal_sidecar_file_sha256": formal_seal_sidecar_file_sha},
            "stage2": {"receipt": {"schema": stage2_receipt.get("schema"), "status": stage2_receipt.get("status"), "s1_verdict": stage2_receipt.get("s1_verdict"), "s2_verdict": stage2_receipt.get("s2_verdict")}, "root_seal": stage2_seal},
            "source_snapshot_path": str(snapshot_path),
            "checks": checks,
        })
        report["status"] = "PASS" if all(checks.values()) else "FAIL"
        report["input_audit_verdict"] = report["status"]
    except Exception as exc:
        report.update({"status": "FAIL", "input_audit_verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}", "checks": checks})
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "TEACHER_STUDENT_INPUT_AUDIT_R3.json", report)
    print(json.dumps({"status": report["status"], "output": str(output_root / "TEACHER_STUDENT_INPUT_AUDIT_R3.json")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
