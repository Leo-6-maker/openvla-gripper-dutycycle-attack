"""Independent fail-closed H1-R9 audit for Detector-v3 D8.

The auditor does not build caches or train a model. It consumes two sealed
independent caches, one sealed P5 artifact and an external SOURCE_SNAPSHOT_V2.
It emits COMMIT_SAFE only when every machine-checkable H1 contract closes.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file
from compare_d8_25d_caches import compare_caches
from d8_source_contract import (
    REVIEW_REQUIRED_SOURCE_FILES,
    load_and_validate_source_snapshot,
    verify_sha256_manifest,
)
from gripper_attack.seal_utils import rename_noreplace

FORMAL_PYTHON_FILES = (
    "scripts/detector_v5/build_d8_25d_cache.py",
    "scripts/detector_v5/d8_source_contract.py",
    "scripts/detector_v5/d8_train_core.py",
    "scripts/detector_v5/run_d8_p5_25d_gpu_smoke.py",
    "scripts/detector_v5/compare_d8_25d_caches.py",
    "scripts/detector_v5/audit_d8_h1_r9.py",
)
REQUIRED_P5_GATES = {
    "source_snapshot_contract",
    "cache_source_binding",
    "cache_seal",
    "input_dim_25",
    "train_val_identity_disjoint",
    "norm_from_train_only",
    "effective_mask_contract",
    "no_privileged_keys",
    "finite_loss",
    "finite_logits",
    "finite_gradients",
    "grad_nonzero",
    "loss_decreases",
    "checkpoint_restore",
    "continuation_parity",
    "validation_completes",
    "val_loss_finite",
}


def _write_seal(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def _assert_free_files() -> list[str]:
    violations = []
    for rel in FORMAL_PYTHON_FILES:
        path = ROOT / rel
        tree = ast.parse(path.read_text("utf-8"), filename=rel)
        if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
            violations.append(rel)
    return violations


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def audit(
    cache_a: Path,
    cache_b: Path,
    p5_root: Path,
    source_snapshot: Path,
) -> dict:
    source = load_and_validate_source_snapshot(
        source_snapshot, ROOT, REVIEW_REQUIRED_SOURCE_FILES
    )
    cache_comparison = compare_caches(cache_a, cache_b)
    p5_seal = verify_sha256_manifest(p5_root, require_all_files_listed=True)
    p5_report = _load_json(p5_root / "P5_REPORT.json")
    p5_receipt = _load_json(p5_root / "EXECUTION_RECEIPT.json")
    access = _load_json(p5_root / "ACCESS_AUDIT.json")
    normalization = _load_json(p5_root / "NORMALIZATION.json")
    batch_schema = _load_json(p5_root / "BATCH_SCHEMA.json")
    cache_manifest_a = _load_json(cache_a / "CACHE_MANIFEST.json")
    cache_manifest_b = _load_json(cache_b / "CACHE_MANIFEST.json")

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    expected_binding = {
        "executable_source_commit": source["executable_source_commit"],
        "executable_source_tree": source["executable_source_tree"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
    }

    checks["source_snapshot_valid"] = True
    checks["cache_ab_canonical_identity"] = cache_comparison["status"] == "PASS"
    checks["cache_a_source_binding"] = all(
        cache_manifest_a.get("code_snapshot", {}).get(key) == value
        for key, value in expected_binding.items()
    )
    checks["cache_b_source_binding"] = all(
        cache_manifest_b.get("code_snapshot", {}).get(key) == value
        for key, value in expected_binding.items()
    )
    checks["cache_nonconsumer"] = all(
        manifest.get("consumer_eligible") is False
        and manifest.get("status") == "BUILT_PENDING_H1"
        for manifest in (cache_manifest_a, cache_manifest_b)
    )
    checks["identity_closure"] = all(
        manifest.get("identity_closure", {}).get("included") == 643
        and manifest.get("identity_closure", {}).get("fully_excluded") == 27
        and manifest.get("identity_closure", {}).get("excluded_by_category") == {"articulated_task": 27}
        for manifest in (cache_manifest_a, cache_manifest_b)
    )
    checks["step_closure"] = all(
        manifest.get("total_steps") == 196_483
        and manifest.get("effective_steps") == 179_674
        and manifest.get("total_episodes") == 670
        for manifest in (cache_manifest_a, cache_manifest_b)
    )

    p5_gates = p5_report.get("gates", {})
    checks["p5_schema"] = p5_report.get("schema") == "D8_P5_25D_GPU_SMOKE_V2"
    checks["p5_all_required_gates_present"] = set(p5_gates) == REQUIRED_P5_GATES
    checks["p5_all_gates_pass"] = bool(p5_report.get("all_gates_pass")) and all(p5_gates.values())
    checks["p5_status_nonconsumer"] = (
        p5_report.get("status") == "PASS_ENGINEERING_NONCONSUMABLE"
        and p5_report.get("consumer_eligible") is False
    )
    checks["p5_source_binding"] = p5_report.get("source_binding") == expected_binding
    checks["p5_receipt_source_binding"] = p5_receipt.get("source_binding") == expected_binding

    cache_seals = {
        cache_comparison["cache_a"]["package_seal"],
        cache_comparison["cache_b"]["package_seal"],
    }
    bound_cache_seal = p5_report.get("cache_binding", {}).get("cache_sha256sums_sha256")
    checks["p5_cache_binding"] = bound_cache_seal in cache_seals
    checks["p5_receipt_cache_binding"] = p5_receipt.get("cache_binding") == p5_report.get("cache_binding")
    if bound_cache_seal == cache_comparison["cache_a"]["package_seal"]:
        bound_cache_manifest = cache_a / "CACHE_MANIFEST.json"
    elif bound_cache_seal == cache_comparison["cache_b"]["package_seal"]:
        bound_cache_manifest = cache_b / "CACHE_MANIFEST.json"
    else:
        bound_cache_manifest = None
    checks["p5_cache_manifest_binding"] = bool(
        bound_cache_manifest
        and p5_report.get("cache_binding", {}).get("cache_manifest_sha256")
        == sha256_file(bound_cache_manifest)
    )
    checks["p5_artifact_sealed"] = p5_seal["listed_file_count"] >= 7
    checks["protected_access_zero"] = all(
        access.get(key) == 0 for key in ("test_reads", "protected_reads", "eval160_reads")
    ) and all(
        access.get(key) is False
        for key in (
            "teacher_records_accessed",
            "sidecar_accessed",
            "relation_data_accessed",
            "telemetry_raw_accessed",
        )
    )
    expected_script_provenance = {
        "p5_script_sha256": sha256_file(ROOT / "scripts/detector_v5/run_d8_p5_25d_gpu_smoke.py"),
        "train_core_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_train_core.py"),
        "source_contract_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_source_contract.py"),
    }
    checks["p5_script_provenance"] = p5_report.get("script_provenance") == expected_script_provenance
    checks["p5_counts"] = (
        p5_report.get("train_samples") == 141_694
        and p5_report.get("val_samples") == 37_980
        and p5_report.get("train_identities") == 507
        and p5_report.get("val_identities") == 136
    )
    checks["p5_normalization_binding"] = (
        normalization.get("schema") == "D8_NORMALIZATION_V2"
        and normalization.get("fit_on") == "outer_training_fold_only"
        and normalization.get("source_identity_digest") == p5_report.get("train_identity_digest")
        and normalization.get("train_sample_count") == p5_report.get("train_samples")
    )
    checks["p5_batch_schema"] = (
        batch_schema.get("schema") == "D8_BATCH_SCHEMA_V1"
        and batch_schema.get("feature_dim") == 25
        and batch_schema.get("effective_mask_required") is True
    )
    try:
        try:
            checkpoint = torch.load(p5_root / "CHECKPOINT.pt", map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(p5_root / "CHECKPOINT.pt", map_location="cpu")
    except Exception:
        checkpoint = {}
    checks["p5_checkpoint_binding"] = (
        checkpoint.get("schema") == "D8_STUDENT_CHECKPOINT_V2"
        and checkpoint.get("source_snapshot_sha256") == source["source_snapshot_sha256"]
        and checkpoint.get("executable_source_commit") == source["executable_source_commit"]
        and checkpoint.get("executable_source_tree") == source["executable_source_tree"]
        and checkpoint.get("normalization") == normalization
    )

    checks["cache_access_zero"] = all(
        manifest.get("test_reads") == 0
        and manifest.get("protected_reads") == 0
        and manifest.get("eval160_reads") == 0
        for manifest in (cache_manifest_a, cache_manifest_b)
    )

    assert_violations = _assert_free_files()
    checks["formal_code_has_no_assert"] = not assert_violations
    legacy_snapshot_references = []
    for rel in FORMAL_PYTHON_FILES:
        text = (ROOT / rel).read_text("utf-8")
        if 'with_name("SOURCE_SNAPSHOT.json")' in text or "SOURCE_SNAPSHOT_V1" in text:
            legacy_snapshot_references.append(rel)
    checks["legacy_static_snapshot_removed"] = (
        not legacy_snapshot_references
        and not (ROOT / "scripts/detector_v5/SOURCE_SNAPSHOT.json").exists()
    )

    details["cache_comparison"] = cache_comparison
    details["assert_violations"] = assert_violations
    details["legacy_snapshot_references"] = legacy_snapshot_references
    details["p5_package_seal"] = p5_seal["sha256sums_sha256"]
    details["expected_source_binding"] = expected_binding

    all_pass = all(checks.values())
    return {
        "schema": "D8_H1_R9_SUBAGENT_REVIEW_V1",
        "review_verdict": "COMMIT_SAFE" if all_pass else "NOT_COMMIT_SAFE",
        "status": "PASS" if all_pass else "FAIL",
        "checks": dict(sorted(checks.items())),
        "details": details,
        "protected_reads": 0,
        "eval160_reads": 0,
        "attack_rollouts_started": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--p5-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    report = audit(
        args.cache_a.resolve(strict=True),
        args.cache_b.resolve(strict=True),
        args.p5_root.resolve(strict=True),
        args.source_snapshot.resolve(strict=True),
    )
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_root.with_name(f".{args.output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "SUBAGENT_REVIEW.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Detector-v3 D8 H1-R9 Review",
        "",
        f"- Verdict: **{report['review_verdict']}**",
        f"- Status: **{report['status']}**",
        "- Eval160 reads: **0**",
        "- Attack rollouts started: **0**",
        "",
        "## Checks",
        "",
    ]
    markdown.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in report["checks"].items()
    )
    (staging / "SUBAGENT_REVIEW.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    digest = _write_seal(staging)
    rename_noreplace(staging, args.output_root)
    print(f"review_verdict={report['review_verdict']}")
    print(f"seal={digest}")
    return 0 if report["review_verdict"] == "COMMIT_SAFE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
