"""Freeze the hash-ranked clean-only Stage V R2 qualification identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SALT = "STAGE_V_R2_CONTROL_QUALIFICATION_20260807"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(pool_path: Path, output_dir: Path, *, expected_sha256: str, per_suite: int, salt: str) -> dict[str, Any]:
    pool_path = pool_path.resolve()
    output_dir = output_dir.resolve()
    if not pool_path.is_file():
        raise ValueError(f"candidate pool missing: {pool_path}")
    actual_sha256 = sha256_file(pool_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("candidate pool SHA256 mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"qualification manifest output must be new/empty: {output_dir}")
    value = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1":
        raise ValueError("unexpected candidate pool schema")
    gates = value.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("candidate pool gates missing")
    required_zero = ("eval160_reads", "protected_eval_reads", "attack_rollouts")
    if any(gates.get(field, 0) != 0 for field in required_zero):
        raise ValueError("candidate pool boundary is non-zero")
    if gates.get("attack_informed_tuning") is not False or gates.get("new_cohort_clean_only_until_freeze") is not True:
        raise ValueError("candidate pool is not clean-only")
    if value.get("selection_frozen_before_new_rollouts") is not True or value.get("final_attack_test_parents_are_separate") is not True:
        raise ValueError("candidate pool selection boundary is not frozen")
    rows = value.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate pool has no candidates")
    required = {"canonical_parent_key", "suite", "task_index", "state_index"}
    by_suite: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError("candidate row is missing identity fields")
        suite = str(raw["suite"])
        key = str(raw["canonical_parent_key"])
        if suite not in by_suite or key in seen:
            raise ValueError("candidate suite or identity is invalid")
        if raw.get("legacy_g10_test_only") is not True:
            raise ValueError("candidate is not explicitly legacy test-only")
        seen.add(key)
        by_suite[suite].append(dict(raw))
    if any(len(by_suite[suite]) < per_suite for suite in SUITES):
        raise ValueError("candidate pool cannot satisfy every suite quota")
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        ranked = sorted(
            by_suite[suite],
            key=lambda row: (hashlib.sha256(f"{salt}::{row['canonical_parent_key']}".encode()).hexdigest(), row["canonical_parent_key"]),
        )
        selected.extend({
            **row,
            "qualification_mode": "FRESH_CLEAN_AB_REPLAY",
            "old_artifacts_reused": False,
            "source_artifact_read": False,
            "qualification_rank_sha256": hashlib.sha256(f"{salt}::{row['canonical_parent_key']}".encode()).hexdigest(),
        } for row in ranked[:per_suite])
    manifest = {
        "schema": "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST_V1",
        "status": "FROZEN",
        "salt": salt,
        "initial_per_suite": per_suite,
        "suites": list(SUITES),
        "selected_count": len(selected),
        "selected_per_suite": {suite: per_suite for suite in SUITES},
        "selected_parents": selected,
        "candidate_pool": str(pool_path),
        "candidate_pool_sha256": actual_sha256,
        "candidate_pool_schema": value["schema"],
        "candidate_role": "clean_control_only_legacy_g10_test_only",
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (output_dir / "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST.sha256").write_text(
        f"{manifest_sha256}  {manifest_path.name}\n", encoding="utf-8",
    )
    atomic_write_json(output_dir / "STAGE_V_R2_QUALIFICATION_CANDIDATE_AUDIT.json", {
        "schema": "STAGE_V_R2_QUALIFICATION_CANDIDATE_AUDIT_V1",
        "verdict": "PASS",
        "candidate_pool_sha256": actual_sha256,
        "manifest_sha256": manifest_sha256,
        "selected_count": len(selected),
        "selected_per_suite": {suite: per_suite for suite in SUITES},
        "hash_order_only": True,
        "vulnerability_outcomes_read": False,
        "old_artifacts_reused": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "audited_utc": manifest["generated_utc"],
    })
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--per-suite", type=int, default=20)
    parser.add_argument("--salt", default=SALT)
    args = parser.parse_args(argv)
    if args.per_suite <= 0 or args.salt != SALT:
        raise SystemExit("invalid frozen qualification parameters")
    prepare(args.candidate_pool, args.output_dir, expected_sha256=args.expected_pool_sha256, per_suite=args.per_suite, salt=args.salt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
