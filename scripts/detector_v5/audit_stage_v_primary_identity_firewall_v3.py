"""Audit the exact55 corridor firewall before the primary clean line starts.

This is a CPU/read-only gate.  It does not copy labels, train a model, or read
any held-out outcome.  The primary manifests are supplied explicitly so the
audit cannot silently fall back to a historical cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.seal_utils import rename_noreplace


IDENTITY_RE = re.compile(r"^libero_[^/]+/task_[^/]+/state_[^/]+$")
IDENTITY_FIELDS = {"canonical_parent_key", "episode_id", "parent_key", "identity"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: Any) -> str | None:
    if isinstance(value, str) and IDENTITY_RE.fullmatch(value):
        return value
    return None


def extract_identities(value: Any) -> set[str]:
    """Collect only canonical identity fields, including episode-binding keys."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in IDENTITY_FIELDS:
                item = _identity(child)
                if item is not None:
                    found.add(item)
            if key in {"episode_bindings", "identity_bindings"} and isinstance(child, Mapping):
                found.update(item for item in (_identity(item) for item in child) if item is not None)
            found.update(extract_identities(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.update(extract_identities(child))
    return found


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_binding(spec: str) -> tuple[str, Path]:
    role, separator, raw_path = spec.partition("=")
    if not separator or not role or not raw_path:
        raise ValueError(f"primary manifest must be ROLE=PATH: {spec}")
    return role, Path(raw_path).resolve()


def _require_zero(value: Any, field: str) -> None:
    if value not in (0, False, [], {}):
        raise ValueError(f"{field} is not zero/false: {value!r}")


def _validate_exact_plan(plan_result: Mapping[str, Any]) -> None:
    if plan_result.get("status") != "PASS" or plan_result.get("manifest_status") != "PASS_EXACT_40X24_PLAN_ONLY":
        raise ValueError("exact plan result is not PASS_EXACT_40X24_PLAN_ONLY")
    if plan_result.get("parent_count") != 40 or plan_result.get("probe_count_total") != 960:
        raise ValueError("exact plan cardinality mismatch")
    if plan_result.get("planned_branch_authority_count") != 3840:
        raise ValueError("exact branch authority count mismatch")
    for field in ("outcomes_read", "intervention_executed"):
        if plan_result.get(field) is not False:
            raise ValueError(f"exact plan {field} is not false")
    for field in ("protected_reads", "eval160_reads", "attack_rollouts", "vis_pgd_attack_rollouts"):
        _require_zero(plan_result.get(field, 0), f"exact plan {field}")


def audit(
    *,
    exact55_registry: Path,
    final_manifest: Path,
    final_split: Path,
    exact_plan_result: Path,
    exact_plan_manifest: Path,
    primary_manifests: list[str],
    historical_quarantine: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if not output_root.is_absolute() or ".." in output_root.parts:
        raise ValueError("output root must be a new absolute path")

    exact55_data = _read_json(exact55_registry.resolve())
    attempted = extract_identities(exact55_data)
    if len(attempted) != 55:
        raise ValueError(f"exact55 registry identity count mismatch: {len(attempted)}")

    final_data = _read_json(final_manifest.resolve())
    final_ids = extract_identities(final_data)
    if len(final_ids) != 40 or not final_ids.issubset(attempted):
        raise ValueError("final40 is not an exact subset of exact55")

    split_data = _read_json(final_split.resolve())
    split_ids = extract_identities(split_data)
    if split_ids != final_ids:
        raise ValueError("final split identity set does not equal final40")
    if split_data.get("counts") != {"TRAIN": 24, "VAL": 8, "TEST": 8}:
        raise ValueError("final split counts are not 24/8/8")

    plan_result = _read_json(exact_plan_result.resolve())
    if not isinstance(plan_result, Mapping):
        raise ValueError("exact plan result is not an object")
    _validate_exact_plan(plan_result)
    plan_data = _read_json(exact_plan_manifest.resolve())
    plan_ids = extract_identities(plan_data)
    if plan_ids != final_ids:
        raise ValueError("exact plan parent set does not equal final40")

    primary_results: list[dict[str, Any]] = []
    primary_union: set[str] = set()
    for spec in primary_manifests:
        role, path = _manifest_binding(spec)
        data = _read_json(path)
        identities = extract_identities(data)
        if not identities:
            raise ValueError(f"primary manifest has no identities: {role}")
        attempted_overlap = sorted(identities & attempted)
        final_overlap = sorted(identities & final_ids)
        if attempted_overlap or final_overlap:
            raise ValueError(f"primary identity overlap for {role}")
        primary_results.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
            "identity_count": len(identities),
            "attempted_overlap": attempted_overlap,
            "final40_overlap": final_overlap,
            "result": "PASS_IDENTITY_DISJOINT",
        })
        primary_union.update(identities)

    historical = None
    if historical_quarantine is not None:
        historical_data = _read_json(historical_quarantine.resolve())
        historical_ids = extract_identities(historical_data)
        historical = {
            "path": str(historical_quarantine.resolve()),
            "sha256": sha256_file(historical_quarantine.resolve()),
            "identity_count": len(historical_ids),
            "final40_overlap": len(historical_ids & final_ids),
            "status": "UNREAD_QUARANTINED",
            "primary_use": False,
        }

    report = {
        "schema": "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3",
        "version": "PRIMARY-DATA-FIREWALL-V3-EXACT55",
        "status": "PASS_PRIMARY_DATA_FIREWALL_EXACT55",
        "purpose": "Exact55 corridor-attempt firewall before a fresh clean-only Teacher/Student primary package.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "m4_outcomes_read": False,
            "teacher_student_started": False,
            "formal_training_authorized": False,
            "formal_inference_authorized": False,
        },
        "exact55_registry": {"path": str(exact55_registry.resolve()), "sha256": sha256_file(exact55_registry.resolve()), "identity_count": len(attempted)},
        "final40": {"path": str(final_manifest.resolve()), "sha256": sha256_file(final_manifest.resolve()), "identity_count": len(final_ids)},
        "final_split": {"path": str(final_split.resolve()), "sha256": sha256_file(final_split.resolve()), "counts": split_data.get("counts")},
        "exact_plan": {
            "result_path": str(exact_plan_result.resolve()),
            "result_sha256": sha256_file(exact_plan_result.resolve()),
            "manifest_path": str(exact_plan_manifest.resolve()),
            "manifest_sha256": sha256_file(exact_plan_manifest.resolve()),
            "status": plan_result.get("manifest_status"),
            "parent_count": plan_result.get("parent_count"),
            "probe_count_total": plan_result.get("probe_count_total"),
            "planned_branch_authority_count": plan_result.get("planned_branch_authority_count"),
            "outcomes_read": False,
            "intervention_executed": False,
        },
        "primary_manifests": primary_results,
        "primary_identity_firewall": {
            "primary_union_count": len(primary_union),
            "attempted_overlap_count": len(primary_union & attempted),
            "final40_overlap_count": len(primary_union & final_ids),
            "zero_overlap": True,
        },
        "historical_quarantine": historical,
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "source_artifacts_modified": False,
        "next_action": "Use this sealed firewall SHA as a parent binding for a fresh clean-only Teacher package; do not read M4 outcomes.",
    }

    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        report_path = staging / "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in files), encoding="utf-8")
        digest = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
        rename_noreplace(staging, output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact55-registry", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--final-split", type=Path, required=True)
    parser.add_argument("--exact-plan-result", type=Path, required=True)
    parser.add_argument("--exact-plan-manifest", type=Path, required=True)
    parser.add_argument("--primary-manifest", action="append", required=True, help="ROLE=PATH; repeat for every FIT/CAL/CHECK source")
    parser.add_argument("--historical-quarantine", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(exact55_registry=args.exact55_registry, final_manifest=args.final_manifest, final_split=args.final_split, exact_plan_result=args.exact_plan_result, exact_plan_manifest=args.exact_plan_manifest, primary_manifests=args.primary_manifest, historical_quarantine=args.historical_quarantine, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
