"""Bind every pre-M4 authority and seal the no-outcome-read lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_r3_contact_input import sha256_file, verify_seal


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _file(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"missing regular file: {value}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_dir() or value.is_symlink():
        raise ValueError(f"invalid root: {value}")
    return value


def _sealed(root: Path, name: str) -> tuple[dict[str, Any], str]:
    root = _root(root)
    seal = verify_seal(root)["sha256sums_sha256"]
    path = _file(root / name)
    return _json(path), seal


def _sealed_exact_plan(root: Path, name: str) -> tuple[dict[str, Any], str]:
    root = _root(root)
    sums = root / "SHA256SUMS"
    pointer = root / "ROOT_SEAL.sha256"
    if not sums.is_file() or not pointer.is_file():
        raise ValueError(f"exact-plan seal missing: {root}")
    digest = sha256_file(sums)
    if pointer.read_text(encoding="utf-8").strip() != f"{digest}  SHA256SUMS":
        raise ValueError(f"exact-plan seal pointer mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        item, separator, relative_name = line.partition("  ")
        if not separator or len(item) != 64 or relative_name in listed or relative_name in {"SHA256SUMS", "ROOT_SEAL.sha256"}:
            raise ValueError(f"invalid exact-plan seal row: {line!r}")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file() or sha256_file(root / relative) != item:
            raise ValueError(f"invalid exact-plan sealed file: {relative_name}")
        listed.add(relative_name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.sha256"}}
    if actual != listed:
        raise ValueError(f"exact-plan seal closure mismatch: missing={sorted(actual - listed)} extra={sorted(listed - actual)}")
    path = _file(root / name)
    return _json(path), digest


def _seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def seal(*, freeze_root: Path, firewall_root: Path, final_manifest: Path, final_split: Path, exact_plan_root: Path, output_root: Path) -> dict[str, Any]:
    freeze, freeze_seal = _sealed(freeze_root, "PRIMARY_TEACHER_STUDENT_FREEZE.json")
    firewall, firewall_seal = _sealed(firewall_root, "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3.json")
    final_manifest = _file(final_manifest); final_split = _file(final_split)
    final = _json(final_manifest); split = _json(final_split)
    plan_root = _root(exact_plan_root)
    plan, plan_seal = _sealed_exact_plan(plan_root, "PLAN_RESULT.json")
    plan_manifest, _ = _sealed_exact_plan(plan_root, "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    if freeze.get("status") != "PASS_PRIMARY_TEACHER_STUDENT_FREEZE" or freeze.get("m4_outcomes_read") is not False or freeze.get("formal_m4_authorized") is not False or freeze.get("protected_counters") != COUNTERS:
        raise ValueError("Teacher/Student freeze is not a clean pre-M4 PASS")
    if freeze.get("student", {}).get("seal_sha256sums_sha256") == "" or freeze.get("primary_data_firewall", {}).get("seal_sha256sums_sha256") != firewall_seal:
        raise ValueError("freeze does not bind its source seals")
    if firewall.get("status") != "PASS_PRIMARY_DATA_FIREWALL_EXACT55" or firewall.get("protected_counters") != COUNTERS:
        raise ValueError("firewall is not PASS")
    if final.get("parent_count") != 40 or final.get("formal_m4_authorized") is not False or final.get("outcomes_read") is not False or split.get("status") != "FROZEN" or split.get("outcomes_read") is not False:
        raise ValueError("final40/split is not frozen")
    if plan.get("status") != "PASS" or plan.get("manifest_status") != "PASS_EXACT_40X24_PLAN_ONLY" or plan.get("parent_count") != 40 or plan.get("probe_count_total") != 960 or plan.get("planned_branch_authority_count") != 3840 or plan.get("outcomes_read") is not False or plan.get("intervention_executed") is not False or plan.get("protected_counters") != COUNTERS:
        raise ValueError("exact plan is not PASS")
    if plan_manifest.get("status") != "PASS_EXACT_40X24_PLAN_ONLY":
        raise ValueError("exact plan manifest is not PASS")
    report = {
        "schema": "STAGE_V_PRE_M4_LOCK_V1",
        "status": "PASS_PRE_M4_LOCK",
        "freeze": {"root": str(_root(freeze_root)), "sha256sums_sha256": freeze_seal, "report_sha256": _sha(_root(freeze_root) / "PRIMARY_TEACHER_STUDENT_FREEZE.json")},
        "primary_data_firewall": {"root": str(_root(firewall_root)), "sha256sums_sha256": firewall_seal, "report_sha256": _sha(_root(firewall_root) / "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3.json")},
        "final40": {"path": str(final_manifest), "sha256": _sha(final_manifest), "split_path": str(final_split), "split_sha256": _sha(final_split)},
        "exact_plan": {"root": str(plan_root), "sha256sums_sha256": plan_seal, "manifest_sha256": _sha(plan_root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"), "plan_result_sha256": _sha(plan_root / "PLAN_RESULT.json")},
        "parent_count": 40,
        "probe_count": 960,
        "planned_branch_count": 3840,
        "m4_outcomes_read": False,
        "v_phys_generated": False,
        "intervention_executed": False,
        "teacher_predictions_read": False,
        "student_predictions_read": False,
        "formal_m4_authorized": False,
        "protected_counters": dict(COUNTERS),
        "failure_action": "HOLD_SEALED_NO_OUTCOME_READ_NO_RUNTIME_PROBE_RESELECTION",
    }
    output_root = output_root.resolve()
    if output_root.parent != _root(freeze_root).parent or output_root.exists():
        raise ValueError("lock output must be a new sibling of freeze")
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "PRE_M4_LOCK.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _seal(staging)
    staging.rename(output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("freeze-root", "firewall-root", "final-manifest", "final-split", "exact-plan-root", "output-root"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(freeze_root=args.freeze_root, firewall_root=args.firewall_root, final_manifest=args.final_manifest, final_split=args.final_split, exact_plan_root=args.exact_plan_root, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
