#!/usr/bin/env python3
"""Aggregate independent Gate-B parent audits into the V1.4 final gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _verify_seal(root: Path) -> list[str]:
    errors: list[str] = []
    sums, header = root / "SHA256SUMS", root / "SHA256SUMS.sha256"
    if not sums.is_file() or not header.is_file():
        return ["SEAL_FILES_MISSING"]
    parts = header.read_text(encoding="utf-8").split()
    if len(parts) != 2 or parts[1] != "SHA256SUMS" or parts[0] != _sha_file(sums):
        errors.append("SEAL_HEADER_INVALID")
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            errors.append(f"SEAL_PATH_INVALID:{relative}")
            continue
        if not separator or not target.is_file() or _sha_file(target) != digest:
            errors.append(f"SEAL_ROW_INVALID:{relative}")
    return errors


def _parent(root: Path, source_commit: str, source_tree: str) -> tuple[dict[str, Any], list[str]]:
    errors = _verify_seal(root)
    try:
        receipt = _load(root / "M3_5_V1_4_GATE_B_RECEIPT.json")
        audit = _load(root / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json")
        labels = [json.loads(line) for line in (root / "COLLAPSED_PROBE_DOSE_LABELS_V1_4.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, errors + [f"PARENT_ARTIFACT_READ_FAIL:{type(exc).__name__}:{exc}"]
    if receipt.get("status") != "PASS_PARENT_DIAGNOSTIC" or audit.get("status") != "PASS_PARENT_INDEPENDENT":
        errors.append("PARENT_GATE_B_NOT_PASS")
    if receipt.get("source_commit") != source_commit or receipt.get("source_tree") != source_tree or audit.get("source_commit") != source_commit or audit.get("source_tree") != source_tree:
        errors.append("PARENT_SOURCE_BINDING_MISMATCH")
    if receipt.get("protected_counters") != COUNTERS or audit.get("protected_counters") != COUNTERS:
        errors.append("PARENT_PROTECTED_COUNTERS_NONZERO")
    if len(labels) != 72 or any(row.get("binary_label_consumable") is not True or row.get("label_class") not in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"} for row in labels):
        errors.append("PARENT_LABEL_CONSUMABILITY_INVALID")
    result = {
        "root": str(root),
        "parent_key": receipt.get("canonical_parent_key"),
        "suite": receipt.get("suite"),
        "label_count": len(labels),
        "receipt_sha256": _sha_file(root / "M3_5_V1_4_GATE_B_RECEIPT.json"),
        "audit_sha256": _sha_file(root / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json"),
    }
    return result, errors


def audit(roots: list[Path], *, source_commit: str, source_tree: str) -> dict[str, Any]:
    errors: list[str] = []
    parents: list[dict[str, Any]] = []
    for root in roots:
        result, root_errors = _parent(root.resolve(), source_commit, source_tree)
        parents.append(result)
        errors.extend(f"{root.name}:{error}" for error in root_errors)
    keys = [row.get("parent_key") for row in parents]
    if len(keys) != len(set(keys)):
        errors.append("DUPLICATE_PARENT_KEYS")
    suite_counts = {suite: sum(row.get("suite") == suite for row in parents) for suite in SUITES}
    if len(parents) != 8 or suite_counts != {suite: 2 for suite in SUITES}:
        errors.append(f"FOUR_SUITE_COVERAGE_INVALID:{suite_counts}")
    status = "PASS" if not errors else "FAIL_SEALED"
    return {
        "schema": "STAGE_V_M3_5_V1_4_FINAL_INDEPENDENT_AUDIT_V1",
        "status": status,
        "auditor_role": "independent_four_suite_gate_b_aggregator_no_producer_decision_helper",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "parent_count": len(parents),
        "suite_counts": suite_counts,
        "parents": parents,
        "errors": sorted(set(errors)),
        "protected_counters": dict(COUNTERS),
    }


def _seal(root: Path) -> None:
    rows = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{_sha_file(path)}  {path.name}\n")
    (root / "SHA256SUMS").write_bytes("".join(rows).encode("utf-8"))
    (root / "SHA256SUMS.sha256").write_bytes(f"{_sha_file(root / 'SHA256SUMS')}  SHA256SUMS\n".encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"REFUSE_OVERWRITE:{output}")
    output.mkdir(parents=True, exist_ok=False)
    result = audit([path.resolve() for path in args.parent_root], source_commit=args.source_commit, source_tree=args.source_tree)
    (output / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json").write_bytes((json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    final = {
        "schema": "STAGE_V_M3_5_V1_4_FINAL_RECEIPT_V1",
        "status": result["status"],
        "M3_5_LABEL_VALIDATION": "PASS" if result["status"] == "PASS" else "HOLD",
        "v7_authorized": result["status"] == "PASS",
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "independent_audit_sha256": _sha_file(output / "M3_5_V1_4_GATE_B_INDEPENDENT_AUDIT.json"),
        "parent_count": result["parent_count"],
        "suite_counts": result["suite_counts"],
        "errors": result["errors"],
        "next_legal_gate": "V7_FRESH_QUALIFICATION" if result["status"] == "PASS" else "HARD_STOP_SEALED_M3_5",
        "protected_counters": dict(COUNTERS),
    }
    (output / "M3_5_V1_4_FINAL_RECEIPT.json").write_bytes((json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    _seal(output)
    print(json.dumps({"status": result["status"], "output_root": str(output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
