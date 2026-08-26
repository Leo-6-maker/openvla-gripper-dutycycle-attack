#!/usr/bin/env python3
"""Build the static AA2 census synthesis from sealed cell receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COMMON_SALT = "STAGE_AA_AA2_COMMON_PARENT_SELECTION_V1_20260826"
FORBIDDEN_COUNTERS = ("open_intervention_steps", "attacked_env_steps", "pgd_calls", "aa_v_phys_reads", "task_success_reads", "eval160_reads", "protected_reads")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rank_parent(parent_key: str) -> str:
    return hashlib.sha256(f"{COMMON_SALT}|{parent_key}".encode()).hexdigest()


def strip_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_evidence(item) for key, item in value.items() if key != "evidence_rows"}
    if isinstance(value, list):
        return [strip_evidence(item) for item in value]
    return value


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = ROOT / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    protocol_path = ROOT / "configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"
    source_path = ROOT / "reports/STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_V1.json"
    manifest = read_json(manifest_path)
    protocol = read_json(protocol_path)
    source = read_json(source_path)
    receipts_root = ROOT / "reports/server_evidence/STAGE_AA_AA2/receipts"
    if manifest.get("cell_count") != 324:
        raise RuntimeError("AA2_MANIFEST_COUNT_INVALID")
    receipt_index: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {}
    for cell in manifest["cells"]:
        path = receipts_root / f"{cell['cell_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"AA2_RECEIPT_MISSING:{cell['cell_id']}")
        receipt = read_json(path)
        if receipt.get("status") != "AA2_CLEAN_CELL_COMPLETE":
            raise RuntimeError(f"AA2_RECEIPT_NOT_COMPLETE:{cell['cell_id']}:{receipt.get('status')}")
        if receipt.get("cell_id") != cell["cell_id"] or receipt.get("model_family") != cell["model_family"] or receipt.get("canonical_parent_key") != cell["canonical_parent_key"]:
            raise RuntimeError(f"AA2_RECEIPT_CELL_BINDING_INVALID:{cell['cell_id']}")
        receipts[cell["cell_id"]] = receipt
        receipt_index.append(
            {
                "cell_id": cell["cell_id"],
                "model_family": cell["model_family"],
                "canonical_parent_key": cell["canonical_parent_key"],
                "status": receipt["status"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "critical_eligible": bool(receipt.get("eligibility", {}).get("critical")),
                "noncritical_eligible": bool(receipt.get("eligibility", {}).get("noncritical")),
                "runtime_counters": receipt.get("runtime_counters", {}),
            }
        )

    by_model: dict[str, dict[str, Any]] = {}
    by_suite: dict[str, dict[str, Any]] = {}
    model_sets: dict[str, set[str]] = defaultdict(set)
    parent_matrix: dict[str, dict[str, Any]] = {}
    aggregate = Counter()
    reason_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    reason_by_suite: dict[str, Counter[str]] = defaultdict(Counter)
    for item in receipt_index:
        family = item["model_family"]
        suite = item["canonical_parent_key"].split("/", 1)[0]
        aggregate.update({key: int(value) for key, value in item["runtime_counters"].items() if isinstance(value, (int, float))})
        receipt = receipts[item["cell_id"]]
        reasons = receipt.get("clean", {}).get("critical_reason_counts", {})
        reason_by_model[family].update({key: int(value) for key, value in reasons.items()})
        reason_by_suite[suite].update({key: int(value) for key, value in reasons.items()})
        model_sets[family].update({item["canonical_parent_key"]} if item["critical_eligible"] else set())
        parent_matrix.setdefault(item["canonical_parent_key"], {"canonical_parent_key": item["canonical_parent_key"]})[family] = {
            "critical_eligible": item["critical_eligible"],
            "noncritical_eligible": item["noncritical_eligible"],
            "critical_anchor": strip_evidence(receipt.get("eligibility", {}).get("critical_anchor")),
            "noncritical_anchor": strip_evidence(receipt.get("eligibility", {}).get("noncritical_anchor")),
            "critical_reason_counts": reasons,
        }

    for family in ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO"):
        rows = [item for item in receipt_index if item["model_family"] == family]
        by_model[family] = {
            "cell_count": len(rows),
            "complete_cell_count": sum(item["status"] == "AA2_CLEAN_CELL_COMPLETE" for item in rows),
            "critical_eligible_count": sum(item["critical_eligible"] for item in rows),
            "noncritical_eligible_count": sum(item["noncritical_eligible"] for item in rows),
            "reason_counts": dict(sorted(reason_by_model[family].items())),
            "eligible_parent_keys": sorted(model_sets[family]),
        }
    for suite in ("libero_10", "libero_object", "libero_spatial"):
        rows = [item for item in receipt_index if item["canonical_parent_key"].startswith(f"{suite}/")]
        by_suite[suite] = {
            "cell_count": len(rows),
            "critical_eligible_count": sum(item["critical_eligible"] for item in rows),
            "noncritical_eligible_count": sum(item["noncritical_eligible"] for item in rows),
            "reason_counts": dict(sorted(reason_by_suite[suite].items())),
        }

    common = sorted(model_sets["M0_OPENVLA"] & model_sets["M1_OPENVLA_OFT"] & model_sets["M2_PI05_LIBERO"])
    ranked_common = sorted(common, key=lambda parent: (rank_parent(parent), parent))
    if len(common) >= 32:
        frozen_primary = ranked_common[:32]
        denominator_status = "N_COMMON_AT_LEAST_32_SELECT_32"
    elif len(common) >= 24:
        frozen_primary = ranked_common
        denominator_status = "N_COMMON_24_TO_31_USE_ALL"
    else:
        frozen_primary = []
        denominator_status = "STAGE_AA_AA2_CAPACITY_LIMIT_STOP_FOR_PI"
    if any(aggregate[key] != 0 for key in FORBIDDEN_COUNTERS):
        raise RuntimeError("AA2_SCIENTIFIC_FIREWALL_COUNTER_NONZERO")

    terminal = {
        "schema": "STAGE_AA_AA2_CLEAN_SCREEN_TERMINAL_V1",
        "status": "STAGE_AA_AA2_COMMON_DENOMINATOR_FROZEN_STOP_FOR_PI" if len(common) >= 24 else "STAGE_AA_AA2_CAPACITY_LIMIT_STOP_FOR_PI",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": protocol["authorization_pi_comment_id"],
        "claim_boundary": "AA2 clean-only denominator census; no treatment, physical endpoint, or cross-model vulnerability claim",
        "manifest_binding": {"path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)},
        "source_authority_binding": {"path": str(source_path.relative_to(ROOT)).replace("\\", "/"), "bytes": source_path.stat().st_size, "sha256": sha256_file(source_path)},
        "protocol_binding": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "bytes": protocol_path.stat().st_size, "sha256": sha256_file(protocol_path)},
        "census": {"manifest_cell_count": 324, "receipt_count": len(receipt_index), "complete_receipt_count": len(receipt_index), "by_model": by_model, "by_suite": by_suite},
        "eligibility_sets": {"E_M0": sorted(model_sets["M0_OPENVLA"]), "E_M1": sorted(model_sets["M1_OPENVLA_OFT"]), "E_M2": sorted(model_sets["M2_PI05_LIBERO"]), "pairwise_M0_M1": sorted(model_sets["M0_OPENVLA"] & model_sets["M1_OPENVLA_OFT"]), "pairwise_M0_M2": sorted(model_sets["M0_OPENVLA"] & model_sets["M2_PI05_LIBERO"]), "pairwise_M1_M2": sorted(model_sets["M1_OPENVLA_OFT"] & model_sets["M2_PI05_LIBERO"]), "E_common": common},
        "parent_matrix": [parent_matrix[key] for key in sorted(parent_matrix)],
        "common_denominator": {"n_common": len(common), "denominator_status": denominator_status, "ranked_common": [{"canonical_parent_key": key, "rank_sha256": rank_parent(key)} for key in ranked_common], "frozen_primary_parent_keys": frozen_primary, "floor_n": 24, "target_n": 32},
        "reason_distributions": {"by_model": {key: dict(sorted(value.items())) for key, value in reason_by_model.items()}, "by_suite": {key: dict(sorted(value.items())) for key, value in reason_by_suite.items()}},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN_COUNTERS},
        "next_legal_action": "STOP_FOR_PI",
    }
    index = {
        "schema": "STAGE_AA_AA2_CLEAN_SCREEN_RECEIPT_INDEX_V1",
        "status": terminal["status"],
        "gate": terminal["gate"],
        "receipt_count": len(receipt_index),
        "receipts": receipt_index,
        "terminal_sha256_after_write": None,
    }
    return index, terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    index, terminal = build()
    index_path = ROOT / "reports/STAGE_AA_AA2_CLEAN_SCREEN_RECEIPT_INDEX_V1.json"
    terminal_path = ROOT / "reports/STAGE_AA_AA2_CLEAN_SCREEN_TERMINAL_V1.json"
    if args.write:
        write_json(index_path, index)
        write_json(terminal_path, terminal)
    print(json.dumps({"status": "AA2_TERMINAL_BUILD_PASS", "receipt_count": terminal["census"]["receipt_count"], "n_common": terminal["common_denominator"]["n_common"], "terminal_status": terminal["status"], "write": args.write}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
