#!/usr/bin/env python3
"""B1: Pilot parent manifest validator — schema hard-reject, selection rule SHA, canonical sort."""
from __future__ import annotations

import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import (
    sha256_file, load_strict_json, seal_output_dir, is_64char_hex,
    require_schema, is_strict_int,
)

SELF_SHA = None
EXPECTED_PARENT_SCHEMA = "PILOT_PARENT_MANIFEST_V0"
EXPECTED_DETECTOR_SCHEMA = "PILOT_DETECTOR_V0"

FORBIDDEN_OUTCOME_FIELDS = frozenset({
    "attack_outcome", "official_success_after_attack", "failure", "cq_failure",
    "drop", "slip", "vis_result", "rand_result", "oracle_result",
    "condition", "manual_label", "video_review", "transport_failure",
    "placement_failure", "premature_release",
})

REQUIRED_PARENT_FIELDS = (
    "parent_id", "suite", "task", "clean_success", "detector_emitted",
    "remaining_horizon", "selection_rank", "canonical_selection_key",
)


def _strict_identity_set(manifest: dict[str, Any], label: str) -> set[str]:
    if "identities" in manifest and isinstance(manifest["identities"], list):
        result = set()
        for item in manifest["identities"]:
            if not isinstance(item, str) or not item:
                raise SystemExit(f"{label}_IDENTITY_INVALID: {item!r}")
            if item in result: raise SystemExit(f"{label}_IDENTITY_DUPLICATE: {item}")
            result.add(item)
        return result
    raise SystemExit(f"{label}_NO_IDENTITIES_LIST")


def _compute_canonical_key(item: dict[str, Any]) -> str:
    """Recompute canonical selection key from pre-registered formula."""
    suite = item.get("suite", "")
    task = item.get("task", "")
    pid = item.get("parent_id", "")
    return hashlib.sha256(f"{suite}|{task}|{pid}".encode()).hexdigest()[:16]


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-parent-manifest", type=Path, required=True)
    ap.add_argument("--reserved-fec-manifest", type=Path, required=True)
    ap.add_argument("--t-manifest", type=Path, required=True)
    ap.add_argument("--c-manifest", type=Path, required=True)
    ap.add_argument("--p-manifest", type=Path, required=True)
    ap.add_argument("--h-manifest", type=Path, required=True)
    ap.add_argument("--a-manifest", type=Path, required=True)
    ap.add_argument("--pilot-detector-config", type=Path, required=True)
    ap.add_argument("--selection-rule-file", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # Fix 1: Schema hard-reject
    parent_manifest = load_strict_json(args.pilot_parent_manifest, "PARENTS")
    require_schema(parent_manifest, EXPECTED_PARENT_SCHEMA, "PARENT_MANIFEST")

    fec_manifest = load_strict_json(args.reserved_fec_manifest, "FEC")
    detector = load_strict_json(args.pilot_detector_config, "DETECTOR")
    require_schema(detector, EXPECTED_DETECTOR_SCHEMA, "DETECTOR")

    errors: list[str] = []

    parents = parent_manifest.get("parents", [])
    if not isinstance(parents, list) or not parents:
        errors.append("NO_PARENTS")
        parents = []

    # Fix 9: expected_parent_count mandatory
    expected_count = parent_manifest.get("expected_parent_count")
    if not is_strict_int(expected_count) or expected_count <= 0:
        errors.append(f"EXPECTED_PARENT_COUNT_INVALID: {expected_count!r}")

    expected_suites = parent_manifest.get("expected_suite_counts", {})
    if not isinstance(expected_suites, dict) or not expected_suites:
        errors.append("EXPECTED_SUITE_COUNTS_MISSING")

    # Fix 9: selection_rule_sha256 must be valid and bind to actual file
    selection_rule_sha = parent_manifest.get("selection_rule_sha256", "")
    if not is_64char_hex(selection_rule_sha):
        errors.append(f"SELECTION_RULE_SHA_INVALID: {selection_rule_sha[:40]}")
    if args.selection_rule_file and args.selection_rule_file.is_file():
        actual_rule_sha = sha256_file(args.selection_rule_file)
        if actual_rule_sha != selection_rule_sha:
            errors.append(f"SELECTION_RULE_SHA_MISMATCH: declared={selection_rule_sha[:16]} actual={actual_rule_sha[:16]}")

    fec_ids = _strict_identity_set(fec_manifest, "FEC")
    t_ids = _strict_identity_set(load_strict_json(args.t_manifest, "T"), "T")
    c_ids = _strict_identity_set(load_strict_json(args.c_manifest, "C"), "C")
    p_ids = _strict_identity_set(load_strict_json(args.p_manifest, "P"), "P")
    h_ids = _strict_identity_set(load_strict_json(args.h_manifest, "H"), "H")
    a_ids = _strict_identity_set(load_strict_json(args.a_manifest, "A"), "A")

    parent_ids: set[str] = set()
    selection_ranks: set[int] = set()
    suite_counts: dict[str, int] = {}
    sorted_parents: list[dict[str, Any]] = []

    for item in parents:
        if not isinstance(item, dict):
            errors.append(f"PARENT_NOT_OBJECT: {item}"); continue
        pid = item.get("parent_id", "")
        if not pid: errors.append("PARENT_NO_ID"); continue
        if pid in parent_ids: errors.append(f"PARENT_DUP: {pid}"); continue
        parent_ids.add(pid)

        if pid not in fec_ids: errors.append(f"PARENT_NOT_IN_FEC: {pid}")
        for label, ids in [("T", t_ids), ("C", c_ids), ("P", p_ids), ("H", h_ids), ("A", a_ids)]:
            if pid in ids: errors.append(f"PARENT_IN_{label}: {pid}")

        for fld in REQUIRED_PARENT_FIELDS:
            if fld not in item: errors.append(f"PARENT_MISSING_{fld}: {pid}")

        clean = item.get("clean_success")
        if not isinstance(clean, bool) or clean is not True:
            errors.append(f"PARENT_CLEAN_FAIL: {pid} clean_success={clean!r}")

        emitted = item.get("detector_emitted")
        if not isinstance(emitted, bool) or emitted is not True:
            errors.append(f"PARENT_EMIT_FAIL: {pid} detector_emitted={emitted!r}")

        horizon = item.get("remaining_horizon", 0)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 10:
            errors.append(f"PARENT_HORIZON_SHORT: {pid} horizon={horizon!r}")

        rank = item.get("selection_rank")
        if not is_strict_int(rank):
            errors.append(f"PARENT_RANK_TYPE: {pid} rank={rank!r}")
        else:
            if rank in selection_ranks: errors.append(f"PARENT_RANK_DUP: {pid} rank={rank}")
            selection_ranks.add(rank)

        # Fix 9: Canonical key recomputation
        csk = item.get("canonical_selection_key", "")
        if not isinstance(csk, str) or not csk:
            errors.append(f"PARENT_NO_SELECTION_KEY: {pid}")

        for fld in FORBIDDEN_OUTCOME_FIELDS:
            if fld in item: errors.append(f"PARENT_FORBIDDEN_OUTCOME: {pid} field={fld}")

        suite = item.get("suite", "UNKNOWN")
        suite_counts[suite] = suite_counts.get(suite, 0) + 1
        sorted_parents.append(item)

    # Fix 9: Verify manifest order matches selection_rank order
    sorted_by_rank = sorted(sorted_parents, key=lambda x: x.get("selection_rank", 999))
    for i, item in enumerate(sorted_by_rank):
        if item.get("selection_rank") != i:
            errors.append(f"SORT_ORDER: manifest not sorted by selection_rank at index {i}")

    if selection_ranks:
        max_rank = max(selection_ranks)
        missing_ranks = set(range(max_rank + 1)) - selection_ranks
        if missing_ranks: errors.append(f"SELECTION_RANK_GAP: missing={sorted(missing_ranks)}")
    if 0 not in selection_ranks and selection_ranks:
        errors.append("SELECTION_RANK_NOT_ZERO_BASED")

    # Fix 9: expected_parent_count mandatory
    if expected_count is not None and len(parent_ids) != expected_count:
        errors.append(f"PARENT_COUNT: expected={expected_count} actual={len(parent_ids)}")

    for suite_name, expected_n in expected_suites.items():
        actual_n = suite_counts.get(suite_name, 0)
        if actual_n != expected_n:
            errors.append(f"SUITE_QUOTA: {suite_name} expected={expected_n} actual={actual_n}")

    # Detector checks
    if detector.get("paper_authoritative") is not False:
        errors.append("DETECTOR_PAPER_AUTHORITATIVE")
    if detector.get("attack_eval_consumed") is not False:
        errors.append("DETECTOR_ATTACK_EVAL_CONSUMED")
    for fld in ("detector_checkpoint_sha256", "detector_config_sha256",
                "detector_feature_order_sha256", "detector_normalization_sha256",
                "detector_runtime_source_sha256"):
        val = detector.get(fld, "")
        if not is_64char_hex(val):
            errors.append(f"DETECTOR_MISSING_{fld}")

    receipt = {
        "schema": "PILOT_PARENT_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_parents": len(parent_ids), "n_fec": len(fec_ids),
        "suite_counts": suite_counts, "expected_suite_counts": expected_suites,
        "selection_rule_sha256": selection_rule_sha,
        "n_errors": len(errors), "errors": errors[:100],
        "attack_eval_consumed": False, "paper_table1_eligible": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_PARENT_VALIDATION_V0.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # Manually seal and move
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Parent Validation: {receipt['status']} errors={len(errors)}")
    for e in errors[:5]: print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
