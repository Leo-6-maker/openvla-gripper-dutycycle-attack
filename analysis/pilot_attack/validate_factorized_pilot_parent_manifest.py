#!/usr/bin/env python3
"""B1: Pilot parent manifest validator — suite quotas, selection rank, detector binding.

P0-2: Validates parent count per suite, selection_rank uniqueness/continuity,
detector emitted, clean success, horizon, forbidden attack outcome fields,
detector checkpoint/source binding, paper_authoritative=false.
"""
from __future__ import annotations

import argparse, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, load_strict_json, seal_output_dir, is_64char_hex

SELF_SHA = None

# P0-2: Forbidden attack outcome fields
FORBIDDEN_OUTCOME_FIELDS = frozenset({
    "attack_outcome", "official_success_after_attack", "failure", "cq_failure",
    "drop", "slip", "vis_result", "rand_result", "oracle_result",
    "condition", "manual_label", "video_review", "transport_failure",
    "placement_failure", "premature_release",
})

# P0-1: Fixed parent manifest schema
REQUIRED_PARENT_FIELDS = (
    "parent_id", "suite", "task", "clean_success", "detector_emitted",
    "remaining_horizon", "selection_rank", "canonical_selection_key",
)

EXPECTED_PARENT_SCHEMA = "PILOT_PARENT_MANIFEST_V0"

# P0-1: Fixed identity manifest schema — must have "identities" list
def _strict_identity_set(manifest: dict[str, Any], label: str) -> set[str]:
    """Extract identities from manifest. Requires 'identities' list."""
    if "identities" in manifest and isinstance(manifest["identities"], list):
        result = set()
        for item in manifest["identities"]:
            if not isinstance(item, str) or not item:
                raise SystemExit(f"{label}_IDENTITY_INVALID: {item!r}")
            if item in result:
                raise SystemExit(f"{label}_IDENTITY_DUPLICATE: {item}")
            result.add(item)
        return result
    raise SystemExit(f"{label}_NO_IDENTITIES_LIST: manifest lacks 'identities' list")


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
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    parent_manifest = load_strict_json(args.pilot_parent_manifest, "PARENTS")
    if parent_manifest.get("schema") != EXPECTED_PARENT_SCHEMA:
        sys.stderr.write(f"WARNING: parent manifest schema={parent_manifest.get('schema')!r} expected={EXPECTED_PARENT_SCHEMA!r}\n")

    fec_manifest = load_strict_json(args.reserved_fec_manifest, "FEC")
    detector = load_strict_json(args.pilot_detector_config, "DETECTOR")

    errors: list[str] = []

    # P0-1: Extract parents from fixed schema
    parents = parent_manifest.get("parents", [])
    if not isinstance(parents, list) or not parents:
        errors.append("NO_PARENTS")
        parents = []

    # P0-2: Suite quotas
    expected_count = parent_manifest.get("expected_parent_count", None)
    expected_suites = parent_manifest.get("expected_suite_counts", {})
    selection_rule_sha = parent_manifest.get("selection_rule_sha256", "")

    fec_ids = _strict_identity_set(fec_manifest, "FEC")
    t_ids = _strict_identity_set(load_strict_json(args.t_manifest, "T"), "T")
    c_ids = _strict_identity_set(load_strict_json(args.c_manifest, "C"), "C")
    p_ids = _strict_identity_set(load_strict_json(args.p_manifest, "P"), "P")
    h_ids = _strict_identity_set(load_strict_json(args.h_manifest, "H"), "H")
    a_ids = _strict_identity_set(load_strict_json(args.a_manifest, "A"), "A")

    parent_ids: set[str] = set()
    selection_ranks: set[int] = set()
    suite_counts: dict[str, int] = {}

    for item in parents:
        if not isinstance(item, dict):
            errors.append(f"PARENT_NOT_OBJECT: {item}"); continue
        pid = item.get("parent_id", "")
        if not pid: errors.append("PARENT_NO_ID"); continue
        if pid in parent_ids: errors.append(f"PARENT_DUP: {pid}"); continue
        parent_ids.add(pid)

        # FEC check
        if pid not in fec_ids:
            errors.append(f"PARENT_NOT_IN_FEC: {pid}")

        # Disjoint from T/C/P/H/A
        for label, ids in [("T", t_ids), ("C", c_ids), ("P", p_ids), ("H", h_ids), ("A", a_ids)]:
            if pid in ids: errors.append(f"PARENT_IN_{label}: {pid}")

        # Required fields
        for fld in REQUIRED_PARENT_FIELDS:
            if fld not in item: errors.append(f"PARENT_MISSING_{fld}: {pid}")

        # P0-2: clean_success must be True (strict bool)
        clean = item.get("clean_success")
        if not isinstance(clean, bool) or clean is not True:
            errors.append(f"PARENT_CLEAN_FAIL: {pid} clean_success={clean!r}")

        # P0-2: detector_emitted must be True (strict bool)
        emitted = item.get("detector_emitted")
        if not isinstance(emitted, bool) or emitted is not True:
            errors.append(f"PARENT_EMIT_FAIL: {pid} detector_emitted={emitted!r}")

        # Horizon
        horizon = item.get("remaining_horizon", 0)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 10:
            errors.append(f"PARENT_HORIZON_SHORT: {pid} horizon={horizon!r}")

        # P0-2: Selection rank: strict int, unique
        rank = item.get("selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            errors.append(f"PARENT_RANK_TYPE: {pid} rank={rank!r}")
        else:
            if rank in selection_ranks: errors.append(f"PARENT_RANK_DUP: {pid} rank={rank}")
            selection_ranks.add(rank)

        # Canonical selection key
        csk = item.get("canonical_selection_key", "")
        if not isinstance(csk, str) or not csk:
            errors.append(f"PARENT_NO_SELECTION_KEY: {pid}")

        # P0-2: Forbidden outcome fields
        for fld in FORBIDDEN_OUTCOME_FIELDS:
            if fld in item: errors.append(f"PARENT_FORBIDDEN_OUTCOME: {pid} field={fld}")

        # Suite counting
        suite = item.get("suite", "UNKNOWN")
        suite_counts[suite] = suite_counts.get(suite, 0) + 1

    # P0-2: Verify selection rank continuity
    if selection_ranks:
        max_rank = max(selection_ranks)
        expected_ranks = set(range(max_rank + 1))
        missing_ranks = expected_ranks - selection_ranks
        if missing_ranks:
            errors.append(f"SELECTION_RANK_GAP: missing={sorted(missing_ranks)}")
    if 0 not in selection_ranks:
        errors.append("SELECTION_RANK_NOT_ZERO_BASED")

    # P0-2: Verify parent count
    if expected_count is not None and len(parent_ids) != expected_count:
        errors.append(f"PARENT_COUNT: expected={expected_count} actual={len(parent_ids)}")

    # P0-2: Verify suite quotas
    if expected_suites:
        for suite_name, expected_n in expected_suites.items():
            actual_n = suite_counts.get(suite_name, 0)
            if actual_n != expected_n:
                errors.append(f"SUITE_QUOTA: {suite_name} expected={expected_n} actual={actual_n}")

    # Detector config checks
    if detector.get("paper_authoritative") is not False:
        errors.append("DETECTOR_PAPER_AUTHORITATIVE")
    if detector.get("attack_eval_consumed") is not False:
        errors.append("DETECTOR_ATTACK_EVAL_CONSUMED")
    for fld in ("detector_checkpoint_sha256", "detector_config_sha256"):
        val = detector.get(fld, "")
        if not is_64char_hex(val):
            errors.append(f"DETECTOR_MISSING_{fld}")

    receipt = {
        "schema": "PILOT_PARENT_VALIDATION_V0",
        "validator_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "n_parents": len(parent_ids), "n_fec": len(fec_ids),
        "suite_counts": suite_counts, "expected_suite_counts": expected_suites,
        "n_errors": len(errors), "errors": errors[:100],
        "attack_eval_consumed": False, "paper_table1_eligible": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "PILOT_PARENT_VALIDATION_V0.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import shutil
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Pilot Parent Validation: {receipt['status']} errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
