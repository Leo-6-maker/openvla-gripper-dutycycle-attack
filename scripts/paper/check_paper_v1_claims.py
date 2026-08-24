#!/usr/bin/env python3
"""Read-only semantic check of the sealed Paper V1 claim ledger.

This entry point deliberately reuses the claim parser and hard checks from the
historical V1 sealer without calling its write or seal functions.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_paper_v1_claims import (
    AUTHORITY,
    E4_LEDGER,
    MANUSCRIPT,
    caption_records,
    hard_checks,
    limitation_records,
    parse_claims,
)


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "paper/PAPER_V1_CLAIM_LEDGER_V1.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PAPER_V1_CLAIM_AUDIT_READ_ONLY_FAIL: {message}")


def main() -> int:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    e4 = json.loads(E4_LEDGER.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    require(
        authority.get("status") == "PAPER_V1_EVIDENCE_AUTHORITY_MAP_PASS",
        "authority map is not PASS",
    )
    require(
        e4.get("status") == "STAGE_X_X1R2_E4_PAPER_LOCK_READY",
        "E4 paper lock is not ready",
    )
    require(e4.get("attack_efficacy") is False, "E4 attack-efficacy boundary changed")

    claims = parse_claims(manuscript)
    claims.extend(limitation_records(manuscript))
    claims.extend(caption_records())
    failures = hard_checks(manuscript, claims, authority, e4)
    require(not failures, "; ".join(failures))

    require(ledger.get("status") == "PAPER_V1_CLAIM_AUDIT_PASS", "ledger is not PASS")
    require(ledger.get("claim_count") == len(claims), "claim count changed")
    require(ledger.get("claims") == claims, "sealed claim records differ from manuscript")
    expected_hard_checks = {
        "parent_unit_primary": True,
        "candidate_slots_non_iid": True,
        "e3_e4_not_physical_efficacy": True,
        "e2_not_attack_failure": True,
        "detector_not_used_to_explain_e3": True,
        "x0_no_formal_mediation": True,
        "historical_invalid_nonpromotional": True,
        "protected_eval160_unread": True,
        "source_denominator_binding": True,
        "stale_result_not_promoted": True,
    }
    require(
        ledger.get("hard_checks") == expected_hard_checks,
        "sealed hard-check record changed",
    )

    expected_boundary = {
        "new_openvla_inference": 0,
        "new_simulator_or_env_step": 0,
        "new_pgd_or_backward": 0,
        "new_physical_intervention": 0,
        "new_vphys_read": 0,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    require(
        ledger.get("protected_boundary") == expected_boundary,
        "protected boundary changed",
    )

    print(
        "PAPER_V1_CLAIM_AUDIT_READ_ONLY_PASS "
        f"claims={len(claims)} eval160=UNREAD protected=UNREAD"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
