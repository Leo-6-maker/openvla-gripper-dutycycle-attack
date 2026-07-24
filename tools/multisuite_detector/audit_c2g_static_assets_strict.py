#!/usr/bin/env python3
"""Strict alias-aware wrapper for the read-only C2g live-asset inventory.

This wrapper preserves the original fail-closed inventory while accepting only the
reviewed compact BDDL spellings that map to existing canonical Teacher-v2 operators.
It also uses the production contact-identity resolver, including the official Panda
numbered jaw aliases.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.gripper_attack.c2g_semantic_aliases import GOAL_OPERATOR_ALIASES
from src.gripper_attack.c2g_teacher_v2_contact_identity import finger_side
from src.gripper_attack.c2g_teacher_v2_target_resolution import _OPERATOR_ROLES
from tools.multisuite_detector.audit_c2g_static_assets import audit_static_assets, write_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-root", action="append", default=[])
    parser.add_argument("--xml-root", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-no-bddl", action="store_true")
    parser.add_argument("--allow-no-xml", action="store_true")
    args = parser.parse_args(argv)

    # The underlying inventory reports the raw compact spelling (for example
    # ``turnon``). Accept it only when it is explicitly mapped to an already
    # supported canonical operator. Unknown spellings continue to fail closed.
    supported = set(_OPERATOR_ROLES)
    supported.update(GOAL_OPERATOR_ALIASES)
    report = audit_static_assets(
        [Path(value) for value in args.bddl_root],
        [Path(value) for value in args.xml_root],
        supported_operators=supported,
        finger_side_fn=finger_side,
        require_bddl=not args.allow_no_bddl,
        require_xml=not args.allow_no_xml,
    )
    report["semantic_alias_contract"] = {
        "goal_operator_aliases": dict(sorted(GOAL_OPERATOR_ALIASES.items())),
        "canonical_supported_operators": sorted(_OPERATOR_ROLES),
        "alias_policy": "EXPLICIT_SYNTAX_ALIAS_ONLY",
        "finger_side_source": "c2g_teacher_v2_contact_identity.finger_side",
    }
    report["exact_command"] = " ".join(shlex.quote(value) for value in sys.argv)
    write_report(report, Path(args.output_json))
    print(json.dumps({
        "status": report["status"],
        "task_files": report["task_inventory"]["file_count"],
        "xml_files": report["xml_inventory"]["file_count"],
        "unsupported_operators": report["task_inventory"]["unsupported_operators"],
        "unresolved_finger_candidates": report["xml_inventory"]["unresolved_finger_candidates"],
        "goal_operator_aliases": report["semantic_alias_contract"]["goal_operator_aliases"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
