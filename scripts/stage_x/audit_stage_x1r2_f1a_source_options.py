"""Read-only audit of identity-source options for the F1-A population hold."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_X_X1R2_F1A_SOURCE_OPTIONS_AUDIT_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
KEY_RE = re.compile(r"libero_(?:10|goal|object|spatial)/task_\d{2}/state_\d{2}")
STATES = range(20)
REQUIRED_PER_SUITE = 14
SCAN_PREFIXES = ("configs/", "reports/", "docs/handoffs/", "paper/")
CURRENT_AUTHORITY = ROOT / "scripts/stage_x/audit_stage_x1r_t1d0r_authority.py"
CURRENT_F1_AUDIT = ROOT / "reports/STAGE_X_X1R2_F1A_STATIC_FEASIBILITY_AUDIT_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def suite_counts(keys: set[str]) -> dict[str, int]:
    return {suite: sum(key.startswith(f"{suite}/") for key in keys) for suite in SUITES}


def main() -> int:
    universe = {
        f"{suite}/task_{task:02d}/state_{state:02d}"
        for suite in SUITES
        for task in range(10)
        for state in STATES
    }
    mentioned: set[str] = set()
    source_paths: dict[str, set[str]] = {}
    scanned_files = 0
    binary_files = 0
    for name in subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"], text=False).decode("utf-8").split("\0"):
        if not name or not name.startswith(SCAN_PREFIXES):
            continue
        path = ROOT / name
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            binary_files += 1
            continue
        scanned_files += 1
        for key in set(KEY_RE.findall(data.decode("utf-8", "ignore"))) & universe:
            mentioned.add(key)
            source_paths.setdefault(key, set()).add(name)

    remaining = universe - mentioned
    report = {
        "schema": "STAGE_X_X1R2_F1A_SOURCE_OPTIONS_AUDIT_V1",
        "status": "ALTERNATIVE_SOURCE_CAPACITY_PRESENT_BUT_PI_AUTH_REQUIRED",
        "scope": "CPU/static/read-only identity census; no rollout, outcome, Student, model, simulator, GPU, V_phys, or protected read",
        "current_f1_hold": {
            "path": CURRENT_F1_AUDIT.relative_to(ROOT).as_posix(),
            "sha256": sha256(CURRENT_F1_AUDIT),
            "status": "HOLD_F1A_FRESH_POPULATION_INSUFFICIENT",
            "current_g10_state_range": "20..49",
            "current_remaining_by_suite": {
                "libero_10": 33,
                "libero_goal": 5,
                "libero_object": 51,
                "libero_spatial": 52,
            },
        },
        "alternative_source": {
            "candidate_definition": "four suites x ten tasks x state 0..19",
            "candidate_count": len(universe),
            "tracked_scan_prefixes": list(SCAN_PREFIXES),
            "tracked_files_scanned": scanned_files,
            "binary_files_skipped": binary_files,
            "conservative_identity_mentions": len(mentioned),
            "mentioned_by_suite": suite_counts(mentioned),
            "potential_remaining_after_mentions": len(remaining),
            "potential_remaining_by_suite": suite_counts(remaining),
            "potential_remaining_key_sha256": hashlib.sha256(
                ("\n".join(sorted(remaining)) + "\n").encode("utf-8")
            ).hexdigest(),
            "mentioned_identity_sources": {
                key: sorted(paths) for key, paths in sorted(source_paths.items())
            },
        },
        "f1_requirement": {
            "dev_per_suite": 6,
            "bridge_per_suite": 8,
            "combined_required_per_suite": REQUIRED_PER_SUITE,
            "capacity_sufficient_under_conservative_scan": all(
                count >= REQUIRED_PER_SUITE for count in suite_counts(remaining).values()
            ),
        },
        "authority_and_risk": {
            "current_stage_x_authority_path": CURRENT_AUTHORITY.relative_to(ROOT).as_posix(),
            "current_stage_x_authority_sha256": sha256(CURRENT_AUTHORITY),
            "current_stage_x_authority_state_rule": "20 <= state <= 49",
            "source_change_is_scientific": True,
            "state_0_19_may_overlap_detector_fit_train_or_old_canaries": True,
            "not_authorized_by_this_audit": [
                "changing_F1_source_universe",
                "changing_parent_ranking_or_denominator",
                "freezing_DEV_or_BRIDGE_from_state_0_19",
                "opening_BRIDGE_outcomes",
                "starting_F1_B_GPU",
            ],
        },
        "decision": {
            "classification": "ALTERNATIVE_SOURCE_CAPACITY_PRESENT_BUT_PI_AUTH_REQUIRED",
            "meaning": "The existing G10 source is capacity-insufficient only because libero_goal has five remaining parents; a larger state source has static capacity, but its split/leakage and scientific authority are unresolved.",
            "next_legal_action": "PI must explicitly authorize the alternative source and its held-out/leakage contract before any new F1-A population freeze.",
        },
        "protected_boundary": {
            "gpu": 0,
            "model_inference": 0,
            "simulator": 0,
            "env_step": 0,
            "pgd": 0,
            "vphys": 0,
            "protected": "UNREAD",
            "eval160": "UNREAD",
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "potential_remaining_by_suite": report["alternative_source"]["potential_remaining_by_suite"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
