#!/usr/bin/env python3
"""Build CODEX_ONE_SHOT_HANDOFF — sealed bundle for final deployment.

Output goes directly to --output-root as a sealed directory.
Handoff package lives inside output-root as CODEX_ONE_SHOT_HANDOFF/.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def _git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT))
    return r.stdout.strip()


def _git_branch() -> str:
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=str(ROOT))
    return r.stdout.strip()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-detector-root", type=Path, required=True)
    ap.add_argument("--h-receipt-root", type=Path, required=True)
    ap.add_argument("--a9-parity-root", type=Path, default=None)
    ap.add_argument("--a10-e2e-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    git_head = _git_head()
    git_branch = _git_branch()

    # ── Verify prerequisites ─────────────────────────────────────────
    detector_root = args.final_detector_root
    detector_seal = sha256_file(detector_root / "SHA256SUMS") if (detector_root / "SHA256SUMS").is_file() else None

    h_receipt = None
    for name in ["HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json", "receipt.json"]:
        p = args.h_receipt_root / name
        if p.is_file():
            h_receipt = json.loads(p.read_text(encoding="utf-8"))
            break

    checks = {
        "detector_sealed": detector_seal is not None,
        "h_receipt_found": h_receipt is not None,
        "h_gate_pass": h_receipt.get("gate_pass", False) if h_receipt else False,
        "attack_authorized": False,
    }

    # ── Write handoff inside staging (atomic) ────────────────────────
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    handoff = staging / "CODEX_ONE_SHOT_HANDOFF"
    handoff.mkdir()

    # Formal matrix configs (skeleton — real manifests built by attack matrix builder)
    CONDITIONS = ["CLEAN", "COMMAND_OPEN_ORACLE", "TRUE_T10", "RAND_T10", "RANDOM_TIME_T10"]
    configs = handoff / "formal_matrix_configs"
    configs.mkdir()

    (configs / "FORMAL_ATTACK_PROTOCOL_V0.json").write_text(json.dumps({
        "schema": "FORMAL_ATTACK_PROTOCOL_V0",
        "conditions": CONDITIONS,
        "attack_authorized": False,
        "a_pool_states": "35-44",
    }, indent=2, sort_keys=True) + "\n")

    (configs / "FORMAL_GO_NO_GO_RULES_V0.json").write_text(json.dumps({
        "schema": "FORMAL_GO_NO_GO_RULES_V0",
        "rules": {
            "min_valid_pairs": 1,
            "min_oracle_actuation_parents": 1,
            "min_true_over_rand_parents": 1,
            "min_true_over_random_time_parents": 1,
            "max_missing_evidence": 0,
            "require_all_conditions_per_group": True,
        },
        "attack_authorized": False,
        "scientific_go_no_go_authorized": False,
    }, indent=2, sort_keys=True) + "\n")

    # Handoff markdown
    md = f"""# Codex One-Shot Handoff

## Integration
- **Git commit:** `{git_head}`
- **Branch:** `{git_branch}`
- **Detector root:** `{detector_root}`
- **Detector seal:** `{detector_seal}`

## Prerequisites
| Check | Status |
|-------|--------|
| Detector sealed | {checks['detector_sealed']} |
| H receipt found | {checks['h_receipt_found']} |
| H gate PASS | {checks['h_gate_pass']} |
| Attack authorized | **FALSE** |

## Conditions
| Condition | k |
|-----------|----|
| CLEAN | 0 |
| COMMAND_OPEN_ORACLE | 10 |
| TRUE_T10 | 10 |
| RAND_T10 | 10 |
| RANDOM_TIME_T10 | 10 |

## STOP Conditions
- Any validator returns HOLD
- Execution validation not PASS
- H receipt not PASS or gate_pass=false
- attack_authorized is not explicitly set to true

## Status Flags
- `attack_authorized = FALSE`
- `scientific_go_no_go_authorized = FALSE`
"""
    (handoff / "CODEX_ONE_SHOT_HANDOFF.md").write_text(md)

    # Copy one-shot launcher
    launcher_src = ROOT / "scripts/run_formal_attack_matrix_one_shot.sh"
    if launcher_src.is_file():
        shutil.copy2(launcher_src, handoff / "run_formal_attack_matrix_one_shot.sh")

    # ── Build receipt ────────────────────────────────────────────────
    receipt = {
        "schema": "CODEX_ONE_SHOT_HANDOFF_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "READY" if all(checks.values()) else "HOLD",
        "integration_head": git_head,
        "integration_branch": git_branch,
        "detector_seal": detector_seal,
        "checks": checks,
        "attack_authorized": False,
    }
    (staging / "CODEX_ONE_SHOT_HANDOFF_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # ── Seal atomically ──────────────────────────────────────────────
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Codex Handoff: {receipt['status']} seal={seal[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
