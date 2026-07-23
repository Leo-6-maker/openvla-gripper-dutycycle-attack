#!/usr/bin/env python3
"""Build CODEX_ONE_SHOT_HANDOFF package — sealed bundle for final Codex deployment.

Produces:
- CODEX_ONE_SHOT_HANDOFF.md with exact SHAs and paths
- Sealed formal matrix configs (attack_authorized=false)
- Single-entry launcher script
- All frozen validator/analysis scripts
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
    ap.add_argument("--final-detector-root", type=Path, required=True,
                    help="FINAL_FACTORIZED_DETECTOR_V1 root from Stage 9")
    ap.add_argument("--h-receipt-root", type=Path, required=True,
                    help="Stage 7b H heldout evaluation receipt")
    ap.add_argument("--a9-parity-root", type=Path, required=True,
                    help="A9 real adapter parity receipt")
    ap.add_argument("--a10-e2e-root", type=Path, required=True,
                    help="A10 full CLI E2E receipt")
    ap.add_argument("--clean2000-root", type=Path, required=True)
    ap.add_argument("--identity-manifests-root", type=Path, required=True,
                    help="Directory containing T/C/P/H/A/FEC identity manifests")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    handoff = out_root / "CODEX_ONE_SHOT_HANDOFF"
    handoff.mkdir(parents=True)

    git_head = _git_head()
    git_branch = _git_branch()

    # ── Verify critical receipts are PASS ───────────────────────────
    h_rec = json.loads((args.h_receipt_root / "receipt.json").read_text())
    a9_rec = json.loads((args.a9_parity_root / "REAL_ADAPTER_PARITY_RECEIPT_V1.json").read_text())
    a10_rec = json.loads((args.a10_e2e_root / "FULL_CLI_E2E_RECEIPT_V1.json").read_text())

    checks = {
        "h_heldout_pass": h_rec.get("status") == "PASS",
        "a9_parity_pass": a9_rec.get("status") == "PASS",
        "a10_e2e_pass": a10_rec.get("status") == "PASS",
    }
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit(f"HANDOFF_PREREQUISITES_FAILED: {failed}")

    # ── Build formal matrix configs (attack_authorized=false) ───────
    CONDITIONS = ["CLEAN", "COMMAND_OPEN_ORACLE", "TRUE_T10", "RAND_T10", "RANDOM_TIME_T10"]
    A_STATES = list(range(35, 45))  # states 35-44

    parent_manifest = {
        "schema": "FORMAL_ATTACK_PARENT_MANIFEST_V0",
        "attack_pool": "A",
        "state_range": [35, 44],
        "conditions": CONDITIONS,
        "expected_parent_count": len(A_STATES) * 10 * 4,  # 400 episodes across 4 suites
        "attack_authorized": False,
        "scientific_go_no_go_authorized": False,
    }

    job_matrix_config = {
        "schema": "FORMAL_ATTACK_JOB_MATRIX_V0",
        "conditions": CONDITIONS,
        "a_pool_parents": f"states 35-44, 400 episodes",
        "attack_authorized": False,
    }

    protocol = {
        "schema": "FORMAL_ATTACK_PROTOCOL_V0",
        "attack_authorized": False,
        "conditions": CONDITIONS,
        "epsilon": 0.01,
        "pgd_steps": 10,
        "pgd_iterations": 1,
        "k": 10,
        "norm": "L2",
        "input_space": "pixel",
    }

    blind_protocol = {
        "schema": "FORMAL_BLIND_REVIEW_PROTOCOL_V0",
        "n_reviewers": 2,
        "labels": ["premature_opening", "slip", "drop", "transport_failure",
                    "placement_failure", "recovery", "uncertain", "not_reviewable"],
    }

    go_rules = {
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
    }

    configs_dir = handoff / "formal_matrix_configs"
    configs_dir.mkdir()
    for name, data in [("FORMAL_ATTACK_PARENT_MANIFEST_V0", parent_manifest),
                        ("FORMAL_ATTACK_JOB_MATRIX_V0", job_matrix_config),
                        ("FORMAL_ATTACK_PROTOCOL_V0", protocol),
                        ("FORMAL_BLIND_REVIEW_PROTOCOL_V0", blind_protocol),
                        ("FORMAL_GO_NO_GO_RULES_V0", go_rules)]:
        (configs_dir / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    # ── Build handoff markdown ──────────────────────────────────────
    md = f"""# Codex One-Shot Handoff

## Integration
- **Git commit:** `{git_head}`
- **Branch:** `{git_branch}`
- **Detector root:** `{args.final_detector_root}`
- **Detector SHA:** `{sha256_file(args.final_detector_root / 'SHA256SUMS')}`

## CLEAN2000
- **Root:** `{args.clean2000_root}`
- **Identity manifests:** `{args.identity_manifests_root}`

## A Pool
- **States:** 35-44
- **Episodes:** 400 (4 suites x 10 tasks x 10 states)

## Conditions
{hr()}
| Condition | k | gradient_aligned | payload | oracle_type |
|-----------|----|------------------|---------|-------------|
| CLEAN | 0 | N/A | N/A | N/A |
| COMMAND_OPEN_ORACLE | 10 | N/A | N/A | command_intervention |
| TRUE_T10 | 10 | True | N/A | N/A |
| RAND_T10 | 10 | False | N/A | N/A |
| RANDOM_TIME_T10 | 10 | N/A | matches_TRUE | N/A |
{hr()}

## Pre-flight Checks
1. Runtime provenance preflight
2. GPU/process safety check (A800 x 1)
3. Detector freeze verification (SHA256SUMS)
4. H receipt verification (PASS)
5. Formal authorization: **FALSE** (await external review)

## One-Shot Launcher
```
./scripts/run_formal_attack_matrix_one_shot.sh
```

## STOP Conditions
- Any validator returns HOLD
- Execution validation not PASS
- Evidence closure fails
- H receipt not PASS
- `attack_authorized` is not explicitly set to true by external reviewer

## Validation Commands (post-rollout)
```bash
# Execution validation
python analysis/pilot_attack/validate_factorized_attack_pilot_execution.py \\
  --pilot-job-matrix-root <job-matrix-root> \\
  --pilot-run-ledger-root <run-ledger-root> \\
  ...

# Analysis
python analysis/pilot_attack/analyze_factorized_attack_pilot.py \\
  --pilot-execution-validation-root <exec-val-root> \\
  ...

# Blind package
python analysis/pilot_attack/build_factorized_pilot_blind_review.py \\
  --pilot-execution-validation-root <exec-val-root> \\
  ...
```

## Status Flags
- `attack_authorized = FALSE`
- `scientific_go_no_go_authorized = FALSE`
- `formal_attack_executed = FALSE`
"""
    (handoff / "CODEX_ONE_SHOT_HANDOFF.md").write_text(md)

    # ── Copy one-shot launcher ──────────────────────────────────────
    launcher_src = ROOT / "scripts/run_formal_attack_matrix_one_shot.sh"
    if launcher_src.is_file():
        shutil.copy2(launcher_src, handoff / "run_formal_attack_matrix_one_shot.sh")

    # ── Seal the handoff package ───────────────────────────────────
    files = sorted(p for p in handoff.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (handoff / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(handoff).as_posix()}\n" for p in files))
    (handoff / "SHA256SUMS.sha256").write_text(f"{sha256_file(handoff / 'SHA256SUMS')}  SHA256SUMS\n")

    # ── Build receipt ──────────────────────────────────────────────
    receipt = {
        "schema": "CODEX_ONE_SHOT_HANDOFF_RECEIPT_V1",
        "builder_code_sha256": SELF_SHA,
        "status": "READY" if all(checks.values()) else "HOLD",
        "integration_head": git_head,
        "integration_branch": git_branch,
        "detector_sha": sha256_file(args.final_detector_root / "SHA256SUMS"),
        "checks": checks,
        "attack_authorized": False,
        "formal_attack_executed": False,
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "CODEX_ONE_SHOT_HANDOFF_RECEIPT_V1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    fs = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in fs))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Codex One-Shot Handoff: {receipt['status']}")
    print(f"  Package: {handoff}")
    print(f"  Integration HEAD: {git_head}")
    return 0


def hr(): return "-" * 60


if __name__ == "__main__":
    raise SystemExit(main())
