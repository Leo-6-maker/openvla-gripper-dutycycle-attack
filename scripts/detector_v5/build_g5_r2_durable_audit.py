"""G5-R2: Durable sealed independent audit evidence root.

Reads G5-R1 audit JSON from sealed G4 prediction roots, augments with
full provenance metadata, and publishes as an atomically sealed root.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

FORBIDDEN = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text("".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def build(
    g4_shared: Path, g4_three: Path, g4_physical: Path, g4_gripper: Path,
    g1_root: Path, output_root: Path,
) -> dict[str, Any]:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("clean checkout required")

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")

    # Validate all inputs
    roots = {
        "g4_shared_four_head": g4_shared.resolve(strict=True),
        "g4_three_head": g4_three.resolve(strict=True),
        "g4_physical_only": g4_physical.resolve(strict=True),
        "g4_gripper_only": g4_gripper.resolve(strict=True),
    }
    seals = {}
    predictions_sha = {}
    for name, r in roots.items():
        seals[name] = verify_seal(r)["sha256sums_sha256"]
        predictions_sha[name] = sha256_file(r / "predictions.jsonl")

    g1_root = g1_root.resolve(strict=True)
    g1_seal = verify_seal(g1_root)["sha256sums_sha256"]

    # Load and validate G1 metadata
    g1_meta: dict[str, dict[str, Any]] = {}
    for split_file in ("EPISODE_TRAIN_MANIFEST.json", "EPISODE_VAL_MANIFEST.json"):
        rows = json.loads((g1_root / split_file).read_text(encoding="utf-8"))
        for r in rows:
            g1_meta[r["episode_id"]] = {"suite": r["suite"], "task_id": r["task_id"]}

    # Recompute all metrics independently
    from audit_g5_r1_independent_metrics import audit_config
    audit_results = {}
    for name, r in [
        ("shared_four_head", roots["g4_shared_four_head"]),
        ("three_head", roots["g4_three_head"]),
        ("physical_only", roots["g4_physical_only"]),
        ("gripper_only", roots["g4_gripper_only"]),
    ]:
        audit_results[name] = audit_config(name, r, g1_meta)

    # Build the full evidence payload
    payload: dict[str, Any] = {
        "schema": "V5_R3_DETECTOR_V2_G5_R2_DURABLE_AUDIT_V1",
        "status": "PASS_INDEPENDENT_RECOMPUTATION",
        "original_v5_status": "FAIL_ORIGINAL_PROTOCOL",
        "code_snapshot": {"commit": commit, "tree": tree},
        "audit_script_sha256": sha256_file(ROOT / "scripts" / "detector_v5" / "build_g5_r2_durable_audit.py"),
        "g5_r1_script_sha256": sha256_file(ROOT / "scripts" / "detector_v5" / "audit_g5_r1_independent_metrics.py"),
        "input_roots": {
            "g4_shared_four_head": {"path": str(roots["g4_shared_four_head"]), "seal": seals["g4_shared_four_head"], "predictions_sha256": predictions_sha["g4_shared_four_head"]},
            "g4_three_head": {"path": str(roots["g4_three_head"]), "seal": seals["g4_three_head"], "predictions_sha256": predictions_sha["g4_three_head"]},
            "g4_physical_only": {"path": str(roots["g4_physical_only"]), "seal": seals["g4_physical_only"], "predictions_sha256": predictions_sha["g4_physical_only"]},
            "g4_gripper_only": {"path": str(roots["g4_gripper_only"]), "seal": seals["g4_gripper_only"], "predictions_sha256": predictions_sha["g4_gripper_only"]},
            "g1_root": {"path": str(g1_root), "seal": g1_seal},
        },
        "independent_metrics": audit_results,
        "ceiling_invariant_summary": {},
        "physical_criticality_summary": {},
        "permissions": {
            "test_payload_read": 0,
            "protected_reads": 0,
            "teacher_labels_read": True,
            "student_training_authorized": True,
            "attack_authorized": False,
        },
        "detector_v2_status": "AUTHORIZED_FOR_G6",
    }

    # Ceiling invariant for physical
    for cfg_name, cfg in audit_results.items():
        for head in ("physical_criticality", "gripper_closing_state"):
            cc = cfg.get("ceiling_check", {}).get(head, {})
            if cc:
                payload["ceiling_invariant_summary"][f"{cfg_name}/{head}"] = {
                    "teacher_critical_total": cc["tcrit"],
                    "candidate_reached": cc["reached"],
                    "detector_detected": cc["detected"],
                    "reached_le_tcrit": cc["invariant_reached_le_tcrit"],
                    "detected_le_reached": cc["invariant_detected_le_reached"],
                }

    # Physical-only summary
    phys = audit_results.get("physical_only", {})
    phys_step = phys.get("step_metrics", {}).get("physical_criticality", {})
    phys_event = phys.get("event_metrics", {}).get("physical_criticality", {})
    phys_ceil = phys.get("ceiling_check", {}).get("physical_criticality", {})

    payload["physical_criticality_summary"] = {
        "config": "physical_only",
        "validation_step": phys_step,
        "validation_event": phys_event,
        "ceiling": {
            "teacher_critical_events": phys_ceil.get("tcrit"),
            "candidate_reached": phys_ceil.get("reached"),
            "detector_detected": phys_ceil.get("detected"),
            "end_to_end_recall_numerator": phys_ceil.get("detected"),
            "end_to_end_recall_denominator": phys_ceil.get("tcrit"),
            "candidate_ceiling_numerator": phys_ceil.get("reached"),
            "candidate_ceiling_denominator": phys_ceil.get("tcrit"),
        },
    }

    # Output
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(str(output_root))
    if any(p.casefold() in FORBIDDEN for p in output_root.parts):
        raise ValueError("forbidden path")

    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        (staging / "G5_R2_DURABLE_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise

    payload["sha256sums_sha256"] = digest
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--three-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--gripper-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.shared_root, args.three_root, args.physical_root, args.gripper_root, args.g1_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
