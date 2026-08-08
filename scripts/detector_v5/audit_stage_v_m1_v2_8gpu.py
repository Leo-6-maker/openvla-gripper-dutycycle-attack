#!/usr/bin/env python3
"""Independent fail-closed audit for an M1-V2 eight-GPU root."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

try:
    from .analyze_stage_v_m1_v2_multigpu import analyze_root
    from .run_stage_v_m1_v2_8gpu import BOUNDARIES, GPU_IDS, IDENTITY, LABELS, REPO_ROOT, V2Error, _load, _write, sha256_file, validate_binding_receipt, validate_protocol
except ImportError:  # direct script execution
    from analyze_stage_v_m1_v2_multigpu import analyze_root
    from run_stage_v_m1_v2_8gpu import BOUNDARIES, GPU_IDS, IDENTITY, LABELS, REPO_ROOT, V2Error, _load, _write, sha256_file, validate_binding_receipt, validate_protocol


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _verify_runs(root: Path, run_set: str) -> None:
    base = root / run_set
    for gpu in GPU_IDS:
        for label in LABELS:
            run = base / f"gpu_{gpu:02d}" / label
            receipt_path = run / "RB1_INDEPENDENT_RECEIPT.json"
            if not receipt_path.is_file():
                raise V2Error(f"V2_RECEIPT_MISSING:{run_set}:gpu_{gpu:02d}:{label}")
            receipt = _load(receipt_path)
            if receipt.get("canonical_parent_key") != IDENTITY:
                raise V2Error(f"V2_IDENTITY_MISMATCH:{run_set}:gpu_{gpu:02d}:{label}")
            if any(receipt.get(field, 0) != 0 for field in BOUNDARIES):
                raise V2Error(f"V2_PROTECTED_BOUNDARY_NONZERO:{run_set}:gpu_{gpu:02d}:{label}")
            binding = _load(run / "M1_V2_WORKER_BINDING_RECEIPT.json")
            validate_binding_receipt(binding, gpu)
            if binding.get("run_set") != run_set or binding.get("phase") != label:
                raise V2Error(f"V2_BINDING_CONTEXT_MISMATCH:{run_set}:gpu_{gpu:02d}:{label}")


def _seal(root: Path, name: str) -> None:
    excluded = {name, f"{name}.sha256"}
    files = [path for path in root.rglob("*") if path.is_file() and path.name not in excluded and not path.name.startswith(".")]
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    target = root / name
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / f"{name}.sha256").write_text(f"{sha256_file(target)}  {name}\n", encoding="utf-8")


def audit_root(root: Path, *, final: bool) -> dict[str, Any]:
    protocol_path = REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json"
    protocol = validate_protocol(protocol_path)
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise V2Error("V2_ROOT_ALREADY_CONSUMED")
    if manifest.get("diagnostic_identity") != IDENTITY:
        raise V2Error("V2_IDENTITY_MISMATCH")
    for field in ("new_science_rollouts_authorized", "formal_parent_promotion_authorized", "vulnerability_label_generation_authorized", "student_training_authorized", "protected_evaluation_authorized", "eval160_authorized", "vis_pgd_authorized"):
        if manifest.get(field) is not False:
            raise V2Error(f"V2_AUTHORIZATION_BOUNDARY_INVALID:{field}")
    if any(manifest.get(field, 0) != 0 for field in BOUNDARIES):
        raise V2Error("V2_PROTECTED_BOUNDARY_NONZERO")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")
    if _git("status", "--porcelain"):
        raise V2Error("V2_AUDITOR_WORKTREE_DIRTY")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise V2Error("V2_PROTOCOL_SHA256_MISMATCH")
    _verify_runs(root, "runs")
    result = analyze_root(root, final=final)
    local_count = sum(len(value["pairs"]) for value in result["local"]["gpus"].values())
    cross_count = sum(len(value) for value in result["cross"]["labels"].values())
    if local_count != 32 or cross_count != 112:
        raise V2Error(f"V2_PAIR_COUNT_MISMATCH:{local_count}:{cross_count}")
    receipt = {
        "schema": "STAGE_V_M1_V2_INDEPENDENT_AUDIT_V1",
        "verdict": "PASS",
        "final": final,
        "source_commit": manifest.get("source_commit"),
        "source_tree": manifest.get("source_tree"),
        "protocol_sha256": sha256_file(protocol_path),
        "r1_run_count": 32,
        "gpu_local_pair_count": 32,
        "cross_gpu_pair_count": 112,
        "classification": result["classification"],
        "protected_boundaries": {field: 0 for field in BOUNDARIES},
    }
    if final:
        _verify_runs(root, "raw_runs")
        receipt["raw_capture_plan_sha256"] = sha256_file(root / "M1_V2_RAW_CAPTURE_PLAN.json")
    status = _load(root / "M1_V2_STATUS.json")
    status.update({"status": "PASS_CLASSIFIED" if final else "R1_AUDITED", "classification": result["classification"], "independent_audit": "PASS", "protected_boundaries": {field: 0 for field in BOUNDARIES}})
    _write(root / "M1_V2_STATUS.json", status)
    _write(root / "M1_V2_INDEPENDENT_AUDIT.json", receipt)
    _seal(root, "M1_V2_SHA256SUMS_FINAL" if final else "M1_V2_SHA256SUMS_R1")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_root(args.root.resolve(), final=args.final)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError, V2Error) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": result["verdict"], "classification": result["classification"], "final": result["final"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
