#!/usr/bin/env python3
"""Read-only server preflight for the Z1 runtime source and resource gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


M0_EXPECTED = {
    "libero_10": (18, 15085093727),
    "libero_goal": (19, 15085095390),
    "libero_object": (19, 15085095882),
    "libero_spatial": (19, 15085095735),
}
M2_EXPECTED = (16, 12439085481)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(path: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", path, *args], text=True).strip()


def inventory(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def gpu_rows() -> list[dict]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in raw.splitlines():
        index, free, used, utilization = [x.strip() for x in line.split(",")]
        rows.append({"index": int(index), "free_memory_mib": int(free), "used_memory_mib": int(used), "utilization_gpu_percent": int(utilization)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--source-authority", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    source = json.loads(args.source_authority.read_text(encoding="utf-8"))
    checks: dict[str, object] = {}

    if config["status"] != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise SystemExit("Z1_RUNTIME_SOURCE_NOT_FROZEN")
    checks["protocol_status"] = config["status"]
    checks["ledger_sha256"] = sha256(args.ledger)
    if checks["ledger_sha256"] != config["canary_ledger"]["sha256"]:
        raise SystemExit("CANARY_LEDGER_SHA256_MISMATCH")
    if ledger["status"] != "STAGE_Z_Z1_ENGINEERING_CANARY_LEDGER_FROZEN" or len(ledger["selected"]) != 8:
        raise SystemExit("CANARY_LEDGER_INVALID")
    checks["canary_rows"] = len(ledger["selected"])
    checks["scientific_overlap_count"] = 0
    checks["engineering_canary_key_count"] = len({r["canonical_parent_key"] for r in ledger["selected"]})

    for name in ("common_libero", "openvla_oft", "openpi"):
        spec = source["external_source_checkouts"][name]
        actual_commit = git(spec["path"], "rev-parse", "HEAD")
        actual_tree = git(spec["path"], "rev-parse", "HEAD^{tree}")
        actual_status = git(spec["path"], "status", "--short")
        if (actual_commit, actual_tree, actual_status) != (spec["commit"], spec["tree"], ""):
            raise SystemExit(f"SOURCE_AUTHORITY_MISMATCH:{name}")
        checks[name] = {"commit": actual_commit, "tree": actual_tree, "status": "CLEAN"}

    libero_config = Path(config["environment"]["libero_config_path"]) / "config.yaml"
    expected_config = "\n".join(
        [
            f"assets: {config['environment']['common_libero_checkout']}/libero/libero/assets",
            f"bddl_files: {config['environment']['common_libero_checkout']}/libero/libero/bddl_files",
            f"benchmark_root: {config['environment']['common_libero_checkout']}/libero/libero",
            f"datasets: {config['environment']['common_libero_checkout']}/datasets",
            f"init_states: {config['environment']['common_libero_checkout']}/libero/libero/init_files",
        ]
    )
    if not libero_config.is_file() or libero_config.read_text(encoding="utf-8").strip() != expected_config:
        raise SystemExit("OFFICIAL_LIBERO_CONFIG_MISMATCH")
    checks["libero_config"] = {"path": str(libero_config), "sha256": sha256(libero_config)}

    m0 = {}
    for suite, path in config["model_families"]["M0_OPENVLA"]["paths"].items():
        p = Path(path)
        if not p.is_dir():
            raise SystemExit(f"M0_CHECKPOINT_MISSING:{suite}")
        actual = inventory(p)
        if actual != M0_EXPECTED[suite]:
            raise SystemExit(f"M0_CHECKPOINT_INVENTORY_MISMATCH:{suite}:{actual}")
        m0[suite] = {"path": str(p), "files": actual[0], "bytes": actual[1]}
    checks["M0_checkpoints"] = m0

    m2_path = Path(config["model_families"]["M2_PI05_LIBERO"]["checkpoint"])
    if inventory(m2_path) != M2_EXPECTED:
        raise SystemExit(f"M2_CHECKPOINT_INVENTORY_MISMATCH:{inventory(m2_path)}")
    checks["M2_checkpoint"] = {"path": str(m2_path), "files": M2_EXPECTED[0], "bytes": M2_EXPECTED[1]}
    m1_root = Path(config["model_families"]["M1_OPENVLA_OFT"]["checkpoint_root"])
    checks["M1_sequential_materialization"] = {"root": str(m1_root), "pre_gpu_status": "PENDING_PER_CELL_MANIFEST_VERIFICATION"}

    gpus = gpu_rows()
    checks["gpu_snapshot"] = gpus
    checks["eligible_gpu_indices"] = [row["index"] for row in gpus if row["free_memory_mib"] > 20480]
    checks["foreign_process_policy"] = "record-only-never-touch"
    checks["source_authority_sha256"] = sha256(args.source_authority)
    checks["runner_git_binding"] = config["runtime_code_binding"]

    result = {
        "schema": "STAGE_Z_Z1_PRE_GPU_STATIC_AUDIT_V1",
        "status": "STAGE_Z_Z1_PRE_GPU_STATIC_AUDIT_PASS",
        "gate": config["gate"],
        "checks": checks,
        "counters": {"gpu_workers": 0, "model_inference": 0, "simulator": 0, "env_step": 0, "physical_intervention": 0, "pgd": 0, "attacked_env_steps": 0, "v_phys": 0, "eval160": 0, "protected_reads": 0, "scientific_parent_exposure": 0},
        "claim_boundary": "Pre-GPU engineering/source/resource audit only; no model or simulator execution.",
        "next_legal_action": "RUN_Z1_ENGINEERING_CANARY_CELLS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "eligible_gpu_indices": checks["eligible_gpu_indices"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
