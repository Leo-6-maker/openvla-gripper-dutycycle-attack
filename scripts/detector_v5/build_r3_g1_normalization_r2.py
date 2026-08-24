"""Rebuild G1 normalization using the CURRENT feature adapter and exact train identities.

Reads the frozen G1 split manifests to get train identities (episode-heldout only).
Materializes features using the current feature adapter code, then saves a new
NORMALIZATION.json sealed root.  Does NOT modify the G1 split manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace
from run_r3_full670_student_development import _load_records, _load_t4

FORBIDDEN = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(p: Path) -> str:
    files = sorted(
        x for x in p.rglob("*")
        if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files),
        encoding="utf-8",
    )
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def build(*, t4_root: Path, g1_root: Path, output_root: Path) -> dict[str, Any]:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("clean checkout required")

    # Validate G1
    t4_root = t4_root.resolve(strict=True)
    g1_root = g1_root.resolve(strict=True)
    _ = verify_seal(g1_root)
    _ = verify_seal(t4_root)

    audit = json.loads((g1_root / "G1_SPLIT_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_SPLIT_CLOSURE_WITH_HEAD_COVERAGE_FLAGS":
        raise ValueError("G1 is not passing split audit")
    if audit.get("checks", {}).get("protected_reads") != 0:
        raise ValueError("G1 has protected reads")

    train_manifest = json.loads((g1_root / "EPISODE_TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    train_ids = [row["episode_id"] for row in train_manifest]
    if len(train_ids) != 445:
        raise ValueError(f"expected 445 episode-train identities, got {len(train_ids)}")

    # Materialize features using CURRENT adapter
    print(f"Materializing features for {len(train_ids)} train identities...")
    transition, *_ = _load_t4(t4_root, allow_descendant_snapshot=True, skip_source_binding=True)
    records, _ = _load_records(
        t4_root, allow_descendant_snapshot=True,
        identity_allowlist=set(train_ids), skip_source_binding=True,
    )
    records_by_id = {r["identity"]: r for r in records}

    all_features = np.concatenate(
        [records_by_id[i]["features"] for i in train_ids], axis=0
    )
    total_steps = len(all_features)
    print(f"Total train steps: {total_steps}")

    mean = all_features.mean(axis=0).astype(np.float64)
    std = np.maximum(all_features.std(axis=0, ddof=0), 1e-8).astype(np.float64)

    # Compare with existing G1 normalization
    old_norm = json.loads((g1_root / "NORMALIZATION.json").read_text(encoding="utf-8"))
    old_mean = np.asarray(old_norm["episode_heldout"]["train"]["mean"], dtype=np.float64)
    old_std = np.asarray(old_norm["episode_heldout"]["train"]["std"], dtype=np.float64)

    FEATURE_ORDER = [
        "gripper_command", "gripper_qpos", "gripper_opening_proxy",
        "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
        "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
        "close_onset", "time_since_close", "eef_speed", "eef_z_delta_since_close",
        "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
        "opening_proxy_variance_5", "eef_speed_variance_5",
    ]

    drift_report = []
    for i in range(25):
        md = abs(float(mean[i] - old_mean[i]))
        sd = abs(float(std[i] - old_std[i]))
        drift_report.append({
            "dim": i, "feature": FEATURE_ORDER[i],
            "old_mean": float(old_mean[i]), "new_mean": float(mean[i]),
            "old_std": float(old_std[i]), "new_std": float(std[i]),
            "mean_abs_diff": md, "std_abs_diff": sd,
        })
    max_mean_drift = max(d["mean_abs_diff"] for d in drift_report)
    max_std_drift = max(d["std_abs_diff"] for d in drift_report)
    print(f"Max mean drift vs G1: {max_mean_drift:.6e}")
    print(f"Max std drift vs G1:  {max_std_drift:.6e}")

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")

    # Build output
    normalization = {
        "episode_heldout": {
            "train": {
                "source_split": "train",
                "identity_count": len(train_ids),
                "step_count": int(total_steps),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "dtype": "float64",
                "ddof": 0,
                "feature_order": FEATURE_ORDER,
                "feature_order_sha256": hashlib.sha256(
                    json.dumps(FEATURE_ORDER, separators=(",", ":"), ensure_ascii=True).encode()
                ).hexdigest(),
            },
        },
        "task_heldout": old_norm.get("task_heldout", {}),
    }

    payload = {
        "schema": "V5_R3_G1_NORMALIZATION_R2_V1",
        "status": "PASS_RECOMPUTED_FROM_CURRENT_ADAPTER",
        "code_snapshot": {"commit": commit, "tree": tree},
        "g1_root": str(g1_root),
        "g1_seal_sha256sums_sha256": verify_seal(g1_root)["sha256sums_sha256"],
        "t4_root": str(t4_root),
        "t4_seal_sha256sums_sha256": verify_seal(t4_root)["sha256sums_sha256"],
        "train_identity_count": len(train_ids),
        "train_step_count": int(total_steps),
        "normalization": normalization,
        "drift_vs_g1_r1": {
            "max_mean_abs_diff": max_mean_drift,
            "max_std_abs_diff": max_std_drift,
            "per_dimension": drift_report,
        },
        "feature_binding_sha256": sha256_file(ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json"),
        "adapter_source_sha256": sha256_file(ROOT / "src" / "gripper_attack" / "sc5_streaming_features_v2.py"),
        "builder_sha256": sha256_file(Path(__file__)),
        "protected_reads": 0,
        "attack_authorized": False,
    }

    # Verify output root
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(str(output_root))
    if any(p.casefold() in FORBIDDEN for p in output_root.parts):
        raise ValueError("output under forbidden path")
    if output_root.parent.resolve(strict=True) != g1_root.parent:
        raise ValueError("output must be G1 sibling")

    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(str(staging))
    staging.mkdir(parents=True)
    try:
        (staging / "NORMALIZATION.json").write_text(
            json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "NORMALIZATION_REBUILD_REPORT.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
    parser.add_argument("--t4-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        t4_root=args.t4_root, g1_root=args.g1_root, output_root=args.output_root,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
