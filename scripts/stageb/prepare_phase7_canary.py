#!/usr/bin/env python3
"""Phase 7D: Canary manifest generator + fail-closed launcher."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER_REPO = Path("/mnt/sdc/dty_user/openvla_attack")

V2_CKPT_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"
V2_CKPT_PATH = SERVER_REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt"


def fail_closed_checks(args):
    """Run all pre-launch checks. Exit non-zero if any fail."""
    errors = []

    # Check git clean
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip():
        errors.append("GIT_DIRTY: uncommitted changes present")

    # Check HEAD matches
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"  HEAD: {head[:16]}")

    # Check checkpoint SHA
    if V2_CKPT_PATH.exists():
        actual = hashlib.sha256(open(V2_CKPT_PATH, "rb").read()).hexdigest()
        match = actual == V2_CKPT_SHA
        print(f"  V2 CKPT SHA: {actual[:16]} match={match}")
        if not match:
            errors.append(f"V2_CHECKPOINT_SHA_MISMATCH: got {actual[:16]}")
    else:
        errors.append(f"V2_CHECKPOINT_MISSING: {V2_CKPT_PATH}")

    # Check backend
    from gripper_attack.openvla_preprocess import resolve_backend
    backend = resolve_backend(args.libero_preprocess_backend)
    jpeg = backend == "upstream_tf_jpeg"
    print(f"  Backend: {args.libero_preprocess_backend} -> {backend} jpeg={jpeg}")
    if not jpeg:
        errors.append(f"BACKEND_NOT_UPSTREAM_TF_JPEG: {backend}")

    # Check output dir
    out = Path(args.output_dir)
    if out.exists():
        errors.append(f"OUTPUT_EXISTS: {out}")

    # Check CUDA
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," in cuda or not cuda:
        errors.append(f"MULTI_GPU_OR_UNSET: CUDA_VISIBLE_DEVICES={cuda}")

    if errors:
        print("FAIL-CLOSED:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


def generate_canary_manifest(args):
    """Generate canary manifest CSV."""
    # Canary A: TV + TRUE_T10 (use butter_s0 from official smoke)
    # Canary B: Same TV + RAND_T10
    # Canary C: First formal NC + TRUE_T10 (TBD after census)
    rows = [
        {
            "run_id": "canary_a", "canary_type": "TV_VIS",
            "task_idx": 6, "state_id": 0, "anchor_audit": 85,
            "condition": "TRUE_T10", "attack_seed": 42, "rollout_seed": 42,
            "checkpoint": str(V2_CKPT_PATH), "checkpoint_sha": V2_CKPT_SHA,
            "backend": "upstream_tf_jpeg", "role": "ENGINEERING_CANARY",
            "included_in_main_analysis": "false",
            "output_dir": str(Path(args.output_dir) / "canary_a_tv_vis"),
        },
        {
            "run_id": "canary_b", "canary_type": "TV_RAND",
            "task_idx": 6, "state_id": 0, "anchor_audit": 85,
            "condition": "RAND_T10", "attack_seed": 42, "rollout_seed": 42,
            "checkpoint": str(V2_CKPT_PATH), "checkpoint_sha": V2_CKPT_SHA,
            "backend": "upstream_tf_jpeg", "role": "ENGINEERING_CANARY",
            "included_in_main_analysis": "false",
            "output_dir": str(Path(args.output_dir) / "canary_b_tv_rand"),
        },
        {
            "run_id": "canary_c", "canary_type": "NC",
            "task_idx": -1, "state_id": -1, "anchor_audit": -1,
            "condition": "TRUE_T10", "attack_seed": 42, "rollout_seed": 42,
            "checkpoint": str(V2_CKPT_PATH), "checkpoint_sha": V2_CKPT_SHA,
            "backend": "upstream_tf_jpeg", "role": "ENGINEERING_CANARY",
            "included_in_main_analysis": "false",
            "output_dir": str(Path(args.output_dir) / "canary_c_nc"),
            "note": "TASK_STATE_TBD_AFTER_CENSUS",
        },
    ]

    manifest_path = Path(args.output_dir) / "PHASE7_CANARY_MANIFEST.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Canary manifest: {manifest_path}")
    return manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="evidence/phase7_object")
    ap.add_argument("--libero_preprocess_backend", default="upstream_tf_jpeg")
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()

    print("=== Phase 7D: Canary Pre-flight ===")
    fail_closed_checks(args)

    if not args.check_only:
        generate_canary_manifest(args)
        print("\nCanary manifest generated. Launch with:")
        print(f"  python scripts/stageb/launch_phase7_canary.py --manifest <path>")


if __name__ == "__main__":
    main()
