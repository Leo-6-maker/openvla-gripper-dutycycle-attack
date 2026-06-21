#!/usr/bin/env python3
"""Orchestrate upstream SC5 pipeline — stages R0 through E0.

Read-only controller: checks dependencies, applies GPU leases, launches stages,
verifies exit codes, and writes pipeline state. Never modifies code or relaxes thresholds.
"""
import os, sys, json, hashlib, subprocess, time, argparse, shutil
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_STATE = os.path.join(REPO_ROOT, "migration_audit", "detector", "pipeline_state.json")
LEASE_DIR = os.path.join(REPO_ROOT, "migration_audit", "detector", "leases")

# Allowed GPUs
ALLOWED_GPUS = [2, 3, 4, 6]

# Stage definitions
STAGES = [
    "R0_collector_repair",
    "C0_observer_canary",
    "C1_parallel_collection",
    "C2_corpus_integrity",
    "T0_training",
    "T1_calibration",
    "X0_transfer_test",
    "E0_eval19_verification",
]


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def load_state():
    if os.path.exists(PIPELINE_STATE):
        with open(PIPELINE_STATE) as f:
            return json.load(f)
    return {"stages": {}, "pipeline": "UPSTREAM_SC5_AUTOMATED_PIPELINE",
            "started_at": datetime.now(timezone.utc).isoformat()}


def save_state(state):
    os.makedirs(os.path.dirname(PIPELINE_STATE), exist_ok=True)
    with open(PIPELINE_STATE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def stage_marker(state, stage, status, metrics=None, pid=None, exit_code=None):
    state["stages"][stage] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": pid, "exit_code": exit_code,
        "metrics": metrics or {},
    }
    save_state(state)


def acquire_gpu_lease(gpu_id, stage_name):
    """Create a lease file for exclusive GPU use."""
    os.makedirs(LEASE_DIR, exist_ok=True)
    lease_path = os.path.join(LEASE_DIR, "gpu%d.lease" % gpu_id)
    if os.path.exists(lease_path):
        with open(lease_path) as f:
            existing = json.load(f)
        # Lease auto-expires after 24 hours
        try:
            ts = datetime.fromisoformat(existing["timestamp"])
            if (datetime.now(timezone.utc) - ts).total_seconds() < 86400:
                return False, existing.get("stage", "unknown")
        except Exception:
            pass
    with open(lease_path, "w") as f:
        json.dump({"gpu": gpu_id, "stage": stage_name,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "pid": os.getpid()}, f)
    return True, None


def release_gpu_lease(gpu_id):
    lease_path = os.path.join(LEASE_DIR, "gpu%d.lease" % gpu_id)
    if os.path.exists(lease_path):
        os.remove(lease_path)


def check_resources():
    """Check disk and GPU resources."""
    issues = []
    root_stat = shutil.disk_usage("/mnt/sdc")
    if root_stat.free < 78 * 1024**3:
        issues.append("disk_low: /mnt/sdc free=%d GiB" % (root_stat.free / 1024**3))
    return issues


def run_stage_cmd(cmd, cwd, env=None):
    """Run a stage command, return (exit_code, stdout, stderr)."""
    full_env = os.environ.copy()
    full_env.setdefault("HOME", "/mnt/sdc/dty_user/openvla_attack/sandbox_home")
    full_env.setdefault("TMPDIR", "/mnt/sdc/dty_user/openvla_attack/tmp")
    full_env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    full_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES + ["status", "reset"])
    parser.add_argument("--gpu", type=int, default=None, choices=ALLOWED_GPUS)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--model_path", default="models/libero-spatial/spatial_c8f03f4_20260620")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    state = load_state()
    python = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"
    repo = "/mnt/sdc/dty_user/openvla_attack"

    if args.stage == "status":
        print(json.dumps(state, indent=2, default=str))
        return 0

    if args.stage == "reset":
        if os.path.exists(PIPELINE_STATE):
            os.remove(PIPELINE_STATE)
        for f in os.listdir(LEASE_DIR) if os.path.exists(LEASE_DIR) else []:
            os.remove(os.path.join(LEASE_DIR, f))
        print("Pipeline state and leases reset.")
        return 0

    # === R0: Collector repair verification ===
    if args.stage == "R0_collector_repair":
        print("R0: Verifying collector uses canonical modules...")
        collector = os.path.join(repo, "scripts/detector/run_upstream_artifact_clean.py")
        with open(collector) as f:
            content = f.read()
        checks = {
            "no_local_SC5_FEATURES": "SC5_FEATURES = [" not in content.split("from gripper_attack.sc5_streaming_features_v2 import")[0] if "from gripper_attack.sc5_streaming_features_v2 import" in content else False,
            "imports_SC5StreamingFeatureAdapterV2": "SC5StreamingFeatureAdapterV2" in content,
            "imports_openvla_preprocess": "from gripper_attack.openvla_preprocess import" in content,
            "imports_sc5mlp_v1": "sc5mlp_v1" in content.lower(),
            "no_qpos_formula": "1.0 - qpos_scalar" not in content,
            "has_mujoco_pose": "_read_mujoco_pose" in content,
            "has_binding_verify": "_verify_binding" in content,
            "has_safe_resume": "episode_manifest.json" in content and "INTEGRITY_FAIL" in content,
        }
        if args.dry_run:
            print("Checks:", json.dumps(checks, indent=2))
        stage_marker(state, args.stage, "PASS" if all(checks.values()) else "FAIL",
                     metrics=checks)
        print("R0: %s" % ("PASS" if all(checks.values()) else "FAIL"))
        return 0 if all(checks.values()) else 1

    # === C0: Observer non-interference canary ===
    if args.stage == "C0_observer_canary":
        if args.gpu is None:
            print("C0 requires --gpu")
            return 1
        ok, holder = acquire_gpu_lease(args.gpu, args.stage)
        if not ok:
            print("C0: GPU%d already leased by %s" % (args.gpu, holder))
            return 1

        resources = check_resources()
        if resources:
            print("C0: Resource check failed: %s" % resources)
            release_gpu_lease(args.gpu)
            return 1

        shard_path = os.path.join(repo, "configs/detector/shard/c0_smoke.json")
        # Create C0 smoke shard on the fly
        c0_eps = [
            {"task_idx": 0, "init_idx": 13, "label": "task0_init13", "split": "c0_smoke"},
            {"task_idx": 5, "init_idx": 13, "label": "task5_init13", "split": "c0_smoke"},
        ]
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        json.dump({"plan_name": "c0_smoke", "episodes": c0_eps}, open(shard_path, "w"))

        print("C0: Launching collector on GPU%d..." % args.gpu)
        cmd = [
            python, os.path.join(repo, "scripts/detector/run_upstream_artifact_clean.py"),
            "--episode_manifest", shard_path,
            "--model_path", args.model_path,
            "--output_dir", os.path.join(repo, "evidence/c0_artifact_canary_gpu%d" % args.gpu),
            "--dtype", "float32", "--attn", "eager",
            "--profile_name", "fp32_eager",
        ]

        if args.dry_run:
            print("DRY_RUN: %s" % " ".join(cmd))
            release_gpu_lease(args.gpu)
            return 0

        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_GL": "egl"}
        exit_code, stdout, stderr = run_stage_cmd(cmd, repo, env)
        print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        if stderr:
            print("STDERR:", stderr[-500:])

        success = exit_code == 0
        stage_marker(state, args.stage, "PASS" if success else "FAIL",
                     exit_code=exit_code, metrics={"exit_code": exit_code})
        release_gpu_lease(args.gpu)
        print("C0: %s (exit=%d)" % ("PASS" if success else "FAIL", exit_code))
        return 0 if success else 1

    # === C1: Parallel collection (launch only — does not wait) ===
    if args.stage == "C1_parallel_collection":
        if not args.gpu or not args.shard:
            print("C1 requires --gpu AND --shard")
            return 1

        ok, holder = acquire_gpu_lease(args.gpu, "%s_%s" % (args.stage, args.shard))
        if not ok:
            print("C1: GPU%d already leased by %s" % (args.gpu, holder))
            return 1

        shard_path = os.path.join(repo, "configs/detector/shard/%s.json" % args.shard)
        if not os.path.exists(shard_path):
            print("C1: Shard not found: %s" % shard_path)
            release_gpu_lease(args.gpu)
            return 1

        # Determine profile from shard name
        is_flash2 = "flash2" in args.shard
        dtype = "bfloat16" if is_flash2 else "float32"
        attn = "flash_attention_2" if is_flash2 else "eager"
        profile = "bf16_flash2" if is_flash2 else "fp32_eager"

        out_dir = os.path.join(repo, "evidence/sc5_corpus_%s" % args.shard)
        cmd = [
            python, os.path.join(repo, "scripts/detector/run_upstream_artifact_clean.py"),
            "--episode_manifest", shard_path,
            "--model_path", args.model_path,
            "--output_dir", out_dir,
            "--dtype", dtype, "--attn", attn,
            "--profile_name", profile,
            "--resume",
        ]
        print("C1: Launching %s on GPU%d -> %s" % (args.shard, args.gpu, out_dir))

        if args.dry_run:
            print("DRY_RUN: %s" % " ".join(cmd))
            release_gpu_lease(args.gpu)
            return 0

        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_GL": "egl"}
        exit_code, stdout, stderr = run_stage_cmd(cmd, repo, env)
        print(stdout[-2000:] if len(stdout) > 2000 else stdout)

        stage_marker(state, "%s_gpu%d_%s" % (args.stage, args.gpu, args.shard),
                     "COMPLETE" if exit_code == 0 else "FAIL",
                     exit_code=exit_code, metrics={"shard": args.shard, "gpu": args.gpu})
        release_gpu_lease(args.gpu)
        return 0 if exit_code == 0 else 1

    # === C2: Corpus integrity check ===
    if args.stage == "C2_corpus_integrity":
        dirs = [
            os.path.join(repo, "evidence/sc5_corpus_fp32_train_a"),
            os.path.join(repo, "evidence/sc5_corpus_fp32_train_b"),
            os.path.join(repo, "evidence/sc5_corpus_fp32_val_cal_xfer"),
            os.path.join(repo, "evidence/sc5_corpus_flash2_xfer"),
        ]
        cmd = [
            python, os.path.join(repo, "scripts/detector/merge_upstream_artifact_corpus.py"),
            "--input_dirs"] + dirs + [
            "--output_dir", os.path.join(repo, "tables"),
            "--split_labels", "train", "train", "val_cal_xfer_fp32", "xfer_flash2",
        ]
        exit_code, stdout, stderr = run_stage_cmd(cmd, repo)
        print(stdout[-3000:] if len(stdout) > 3000 else stdout)
        stage_marker(state, args.stage, "PASS" if exit_code == 0 else "FAIL",
                     exit_code=exit_code)
        return 0 if exit_code == 0 else 1

    # === T0/T1/X0/E0: Training, calibration, transfer, EVAL19 ===
    # These stages are defined but require model training infrastructure
    # that runs on A800 with GPU. Placeholder for now.
    for stage_name in ["T0_training", "T1_calibration", "X0_transfer_test", "E0_eval19_verification"]:
        if args.stage == stage_name:
            print("%s: Stage defined but requires A800 execution with trained model." % stage_name)
            print("Prerequisites: C2 must PASS, teacher labeling must be complete.")
            stage_marker(state, args.stage, "PENDING_PREREQUISITES")
            return 0

    print("Unknown stage: %s" % args.stage)
    return 1


if __name__ == "__main__":
    sys.exit(main())
