#!/usr/bin/env python3
"""M1: A800-local paired runtime profile study.

Evaluates clean policy + frozen Object detector under 3 profiles on same GPU:
  B0: BF16 + Eager (baseline)
  A1: BF16 + FlashAttention2
  D1: FP32 + Eager

Manifest: 30 episodes (10 tasks × states 0,1,2) from libero_object.
Latin-square ordering to mitigate order/cache/warmup confounding.
"""
import os, sys, json, hashlib, time, csv, argparse, subprocess
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

MANIFEST_PATH = os.path.join(REPO, "migration_audit", "object_checkpoint_migration", "manifests", "m1_object_30.json")
CONTRACT_PATH = os.path.join(REPO, "migration_audit", "object_checkpoint_migration", "m1_runtime", "m1_contract.json")
OUT_DIR = os.path.join(REPO, "migration_audit", "object_checkpoint_migration", "m1_runtime")
EVIDENCE_DIR = os.path.join(REPO, "evidence", "object_checkpoint_migration", "m1_runtime")

MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", os.path.join(REPO, "models", "openvla-7b-finetuned-libero-object"))
CKPT_PATH = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")

PROFILES = {
    "B0": {"dtype": "bfloat16", "attn": "eager", "label": "BF16+Eager"},
    "A1": {"dtype": "bfloat16", "attn": "flash_attention_2", "label": "BF16+Flash2"},
    "D1": {"dtype": "float32", "attn": "eager", "label": "FP32+Eager"},
}

# Latin-square groups: (group_id, [profile_order])
# 30 episodes → 3 groups of 10
LATIN_SQUARE = [
    (1, ["B0", "A1", "D1"]),
    (2, ["A1", "D1", "B0"]),
    (3, ["D1", "B0", "A1"]),
]

# Sentinel episodes
SENTINEL_KEYS = [
    "butter_s0",       # known A800 success
    "butter_s2",       # known A800 fail
    "ketchup_s1",      # borderline
    "tomato_sauce_s1", # borderline
    "milk_s0",         # easy-success candidate
    "orange_juice_s0", # easy-success candidate
]


def load_manifest():
    return json.load(open(MANIFEST_PATH))


def build_latin_square_schedule(manifest):
    """Assign each episode to a Latin-square group and order profiles."""
    episodes = manifest["episodes"]
    schedule = []
    for i, ep in enumerate(episodes):
        group_idx = i % 3
        group_id, profile_order = LATIN_SQUARE[group_idx]
        for profile in profile_order:
            schedule.append({
                "episode": ep,
                "profile": profile,
                "group": group_id,
            })
    return schedule


def run_episode(ep, profile_key, gpu):
    """Run one episode with the given profile on the specified GPU.

    Returns dict with results or None on failure.
    Uses subprocess to call the bridge with profile-specific env vars.
    """
    profile = PROFILES[profile_key]
    episode_key = ep["episode_key"]
    label = f"{episode_key}_{profile_key}"
    cell_dir = os.path.join(EVIDENCE_DIR, label)
    os.makedirs(cell_dir, exist_ok=True)

    bridge = os.path.join(REPO, "scripts", "stageb", "run_v2_vis_sc5_mlp_bridge.py")
    python = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"

    cmd = [
        python, bridge,
        "--condition", "CLEAN",
        "--state_id", str(ep["state_id"]),
        "--task_idx", str(ep["task_idx"]),
        "--anchor", "999",  # dummy — CLEAN mode ignores anchor
        "--seed_id", "42",
        "--output_dir", cell_dir,
        "--render_gpu", str(gpu),
        "--mlp_path", CKPT_PATH,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
    env["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
    env["OPENVLA_ATTN_IMPLEMENTATION"] = profile["attn"]
    env["OPENVLA_DTYPE"] = profile["dtype"]
    env["OPENVLA_MODEL_PATH"] = MODEL_PATH

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0

    # Save logs
    with open(os.path.join(cell_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(cell_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    # Parse result
    summary_path = os.path.join(cell_dir, "episode_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    # Detect FlashAttention2 fallback
    actual_attn = "unknown"
    if profile_key == "A1":
        stderr_lower = result.stderr.lower()
        if "flash attention" in stderr_lower and "not support" in stderr_lower:
            actual_attn = "fallback_eager"
        elif "flash" in stderr_lower:
            actual_attn = "flash_attention_2"
        else:
            # Check stdout for model loading message
            stdout_lower = result.stdout.lower()
            if "flash" in stdout_lower:
                actual_attn = "flash_attention_2"
            else:
                actual_attn = "unknown"

    telemetry_path = os.path.join(cell_dir, "step_telemetry.csv")
    telemetry_sha = hashlib.sha256(open(telemetry_path, "rb").read()).hexdigest() if os.path.exists(telemetry_path) else "MISSING"

    return {
        "episode_key": episode_key,
        "task_idx": ep["task_idx"],
        "task_name": ep["task_name"],
        "state_id": ep["state_id"],
        "profile": profile_key,
        "requested_dtype": profile["dtype"],
        "requested_attn": profile["attn"],
        "actual_attn": actual_attn,
        "exit_code": result.returncode,
        "duration_s": round(dt, 1),
        "telemetry_sha": telemetry_sha,
        "success": summary.get("task_success", False),
        "emit_step": summary.get("mlp_emit_step", -1),
        "steps": summary.get("n_steps", -1),
        "anchor_error": summary.get("anchor_error", None),
        "attack_frames": summary.get("attack_frames", 0),
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser(description="M1 Runtime Profile Study")
    ap.add_argument("--main_gpu", type=int, default=2, help="GPU for main matrix")
    ap.add_argument("--sentinel_gpu", type=int, default=3, help="GPU for sentinel repeats")
    ap.add_argument("--dry_run", action="store_true", help="Validate schedule without running")
    ap.add_argument("--sentinel_only", action="store_true", help="Run only sentinel episodes")
    ap.add_argument("--profile", choices=["B0","A1","D1"], help="Run only one profile")
    args = ap.parse_args()

    manifest = load_manifest()
    schedule = build_latin_square_schedule(manifest)

    print(f"M1 Runtime Profile Study")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Episodes: {len(manifest['episodes'])}")
    print(f"  Total runs (main): {len(schedule)}")
    print(f"  Main GPU: {args.main_gpu}")
    print(f"  Sentinel GPU: {args.sentinel_gpu}")
    print(f"  Profiles: {list(PROFILES.keys())}")

    if args.profile:
        schedule = [s for s in schedule if s["profile"] == args.profile]
        print(f"  Filtered to profile {args.profile}: {len(schedule)} runs")

    if args.dry_run:
        print("\nSchedule (dry run):")
        for i, s in enumerate(schedule):
            print(f"  {i:3d}: {s['episode']['episode_key']:25s} {s['profile']} group={s['group']}")
        print("\nSentinel episodes:")
        for sk in SENTINEL_KEYS:
            print(f"  {sk}")
        return

    # --- Sentinel repeatability (run first to fail fast) ---
    sentinel_results = []
    if not args.sentinel_only or True:  # Always run sentinel
        print("\n=== Sentinel Repeatability (GPU {}) ===".format(args.sentinel_gpu))
        sentinel_eps = [ep for ep in manifest["episodes"] if ep["episode_key"] in SENTINEL_KEYS]
        for profile_key in (["B0", "A1", "D1"] if not args.profile else [args.profile]):
            for ep in sentinel_eps:
                for repeat in range(2):
                    label = f"sentinel_{ep['episode_key']}_{profile_key}_r{repeat}"
                    print(f"  {label}...")
                    result = run_episode(ep, profile_key, args.sentinel_gpu)
                    result["sentinel_repeat"] = repeat
                    sentinel_results.append(result)
                    if result["telemetry_sha"] == "MISSING":
                        print(f"    WARNING: telemetry missing — bridge may have crashed")

        # Write sentinel results
        sentinel_path = os.path.join(OUT_DIR, "sentinel_repeatability.csv")
        with open(sentinel_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sentinel_results[0].keys())
            w.writeheader()
            w.writerows(sentinel_results)
        print(f"  Sentinel results: {sentinel_path}")

    if args.sentinel_only:
        return

    # --- Main matrix (GPU2) ---
    print(f"\n=== Main Matrix (GPU {args.main_gpu}) ===")
    print(f"  Schedule: {len(schedule)} runs, Latin-square groups")

    all_results = []
    failure_ledger = []
    for i, s in enumerate(schedule):
        ep = s["episode"]
        profile_key = s["profile"]
        label = f"{ep['episode_key']}_{profile_key}"
        print(f"\n[{i+1}/{len(schedule)}] {label} (group={s['group']})")

        result = run_episode(ep, profile_key, args.main_gpu)
        all_results.append(result)

        if result["telemetry_sha"] == "MISSING":
            failure_ledger.append({
                "index": i,
                "episode_key": ep["episode_key"],
                "profile": profile_key,
                "error": "telemetry_missing",
            })
            print(f"    FAILED: telemetry missing")

        # Flush results incrementally
        results_path = os.path.join(OUT_DIR, "episode_results.csv")
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)

        # Check FlashAttention2 fallback
        if profile_key == "A1" and result.get("actual_attn") == "fallback_eager":
            print(f"    FATAL: FlashAttention2 fallback detected — marking A1 INVALID")
            failure_ledger.append({
                "index": i,
                "episode_key": ep["episode_key"],
                "profile": profile_key,
                "error": "flash_attention_fallback",
            })

    # Save final outputs
    json.dump(all_results, open(os.path.join(OUT_DIR, "profile_results.json"), "w"), indent=2)
    json.dump(failure_ledger, open(os.path.join(OUT_DIR, "failure_ledger.json"), "w"), indent=2)

    print(f"\n=== Complete ===")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Failures: {len(failure_ledger)}")
    print(f"  Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
