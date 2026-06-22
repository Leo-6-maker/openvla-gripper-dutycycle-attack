#!/usr/bin/env python3
"""M1: A800-local paired runtime profile study.

Evaluates clean policy + frozen Object detector under 3 profiles on same GPU:
  B0: BF16 + Eager (baseline)
  A1: BF16 + FlashAttention2
  D1: FP32 + Eager

Manifest: 30 episodes (10 tasks x states 0,1,2) from libero_object.
Latin-square ordering. Sentinel repeatability on separate GPU.
Evidence directories are isolated: smoke/, sentinel/<ep>/<profile>/repeat_N/, main/<group>/<ep>/<profile/

Flash2 fallback = immediate full stop. anchor=null for clean.
"""
import os, sys, json, hashlib, time, csv, argparse, subprocess, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/manifests/m1_object_30.json"
OUT_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime"
BRIDGE = REPO / "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"
CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", str(REPO / "models/openvla-7b-finetuned-libero-object"))
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"

PROFILES = {
    "B0": {"dtype": "bfloat16", "attn": "eager", "label": "BF16+Eager"},
    "A1": {"dtype": "bfloat16", "attn": "flash_attention_2", "label": "BF16+Flash2"},
    "D1": {"dtype": "float32", "attn": "eager", "label": "FP32+Eager"},
}

LATIN_SQUARE = [
    (1, ["B0", "A1", "D1"]),
    (2, ["A1", "D1", "B0"]),
    (3, ["D1", "B0", "A1"]),
]

SENTINEL_KEYS = [
    "butter_s0", "butter_s2", "ketchup_s1",
    "tomato_sauce_s1", "milk_s0", "orange_juice_s0",
]


def env_for_profile(profile_key, gpu):
    p = PROFILES[profile_key]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
    env["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
    env["OPENVLA_DTYPE"] = p["dtype"]
    env["OPENVLA_ATTN_IMPLEMENTATION"] = p["attn"]
    env["OPENVLA_MODEL_PATH"] = MODEL_PATH
    return env


def run_episode(ep, profile_key, gpu, output_dir, source_commit, save_video=False):
    """Run one episode. Returns result dict. Raises on critical failure."""
    cell_dir = Path(output_dir)
    done_file = cell_dir / ".done"

    if done_file.exists():
        existing = json.load(open(done_file))
        if existing.get("telemetry_sha") and existing["telemetry_sha"] != "MISSING":
            print(f"    SKIP: already complete (telemetry_sha={existing['telemetry_sha'][:12]})")
            return existing

    # Clean incomplete dir
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)

    cmd = [
        PYTHON, str(BRIDGE),
        "--condition", "CLEAN",
        "--state_id", str(ep["state_id"]),
        "--task_idx", str(ep["task_idx"]),
        "--anchor", "0",  # null anchor — CLEAN ignores it; error computed later from Teacher
        "--seed_id", "42",
        "--output_dir", str(cell_dir),
        "--render_gpu", str(gpu),
        "--mlp_path", str(CKPT),
    ]
    if save_video:
        cmd.extend(["--save_video", "--source_commit", source_commit])

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO), env=env_for_profile(profile_key, gpu),
                            capture_output=True, text=True)
    dt = time.time() - t0

    # Save logs
    (cell_dir / "stdout.log").write_text(result.stdout)
    (cell_dir / "stderr.log").write_text(result.stderr)

    # Parse
    summary_path = cell_dir / "episode_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    telemetry_path = cell_dir / "step_telemetry.csv"
    telemetry_sha = hashlib.sha256(open(telemetry_path, "rb").read()).hexdigest() if telemetry_path.exists() else "MISSING"

    r = {
        "episode_key": ep["episode_key"], "task_idx": ep["task_idx"],
        "task_name": ep["task_name"], "state_id": ep["state_id"],
        "profile": profile_key,
        "requested_dtype": PROFILES[profile_key]["dtype"],
        "requested_attn": PROFILES[profile_key]["attn"],
        "actual_dtype": summary.get("actual_dtype", "unknown"),
        "actual_attn": summary.get("actual_attn", "unknown"),
        "exit_code": result.returncode, "duration_s": round(dt, 1),
        "telemetry_sha": telemetry_sha,
        "success": summary.get("task_success", False),
        "emit_step": summary.get("mlp_emit_step", -1),
        "steps": summary.get("n_steps", -1),
        "attack_frames": summary.get("attack_frames", 0),
        "checkpoint_sha": summary.get("checkpoint_sha256", "")[:16],
        "gpu": gpu,
        "output_dir": str(cell_dir),
    }

    # Validate runtime attestation
    if profile_key == "A1":
        if r["actual_attn"] != "flash_attention_2":
            r["flash2_fallback"] = True
            r["flash2_actual"] = r["actual_attn"]
            print(f"    FATAL: Flash2 fallback! actual_attn={r['actual_attn']}")
            # Write done file with failure marker
            r["_fatal"] = "flash2_fallback"
            json.dump(r, open(done_file, "w"))
            return r
    if profile_key == "D1":
        if r["actual_dtype"] != "float32":
            r["fp32_fallback"] = True
            print(f"    FATAL: FP32 fallback! actual_dtype={r['actual_dtype']}")
            r["_fatal"] = "fp32_fallback"
            json.dump(r, open(done_file, "w"))
            return r

    # Validate basics
    if result.returncode != 0:
        r["_fatal"] = f"exit_code={result.returncode}"
        print(f"    FATAL: non-zero exit code {result.returncode}")
    if telemetry_sha == "MISSING":
        r["_fatal"] = "telemetry_missing"
        print(f"    FATAL: telemetry missing")
    if r["attack_frames"] != 0:
        r["_fatal"] = "attack_frames_nonzero"
        print(f"    FATAL: attack_frames={r['attack_frames']}")

    json.dump(r, open(done_file, "w"))
    return r


def run_smoke(source_commit, gpu):
    """Three-profile smoke on butter_s0."""
    manifest = json.load(open(MANIFEST_PATH))
    ep = [e for e in manifest["episodes"] if e["episode_key"] == "butter_s0"][0]
    results = {}
    for pk in ["B0", "A1", "D1"]:
        out_dir = OUT_BASE / "smoke" / pk / "butter_s0"
        print(f"\n=== SMOKE {pk} ===")
        r = run_episode(ep, pk, gpu, str(out_dir), source_commit, save_video=True)
        results[pk] = r
        if r.get("_fatal"):
            print(f"SMOKE {pk} FAILED: {r['_fatal']}")
            if pk == "A1" and r.get("flash2_fallback"):
                print("FLASH2 NOT AVAILABLE — stopping M1")
                sys.exit(1)
    return results


def run_sentinel(source_commit, gpu):
    """6 episodes x 3 profiles x 2 repeats."""
    manifest = json.load(open(MANIFEST_PATH))
    sentinel_eps = [e for e in manifest["episodes"] if e["episode_key"] in SENTINEL_KEYS]
    all_results = []
    for pk in PROFILES:
        for ep in sentinel_eps:
            for repeat in range(2):
                out_dir = OUT_BASE / "sentinel" / ep["episode_key"] / pk / f"repeat_{repeat}"
                label = f"sentinel/{ep['episode_key']}/{pk}/r{repeat}"
                print(f"\n=== SENTINEL {label} ===")
                r = run_episode(ep, pk, gpu, str(out_dir), source_commit, save_video=True)
                r["sentinel_repeat"] = repeat
                all_results.append(r)
                if r.get("_fatal"):
                    print(f"SENTINEL FATAL: {r['_fatal']}")
                    if r.get("flash2_fallback"):
                        sys.exit(1)
    # Write summary
    csv_path = OUT_BASE / "sentinel_repeatability.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSentinel results: {csv_path}")
    return all_results


def run_main_matrix(source_commit, gpu):
    """30 episodes x 3 profiles with Latin-square ordering."""
    manifest = json.load(open(MANIFEST_PATH))
    episodes = manifest["episodes"]

    # Build Latin-square schedule
    schedule = []
    for i, ep in enumerate(episodes):
        group_idx = i % 3
        group_id, profile_order = LATIN_SQUARE[group_idx]
        for pk in profile_order:
            schedule.append({"episode": ep, "profile": pk, "group": group_id})

    all_results = []
    fatal_count = 0
    for i, s in enumerate(schedule):
        ep = s["episode"]
        pk = s["profile"]
        out_dir = OUT_BASE / "main" / f"group_{s['group']}" / ep["episode_key"] / pk
        label = f"[{i+1}/{len(schedule)}] g{s['group']} {ep['episode_key']}/{pk}"
        print(f"\n{label}")

        r = run_episode(ep, pk, gpu, str(out_dir), source_commit, save_video=False)
        r["group"] = s["group"]
        r["schedule_index"] = i
        all_results.append(r)

        if r.get("_fatal"):
            fatal_count += 1
            if r.get("flash2_fallback"):
                print("FLASH2 FALLBACK — stopping main matrix")
                break

        # Incremental save
        csv_path = OUT_BASE / "episode_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)

        # Heartbeat
        heartbeat = {"last_episode": ep["episode_key"], "last_profile": pk,
                     "completed": i+1, "total": len(schedule), "fatal_count": fatal_count,
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(heartbeat, open(OUT_BASE / "heartbeat.json", "w"))

    return all_results


def main():
    ap = argparse.ArgumentParser(description="M1 Runtime Profile Study")
    ap.add_argument("--source_commit", required=True, help="Git commit SHA for provenance")
    ap.add_argument("--main_gpu", type=int, default=2)
    ap.add_argument("--sentinel_gpu", type=int, default=3)
    ap.add_argument("--smoke_only", action="store_true")
    ap.add_argument("--sentinel_only", action="store_true")
    ap.add_argument("--main_only", action="store_true")
    ap.add_argument("--profile", choices=["B0","A1","D1"])
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"M1 Runtime Profile Study — commit={args.source_commit}")
    print(f"  Output: {OUT_BASE}")

    # Smoke
    if not args.sentinel_only and not args.main_only:
        smoke_results = run_smoke(args.source_commit, args.main_gpu)
        json.dump(smoke_results, open(OUT_BASE / "smoke_results.json", "w"), indent=2)
        print("\n=== SMOKE COMPLETE ===")
        for pk, r in smoke_results.items():
            status = "PASS" if not r.get("_fatal") else f"FAIL({r['_fatal']})"
            print(f"  {pk}: {status} succ={r['success']} dtype={r['actual_dtype']} attn={r['actual_attn']}")
    if args.smoke_only:
        return

    # Sentinel
    if not args.main_only:
        sentinel_results = run_sentinel(args.source_commit, args.sentinel_gpu)
        print(f"\n=== SENTINEL COMPLETE: {len(sentinel_results)} runs ===")
    if args.sentinel_only:
        return

    # Main matrix
    main_results = run_main_matrix(args.source_commit, args.main_gpu)
    print(f"\n=== MAIN MATRIX COMPLETE: {len(main_results)} runs ===")

    json.dump(main_results, open(OUT_BASE / "profile_results.json", "w"), indent=2)
    print(f"Results: {OUT_BASE}")


if __name__ == "__main__":
    main()
