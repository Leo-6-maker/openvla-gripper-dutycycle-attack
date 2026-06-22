#!/usr/bin/env python3
"""M1B: B0 + D1 dual-profile paired runtime study.

B0 = BF16+Eager (baseline), D1 = FP32+Eager.
30 episodes x 2 profiles = 60 main runs on GPU2.
6 sentinel x 2 profiles x 2 repeats = 24 runs on GPU3.
AB/BA balanced ordering. Same-GPU smoke required before main matrix.
"""
import os, sys, json, hashlib, time, csv, argparse, subprocess, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/manifests/m1_object_30.json"
OUT_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime_b0_d1"
BRIDGE = REPO / "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"
CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", str(REPO / "models/openvla-7b-finetuned-libero-object"))
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"

ACTIVE_PROFILES = {
    "B0": {"dtype": "bfloat16", "attn": "eager"},
    "D1": {"dtype": "float32", "attn": "eager"},
}

SENTINEL_KEYS = [
    "butter_s0", "butter_s2", "ketchup_s1",
    "tomato_sauce_s1", "milk_s0", "orange_juice_s0",
]


def env_for(profile_key, gpu):
    p = ACTIVE_PROFILES[profile_key]
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


def run_one(ep, profile_key, gpu, output_dir, source_commit, save_video=False):
    cell_dir = Path(output_dir)
    done_file = cell_dir / ".done"

    if done_file.exists():
        try:
            existing = json.load(open(done_file))
            if existing.get("telemetry_sha") and existing["telemetry_sha"] != "MISSING":
                return existing
        except Exception:
            pass

    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True)

    cmd = [PYTHON, str(BRIDGE),
           "--condition", "CLEAN",
           "--state_id", str(ep["state_id"]),
           "--task_idx", str(ep["task_idx"]),
           "--anchor", "0",
           "--seed_id", "42",
           "--output_dir", str(cell_dir),
           "--render_gpu", str(gpu),
           "--mlp_path", str(CKPT)]
    if save_video:
        cmd.extend(["--save_video", "--source_commit", source_commit])

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO), env=env_for(profile_key, gpu),
                            capture_output=True, text=True)
    dt = time.time() - t0

    (cell_dir / "stdout.log").write_text(result.stdout)
    (cell_dir / "stderr.log").write_text(result.stderr)

    summary_path = cell_dir / "episode_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    telemetry_path = cell_dir / "step_telemetry.csv"
    telemetry_sha = hashlib.sha256(open(telemetry_path, "rb").read()).hexdigest() if telemetry_path.exists() else "MISSING"

    r = {
        "episode_key": ep["episode_key"], "task_idx": ep["task_idx"],
        "task_name": ep["task_name"], "state_id": ep["state_id"],
        "profile": profile_key,
        "requested_dtype": ACTIVE_PROFILES[profile_key]["dtype"],
        "requested_attn": ACTIVE_PROFILES[profile_key]["attn"],
        "actual_dtype": summary.get("actual_dtype", "unknown"),
        "actual_attn": summary.get("actual_attn", "unknown"),
        "exit_code": result.returncode, "duration_s": round(dt, 1),
        "telemetry_sha": telemetry_sha,
        "success": summary.get("task_success", False),
        "emit_step": summary.get("mlp_emit_step", -1),
        "steps": summary.get("n_steps", -1),
        "attack_frames": summary.get("attack_frames", 0),
        "checkpoint_sha": summary.get("checkpoint_sha256", "")[:16],
        "gpu": gpu, "output_dir": str(cell_dir),
    }

    # Attestation check
    expected_dtype = ACTIVE_PROFILES[profile_key]["dtype"]
    expected_attn = ACTIVE_PROFILES[profile_key]["attn"]
    if r["actual_dtype"] != expected_dtype:
        r["_fatal"] = f"dtype_mismatch: expected={expected_dtype} actual={r['actual_dtype']}"
    if r["actual_attn"] != expected_attn:
        r["_fatal"] = f"attn_mismatch: expected={expected_attn} actual={r['actual_attn']}"
    if result.returncode != 0:
        r["_fatal"] = f"exit_code={result.returncode}"
    if telemetry_sha == "MISSING":
        r["_fatal"] = "telemetry_missing"
    if r["attack_frames"] != 0:
        r["_fatal"] = f"attack_frames={r['attack_frames']}"

    json.dump(r, open(done_file, "w"))
    return r


def run_smoke(source_commit, gpu):
    manifest = json.load(open(MANIFEST_PATH))
    ep = [e for e in manifest["episodes"] if e["episode_key"] == "butter_s0"][0]
    results = {}
    for pk in ["B0", "D1"]:
        out = OUT_BASE / "smoke" / pk / "butter_s0"
        print(f"\n=== SMOKE {pk} ===")
        r = run_one(ep, pk, gpu, str(out), source_commit, save_video=True)
        results[pk] = r
        status = "PASS" if not r.get("_fatal") else f"FAIL({r['_fatal']})"
        print(f"  {status} succ={r['success']} dtype={r['actual_dtype']} attn={r['actual_attn']} emit={r['emit_step']} steps={r['steps']}")
    return results


def run_sentinel(source_commit, gpu):
    manifest = json.load(open(MANIFEST_PATH))
    sentinel_eps = [e for e in manifest["episodes"] if e["episode_key"] in SENTINEL_KEYS]
    all_results = []
    for pk in ["B0", "D1"]:
        for ep in sentinel_eps:
            for repeat in range(2):
                out = OUT_BASE / "sentinel" / ep["episode_key"] / pk / f"repeat_{repeat}"
                label = f"sentinel/{ep['episode_key']}/{pk}/r{repeat}"
                print(f"\n=== {label} ===")
                r = run_one(ep, pk, gpu, str(out), source_commit, save_video=True)
                r["sentinel_repeat"] = repeat
                all_results.append(r)
                if r.get("_fatal"):
                    print(f"  FATAL: {r['_fatal']}")
    csv_path = OUT_BASE / "sentinel_repeatability.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    return all_results


def run_main(source_commit, gpu):
    manifest = json.load(open(MANIFEST_PATH))
    episodes = manifest["episodes"]

    schedule = []
    for ep in episodes:
        idx = ep["fixed_order_index"]
        if idx % 2 == 0:
            schedule.append({"episode": ep, "profile": "B0", "group": "AB"})
            schedule.append({"episode": ep, "profile": "D1", "group": "AB"})
        else:
            schedule.append({"episode": ep, "profile": "D1", "group": "BA"})
            schedule.append({"episode": ep, "profile": "B0", "group": "BA"})

    all_results = []
    for i, s in enumerate(schedule):
        ep, pk = s["episode"], s["profile"]
        out = OUT_BASE / "main" / f"group_{s['group']}" / ep["episode_key"] / pk
        label = f"[{i+1}/{len(schedule)}] {s['group']} {ep['episode_key']}/{pk}"
        print(f"\n{label}")
        r = run_one(ep, pk, gpu, str(out), source_commit, save_video=False)
        r["group"] = s["group"]
        r["schedule_index"] = i
        all_results.append(r)
        if r.get("_fatal"):
            print(f"  FATAL: {r['_fatal']}")

        csv_path = OUT_BASE / "episode_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)

        heartbeat = {"last_episode": ep["episode_key"], "last_profile": pk,
                     "completed": i+1, "total": len(schedule),
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(heartbeat, open(OUT_BASE / "heartbeat.json", "w"))

    return all_results


def main():
    ap = argparse.ArgumentParser(description="M1B B0+D1 Dual Profile Study")
    ap.add_argument("--source_commit", required=True)
    ap.add_argument("--main_gpu", type=int, default=2)
    ap.add_argument("--sentinel_gpu", type=int, default=3)
    ap.add_argument("--smoke_only", action="store_true")
    ap.add_argument("--sentinel_only", action="store_true")
    ap.add_argument("--main_only", action="store_true")
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"M1B B0+D1 Dual Profile — commit={args.source_commit}")
    print(f"  Output: {OUT_BASE}")
    print(f"  Main GPU: {args.main_gpu}, Sentinel GPU: {args.sentinel_gpu}")

    if not args.sentinel_only and not args.main_only:
        smoke = run_smoke(args.source_commit, args.main_gpu)
        json.dump(smoke, open(OUT_BASE / "smoke_results.json", "w"), indent=2)
        print("\n=== SMOKE COMPLETE ===")
    if args.smoke_only:
        return

    if not args.main_only:
        sentinel = run_sentinel(args.source_commit, args.sentinel_gpu)
        print(f"\n=== SENTINEL: {len(sentinel)} runs ===")
    if args.sentinel_only:
        return

    main_results = run_main(args.source_commit, args.main_gpu)
    print(f"\n=== MAIN: {len(main_results)} runs ===")
    json.dump(main_results, open(OUT_BASE / "profile_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
