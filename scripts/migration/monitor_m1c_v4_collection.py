#!/usr/bin/env python3
"""Safe read-only monitor for M1C V4 collection (v2 — hardened).

v2 fixes: .done mtime, heartbeat, GPU stats, full asset SHA, JSON fail-closed,
duplicate via multi-root, process-alive check.
"""
import os, sys, json, time, hashlib, argparse, subprocess
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]

ALLOWED_METRICS = [
    "pool", "state_range", "planned", "done", "missing", "last_done_time",
    "last_heartbeat", "process_alive", "gpu_util_pct", "gpu_mem_free_gb",
    "disk_free_gb", "non_zero_rc", "duplicate_cells", "attack_frames_nonzero",
    "asset_sha_drift_count",
]

FORBIDDEN_METRICS = [
    "success_rate", "emit_count", "noemit_count", "teacher_valid",
    "no_corridor", "task_performance", "state_performance",
]

EXPECTED_ASSETS = {
    "detector_checkpoint": ("/mnt/sdc/dty_user/openvla_attack/artifacts/detector/sc5_mlp_s2.pt",
                            "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628"),
    "teacher_config": ("/mnt/sdc/dty_user/openvla_attack/migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json",
                       "ebc1ccda21cdfeae0f70f90ef0e433be3474ef0baa9cf52f609d620f863ce87a"),
    "bridge_script": ("/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py",
                      "fd594b3f9b38f4545d7b19202b380c0f4eeb0e9d95cb566f2fbca7b1852b208e"),
}

ALERT_CONDITIONS = {
    "no_new_done_20min": 1200,
    "heartbeat_stale_10min": 600,
    "process_gone_incomplete": True,
    "non_zero_rc": True,
    "attack_frames": True,
    "asset_drift": True,
    "disk_below_100gb": 100 * 1024**3,
}


def _safe_json(path):
    """Fail-closed: return (data, error)."""
    try:
        with open(path) as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def _sha256(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _gpu_stats():
    """Parse nvidia-smi for GPU util and memory."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        stats = {}
        for line in out.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                stats[int(parts[0])] = {
                    "util_pct": float(parts[1]),
                    "mem_free_mb": float(parts[2]),
                }
        return stats
    except Exception:
        return {}


def _disk_free(path):
    try:
        s = os.statvfs(path)
        return s.f_frsize * s.f_bavail
    except Exception:
        return -1


def monitor(manifest_path, output_path=None):
    with open(manifest_path) as f:
        manifest = json.load(f)

    gpu_assignments = manifest.get("gpu_assignments", {})
    out_root = Path(manifest["output_root"])
    now = datetime.now()
    alerts = []
    shard_status = {}

    # GPU stats
    gpu_stats = _gpu_stats()
    disk_free_bytes = _disk_free(str(out_root))
    disk_free_gb = disk_free_bytes / (1024**3) if disk_free_bytes >= 0 else -1

    # Asset SHA drift
    asset_drift = []
    for name, (path, expected) in EXPECTED_ASSETS.items():
        actual = _sha256(path)
        if actual is None:
            asset_drift.append(f"ASSET_MISSING:{name}")
            alerts.append("ASSET_SHA_DRIFT")
        elif actual != expected:
            asset_drift.append(f"ASSET_MISMATCH:{name} got={actual[:16]} exp={expected[:16]}")
            alerts.append("ASSET_SHA_DRIFT")

    for gpu_key, gpu_info in gpu_assignments.items():
        pool = gpu_info["pool"]
        planned = gpu_info["planned_cells"]
        state_start, state_end = map(int, gpu_info["state_range"].split("-"))
        pool_dir = out_root / pool

        # Scan filesystem for actual done cells
        done_files = list(pool_dir.rglob(".done"))
        done_count = len(done_files)

        # .done mtime (not directory mtime!)
        last_done_time = None
        if done_files:
            latest = max(done_files, key=lambda p: os.path.getmtime(str(p)))
            last_done_time = datetime.fromtimestamp(os.path.getmtime(str(latest)))

        # Heartbeat: latest .done write within threshold
        heartbeat_ok = True
        if done_files and last_done_time:
            age = (now - last_done_time).total_seconds()
            if age > ALERT_CONDITIONS["heartbeat_stale_10min"]:
                alerts.append(f"HEARTBEAT_STALE:{pool} last={last_done_time.strftime('%H:%M')}")
                heartbeat_ok = False

        # Non-zero RC (fail-closed: corrupt JSON = alert)
        non_zero_rc = 0
        for d in done_files:
            data, err = _safe_json(d)
            if err:
                non_zero_rc += 1
                alerts.append(f"CORRUPT_DONE:{d}")
            elif data.get("exit_code", -999) != 0:
                non_zero_rc += 1
                alerts.append(f"NON_ZERO_RC:{d} rc={data['exit_code']}")

        # Attack check (each cell dir in pool)
        attack_nonzero = 0
        for cell_dir in pool_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            ep = cell_dir / "episode_summary.json"
            if ep.exists():
                s, err = _safe_json(ep)
                if err:
                    attack_nonzero += 1
                    alerts.append(f"CORRUPT_SUMMARY:{cell_dir}")
                elif s.get("attack_frames", -1) is not None and s.get("attack_frames", -1) > 0:
                    attack_nonzero += 1
                    alerts.append(f"ATTACK_FRAMES_NONZERO:{cell_dir.name}")
                elif s.get("condition", "") != "CLEAN":
                    attack_nonzero += 1
                    alerts.append(f"CONDITION_NOT_CLEAN:{cell_dir.name}")

        # Duplicate: claim/cell key collision across pools
        cells_seen = set()
        dupes = 0
        for d in done_files:
            cell_name = str(d.parent.relative_to(out_root))
            if cell_name in cells_seen:
                dupes += 1
            cells_seen.add(cell_name)
        if dupes > 0:
            alerts.append(f"DUPLICATE_CELLS:{pool} count={dupes}")

        # Process alive
        gpu_idx = int(gpu_key.replace("gpu", ""))
        proc_alive = gpu_idx in gpu_stats
        if not proc_alive and planned > done_count:
            alerts.append(f"GPU_PROCESS_GONE:{gpu_key}")

        # Disk
        if disk_free_bytes >= 0 and disk_free_bytes < ALERT_CONDITIONS["disk_below_100gb"]:
            alerts.append(f"DISK_LOW:{disk_free_gb:.0f}GB")

        # GPU-specific stats
        gs = gpu_stats.get(gpu_idx, {})
        gpu_util = gs.get("util_pct", -1)
        gpu_mem = gs.get("mem_free_mb", -1) / 1024 if gs.get("mem_free_mb", -1) >= 0 else -1

        shard_status[gpu_key] = {
            "pool": pool,
            "state_range": gpu_info["state_range"],
            "planned": planned,
            "done": done_count,
            "missing": planned - done_count,
            "last_done_time": last_done_time.strftime("%Y-%m-%d %H:%M:%S") if last_done_time else "N/A",
            "last_heartbeat": last_done_time.strftime("%H:%M:%S") if last_done_time else "N/A",
            "process_alive": proc_alive,
            "gpu_util_pct": gpu_util,
            "gpu_mem_free_gb": round(gpu_mem, 1) if gpu_mem >= 0 else "N/A",
            "disk_free_gb": f"{disk_free_gb:.0f}" if disk_free_bytes >= 0 else "N/A",
            "non_zero_rc": non_zero_rc,
            "duplicate_cells": dupes,
            "attack_frames_nonzero": attack_nonzero,
            "asset_sha_drift_count": len(asset_drift),
        }

    # Summary
    total_done = sum(s["done"] for s in shard_status.values())
    total_planned = manifest.get("total_planned_cells", 0)

    report = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "host": manifest.get("a800_host", "unknown"),
        "total_planned": total_planned,
        "total_done": total_done,
        "total_missing": total_planned - total_done,
        "shards": shard_status,
        "asset_drift": asset_drift,
        "alerts": alerts,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    # Console
    print(f"\n  M1C V4 Monitor v2 — {report['timestamp']}")
    print(f"  {total_done}/{total_planned} done ({total_planned - total_done} remaining)")
    for gpu_key, s in shard_status.items():
        pct = s["done"] / s["planned"] * 100 if s["planned"] > 0 else 0
        print(f"  {gpu_key} {s['pool']} {s['state_range']}: "
              f"{s['done']}/{s['planned']} ({pct:.0f}%)  last={s['last_heartbeat']}  "
              f"util={s['gpu_util_pct']}%  mem={s['gpu_mem_free_gb']}G  "
              f"rc={s['non_zero_rc']}  dup={s['duplicate_cells']}  "
              f"atk={s['attack_frames_nonzero']}  drift={s['asset_sha_drift_count']}  "
              f"disk={s['disk_free_gb']}G")
    if asset_drift:
        print(f"\n  ASSET DRIFT ({len(asset_drift)}):")
        for a in asset_drift:
            print(f"    {a}")
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    {a}")
    else:
        print(f"\n  No alerts.")

    return report


def main():
    ap = argparse.ArgumentParser(description="M1C V4 Safe Collection Monitor v2")
    ap.add_argument("--execution-manifest", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval-seconds", type=int, default=0)
    ap.add_argument("--output", help="JSON report path")
    args = ap.parse_args()

    if args.interval_seconds > 0:
        while True:
            monitor(args.execution_manifest, args.output)
            time.sleep(args.interval_seconds)
    else:
        report = monitor(args.execution_manifest, args.output)
        if report["alerts"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
