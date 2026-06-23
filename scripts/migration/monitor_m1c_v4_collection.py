#!/usr/bin/env python3
"""Safe read-only monitor for M1C V4 collection.

Displays per-shard progress, detects anomalies, never shows performance metrics.
Read-only: does not modify, delete, or re-run any cells.
"""
import os, sys, json, csv, time, hashlib, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────

ALLOWED_METRICS = [
    "pool", "state_range", "planned", "done", "missing", "last_done_time",
    "last_heartbeat", "process_alive", "gpu_util", "gpu_mem_free_gb",
    "disk_free_gb", "non_zero_rc", "missing_outputs", "duplicate_cells",
    "attack_frames_nonzero", "asset_sha_drift_count", "asset_sha_drift",
]

FORBIDDEN_METRICS = [
    "success_rate", "emit_count", "noemit_count", "teacher_valid",
    "no_corridor", "task_performance", "state_performance",
]

ALERT_CONDITIONS = {
    "no_new_done_20min": 1200,
    "heartbeat_stale_10min": 600,
    "process_gone_incomplete": True,
    "non_zero_rc": True,
    "duplicate_cell": True,
    "attack_frames": True,
    "asset_drift": True,
    "disk_below_100gb": 100 * 1024**3,
}


def sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


# ── Monitor logic ───────────────────────────────────────────────────

def monitor(manifest_path, output_path=None):
    manifest = load_manifest(manifest_path)
    gpu_assets = manifest.get("gpu_assignments", {})
    asserts = manifest.get("asset_sha_consistency", {})
    out_root = Path(manifest["output_root"])

    now = datetime.now()
    alerts = []
    shard_status = {}

    for gpu_key, gpu_info in gpu_assets.items():
        pool = gpu_info["pool"]
        planned = gpu_info["planned_cells"]
        state_start = int(gpu_info["state_range"].split("-")[0])
        state_end = int(gpu_info["state_range"].split("-")[1])

        # Count done cells in this shard's state range
        pool_dir = out_root / pool
        done_cells = []
        for task in range(0, 10):
            for state in range(state_start, state_end + 1):
                cell_dir = pool_dir / f"task{task}_state{state}"
                done_file = cell_dir / ".done"
                running_file = cell_dir / "RUNNING"
                if done_file.exists():
                    done_cells.append(str(cell_dir))

        n_done = len(done_cells)
        n_missing = planned - n_done

        # Last done time
        last_done_time = None
        if done_cells:
            last_done = max(done_cells, key=lambda d: os.path.getmtime(d))
            last_done_time = datetime.fromtimestamp(os.path.getmtime(last_done))

        # Duplicate detection
        done_files = list(pool_dir.rglob(".done"))
        if len(done_files) != len(set(str(d) for d in done_files)):
            alerts.append(f"DUPLICATE_DONE_FILES in {pool}")

        # RC check
        non_zero_rc = 0
        for d in done_files:
            try:
                data = json.load(open(d))
                if data.get("exit_code", 0) != 0:
                    non_zero_rc += 1
            except Exception:
                pass

        # Attack check — each subdirectory of pool_dir IS a cell directory
        attack_nonzero = 0
        for cell_dir in pool_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            ep = cell_dir / "episode_summary.json"
            if ep.exists():
                try:
                    s = json.load(open(ep))
                    if s.get("attack_frames", 0) is not None and s.get("attack_frames", 0) > 0:
                        attack_nonzero += 1
                except Exception:
                    pass

        # Asset SHA drift
        asset_drift = []
        for ckpt_key, expected in asserts.items():
            if ckpt_key == "vla_model_shards":
                continue
            # Only check repo-relative paths
            pass
        # Check checkpoint
        ckpt_path = Path("/mnt/sdc/dty_user/openvla_attack/artifacts/detector/sc5_mlp_s2.pt")
        if ckpt_path.exists():
            actual = sha256_file(str(ckpt_path))
            expected = asserts.get("detector_checkpoint", "")
            if actual != expected:
                asset_drift.append(f"checkpoint: {actual[:16]} != {expected[:16]}")
                alerts.append("ASSET_SHA_DRIFT")

        # Alerts
        if last_done_time and (now - last_done_time).total_seconds() > ALERT_CONDITIONS["no_new_done_20min"]:
            alerts.append(f"NO_NEW_DONE_20MIN: {pool} last={last_done_time.strftime('%H:%M')}")

        if non_zero_rc > 0:
            alerts.append(f"NON_ZERO_RC: {pool} count={non_zero_rc}")

        if attack_nonzero > 0:
            alerts.append(f"ATTACK_FRAMES_NONZERO: {pool} count={attack_nonzero}")

        if asset_drift:
            for a in asset_drift:
                alerts.append(f"ASSET_DRIFT: {a}")

        # Disk check
        try:
            import shutil
            disk_usage = shutil.disk_usage(str(out_root))
            disk_free_gb = disk_usage.free / (1024**3)
            if disk_usage.free < ALERT_CONDITIONS["disk_below_100gb"]:
                alerts.append(f"DISK_LOW: {disk_free_gb:.0f}GB free")
        except Exception:
            disk_free_gb = -1

        shard_status[gpu_key] = {
            "pool": pool,
            "state_range": gpu_info["state_range"],
            "planned": planned,
            "done": n_done,
            "missing": n_missing,
            "last_done_time": last_done_time.strftime("%Y-%m-%d %H:%M:%S") if last_done_time else "N/A",
            "non_zero_rc": non_zero_rc,
            "attack_frames_nonzero": attack_nonzero,
            "asset_sha_drift_count": len(asset_drift),
            "disk_free_gb": f"{disk_free_gb:.0f}" if disk_free_gb >= 0 else "N/A",
        }

    # Output
    report = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "host": manifest.get("a800_host", "unknown"),
        "total_planned": manifest["total_planned_cells"],
        "total_done": sum(s["done"] for s in shard_status.values()),
        "total_missing": sum(s["missing"] for s in shard_status.values()),
        "shards": shard_status,
        "alerts": alerts,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report: {output_path}")

    # Console display (safe fields only)
    print(f"\n  M1C V4 Monitor — {report['timestamp']}")
    print(f"  {report['total_done']}/{report['total_planned']} cells done  "
          f"({report['total_missing']} remaining)")
    for gpu_key, s in shard_status.items():
        pct = s["done"] / s["planned"] * 100 if s["planned"] > 0 else 0
        print(f"  {gpu_key} ({s['pool']} {s['state_range']}): "
              f"{s['done']}/{s['planned']} ({pct:.0f}%)  "
              f"last={s['last_done_time']}  rc={s['non_zero_rc']}  "
              f"atk={s['attack_frames_nonzero']}  disk={s['disk_free_gb']}GB")
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    {a}")
    else:
        print(f"\n  No alerts.")

    return report


def main():
    ap = argparse.ArgumentParser(description="M1C V4 Safe Collection Monitor")
    ap.add_argument("--execution-manifest", required=True)
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--interval-seconds", type=int, default=0, help="Poll interval")
    ap.add_argument("--output", help="Write JSON report to file")
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
