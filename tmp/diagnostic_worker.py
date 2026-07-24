#!/usr/bin/env python3
"""Persistent diagnostic worker — one per GPU, atomic cell claiming."""
import argparse, csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from datetime import datetime

REPO = Path(os.environ.get("REPO", "/mnt/sdc/dty_user/openvla_attack"))
COLLECTOR = REPO / "scripts/stageb/run_v6_perturbed_collector.py"
CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
CLAIM_DIR = None  # set per-run

def heartbeat(msg):
    print("%s [GPU%d] %s" % (datetime.now().strftime("%H:%M:%S"), GPU, msg), flush=True)

def claim_cell(manifest_path, gpu):
    """Atomically claim next unclaimed cell for this GPU. Returns row dict or None."""
    lock = CLAIM_DIR / "claim.lock"
    claimed = CLAIM_DIR / "claimed.txt"

    # Simple file-based lock
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None  # another worker is claiming

    try:
        manifest = list(csv.DictReader(open(manifest_path)))

        # Read already-claimed cell IDs
        claimed_ids = set()
        if claimed.exists():
            claimed_ids = set(open(claimed).read().strip().split('\n'))

        # Find next unclaimed cell for this GPU
        for row in manifest:
            cid = row['diagnostic_cell_id']
            if cid in claimed_ids:
                continue
            if int(row['assigned_gpu']) != gpu:
                continue
            # Claim it
            with open(claimed, 'a') as f:
                f.write(cid + '\n')
            return row

        return None  # no more cells for this GPU
    finally:
        if lock.exists():
            lock.unlink()

def run_cell(row, out_base, vla_sha):
    """Run a single diagnostic cell. Returns (success, output_dir)."""
    cid = row['diagnostic_cell_id']
    task = row['task_idx']; state = row['state_id']
    tmpl = row['perturbation_template']; seed = row['perturbation_seed']

    final_dir = out_base / cid
    tmp_dir = out_base / (cid + ".tmp." + str(os.getpid()))

    if final_dir.exists():
        if (final_dir / ".done").exists():
            heartbeat("%s: already complete, skipping" % cid)
            return True, final_dir
        shutil.rmtree(str(final_dir))

    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(REPO / "envs/openvla-official-a800/bin/python3"),
        str(COLLECTOR),
        "--task_idx", task, "--state_id", state,
        "--seed_id", seed, "--perturbation_template", tmpl,
        "--pool", "smoke",  # diagnostic — not train/dev
        "--output_dir", str(tmp_dir),
        "--render_gpu", str(GPU),
        "--mlp_path", str(CKPT),
        "--vla_manifest_sha256", vla_sha,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["MUJOCO_GL"] = "egl"

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1500)
        if proc.returncode != 0:
            heartbeat("%s: FAILED rc=%d stderr=%s" % (cid, proc.returncode, proc.stderr[-200:]))
            return False, tmp_dir

        # Write diagnostic metadata sidecar
        sidecar = {
            "diagnostic_cell_id": cid, "usage": "DIAGNOSTIC_ONLY",
            "eligible_for_training": False, "eligible_for_dev": False,
            "eligible_for_checkpoint_selection": False,
            "worker_gpu": GPU, "collector_sha256": COLLECTOR_SHA,
            "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        }
        with open(tmp_dir / "diagnostic_metadata.json", "w") as f:
            json.dump(sidecar, f)

        # Atomic rename
        if final_dir.exists():
            shutil.rmtree(str(final_dir))
        os.rename(str(tmp_dir), str(final_dir))

        # Count steps from telemetry
        tel = final_dir / "step_telemetry.csv"
        steps = 0
        if tel.exists():
            steps = tel.read_bytes().decode().count('\n') - 1

        heartbeat("%s: OK steps=%d" % (cid, steps))
        heartbeat("HEARTBEAT OK")
        return True, final_dir

    except subprocess.TimeoutExpired:
        heartbeat("%s: TIMEOUT (25min)" % cid)
        proc.kill()
        return False, tmp_dir
    except Exception as e:
        heartbeat("%s: ERROR %s" % (cid, e))
        return False, tmp_dir

def main():
    global GPU, COLLECTOR_SHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--claim_dir", required=True)
    ap.add_argument("--vla_sha", required=True)
    args = ap.parse_args()

    GPU = args.gpu
    COLLECTOR_SHA = hashlib.sha256(open(COLLECTOR,"rb").read()).hexdigest()

    global CLAIM_DIR
    CLAIM_DIR = Path(args.claim_dir)
    CLAIM_DIR.mkdir(parents=True, exist_ok=True)

    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    heartbeat("STARTED collector_sha=%s" % COLLECTOR_SHA[:16])

    done = 0; failed = 0; retries = {}

    stall_count = 0
    while True:
        row = claim_cell(args.manifest, GPU)
        if row is None:
            stall_count += 1
            if stall_count > 30:  # 30 × 10s = 5 min of no cells
                heartbeat("NO_MORE_CELLS (stalled %d times)" % stall_count)
                break
            time.sleep(10)
            continue
        stall_count = 0  # reset on successful claim

        cid = row['diagnostic_cell_id']
        ok, out_dir = run_cell(row, out_base, args.vla_sha)

        if ok:
            done += 1
            if cid in retries:
                del retries[cid]
        else:
            # First failure: retry once
            if cid not in retries:
                retries[cid] = 1
                heartbeat("%s: RETRY_1/1" % cid)
                ok2, out_dir2 = run_cell(row, out_base, args.vla_sha)
                if ok2:
                    done += 1
                    del retries[cid]
                else:
                    failed += 1
                    # Move to quarantine
                    q_dir = out_base / ("QUARANTINE_" + cid)
                    if out_dir.exists():
                        os.rename(str(out_dir), str(q_dir))
                    heartbeat("%s: QUARANTINED (2 failures)" % cid)
            else:
                failed += 1
                q_dir = out_base / ("QUARANTINE_" + cid)
                if out_dir.exists():
                    os.rename(str(out_dir), str(q_dir))
                heartbeat("%s: QUARANTINED" % cid)

    heartbeat("DONE done=%d failed=%d" % (done, failed))

if __name__ == "__main__":
    main()
