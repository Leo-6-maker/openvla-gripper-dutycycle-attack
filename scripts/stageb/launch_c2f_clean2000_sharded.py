#!/usr/bin/env python3
"""Launch C2f clean collection in suite-aware worker shards.

Why this exists
---------------
`collect_c2f_observation_clean_rollouts.py` writes a root-level manifest and
SHA256SUMS at the end of each run.  Running many collectors against the same
output root would race and corrupt those root-level files.  This launcher instead
creates one isolated output root per worker:

  <output-root>/shards/<suite>/worker_<NN>/

A separate merge step can then combine successful shards into one canonical C2f
collection root for hygiene checks and materialization.

Recommended clean2000 layout on 4 GPUs
--------------------------------------
Use one suite per GPU to avoid a single process caching multiple OpenVLA suite
checkpoints:

  GPU4: libero_10      x 3 workers
  GPU5: libero_object  x 3 workers
  GPU6: libero_goal    x 3 workers
  GPU7: libero_spatial x 3 workers

Each worker uses the same C2f adapter and official OpenVLA preprocessing path
inside the adapter (`upstream_tf_jpeg`, center crop, resize 224).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_SUITES = ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
DEFAULT_ADAPTER = "scripts.stageb.c2f_libero_openvla_adapter:make_adapter"


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if "episodes" in obj:
            return list(obj["episodes"])
        raise ValueError("JSON manifest must be a list or contain an episodes list")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def split_even(rows: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(n)]
    for i, row in enumerate(rows):
        shards[i % n].append(row)
    return shards


def shell_quote_list(items: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in items)


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch suite-aware sharded C2f clean collection")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True, help="Root containing shards/, logs/, launch_manifest.json")
    ap.add_argument("--gpus", default="4,5,6,7", help="Comma-separated physical GPU ids")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES), help="Comma-separated suite order, mapped 1:1 to GPUs")
    ap.add_argument("--workers-per-gpu", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=3, help="OMP/MKL/OPENBLAS threads per worker; 12*3=36 cores")
    ap.add_argument("--adapter-module", default=DEFAULT_ADAPTER)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--python", default="python")
    ap.add_argument("--dry-run", action="store_true", help="Write shards/scripts but do not start workers")
    ap.add_argument("--overwrite", action="store_true", help="Pass --overwrite to each worker collector")
    ap.add_argument("--max-total", type=int, default=0, help="Optional global cap for testing; 0=all")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    if len(gpus) != len(suites):
        raise SystemExit(f"--gpus and --suites must have same length, got {gpus} vs {suites}")

    out = Path(args.output_root)
    shard_root = out / "shards"
    log_root = out / "logs"
    script_root = out / "worker_scripts"
    for p in [shard_root, log_root, script_root]:
        p.mkdir(parents=True, exist_ok=True)

    episodes = read_manifest(Path(args.manifest))
    if args.max_total > 0:
        episodes = episodes[: args.max_total]

    suite_to_gpu = dict(zip(suites, gpus))
    launch_rows: List[Dict[str, Any]] = []
    procs: List[subprocess.Popen] = []

    for suite in suites:
        suite_rows = [e for e in episodes if str(e.get("suite")) == suite]
        shards = split_even(suite_rows, args.workers_per_gpu)
        gpu = suite_to_gpu[suite]
        for wi, shard in enumerate(shards):
            worker_name = f"{suite}_w{wi:02d}"
            worker_dir = shard_root / suite / f"worker_{wi:02d}"
            manifest_path = worker_dir / "worker_manifest.jsonl"
            write_jsonl(manifest_path, shard)
            log_path = log_root / f"{worker_name}.log"
            cmd = [
                args.python,
                "scripts/stageb/collect_c2f_observation_clean_rollouts.py",
                "--manifest", str(manifest_path),
                "--output-root", str(worker_dir),
                "--adapter-module", args.adapter_module,
                "--suite", suite,
                "--git-commit", args.git_commit,
                "--source-commit", args.source_commit or args.git_commit,
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env_exports = (
                f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}; "
                f"export OMP_NUM_THREADS={args.threads_per_worker}; "
                f"export MKL_NUM_THREADS={args.threads_per_worker}; "
                f"export OPENBLAS_NUM_THREADS={args.threads_per_worker}; "
                f"export NUMEXPR_NUM_THREADS={args.threads_per_worker}; "
                "export TOKENIZERS_PARALLELISM=false; "
            )
            shell_cmd = env_exports + "nice -n 10 ionice -c2 -n7 " + shell_quote_list(cmd)
            script_path = script_root / f"run_{worker_name}.sh"
            script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(Path.cwd())) + "\n" + shell_cmd + "\n", encoding="utf-8")
            script_path.chmod(0o755)
            row = {
                "worker": worker_name,
                "suite": suite,
                "gpu": gpu,
                "n_episodes": len(shard),
                "manifest": str(manifest_path),
                "output_root": str(worker_dir),
                "log": str(log_path),
                "script": str(script_path),
                "command": shell_cmd,
            }
            launch_rows.append(row)
            if not args.dry_run and shard:
                log_f = log_path.open("w", encoding="utf-8")
                p = subprocess.Popen(["bash", "-lc", shell_cmd], stdout=log_f, stderr=subprocess.STDOUT)
                row["pid"] = p.pid
                procs.append(p)
                time.sleep(3)

    report = {
        "gate": "C2F_CLEAN2000_SHARDED_LAUNCH",
        "status": "DRY_RUN_WRITTEN" if args.dry_run else "LAUNCHED",
        "created_at_unix": time.time(),
        "manifest": str(Path(args.manifest)),
        "output_root": str(out),
        "n_input_episodes": len(episodes),
        "gpus": gpus,
        "suites": suites,
        "workers_per_gpu": args.workers_per_gpu,
        "threads_per_worker": args.threads_per_worker,
        "total_workers": len(launch_rows),
        "adapter_module": args.adapter_module,
        "git_commit": args.git_commit,
        "source_commit": args.source_commit or args.git_commit,
        "workers": launch_rows,
        "boundaries": {
            "attack": "NOT_PERFORMED",
            "d7b2_outcome_read": False,
            "same_output_root_per_worker": False,
            "suite_aware_model_cache": True,
        },
    }
    write_json(out / "launch_manifest.json", report)
    print(json.dumps({"status": report["status"], "workers": len(launch_rows), "output_root": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
