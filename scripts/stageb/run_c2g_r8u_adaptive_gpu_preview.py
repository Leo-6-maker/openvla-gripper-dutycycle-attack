#!/usr/bin/env python3
"""Preview-only adaptive GPU worker planner for R8U.

Calculates worker slots per GPU from free memory and utilization,
plans serialized model loading, and assigns microshards.
The `run` mode is explicitly denied — this is preview ONLY.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple


def read_jsonl(path: _Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

AUTHORIZATION_DENIED = "R8U_ADAPTIVE_COLLECTION_NOT_AUTHORIZED"

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_GPUS = [4, 5, 6, 7]
DEFAULT_RESERVE_MIB = 12000
DEFAULT_WORKER_BUDGET_MIB = 22000
DEFAULT_MAX_WORKERS_PER_GPU = 2
DEFAULT_GLOBAL_LOADING_SLOTS = 1
DEFAULT_POLL_SECONDS = 20
DEFAULT_STABILIZATION_POLLS = 3
DEFAULT_MAX_UTILIZATION_PERCENT = 40
DEFAULT_STABILIZATION_SECONDS = 60

@dataclass
class OomBackoff:
    gpu_index: int
    current_worker_cap: int
    original_worker_cap: int
    backoff_count: int = 0

    def reduce(self) -> int:
        self.backoff_count += 1
        self.current_worker_cap = max(0, self.original_worker_cap - self.backoff_count)
        return self.current_worker_cap


@dataclass
class GpuSnapshot:
    index: int
    memory_total_mib: float
    memory_free_mib: float
    utilization_percent: float

    @property
    def available_slots(self) -> int:
        return max(
            0,
            int((self.memory_free_mib - DEFAULT_RESERVE_MIB) // DEFAULT_WORKER_BUDGET_MIB),
        )

    @property
    def can_accept_worker(self) -> bool:
        return self.available_slots > 0 and self.utilization_percent <= DEFAULT_MAX_UTILIZATION_PERCENT


@dataclass
class Microshard:
    shard_id: str
    suite: str
    episode_count: int


@dataclass
class WorkerPlan:
    worker_id: str
    physical_gpu: int
    shard_id: str
    output_root: str
    stdout_log: str
    stderr_log: str
    planned_start_delay_seconds: float = 0.0
    load_order: int = 0


@dataclass
class PreviewResult:
    gpu_snapshot: Dict[int, dict] = field(default_factory=dict)
    slot_counts: Dict[int, int] = field(default_factory=dict)
    microshards: List[dict] = field(default_factory=list)
    workers: List[dict] = field(default_factory=list)
    loading_order: List[str] = field(default_factory=list)
    planned_start_delays: List[float] = field(default_factory=list)
    status: str = "PASS_C2G_R8U_ADAPTIVE_GPU_PREVIEW"


def calculate_slots(snapshot: GpuSnapshot) -> Tuple[int, int]:
    """Return (available_slots, target_workers). Target caps at max_workers_per_gpu."""
    available = snapshot.available_slots
    target = min(DEFAULT_MAX_WORKERS_PER_GPU, available)
    return available, target


def build_preview(
    gpu_indices: List[int],
    gpu_snapshots: Dict[int, GpuSnapshot],
    shards: List[Microshard],
    output_base: str,
) -> PreviewResult:
    result = PreviewResult()

    for idx in gpu_indices:
        snap = gpu_snapshots.get(idx)
        if snap is None:
            continue
        result.gpu_snapshot[idx] = {
            "memory_total_mib": snap.memory_total_mib,
            "memory_free_mib": snap.memory_free_mib,
            "utilization_percent": snap.utilization_percent,
        }
        available, target = calculate_slots(snap)
        result.slot_counts[idx] = {
            "available_slots": available,
            "target_workers": target,
            "can_accept": snap.can_accept_worker,
        }

    # Assign shards to GPUs round-robin
    gpu_order = sorted(gpu_indices)
    load_order = 0
    for i, shard in enumerate(shards):
        gpu = gpu_order[i % len(gpu_order)]
        if not gpu_snapshots.get(gpu, GpuSnapshot(gpu, 0, 0, 100)).can_accept_worker:
            continue

        worker_id = f"r8u_adaptive__{shard.shard_id}__w{load_order:03d}"
        out = f"{output_base}/workers/{worker_id}"

        delay = 0.0
        if load_order > 0:
            delay = DEFAULT_STABILIZATION_SECONDS * load_order

        worker = WorkerPlan(
            worker_id=worker_id,
            physical_gpu=gpu,
            shard_id=shard.shard_id,
            output_root=out,
            stdout_log=f"{output_base}/logs/{worker_id}.stdout.log",
            stderr_log=f"{output_base}/logs/{worker_id}.stderr.log",
            planned_start_delay_seconds=delay,
            load_order=load_order,
        )
        result.workers.append({
            "worker_id": worker.worker_id,
            "physical_gpu": worker.physical_gpu,
            "shard_id": worker.shard_id,
            "output_root": worker.output_root,
            "planned_start_delay_seconds": worker.planned_start_delay_seconds,
            "load_order": worker.load_order,
        })
        result.loading_order.append(worker.worker_id)
        result.planned_start_delays.append(delay)
        load_order += 1

    for sh in shards:
        result.microshards.append({
            "shard_id": sh.shard_id,
            "suite": sh.suite,
            "episode_count": sh.episode_count,
        })

    return result


def get_snapshot(gpu_indices: List[int]) -> Dict[int, GpuSnapshot]:
    """Read GPU state from nvidia-smi. Returns empty dict if unavailable."""
    import subprocess
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    snapshots = {}
    for line in output.strip().split("\n"):
        parts = line.split(", ")
        if len(parts) < 4:
            continue
        idx = int(parts[0].strip())
        if idx not in gpu_indices:
            continue
        snapshots[idx] = GpuSnapshot(
            index=idx,
            memory_total_mib=float(parts[1].strip()),
            memory_free_mib=float(parts[2].strip()),
            utilization_percent=float(parts[3].strip()),
        )
    return snapshots


def main() -> int:
    ap = argparse.ArgumentParser(description="R8U adaptive GPU preview")
    ap.add_argument("mode", choices=["preview", "run"], help="preview only; run is denied")
    ap.add_argument("--gpus", default="4,5,6,7", help="GPU indices")
    ap.add_argument("--output-root", default="/tmp/r8u_adaptive_preview")
    ap.add_argument("--shard-count", type=int, default=4, help="number of microshards")
    ap.add_argument("--episodes-per-shard", type=int, default=6)
    ap.add_argument("--synthetic-test-mode", action="store_true", help="use synthetic GPU snapshots for CI/testing")
    ap.add_argument("--plan-manifest", default="", help="hash-bound plan manifest for real microshards")
    ap.add_argument("--max-utilization-percent", type=int, default=DEFAULT_MAX_UTILIZATION_PERCENT)
    args = ap.parse_args()

    if args.mode == "run":
        raise PermissionError(AUTHORIZATION_DENIED)

    gpu_indices = [int(g.strip()) for g in args.gpus.split(",")]
    snapshots = get_snapshot(gpu_indices)

    if not snapshots:
        if args.synthetic_test_mode:
            for idx in gpu_indices:
                snapshots[idx] = GpuSnapshot(idx, 81920, 70000 - (idx * 5000), 5.0)
        else:
            print("ERROR: nvidia-smi unavailable. Use --synthetic-test-mode for CI only.", file=sys.stderr)
            return 1

    # Load real microshards from plan manifest if provided
    shards = []
    if args.plan_manifest and _Path(args.plan_manifest).is_file():
        for row in read_jsonl(_Path(args.plan_manifest)):
            shards.append(Microshard(
                f"r8u_{row['suite']}_{row.get('task_index', 0):02d}",
                row["suite"],
                1,  # per-parent microshard
            ))
    else:
        shards = [
            Microshard(f"r8u_shard_{i:03d}", f"suite_{i}", args.episodes_per_shard)
            for i in range(args.shard_count)
        ]

    result = build_preview(gpu_indices, snapshots, shards, args.output_root)
    print(json.dumps({
        "schema": "c2g.r8u.adaptive_gpu_preview.2026-07-11.v1",
        "status": result.status,
        "gpu_snapshot": result.gpu_snapshot,
        "slot_counts": result.slot_counts,
        "microshards": result.microshards,
        "workers": result.workers,
        "loading_order": result.loading_order,
        "planned_start_delays": result.planned_start_delays,
        "max_workers_per_gpu": DEFAULT_MAX_WORKERS_PER_GPU,
        "global_loading_slots": DEFAULT_GLOBAL_LOADING_SLOTS,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
