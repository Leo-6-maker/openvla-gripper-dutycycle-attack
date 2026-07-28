"""[DeepSeek] FIT670 Shard Plan Builder — Gate F670-D.

Cost-based greedy bin-packing of 670 identities into 6 (or 8) GPU shards.
Deterministic: same inputs produce same output every time.

Cost = HORIZONS[suite] (libero_10=520, libero_goal=300, etc.)

Usage:
  python n5/phase2_labels/build_fit670_shard_plan.py \
    --identity-allowlist /path/to/FIT670_IDENTITY_ALLOWLIST.json \
    --out /path/to/shard_plan_output \
    --n-shards 6
"""
import argparse, json, os, shutil, time, uuid
from pathlib import Path

HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FOUR_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def seal_output(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return sums_sha


def build_shard_plan(identities, n_shards=6):
    """Greedy bin-packing by cost, with suite spread balancing.

    Returns dict with shard plan.
    """
    # 1. Compute cost and sort
    for ident in identities:
        ident["_cost"] = HORIZONS.get(ident["suite"], 300)
    # Sort by cost descending, then episode_id ascending (deterministic tiebreak)
    sorted_ids = sorted(identities, key=lambda x: (-x["_cost"], x["episode_id"]))

    # 2. Greedy assignment
    shards = [{"shard_id": i, "gpu": i, "identities": [], "total_cost": 0,
               "suite_counts": {s: 0 for s in FOUR_SUITES}}
              for i in range(n_shards)]

    for ident in sorted_ids:
        # Find shard with lowest total_cost
        best = min(shards, key=lambda s: s["total_cost"])
        best["identities"].append(ident)
        best["total_cost"] += ident["_cost"]
        best["suite_counts"][ident["suite"]] += 1

    # 3. Balance suite spread: no shard should have > ceil(count/n_shards) + 1 from any suite
    suite_totals = {}
    for s in FOUR_SUITES:
        suite_totals[s] = sum(1 for i in identities if i["suite"] == s)

    max_per_shard = {s: max(1, -(-suite_totals[s] // n_shards) + 1) for s in FOUR_SUITES}

    # Greedy swap to improve spread
    for _ in range(10):  # max 10 swap passes
        improved = False
        for s in FOUR_SUITES:
            overloaded = [sh for sh in shards if sh["suite_counts"][s] > max_per_shard[s]]
            underloaded = [sh for sh in shards if sh["suite_counts"][s] < max_per_shard[s] - 1]
            while overloaded and underloaded:
                src = overloaded[0]
                dst = underloaded[0]
                # Find an identity from suite s in src to move to dst
                swap_idx = None
                for idx, ident in enumerate(src["identities"]):
                    if ident["suite"] == s:
                        swap_idx = idx
                        break
                if swap_idx is not None:
                    ident = src["identities"].pop(swap_idx)
                    dst["identities"].append(ident)
                    src["total_cost"] -= ident["_cost"]
                    dst["total_cost"] += ident["_cost"]
                    src["suite_counts"][s] -= 1
                    dst["suite_counts"][s] += 1
                    improved = True
                # Refresh overloaded/underloaded
                overloaded = [sh for sh in shards if sh["suite_counts"][s] > max_per_shard[s]]
                underloaded = [sh for sh in shards if sh["suite_counts"][s] < max_per_shard[s] - 1]
        if not improved:
            break

    # 4. Build output
    for shard in shards:
        shard["n_identities"] = len(shard["identities"])
        # Sort identities within shard by episode_id
        shard["identities"].sort(key=lambda x: x["episode_id"])
        # Strip internal _cost
        for ident in shard["identities"]:
            ident.pop("_cost", None)

    total_cost = sum(s["total_cost"] for s in shards)
    # Recalculate suite_counts from sorted identities
    for shard in shards:
        sc = {s: 0 for s in FOUR_SUITES}
        for ident in shard["identities"]:
            sc[ident["suite"]] += 1
        shard["suite_counts"] = sc

    return {
        "schema": "FIT670_GPU_SHARD_PLAN_V1",
        "status": "FROZEN",
        "n_shards": n_shards,
        "n_identities": len(identities),
        "algorithm": "COST_DESCENDING_GREEDY_BIN_PACK_WITH_SUITE_SPREAD",
        "cost_total": total_cost,
        "cost_per_shard": {f"shard_{i}": s["total_cost"] for i, s in enumerate(shards)},
        "cost_imbalance_pct": round(
            (max(s["total_cost"] for s in shards) - min(s["total_cost"] for s in shards))
            / (total_cost / n_shards) * 100, 2
        ) if total_cost > 0 else 0,
        "shards": shards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-allowlist", type=Path, required=True,
                        help="Path to FIT670_IDENTITY_ALLOWLIST.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, default=6, choices=[6, 8])
    args = parser.parse_args()

    allowlist_path = Path(args.identity_allowlist).resolve()
    if not allowlist_path.is_file():
        raise SystemExit(f"allowlist not found: {allowlist_path}")

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    identities = allowlist["identities"]

    if len(identities) != 670:
        raise SystemExit(f"expected 670 identities, got {len(identities)}")

    # Verify identity fields
    for ident in identities:
        for field in ["episode_id", "suite", "task_id", "state_id",
                       "collection_seed", "initial_state_sha256"]:
            if field not in ident:
                raise SystemExit(f"missing field '{field}' in identity {ident.get('episode_id', '?')}")

    print(f"Building shard plan: {len(identities)} identities → {args.n_shards} shards")

    plan = build_shard_plan(identities, args.n_shards)

    print(f"  total cost: {plan['cost_total']}")
    print(f"  cost imbalance: {plan['cost_imbalance_pct']}%")
    for i, shard in enumerate(plan["shards"]):
        sc = shard["suite_counts"]
        print(f"  shard {i}: {shard['n_identities']} identities, "
              f"cost={shard['total_cost']}, "
              f"suites=[L10:{sc['libero_10']} LG:{sc['libero_goal']} "
              f"LO:{sc['libero_object']} LS:{sc['libero_spatial']}]")

    # ── Seal output ──
    staging = out.parent / f".{out.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    published = False
    try:
        plan["input_allowlist_sha256"] = sha256_file(allowlist_path)
        plan["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        (staging / "FIT670_GPU_SHARD_PLAN.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

        # Per-shard identity lists
        shards_dir = staging / "shards"
        shards_dir.mkdir()
        for shard in plan["shards"]:
            (shards_dir / f"shard_{shard['shard_id']}_identities.json").write_text(
                json.dumps(shard["identities"], indent=2, sort_keys=True), encoding="utf-8")

        manifest = {
            "gate": "F670-D_SHARD_PLAN",
            "input_allowlist_sha256": plan["input_allowlist_sha256"],
            "n_shards": args.n_shards,
            "n_identities": len(identities),
            "cost_total": plan["cost_total"],
            "cost_imbalance_pct": plan["cost_imbalance_pct"],
            "created_at": plan["created_at"],
            "status": "FROZEN",
        }
        (staging / "SHARD_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        seal_output(staging)
        staging.rename(out)
        published = True

        print(f"\nShard plan sealed: {out}")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
