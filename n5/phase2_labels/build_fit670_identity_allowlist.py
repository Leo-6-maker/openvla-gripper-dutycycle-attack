"""[DeepSeek] FIT670 Identity Allowlist Builder — Gate F670-B.

Reads the D0-R2 670-identity CSV, initializes LIBERO for each identity,
computes initial_state_sha256 via pickle.dumps(canonical_state, protocol=4),
and produces FIT670_IDENTITY_ALLOWLIST.json.

Must run on the server (requires LIBERO).

Usage (server):
  python n5/phase2_labels/build_fit670_identity_allowlist.py \
    --dev-pool-manifest /path/to/DEV_POOL_IDENTITY_MANIFEST.csv \
    --out /path/to/allowlist_output \
    --seed 20260717
"""
import argparse, copy, csv, hashlib, json, os, pickle, shutil, sys, time, uuid
from pathlib import Path

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-pool-manifest", type=Path, required=True,
                        help="Path to D0-R2 DEV_POOL_IDENTITY_MANIFEST.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    csv_path = Path(args.dev_pool_manifest).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # ── Parse CSV ──
    identities = []
    seen = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suite = row["suite"].strip()
            task_id = int(row["task_id"].strip())
            state_id = int(row["state_id"].strip())
            ep_id = f"{suite}/task_{task_id:02d}/state_{state_id:02d}"
            if ep_id in seen:
                raise SystemExit(f"duplicate in CSV: {ep_id}")
            seen.add(ep_id)
            identities.append({
                "suite": suite, "task_id": task_id, "state_id": state_id,
                "episode_id": ep_id,
            })

    if len(identities) != 670:
        raise SystemExit(f"expected 670 identities, got {len(identities)}")

    suite_counts = {}
    for ident in identities:
        suite_counts[ident["suite"]] = suite_counts.get(ident["suite"], 0) + 1
    print(f"Loaded {len(identities)} identities from CSV")
    for suite, count in sorted(suite_counts.items()):
        print(f"  {suite}: {count}")

    # ── Compute initial_state_sha256 via LIBERO ──
    print("\nComputing initial_state_sha256 for all identities...")
    print("(this requires LIBERO — ~2-3 min with cached models)\n")

    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()

    suite_cache = {}
    n_done = 0
    t_start = time.time()

    for ident in identities:
        suite = ident["suite"]
        task_id = ident["task_id"]
        state_id = ident["state_id"]

        # Cache suite_obj per suite to avoid re-initialization
        if suite not in suite_cache:
            suite_cache[suite] = benchmark_dict[suite]()

        suite_obj = suite_cache[suite]
        states = suite_obj.get_task_init_states(task_id)

        if state_id >= len(states):
            raise SystemExit(
                f"state_id {state_id} >= {len(states)} for {ident['episode_id']}")

        canonical_state = copy.deepcopy(states[state_id])
        init_sha = sha256_bytes(pickle.dumps(canonical_state, protocol=4))

        ident["initial_state_sha256"] = init_sha
        ident["collection_seed"] = args.seed
        ident["fold"] = None  # derived later when inner-CV split is available

        n_done += 1
        if n_done % 50 == 0 or n_done == len(identities):
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(identities) - n_done) / rate if rate > 0 else 0
            print(f"  {n_done}/{len(identities)}  sha={init_sha[:16]}...  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  last={ident['episode_id']}")

    # ── Sort by episode_id for deterministic output ──
    identities.sort(key=lambda x: x["episode_id"])

    # ── Verify ──
    seen_after = set()
    for ident in identities:
        if ident["episode_id"] in seen_after:
            raise SystemExit(f"duplicate after sort: {ident['episode_id']}")
        seen_after.add(ident["episode_id"])
        if not ident.get("initial_state_sha256") or len(ident["initial_state_sha256"]) != 64:
            raise SystemExit(f"missing/invalid SHA for {ident['episode_id']}")

    # ── Compute identity_set_digest ──
    slim = [{k: v for k, v in i.items() if k != "fold"}
            for i in identities]
    id_set_digest = hashlib.sha256(
        json.dumps(slim, sort_keys=True).encode()).hexdigest()

    # ── Write allowlist ──
    staging = out.parent / f".{out.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    published = False
    try:
        allowlist = {
            "gate": "FIT670-INFERENCE_IDENTITY_ALLOWLIST",
            "schema": "FIT670_IDENTITY_ALLOWLIST_V1",
            "n_identities": len(identities),
            "collection_seed": args.seed,
            "identity_set_digest": id_set_digest,
            "identity_pool": "D0-R2_DEV_POOL_670",
            "protected_overlap": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "identities": identities,
        }

        (staging / "FIT670_IDENTITY_ALLOWLIST.json").write_text(
            json.dumps(allowlist, indent=2, sort_keys=True), encoding="utf-8")

        # Write build manifest
        manifest = {
            "gate": "F670-B_IDENTITY_ALLOWLIST_BUILD",
            "input_csv": str(csv_path),
            "input_csv_sha256": sha256_file(csv_path),
            "n_identities": len(identities),
            "suite_counts": suite_counts,
            "collection_seed": args.seed,
            "identity_set_digest": id_set_digest,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "FROZEN",
        }
        (staging / "BUILD_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        seal_output(staging)
        staging.rename(out)
        published = True

        elapsed = time.time() - t_start
        print(f"\nAllowlist sealed: {out}")
        print(f"  identities: {len(identities)}")
        print(f"  identity_set_digest: {id_set_digest}")
        print(f"  elapsed: {elapsed:.0f}s")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
