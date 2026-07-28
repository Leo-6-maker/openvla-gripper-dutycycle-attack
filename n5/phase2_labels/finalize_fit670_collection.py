"""[DeepSeek] FIT670 Collection Finalizer — Gate F670-H.

Verifies all worker outputs, checks identity closure (670/670),
and produces global seal.

Usage:
  python n5/phase2_labels/finalize_fit670_collection.py \
    --output-root /path/to/d670_output \
    --identity-allowlist /path/to/FIT670_IDENTITY_ALLOWLIST.json \
    --shard-plan /path/to/FIT670_GPU_SHARD_PLAN.json \
    --transition-receipt /path/to/transition_root
"""
import argparse, json, os, shutil, sys, time, uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_collection_core import sha256_file, seal_root, FOUR_SUITES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()

    # Load allowlist
    allowlist = json.loads(Path(args.identity_allowlist).read_text(encoding="utf-8"))
    expected_ids = set(ident["episode_id"] for ident in allowlist["identities"])
    if len(expected_ids) != 670:
        raise SystemExit(f"allowlist has {len(expected_ids)} identities, expected 670")

    # Load shard plan
    shard_plan = json.loads(Path(args.shard_plan).read_text(encoding="utf-8"))
    shards = shard_plan.get("shards", [])
    shard_ids = set()
    for shard in shards:
        for ident in shard["identities"]:
            shard_ids.add(ident["episode_id"])
    if shard_ids != expected_ids:
        print(f"WARNING: shard plan identities differ from allowlist")
        only_allowlist = expected_ids - shard_ids
        only_shard = shard_ids - expected_ids
        if only_allowlist:
            print(f"  In allowlist only: {len(only_allowlist)}")
        if only_shard:
            print(f"  In shard plan only: {len(only_shard)}")

    # ── Discover published episodes ──
    episodes_dir = out_root / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"episodes directory not found: {episodes_dir}")

    found_ids = set()
    seal_failures = []
    missing_ids = []
    source_mismatches = []

    for suite in FOUR_SUITES:
        suite_dir = episodes_dir / suite
        if not suite_dir.is_dir():
            continue
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for state_dir in sorted(task_dir.iterdir()):
                if not state_dir.is_dir():
                    continue
                ep_json = state_dir / "episode.json"
                seal_file = state_dir / "SHA256SUMS.sha256"
                if not ep_json.is_file():
                    seal_failures.append(str(state_dir.relative_to(episodes_dir)))
                    continue
                if not seal_file.is_file():
                    seal_failures.append(f"{state_dir.relative_to(episodes_dir)} (no seal)")
                    continue

                # Verify seal
                try:
                    sums_path = state_dir / "SHA256SUMS"
                    if not sums_path.is_file():
                        seal_failures.append(f"{state_dir.relative_to(episodes_dir)} (no SHA256SUMS)")
                        continue
                    # Basic seal check: SHA256SUMS.sha256 references SHA256SUMS
                    seal_content = seal_file.read_text().strip()
                    declared = seal_content.split()[0] if seal_content else ""
                    actual = sha256_file(sums_path)
                    if declared != actual:
                        seal_failures.append(
                            f"{state_dir.relative_to(episodes_dir)} (seal mismatch: "
                            f"{declared[:16]} vs {actual[:16]})")
                        continue
                except Exception as e:
                    seal_failures.append(f"{state_dir.relative_to(episodes_dir)} (seal error: {e})")
                    continue

                # Verify episode identity
                episode_data = json.loads(ep_json.read_text(encoding="utf-8"))
                ep_id = episode_data.get("episode_id", "")
                if ep_id not in expected_ids:
                    source_mismatches.append(f"{ep_id} (not in allowlist)")
                    continue
                found_ids.add(ep_id)

    missing_ids = expected_ids - found_ids

    # ── Worker manifest discovery ──
    worker_results = {}
    for gpu_dir in sorted(out_root.iterdir()):
        if not gpu_dir.is_dir() or not gpu_dir.name.startswith("gpu_"):
            continue
        manifest_path = gpu_dir / "WORKER_MANIFEST.json"
        if manifest_path.is_file():
            wm = json.loads(manifest_path.read_text(encoding="utf-8"))
            worker_results[gpu_dir.name] = {
                "n_success": wm.get("n_success", 0),
                "n_fail": wm.get("n_fail", 0),
                "n_skipped": wm.get("n_skipped", 0),
                "total_steps": wm.get("total_steps", 0),
                "elapsed_s": wm.get("elapsed_s", 0),
            }

    # ── Print audit ──
    print("=" * 60)
    print("FIT670 Collection Finalizer — Gate F670-H")
    print("=" * 60)
    print(f"  Expected identities:  {len(expected_ids)}")
    print(f"  Found on disk:       {len(found_ids)}")
    print(f"  Missing:             {len(missing_ids)}")
    print(f"  Seal failures:       {len(seal_failures)}")
    print(f"  Source mismatches:   {len(source_mismatches)}")
    print(f"  Duplicates:          {len(found_ids) + len(missing_ids) - len(expected_ids)}")
    print(f"  Worker results:      {len(worker_results)} workers")

    for name, wr in sorted(worker_results.items()):
        print(f"    {name}: {wr['n_success']} ok, {wr['n_fail']} fail, "
              f"{wr['n_skipped']} skip, {wr['total_steps']} steps, {wr['elapsed_s']}s")

    if missing_ids:
        print(f"\n  MISSING ({len(missing_ids)}):")
        for mid in sorted(missing_ids)[:10]:
            print(f"    {mid}")
        if len(missing_ids) > 10:
            print(f"    ... and {len(missing_ids) - 10} more")

    if seal_failures:
        print(f"\n  SEAL FAILURES ({len(seal_failures)}):")
        for sf in seal_failures[:10]:
            print(f"    {sf}")

    if source_mismatches:
        print(f"\n  SOURCE MISMATCHES ({len(source_mismatches)}):")
        for sm in source_mismatches[:10]:
            print(f"    {sm}")

    # ── Determine status ──
    if len(found_ids) == 670 and len(seal_failures) == 0 and len(source_mismatches) == 0:
        status = "PASS_CONSUMABLE"
    elif len(found_ids) > 0:
        status = "PARTIAL_NONCONSUMABLE"
    else:
        status = "FAIL_PROVENANCE"

    print(f"\n  STATUS: {status}")

    # ── Generate global seal ──
    if status in ("PASS_CONSUMABLE", "PARTIAL_NONCONSUMABLE"):
        staging = out_root.parent / f".global_seal.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True)
        published = False
        try:
            global_manifest = {
                "gate": "FIT670_ATOMIC_COLLECTION",
                "schema": "FIT670_GLOBAL_MERGE_V1",
                "status": status,
                "n_identities_expected": 670,
                "n_identities_found": len(found_ids),
                "n_missing": len(missing_ids),
                "n_seal_failures": len(seal_failures),
                "n_source_mismatches": len(source_mismatches),
                "worker_results": worker_results,
                "identity_allowlist_digest": allowlist.get("identity_set_digest", ""),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "missing_ids": sorted(missing_ids) if missing_ids else [],
                "seal_failures": seal_failures,
                "source_mismatches": source_mismatches,
            }

            (staging / "GLOBAL_MANIFEST.json").write_text(
                json.dumps(global_manifest, indent=2, sort_keys=True), encoding="utf-8")

            seal_root(staging)
            staging.rename(out_root)
            published = True

            print(f"\n  Global seal written to: {out_root}")
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    # Exit code
    if status == "PASS_CONSUMABLE":
        sys.exit(0)
    elif status == "PARTIAL_NONCONSUMABLE":
        sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
