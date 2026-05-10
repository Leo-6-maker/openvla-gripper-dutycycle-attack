#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.io import read_json, read_jsonl
from gripper_attack.logging_schema import validate_episode_record, validate_run_manifest, validate_step_record

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--tolerance", type=float, default=0.02); ap.add_argument("--allow_failed", action="store_true"); args = ap.parse_args()
    n = 0
    for mf in Path(args.root).rglob("run_manifest.json"):
        m = read_json(str(mf)); validate_run_manifest(m); n += 1
        status = str(m.get("status", ""))
        if status != "done" and not args.allow_failed:
            raise SystemExit(f"run status is {status!r} in {mf}; pass --allow_failed only when validating crash artifacts")
        progress_path = mf.parent / "progress.json"
        if progress_path.exists():
            progress = read_json(str(progress_path))
            if str(progress.get("status", "")) != status and not (status == "done" and progress.get("status") == "done"):
                raise SystemExit(f"progress/manifest status mismatch in {mf}: progress={progress.get('status')!r} manifest={status!r}")
        steps = read_jsonl(m["output_files"]["steps"]); episodes = read_jsonl(m["output_files"]["episodes"])
        for r in steps: validate_step_record(r)
        for e in episodes:
            validate_episode_record(e)
            if e["trigger_name"] != "dense" and float(e["attacked_step_ratio"]) > float(e["rho"]) + args.tolerance:
                raise SystemExit(f"attacked_step_ratio exceeds rho in {mf}")
        if m["trigger_name"] not in ("oracle_offline_entropy_topk","oracle_offline_margin_topk") and any(r.get("oracle") for r in steps):
            raise SystemExit(f"non-oracle run has oracle steps: {mf}")
    if n == 0: raise SystemExit(f"no run_manifest.json under {args.root}")
    print("[ok] validated runs=", n)

if __name__ == "__main__": main()
