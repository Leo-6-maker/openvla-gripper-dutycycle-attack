#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.io import read_jsonl, write_csv, write_json
from gripper_attack.metrics import aggregate_run

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input_root", required=True); ap.add_argument("--output_dir", required=True); args = ap.parse_args()
    rows = []
    for epfile in Path(args.input_root).glob("*/episode_records.jsonl"):
        eps = read_jsonl(str(epfile)); s = aggregate_run(eps); first = eps[0] if eps else {}
        s.update({k:first.get(k,"") for k in ["run_id","task_id","suite","seed","trigger_name","rho"]}); rows.append(s)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); write_csv(str(out / "main_table.csv"), rows); write_json(str(out / "bootstrap_ci.json"), {"note":"paired bootstrap added after full_main"})
    (out / "acceptance_report.md").write_text("# V4 Acceptance Report\n\nGenerated rows: %d\n" % len(rows), encoding="utf-8")
    print("[ok] aggregate rows=", len(rows))

if __name__ == "__main__": main()
