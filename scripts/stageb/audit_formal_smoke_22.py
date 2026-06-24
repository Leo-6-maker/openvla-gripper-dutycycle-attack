#!/usr/bin/env python3
"""Audit 22-cell formal smoke against frozen manifest and V6 completion protocol."""
import argparse, csv, json, hashlib, os, sys
from pathlib import Path
from collections import defaultdict

def fail(msg):
    print("FAIL: " + msg)
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--strict", action="store_true", default=True)
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(args.manifest)))
    base = Path(args.output_base)
    errors = 0
    seen_dirs = set()

    replay_groups = defaultdict(list)

    for i, row in enumerate(manifest):
        task = int(row["task"])
        tmpl = row["template"]
        out_name = row["output"]
        cell_dir = base / out_name
        seen_dirs.add(str(cell_dir))

        done = cell_dir / ".done"
        tel = cell_dir / "step_telemetry.csv"
        ep = cell_dir / "episode_summary.json"

        # File presence
        if not done.exists(): errors += fail("[%d] %s: .done missing" % (i, out_name)); continue
        if not tel.exists(): errors += fail("[%d] %s: telemetry missing" % (i, out_name)); continue
        if not ep.exists(): errors += fail("[%d] %s: summary missing" % (i, out_name)); continue

        # .done checks
        done_data = json.loads(done.read_text())
        if done_data.get("exit_code") != 0:
            errors += fail("[%d] %s: exit_code=%s" % (i, out_name, done_data.get("exit_code")))

        # telemetry checks
        tel_rows = list(csv.DictReader(open(tel)))
        if not tel_rows:
            errors += fail("[%d] %s: empty telemetry" % (i, out_name)); continue

        # Per-row provenance
        for j, tr in enumerate(tel_rows):
            if tr.get("condition", "") != "CLEAN":
                errors += fail("[%d] %s row %d: condition=%s" % (i, out_name, j, tr.get("condition")))
            if tr.get("pool", "") != "smoke":
                errors += fail("[%d] %s row %d: pool=%s" % (i, out_name, j, tr.get("pool")))
            if int(tr.get("task_idx", -1)) != task:
                errors += fail("[%d] %s row %d: task_idx mismatch" % (i, out_name, j))
            if int(tr.get("attack_count", -1)) != 0:
                errors += fail("[%d] %s row %d: attack_count=%s" % (i, out_name, j, tr.get("attack_count")))

        # Summary checks
        ep_data = json.loads(ep.read_text())
        if ep_data.get("condition", "") != "CLEAN":
            errors += fail("[%d] %s: summary condition not CLEAN" % (i, out_name))
        if ep_data.get("attack_frames", -1) != 0:
            errors += fail("[%d] %s: attack_frames=%s" % (i, out_name, ep_data.get("attack_frames")))
        if ep_data.get("pool", "") != "smoke":
            errors += fail("[%d] %s: summary pool=%s" % (i, out_name, ep_data.get("pool")))

        # n_steps consistency
        n_steps = ep_data.get("n_steps", -1)
        if len(tel_rows) != n_steps:
            errors += fail("[%d] %s: n_steps mismatch csv=%d summary=%d" % (i, out_name, len(tel_rows), n_steps))

        # UUID consistency
        summary_uuid = ep_data.get("run_uuid", "")
        done_uuid = done_data.get("run_uuid", "")
        if summary_uuid and done_uuid and summary_uuid != done_uuid:
            errors += fail("[%d] %s: UUID mismatch summary=%s done=%s" % (i, out_name, summary_uuid[:8], done_uuid[:8]))

        # Hash invariant
        orig_sha = ep_data.get("selected_original_state_sha256", "")
        pert_sha = ep_data.get("perturbed_pre_wait_sha256", "")
        if not orig_sha or not pert_sha:
            errors += fail("[%d] %s: missing SHA fields" % (i, out_name)); continue

        if tmpl == "P0":
            if orig_sha != pert_sha:
                errors += fail("[%d] %s P0: hash changed orig=%s pert=%s" % (i, out_name, orig_sha[:16], pert_sha[:16]))
        else:
            if orig_sha == pert_sha:
                errors += fail("[%d] %s %s: hash unchanged (NOOP)" % (i, out_name, tmpl))

        # Post-wait drift
        drift = ep_data.get("post_wait_translation_drift_m", -1)
        if isinstance(drift, (int, float)) and drift > 0.01:
            errors += fail("[%d] %s: post-wait drift %.4f m > 10mm" % (i, out_name, drift))

        # No states 28-49
        ps = ep_data.get("parent_state_id", -1)
        if 28 <= ps <= 49:
            errors += fail("[%d] %s: forbidden state %d in range 28-49" % (i, out_name, ps))

        # Replay group tracking
        rg = row.get("replay_group", "")
        if rg:
            key = (rg, task, str(ps), tmpl)
            tel_sha = hashlib.sha256(tel.read_bytes()).hexdigest()
            replay_groups[rg].append({"cell": out_name, "key": key, "tel_sha": tel_sha})

    # Replay consistency
    for rg, cells in replay_groups.items():
        if len(cells) >= 2:
            shas = [c["tel_sha"] for c in cells]
            if len(set(shas)) > 1:
                errors += fail("REPLAY_%s MISMATCH: different telemetry SHAs %s" % (rg, [s[:16] for s in shas]))
            else:
                print("REPLAY_%s MATCH: %d cells identical SHA=%s" % (rg, len(cells), shas[0][:16]))

    print("\nTotal: %d cells checked, %d errors" % (len(manifest), errors))
    if errors == 0:
        print("FORMAL SMOKE 22: PASS")
    else:
        print("FORMAL SMOKE 22: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
