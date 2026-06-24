#!/usr/bin/env python3
"""Audit 22-cell formal smoke against frozen manifest — complete implementation."""
import argparse, csv, json, hashlib, math, os, sys
from pathlib import Path
from collections import defaultdict

CANONICAL_FIELDS = [  # fields compared for replay identity (excludes volatile UUIDs/timing)
    "step", "raw_gripper", "env_gripper", "qpos_sum",
    "eef_x", "eef_y", "eef_z", "obj_x", "obj_y", "obj_z",
    "eef_obj_dist", "target_x", "target_y", "target_z",
    "detector_state", "corridor_p", "release_p", "pred_phase",
    "mlp_emit", "perturbation_template", "perturbation_seed",
    "raw_action_7d", "env_action_7d", "feat_valid",
]

def canonical_hash(tel_path):
    """SHA256 of canonical fields only (excludes run_uuid, cell_uuid, model_ms)."""
    rows = list(csv.DictReader(open(tel_path)))
    h = hashlib.sha256()
    for r in rows:
        vals = [str(r.get(f, "")) for f in CANONICAL_FIELDS]
        h.update("\x00".join(vals).encode())
    return h.hexdigest()

def fail(msg):
    print("FAIL: " + msg)
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_base", required=True)
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(args.manifest)))
    base = Path(args.output_base)
    errors = 0

    # ── Manifest preflight ──
    if len(manifest) != 22:
        errors += fail("Manifest has %d rows, expected 22" % len(manifest))

    outputs = [r["output"] for r in manifest]
    if len(set(outputs)) != len(outputs):
        errors += fail("Manifest has duplicate output paths")

    p0_base = [r for r in manifest if r["template"] == "P0" and not r["replay_group"]]
    nonzero_base = [r for r in manifest if r["template"] != "P0" and not r["replay_group"]]
    replays = [r for r in manifest if r["replay_group"]]
    templates_seen = set(r["template"] for r in nonzero_base)
    expected_templates = {"P1","P2","P3","P4","P5","P6","P7"}
    missing = expected_templates - templates_seen
    if missing:
        errors += fail("Nonzero templates missing from base: %s" % missing)

    # Replay groups validation
    groups = defaultdict(list)
    for r in replays:
        groups[r["replay_group"]].append(r)
    if set(groups.keys()) != {"A", "B"}:
        errors += fail("Expected replay groups A,B, got: %s" % set(groups.keys()))
    for g, cells in groups.items():
        if len(cells) != 2:
            errors += fail("Replay group %s has %d cells (expected 2)" % (g, len(cells)))
            continue
        c0, c1 = cells[0], cells[1]
        for k in ["task","state","template","seed"]:
            if c0[k] != c1[k]:
                errors += fail("Replay %s: %s mismatch (%s vs %s)" % (g, k, c0[k], c1[k]))

    # No forbidden states
    for r in manifest:
        s = int(r["state"])
        if 28 <= s <= 49:
            errors += fail("%s: forbidden state %d in range 28-49" % (r["output"], s))

    print("Preflight: %d manifest errors" % errors)

    # ── Per-cell integrity ──
    replay_hashes = defaultdict(list)
    for i, row in enumerate(manifest):
        out_name = row["output"]
        cell_dir = base / out_name
        exp_task = int(row["task"])
        exp_state = int(row["state"])
        exp_tmpl = row["template"]
        exp_seed = int(row["seed"])
        exp_dx = float(row["dx_m"]); exp_dy = float(row["dy_m"]); exp_dyaw = float(row["dyaw_rad"])

        done = cell_dir / ".done"
        tel = cell_dir / "step_telemetry.csv"
        ep = cell_dir / "episode_summary.json"

        if not done.exists(): errors += fail("[%d] %s: .done missing" % (i, out_name)); continue
        if not tel.exists(): errors += fail("[%d] %s: telemetry missing" % (i, out_name)); continue
        if not ep.exists(): errors += fail("[%d] %s: summary missing" % (i, out_name)); continue

        # .done
        done_data = json.loads(done.read_text())
        if done_data.get("exit_code") != 0:
            errors += fail("[%d] %s: exit_code=%s" % (i, out_name, done_data.get("exit_code")))
        d_tel_sha = done_data.get("telemetry_sha256") or done_data.get("telemetry_sha", "")
        d_ep_sha = done_data.get("summary_sha256", "")

        # telemetry
        tel_rows = list(csv.DictReader(open(tel)))
        if not tel_rows: errors += fail("[%d] %s: empty telemetry" % (i, out_name)); continue
        tel_sha = hashlib.sha256(tel.read_bytes()).hexdigest()

        # summary
        ep_data = json.loads(ep.read_text())
        ep_sha = hashlib.sha256(ep.read_bytes()).hexdigest()

        # 1. SHA cross-check
        if d_tel_sha and d_tel_sha != tel_sha:
            errors += fail("[%d] %s: done.telemetry_sha != actual telemetry SHA" % (i, out_name))
        if d_ep_sha and d_ep_sha != ep_sha:
            errors += fail("[%d] %s: done.summary_sha != actual summary SHA" % (i, out_name))

        # 2. n_steps
        n_steps = ep_data.get("n_steps", -1)
        if len(tel_rows) != n_steps:
            errors += fail("[%d] %s: n_steps mismatch csv=%d summary=%d" % (i, out_name, len(tel_rows), n_steps))

        # 3. condition/pool/attack_frames
        if ep_data.get("condition", "") != "CLEAN":
            errors += fail("[%d] %s: condition=%s" % (i, out_name, ep_data.get("condition")))
        if ep_data.get("pool", "") != "smoke":
            errors += fail("[%d] %s: pool=%s" % (i, out_name, ep_data.get("pool")))
        if ep_data.get("attack_frames", -1) != 0:
            errors += fail("[%d] %s: attack_frames=%s" % (i, out_name, ep_data.get("attack_frames")))

        # 4. task/state/template/seed match manifest
        if ep_data.get("task_idx", -1) != exp_task:
            errors += fail("[%d] %s: task_idx=%s expected=%d" % (i, out_name, ep_data.get("task_idx"), exp_task))
        if ep_data.get("parent_state_id", -1) != exp_state:
            errors += fail("[%d] %s: state=%s expected=%d" % (i, out_name, ep_data.get("parent_state_id"), exp_state))
        if ep_data.get("perturbation_template", "") != exp_tmpl:
            errors += fail("[%d] %s: template=%s expected=%s" % (i, out_name, ep_data.get("perturbation_template"), exp_tmpl))

        # 5. UUID consistency across all three artifacts
        ep_uuid = ep_data.get("run_uuid", "")
        done_uuid = done_data.get("run_uuid", "")
        tel_uuids = set(r.get("run_uuid", "") for r in tel_rows)
        ep_cell = ep_data.get("cell_uuid", "")
        done_cell = done_data.get("cell_uuid", "")
        tel_cells = set(r.get("cell_uuid", "") for r in tel_rows)

        if not ep_uuid: errors += fail("[%d] %s: summary.run_uuid empty" % (i, out_name))
        if not ep_cell: errors += fail("[%d] %s: summary.cell_uuid empty" % (i, out_name))
        if ep_uuid and done_uuid and ep_uuid != done_uuid:
            errors += fail("[%d] %s: run_uuid mismatch summary=%s done=%s" % (i, out_name, ep_uuid[:8], done_uuid[:8]))
        if ep_uuid and len(tel_uuids) == 1 and ep_uuid not in tel_uuids:
            errors += fail("[%d] %s: run_uuid mismatch summary vs telemetry" % (i, out_name))
        if ep_cell and done_cell and ep_cell != done_cell:
            errors += fail("[%d] %s: cell_uuid mismatch" % (i, out_name))
        if ep_cell and len(tel_cells) == 1 and ep_cell not in tel_cells:
            errors += fail("[%d] %s: cell_uuid mismatch summary vs telemetry" % (i, out_name))

        # Per-row fields in telemetry
        for j, tr in enumerate(tel_rows):
            if tr.get("condition", "") != "CLEAN":
                errors += fail("[%d] %s row%d: condition=%s" % (i, out_name, j, tr.get("condition")))
            if tr.get("pool", "") != "smoke":
                errors += fail("[%d] %s row%d: pool=%s" % (i, out_name, j, tr.get("pool")))
            if int(tr.get("task_idx", -1)) != exp_task:
                errors += fail("[%d] %s row%d: task_idx mismatch" % (i, out_name, j))

        # 6. Hash invariants
        orig_sha = ep_data.get("selected_original_state_sha256", "")
        pert_sha = ep_data.get("perturbed_pre_wait_sha256", "")
        if not orig_sha or not pert_sha:
            errors += fail("[%d] %s: missing SHA fields" % (i, out_name))
        elif exp_tmpl == "P0":
            if orig_sha != pert_sha:
                errors += fail("[%d] %s P0: hash changed" % (i, out_name))
        else:
            if orig_sha == pert_sha:
                errors += fail("[%d] %s %s: hash unchanged (NOOP)" % (i, out_name, exp_tmpl))

        # 7. Pre-wait pose delta matches manifest
        orig_pos = ep_data.get("selected_original_body_pos", [])
        pert_pos = ep_data.get("perturbed_pre_wait_body_pos", [])
        if len(orig_pos) >= 3 and len(pert_pos) >= 3:
            actual_dx = pert_pos[0] - orig_pos[0]
            actual_dy = pert_pos[1] - orig_pos[1]
            if abs(actual_dx - exp_dx) > 0.002:
                errors += fail("[%d] %s: dx mismatch manifest=%.6f actual=%.6f" % (i, out_name, exp_dx, actual_dx))
            if abs(actual_dy - exp_dy) > 0.002:
                errors += fail("[%d] %s: dy mismatch manifest=%.6f actual=%.6f" % (i, out_name, exp_dy, actual_dy))
            if exp_dyaw != 0:
                errors += fail("[%d] %s: dyaw check requires quaternion comparison in summary" % (i, out_name))
        else:
            errors += fail("[%d] %s: missing body pos fields" % (i, out_name))

        # 8. Post-wait perturbation preservation
        post_pos = ep_data.get("rollout_start_post_wait_body_pos", [])
        if len(pert_pos) >= 3 and len(post_pos) >= 3:
            drift = math.sqrt(sum((post_pos[j] - pert_pos[j])**2 for j in range(3)))
            if drift > 0.01:
                errors += fail("[%d] %s: post-wait drift %.6f m > 10mm" % (i, out_name, drift))

        # 9. Asset SHAs
        for key in ["bridge_sha256", "checkpoint_sha256", "teacher_config_sha256",
                     "target_resolver_sha256", "perturbation_generator_sha256"]:
            val = ep_data.get(key, "")
            if not val or val == "MISSING" or val == "NOT_COMPUTED":
                errors += fail("[%d] %s: %s is %s" % (i, out_name, key, val or "empty"))

        # 10. Replay canonical hash
        rg = row["replay_group"]
        if rg:
            ch = canonical_hash(str(tel))
            replay_hashes[rg].append({"cell": out_name, "canonical_hash": ch})

    # ── Replay consistency ──
    for g in ["A", "B"]:
        cells = replay_hashes.get(g, [])
        if len(cells) != 2:
            errors += fail("REPLAY_%s: expected 2 cells, got %d" % (g, len(cells)))
            continue
        if cells[0]["canonical_hash"] != cells[1]["canonical_hash"]:
            errors += fail("REPLAY_%s MISMATCH: %s != %s" % (g, cells[0]["canonical_hash"][:16], cells[1]["canonical_hash"][:16]))
        else:
            print("REPLAY_%s MATCH: %s == %s" % (g, cells[0]["cell"], cells[1]["cell"]))

    print("\n=== FORMAL SMOKE 22 AUDIT ===")
    print("Total: %d cells checked, %d errors" % (len(manifest), errors))
    if errors == 0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (%d errors)" % errors)
        sys.exit(1)

if __name__ == "__main__":
    main()
