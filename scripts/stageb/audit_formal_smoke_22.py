#!/usr/bin/env python3
"""Audit 22-cell formal smoke — complete fail-closed implementation."""
import argparse, csv, json, hashlib, math, os, sys
from pathlib import Path
from collections import defaultdict

CANONICAL_FIELDS = [
    "step", "raw_gripper", "env_gripper", "qpos_sum",
    "eef_x", "eef_y", "eef_z", "obj_x", "obj_y", "obj_z",
    "eef_obj_dist", "target_x", "target_y", "target_z",
    "detector_state", "corridor_p", "release_p", "pred_phase",
    "mlp_emit", "perturbation_template", "perturbation_seed",
    "raw_action_7d", "env_action_7d", "feat_valid",
]

def canonical_hash(tel_path):
    rows = list(csv.DictReader(open(tel_path)))
    if not rows:
        raise ValueError("EMPTY_CSV")
    header = set(rows[0].keys())
    missing = set(CANONICAL_FIELDS) - header
    if missing:
        raise ValueError("MISSING_COLUMNS:" + ",".join(sorted(missing)))
    # Reject None values in canonical fields (short/malformed rows)
    malformed = {
        field for field in CANONICAL_FIELDS
        if any(row[field] is None for row in rows)
    }
    if malformed:
        raise ValueError("MALFORMED_COLUMNS:" + ",".join(sorted(malformed)))
    payload = [{f: row[f] for f in CANONICAL_FIELDS} for row in rows]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()

def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

def quat_to_rot(q):
    """q in muJoCo [qw,qx,qy,qz] -> scipy [x,y,z,w]."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])

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

    # ═══ Manifest preflight ═══
    if len(manifest) != 22:
        errors += fail("Manifest has %d rows, expected 22" % len(manifest))

    outputs = [r["output"] for r in manifest]
    if len(set(outputs)) != len(outputs):
        errors += fail("Manifest has duplicate output paths")

    base_cells = [r for r in manifest if r["role"] == "base"]
    replay_cells = [r for r in manifest if r["role"] == "replay"]
    p0_base = [r for r in base_cells if r["template"] == "P0"]
    nz_base = [r for r in base_cells if r["template"] != "P0"]

    if len(p0_base) != 10:
        errors += fail("Expected 10 P0 base cells, got %d" % len(p0_base))
    if len(nz_base) != 10:
        errors += fail("Expected 10 nonzero base cells, got %d" % len(nz_base))
    if len(replay_cells) != 2:
        errors += fail("Expected 2 replay cells, got %d" % len(replay_cells))

    nz_templates = set(r["template"] for r in nz_base)
    for t in ["P1","P2","P3","P4","P5","P6","P7"]:
        if t not in nz_templates:
            errors += fail("Template %s missing from nonzero base" % t)

    # Replay groups: exactly A and B, 2 cells each, identical configs
    groups = defaultdict(list)
    for r in manifest:
        g = r["replay_group"]
        if g:
            groups[g].append(r)
    if set(groups.keys()) != {"A", "B"}:
        errors += fail("Expected replay groups {A,B}, got %s" % set(groups.keys()))
    for g in ["A", "B"]:
        cells = groups.get(g, [])
        if len(cells) != 2:
            errors += fail("Replay group %s has %d cells (expected 2)" % (g, len(cells)))
            continue
        # One must be role=base, one role=replay
        roles = set(c["role"] for c in cells)
        if roles != {"base", "replay"}:
            errors += fail("Replay group %s: expected roles {base,replay}, got %s" % (g, roles))
        # Config must be identical
        c0, c1 = cells[0], cells[1]
        for k in ["task","state","template","seed"]:
            if c0[k] != c1[k]:
                errors += fail("Replay %s: %s mismatch (%s vs %s)" % (g, k, c0[k], c1[k]))

    # Forbidden states
    for r in manifest:
        s = int(r["state"])
        if 28 <= s <= 49:
            errors += fail("%s: forbidden state %d" % (r["output"], s))

    print("Preflight: %d errors" % (errors - (0 if errors == 0 else 0)))
    if errors:
        sys.exit(1)

    # ═══ Per-cell integrity ═══
    replay_hashes = defaultdict(list)
    for i, row in enumerate(manifest):
        out_name = row["output"]
        cell_dir = base / out_name
        exp_task = int(row["task"]); exp_state = int(row["state"])
        exp_tmpl = row["template"]; exp_seed = int(row["seed"])
        exp_dx = float(row["dx_m"]); exp_dy = float(row["dy_m"])
        exp_dyaw = float(row["dyaw_rad"])

        done = cell_dir / ".done"; tel = cell_dir / "step_telemetry.csv"
        ep = cell_dir / "episode_summary.json"

        if not done.exists(): errors += fail("[%d] %s: .done missing" % (i, out_name)); continue
        if not tel.exists(): errors += fail("[%d] %s: telemetry missing" % (i, out_name)); continue
        if not ep.exists(): errors += fail("[%d] %s: summary missing" % (i, out_name)); continue

        done_data = json.loads(done.read_text())
        ep_data = json.loads(ep.read_text())
        tel_rows = list(csv.DictReader(open(tel)))
        if not tel_rows: errors += fail("[%d] %s: empty telemetry" % (i, out_name)); continue

        tel_sha = hashlib.sha256(tel.read_bytes()).hexdigest()
        ep_sha = hashlib.sha256(ep.read_bytes()).hexdigest()

        # ── .done checks (fail-closed) ──
        if done_data.get("exit_code") != 0:
            errors += fail("[%d] %s: exit_code=%s" % (i, out_name, done_data.get("exit_code")))
        d_tel_sha = done_data.get("telemetry_sha256") or done_data.get("telemetry_sha", "")
        d_ep_sha = done_data.get("summary_sha256", "")
        if not d_tel_sha:
            errors += fail("[%d] %s: .done missing telemetry SHA" % (i, out_name))
        elif d_tel_sha != tel_sha:
            errors += fail("[%d] %s: .done telemetry SHA mismatch" % (i, out_name))
        if not d_ep_sha:
            errors += fail("[%d] %s: .done missing summary SHA" % (i, out_name))
        elif d_ep_sha != ep_sha:
            errors += fail("[%d] %s: .done summary SHA mismatch" % (i, out_name))

        # ── n_steps ──
        n_steps = ep_data.get("n_steps", -1)
        if len(tel_rows) != n_steps:
            errors += fail("[%d] %s: n_steps csv=%d != summary=%d" % (i, out_name, len(tel_rows), n_steps))

        # ── condition/pool/attack ──
        if ep_data.get("condition","") != "CLEAN":
            errors += fail("[%d] %s: condition=%s" % (i, out_name, ep_data.get("condition")))
        if ep_data.get("pool","") != "smoke":
            errors += fail("[%d] %s: pool=%s" % (i, out_name, ep_data.get("pool")))
        if ep_data.get("attack_frames",-1) != 0:
            errors += fail("[%d] %s: attack_frames=%s" % (i, out_name, ep_data.get("attack_frames")))

        # ── task/state/template/seed match ──
        if ep_data.get("task_idx",-1) != exp_task:
            errors += fail("[%d] %s: task_idx=%s expected=%d" % (i, out_name, ep_data.get("task_idx"), exp_task))
        if ep_data.get("parent_state_id",-1) != exp_state:
            errors += fail("[%d] %s: state=%s expected=%d" % (i, out_name, ep_data.get("parent_state_id"), exp_state))
        if ep_data.get("perturbation_template","") != exp_tmpl:
            errors += fail("[%d] %s: template=%s expected=%s" % (i, out_name, ep_data.get("perturbation_template"), exp_tmpl))
        if ep_data.get("perturbation_seed",-1) != exp_seed:
            errors += fail("[%d] %s: seed=%s expected=%d" % (i, out_name, ep_data.get("perturbation_seed"), exp_seed))

        # ── UUID three-way strict ──
        ep_uuid = ep_data.get("run_uuid","")
        done_uuid = done_data.get("run_uuid","")
        tel_uuids = set(r.get("run_uuid","") for r in tel_rows)
        ep_cell = ep_data.get("cell_uuid","")
        done_cell = done_data.get("cell_uuid","")
        tel_cells = set(r.get("cell_uuid","") for r in tel_rows)

        if not ep_uuid: errors += fail("[%d] %s: summary.run_uuid empty" % (i, out_name))
        if not done_uuid: errors += fail("[%d] %s: .done.run_uuid empty" % (i, out_name))
        if tel_uuids != {ep_uuid}: errors += fail("[%d] %s: telemetry run_uuid not uniform or != summary" % (i, out_name))
        if done_uuid != ep_uuid: errors += fail("[%d] %s: run_uuid mismatch done vs summary" % (i, out_name))
        if not ep_cell: errors += fail("[%d] %s: summary.cell_uuid empty" % (i, out_name))
        if not done_cell: errors += fail("[%d] %s: .done.cell_uuid empty" % (i, out_name))
        if tel_cells != {ep_cell}: errors += fail("[%d] %s: telemetry cell_uuid not uniform or != summary" % (i, out_name))
        if done_cell != ep_cell: errors += fail("[%d] %s: cell_uuid mismatch" % (i, out_name))

        # ── Per-row telemetry fields ──
        for j, tr in enumerate(tel_rows):
            for fld in ["condition","pool","perturbation_template"]:
                if fld not in tr:
                    errors += fail("[%d] %s row%d: missing field '%s'" % (i, out_name, j, fld))
            if tr.get("condition","") != "CLEAN":
                errors += fail("[%d] %s row%d: condition" % (i, out_name, j))
            if tr.get("pool","") != "smoke":
                errors += fail("[%d] %s row%d: pool" % (i, out_name, j))
            if int(tr.get("task_idx",-1)) != exp_task:
                errors += fail("[%d] %s row%d: task_idx" % (i, out_name, j))
            if int(tr.get("parent_state_id",-1)) != exp_state:
                errors += fail("[%d] %s row%d: parent_state_id" % (i, out_name, j))
            if tr.get("perturbation_template","") != exp_tmpl:
                errors += fail("[%d] %s row%d: template" % (i, out_name, j))
            if int(tr.get("perturbation_seed",-1)) != exp_seed:
                errors += fail("[%d] %s row%d: seed" % (i, out_name, j))
            # attack_this: field must exist and be False/0 (not empty)
            if "attack_this" not in tr:
                errors += fail("[%d] %s row%d: missing attack_this" % (i, out_name, j))
            elif tr["attack_this"] not in ("False", "0"):
                errors += fail("[%d] %s row%d: attack_this=%s" % (i, out_name, j, tr["attack_this"]))
            # attack_count: field must exist and be 0
            if "attack_count" not in tr:
                errors += fail("[%d] %s row%d: missing attack_count" % (i, out_name, j))
            elif int(tr.get("attack_count",-1)) != 0:
                errors += fail("[%d] %s row%d: attack_count=%s" % (i, out_name, j, tr.get("attack_count")))

        # ── Hash invariants ──
        orig_sha = ep_data.get("selected_original_state_sha256","")
        pert_sha = ep_data.get("perturbed_pre_wait_sha256","")
        if not orig_sha or not pert_sha:
            errors += fail("[%d] %s: missing SHA" % (i, out_name))
        elif exp_tmpl == "P0":
            if orig_sha != pert_sha:
                errors += fail("[%d] %s P0: hash changed" % (i, out_name))
        else:
            if orig_sha == pert_sha:
                errors += fail("[%d] %s %s: hash unchanged" % (i, out_name, exp_tmpl))

        # ── Pre-wait pose delta vs manifest ──
        orig_pos = ep_data.get("selected_original_body_pos",[])
        pert_pos = ep_data.get("perturbed_pre_wait_body_pos",[])
        if len(orig_pos) < 3 or len(pert_pos) < 3:
            errors += fail("[%d] %s: missing body pos" % (i, out_name))
        else:
            actual_dx = pert_pos[0] - orig_pos[0]
            actual_dy = pert_pos[1] - orig_pos[1]
            if abs(actual_dx - exp_dx) > 0.002:
                errors += fail("[%d] %s: dx mismatch exp=%.6f act=%.6f" % (i, out_name, exp_dx, actual_dx))
            if abs(actual_dy - exp_dy) > 0.002:
                errors += fail("[%d] %s: dy mismatch exp=%.6f act=%.6f" % (i, out_name, exp_dy, actual_dy))

            # Yaw check for P5/P6
            if exp_dyaw != 0:
                orig_q = ep_data.get("selected_original_body_quat",[])
                pert_q = ep_data.get("perturbed_pre_wait_body_quat",[])
                if len(orig_q) >= 4 and len(pert_q) >= 4:
                    from scipy.spatial.transform import Rotation
                    orig_rot = Rotation.from_quat([orig_q[1],orig_q[2],orig_q[3],orig_q[0]])
                    pert_rot = Rotation.from_quat([pert_q[1],pert_q[2],pert_q[3],pert_q[0]])
                    rel = orig_rot.inv() * pert_rot
                    actual_dyaw = wrap_pi(rel.as_euler("xyz")[2])
                    if abs(wrap_pi(actual_dyaw - exp_dyaw)) > math.radians(2):
                        errors += fail("[%d] %s: dyaw mismatch exp=%.6f act=%.6f" % (i, out_name, exp_dyaw, actual_dyaw))
                else:
                    errors += fail("[%d] %s: missing quat for dyaw check" % (i, out_name))

            # Yaw check: P1-P4,P0,P7 should have near-zero rotation
            if exp_dyaw == 0:
                orig_q = ep_data.get("selected_original_body_quat",[])
                pert_q = ep_data.get("perturbed_pre_wait_body_quat",[])
                if len(orig_q) < 4 or len(pert_q) < 4:
                    errors += fail("[%d] %s: missing quat for zero-yaw rotation check" % (i, out_name))
                else:
                    from scipy.spatial.transform import Rotation
                    orig_rot = Rotation.from_quat([orig_q[1],orig_q[2],orig_q[3],orig_q[0]])
                    pert_rot = Rotation.from_quat([pert_q[1],pert_q[2],pert_q[3],pert_q[0]])
                    rel = orig_rot.inv() * pert_rot
                    rot_mag = float(np.linalg.norm(rel.as_rotvec()))
                    if rot_mag > math.radians(1):
                        errors += fail("[%d] %s: unexpected rotation %.4f deg (template=%s dyaw=0)" % (
                            i, out_name, math.degrees(rot_mag), exp_tmpl))

        # ── Post-wait perturbation preservation ──
        post_pos = ep_data.get("rollout_start_post_wait_body_pos",[])
        post_q = ep_data.get("rollout_start_post_wait_body_quat",[])
        if len(post_pos) < 3:
            errors += fail("[%d] %s: missing rollout_start_post_wait_body_pos" % (i, out_name))
        elif len(orig_pos) < 3:
            errors += fail("[%d] %s: missing selected_original_body_pos" % (i, out_name))
        else:
            post_dx = post_pos[0] - orig_pos[0]
            post_dy = post_pos[1] - orig_pos[1]
            if abs(post_dx - exp_dx) > 0.003:
                errors += fail("[%d] %s: post-wait dx lost: exp=%.6f post_vs_orig=%.6f" % (i, out_name, exp_dx, post_dx))
            if abs(post_dy - exp_dy) > 0.003:
                errors += fail("[%d] %s: post-wait dy lost: exp=%.6f post_vs_orig=%.6f" % (i, out_name, exp_dy, post_dy))
            drift = math.sqrt(sum((post_pos[j] - pert_pos[j])**2 for j in range(3)))
            if drift > 0.01:
                errors += fail("[%d] %s: post-wait drift %.6f m > 10mm" % (i, out_name, drift))

        # Post-wait yaw preservation
        orig_q = ep_data.get("selected_original_body_quat",[])
        if len(post_q) < 4:
            errors += fail("[%d] %s: missing rollout_start_post_wait_body_quat" % (i, out_name))
        elif len(orig_q) >= 4:
            from scipy.spatial.transform import Rotation
            orig_rot = Rotation.from_quat([orig_q[1],orig_q[2],orig_q[3],orig_q[0]])
            post_rot = Rotation.from_quat([post_q[1],post_q[2],post_q[3],post_q[0]])
            post_rel = orig_rot.inv() * post_rot
            post_yaw = wrap_pi(post_rel.as_euler("xyz")[2])
            post_rot_mag = float(np.linalg.norm(post_rel.as_rotvec()))
            if exp_dyaw != 0:
                if abs(wrap_pi(post_yaw - exp_dyaw)) > math.radians(3):
                    errors += fail("[%d] %s: post-wait yaw lost: exp=%.6f post_vs_orig=%.6f" % (
                        i, out_name, exp_dyaw, post_yaw))
            else:
                if post_rot_mag > math.radians(2):
                    errors += fail("[%d] %s: post-wait unexpected rotation %.4f deg" % (
                        i, out_name, math.degrees(post_rot_mag)))

        # ── Asset SHAs ──
        for key in ["bridge_sha256","checkpoint_sha256","teacher_config_sha256",
                     "target_resolver_sha256","perturbation_generator_sha256",
                     "vla_model_manifest_sha256","dataset_sha256","runner_sha256"]:
            val = ep_data.get(key,"")
            if not val or val in ("MISSING","NOT_COMPUTED"):
                errors += fail("[%d] %s: %s=%s" % (i, out_name, key, val or "empty"))

        # ── Perturbation spec exact match for P7 ──
        if exp_tmpl == "P7":
            ps = ep_data.get("perturbation_spec",{})
            if not isinstance(ps, dict):
                errors += fail("[%d] %s: perturbation_spec missing or not dict" % (i, out_name))
            else:
                for fld in ["dx_m","dy_m","base_seed","template_id"]:
                    if fld not in ps:
                        errors += fail("[%d] %s: P7 perturbation_spec missing '%s'" % (i, out_name, fld))
                if "dx_m" in ps and abs(float(ps["dx_m"]) - exp_dx) > 1e-9:
                    errors += fail("[%d] %s: P7 spec.dx_m mismatch manifest=%.15f spec=%.15f" % (i, out_name, exp_dx, float(ps["dx_m"])))
                if "dy_m" in ps and abs(float(ps["dy_m"]) - exp_dy) > 1e-9:
                    errors += fail("[%d] %s: P7 spec.dy_m mismatch manifest=%.15f spec=%.15f" % (i, out_name, exp_dy, float(ps["dy_m"])))
                if "template_id" in ps and ps["template_id"] != exp_tmpl:
                    errors += fail("[%d] %s: P7 spec.template_id=%s expected=%s" % (i, out_name, ps["template_id"], exp_tmpl))
                if "base_seed" in ps and int(ps["base_seed"]) != exp_seed:
                    errors += fail("[%d] %s: P7 spec.base_seed=%s expected=%d" % (i, out_name, ps["base_seed"], exp_seed))

        # ── Replay canonical hash ──
        rg = row["replay_group"]
        if rg:
            try:
                ch = canonical_hash(str(tel))
            except ValueError as exc:
                errors += fail("[%d] %s: canonical_hash %s" % (i, out_name, exc))
                ch = "ERROR:" + str(exc)[:80]
            replay_hashes[rg].append({"cell": out_name, "canonical_hash": ch})

    # ═══ Cross-cell asset SHA consistency ═══
    asset_keys = ["bridge_sha256","runner_sha256","checkpoint_sha256",
                  "teacher_config_sha256","target_resolver_sha256",
                  "perturbation_generator_sha256",
                  "vla_model_manifest_sha256","dataset_sha256"]
    for key in asset_keys:
        values = set()
        for row in manifest:
            ep_data = json.loads((base / row["output"] / "episode_summary.json").read_text())
            v = ep_data.get(key,"")
            if v and v not in ("MISSING","NOT_COMPUTED"):
                values.add(v)
        if len(values) == 0:
            errors += fail("Asset %s: no valid value in any cell" % key)
        elif len(values) > 1:
            errors += fail("Asset %s: inconsistent across cells (%d unique values)" % (key, len(values)))

    # ═══ Replay consistency ═══
    for g in ["A","B"]:
        cells = replay_hashes.get(g,[])
        if len(cells) != 2:
            errors += fail("REPLAY_%s: expected 2, got %d" % (g, len(cells)))
            continue
        if cells[0]["canonical_hash"] != cells[1]["canonical_hash"]:
            errors += fail("REPLAY_%s MISMATCH: %s != %s" % (g,
                cells[0]["canonical_hash"][:16], cells[1]["canonical_hash"][:16]))
        else:
            print("REPLAY_%s MATCH: %s == %s" % (g, cells[0]["cell"], cells[1]["cell"]))

    # ═══ Negative completion test ═══
    neg_test_file = base / "negative_duplicate_test.json"
    if not neg_test_file.exists():
        errors += fail("negative_duplicate_test.json not found")
    else:
        neg = json.loads(neg_test_file.read_text())
        neg_ok = True
        if neg.get("collector_nonzero_exit") != True:
            errors += fail("negative_test: collector did not exit non-zero"); neg_ok = False
        if not neg.get("stderr_contains_CELL_ALREADY_COMPLETE"):
            errors += fail("negative_test: stderr missing CELL_ALREADY_COMPLETE"); neg_ok = False
        if neg.get("files_unchanged") != True:
            errors += fail("negative_test: original files modified"); neg_ok = False
        if neg_ok:
            print("NEGATIVE_COMPLETION_TEST: PASS")
        else:
            print("NEGATIVE_COMPLETION_TEST: FAIL")

    print("\n=== FORMAL SMOKE 22 AUDIT ===")
    print("Total: %d cells, %d errors" % (len(manifest), errors))
    if errors == 0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (%d errors)" % errors)
        sys.exit(1)

if __name__ == "__main__":
    import numpy as np
    main()
