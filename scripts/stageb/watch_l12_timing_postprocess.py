#!/usr/bin/env python3
"""L12 Post-Freeze CPU Postprocess Watcher.

Waits for GPU watcher PID 32355 to COMPLETE/FAIL, then:
  P1: Quota audit
  P2: Episode-by-episode audit
  P3: Recorder identity strict comparison
  P4: Repeatability comparison (original vs rerun)
  P5: Build handoff v2
  P6: Codex Layer3 alignment (if available)
  P7: Write reports
  P8: Commit/push

CPU-only. Does NOT use GPU. Does NOT restart PID 32355.
"""
import csv, hashlib, json, math, os, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.environ.get("L12_REPO_ROOT", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605")
PANEL_DIR = "/data/liuyu/outputs/l12_timing_panel_v2"
STATE_FILE = os.path.join(PANEL_DIR, "watcher_state.json")
OUT_DIR = os.path.join(PANEL_DIR, "postprocess")
WATCHER_PID = 32355
PY = "/data/aviary/envs/openvla_official_libero_20260525/bin/python"

EXPECTED_PARENTS = [
    ("butter", "11", "exact"), ("ketchup", "18", "exact"), ("orange_juice", "29", "exact"),
    ("milk", "7", "exact"), ("bbq_sauce", "40", "exact"), ("bbq_sauce", "27", "exact"),
    ("tomato_sauce", "23", "early"), ("salad_dressing", "32", "early"),
    ("cream_cheese", "1", "early"), ("cream_cheese", "20", "early"),
    ("salad_dressing", "24", "early"),
    ("salad_dressing", "11", "late"),
    ("ketchup", "34", "miss"), ("salad_dressing", "45", "miss"),
]
REPEAT_PARENTS = [("butter", "11"), ("tomato_sauce", "23"), ("salad_dressing", "11")]


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def wait_gpu_done():
    """P0: Wait for GPU watcher to complete."""
    print(f"P0: Waiting for GPU watcher PID {WATCHER_PID}...")
    while True:
        alive = pid_alive(WATCHER_PID)
        if os.path.exists(STATE_FILE):
            try:
                s = json.load(open(STATE_FILE))
                state = s.get("state", "UNKNOWN")
                episodes = len(s.get("episodes", {}))
                print(f"  state={state} episodes={episodes} alive={alive}")
                if state in ("COMPLETE", "FAILED") and not alive:
                    print(f"P0: GPU watcher {state}, PID exited. Proceeding.")
                    return s
            except Exception:
                pass
        if not alive and os.path.exists(STATE_FILE):
            try:
                s = json.load(open(STATE_FILE))
                if s.get("state") in ("COMPLETE", "FAILED"):
                    return s
            except Exception:
                pass
        time.sleep(30)


def audit_episode(ep_dir, expect_success=True):
    """P2: Strict episode audit. Returns dict of findings."""
    result = {"dir": ep_dir, "errors": [], "warnings": []}

    # Required files
    for fname in ["step_trace.csv", "action_identity.csv", "detector_candidates.csv",
                  "detector_emission.json", "latency.csv"]:
        if not os.path.exists(os.path.join(ep_dir, fname)):
            result["errors"].append(f"missing_{fname}")

    st_path = os.path.join(ep_dir, "step_trace.csv")
    if os.path.exists(st_path):
        rows = list(csv.DictReader(open(st_path)))
        if not rows:
            result["errors"].append("empty_trace")
        else:
            n = len(rows)
            result["n_steps"] = n
            success = rows[-1].get("success_done", "0")
            result["success"] = success
            if expect_success and success != "1":
                result["errors"].append(f"expected_success=1 got {success}")

    cands_path = os.path.join(ep_dir, "detector_candidates.csv")
    if os.path.exists(cands_path):
        cands = list(csv.DictReader(open(cands_path)))
        result["n_candidates"] = len(cands)

    emit_path = os.path.join(ep_dir, "detector_emission.json")
    if os.path.exists(emit_path):
        emit = json.load(open(emit_path))
        result["emit_step"] = emit.get("emit_step", -1)
        result["emit_score"] = emit.get("emit_score", 0)

    act_path = os.path.join(ep_dir, "action_identity.csv")
    if os.path.exists(act_path):
        act_rows = list(csv.DictReader(open(act_path)))
        has_fail = any(r.get("action_identical", "1") == "0" for r in act_rows)
        if has_fail:
            result["errors"].append("action_identity_fail")

    return result


def audit_quota(audit_results):
    """P1: Verify quota."""
    timing = [v for k, v in audit_results.items() if "timing" in k or k.startswith(("butter_","ketchup_","orange_juice_","milk_","bbq_","tomato_","salad_","cream_"))]
    primary = [r for r in timing if r.get("cat") != "miss"]
    n_ok = sum(1 for r in primary if not r.get("errors"))
    print(f"P1: {len(timing)} timing episodes, {n_ok} primary OK")
    return n_ok >= 10


def compare_recorder_identity():
    """P3: Strict OFF vs ON comparison."""
    off_dir = os.path.join(PANEL_DIR, "alphabet_soup_s2_reference_attempt1")
    on_dir = os.path.join(PANEL_DIR, "alphabet_soup_s2_shadow_attempt1")

    for d in [off_dir, on_dir]:
        if not os.path.exists(os.path.join(d, "step_trace.csv")):
            return {"errors": ["missing_recorder_episode"]}

    off_act = list(csv.DictReader(open(os.path.join(off_dir, "action_identity.csv"))))
    on_act = list(csv.DictReader(open(os.path.join(on_dir, "action_identity.csv"))))
    off_rows = list(csv.DictReader(open(os.path.join(off_dir, "step_trace.csv"))))
    on_rows = list(csv.DictReader(open(os.path.join(on_dir, "step_trace.csv"))))

    errors = []
    if len(off_rows) != len(on_rows):
        errors.append(f"length: {len(off_rows)} vs {len(on_rows)}")
    off_succ = off_rows[-1].get("success_done", "?") if off_rows else "?"
    on_succ = on_rows[-1].get("success_done", "?") if on_rows else "?"
    if off_succ != on_succ:
        errors.append(f"success: {off_succ} vs {on_succ}")

    n = min(len(off_act), len(on_act))
    act_diffs = sum(1 for i in range(n) if off_act[i].get("action_hash_post") != on_act[i].get("action_hash_post"))
    env_diffs = sum(1 for i in range(n) if off_act[i].get("env_action_hash") != on_act[i].get("env_action_hash"))
    obs_diffs = sum(1 for i in range(n) if off_act[i].get("obs_hash", "") != on_act[i].get("obs_hash", ""))

    if act_diffs > 0: errors.append(f"act_diffs={act_diffs}")
    if env_diffs > 0: errors.append(f"env_diffs={env_diffs}")
    if obs_diffs > 0: errors.append(f"obs_diffs={obs_diffs}")

    return {
        "off_steps": len(off_rows), "on_steps": len(on_rows),
        "off_success": off_succ, "on_success": on_succ,
        "act_diffs": act_diffs, "env_diffs": env_diffs, "obs_diffs": obs_diffs,
        "errors": errors,
    }


def compare_repeatability(task, sid):
    """P4: Compare original vs repeat run."""
    orig_dir = os.path.join(PANEL_DIR, f"{task}_s{sid}_shadow_attempt1")
    rep_dir = os.path.join(PANEL_DIR, f"{task}_s{sid}_shadow_attempt2")

    errors = []
    if not os.path.exists(os.path.join(orig_dir, "step_trace.csv")):
        return {"errors": ["missing_original"]}
    if not os.path.exists(os.path.join(rep_dir, "step_trace.csv")):
        return {"errors": ["missing_repeat"]}

    orig_rows = list(csv.DictReader(open(os.path.join(orig_dir, "step_trace.csv"))))
    rep_rows = list(csv.DictReader(open(os.path.join(rep_dir, "step_trace.csv"))))
    orig_act = list(csv.DictReader(open(os.path.join(orig_dir, "action_identity.csv"))))
    rep_act = list(csv.DictReader(open(os.path.join(rep_dir, "action_identity.csv"))))

    if len(orig_rows) != len(rep_rows):
        errors.append(f"length mismatch: {len(orig_rows)} vs {len(rep_rows)}")
    orig_succ = orig_rows[-1].get("success_done", "?") if orig_rows else "?"
    rep_succ = rep_rows[-1].get("success_done", "?") if rep_rows else "?"
    if orig_succ != rep_succ:
        errors.append(f"success mismatch: {orig_succ} vs {rep_succ}")

    n = min(len(orig_act), len(rep_act))
    act_diffs = sum(1 for i in range(n) if orig_act[i]["action_hash_post"] != rep_act[i]["action_hash_post"])
    env_diffs = sum(1 for i in range(n) if orig_act[i]["env_action_hash"] != rep_act[i]["env_action_hash"])
    if act_diffs > 0: errors.append(f"act_diffs={act_diffs}")
    if env_diffs > 0: errors.append(f"env_diffs={env_diffs}")

    # Compare candidates
    orig_cands = list(csv.DictReader(open(os.path.join(orig_dir, "detector_candidates.csv"))))
    rep_cands = list(csv.DictReader(open(os.path.join(rep_dir, "detector_candidates.csv"))))
    if len(orig_cands) != len(rep_cands):
        errors.append(f"cand_count: {len(orig_cands)} vs {len(rep_cands)}")
    else:
        for i in range(len(orig_cands)):
            if orig_cands[i].get("step") != rep_cands[i].get("step"):
                errors.append(f"cand_step_mismatch at {i}")
            if orig_cands[i].get("abstained") != rep_cands[i].get("abstained"):
                errors.append(f"abstain_mismatch at {i}")
            try:
                s1 = float(orig_cands[i].get("score", "nan"))
                s2 = float(rep_cands[i].get("score", "nan"))
                if abs(s1 - s2) > 1e-6:
                    errors.append(f"score_diff at {i}: {abs(s1-s2):.2e}")
            except Exception:
                pass

    orig_emit = json.load(open(os.path.join(orig_dir, "detector_emission.json")))
    rep_emit = json.load(open(os.path.join(rep_dir, "detector_emission.json")))
    if orig_emit.get("emit_step") != rep_emit.get("emit_step"):
        errors.append(f"emit mismatch: {orig_emit.get('emit_step')} vs {rep_emit.get('emit_step')}")

    return {
        "task": task, "sid": sid,
        "orig_steps": len(orig_rows), "rep_steps": len(rep_rows),
        "orig_success": orig_succ, "rep_success": rep_succ,
        "act_diffs": act_diffs, "env_diffs": env_diffs,
        "orig_cands": len(orig_cands), "rep_cands": len(rep_cands),
        "orig_emit": orig_emit.get("emit_step"), "rep_emit": rep_emit.get("emit_step"),
        "errors": errors,
    }


def build_handoff_v2(audit_results):
    """P5: Build handoff v2 from fresh episodes."""
    rows = []
    for task, sid, cat in EXPECTED_PARENTS:
        ep_dir = os.path.join(PANEL_DIR, f"{task}_s{sid}_shadow_attempt1")
        st_path = os.path.join(ep_dir, "step_trace.csv")
        if not os.path.exists(st_path):
            rows.append({"task": task, "state_id": sid, "status": "MISSING_TRACE"})
            continue

        trace_rows = list(csv.DictReader(open(st_path)))
        n = len(trace_rows)
        success = trace_rows[-1].get("success_done", "0") if trace_rows else "0"

        emit_path = os.path.join(ep_dir, "detector_emission.json")
        emit = json.load(open(emit_path)) if os.path.exists(emit_path) else {}
        d5_emit = emit.get("emit_step", -1)
        d5_score = emit.get("emit_score", 0)

        rows.append({
            "production_tag": "l12-d5-v1-production-20260617",
            "task": task, "state_id": sid, "split": "train_or_val",
            "clean_success": success,
            "d5_emit": d5_emit, "d5_score": d5_score,
            "d5_tau_margin": round(d5_score - 0.050, 6),
            "emit_anchor_offset": "",
            "window_relation": cat,
            "trace_path": ep_dir + "/step_trace.csv",
            "trace_sha256": sha256_file(st_path),
            "candidate_path": ep_dir + "/detector_candidates.csv",
            "candidate_sha256": sha256_file(os.path.join(ep_dir, "detector_candidates.csv")),
            "raw_frame_path": "FRAME_ARTIFACT_NOT_CAPTURED",
            "raw_frame_sha256": "",
            "processor_tensor_sha256": "",
            "frame_step": "",
            "frame_role": "",
            "preprocess_contract": "official_pil_lanczos_224_center_crop",
            "source_commit": os.popen("cd " + REPO + " && git rev-parse HEAD").read().strip()[:16],
            "source_gpu_pair": "2,6",
            "provenance_status": "audited" if success == "1" else "audited_diagnostic",
            "primary_or_diagnostic": "diagnostic" if cat == "miss" else "primary",
            "repeatability_status": "",
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    state = {"state": "P0_WAIT_GPU", "started": now_utc()}

    # P0
    gpu_state = wait_gpu_done()
    print(f"P0 complete: GPU watcher {gpu_state.get('state')}")

    # P1
    state["state"] = "P1_AUDIT_QUOTA"
    audit_results = {}
    for task, sid, cat in EXPECTED_PARENTS:
        ep_dir = os.path.join(PANEL_DIR, f"{task}_s{sid}_shadow_attempt1")
        r = audit_episode(ep_dir, cat != "miss")
        r["cat"] = cat
        audit_results[f"{task}_{sid}"] = r
    quota_ok = audit_quota(audit_results)
    print(f"P1: quota_ok={quota_ok}")

    # P2
    state["state"] = "P2_AUDIT_EPISODES"
    primary_errors = sum(1 for r in audit_results.values() if r.get("cat") != "miss" and r.get("errors"))
    print(f"P2: {primary_errors} primary episodes with errors")

    # P3
    state["state"] = "P3_RECORDER_IDENTITY"
    rec = compare_recorder_identity()
    rec_ok = len(rec.get("errors", [])) == 0
    print(f"P3: recorder_ok={rec_ok} errors={rec.get('errors')}")

    # P4
    state["state"] = "P4_REPEATABILITY"
    rep_results = {}
    for task, sid in REPEAT_PARENTS:
        r = compare_repeatability(task, sid)
        rep_results[f"{task}_{sid}"] = r
    rep_all_ok = all(len(r.get("errors", [])) == 0 for r in rep_results.values())
    timing_deterministic = rep_all_ok
    print(f"P4: repeatability_all_ok={rep_all_ok}")
    for k, r in rep_results.items():
        if r["errors"]:
            print(f"  {k}: errors={r['errors']}")

    # P5
    state["state"] = "P5_BUILD_HANDOFF"
    handoff_rows = build_handoff_v2(audit_results)
    handoff_csv = os.path.join(OUT_DIR, "l12_to_l3_timing_handoff_v2.csv")
    with open(handoff_csv, "w", newline="") as f:
        if handoff_rows:
            w = csv.DictWriter(f, fieldnames=list(handoff_rows[0].keys()))
            w.writeheader()
            w.writerows(handoff_rows)
    print(f"P5: handoff written ({len(handoff_rows)} rows)")

    # Write repeatability
    rep_csv = os.path.join(OUT_DIR, "l12_timing_repeatability_v2.csv")
    with open(rep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_sid", "orig_steps", "rep_steps",
                                          "orig_success", "rep_success",
                                          "act_diffs", "env_diffs",
                                          "orig_emit", "rep_emit", "errors"])
        w.writeheader()
        for k, r in rep_results.items():
            w.writerow({"task_sid": k, **{kk: str(vv) for kk, vv in r.items() if kk != "task" and kk != "sid"}})

    # Write summary
    summary = {
        "completed": now_utc(),
        "gpu_watcher_state": gpu_state.get("state"),
        "quota_ok": quota_ok,
        "primary_errors": primary_errors,
        "recorder_identity_ok": rec_ok,
        "repeatability_all_ok": rep_all_ok,
        "timing_deterministic": timing_deterministic,
        "n_timing_episodes": len(audit_results),
        "n_handoff_rows": len(handoff_rows),
    }
    with open(os.path.join(OUT_DIR, "postprocess_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # P6
    state["state"] = "P6_L3_ALIGNMENT"
    codex_files = [
        "/data/liuyu/outputs/tables/m3_v3_fixed_frame_panel.csv",
        "/data/liuyu/outputs/tables/l3_fixed_frame_timing_map.csv",
    ]
    has_codex = any(os.path.exists(f) for f in codex_files)
    if has_codex:
        for f in codex_files:
            if os.path.exists(f):
                print(f"P6: Codex evidence found at {f}")
                # Run alignment
                rc, out, _ = subprocess.run([
                    PY, f"{REPO}/scripts/stageb/analyze_l12_l3_window_alignment.py",
                    "--handoff", handoff_csv,
                    "--attack-windows", f,
                    "--delay-min", "0", "--delay-max", "20",
                    "--output", os.path.join(OUT_DIR, "l12_l3_alignment_v1.csv"),
                ], cwd=REPO, capture_output=True, text=True)
                print(out[-500:])
                break
    else:
        print("P6: WAITING_FOR_L3_EVIDENCE")

    # P7: Write report
    state["state"] = "P7_REPORT"
    report_path = os.path.join(OUT_DIR, "L12_POSTPROCESS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# L12 Post-Freeze Timing Panel — Postprocess Report\n\n")
        f.write(f"**Completed:** {now_utc()}\n\n")
        f.write(f"## GPU Watcher\n- State: {gpu_state.get('state')}\n")
        f.write(f"- Episodes: {len(gpu_state.get('episodes', {}))}\n\n")
        f.write(f"## Audit\n- Quota OK: {quota_ok}\n- Primary errors: {primary_errors}\n\n")
        f.write(f"## Recorder Identity\n- OK: {rec_ok}\n- Errors: {rec.get('errors')}\n\n")
        f.write(f"## Repeatability\n- All OK: {rep_all_ok}\n- Timing: {'DETERMINISTIC' if timing_deterministic else 'NONDETERMINISTIC_TIMING'}\n")
        for k, r in rep_results.items():
            f.write(f"- {k}: errors={r['errors']}\n")
        f.write(f"\n## Handoff\n- Rows: {len(handoff_rows)}\n")
        f.write(f"\n## Layer3 Alignment\n- Evidence: {'AVAILABLE' if has_codex else 'WAITING_FOR_L3_EVIDENCE'}\n")
    print(f"P7: report written")

    # P8: Commit/push
    state["state"] = "P8_COMMIT_PUSH"
    all_gates = (
        gpu_state.get("state") == "COMPLETE"
        and quota_ok
        and rec_ok
        and rep_all_ok
        and primary_errors == 0
    )
    if all_gates:
        print("P8: All gates PASS. Attempting commit...")
        # Add files
        subprocess.run(["git", "add", OUT_DIR.replace(REPO + "/", "")], cwd=REPO)
        subprocess.run(["git", "add", "scripts/stageb/watch_l12_timing_postprocess.py"], cwd=REPO)
        msg = "feat(l12): seal post-freeze timing panel v2 and repeatability audit"
        rc, _, _ = subprocess.run(["git", "commit", "-m", msg], cwd=REPO, capture_output=True, text=True)
        if rc == 0:
            subprocess.run(["git", "push", "origin", "exp/l12-timing-postprocess-20260617"], cwd=REPO)
            print("P8: Committed and pushed.")
        else:
            print("P8: Commit failed or nothing to commit.")
    else:
        print("P8: Gates not all PASS. Skipping commit.")
        print(f"  gpu_complete={gpu_state.get('state')=='COMPLETE'} quota={quota_ok} rec={rec_ok} rep={rep_all_ok} prim_err={primary_errors}")

    state["state"] = "COMPLETE" if all_gates else "PARTIAL"
    with open(os.path.join(OUT_DIR, "postprocess_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"Postprocess: {state['state']}")
    return 0 if all_gates else 1


if __name__ == "__main__":
    sys.exit(main())
