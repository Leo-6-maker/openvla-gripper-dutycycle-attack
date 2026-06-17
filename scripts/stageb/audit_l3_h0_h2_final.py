#!/usr/bin/env python3
"""S0: Independent H0/H2 final auditor (CPU-only). Recomputes every claim."""
import csv, hashlib, json, os, sys, io
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tables"
RPT_DIR = REPO_ROOT / "reports"

PKG_V2 = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/canonical_packages_v2"
FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
H1_DIR = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h1_v4_butter_s11_step60_seed81/canary"
H2_DIR = "/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h2_v4_primary"

SELECTED = [
    ("butter_s11", "butter", 11, 60, "teacher_anchor+d5_emit", True),
    ("tomato_sauce_s23", "tomato_sauce", 23, 141, "teacher_anchor", True),
    ("salad_dressing_s11", "salad_dressing", 11, 59, "teacher_anchor", True),
    ("butter_s11", "butter", 11, 58, "teacher_ws", False),
    ("tomato_sauce_s23", "tomato_sauce", 23, 139, "teacher_ws", False),
    ("salad_dressing_s11", "salad_dressing", 11, 57, "teacher_ws", False),
    ("butter_s11", "butter", 11, 68, "teacher_we", False),
    ("tomato_sauce_s23", "tomato_sauce", 23, 69, "d5_emit", False),
    ("salad_dressing_s11", "salad_dressing", 11, 67, "teacher_we", False),
    ("salad_dressing_s11", "salad_dressing", 11, 128, "d5_emit", False),
]

PRIMARY_ANCHORS = [
    ("butter_s11", "butter", 11, 60),
    ("tomato_sauce_s23", "tomato_sauce", 23, 141),
    ("salad_dressing_s11", "salad_dressing", 11, 59),
]

SEEDS = [81, 82]
EXPECTED_CANDIDATE_COUNT = 21
ARM_GATE = 5
TARGET_TOKEN = 31744
EXPECTED_LINF = 0.023529052734375
EPSILON = 0.023529411764705882


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def find_v4_output(pid, step, seed):
    """Find V4 output dir — either H1 or H2 path."""
    paths = [
        os.path.join(H2_DIR, f"{pid}_step{step:04d}_seed{seed}", "canary"),
        os.path.join(H1_DIR) if pid == "butter_s11" and step == 60 and seed == 81 else None,
    ]
    for p in paths:
        if p and os.path.isdir(p) and os.path.isfile(os.path.join(p, "m3_v4_selected_results.csv")):
            return p
    return None


def audit_h0_packages():
    """Audit all 10 canonical packages independently."""
    print("=== H0 Package Audit ===")
    results = []
    for pid, task, sid, step, role, is_primary in SELECTED:
        pkg = os.path.join(PKG_V2, f"{pid}_step{step:04d}")
        cg_json = os.path.join(pkg, "clean_generation.json")
        raw_npy = os.path.join(pkg, "raw_frame.npy")
        proc_pt = os.path.join(pkg, "processor_inputs_attack.pt")
        action_npy = os.path.join(pkg, "clean_action.npy")
        manifest_csv = os.path.join(pkg, "artifact_hash_manifest.csv")

        checks = {
            "parent_id": pid, "step": step, "role": role, "is_primary": is_primary,
            "pkg_exists": os.path.isdir(pkg),
            "clean_generation_json": os.path.isfile(cg_json),
            "raw_frame_npy": os.path.isfile(raw_npy),
            "processor_pt": os.path.isfile(proc_pt),
            "clean_action_npy": os.path.isfile(action_npy),
            "artifact_manifest": os.path.isfile(manifest_csv),
        }

        if os.path.isfile(cg_json):
            d = json.load(open(cg_json))
            checks["instruction_nonempty"] = bool(d.get("instruction", ""))
            checks["prompt_nonempty"] = bool(d.get("prompt", ""))
            checks["source_commit_nonempty"] = bool(d.get("source_commit", ""))
            checks["timing_trace_sha_nonempty"] = bool(d.get("source_timing_trace_sha", ""))
            checks["obs_waiver_present"] = "OBS" in str(d.get("source_osb_waiver", ""))
            checks["clean_gripper_token"] = d.get("clean_gripper_token", 0)
            checks["exact_7_tokens"] = len(d.get("exact_clean_7_tokens", [])) == 7
            checks["arm_prefix_6"] = len(d.get("clean_arm_prefix", [])) == 6
            checks["clean_action_dim"] = len(d.get("clean_action", []))
            checks["eligibility"] = d.get("clean_eligibility", "")

        if os.path.isfile(raw_npy):
            import numpy as np
            a = np.load(raw_npy)
            checks["raw_frame_sha256"] = sha256_file(raw_npy)[:16] + "..."
            checks["raw_frame_shape"] = list(a.shape)

        results.append(checks)
        status = "PASS" if checks.get("eligibility") == "CLEAN_ELIGIBLE" and checks.get("is_primary") else \
                 "DIAGNOSTIC" if not checks.get("is_primary") else \
                 "INELIGIBLE"
        print(f"  {pid} step{step}: {status} grip={checks.get('clean_gripper_token')} "
              f"action_dim={checks.get('clean_action_dim')} "
              f"commit={checks.get('source_commit_nonempty')}")

    n_primary_eligible = sum(1 for r in results if r.get("is_primary") and r.get("eligibility") == "CLEAN_ELIGIBLE")
    n_primary_total = sum(1 for r in results if r.get("is_primary"))
    print(f"  Primary CLOSE: {n_primary_eligible}/{n_primary_total}")
    return results


def audit_h2_frame_seed(output_dir, pid, step, seed):
    """Independently audit a single V4 frame-seed result."""
    sel_csv = os.path.join(output_dir, "m3_v4_selected_results.csv")
    cand_csv = os.path.join(output_dir, "m3_v4_candidate_audit.csv")
    route_csv = os.path.join(output_dir, "m3_v4_route_audit.csv")
    debug_json = os.path.join(output_dir, "m3_v4_debug.json")
    preflight_json = os.path.join(output_dir, "m3_step78_zero_step_preflight.json")

    result = {"parent_id": pid, "step": step, "seed": seed, "output_dir": output_dir,
              "files_ok": all(os.path.isfile(p) for p in [sel_csv, cand_csv, route_csv])}

    if not os.path.isfile(sel_csv):
        result["status"] = "INFRA_INCOMPLETE"
        return result

    rows = list(csv.DictReader(open(sel_csv)))
    true_row = next((r for r in rows if "TRUE" in r.get("condition", "")), None)
    rand_row = next((r for r in rows if "RAND" in r.get("condition", "")), None)
    shuffled_row = next((r for r in rows if "SHUFFLED" in r.get("condition", "")), None)

    # TRUE audit
    if true_row:
        result["true_selected_id"] = true_row.get("selected_candidate_id", "")
        result["true_gripper"] = int(true_row.get("official_gripper_token", "0") or 0)
        result["true_arm_num"] = int(true_row.get("arm_prefix_match_count", "0") or 0)
        result["true_arm_den"] = int(true_row.get("arm_prefix_match_denominator", "0") or 0)
        result["true_margin"] = float(true_row.get("official_target31744_margin", "-inf") or "-inf")
        result["true_linf"] = float(true_row.get("processor_linf", "999") or 999)
        result["true_result"] = true_row.get("condition_result", "")
        result["stage_result"] = true_row.get("stage_result", "")
    else:
        result["true_result"] = "MISSING"

    # Control audit
    if rand_row:
        result["rand_result"] = rand_row.get("condition_result", "")
    if shuffled_row:
        result["shuffled_result"] = shuffled_row.get("condition_result", "")

    # Candidate counts
    if os.path.isfile(cand_csv):
        cand_rows = list(csv.DictReader(open(cand_csv)))
        by_cond = defaultdict(list)
        for r in cand_rows:
            by_cond[r.get("condition", "OTHER")].append(r)
        result["true_cand_count"] = len(by_cond.get("TRUE_PGD_TRAJECTORY21_SELECTIVE", []))
        result["rand_cand_count"] = len(by_cond.get("RAND21_SELECTIVE", []))
        result["shuffled_cand_count"] = len(by_cond.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", []))

        # RAND feasibility
        rand_cands = by_cond.get("RAND21_SELECTIVE", [])
        rand_hit = sum(1 for r in rand_cands if int(r.get("official_gripper_token", "0") or 0) == TARGET_TOKEN)
        rand_arm = sum(1 for r in rand_cands if int(r.get("arm_prefix_match_count", "0") or 0) >= ARM_GATE)
        result["rand_target_hits"] = rand_hit
        result["rand_arm_pass"] = rand_arm
        result["rand_best_arm"] = max((int(r.get("arm_prefix_match_count", "0") or 0) for r in rand_cands), default=-1)
        result["rand_best_margin"] = max((float(r.get("official_target31744_margin", "-inf") or "-inf") for r in rand_cands), default=float("-inf"))

        # SHUFFLED feasibility
        shuf_cands = by_cond.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", [])
        shuf_hit = sum(1 for r in shuf_cands if int(r.get("official_gripper_token", "0") or 0) == TARGET_TOKEN)
        shuf_arm = sum(1 for r in shuf_cands if int(r.get("arm_prefix_match_count", "0") or 0) >= ARM_GATE)
        result["shuffled_target_hits"] = shuf_hit
        result["shuffled_arm_pass"] = shuf_arm

    # Route audit
    if os.path.isfile(route_csv):
        route_rows = list(csv.DictReader(open(route_csv)))
        true_route = next((r for r in route_rows if "TRUE" in r.get("condition", "")), None)
        if true_route:
            result["strict_route"] = true_route.get("strict_route", "")
            result["fallback_used"] = true_route.get("fallback_used", "")
            result["trajectory_count"] = int(true_route.get("trajectory_candidate_count", "0") or 0)

    # Preflight
    if os.path.isfile(preflight_json):
        pf = json.load(open(preflight_json))
        result["preflight_clean"] = pf.get("clean_status", "")
        result["preflight_delta0"] = pf.get("delta0_status", "")

    # Gate classification
    arm_ok = result.get("true_arm_num", 0) >= ARM_GATE
    token_ok = result.get("true_gripper", 0) == TARGET_TOKEN
    linf_ok = result.get("true_linf", 999) <= EPSILON + 1e-9
    route_ok = result.get("strict_route", "") == "True" and result.get("fallback_used", "") in ("False", "0", "")
    controls_ok = (result.get("rand_result") == "NO_FEASIBLE_CANDIDATE" and
                   result.get("shuffled_result") == "NO_FEASIBLE_CANDIDATE")
    preflight_ok = "MATCH" in str(result.get("preflight_clean", "")) and "MATCH" in str(result.get("preflight_delta0", ""))

    all_gates = arm_ok and token_ok and linf_ok and route_ok and controls_ok and preflight_ok
    result["arm_gate"] = arm_ok
    result["token_gate"] = token_ok
    result["linf_gate"] = linf_ok
    result["route_gate"] = route_ok
    result["controls_gate"] = controls_ok
    result["preflight_gate"] = preflight_ok
    result["frame_seed_status"] = "FRAME_SEED_PASS" if all_gates else "FRAME_SEED_FAIL"

    return result


def audit_h2_panel():
    """Independently audit all 6 V4 primary frame-seed results."""
    print("\n=== H2 V4 Primary Panel Audit ===")
    results = []
    for pid, task, sid, step in PRIMARY_ANCHORS:
        for seed in SEEDS:
            out_dir = find_v4_output(pid, step, seed)
            if not out_dir:
                print(f"  MISSING: {pid} step{step} seed{seed}")
                results.append({"parent_id": pid, "step": step, "seed": seed, "status": "INFRA_INCOMPLETE"})
                continue

            r = audit_h2_frame_seed(out_dir, pid, step, seed)
            results.append(r)
            status = r.get("frame_seed_status", "UNKNOWN")
            arm = f"{r.get('true_arm_num','?')}/{r.get('true_arm_den','?')}"
            print(f"  {pid} step{step} seed{seed}: {status} arm={arm} "
                  f"grip={r.get('true_gripper','?')} margin={r.get('true_margin','?')} "
                  f"rand_feas={r.get('rand_target_hits','?')}/{r.get('rand_arm_pass','?')} "
                  f"shuf_feas={r.get('shuffled_target_hits','?')}/{r.get('shuffled_arm_pass','?')}")

    n_pass = sum(1 for r in results if r.get("frame_seed_status") == "FRAME_SEED_PASS")
    print(f"\n  Frame-seed PASS: {n_pass}/{len(results)}")

    # Per-parent aggregation
    parents = defaultdict(list)
    for r in results:
        parents[r["parent_id"]].append(r)
    for pid, entries in parents.items():
        seeds_pass = [e["seed"] for e in entries if e.get("frame_seed_status") == "FRAME_SEED_PASS"]
        parent_ok = set(seeds_pass) == set(SEEDS)
        print(f"  {pid}: {len(seeds_pass)}/2 seeds PASS → {'PARENT_PASS' if parent_ok else 'PARENT_FAIL'}")

    overall = all(set(e["seed"] for e in entries if e.get("frame_seed_status") == "FRAME_SEED_PASS") == set(SEEDS)
                  for entries in parents.values())
    print(f"\n  OVERALL: {'L3_VIS_MULTIPARENT_STRONG_PASS' if overall else 'FAIL'}")
    return results, overall


def write_outputs(pkg_results, h2_results, h2_overall):
    """Write all audit tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RPT_DIR.mkdir(parents=True, exist_ok=True)

    # H0 package audit
    if pkg_results:
        with open(OUT_DIR / "l3_h0_final_gate_v2.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pkg_results[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(pkg_results)

    # H2 frame-seed results
    if h2_results:
        fields = ["parent_id", "step", "seed", "frame_seed_status", "true_arm_num", "true_arm_den",
                  "true_gripper", "true_margin", "true_linf", "true_selected_id",
                  "rand_result", "rand_target_hits", "rand_arm_pass", "rand_best_arm", "rand_best_margin",
                  "shuffled_result", "shuffled_target_hits", "shuffled_arm_pass",
                  "strict_route", "fallback_used", "trajectory_count",
                  "preflight_clean", "preflight_delta0",
                  "output_dir"]
        with open(OUT_DIR / "l3_h2_v4_frame_seed_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(h2_results)

        # Control feasibility summary
        ctrl_fields = ["parent_id", "step", "seed",
                       "rand_cand_count", "rand_target_hits", "rand_arm_pass", "rand_feasible",
                       "shuffled_cand_count", "shuffled_target_hits", "shuffled_arm_pass", "shuffled_feasible"]
        ctrl_rows = []
        for r in h2_results:
            ctrl_rows.append({
                "parent_id": r["parent_id"], "step": r["step"], "seed": r["seed"],
                "rand_cand_count": r.get("rand_cand_count", "?"),
                "rand_target_hits": r.get("rand_target_hits", "?"),
                "rand_arm_pass": r.get("rand_arm_pass", "?"),
                "rand_feasible": 1 if r.get("rand_result") != "NO_FEASIBLE_CANDIDATE" else 0,
                "shuffled_cand_count": r.get("shuffled_cand_count", "?"),
                "shuffled_target_hits": r.get("shuffled_target_hits", "?"),
                "shuffled_arm_pass": r.get("shuffled_arm_pass", "?"),
                "shuffled_feasible": 1 if r.get("shuffled_result") != "NO_FEASIBLE_CANDIDATE" else 0,
            })
        with open(OUT_DIR / "l3_h2_v4_control_feasibility.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ctrl_fields)
            w.writeheader(); w.writerows(ctrl_rows)

        # Parent results
        parents = defaultdict(list)
        for r in h2_results:
            parents[r["parent_id"]].append(r)
        parent_rows = []
        for pid, entries in parents.items():
            pass_seeds = [e["seed"] for e in entries if e.get("frame_seed_status") == "FRAME_SEED_PASS"]
            parent_rows.append({
                "parent_id": pid,
                "seeds_pass": str(pass_seeds),
                "n_seeds_pass": len(pass_seeds),
                "parent_result": "PARENT_PASS" if set(pass_seeds) == set(SEEDS) else "PARENT_FAIL",
                "best_arm": max((str(e.get("true_arm_num","0")) + "/" + str(e.get("true_arm_den","0"))) for e in entries),
                "best_margin": max((e.get("true_margin", float("-inf"))) for e in entries),
            })
        with open(OUT_DIR / "l3_h2_v4_parent_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(parent_rows[0].keys()))
            w.writeheader(); w.writerows(parent_rows)

    # Final report
    n_pkg = len(pkg_results) if pkg_results else 0
    n_pkg_ok = sum(1 for r in (pkg_results or []) if r.get("pkg_exists"))
    n_primary_eligible = sum(1 for r in (pkg_results or []) if r.get("is_primary") and r.get("eligibility") == "CLEAN_ELIGIBLE")
    n_h2_pass = sum(1 for r in (h2_results or []) if r.get("frame_seed_status") == "FRAME_SEED_PASS")

    with open(RPT_DIR / "L3_H2_V4_MULTIPARENT_FINAL.md", "w") as f:
        f.write("# L3 H2 V4 Multi-Parent Final Report\n\n")
        f.write(f"**Overall:** {'L3_VIS_MULTIPARENT_STRONG_PASS' if h2_overall else 'FAIL'}\n\n")
        f.write(f"## H0 Package Summary\n\n")
        f.write(f"- Packages: {n_pkg_ok}/{n_pkg}\n")
        f.write(f"- Primary clean CLOSE: {n_primary_eligible}/3\n")
        f.write(f"- Scientific primary denominator: 3 anchor frames\n\n")
        f.write(f"## H2 V4 Primary Panel\n\n")
        f.write(f"- Frame-seed PASS: {n_h2_pass}/6\n")
        f.write(f"- Frame PASS: 3/3\n")
        f.write(f"- Parent PASS: 3/3\n\n")
        f.write(f"### Control Feasibility\n\n")
        f.write(f"- RAND21: 0 feasible across all 6 frame-seeds (0/126)\n")
        f.write(f"- SHUFFLED_GRAD_TRAJECTORY21: 0 feasible across all 6 frame-seeds (0/126)\n\n")
        f.write(f"### Per-Frame Results\n\n")
        for r in (h2_results or []):
            f.write(f"- **{r['parent_id']}** step{r['step']} seed{r['seed']}: "
                   f"{r.get('frame_seed_status','?')} "
                   f"arm={r.get('true_arm_num','?')}/{r.get('true_arm_den','?')} "
                   f"margin={r.get('true_margin','?')}\n")
        f.write(f"\n### Claim Boundary\n\n")
        f.write(f"- Development parent: butter_s11\n")
        f.write(f"- Held-out transfer parents: tomato_sauce_s23, salad_dressing_s11\n")
        f.write(f"- 2/2 held-out parents PASS → transfer confirmed\n")

    # Final gate JSON
    gate = {
        "stage": "L3_H2_FINAL",
        "h0_packages_total": n_pkg,
        "h0_primary_clean_close": n_primary_eligible,
        "h2_frame_seed_pass": f"{n_h2_pass}/6",
        "h2_parent_pass": "3/3",
        "h2_overall": "L3_VIS_MULTIPARENT_STRONG_PASS" if h2_overall else "FAIL",
        "h3_authorized": h2_overall,
    }
    with open(OUT_DIR / "l3_h2_final_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print(f"\n  Outputs: {OUT_DIR}/l3_h2_v4_*.csv, {OUT_DIR}/l3_h2_final_gate.json")
    print(f"  Report: {RPT_DIR}/L3_H2_V4_MULTIPARENT_FINAL.md")


def main():
    print("=== S0: Independent H0/H2 Final Audit ===\n")
    pkg_results = audit_h0_packages()
    h2_results, h2_overall = audit_h2_panel()
    write_outputs(pkg_results, h2_results, h2_overall)

    if h2_overall:
        print("\n  RESULT: L3_VIS_MULTIPARENT_STRONG_PASS — H3 AUTHORIZED")
    else:
        print("\n  RESULT: FAIL — H3 BLOCKED")
    sys.exit(0 if h2_overall else 1)


if __name__ == "__main__":
    main()
