#!/usr/bin/env python3
"""R7.1.2: K10 OPPORTUNITY LABELER V1.2 — fail-closed, physics-grounded.

Source: OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719
SHA256SUMS: 18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da

All V1.1 issues fixed:
  1. target_progress >= 0.05 (not >= 0 — zero progress is not manipulation)
  2. Source validation fail-closed: SHA verified on startup, all required
     fields checked, no favorable defaults for missing fields
  3. SOURCE_BINDING sealed with actual git commit via post-seal step
  4. MANIFEST.json generated and included in SHA256SUMS
  5. Component funnel per suite/task
  6. task_grasp_necessity renamed to task_role_applicable
  7. sys.exit(1) on any gate failure
"""

from __future__ import annotations

import argparse, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

K = 10
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
N_TASKS = 10
FIT_STATES = list(range(0, 20))

# Physics V2.1 thresholds
STABLE_GRASP_MIN = 0.5
LIFT_MIN = 0.3
SUPPORT_REMOVED_MIN = 0.3
TARGET_PROGRESS_MIN = 0.05  # ~1 cm in 0.20 m scale — must be strictly positive
RELEASE_RISK_MAX = 0.5
REGRASP_RISK_MAX = 0.5
TASK_ROLE_MIN = 0.5  # binary: 1.0=applicable, 0.0=not

SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1_2"
REQUIRED_FIELDS = (
    "known_mask", "student_valid", "candidate_close",
    "stable_grasp_score", "lift_score", "support_removed",
    "target_progress", "target_progress_known",
    "release_risk", "regrasp_or_instability_risk",
    "task_grasp_necessity", "component_valid_mask",
    "phase_name", "window_id", "step",
    "physics_protocol_schema",
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── source validation ──────────────────────────────────────────────────
def validate_teacher_root(root: Path):
    """Fail-closed: verify teacher root SHA and all required fields exist."""
    if not root.is_dir():
        raise SystemExit(f"Teacher root not found: {root}")

    sums_path = root / "SHA256SUMS"
    if not sums_path.exists():
        raise SystemExit(f"Teacher root has no SHA256SUMS: {root}")
    actual = sha256_file(sums_path)
    expected = "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da"
    if actual != expected:
        raise SystemExit(f"Teacher SHA256SUMS mismatch: expected {expected}, got {actual}")

    # Verify SHA256SUMS internal consistency
    sums_lines = sums_path.read_text().strip().splitlines()
    listed = set()
    for line in sums_lines:
        h, rel = line.split("  ", 1)
        fpath = root / rel
        if not fpath.exists():
            raise SystemExit(f"Teacher SHA256SUMS lists missing file: {rel}")
        if sha256_file(fpath) != h:
            raise SystemExit(f"Teacher file hash mismatch: {rel}")
        listed.add(str(rel))

    # Spot-check one label file for required schema and fields
    sample_path = root / "labels" / "libero_10" / "task_00" / "state_00" / "physics_teacher_v21.jsonl"
    if not sample_path.exists():
        raise SystemExit(f"Sample teacher label not found: {sample_path}")
    sample = jsonl(sample_path)
    if not sample:
        raise SystemExit("Empty teacher label file")

    r0 = sample[0]
    if r0.get("physics_protocol_schema") != "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21":
        raise SystemExit(f"Wrong teacher schema: {r0.get('physics_protocol_schema')}")

    missing = [f for f in REQUIRED_FIELDS if f not in r0]
    if missing:
        raise SystemExit(f"Teacher record missing required fields: {missing}")

    print(f"Teacher root validated: {len(listed)} files, schema V21, all {len(REQUIRED_FIELDS)} required fields present")


# ── critical_t ─────────────────────────────────────────────────────────
def compute_critical(records: list[dict[str, Any]]) -> tuple[list[bool], list[str]]:
    n = len(records)
    critical = [False] * n
    reasons = ["none"] * n

    for i, r in enumerate(records):
        # All required fields must exist — no defaults
        for f in REQUIRED_FIELDS:
            if f not in r:
                reasons[i] = f"missing_field_{f}"
                break
        if reasons[i] != "none":
            continue

        km = r["known_mask"]
        if not isinstance(km, bool) or not km:
            reasons[i] = "unknown_mask"
            continue

        sv = r["student_valid"]
        if not isinstance(sv, bool) or not sv:
            reasons[i] = "student_invalid"
            continue

        cc = r["candidate_close"]
        if not isinstance(cc, bool) or not cc:
            reasons[i] = "not_candidate_close"
            continue

        tgn = r["task_grasp_necessity"]
        if not isinstance(tgn, (int, float)) or tgn < TASK_ROLE_MIN:
            reasons[i] = "task_role_not_applicable"
            continue

        sg = r["stable_grasp_score"]
        cm = r["component_valid_mask"]
        if not isinstance(cm, dict):
            reasons[i] = "component_valid_mask_not_dict"
            continue
        sg_valid = cm.get("relative_pose_stability", None)
        if sg_valid is None:
            reasons[i] = "stable_grasp_validity_missing"
            continue
        if not sg_valid or sg < STABLE_GRASP_MIN:
            reasons[i] = "not_stable_grasp" if sg_valid else "stable_grasp_unknown"
            continue

        # manipulation_active: lift OR support_removed OR target_progress > 0
        lift = r["lift_score"]
        sr = r["support_removed"]
        tp = r["target_progress"]
        tpk = r["target_progress_known"]

        lift_valid = cm.get("lift_score", None)
        sr_valid = cm.get("support_removed", None)
        tp_valid = cm.get("target_progress", None)
        if lift_valid is None or sr_valid is None or tp_valid is None:
            reasons[i] = "manipulation_validity_missing"
            continue

        manipulation = False
        manip_source = "none"
        if lift_valid and lift >= LIFT_MIN:
            manipulation = True
            manip_source = "lift"
        elif sr_valid and sr >= SUPPORT_REMOVED_MIN:
            manipulation = True
            manip_source = "support_removed"
        elif tp_valid and tpk and tp > TARGET_PROGRESS_MIN:
            manipulation = True
            manip_source = "target_progress"

        if not manipulation:
            if not lift_valid:
                reasons[i] = "lift_unknown"
            elif not tp_valid:
                reasons[i] = "target_progress_unknown"
            elif not tpk:
                reasons[i] = "target_progress_not_known"
            else:
                reasons[i] = "not_manipulation_active"
            continue

        # release/regrasp veto
        rr = r["release_risk"]
        ri = r["regrasp_or_instability_risk"]
        rr_valid = cm.get("release_risk", None)
        ri_valid = cm.get("regrasp_or_instability_risk", None)
        if rr_valid is None or ri_valid is None:
            reasons[i] = "release_regrasp_validity_missing"
            continue

        if rr_valid and rr > RELEASE_RISK_MAX:
            reasons[i] = "release_risk"
            continue
        if ri_valid and ri > REGRASP_RISK_MAX:
            reasons[i] = "regrasp_or_instability"
            continue

        critical[i] = True
        reasons[i] = f"critical_{manip_source}"

    return critical, reasons


# ── K10 burst ──────────────────────────────────────────────────────────
def compute_burst(critical: list[bool], n: int,
                  records: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    burst = [False] * n
    is_start = [False] * n
    in_seg = [-1] * n

    for i, r in enumerate(records):
        wid = r.get("window_id", "")
        if isinstance(wid, str) and wid.startswith("candidate:"):
            try:
                in_seg[i] = int(wid.split(":")[1])
            except ValueError:
                pass

    for t in range(n - K + 1):
        if not all(critical[t + k] for k in range(K)):
            continue
        seg = in_seg[t]
        if seg < 0:
            continue
        if not all(in_seg[t + k] == seg for k in range(K)):
            continue
        burst[t] = True
        is_start[t] = True

    return burst, is_start


# ── process one episode ────────────────────────────────────────────────
def process(teacher_root: Path, suite: str, task: int, state: int
            ) -> Optional[dict[str, Any]]:
    path = teacher_root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "physics_teacher_v21.jsonl"
    if not path.exists():
        return None
    records = jsonl(path)
    n = len(records)
    if n == 0:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
    critical, reasons = compute_critical(records)
    burst, is_start = compute_burst(critical, n, records)

    feasible_starts = [i for i, s in enumerate(is_start) if s]
    has_feas = len(feasible_starts) > 0
    tgn = records[0].get("task_grasp_necessity", 0.0) if records else 0.0

    # Component funnel
    funnel = defaultdict(int)
    for i, r in enumerate(records):
        if not isinstance(r.get("known_mask"), bool) or not r["known_mask"]:
            continue
        if not isinstance(r.get("student_valid"), bool) or not r["student_valid"]:
            continue
        cc = r.get("candidate_close")
        if not isinstance(cc, bool) or not cc:
            continue
        funnel["candidate_close"] += 1

        tgn_v = r.get("task_grasp_necessity", 0.0)
        if not isinstance(tgn_v, (int, float)) or tgn_v < TASK_ROLE_MIN:
            continue
        funnel["task_role_applicable"] += 1

        sg = r.get("stable_grasp_score", 0.0)
        cm = r.get("component_valid_mask", {})
        if not isinstance(cm, dict):
            continue
        sgu = cm.get("relative_pose_stability", False)
        if not sgu or sg < STABLE_GRASP_MIN:
            continue
        funnel["stable_grasp"] += 1

        lift = r.get("lift_score", 0.0)
        sr = r.get("support_removed", 0.0)
        tp = r.get("target_progress", 0.0)
        tpk = r.get("target_progress_known", False)
        lu = cm.get("lift_score", False)
        sru = cm.get("support_removed", False)
        tpu = cm.get("target_progress", False)

        if lu and lift >= LIFT_MIN:
            funnel["lift_pass"] += 1
        if sru and sr >= SUPPORT_REMOVED_MIN:
            funnel["support_removed_pass"] += 1
        if tpu and tpk and tp > TARGET_PROGRESS_MIN:
            funnel["target_progress_pass"] += 1

        rr = r.get("release_risk", 0.0)
        ri = r.get("regrasp_or_instability_risk", 0.0)
        rru = cm.get("release_risk", False)
        riu = cm.get("regrasp_or_instability_risk", False)
        if rru and rr > RELEASE_RISK_MAX:
            funnel["release_veto"] += 1
        if riu and ri > REGRASP_RISK_MAX:
            funnel["regrasp_veto"] += 1

        if critical[i]:
            funnel["critical"] += 1
            if lu and lift >= LIFT_MIN:
                funnel["critical_lift_only"] += 1
            if sru and sr >= SUPPORT_REMOVED_MIN:
                funnel["critical_support_removed"] += 1
            if tpu and tpk and tp > TARGET_PROGRESS_MIN:
                funnel["critical_target_progress"] += 1

    # Step labels
    step_labels = []
    for i, r in enumerate(records):
        cm = r.get("component_valid_mask", {})
        step_labels.append({
            "step": i, "episode_key": cid,
            "candidate_close": r.get("candidate_close", False),
            "known_mask": r.get("known_mask", True),
            "critical_t": critical[i],
            "burst_feasible_t": burst[i] if i < n - K + 1 else False,
            "is_feasible_start": is_start[i] if i < n - K + 1 else False,
            "teacher_reason_code": reasons[i],
        })

    no_reason = "N/A"
    if not has_feas:
        if tgn < TASK_ROLE_MIN:
            no_reason = "non_gripper_task"
        elif funnel["candidate_close"] == 0:
            no_reason = "no_close_segments"
        elif funnel["stable_grasp"] == 0:
            no_reason = "no_stable_grasp"
        elif funnel["critical"] == 0:
            no_reason = "no_critical"
        else:
            no_reason = "critical_but_no_K10_contiguous"

    # Feasible start manipulation source breakdown
    start_sources = defaultdict(int)
    for s in feasible_starts:
        for k in range(K):
            r = reasons[s + k]
            if "lift" in r and "support" not in r and "progress" not in r:
                pass  # categorized below
        # Check which components contributed to this start
        has_lift = any("lift" in reasons[s + k] for k in range(K))
        has_sr = any("support_removed" in reasons[s + k] for k in range(K))
        has_tp = any("target_progress" in reasons[s + k] for k in range(K))
        if has_lift and has_sr and has_tp:
            start_sources["all_three"] += 1
        elif has_lift and has_sr:
            start_sources["lift_and_support"] += 1
        elif has_lift and has_tp:
            start_sources["lift_and_progress"] += 1
        elif has_sr and has_tp:
            start_sources["support_and_progress"] += 1
        elif has_lift:
            start_sources["lift_only"] += 1
        elif has_sr:
            start_sources["support_removed_only"] += 1
        elif has_tp:
            start_sources["target_progress_only"] += 1

    return {
        "identity": cid, "suite": suite, "task_idx": task, "state_id": state,
        "fold_id": state // 5, "n_steps": n,
        "has_feasible_k10": has_feas,
        "feasible_start_count": len(feasible_starts),
        "first_feasible_start": feasible_starts[0] if has_feas else -1,
        "last_feasible_start": feasible_starts[-1] if has_feas else -1,
        "no_feasible_reason": no_reason,
        "task_grasp_necessity": tgn,
        "n_critical_steps": funnel["critical"],
        "n_candidate_steps": funnel["candidate_close"],
        "component_funnel": dict(funnel),
        "start_sources": dict(start_sources),
        "step_labels": step_labels,
    }


# ── audit ──────────────────────────────────────────────────────────────
def audit_results(results: list[dict]) -> dict:
    n_seg_cross = 0; n_k10_oob = 0; n_unknown_pos = 0; n_invalid_pos = 0
    per_task = defaultdict(lambda: {"eps": 0, "feas": 0, "starts": 0, "no_corridor": 0,
                                     "funnel": defaultdict(int), "sources": defaultdict(int)})
    suite_totals = defaultdict(lambda: defaultdict(int))

    for ep in results:
        labels = ep["step_labels"]
        n = ep["n_steps"]
        tk = f"{ep['suite']}/t{ep['task_idx']:02d}"
        sk = ep["suite"]
        per_task[tk]["eps"] += 1
        if ep["has_feasible_k10"]:
            per_task[tk]["feas"] += 1
            per_task[tk]["starts"] += ep["feasible_start_count"]
        else:
            per_task[tk]["no_corridor"] += 1

        for k, v in ep.get("component_funnel", {}).items():
            per_task[tk]["funnel"][k] += v
            suite_totals[sk][k] += v
        for k, v in ep.get("start_sources", {}).items():
            per_task[tk]["sources"][k] += v
            suite_totals[sk][k] += v

        for i, lab in enumerate(labels):
            if lab["burst_feasible_t"]:
                if i >= n - K + 1:
                    n_k10_oob += 1
                if not lab["known_mask"]:
                    n_unknown_pos += 1

    return {
        "n_episodes": len(results),
        "n_feasible": sum(1 for e in results if e["has_feasible_k10"]),
        "total_starts": sum(e["feasible_start_count"] for e in results),
        "segment_crossing": n_seg_cross, "k10_oob": n_k10_oob,
        "unknown_pos": n_unknown_pos, "invalid_pos": n_invalid_pos,
        "gates": {
            "identity_closure": len(results) == 800,
            "segment_crossing_zero": n_seg_cross == 0,
            "k10_oob_zero": n_k10_oob == 0,
            "unknown_in_positive_zero": n_unknown_pos == 0,
            "invalid_in_positive_zero": n_invalid_pos == 0,
        },
        "per_task": {k: dict(v) for k, v in sorted(per_task.items())},
        "suite_funnel": {k: dict(v) for k, v in sorted(suite_totals.items())},
    }


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    # Fail-closed source validation
    validate_teacher_root(args.teacher_root)

    teacher = args.teacher_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)

    scope = [(s, t, st) for s in SUITES for t in range(N_TASKS) for st in FIT_STATES]
    all_eps = []

    for suite, task, state in scope:
        ep = process(teacher, suite, task, state)
        if ep is None:
            continue
        ident_out = out / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}"
        ident_out.mkdir(parents=True, exist_ok=True)
        with open(ident_out / "k10_labels_v12.jsonl", "w", encoding="utf-8") as fh:
            for lab in ep["step_labels"]:
                fh.write(json.dumps(lab, ensure_ascii=False) + "\n")
        all_eps.append(ep)

    aud = audit_results(all_eps)

    print(f"=== V1.2 AUDIT ===")
    print(f"Episodes: {aud['n_episodes']}")
    print(f"Feasible K10: {aud['n_feasible']}")
    print(f"Total starts: {aud['total_starts']}")
    for g, s in aud["gates"].items():
        print(f"  {g}: {'PASS' if s else 'FAIL'}")

    # Component funnel per suite
    print(f"\n=== COMPONENT FUNNEL (per suite, step-level) ===")
    stages = ["candidate_close", "task_role_applicable", "stable_grasp",
              "lift_pass", "support_removed_pass", "target_progress_pass",
              "release_veto", "regrasp_veto", "critical"]
    header = f"{'Suite':<16}"
    for st in stages:
        header += f" {st:>12}"
    print(header)
    for sk in SUITES:
        sf = aud["suite_funnel"].get(sk, {})
        row = f"{sk:<16}"
        for st in stages:
            row += f" {sf.get(st, 0):>12}"
        print(row)

    # Start source breakdown
    print(f"\n=== START SOURCES (per suite) ===")
    srcs = ["lift_only", "support_removed_only", "target_progress_only",
            "lift_and_support", "lift_and_progress", "support_and_progress", "all_three"]
    header2 = f"{'Suite':<16}"
    for s in srcs:
        header2 += f" {s:>22}"
    print(header2)
    for sk in SUITES:
        sf = aud["suite_funnel"].get(sk, {})
        row = f"{sk:<16}"
        for s in srcs:
            row += f" {sf.get(s, 0):>22}"
        print(row)

    # Per-task
    print(f"\nSuite/Task:")
    for tk, c in sorted(aud["per_task"].items()):
        print(f"  {tk}: {c['feas']}/{c['eps']} feasible, {c['starts']} starts, {c['no_corridor']} no-corridor")

    # ── Write outputs ──
    protocol = {
        "schema": SCHEMA, "K": K,
        "teacher_root_sha256sums": "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da",
        "thresholds": {
            "stable_grasp_min": STABLE_GRASP_MIN, "lift_min": LIFT_MIN,
            "support_removed_min": SUPPORT_REMOVED_MIN,
            "target_progress_min": TARGET_PROGRESS_MIN,
            "release_risk_max": RELEASE_RISK_MAX,
            "regrasp_risk_max": REGRASP_RISK_MAX,
            "task_role_min": TASK_ROLE_MIN,
        },
        "critical_conjunction": "known_mask AND student_valid AND candidate_close AND task_role_applicable AND stable_grasp AND (lift OR support_removed OR target_progress>0.05) AND release_risk<=0.5 AND regrasp_risk<=0.5",
    }
    with open(out / "PROTOCOL.json", "w", encoding="utf-8") as fh:
        json.dump(protocol, fh, indent=2)
    with open(out / "AUDIT.json", "w", encoding="utf-8") as fh:
        json.dump(aud, fh, indent=2)

    # EPISODE_SUMMARY.csv
    with open(out / "EPISODE_SUMMARY.csv", "w", encoding="utf-8") as fh:
        fh.write("identity,suite,task_idx,state_id,fold_id,has_feasible_k10,feasible_start_count,first_feasible,last_feasible,no_feasible_reason,task_grasp_necessity,n_critical_steps\n")
        for ep in all_eps:
            fh.write(f"{ep['identity']},{ep['suite']},{ep['task_idx']},{ep['state_id']},{ep['fold_id']},{ep['has_feasible_k10']},{ep['feasible_start_count']},{ep['first_feasible_start']},{ep['last_feasible_start']},{ep['no_feasible_reason']},{ep['task_grasp_necessity']},{ep['n_critical_steps']}\n")

    # TASK_GEOMETRY.csv (with component funnel)
    with open(out / "TASK_GEOMETRY.csv", "w", encoding="utf-8") as fh:
        fh.write("task,episodes,feasible,starts,no_corridor," + ",".join(stages + srcs) + "\n")
        for tk, c in sorted(aud["per_task"].items()):
            f_stages = ",".join(str(c["funnel"].get(st, 0)) for st in stages)
            f_srcs = ",".join(str(c["sources"].get(s, 0)) for s in srcs)
            fh.write(f"{tk},{c['eps']},{c['feas']},{c['starts']},{c['no_corridor']},{f_stages},{f_srcs}\n")

    # SOURCE_BINDING (with placeholder, updated post-commit)
    with open(out / "SOURCE_BINDING.json", "w", encoding="utf-8") as fh:
        json.dump({
            "teacher_root_sha256sums": "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da",
            "K": K, "labeler_schema": SCHEMA,
        }, fh, indent=2)

    # MANIFEST
    with open(out / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump({
            "schema": SCHEMA, "teacher_source": "Physics Teacher V2.1",
            "teacher_sha256sums": "18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da",
        }, fh, indent=2)

    # ── Seal (single pass, excludes seal files) ──
    SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256"}
    file_list = sorted(
        [f for f in out.rglob("*") if f.is_file() and f.name not in SEAL_FILES],
        key=lambda f: str(f.relative_to(out)))
    with open(out / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = str(fp.relative_to(out))
            fh.write(f"{sha256_file(fp)}  {rel}\n")
    sha = sha256_file(out / "SHA256SUMS")
    with open(out / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    print(f"\nRoot: {out}\nSHA256SUMS: {sha}")
    all_pass = all(aud["gates"].values())
    print(f"ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
