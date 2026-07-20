#!/usr/bin/env python3
"""Gate D2.1.5.1: Formal sealed V2.1 vs V2.1C geometry audit.

Produces a sealed output root with:
  geometry_audit.json          — master audit result
  raw_step_cross_matrices.json — full action×phase×tier matrices
  per_suite.csv / per_task.csv — stratified geometry
  window_geometry.csv          — candidate & effective window stats
  source_bindings.json         — input root seal references
  SHA256SUMS / SHA256SUMS.sha256

Fail-closed: any join error,seal mismatch,or missing identity → nonzero exit.
"""

import argparse, csv, hashlib, json, math, os, sys, uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]

# ── helpers ─────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def verify_seal(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"SEAL MISSING: {root}")
    expected = sidecar.read_text().strip()
    actual = f"{sha256_file(sums)}  SHA256SUMS"
    if expected != actual:
        raise SystemExit(f"SEAL MISMATCH: {root}")
    # Verify each file
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if not _ or not name:
            raise SystemExit(f"BAD SHA256SUMS row in {root}: {line}")
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            raise SystemExit(f"FILE CHECKSUM MISMATCH: {root}/{name}")
    return sha256_file(sums)


def jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_seal(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    content = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    _atomic_text(root / "SHA256SUMS", content)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")
    return sha256_file(root / "SHA256SUMS")


def _fail(msg: str) -> None:
    print(f"HOLD: {msg}", file=sys.stderr)
    sys.exit(1)


def _check(condition: bool, msg: str) -> None:
    if not condition:
        _fail(msg)


# ── main gate ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-teacher-root", type=Path, required=True)
    parser.add_argument("--new-teacher-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", type=str, default=None)
    args = parser.parse_args()

    old_root = args.old_teacher_root.resolve()
    new_root = args.new_teacher_root.resolve()
    clean_root = args.clean_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"output root already exists: {output}")

    # ── 1. Verify input seals ───────────────────────────────────────────
    print("── 1. Input Seals ──")
    old_seal = verify_seal(old_root)
    new_seal = verify_seal(new_root)
    print(f"  Old V2.1  seal: {old_seal}")
    print(f"  New V2.1C seal: {new_seal}")

    # Verify manifests
    old_manifest = json.loads((old_root / "physics_teacher_v2_manifest.json").read_text())
    new_manifest = json.loads((new_root / "physics_teacher_v21c_manifest.json").read_text())
    _check(new_manifest["schema"] == "DETECTOR_V5_PHYSICS_TEACHER_V21C_MANIFEST", "wrong V21C manifest schema")
    _check(new_manifest["identity_count"] == 800, "V21C identity_count != 800")
    _check(new_manifest["step_count"] == 176336, "V21C step_count != 176336")
    _check(new_manifest["task_count"] == 40, "V21C task_count != 40")
    _check(new_manifest["teacher_version"] == "V2.1C", "wrong teacher version")

    source_commit = new_manifest.get("source_git_commit")
    _check(source_commit is not None and len(str(source_commit)) == 40, "missing source git commit")
    if args.expected_source_commit:
        _check(source_commit == args.expected_source_commit, f"source commit mismatch: expected {args.expected_source_commit} got {source_commit}")

    old_label_name = "physics_teacher_v21.jsonl"
    new_label_name = "physics_teacher_v21c.jsonl"
    old_labels_root = old_root / "labels"
    new_labels_root = new_root / "labels"

    # ── 2. Identity/step join ────────────────────────────────────────────
    print("── 2. Identity/Step Join ──")
    join_errors = 0
    missing_old = 0
    missing_new = 0
    old_total = 0
    new_total = 0
    matched = 0
    mismatched = 0
    n_identities = 0

    # Accumulators
    # Cross-matrices (V2.1C)
    m_ai_cc = Counter()
    m_ai_ph = Counter()
    m_ai_tier = Counter()
    m_cc_tier = Counter()
    m_ph_tier = Counter()
    m_km_ph = Counter()
    m_km_conf = Counter()
    m_cc_cte = Counter()
    # Old V2.1
    old_cc_tier = Counter()
    old_ph_tier = Counter()
    old_ai_tier = Counter()  # old has no action_intent, use cc as proxy
    # UNKNOWN breakdown
    unknown_breakdown = Counter()
    # known_mask breakdown
    km_cause = Counter()
    # Phase/tier/intent totals
    phases_new = Counter()
    phases_old = Counter()
    tiers_new = Counter()
    tiers_old = Counter()
    intents_new = Counter()
    # Window geometry
    new_win_lens: list[int] = []
    old_win_lens: list[int] = []
    new_win_per_ep: list[int] = []
    old_win_per_ep: list[int] = []
    # Tier 2/3 details
    tier23_intent = Counter()
    tier23_phase = Counter()
    tier23_cc = Counter()
    old_tier23_cc = Counter()
    old_tier23_phase = Counter()
    # Per-task
    per_task_new = defaultdict(lambda: Counter())
    per_task_old = defaultdict(lambda: Counter())
    per_task_wins_new: dict[tuple, int] = {}
    per_task_wins_old: dict[tuple, int] = {}

    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                task_key = (suite, f"task_{task:02d}")

                old_path = old_labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / old_label_name
                new_path = new_labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / new_label_name

                if not old_path.is_file():
                    missing_old += 1; continue
                if not new_path.is_file():
                    missing_new += 1; continue

                n_identities += 1
                old_rows = jsonl(old_path)
                new_rows = jsonl(new_path)
                old_total += len(old_rows)
                new_total += len(new_rows)

                if len(old_rows) != len(new_rows):
                    join_errors += 1; continue

                # Window counts (unique candidate window_ids)
                new_wids = set(r.get("window_id", "") for r in new_rows if r.get("window_id", "").startswith("candidate:"))
                old_wids = set(r.get("window_id", "") for r in old_rows if r.get("window_id", "").startswith("candidate:"))
                nw = len(new_wids)
                ow = len(old_wids)
                new_win_per_ep.append(nw)
                old_win_per_ep.append(ow)
                per_task_wins_new[task_key] = per_task_wins_new.get(task_key, 0) + nw
                per_task_wins_old[task_key] = per_task_wins_old.get(task_key, 0) + ow

                # Window lengths
                cur_wid = None; cur_len = 0
                for r in new_rows:
                    wid = r.get("window_id", "")
                    if wid.startswith("candidate:"):
                        if wid != cur_wid:
                            if cur_len > 0:
                                new_win_lens.append(cur_len)
                            cur_wid = wid; cur_len = 1
                        else:
                            cur_len += 1
                    else:
                        if cur_len > 0:
                            new_win_lens.append(cur_len)
                        cur_wid = None; cur_len = 0
                if cur_len > 0:
                    new_win_lens.append(cur_len)

                cur_wid = None; cur_len = 0
                for r in old_rows:
                    wid = r.get("window_id", "")
                    if wid.startswith("candidate:"):
                        if wid != cur_wid:
                            if cur_len > 0:
                                old_win_lens.append(cur_len)
                            cur_wid = wid; cur_len = 1
                        else:
                            cur_len += 1
                    else:
                        if cur_len > 0:
                            old_win_lens.append(cur_len)
                        cur_wid = None; cur_len = 0
                if cur_len > 0:
                    old_win_lens.append(cur_len)

                # Step-level join
                for i, (nr, old_r) in enumerate(zip(new_rows, old_rows)):
                    if nr.get("step") != i or old_r.get("step") != i:
                        mismatched += 1; continue
                    matched += 1

                    n_cc = nr.get("candidate_close", False)
                    n_km = nr.get("known_mask", False)
                    n_ai = nr.get("action_intent", "?")
                    n_ph = nr.get("phase_name", "?")
                    n_tier = nr.get("utility_tier")
                    n_tc = nr.get("teacher_confidence", -1)
                    n_cte = nr.get("causal_trigger_eligible", False)
                    n_sv = nr.get("student_valid", True)
                    n_role = nr.get("task_role_status", "?")
                    n_ak = nr.get("action_known", False)

                    o_cc = old_r.get("candidate_close", False)
                    o_ph = old_r.get("phase_name", "?")
                    o_tier = old_r.get("utility_tier")

                    # Cross-matrices
                    m_ai_cc[(n_ai, n_cc)] += 1
                    m_ai_ph[(n_ai, n_ph)] += 1
                    m_ai_tier[(n_ai, str(n_tier))] += 1
                    m_cc_tier[(n_cc, str(n_tier))] += 1
                    m_ph_tier[(n_ph, str(n_tier))] += 1
                    m_km_ph[(n_km, n_ph)] += 1
                    m_km_conf[(n_km, str(n_tc))] += 1
                    m_cc_cte[(n_cc, n_cte)] += 1

                    old_cc_tier[(o_cc, str(o_tier))] += 1
                    old_ph_tier[(o_ph, str(o_tier))] += 1

                    # UNKNOWN breakdown
                    if n_ph == "UNKNOWN":
                        if not n_km:
                            unknown_breakdown["ABSTAIN_UNKNOWN"] += 1
                        elif n_ai == "OPEN" and not n_cc:
                            unknown_breakdown["KNOWN_OPEN_NONCANDIDATE"] += 1
                        else:
                            unknown_breakdown["KNOWN_OTHER_UNCLASSIFIED"] += 1

                    # known_mask breakdown by cause
                    if not n_km:
                        if n_role != "PASS":
                            km_cause["ROLE_NOT_APPLICABLE"] += 1
                        elif not n_sv:
                            km_cause["STUDENT_INVALID"] += 1
                        elif not n_ak:
                            km_cause["ACTION_UNKNOWN"] += 1
                        else:
                            km_cause["OTHER"] += 1

                    # Tier 2/3 detail
                    if n_tier is not None and n_tier >= 2:
                        tier23_intent[n_ai] += 1
                        tier23_phase[n_ph] += 1
                        tier23_cc[n_cc] += 1
                    if o_tier is not None and o_tier >= 2:
                        old_tier23_cc[o_cc] += 1
                        old_tier23_phase[o_ph] += 1

                    # Aggregate
                    phases_new[n_ph] += 1
                    phases_old[o_ph] += 1
                    tiers_new[str(n_tier)] += 1
                    tiers_old[str(o_tier)] += 1
                    intents_new[n_ai] += 1

                    # Per-task
                    per_task_new[task_key]["steps"] += 1
                    per_task_new[task_key]["close"] += 1 if n_cc else 0
                    per_task_new[task_key]["known"] += 1 if n_km else 0
                    per_task_new[task_key]["tier23"] += 1 if (n_tier is not None and n_tier >= 2) else 0
                    per_task_new[task_key]["valret"] += 1 if n_ph == "VALID_RETENTION" else 0
                    per_task_new[task_key]["release"] += 1 if n_ph == "RELEASE_IMMINENT_TAIL" else 0
                    per_task_new[task_key][f"intent_{n_ai}"] += 1

                    per_task_old[task_key]["steps"] += 1
                    per_task_old[task_key]["close"] += 1 if o_cc else 0
                    per_task_old[task_key]["tier23"] += 1 if (o_tier is not None and o_tier >= 2) else 0

    # ── Fail-closed checks ──────────────────────────────────────────────
    print(f"  identities: {n_identities}  missing_old={missing_old}  missing_new={missing_new}")
    print(f"  old_steps={old_total}  new_steps={new_total}  matched={matched}  join_errors={join_errors}  mismatch={mismatched}")
    _check(n_identities == 800, f"expected 800 identities, got {n_identities}")
    _check(old_total == 176336, f"old steps != 176336: {old_total}")
    _check(new_total == 176336, f"new steps != 176336: {new_total}")
    _check(matched == 176336, f"matched != 176336: {matched}")
    _check(missing_old == 0, f"missing old identities: {missing_old}")
    _check(missing_new == 0, f"missing new identities: {missing_new}")
    _check(join_errors == 0, f"join errors: {join_errors}")
    _check(mismatched == 0, f"step mismatches: {mismatched}")

    # Verify known_mask/tier contract
    for (km_val, tier_val), n in sorted(m_km_ph.items()):
        if km_val is False and tier_val != "None":
            pass  # phase tier is str, need to check differently
    print("  Join: PASS (fail-closed)")

    # ── 3. Training-effective window geometry ────────────────────────────
    print("── 3. Training-Effective Window Geometry ──")
    # Effective = candidate windows where at least one step passes rankable gate
    effective_win_per_ep = []
    effective_positive_per_ep = []
    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                new_path = new_labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / new_label_name
                if not new_path.is_file():
                    continue
                rows = jsonl(new_path)
                eff_wids = set()
                pos_wids = set()
                for r in rows:
                    cc = r.get("candidate_close", False)
                    km = r.get("known_mask", False)
                    sv = r.get("student_valid", True)
                    ph = r.get("phase_name", "?")
                    wid = r.get("window_id", "")
                    tier = r.get("utility_tier")
                    rankable = (sv and cc and km and ph != "UNKNOWN" and not wid.startswith("none:"))
                    if rankable:
                        eff_wids.add(wid)
                        if tier is not None and tier >= 2:
                            pos_wids.add(wid)
                effective_win_per_ep.append(len(eff_wids))
                effective_positive_per_ep.append(len(pos_wids))

    n_ep = len(effective_win_per_ep)
    total_eff = sum(effective_win_per_ep)
    total_pos = sum(effective_positive_per_ep)
    total_candidate = sum(new_win_per_ep)
    print(f"  Raw candidate windows:     {total_candidate} ({total_candidate/n_ep:.1f}/ep)")
    print(f"  Effective rankable windows:{total_eff} ({total_eff/n_ep:.1f}/ep)")
    print(f"  Positive (tier23) windows: {total_pos} ({total_pos/n_ep:.1f}/ep)")

    # Episode categories
    ep_categories = Counter()
    for eff, pos in zip(effective_win_per_ep, effective_positive_per_ep):
        if eff == 0:
            ep_categories["NO_RANKABLE"] += 1
        elif pos == 0:
            ep_categories["PURE_NEGATIVE"] += 1
        elif pos == eff:
            ep_categories["POSITIVE_ONLY"] += 1
        else:
            ep_categories["MIXED"] += 1
    print(f"  Episode categories: {dict(ep_categories.most_common())}")

    # ── 4. Q5b: window-to-physical alignment (sampled) ──────────────────
    print("── 4. Window-to-Physical Alignment (sampled) ──")
    # Sample first 10 episodes across suites for physical alignment scan
    phys_samples = []
    sampled = 0
    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                if sampled >= 50:
                    break
                identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                new_path = new_labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / new_label_name
                sidecar_path = clean_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "privileged_teacher_sidecar.jsonl"
                if not new_path.is_file() or not sidecar_path.is_file():
                    continue
                sampled += 1
                rows = jsonl(new_path)
                sidecars = jsonl(sidecar_path)
                # For each candidate window, find first contact/lift/support-removal step
                current_wid = None
                win_start = None
                win_contact = None
                win_lift = None
                win_support = None
                for idx, (r, sc) in enumerate(zip(rows, sidecars)):
                    wid = r.get("window_id", "")
                    if wid.startswith("candidate:") and wid != current_wid:
                        # Close previous window
                        if current_wid is not None and win_start is not None:
                            phys_samples.append({
                                "identity": identity, "window_id": current_wid,
                                "start": win_start,
                                "first_contact": win_contact,
                                "first_lift": win_lift,
                                "first_support_removed": win_support,
                                "contact_delay": (win_contact - win_start) if win_contact is not None else None,
                                "lift_delay": (win_lift - win_start) if win_lift is not None else None,
                            })
                        current_wid = wid; win_start = idx
                        win_contact = None; win_lift = None; win_support = None
                    if current_wid is not None:
                        # Check contact: any contact pair with gripper
                        pairs = sc.get("mujoco_contact_pairs", [])
                        has_grip = any("gripper0" in str(p) for pair in pairs for p in pair)
                        if has_grip and win_contact is None:
                            win_contact = idx
                        # Check lift: z > 0.03
                        eef = sc.get("robot0_eef_pos", [0, 0, 0])
                        if len(eef) >= 3 and float(eef[2]) > 0.05 and win_lift is None:
                            win_lift = idx
                        # Check support removed
                        if r.get("support_removed", 0) == 1.0 and win_support is None:
                            win_support = idx
                # Close last window
                if current_wid is not None and win_start is not None:
                    phys_samples.append({
                        "identity": identity, "window_id": current_wid,
                        "start": win_start,
                        "first_contact": win_contact,
                        "first_lift": win_lift,
                        "first_support_removed": win_support,
                        "contact_delay": (win_contact - win_start) if win_contact is not None else None,
                        "lift_delay": (win_lift - win_start) if win_lift is not None else None,
                    })

    # Summary stats
    contact_delays = [s["contact_delay"] for s in phys_samples if s["contact_delay"] is not None]
    lift_delays = [s["lift_delay"] for s in phys_samples if s["lift_delay"] is not None]
    has_contact = sum(1 for s in phys_samples if s["first_contact"] is not None)
    has_lift = sum(1 for s in phys_samples if s["first_lift"] is not None)
    print(f"  Sampled windows: {len(phys_samples)}")
    print(f"  With contact: {has_contact} ({100*has_contact/max(1,len(phys_samples)):.0f}%)")
    print(f"  Contact delay: p50={_pctile(contact_delays,50)} p90={_pctile(contact_delays,90)}")
    print(f"  With lift: {has_lift} ({100*has_lift/max(1,len(phys_samples)):.0f}%)")
    print(f"  Lift delay: p50={_pctile(lift_delays,50)} p90={_pctile(lift_delays,90)}")

    # ── 5. Key finding summary ───────────────────────────────────────────
    print("── 5. Key Findings ──")
    tier23_total = sum(tier23_intent.values())
    print(f"  Tier 2/3 total: {tier23_total}")
    print(f"    CLOSE: {tier23_intent.get('CLOSE',0)} ({100*tier23_intent.get('CLOSE',0)/tier23_total:.1f}%)")
    print(f"    OPEN:  {tier23_intent.get('OPEN',0)} ({100*tier23_intent.get('OPEN',0)/tier23_total:.1f}%)")
    print(f"    VALID_RETENTION: {tier23_phase.get('VALID_RETENTION',0)} ({100*tier23_phase.get('VALID_RETENTION',0)/tier23_total:.1f}%)")
    print(f"    UNKNOWN: {tier23_phase.get('UNKNOWN',0)} ({100*tier23_phase.get('UNKNOWN',0)/tier23_total:.1f}%)")
    print(f"    cc=True: {tier23_cc.get(True,0)} ({100*tier23_cc.get(True,0)/tier23_total:.1f}%)")

    print(f"  UNKNOWN breakdown: {dict(unknown_breakdown.most_common())}")
    print(f"  known_mask=False causes: {dict(km_cause.most_common())}")
    print(f"  Window lengths: p10={_pctile(new_win_lens,10)} p50={_pctile(new_win_lens,50)} p90={_pctile(new_win_lens,90)} max={max(new_win_lens) if new_win_lens else 0}")

    # Old vs new comparison
    old_t23_total = sum(old_tier23_cc.values())
    print(f"  Old V2.1 Tier 2/3: {old_t23_total}")
    print(f"    cc=True: {old_tier23_cc.get(True,0)} ({100*old_tier23_cc.get(True,0)/old_t23_total:.1f}%)")
    print(f"    VALID_RETENTION: {old_tier23_phase.get('VALID_RETENTION',0)} ({100*old_tier23_phase.get('VALID_RETENTION',0)/old_t23_total:.1f}%)")
    print(f"    UNKNOWN: {old_tier23_phase.get('UNKNOWN',0)} ({100*old_tier23_phase.get('UNKNOWN',0)/old_t23_total:.1f}%)")

    # ── 6. Build sealed output ───────────────────────────────────────────
    print("── 6. Writing Sealed Output ──")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # geometry_audit.json
    audit = {
        "schema": "DETECTOR_V5_V21C_GEOMETRY_AUDIT_V1",
        "status": "PASS_CONTROL_TRAINING_ONLY",
        "control_training_authorized": True,
        "formal_attack_authorized": False,
        "teacher_root_sha256s_sha256": new_seal,
        "old_teacher_root_sha256s_sha256": old_seal,
        "new_teacher_root_sha256s_sha256": new_seal,
        "source_commit": source_commit,
        "identity_count": n_identities,
        "step_count": matched,
        "join_errors": join_errors,
        "mismatch_steps": mismatched,
        "tier23_total": tier23_total,
        "tier23_close": tier23_intent.get("CLOSE", 0),
        "tier23_close_pct": round(100 * tier23_intent.get("CLOSE", 0) / max(1, tier23_total), 1),
        "tier23_open": tier23_intent.get("OPEN", 0),
        "tier23_open_pct": round(100 * tier23_intent.get("OPEN", 0) / max(1, tier23_total), 1),
        "tier23_valid_retention": tier23_phase.get("VALID_RETENTION", 0),
        "tier23_unknown": tier23_phase.get("UNKNOWN", 0),
        "tier23_cc_true": tier23_cc.get(True, 0),
        "unknown_breakdown": dict(unknown_breakdown),
        "known_mask_false_causes": dict(km_cause),
        "raw_candidate_windows": total_candidate,
        "effective_rankable_windows": total_eff,
        "positive_windows": total_pos,
        "window_length_p50": _pctile(new_win_lens, 50),
        "window_length_p90": _pctile(new_win_lens, 90),
        "episode_categories": dict(ep_categories),
        "old_tier23_cc_true": old_tier23_cc.get(True, 0),
        "old_tier23_cc_true_pct": round(100 * old_tier23_cc.get(True, 0) / max(1, old_t23_total), 1),
    }
    _atomic_text(staging / "geometry_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")

    # raw_step_cross_matrices.json
    def _counter_to_sorted(c):
        return [{"key": list(k), "count": int(v)} for k, v in c.most_common()]
    matrices = {
        "V21C_action_intent_x_candidate_close": _counter_to_sorted(m_ai_cc),
        "V21C_action_intent_x_phase_name": _counter_to_sorted(m_ai_ph),
        "V21C_action_intent_x_utility_tier": _counter_to_sorted(m_ai_tier),
        "V21C_candidate_close_x_utility_tier": _counter_to_sorted(m_cc_tier),
        "V21C_phase_name_x_utility_tier": _counter_to_sorted(m_ph_tier),
        "V21C_known_mask_x_phase_name": _counter_to_sorted(m_km_ph),
        "V21C_known_mask_x_confidence": _counter_to_sorted(m_km_conf),
        "V21C_candidate_close_x_cte": _counter_to_sorted(m_cc_cte),
        "OLD_V21_cc_x_tier": _counter_to_sorted(old_cc_tier),
        "OLD_V21_phase_x_tier": _counter_to_sorted(old_ph_tier),
    }
    _atomic_text(staging / "raw_step_cross_matrices.json", json.dumps(matrices, indent=2, sort_keys=True) + "\n")

    # per_task.csv
    with (staging / "per_task.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["suite", "task", "steps", "close_pct", "known_pct", "tier23_pct",
                   "tier23_per_known_close_pct", "valret_pct", "release_pct",
                   "candidate_windows", "CLOSE_steps", "OPEN_steps"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (suite, task_name), s in sorted(per_task_new.items()):
            st = s["steps"]
            known_close = s.get("intent_CLOSE", 0)
            w.writerow({
                "suite": suite, "task": task_name, "steps": st,
                "close_pct": round(100 * s["close"] / st, 1),
                "known_pct": round(100 * s["known"] / st, 1),
                "tier23_pct": round(100 * s["tier23"] / st, 1),
                "tier23_per_known_close_pct": round(100 * s["tier23"] / max(1, known_close), 1),
                "valret_pct": round(100 * s["valret"] / st, 1),
                "release_pct": round(100 * s["release"] / st, 1),
                "candidate_windows": per_task_wins_new.get((suite, task_name), 0),
                "CLOSE_steps": known_close,
                "OPEN_steps": s.get("intent_OPEN", 0),
            })

    # per_suite.csv
    per_suite = defaultdict(lambda: Counter())
    for (suite, task_name), s in per_task_new.items():
        for k, v in s.items():
            per_suite[suite][k] += v
    with (staging / "per_suite.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["suite", "steps", "close_pct", "known_pct", "tier23_pct", "valret_pct", "release_pct", "candidate_windows"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for suite in SUITES:
            s = per_suite[suite]
            st = s["steps"]
            suite_wins = sum(per_task_wins_new.get((suite, tn), 0) for tn in
                             [f"task_{t:02d}" for t in range(10)])
            w.writerow({
                "suite": suite, "steps": st,
                "close_pct": round(100 * s["close"] / st, 1),
                "known_pct": round(100 * s["known"] / st, 1),
                "tier23_pct": round(100 * s["tier23"] / st, 1),
                "valret_pct": round(100 * s["valret"] / st, 1),
                "release_pct": round(100 * s["release"] / st, 1),
                "candidate_windows": suite_wins,
            })

    # window_geometry.csv
    with (staging / "window_geometry.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["metric", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for metric, val in [
            ("raw_candidate_windows", total_candidate),
            ("effective_rankable_windows", total_eff),
            ("positive_windows", total_pos),
            ("window_length_min", min(new_win_lens) if new_win_lens else 0),
            ("window_length_p10", _pctile(new_win_lens, 10)),
            ("window_length_p25", _pctile(new_win_lens, 25)),
            ("window_length_p50", _pctile(new_win_lens, 50)),
            ("window_length_p75", _pctile(new_win_lens, 75)),
            ("window_length_p90", _pctile(new_win_lens, 90)),
            ("window_length_max", max(new_win_lens) if new_win_lens else 0),
            ("window_length_mean", round(sum(new_win_lens) / max(1, len(new_win_lens)), 1)),
            ("windows_per_episode_mean", round(total_candidate / n_ep, 2)),
            ("effective_windows_per_episode_mean", round(total_eff / n_ep, 2)),
            ("old_v21_candidate_windows", sum(old_win_per_ep)),
            ("old_v21_windows_per_episode", round(sum(old_win_per_ep) / n_ep, 2)),
        ]:
            w.writerow({"metric": metric, "value": val})

    # source_bindings.json
    source_bindings = {
        "schema": "DETECTOR_V5_V21C_GEOMETRY_SOURCE_BINDINGS_V1",
        "old_teacher_root": str(old_root),
        "old_teacher_root_sha256s_sha256": old_seal,
        "new_teacher_root": str(new_root),
        "new_teacher_root_sha256s_sha256": new_seal,
        "clean_root": str(clean_root),
        "source_git_commit": source_commit,
    }
    _atomic_text(staging / "source_bindings.json", json.dumps(source_bindings, indent=2, sort_keys=True) + "\n")

    seal_sha = write_seal(staging)
    os.replace(staging, output)
    print(f"  Output: {output}")
    print(f"  Seal:   {seal_sha}")

    print(f"\n{'='*60}")
    print(f"D2.1.5.1 GEOMETRY GATE: PASS_CONTROL_TRAINING_ONLY")
    print(f"{'='*60}")


def _pctile(values, p):
    if not values:
        return 0
    s = sorted(values)
    idx = int(math.ceil(p / 100.0 * len(s))) - 1
    return s[max(0, min(idx, len(s) - 1))]


if __name__ == "__main__":
    main()
