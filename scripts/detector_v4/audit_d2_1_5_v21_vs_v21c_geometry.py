#!/usr/bin/env python3
"""Gate D2.1.5: V2.1 vs V2.1C label-geometry audit.

Answers the forensic question: after fixing the action inversion, does the
old V2.1 Teacher produce physically reasonable label geometry, or is it still
structurally broken by window/phase/tier formulation issues?

Key questions:
  Q1: How many Tier 2/3 steps are CLOSE vs OPEN vs phase=UNKNOWN?
  Q2: Full phase breakdown — where are the missing 110 steps?
  Q3: UNKNOWN breakdown: abstain vs known-OPEN-noncandidate
  Q4: Old V2.1 vs V2.1C comparison on all metrics
  Q5: Window geometry vs physical evidence alignment
  Q6: Per-suite/per-task collapse/hotspot detection
"""

import json, sys, csv
from collections import Counter, defaultdict
from pathlib import Path


OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
CLEAN = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean")
OLD_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719"
NEW_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21C_b7cc5b8_20260720"
OLD_LABELS = OLD_ROOT / "labels"
NEW_LABELS = NEW_ROOT / "labels"

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
NTASKS = 10
NSTATES = 20


def jsonl(path):
    if not path.is_file():
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def iter_all(labels_root):
    for suite in SUITES:
        for task in range(NTASKS):
            for state in range(NSTATES):
                identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                p = labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
                if not p.is_dir():
                    yield identity, None, None
                    continue
                label_file = p / "physics_teacher_v21c.jsonl" if "V21C" in str(labels_root) else p / "physics_teacher_v21.jsonl"
                rows = jsonl(label_file)
                yield identity, rows, label_file


def compute_stats(rows, label_name):
    """Compute per-episode and aggregate stats."""
    if rows is None:
        return {"status": "MISSING"}, None

    stats = Counter()
    details = []
    for r in rows:
        cc = r.get("candidate_close", False)
        km = r.get("known_mask", False)
        ai = r.get("action_intent", "?")
        ph = r.get("phase_name", "?")
        tier = r.get("utility_tier")
        tc = r.get("teacher_confidence", -1)
        cte = r.get("causal_trigger_eligible", False)
        stats["steps"] += 1
        stats[f"cc_{cc}"] += 1
        stats[f"km_{km}"] += 1
        stats[f"intent_{ai}"] += 1
        stats[f"phase_{ph}"] += 1
        stats[f"tier_{tier}"] += 1
        stats[f"conf_{tc}"] += 1
        stats[f"cte_{cte}"] += 1
        # Tier 2/3 disaggregation
        if tier is not None and tier >= 2:
            stats[f"tier23_intent_{ai}"] += 1
            stats[f"tier23_phase_{ph}"] += 1
            stats[f"tier23_cc_{cc}"] += 1
        details.append({
            "step": r.get("step", -1),
            "cc": cc, "km": km, "intent": ai, "phase": ph,
            "tier": tier, "conf": tc, "cte": cte,
        })
    return stats, details


def main():
    print("=" * 70)
    print("D2.1.5: V2.1 vs V2.1C Label Geometry Audit")
    print("=" * 70)

    # ── 1. Exact join ─────────────────────────────────────────────────────
    print("\n── 1. Identity/Step Join ──")
    join_errors = 0
    old_total = 0
    new_total = 0
    matched_steps = 0
    mismatch_steps = 0

    # ── 2. Aggregate cross-matrices ───────────────────────────────────────
    # V2.1C cross-matrices
    matrix_ai_cc = Counter()       # action_intent × candidate_close
    matrix_ai_ph = Counter()       # action_intent × phase_name
    matrix_ai_tier = Counter()     # action_intent × utility_tier
    matrix_cc_tier = Counter()     # candidate_close × utility_tier
    matrix_ph_tier = Counter()     # phase_name × utility_tier
    matrix_km_ph = Counter()       # known_mask × phase_name
    matrix_km_conf = Counter()     # known_mask × teacher_confidence
    matrix_cc_cte = Counter()      # candidate_close × causal_trigger_eligible

    # Old V2.1 matrices for comparison
    old_matrix_ai_cc = Counter()
    old_matrix_cc_tier = Counter()
    old_matrix_ai_tier = Counter()
    old_matrix_ph_tier = Counter()

    # ── 3. UNKNOWN breakdown ──────────────────────────────────────────────
    unknown_breakdown = Counter()  # ABSTAIN / KNOWN_OPEN_NONCANDIDATE / KNOWN_OTHER

    # ── 4. Window geometry ────────────────────────────────────────────────
    new_windows_per_ep = []
    old_windows_per_ep = []
    new_window_lengths = []
    old_window_lengths = []

    # ── 5. Per-suite/per-task ─────────────────────────────────────────────
    per_task_new = defaultdict(lambda: Counter())
    per_task_old = defaultdict(lambda: Counter())

    # ── 6. Phase full enumeration (find the 110 missing steps) ────────────
    all_phases_new = Counter()
    all_phases_old = Counter()
    all_tiers_new = Counter()
    all_tiers_old = Counter()
    all_intents_new = Counter()

    n_identities = 0
    n_missing_old = 0
    n_missing_new = 0

    for identity, new_rows, new_path in iter_all(NEW_LABELS):
        old_path = OLD_LABELS / identity.split("/")[0] / identity.split("/")[1] / identity.split("/")[2] / "physics_teacher_v21.jsonl"
        old_rows = jsonl(old_path)

        if new_rows is None:
            n_missing_new += 1
            continue
        if old_rows is None:
            n_missing_old += 1
            continue

        n_identities += 1
        new_total += len(new_rows)
        old_total += len(old_rows)

        if len(new_rows) != len(old_rows):
            join_errors += 1
            continue

        # Per-episode window IDs
        new_wids = set(r.get("window_id", "") for r in new_rows if r.get("candidate_close"))
        old_wids = set(r.get("window_id", "") for r in old_rows if r.get("candidate_close"))
        new_windows_per_ep.append(len(new_wids))
        old_windows_per_ep.append(len(old_wids))

        suite = identity.split("/")[0]
        task_key = (suite, identity.split("/")[1])

        for i, (nr, old_r) in enumerate(zip(new_rows, old_rows)):
            if nr.get("step") != i or old_r.get("step") != i:
                mismatch_steps += 1
                continue
            matched_steps += 1

            # ── New V2.1C fields ──
            n_cc = nr.get("candidate_close", False)
            n_km = nr.get("known_mask", False)
            n_ai = nr.get("action_intent", "?")
            n_ph = nr.get("phase_name", "?")
            n_tier = nr.get("utility_tier")
            n_tc = nr.get("teacher_confidence", -1)
            n_cte = nr.get("causal_trigger_eligible", False)

            # ── Old V2.1 fields ──
            o_cc = old_r.get("candidate_close", False)
            o_ph = old_r.get("phase_name", "?")
            o_tier = old_r.get("utility_tier")

            # ── Cross-matrices (V2.1C) ──
            matrix_ai_cc[(n_ai, n_cc)] += 1
            matrix_ai_ph[(n_ai, n_ph)] += 1
            matrix_ai_tier[(n_ai, str(n_tier))] += 1
            matrix_cc_tier[(n_cc, str(n_tier))] += 1
            matrix_ph_tier[(n_ph, str(n_tier))] += 1
            matrix_km_ph[(n_km, n_ph)] += 1
            matrix_km_conf[(n_km, str(n_tc))] += 1
            matrix_cc_cte[(n_cc, n_cte)] += 1

            # ── Cross-matrices (Old V2.1) ──
            old_matrix_ai_cc[(o_cc, "N/A")] += 1  # old has no action_intent
            old_matrix_cc_tier[(o_cc, str(o_tier))] += 1
            old_matrix_ph_tier[(o_ph, str(o_tier))] += 1

            # ── UNKNOWN breakdown (V2.1C) ──
            if n_ph == "UNKNOWN":
                if not n_km:
                    unknown_breakdown["ABSTAIN_UNKNOWN"] += 1
                elif n_ai == "OPEN" and not n_cc:
                    unknown_breakdown["KNOWN_OPEN_NONCANDIDATE"] += 1
                else:
                    unknown_breakdown["KNOWN_OTHER_UNCLASSIFIED"] += 1

            # ── Aggregate ──
            all_phases_new[n_ph] += 1
            all_phases_old[o_ph] += 1
            all_tiers_new[str(n_tier)] += 1
            all_tiers_old[str(o_tier)] += 1
            all_intents_new[n_ai] += 1

            # ── Per-task ──
            per_task_new[task_key]["steps"] += 1
            per_task_new[task_key][f"close"] += 1 if n_cc else 0
            per_task_new[task_key][f"known"] += 1 if n_km else 0
            per_task_new[task_key][f"tier23"] += 1 if (n_tier is not None and n_tier >= 2) else 0
            per_task_new[task_key][f"valret"] += 1 if n_ph == "VALID_RETENTION" else 0
            per_task_new[task_key][f"release"] += 1 if n_ph == "RELEASE_IMMINENT_TAIL" else 0
            per_task_new[task_key][f"phase_{n_ph}"] += 1
            per_task_new[task_key][f"tier_{n_tier}"] += 1

            per_task_old[task_key]["steps"] += 1
            per_task_old[task_key][f"close"] += 1 if o_cc else 0
            per_task_old[task_key][f"tier23"] += 1 if (o_tier is not None and o_tier >= 2) else 0
            per_task_old[task_key][f"valret"] += 1 if o_ph == "VALID_RETENTION" else 0

    # ── Report ─────────────────────────────────────────────────────────────

    print(f"  Identities: {n_identities}/800 (missing old={n_missing_old} new={n_missing_new})")
    print(f"  Old steps: {old_total}  New steps: {new_total}")
    print(f"  Matched: {matched_steps}  Join errors: {join_errors}  Mismatch: {mismatch_steps}")

    # Q2: Full phase enumeration
    print(f"\n── Q2: Full Phase Breakdown ──")
    print(f"  V2.1C phases:")
    total_ph = sum(all_phases_new.values())
    for ph, n in all_phases_new.most_common():
        print(f"    {ph}: {n} ({100*n/total_ph:.1f}%)")
    print(f"    TOTAL: {total_ph}")

    print(f"  V2.1 phases:")
    total_ph_old = sum(all_phases_old.values())
    for ph, n in all_phases_old.most_common():
        print(f"    {ph}: {n} ({100*n/total_ph_old:.1f}%)")
    print(f"    TOTAL: {total_ph_old}")

    # Q3: UNKNOWN breakdown
    print(f"\n── Q3: UNKNOWN Breakdown (V2.1C) ──")
    for k, n in unknown_breakdown.most_common():
        print(f"    {k}: {n}")

    # Q1: Tier 2/3 × action_intent × phase
    print(f"\n── Q1: Tier 2/3 × Action Intent × Phase (V2.1C) ──")
    # Direct computation for clarity
    tier23_by_intent = Counter()
    tier23_by_phase = Counter()
    tier23_by_cc = Counter()
    tier23_total = 0
    for (nt, np), n in sorted(zip(
        [v for v in matrix_ai_tier if v[1] in ("2", "3")],
        [v for v in matrix_ai_tier if v[1] in ("2", "3")]
    )):
        pass  # need to recompute more cleanly below

    # Recompute directly
    tier23_intent = Counter()
    tier23_phase = Counter()
    tier23_cc = Counter()
    tier23_steps = 0
    for identity, new_rows, _ in iter_all(NEW_LABELS):
        if new_rows is None:
            continue
        for nr in new_rows:
            tier = nr.get("utility_tier")
            if tier is not None and tier >= 2:
                tier23_steps += 1
                tier23_intent[nr.get("action_intent", "?")] += 1
                tier23_phase[nr.get("phase_name", "?")] += 1
                tier23_cc[nr.get("candidate_close", False)] += 1

    print(f"  Tier 2/3 total steps: {tier23_steps}")
    print(f"  By action_intent: {dict(tier23_intent.most_common())}")
    print(f"  By phase_name: {dict(tier23_phase.most_common())}")
    print(f"  By candidate_close: {dict(tier23_cc.most_common())}")

    # Q1 also for old V2.1
    old_tier23_intent = Counter()
    old_tier23_phase = Counter()
    old_tier23_cc = Counter()
    old_tier23_steps = 0
    for identity in [id for id, _, _ in iter_all(OLD_LABELS)]:
        parts = identity.split("/")
        old_path = OLD_LABELS / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        old_rows = jsonl(old_path)
        if old_rows is None:
            continue
        for old_r in old_rows:
            tier = old_r.get("utility_tier")
            if tier is not None and tier >= 2:
                old_tier23_steps += 1
                old_tier23_cc[old_r.get("candidate_close", False)] += 1
                old_tier23_phase[old_r.get("phase_name", "?")] += 1

    print(f"\n  Old V2.1 Tier 2/3 total: {old_tier23_steps}")
    print(f"  Old V2.1 Tier 2/3 by phase: {dict(old_tier23_phase.most_common())}")
    print(f"  Old V2.1 Tier 2/3 by cc: {dict(old_tier23_cc.most_common())}")

    # Q4: Key cross-matrices
    print(f"\n── Q4: V2.1C Key Cross-Matrices ──")
    print(f"  action_intent × candidate_close:")
    for (ai, cc), n in sorted(matrix_ai_cc.items()):
        print(f"    {ai} × cc={cc}: {n}")
    print(f"  phase_name × utility_tier (top rows):")
    for (ph, tier), n in sorted(matrix_ph_tier.items(), key=lambda x: -x[1])[:20]:
        print(f"    {ph} × tier={tier}: {n}")
    print(f"  known_mask × teacher_confidence:")
    for (km, tc), n in sorted(matrix_km_conf.items()):
        print(f"    km={km} × conf={tc}: {n}")
    print(f"  candidate_close × causal_trigger_eligible:")
    for (cc, cte), n in sorted(matrix_cc_cte.items()):
        print(f"    cc={cc} × cte={cte}: {n}")

    # Q5: Window geometry
    print(f"\n── Q5: Window Geometry ──")
    if new_windows_per_ep:
        nw = sorted(new_windows_per_ep)
        print(f"  V2.1C windows per episode: min={min(nw)} max={max(nw)} median={nw[len(nw)//2]} total={sum(nw)}")
        print(f"  V2.1C total windows: {sum(nw)}")
        print(f"  V2.1C windows/episode: {sum(nw)/len(nw):.2f}")
    if old_windows_per_ep:
        ow = sorted(old_windows_per_ep)
        print(f"  V2.1 windows per episode: min={min(ow)} max={max(ow)} median={ow[len(ow)//2]} total={sum(ow)}")
        print(f"  V2.1 total windows: {sum(ow)}")
        print(f"  V2.1 windows/episode: {sum(ow)/len(ow):.2f}")

    # Window length distribution (V2.1C)
    new_win_lens = Counter()
    for identity, new_rows, _ in iter_all(NEW_LABELS):
        if new_rows is None:
            continue
        current_wid = None
        current_len = 0
        for nr in new_rows:
            wid = nr.get("window_id", "")
            if wid.startswith("candidate:"):
                if wid != current_wid:
                    if current_len > 0:
                        new_win_lens[current_len] += 1
                    current_wid = wid
                    current_len = 1
                else:
                    current_len += 1
            else:
                if current_len > 0:
                    new_win_lens[current_len] += 1
                current_wid = None
                current_len = 0
        if current_len > 0:
            new_win_lens[current_len] += 1

    lens = sorted(new_win_lens.elements())
    if lens:
        n = len(lens)
        print(f"  V2.1C window lengths: min={min(lens)} p10={lens[n//10]} p50={lens[n//2]} p90={lens[9*n//10]} max={max(lens)} mean={sum(lens)/n:.1f}")

    # Q6: Per-task summary
    print(f"\n── Q6: Per-Task Summary (V2.1C) ──")
    print(f"  {'task':<35s} {'steps':>6s} {'close%':>7s} {'known%':>7s} {'tier23%':>8s} {'valret%':>8s} {'release%':>8s} {'wins':>5s}")
    for (suite, task_name), stats in sorted(per_task_new.items()):
        s = stats["steps"]
        if s == 0:
            continue
        # Count windows from V2.1C manifest
        nw = int(new_windows_per_ep[n_identities - 1]) if False else 0
        print(f"  {suite}/{task_name:<20s} {s:>6d} {100*stats['close']/s:>6.1f}% {100*stats['known']/s:>6.1f}% {100*stats['tier23']/s:>7.1f}% {100*stats['valret']/s:>7.1f}% {100*stats['release']/s:>7.1f}% {nw:>5d}")

    # ── Per-suite summary ──
    print(f"\n── Per-Suite Summary (V2.1C) ──")
    per_suite = defaultdict(lambda: Counter())
    for (suite, task_name), stats in per_task_new.items():
        for k, v in stats.items():
            per_suite[suite][k] += v
    # Count windows per suite
    suite_windows = defaultdict(int)
    for identity, new_rows, _ in iter_all(NEW_LABELS):
        if new_rows is None:
            continue
        suite = identity.split("/")[0]
        wids = set(r.get("window_id", "") for r in new_rows if r.get("candidate_close"))
        suite_windows[suite] += len(wids)

    for suite in SUITES:
        stats = per_suite[suite]
        s = stats["steps"]
        if s == 0:
            continue
        print(f"  {suite}: steps={s} close={100*stats['close']/s:.1f}% known={100*stats['known']/s:.1f}% tier23={100*stats['tier23']/s:.1f}% valret={100*stats['valret']/s:.1f}% release={100*stats['release']/s:.1f}% windows={suite_windows[suite]}")

    # ── Tier distribution ──
    print(f"\n── Tier Distribution ──")
    print(f"  V2.1C tiers: {dict(all_tiers_new.most_common())}")
    print(f"  V2.1  tiers: {dict(all_tiers_old.most_common())}")

    print(f"\n── Action Intent Distribution ──")
    print(f"  V2.1C intents: {dict(all_intents_new.most_common())}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"D2.1.5 AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
