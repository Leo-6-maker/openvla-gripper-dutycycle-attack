#!/usr/bin/env python3
"""Factorized V2 L3 analysis. FAIL-CLOSED. Awaiting Codex V3 adapter."""
from __future__ import annotations

import argparse, hashlib, json, math, os, statistics, sys, time, uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
from validate_factorized_codex_handoff import validate_handoff_static, validate_handoff_execution

EXPECTED_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))


# ── Episode validation ──

def validate_episode_step_sequence(rows, step_field):
    steps = sorted(r[step_field] for r in rows)
    if not steps: raise SystemExit("EMPTY_EPISODE")
    if steps[0] != 0: raise SystemExit(f"FIRST_STEP_NOT_ZERO: {steps[0]}")
    for i, s in enumerate(steps):
        if s != i: raise SystemExit(f"STEP_GAP: expected {i} got {s}")
    return len(steps)


# ── Classification + K10 (use step_field) ──

def classify_episode(offline_rows, step_field):
    T = len(offline_rows)
    if T < 10: return "unknown"
    last_eligible = T - 10
    eligible = offline_rows[:last_eligible + 1]
    known_all = all(r.get("strict_k10_known_mask", False) for r in eligible)
    has_pos = any(
        r.get("strict_k10_feasible", False) and r.get("strict_k10_known_mask", False)
        for r in eligible)
    if has_pos: return "positive"
    if known_all: return "negative"
    return "unknown"


def is_valid_start(row):
    return (row.get("strict_k10_feasible", False) and
            row.get("strict_k10_known_mask", False))


def compute_timing(offline_rows, emit_step, step_field):
    valid = sorted(r[step_field] for r in offline_rows if is_valid_start(r))
    if not valid: return None, None, None
    regions = []; rs = valid[0]; prev = valid[0]
    for t in valid[1:]:
        if t == prev + 1: prev = t
        else: regions.append((rs, prev)); rs = t; prev = t
    regions.append((rs, prev))
    for rs, re in regions:
        if rs <= emit_step <= re:
            return emit_step - rs, re - rs + 1, (emit_step - rs) / max(1, re - rs)
    return None, None, None


# ── Exact join ──

def exact_join(rt_file, ol_file, ep_field, step_field):
    rt_rows = []; rt_keys = set()
    with open(rt_file) as f:
        for line in f:
            r = json.loads(line)
            key = (r[ep_field], r[step_field])
            if key in rt_keys: raise SystemExit(f"DUP_RT: {key}")
            rt_keys.add(key); rt_rows.append(r)
    ol_rows = []; ol_keys = set()
    with open(ol_file) as f:
        for line in f:
            r = json.loads(line)
            key = (r[ep_field], r[step_field])
            if key in ol_keys: raise SystemExit(f"DUP_OL: {key}")
            ol_keys.add(key); ol_rows.append(r)
    if rt_keys != ol_keys:
        raise SystemExit(f"JOIN_MISMATCH: rt={len(rt_keys-ol_keys)} ol={len(ol_keys-rt_keys)}")
    return rt_rows, ol_rows


# ── L3 Metrics ──

def compute_l3_metrics(offline_episodes, scheduler_results, step_field):
    n_pos = 0; n_neg = 0; n_unk = 0
    neg_emits = 0; pos_on = 0; pos_off = 0; pos_abstain = 0; unk_emits = 0
    offsets = []; per_ep = []

    ep_keys = set(offline_episodes.keys())
    res_keys = set(scheduler_results.keys())
    if ep_keys != res_keys:
        raise SystemExit(f"EP_CLOSURE: eps={len(ep_keys-res_keys)} res={len(res_keys-ep_keys)}")

    for ep_key in sorted(offline_episodes.keys()):
        ol_rows = offline_episodes[ep_key]
        result = scheduler_results[ep_key]
        emitted = result.get("emitted", False)
        emit_step = result.get("emit_step", -1)
        ep_class = classify_episode(ol_rows, step_field)
        per_ep.append({"episode_key": ep_key, "classification": ep_class,
                       "scheduler_emitted": emitted, "emit_step": emit_step})

        if ep_class == "unknown":
            n_unk += 1
            if emitted: unk_emits += 1
            continue

        if ep_class == "negative":
            n_neg += 1
            if emitted: neg_emits += 1
        else:
            n_pos += 1
            if emitted:
                row = next((r for r in ol_rows if r[step_field] == emit_step), None)
                if row is None: raise SystemExit(f"EMIT_STEP_NOT_FOUND: {ep_key}/{emit_step}")
                if is_valid_start(row):
                    pos_on += 1
                    o, rl, rp = compute_timing(ol_rows, emit_step, step_field)
                    if o is not None: offsets.append(o)
                else:
                    pos_off += 1
            else:
                pos_abstain += 1

    total = neg_emits + pos_on + pos_off + unk_emits
    def sr(n,d): return n/d if d>0 else None

    return {
        "negative_episodes": n_neg, "positive_episodes": n_pos, "unknown_episodes": n_unk,
        "negative_episode_emits": neg_emits, "positive_on_corridor_emits": pos_on,
        "positive_off_corridor_emits": pos_off, "positive_abstentions": pos_abstain,
        "unknown_episode_emits": unk_emits, "total_emitted_episodes": total,
        "negative_episode_false_start_rate": sr(neg_emits, n_neg),
        "valid_opportunity_recall": sr(pos_on, n_pos),
        "all_emit_precision": sr(pos_on, total),
        "verified_emit_precision": sr(pos_on, neg_emits+pos_on+pos_off),
        "abstention_rate": sr(pos_abstain, n_pos),
        "unknown_emit_rate": sr(unk_emits, n_unk),
        "unverifiable_emit_fraction": sr(unk_emits, total),
        "median_timing_offset": float(statistics.median(offsets)) if offsets else None,
        "unknown_fraction": sr(n_unk, n_pos+n_neg+n_unk),
        "per_episode": per_ep,
    }


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-handoff", type=Path, required=True)
    ap.add_argument("--runtime-bundle-root", type=Path, required=True)
    ap.add_argument("--offline-eval-bundle-root", type=Path, required=True)
    ap.add_argument("--calibration-contract-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--blocker-receipt-root", type=Path, default=None)
    ap.add_argument("--mode", choices=["authoritative","diagnostic"], default="diagnostic")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # ── Strict handoff load ──
    from load_factorized_handoff import load_handoff_file
    handoff = load_handoff_file(args.codex_handoff.resolve(), ROOT)

    # ── Call formal validator ──
    if args.mode == "authoritative":
        ok, errors = validate_handoff_execution(handoff)
    else:
        ok, errors = validate_handoff_static(handoff)
    if not ok:
        for e in errors: print(f"  {e}")
        raise SystemExit(f"HANDOFF_VALIDATION_FAILED ({len(errors)} errors)")

    # ── Extract from V3.1 nested structure (no defaults) ──
    ra = handoff["runtime_adapter"]
    adapter_path = ra["source"]["path"]
    adapter_class = ra["class"]

    rb = handoff["runtime_bundle"]
    ep_field = rb["episode_field"]
    step_field = rb["step_field"]
    rt_fn = rb["data_filename"]

    ob = handoff["offline_bundles"]["evaluation"]
    ol_fn = ob["data_filename"]

    # ── Adapter import (ALL modes fail-closed) ──
    handoff_path = args.codex_handoff.resolve()
    handoff_sha = sha256_file(handoff_path)
    import importlib
    sys.path.insert(0, str(ROOT / "src"))
    # Derive module from verified source path: src/gripper_attack/factorized_scheduler_adapter.py → gripper_attack.factorized_scheduler_adapter
    adapter_rel = handoff["runtime_adapter"]["source"]["path"]
    if not adapter_rel.startswith("src/") or not adapter_rel.endswith(".py"):
        raise SystemExit(f"ADAPTER_PATH_UNEXPECTED: {adapter_rel}")
    adapter_module = adapter_rel[len("src/"):-len(".py")].replace("/", ".")
    try:
        mod = importlib.import_module(adapter_module)
        AdapterCls = getattr(mod, adapter_class)
    except Exception as e:
        if args.blocker_receipt_root:
            br = Path(args.blocker_receipt_root); br.mkdir(parents=True)
            (br / "BLOCKER_RECEIPT.json").write_text(json.dumps({
                "status": "SCHEDULER_ADAPTER_IMPORT_FAILED",
                "error": str(e), "authoritative_l3": False,
            }, indent=2) + "\n")
        raise SystemExit(f"SCHEDULER_ADAPTER_IMPORT_FAILED: {e}")

    # ── Verify bundles ──
    rt_root = args.runtime_bundle_root.resolve()
    off_root = args.offline_eval_bundle_root.resolve()
    rt_found = set(d.name for d in rt_root.iterdir() if d.is_dir())
    off_found = set(d.name for d in off_root.iterdir() if d.is_dir())
    if rt_found != off_found: raise SystemExit("RT_OL_SPLIT_MISMATCH")
    missing = EXPECTED_SPLITS - rt_found
    extra = rt_found - EXPECTED_SPLITS
    if missing: raise SystemExit(f"MISSING: {sorted(missing)}")
    if extra: raise SystemExit(f"EXTRA: {sorted(extra)}")

    # ── Load calibration contracts (V3 schema) ──
    cal_data = {}
    for split_key in sorted(EXPECTED_SPLITS):
        cf = Path(args.calibration_contract_root or "") / split_key / "calibration_and_threshold_contract.json"
        if not cf.is_file():
            if args.mode == "authoritative": raise SystemExit(f"CAL_CONTRACT_MISSING: {split_key}")
            continue
        c = json.loads(cf.read_text())
        if c.get("schema") != "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3":
            raise SystemExit(f"CAL_WRONG_SCHEMA: {split_key} expected V3")
        if args.mode == "authoritative":
            if not c.get("l3_evaluation_eligible", False):
                raise SystemExit(f"CAL_NOT_L3_ELIGIBLE: {split_key}")
        cal_data[split_key] = c

    # ── Load structural config ──
    structure = json.loads((ROOT / handoff["structural_config"]["path"]).read_text())

    # ── Per-split replay via Codex adapter ──
    all_metrics = {}
    for split_key in sorted(EXPECTED_SPLITS):
        rt_file = rt_root / split_key / rt_fn
        ol_file = off_root / split_key / ol_fn
        rt_rows, ol_rows = exact_join(rt_file, ol_file, ep_field, step_field)

        rt_eps = defaultdict(list)
        for r in rt_rows: rt_eps[r[ep_field]].append(r)
        for rows in rt_eps.values(): rows.sort(key=lambda r: r[step_field])
        for ep_key, rows in rt_eps.items():
            validate_episode_step_sequence(rows, step_field)

        ol_eps = defaultdict(list)
        for r in ol_rows: ol_eps[r[ep_field]].append(r)
        for rows in ol_eps.values(): rows.sort(key=lambda r: r[step_field])
        for ep_key, rows in ol_eps.items():
            validate_episode_step_sequence(rows, step_field)

        split_cal = cal_data.get(split_key, {})
        if args.mode == "authoritative":
            for head in ["grasp", "manipulation", "release"]:
                hc = split_cal.get(head)
                if hc is None: raise SystemExit(f"CAL_HEAD_MISSING: {split_key}/{head}")
                for k in ["a","b","threshold","method_valid","transform_valid","fit_data_valid"]:
                    if hc.get(k) is None: raise SystemExit(f"CAL_MISSING_{k}: {split_key}/{head}")
                if hc["provenance_class"] != "INDEPENDENT_CALIBRATION":
                    raise SystemExit(f"CAL_NOT_INDEPENDENT: {split_key}/{head}")

        adapter = AdapterCls(
            structure=structure,
            calibration_contract=split_cal,
            require_l3_eligible=(args.mode == "authoritative"),
        )

        sched_results = {}
        for ep_key in sorted(rt_eps.keys()):
            result = adapter.run_episode(rt_eps[ep_key])
            # Strict field extraction — no defaults
            req = ["per_step_trace", "ever_emitted", "first_emit_step", "first_emit_trace",
                   "final_state", "reason_histogram", "l3_evaluation_eligible", "diagnostic_only"]
            for fld in req:
                if fld not in result:
                    raise SystemExit(f"ADAPTER_MISSING_FIELD: {fld}")
            sched_results[ep_key] = {
                "emitted": result["ever_emitted"],
                "emit_step": result["first_emit_step"] if result["first_emit_step"] is not None else -1,
                "final_state": result["final_state"],
            }

        m = compute_l3_metrics(ol_eps, sched_results, step_field)
        all_metrics[split_key] = m

    # Worst-split
    false_rates = {}
    for sk, m in all_metrics.items():
        r = m["negative_episode_false_start_rate"]
        false_rates[sk] = float(r) if r is not None else None
    defined = {k: v for k, v in false_rates.items() if v is not None}
    worst = max(defined.values()) if defined else None
    undefined = [k for k, v in false_rates.items() if v is None]

    # Staged output
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    auth_l3 = (args.mode == "authoritative")
    manifest = {
        "analysis": "FACTORIZED_V2_L3_ANALYSIS_V1",
        "authoritative_l3": auth_l3,
        "runner_status": "AUTHORITATIVE_L3_COMPLETE" if auth_l3 else "NON_AUTHORITATIVE_DIAGNOSTIC_COMPLETE",
        "mode": args.mode,
        "codex_handoff_sha256": handoff_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (staging / "l3_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "per_split_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    (staging / "summary.json").write_text(json.dumps({
        "n_splits": len(all_metrics), "worst_false_start": worst,
        "undefined_splits": undefined, "n_undefined": len(undefined),
    }, indent=2) + "\n")

    sums = {}
    for f in staging.rglob("*"):
        if f.is_file() and f.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            sums[f.relative_to(staging).as_posix()] = sha256_file(f)
    (staging / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")

    os.replace(staging, out_root)
    print(f"Sealed: {out_root}")


if __name__ == "__main__":
    main()
