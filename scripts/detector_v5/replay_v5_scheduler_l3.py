#!/usr/bin/env python3
"""V5 scheduler L3 replay: TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY.

Authoritative L3 is BLOCKED on:
  RUNTIME_CANDIDATE_CLOSE  — policy raw_gripper not in prediction bundle
  STUDENT_VALID            — route_supported is proxy, not runtime student_valid
  UTILITY_SEMANTICS        — grasp_prob used as proxy for utility
  REGRASP_SEMANTICS        — manipulation_prob used as proxy for regrasp
  PROPER_INNER_TRAIN_PLATT — inner-train calibration inference not generated
  K10_CONTAINMENT          — no frozen K10 label in prediction

All metrics are placed under proxy_diagnostic_metrics.
authoritative_l3_metrics is always null.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, time, uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig

# ── Frozen expected 12-split closure ──

EXPECTED_SPLITS = frozenset(
    f"predict_V2B_EXACT_W32_H64_D0.1_WD1e-4_o{o}_i{i}_s42"
    for o in range(4) for i in range(3)
)

# ── SHA & sigmoid ──

def _sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            d.update(chunk)
    return d.hexdigest()


def _sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _sig(z):
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ── Scheduler config ──

SCHEDULER_CONFIG_SCHEMA = {
    "utility_threshold", "release_veto_threshold", "regrasp_veto_threshold",
    "uncertainty_veto_threshold", "release_veto_enabled", "regrasp_veto_enabled",
    "uncertainty_veto_enabled", "minimum_candidate_dwell",
    "persistence_window", "persistence_required",
}

FROZEN_DEFAULTS = {
    "utility_threshold": 0.5, "release_veto_threshold": 0.5,
    "regrasp_veto_threshold": 0.5, "uncertainty_veto_threshold": 0.5,
    "release_veto_enabled": True, "regrasp_veto_enabled": True,
    "uncertainty_veto_enabled": False,
    "minimum_candidate_dwell": 10, "persistence_window": 5, "persistence_required": 3,
}


def load_scheduler_config(config_path):
    """Load and validate scheduler config JSON. Returns (V5SchedulerConfig, effective_dict, sha256)."""
    if config_path is None:
        cfg = V5SchedulerConfig()
        effective = dict(FROZEN_DEFAULTS)
        effective_sha = _sha256_str(json.dumps(effective, sort_keys=True))
        return cfg, effective, effective_sha, None

    path = Path(config_path)
    if not path.is_file():
        raise SystemExit(f"Scheduler config not found: {path}")
    raw = json.loads(path.read_text())
    raw_sha = _sha256_file(path)

    unknown = set(raw.keys()) - SCHEDULER_CONFIG_SCHEMA
    if unknown:
        raise SystemExit(f"Unknown scheduler config fields: {sorted(unknown)}")

    for k, v in raw.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise SystemExit(f"Scheduler config field {k} is NaN/Inf")

    cfg = V5SchedulerConfig(**{k: v for k, v in raw.items() if k in SCHEDULER_CONFIG_SCHEMA})

    if cfg.minimum_candidate_dwell != 10:
        raise SystemExit("Frozen minimum_candidate_dwell must be 10")
    if cfg.persistence_window != 5 or cfg.persistence_required != 3:
        raise SystemExit("Frozen persistence rule must be 3-of-5")

    effective = {k: getattr(cfg, k) for k in SCHEDULER_CONFIG_SCHEMA}
    effective_sha = _sha256_str(json.dumps(effective, sort_keys=True))
    return cfg, effective, effective_sha, raw_sha


# ── Data loading ──

def load_steps(pred_dir):
    path = Path(pred_dir) / "heldout_step_predictions.jsonl"
    steps = []
    with open(path) as f:
        for line in f:
            steps.append(json.loads(line))
    return steps


def group_episodes(steps):
    eps = defaultdict(list)
    for s in steps:
        eps[s["canonical_parent_key"]].append(s)
    return dict(eps)


# ── Opportunity corridors ──

def find_teacher_event_corridors(ep_steps, min_length=8):
    ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
    corridors = []
    i = 0
    while i < len(ep_sorted):
        s = ep_sorted[i]
        if s.get("route_supported") and s.get("event_id", -1) >= 0:
            start = s["step_index"]
            j = i
            while (j < len(ep_sorted)
                   and ep_sorted[j].get("route_supported")
                   and ep_sorted[j].get("event_id", -1) >= 0
                   and ep_sorted[j]["step_index"] == start + (j - i)):
                j += 1
            if j - i >= min_length:
                corridors.append({
                    "start": start, "end": ep_sorted[j - 1]["step_index"],
                    "length": j - i, "source": "teacher_event_proxy",
                })
            i = j
        else:
            i += 1
    return corridors


# ── Scheduler replay ──

def replay_scheduler(episodes, calibrator, candidate_close_fn, valid_fn, sched_cfg):
    results = {}
    if calibrator is None:
        a_g, b_g = 1.0, 0.0
        a_r, b_r = 1.0, 0.0
    else:
        a_g = calibrator["grasp_a"]; b_g = calibrator["grasp_b"]
        a_r = calibrator["release_a"]; b_r = calibrator["release_b"]

    for ep_key, ep_steps in sorted(episodes.items()):
        ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
        scheduler = V5OneShotScheduler(sched_cfg)
        final = {}
        for s in ep_sorted:
            gp = s.get("grasp_prob", 0); rp = s.get("release_prob", 0)
            mp = s.get("manipulation_prob", 0)
            if calibrator is not None:
                if gp <= 0: gp = 1e-7
                if gp >= 1: gp = 1 - 1e-7
                if rp <= 0: rp = 1e-7
                if rp >= 1: rp = 1 - 1e-7
                gp = _sig(a_g * math.log(gp / (1 - gp)) + b_g)
                rp = _sig(a_r * math.log(rp / (1 - rp)) + b_r)

            final = scheduler.update(
                step=s["step_index"],
                candidate_close=candidate_close_fn(s),
                valid=valid_fn(s),
                utility_probability=gp,
                release_probability=rp,
                regrasp_probability=mp,
                uncertainty_probability=0.0,
            )

        results[ep_key] = {
            "emitted": final["one_shot_emitted"],
            "emit_step": final["emit_step"],
            "final_state": final["state"],
        }
    return results


# ── Proxy diagnostic metrics ──

def compute_proxy_metrics(episodes, scheduler_results, min_corridor_len=8):
    n_total = len(episodes)
    n_no_teacher = 0     # episodes with zero teacher events (pure bg)
    n_with_teacher = 0   # episodes with teacher events
    n_with_corridor = 0
    n_no_corridor = 0
    emits_on_corridor = 0
    emits_off_corridor = 0
    emits_no_teacher = 0
    abstain_with_corridor = 0

    per_ep = []

    for ep_key, ep_steps in sorted(episodes.items()):
        result = scheduler_results.get(ep_key, {})
        emitted = result.get("emitted", False)
        emit_step = result.get("emit_step", -1)
        corridors = find_teacher_event_corridors(ep_steps, min_corridor_len)

        has_teacher = any(s.get("event_id", -1) >= 0 for s in ep_steps if s.get("route_supported"))

        if not has_teacher:
            n_no_teacher += 1
            if emitted:
                emits_no_teacher += 1
            cls = "no_teacher_events_background_exposure_censored"
        elif not corridors:
            n_no_corridor += 1
            cls = "teacher_events_no_valid_corridor"
        else:
            n_with_corridor += 1
            if emitted:
                on_c = any(c["start"] <= emit_step <= c["end"] for c in corridors)
                if on_c:
                    emits_on_corridor += 1
                    cls = "emit_on_teacher_corridor"
                else:
                    emits_off_corridor += 1
                    cls = "emit_off_teacher_corridor"
            else:
                abstain_with_corridor += 1
                cls = "abstain_with_corridor"

        per_ep.append({
            "split": ep_steps[0].get("canonical_parent_key", "").split("/")[0] if ep_steps else "",
            "episode_key": ep_key,
            "has_teacher_events": has_teacher,
            "corridor_count": len(corridors),
            "candidate_source": "teacher_event_gate",
            "corridor_source": "teacher_event_proxy",
            "scheduler_emitted": emitted,
            "emit_step": emit_step,
            "on_proxy_corridor": cls == "emit_on_teacher_corridor",
            "proxy_classification": cls,
        })

    total_emitted = emits_no_teacher + emits_on_corridor + emits_off_corridor
    # verify
    assert total_emitted == sum(1 for v in scheduler_results.values() if v.get("emitted")), \
        f"emit count mismatch: {total_emitted} != sum"

    return {
        "total_episodes": n_total,
        "episodes_without_teacher_events": n_no_teacher,
        "episodes_with_teacher_events": n_with_teacher,
        "episodes_with_valid_corridor": n_with_corridor,
        "proxy_teacher_gated_negative_emit_rate":
            emits_no_teacher / max(1, n_no_teacher) if n_no_teacher > 0 else 0.0,
        "proxy_teacher_gated_on_corridor_rate":
            emits_on_corridor / max(1, n_with_corridor) if n_with_corridor > 0 else 0.0,
        "proxy_teacher_gated_off_corridor_rate":
            emits_off_corridor / max(1, n_with_corridor) if n_with_corridor > 0 else 0.0,
        "proxy_emit_precision":
            emits_on_corridor / max(1, total_emitted) if total_emitted > 0 else 0.0,
        "proxy_abstention_rate":
            abstain_with_corridor / max(1, n_with_corridor) if n_with_corridor > 0 else 0.0,
        "_counts": {
            "emits_no_teacher": emits_no_teacher,
            "emits_on_corridor": emits_on_corridor,
            "emits_off_corridor": emits_off_corridor,
            "abstain_with_corridor": abstain_with_corridor,
            "total_emitted": total_emitted,
        },
        "per_episode": per_ep,
    }


# ── Known-mask background emit (3 variants) ──

def compute_legacy_stage1_background_emit(steps, tau=0.5):
    """Legacy Stage-1 evaluator denominator: bg steps where ANY head known.
    Producer: evaluate_factorized_v2_inner_cv.py lines 162-166.
    """
    bg_known = 0; bg_emit = 0
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported"):
            gk = s.get("grasp_known_mask", False)
            mk = s.get("manipulation_known_mask", False)
            rk = s.get("release_known_mask", False)
            if gk or mk or rk:
                bg_known += 1
                if ((gk and s.get("grasp_prob", 0) >= tau) or
                    (mk and s.get("manipulation_prob", 0) >= tau) or
                    (rk and s.get("release_prob", 0) >= tau)):
                    bg_emit += 1
    return bg_emit / max(1, bg_known), bg_known


def compute_head_conditional_emit(steps, head, tau=0.5):
    known = 0; emit = 0
    km = f"{head}_known_mask"; pk = f"{head}_prob"
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported") and s.get(km, False):
            known += 1
            if s.get(pk, 0) >= tau: emit += 1
    return emit / max(1, known), known


def compute_all_bg_steps_emit(steps, tau=0.5):
    """Denominator: ALL supported bg steps (including those with zero known heads)."""
    total = 0; emit = 0
    for s in steps:
        if s.get("event_id", -1) < 0 and s.get("route_supported"):
            total += 1
            gk = s.get("grasp_known_mask", False)
            mk = s.get("manipulation_known_mask", False)
            rk = s.get("release_known_mask", False)
            if ((gk and s.get("grasp_prob", 0) >= tau) or
                (mk and s.get("manipulation_prob", 0) >= tau) or
                (rk and s.get("release_prob", 0) >= tau)):
                emit += 1
    return emit / max(1, total), total


# ── Calibration validation (fail-closed) ──

def validate_calibration_bundle(bundle_data, fit_manifest, heldout_manifest):
    errors = []
    # fit manifest required
    if fit_manifest is None:
        errors.append("calibration-fit-manifest is required with calibration-bundle")
    else:
        fit_ids = set(fit_manifest.get("fit_identities", []))
        if not fit_ids:
            errors.append("fit_identities is empty")
        if "split" not in fit_manifest:
            errors.append("fit manifest missing 'split' key")
        if "checkpoint_sha256" not in fit_manifest:
            errors.append("fit manifest missing 'checkpoint_sha256'")

    # heldout manifest required
    if heldout_manifest is None:
        errors.append("prediction-manifest is required with calibration-bundle")
    else:
        ho_ids = set(heldout_manifest.get("heldout_identities", []))
        if not ho_ids:
            errors.append("heldout_identities is empty")
        if fit_manifest and ho_ids:
            overlap = fit_ids & ho_ids
            if overlap:
                errors.append(f"CALIBRATION_LEAKAGE: {len(overlap)} identities in fit ∩ heldout")

    # Bundle validation
    if not isinstance(bundle_data, list):
        errors.append("calibration bundle must be a JSON array")
    else:
        heads_seen = set()
        for entry in bundle_data:
            h = entry.get("head", "")
            if h in heads_seen:
                errors.append(f"duplicate head in bundle: {h}")
            heads_seen.add(h)
            if h not in ("grasp", "release", "manipulation"):
                errors.append(f"unknown head: {h}")
            for k in ("a", "b"):
                v = entry.get(k)
                if not isinstance(v, (int, float)):
                    errors.append(f"head {h}: {k} must be numeric, got {type(v).__name__}")
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    errors.append(f"head {h}: {k} is NaN/Inf")

    if errors:
        raise SystemExit("CALIBRATION_VALIDATION_FAILED:\n  " + "\n  ".join(errors))

    # Build calibrator dict
    cal = {}
    for entry in bundle_data:
        h = entry["head"]
        cal[f"{h}_a"] = float(entry["a"])
        cal[f"{h}_b"] = float(entry["b"])
    return cal


# ── Split closure verification ──

def verify_split_closure(pred_root):
    found = set()
    for d in pred_root.iterdir():
        if d.is_dir() and d.name.startswith("predict_V2B_EXACT_"):
            found.add(d.name)

    missing = EXPECTED_SPLITS - found
    extra = found - EXPECTED_SPLITS
    if missing:
        raise SystemExit(f"MISSING_SPLITS: {sorted(missing)}")
    if extra:
        raise SystemExit(f"EXTRA_SPLITS: {sorted(extra)}")

    # Verify each has prediction file and SHA256SUMS seal
    verified = []
    for name in sorted(EXPECTED_SPLITS):
        d = pred_root / name
        pred_file = d / "heldout_step_predictions.jsonl"
        if not pred_file.is_file():
            raise SystemExit(f"PREDICTION_MISSING: {name}/heldout_step_predictions.jsonl")
        seal_file = d / "SHA256SUMS"
        if not seal_file.is_file():
            print(f"  Warning: {name} has no SHA256SUMS seal — continuing without seal verification")
        verified.append(name)
    return verified


# ── Staged output ──

def atomic_output_write(out_root, files):
    """Write files to staging dir, create SHA256SUMS, atomic rename."""
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    written = []
    for relname, content in files.items():
        p = staging / relname
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            p.write_text(content)
        elif isinstance(content, bytes):
            p.write_bytes(content)
        elif isinstance(content, (list, dict)):
            p.write_text(json.dumps(content, indent=2) + "\n")
        elif hasattr(content, 'read'):
            with open(p, 'wb') as dst:
                for chunk in iter(lambda: content.read(1048576), b""):
                    dst.write(chunk)
        written.append(relname)

    # SHA256SUMS
    sums_lines = []
    for relname in sorted(written):
        h = _sha256_file(staging / relname)
        sums_lines.append(f"{h}  {relname}\n")
    (staging / "SHA256SUMS").write_text("".join(sums_lines))
    seal_h = _sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal_h}  SHA256SUMS\n")

    os.replace(staging, out_root)


# ── Main ──

def main():
    ap = argparse.ArgumentParser(
        description="V5 scheduler L3 replay — TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY")
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--prediction-manifest", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--calibration-mode", choices=["raw", "external_bundle"], default="raw")
    ap.add_argument("--calibration-bundle", type=Path, default=None)
    ap.add_argument("--calibration-fit-manifest", type=Path, default=None)
    ap.add_argument("--opportunity-source", type=str, default="teacher_event_proxy",
                    choices=["teacher_event_proxy"])
    ap.add_argument("--min-corridor-length", type=int, default=8)
    ap.add_argument("--scheduler-config", type=Path, default=None)
    args = ap.parse_args()

    pred_root = args.prediction_root.resolve()
    out_root = args.output_root.resolve()

    # Reject overwrite
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root} — refusing to overwrite")

    # Scheduler config
    sched_cfg, effective_cfg, effective_cfg_sha, raw_cfg_sha = load_scheduler_config(args.scheduler_config)

    # Calibration
    calibrator = None
    calib_status = "RAW_UNCALIBRATED_DIAGNOSTIC"
    calib_bundle_sha = None

    if args.calibration_mode == "external_bundle":
        calib_status = "NON_AUTHORITATIVE"
        if args.calibration_bundle is None:
            raise SystemExit("--calibration-mode external_bundle requires --calibration-bundle")
        if args.calibration_fit_manifest is None:
            raise SystemExit("--calibration-mode external_bundle requires --calibration-fit-manifest")
        if args.prediction_manifest is None:
            raise SystemExit("--calibration-mode external_bundle requires --prediction-manifest")

        cb = Path(args.calibration_bundle)
        calib_bundle_sha = _sha256_file(cb)
        bundle_data = json.loads(cb.read_text())
        fit_manifest = json.loads(Path(args.calibration_fit_manifest).read_text())
        heldout_manifest = json.loads(Path(args.prediction_manifest).read_text()) if args.prediction_manifest else None

        calibrator = validate_calibration_bundle(bundle_data, fit_manifest, heldout_manifest)
    elif args.calibration_bundle is not None:
        raise SystemExit("--calibration-bundle requires --calibration-mode external_bundle")

    # Artifact identity
    script_sha = _sha256_file(Path(__file__))
    scheduler_sha = _sha256_file(ROOT / "src/gripper_attack/v5_scheduler.py")
    source_commit = None
    try:
        import subprocess
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass

    # Split closure
    verified_splits = verify_split_closure(pred_root)
    n_verified = len(verified_splits)
    if n_verified != 12:
        raise SystemExit(f"SPLIT_CLOSURE_FAILED: expected 12, verified {n_verified}")

    # ── Proxy candidate gate ──
    def candidate_close_fn(step):
        return step.get("route_supported", False) and step.get("event_id", -1) >= 0

    def valid_fn(step):
        return step.get("route_supported", False)

    # ── Replay ──
    all_metrics = {}
    csv_rows = []
    per_ep_all = []

    for sl in verified_splits:
        steps = load_steps(pred_root / sl)
        episodes = group_episodes(steps)
        short = sl.replace("predict_V2B_EXACT_W32_H64_D0.1_WD1e-4_", "")

        sched_results = replay_scheduler(episodes, calibrator, candidate_close_fn, valid_fn, sched_cfg)
        m = compute_proxy_metrics(episodes, sched_results, args.min_corridor_length)
        all_metrics[short] = m
        per_ep_all.extend(m["per_episode"])

        c = m["_counts"]
        print(f"{short:<10} no_teacher={m['episodes_without_teacher_events']:>3} "
              f"w_corridor={m['episodes_with_valid_corridor']:>3} "
              f"total_emit={c['total_emitted']:>3} "
              f"no_teacher_emit={c['emits_no_teacher']:>2} "
              f"on_c={c['emits_on_corridor']:>2} "
              f"off_c={c['emits_off_corridor']:>2} "
              f"abstain={c['abstain_with_corridor']:>2}")

        csv_rows.append({
            "split": short,
            "total_episodes": m["total_episodes"],
            "episodes_without_teacher_events": m["episodes_without_teacher_events"],
            "episodes_with_valid_corridor": m["episodes_with_valid_corridor"],
            "proxy_teacher_gated_negative_emit_rate": round(m["proxy_teacher_gated_negative_emit_rate"], 6),
            "proxy_teacher_gated_on_corridor_rate": round(m["proxy_teacher_gated_on_corridor_rate"], 6),
            "proxy_teacher_gated_off_corridor_rate": round(m["proxy_teacher_gated_off_corridor_rate"], 6),
            "proxy_emit_precision": round(m["proxy_emit_precision"], 6),
            "proxy_abstention_rate": round(m["proxy_abstention_rate"], 6),
        })

    # L1 denominator audit
    first_sl = verified_splits[0]
    steps0 = load_steps(pred_root / first_sl)
    leg_rate, leg_n = compute_legacy_stage1_background_emit(steps0)
    all_rate, all_n = compute_all_bg_steps_emit(steps0)
    g_rate, g_n = compute_head_conditional_emit(steps0, "grasp")
    m_rate, m_n = compute_head_conditional_emit(steps0, "manipulation")

    known_denom_audit = {
        "producer": {
            "file": "scripts/detector_v5/evaluate_factorized_v2_inner_cv.py",
            "lines": "162-166",
            "description": "Legacy denominator = bg steps where any head known",
        },
        "legacy_stage1_background_emit": {
            "rate": leg_rate, "denominator": leg_n,
            "definition": "Denominator = bg steps where any(grasp_known, manipulation_known, release_known). Matches Stage-1 evaluator lines 162-166.",
        },
        "head_conditional_grasp": {"rate": g_rate, "denominator": g_n,
                                    "definition": "Denominator = bg steps where grasp_known_mask is True."},
        "head_conditional_manipulation": {"rate": m_rate, "denominator": m_n,
                                           "definition": "Denominator = bg steps where manipulation_known_mask is True."},
        "all_supported_bg_steps": {"rate": all_rate, "denominator": all_n,
                                    "definition": "Denominator = ALL bg steps with route_supported. Includes steps where zero heads are known. NOT the Stage-1 evaluator denominator."},
    }

    # ── Build output ──
    proxy_interpretation = (
        "Does not measure runtime background false starts. "
        "Teacher event membership (event_id >= 0) gates candidate_close, "
        "so pure-background episodes always have candidate_close=False. "
        "runtime_background_exposure_measured = false."
    )

    manifest = {
        "replay_status": "TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY",
        "authoritative_l3": False,
        "proxy_metric_interpretation": proxy_interpretation,
        "runtime_background_exposure_measured": False,

        "candidate_close_status": "TEACHER_EVENT_GATE",
        "candidate_source": "teacher_event_gate",
        "background_candidate_exposure": "STRUCTURALLY_CENSORED",
        "RUNTIME_CANDIDATE_CLOSE": "BLOCKED",

        "student_valid_status": "UNVALIDATED_PROXY_ROUTE_SUPPORTED",
        "STUDENT_VALID": "BLOCKED_MISSING_RUNTIME_FIELD",

        "utility_mapping_status": "UNVALIDATED_SEMANTIC_PROXY (grasp_prob)",
        "UTILITY_SEMANTICS": "BLOCKED",
        "regrasp_mapping_status": "UNVALIDATED_SEMANTIC_PROXY (manipulation_prob)",
        "REGRASP_SEMANTICS": "BLOCKED",
        "release_mapping_status": "DIRECT (release_prob)",

        "opportunity_label_status": f"TEACHER_EVENT_PROXY (source={args.opportunity_source})",
        "calibration_status": calib_status,
        "k10_containment_status": "NOT_MEASURED",
        "PROPER_INNER_TRAIN_PLATT": "BLOCKED",
        "K10_CONTAINMENT": "NOT_MEASURED",

        "script_sha256": script_sha,
        "scheduler_sha256": scheduler_sha,
        "scheduler_source": "src/gripper_attack/v5_scheduler.py",
        "scheduler_config_source": "CLI" if args.scheduler_config else "DEFAULT_CODE_FROZEN_CONFIG",
        "scheduler_config_raw_sha256": raw_cfg_sha,
        "scheduler_config_effective": effective_cfg,
        "scheduler_config_effective_sha256": effective_cfg_sha,
        "calibration_bundle_sha256": calib_bundle_sha,
        "source_commit": source_commit,
        "n_splits": n_verified,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "proxy_diagnostic_metrics": all_metrics,
        "authoritative_l3_metrics": None,
    }

    # Staged output
    atomic_output_write(out_root, {
        "replay_manifest.json": manifest,
        "proxy_diagnostic_metrics.json": all_metrics,
        "proxy_diagnostic_per_split.csv": csv_rows,
        "per_episode_classification.jsonl":
            "".join(json.dumps(e) + "\n" for e in per_ep_all),
        "known_mask_denominator_audit.json": known_denom_audit,
    })

    print(f"\nOutput sealed: {out_root}")
    print(f"Status: {manifest['replay_status']}")
    print(f"Authoritative L3: {manifest['authoritative_l3']}")


if __name__ == "__main__":
    main()
