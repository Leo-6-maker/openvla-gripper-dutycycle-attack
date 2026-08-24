#!/usr/bin/env python3
"""A9: Real adapter parity — fail-closed verification of offline vs runtime equivalence."""
from __future__ import annotations

import argparse, json, math, os, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))
from pilot_integrity import sha256_file, is_64char_hex, consume_sealed_root

SELF_SHA = None
EXPECTED_FEATURE_DIM = 25
FLOAT_TOLERANCE = 1e-7


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-features-root", type=Path, required=True)
    ap.add_argument("--runtime-features-root", type=Path, required=True)
    ap.add_argument("--offline-scores-root", type=Path, required=True)
    ap.add_argument("--runtime-scores-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    errors: list[str] = []
    checks: dict[str, Any] = {}

    # ── Consume sealed roots ─────────────────────────────────────────
    off_feat_data, off_feat_seal = consume_sealed_root(
        args.offline_features_root, "FEATURE_BUNDLE", "OFF_FEAT")
    run_feat_data, run_feat_seal = consume_sealed_root(
        args.runtime_features_root, "FEATURE_BUNDLE", "RUN_FEAT")
    off_score_data, off_score_seal = consume_sealed_root(
        args.offline_scores_root, "SCORE_BUNDLE", "OFF_SCORE")
    run_score_data, run_score_seal = consume_sealed_root(
        args.runtime_scores_root, "SCORE_BUNDLE", "RUN_SCORE")

    # ── Feature parity ───────────────────────────────────────────────
    off_vecs = off_feat_data.get("features", off_feat_data.get("vectors", []))
    run_vecs = run_feat_data.get("features", run_feat_data.get("vectors", []))

    if not isinstance(off_vecs, list) or len(off_vecs) == 0:
        errors.append("OFFLINE_FEATURES_EMPTY")
    if not isinstance(run_vecs, list) or len(run_vecs) == 0:
        errors.append("RUNTIME_FEATURES_EMPTY")

    if off_vecs and run_vecs:
        if len(off_vecs) != len(run_vecs):
            errors.append(f"FEATURE_TIMESTEP_MISMATCH: offline={len(off_vecs)} runtime={len(run_vecs)}")
        else:
            max_diff = 0.0
            for i, (ov, rv) in enumerate(zip(off_vecs, run_vecs)):
                if not isinstance(ov, list) or not isinstance(rv, list):
                    errors.append(f"FEATURE_NOT_LIST: step={i}")
                    continue
                if len(ov) != EXPECTED_FEATURE_DIM or len(rv) != EXPECTED_FEATURE_DIM:
                    errors.append(f"FEATURE_DIM_MISMATCH: step={i} offline={len(ov)} runtime={len(rv)}")
                    continue
                for j, (ox, rx) in enumerate(zip(ov, rv)):
                    if not isinstance(ox, (int, float)) or isinstance(ox, bool):
                        errors.append(f"FEATURE_NON_NUMERIC: step={i} dim={j} offline={ox!r}")
                    if not isinstance(rx, (int, float)) or isinstance(rx, bool):
                        errors.append(f"FEATURE_NON_NUMERIC: step={i} dim={j} runtime={rx!r}")
                    oxf, rxf = float(ox), float(rx)
                    if not math.isfinite(oxf): errors.append(f"FEATURE_NAN_INF: step={i} dim={j} offline={oxf}")
                    if not math.isfinite(rxf): errors.append(f"FEATURE_NAN_INF: step={i} dim={j} runtime={rxf}")
                    d = abs(oxf - rxf)
                    if d > max_diff: max_diff = d
            checks["feature_count"] = len(off_vecs)
            checks["feature_dim"] = EXPECTED_FEATURE_DIM
            checks["feature_max_abs_diff"] = max_diff
            if max_diff > 0:
                errors.append(f"FEATURE_DIVERGENCE: max_abs_diff={max_diff}")

    # ── Score parity ─────────────────────────────────────────────────
    off_vals = off_score_data.get("scores", off_score_data.get("predictions", []))
    run_vals = run_score_data.get("scores", run_score_data.get("predictions", []))

    if not isinstance(off_vals, list) or len(off_vals) == 0:
        errors.append("OFFLINE_SCORES_EMPTY")
    if not isinstance(run_vals, list) or len(run_vals) == 0:
        errors.append("RUNTIME_SCORES_EMPTY")

    if off_vals and run_vals and len(off_vals) == len(run_vals):
        max_score_diff = 0.0
        for i, (ov, rv) in enumerate(zip(off_vals, run_vals)):
            if isinstance(ov, dict) and isinstance(rv, dict):
                for head in ["grasp", "manipulation", "release"]:
                    if head in ov and head in rv:
                        d = abs(float(ov[head]) - float(rv[head]))
                        if d > max_score_diff: max_score_diff = d
            elif isinstance(ov, (int, float)) and isinstance(rv, (int, float)):
                d = abs(float(ov) - float(rv))
                if d > max_score_diff: max_score_diff = d
            else:
                errors.append(f"SCORE_TYPE_MISMATCH: step={i}")
        checks["score_max_abs_diff"] = max_score_diff
        if max_score_diff > FLOAT_TOLERANCE:
            errors.append(f"SCORE_DIVERGENCE: max_abs_diff={max_score_diff} > {FLOAT_TOLERANCE}")

    # ── FSM parity ───────────────────────────────────────────────────
    off_fsm = off_score_data.get("fsm", off_score_data.get("decisions", {}))
    run_fsm = run_score_data.get("fsm", run_score_data.get("decisions", {}))
    if not isinstance(off_fsm, dict) or not isinstance(run_fsm, dict):
        errors.append("FSM_NOT_DICT")
    elif not off_fsm and not run_fsm:
        errors.append("FSM_EMPTY_BOTH: cannot verify parity with empty FSM")
    else:
        fsm_identical = (off_fsm == run_fsm)
        checks["fsm_identical"] = fsm_identical
        if not fsm_identical:
            errors.append("FSM_DECISIONS_DIVERGE")

    receipt = {
        "schema": "REAL_ADAPTER_PARITY_RECEIPT_V1",
        "verifier_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "checks": checks,
        "n_errors": len(errors),
        "errors": errors,
        "offline_feature_seal": off_feat_seal,
        "runtime_feature_seal": run_feat_seal,
        "offline_score_seal": off_score_seal,
        "runtime_score_seal": run_score_seal,
    }

    import shutil
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "REAL_ADAPTER_PARITY_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file()
                   and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Adapter Parity: {receipt['status']} errors={len(errors)}")
    for e in errors: print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
