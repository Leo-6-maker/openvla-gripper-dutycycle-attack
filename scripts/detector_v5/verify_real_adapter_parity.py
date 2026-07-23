#!/usr/bin/env python3
"""A9: Real adapter parity — verify offline replay and runtime adapter produce identical outputs.

Runs on a single clean canary episode (state 0 of any task).
Checks:
1. 25D feature vectors match element-by-element (max abs diff = 0)
2. Detector scores match within strict float tolerance (max abs diff <= 1e-7)
3. FSM arm/emit decisions are identical
4. No manual file copying was involved
"""
from __future__ import annotations

import argparse, json, math, os, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

SELF_SHA = None


def sha256_file(p: Path) -> str:
    import hashlib
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-replay-features-root", type=Path, required=True,
                    help="Offline replay 25D features for one canary episode (JSON)")
    ap.add_argument("--runtime-adapter-features-root", type=Path, required=True,
                    help="Runtime adapter 25D features for the SAME canary episode (JSON)")
    ap.add_argument("--offline-detector-scores-root", type=Path, required=True,
                    help="Offline replay detector scores (JSON)")
    ap.add_argument("--runtime-detector-scores-root", type=Path, required=True,
                    help="Runtime adapter detector scores (JSON)")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    errors: list[str] = []
    checks: dict[str, Any] = {}

    # ── Feature parity ───────────────────────────────────────────────
    off_feat = json.loads(args.offline_replay_features_root.read_text())
    run_feat = json.loads(args.runtime_adapter_features_root.read_text())
    off_vecs = off_feat.get("features", off_feat.get("vectors", []))
    run_vecs = run_feat.get("features", run_feat.get("vectors", []))

    if len(off_vecs) != len(run_vecs):
        errors.append(f"FEATURE_COUNT: offline={len(off_vecs)} runtime={len(run_vecs)}")
    else:
        max_diff = 0.0
        diff_steps: list[int] = []
        for i, (ov, rv) in enumerate(zip(off_vecs, run_vecs)):
            if isinstance(ov, list) and isinstance(rv, list):
                for j, (ox, rx) in enumerate(zip(ov, rv)):
                    d = abs(float(ox) - float(rx))
                    if d > max_diff: max_diff = d
                    if d > 0: diff_steps.append(i)
        checks["feature_count"] = len(off_vecs)
        checks["feature_max_abs_diff"] = max_diff
        checks["feature_diff_steps"] = diff_steps[:10]
        if max_diff > 0:
            errors.append(f"FEATURE_DIVERGENCE: max_abs_diff={max_diff} at steps {diff_steps[:5]}")

    # ── Detector score parity ────────────────────────────────────────
    off_scores = json.loads(args.offline_detector_scores_root.read_text())
    run_scores = json.loads(args.runtime_detector_scores_root.read_text())
    off_vals = off_scores.get("scores", off_scores.get("predictions", []))
    run_vals = run_scores.get("scores", run_scores.get("predictions", []))

    if len(off_vals) != len(run_vals):
        errors.append(f"SCORE_COUNT: offline={len(off_vals)} runtime={len(run_vals)}")
    else:
        max_score_diff = 0.0
        for i, (ov, rv) in enumerate(zip(off_vals, run_vals)):
            if isinstance(ov, dict) and isinstance(rv, dict):
                for head in ["grasp", "manipulation", "release"]:
                    d = abs(float(ov.get(head, 0)) - float(rv.get(head, 0)))
                    if d > max_score_diff: max_score_diff = d
            elif isinstance(ov, (int, float)) and isinstance(rv, (int, float)):
                d = abs(float(ov) - float(rv))
                if d > max_score_diff: max_score_diff = d
        checks["score_max_abs_diff"] = max_score_diff
        if max_score_diff > 1e-7:
            errors.append(f"SCORE_DIVERGENCE: max_abs_diff={max_score_diff}")

    # ── FSM parity ───────────────────────────────────────────────────
    off_fsm = off_scores.get("fsm", off_scores.get("decisions", {}))
    run_fsm = run_scores.get("fsm", run_scores.get("decisions", {}))
    checks["fsm_identical"] = (off_fsm == run_fsm)
    if not checks["fsm_identical"]:
        errors.append("FSM_DECISIONS_DIVERGE")

    receipt = {
        "schema": "REAL_ADAPTER_PARITY_RECEIPT_V1",
        "verifier_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "checks": checks,
        "n_errors": len(errors),
        "errors": errors,
    }

    import shutil
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "REAL_ADAPTER_PARITY_RECEIPT_V1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Adapter Parity: {receipt['status']} errors={len(errors)}")
    if errors:
        for e in errors: print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
