#!/usr/bin/env python3
"""Audit Factorized Student OOF predictions for completeness and integrity.

Checks:
- All 24 shards present and sealed
- Step count matches manifest
- No missing episodes vs val set
- No cross-fold contamination
- Unsupported route abstention verified
- No NaN/Inf probabilities
- Route coverage
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory

SUPPORTED_ROUTES = ["single_object_pick_place", "multi_object_transfer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-base", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    pred_base = args.predictions_base.resolve()
    model_types = ["25D9D", "25D"]
    folds = [0, 1, 2, 3]
    seeds = [42, 123, 456]

    # Load fold validation identities
    fold_bundle = load_fit_fold_bundle(args.fold_root)
    fold_val_ids = {}
    for f in fold_bundle["folds"]:
        fold_val_ids[f["fold_id"]] = set(f["validation_identities"])

    # Load registry
    rows = list(csv.DictReader(open(args.registry)))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]

    issues = []
    stats = {"total_shards": 0, "sealed_ok": 0, "total_steps": 0, "total_events": 0,
             "total_episodes": 0, "unsupported_episodes": 0, "cross_fold_leak": 0,
             "nan_inf_probs": 0, "missing_episodes": 0, "unsupported_route_emit": 0}

    print("=== Prediction Audit ===")
    for mt in model_types:
        for fold in folds:
            val_ids = fold_val_ids[fold]
            for seed in seeds:
                shard = pred_base / f"predict_{mt}_fold{fold}_seed{seed}"
                stats["total_shards"] += 1

                if not shard.is_dir():
                    issues.append(f"MISSING_SHARD: {shard.name}")
                    continue

                # Verify seal
                try:
                    verify_sealed_directory(shard)
                    stats["sealed_ok"] += 1
                except Exception as e:
                    issues.append(f"SEAL_FAIL: {shard.name}: {e}")
                    continue

                # Load manifest
                manifest = json.loads((shard / "prediction_manifest.json").read_text())
                if manifest["model_type"] != mt:
                    issues.append(f"MANIFEST_MT: {shard.name}: {manifest['model_type']} != {mt}")
                if manifest["fold_id"] != fold:
                    issues.append(f"MANIFEST_FOLD: {shard.name}: {manifest['fold_id']} != {fold}")
                if manifest["seed"] != seed:
                    issues.append(f"MANIFEST_SEED: {shard.name}: {manifest['seed']} != {seed}")

                # Load step predictions
                sp_file = shard / "heldout_step_predictions.jsonl"
                ep_file = shard / "heldout_event_predictions.jsonl"

                if not sp_file.is_file():
                    issues.append(f"MISSING_STEP_PREDICTIONS: {shard.name}")
                    continue
                if not ep_file.is_file():
                    issues.append(f"MISSING_EVENT_PREDICTIONS: {shard.name}")
                    continue

                step_recs = [json.loads(l) for l in sp_file.read_text().splitlines() if l.strip()]
                event_recs = [json.loads(l) for l in ep_file.read_text().splitlines() if l.strip()]

                stats["total_steps"] += len(step_recs)
                stats["total_events"] += len(event_recs)

                # Check predicted identities are exactly val set for this fold
                pred_ids = set(r["canonical_parent_key"] for r in step_recs)
                stats["total_episodes"] += len(pred_ids)

                # Cross-fold leak
                expected_ids = val_ids
                extra = pred_ids - expected_ids
                missing = expected_ids - pred_ids
                if extra:
                    stats["cross_fold_leak"] += len(extra)
                    issues.append(f"CROSS_FOLD_LEAK: {shard.name}: {len(extra)} extra identities")
                if missing:
                    stats["missing_episodes"] += len(missing)
                    issues.append(f"MISSING_EPISODES: {shard.name}: {len(missing)} missing identities")

                # Check NaN/Inf
                for r in step_recs:
                    for k in ["grasp_prob", "manipulation_prob", "release_prob",
                               "grasp_logit", "manipulation_logit", "release_logit"]:
                        v = r.get(k, 0)
                        if v != v or v == float('inf') or v == float('-inf'):
                            stats["nan_inf_probs"] += 1
                            issues.append(f"NaN/Inf: {shard.name}: {r['canonical_parent_key']} step={r['step_index']} {k}={v}")
                            break

                # Unsupported route: must have 0 probs
                for r in step_recs:
                    if not r["route_supported"]:
                        if r["grasp_prob"] > 0 or r["manipulation_prob"] > 0 or r["release_prob"] > 0:
                            stats["unsupported_route_emit"] += 1
                            issues.append(f"UNSUPPORTED_EMIT: {shard.name}: {r['canonical_parent_key']} step={r['step_index']}")

                # Verify step count matches manifest
                if len(step_recs) != manifest.get("total_steps", -1):
                    issues.append(f"STEP_COUNT: {shard.name}: {len(step_recs)} != manifest {manifest.get('total_steps')}")

                # Route coverage
                routes_seen = set(r["mechanism_route"] for r in step_recs)
                print(f"  {mt}/fold{fold}_seed{seed}: {len(step_recs)} steps, {len(event_recs)} events, {len(pred_ids)} eps, routes={routes_seen}")

    # ── Summary ──
    print(f"\n=== Audit Summary ===")
    print(f"Shards: {stats['total_shards']} total, {stats['sealed_ok']} sealed OK")
    print(f"Steps: {stats['total_steps']}")
    print(f"Events: {stats['total_events']}")
    print(f"Episodes: {stats['total_episodes']}")
    print(f"Issues: {len(issues)}")
    for issue in issues:
        print(f"  {issue}")

    audit_result = {
        "status": "PASS" if len(issues) == 0 else "FAIL",
        "issue_count": len(issues),
        "issues": issues,
        "stats": stats,
    }

    if args.output:
        args.output.write_text(json.dumps(audit_result, indent=2) + "\n")

    if issues:
        print(f"\nAUDIT: FAIL ({len(issues)} issues)")
        sys.exit(1)
    else:
        print(f"\nAUDIT: PASS")


if __name__ == "__main__":
    main()
