#!/usr/bin/env python3
"""Aggregate 864 Stage-1 runs into 72 configuration records.

Each config = (candidate, W, hidden_dim, dropout, weight_decay) × 12 inner-CV runs.
Fail-closed: missing/invalid runs → HOLD.
Outputs: per-config metrics, safety elimination, LR comparison, shortlist.
"""
import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
DEFAULT_RUNS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_RUNS_V1_20260721'
DEFAULT_LR = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_LR_BASELINE_V1_20260721'
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_AGGREGATION_V1_20260721'

SEEDS = [42]
OUTER_FOLDS = [0, 1, 2, 3]
INNER_FOLDS = [0, 1, 2]
SAFETY_LIMITS = {
    'background_false_emit_rate': 0.10,
    'release_overlap_emit_rate': 0.05,
    'unsupported_route_emit_rate': 0.0,
}


def sha256_file(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.tmp')
    with t.open('x') as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {'SHA256SUMS', 'SHA256SUMS.sha256'}
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = ''.join(f'{sha256_file(p)}  {p.relative_to(root).as_posix()}\n' for p in fs)
    _atomic_text(root / 'SHA256SUMS', c)
    _atomic_text(root / 'SHA256SUMS.sha256', f'{sha256_file(root / "SHA256SUMS")}  SHA256SUMS\n')


def make_config_key(candidate, W, hidden_dim, dropout, weight_decay):
    return f'{candidate}_W{W}_H{hidden_dim}_D{dropout}_WD{weight_decay}'


def make_run_label(candidate, W, hidden_dim, dropout, weight_decay, outer, inner, seed):
    return f'{candidate}_W{W}_H{hidden_dim}_D{dropout}_WD{weight_decay}_o{outer}_i{inner}_s{seed}'


def load_run_metrics(runs_root, label):
    """Load evaluation metrics for one completed run."""
    eval_file = runs_root / f'eval_{label}.json'
    if not eval_file.is_file():
        return None
    data = json.loads(eval_file.read_text())
    key = list(data.get('per_run', {}).keys())
    if not key:
        return None
    return data['per_run'][key[0]]


def check_run_integrity(runs_root, label, auth_seal, source_commit):
    """Verify all artifacts for one run. Returns (ok, issues)."""
    issues = []
    train_dir = runs_root / label
    pred_dir = runs_root / f'predict_{label}'

    # Training seal
    if not (train_dir / 'SHA256SUMS').is_file():
        issues.append('TRAIN_NOT_SEALED')
    else:
        try:
            rc = json.loads((train_dir / 'run_config.json').read_text())
            ar = json.loads((train_dir / 'authorization_receipt.json').read_text())
            if ar.get('authorization_seal') != auth_seal:
                issues.append('AUTH_SEAL_MISMATCH')
            if rc.get('source_commit', '') != source_commit:
                issues.append('COMMIT_MISMATCH')
        except Exception as e:
            issues.append(f'TRAIN_METADATA: {e}')

    # Prediction seal
    if not (pred_dir / 'SHA256SUMS').is_file():
        issues.append('PREDICT_NOT_SEALED')

    # Evaluation
    if load_run_metrics(runs_root, label) is None:
        issues.append('EVAL_MISSING')

    # Audit
    audit_file = runs_root / f'audit_{label}.json'
    if not audit_file.is_file():
        issues.append('AUDIT_MISSING')
    else:
        audit = json.loads(audit_file.read_text())
        if audit.get('status') != 'PASS':
            issues.append(f'AUDIT_{audit.get("status", "UNKNOWN")}')

    return len(issues) == 0, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--lr-baseline-root', type=Path, default=DEFAULT_LR)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--auth-seal', type=str, required=True)
    ap.add_argument('--source-commit', type=str, required=True)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')

    # Collect all 864 runs by config
    candidates = ['V2A', 'V2B', 'V2C']
    W_vals = [16, 32, 64]
    H_vals = [64, 128]
    D_vals = [0.0, 0.1]
    WD_vals = [1e-5, 1e-4]

    configs = {}
    all_runs = {}
    integrity_issues = []
    missing_runs = []
    invalid_runs = []

    for candidate in candidates:
        for W in W_vals:
            for H in H_vals:
                for D in D_vals:
                    for WD in WD_vals:
                        ck = make_config_key(candidate, W, H, D, WD)
                        configs[ck] = {
                            'candidate': candidate, 'W': W, 'hidden_dim': H,
                            'dropout': D, 'weight_decay': WD, 'runs': {}
                        }
                        for outer in OUTER_FOLDS:
                            for inner in INNER_FOLDS:
                                for seed in SEEDS:
                                    label = make_run_label(candidate, W, H, D, WD, outer, inner, seed)
                                    ok, issues = check_run_integrity(
                                        args.runs_root, label, args.auth_seal, args.source_commit)
                                    run_info = {'label': label, 'ok': ok, 'issues': issues}
                                    configs[ck]['runs'][(outer, inner)] = run_info
                                    all_runs[label] = run_info
                                    if not ok:
                                        invalid_runs.append(run_info)
                                    if not (args.runs_root / label).exists():
                                        missing_runs.append(label)

    print(f'Total configs: {len(configs)}')
    print(f'Total runs expected: {864}')
    print(f'Missing: {len(missing_runs)}')
    print(f'Invalid: {len(invalid_runs)}')

    if missing_runs or invalid_runs:
        print('HOLD: missing or invalid runs')
        if missing_runs:
            print(f'  Missing: {missing_runs[:10]}...')
        if invalid_runs:
            for r in invalid_runs[:5]:
                print(f'  Invalid {r["label"]}: {r["issues"]}')
        # Still produce partial output for diagnosis
        partial = {
            'status': 'HOLD_INCOMPLETE',
            'missing_runs': len(missing_runs),
            'invalid_runs': len(invalid_runs),
        }
        staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
        staging.mkdir(parents=True)
        _atomic_text(staging / 'integrity_report.json', json.dumps(partial, indent=2))
        write_seal(staging)
        os.replace(staging, out)
        sys.exit(1)

    # ── Aggregate metrics per config ──
    lr_data = json.loads((args.lr_baseline_root / 'pooled_metrics.json').read_text())
    lr_per_split = json.loads((args.lr_baseline_root / 'per_split_metrics.json').read_text())

    config_metrics = {}
    safety_eliminated = {}
    lr_comparison = {}

    for ck, cfg in configs.items():
        metrics_list = []
        safety_failures = []
        for (outer, inner), run_info in cfg['runs'].items():
            m = load_run_metrics(args.runs_root, run_info['label'])
            if m is None:
                safety_failures.append(f'o{outer}_i{inner}: EVAL_MISSING')
                continue
            metrics_list.append(m)

            # Check safety per-split
            for sk, limit in SAFETY_LIMITS.items():
                val = m.get(sk)
                if val is not None and val > limit:
                    safety_failures.append(f'o{outer}_i{inner}: {sk}={val:.4f} > {limit}')

        if not metrics_list:
            safety_eliminated[ck] = {'reason': 'NO_VALID_RUNS', 'failures': safety_failures}
            continue

        # Aggregate across 12 splits
        agg = {}
        metric_keys = ['release_auroc', 'release_auprc', 'release_short_auprc',
                       'release_recall_05', 'background_false_emit_rate',
                       'release_overlap_emit_rate', 'unsupported_route_emit_rate',
                       'release_first_recall_05', 'release_later_recall_05']

        for mk in metric_keys:
            vals = [m[mk] for m in metrics_list if m.get(mk) is not None]
            if vals:
                agg[f'{mk}_mean'] = mean(vals)
                agg[f'{mk}_stdev'] = stdev(vals) if len(vals) > 1 else 0
                agg[f'{mk}_per_split'] = vals
                if mk in SAFETY_LIMITS:
                    agg[f'{mk}_worst'] = max(vals)

        agg['n_runs'] = len(metrics_list)
        agg['n_expected'] = 12

        # Safety disposition (worst-split)
        safety_pass = True
        for sk, limit in SAFETY_LIMITS.items():
            worst = agg.get(f'{sk}_worst')
            if worst is not None and worst > limit:
                safety_pass = False
                safety_failures.append(f'WORST: {sk}={worst:.4f} > {limit}')

        if not safety_pass:
            safety_eliminated[ck] = {
                'reason': 'SAFETY_FAILURE',
                'failures': safety_failures,
                'metrics': {k: v for k, v in agg.items() if not isinstance(v, list)},
            }
            config_metrics[ck] = agg
            continue

        agg['safety_pass'] = True
        config_metrics[ck] = agg

        # LR comparison
        lr_auroc = lr_data.get('release_auroc', 0)
        lr_auprc = lr_data.get('release_auprc', 0)
        v2_auroc = agg.get('release_auroc_mean', 0)
        v2_auprc = agg.get('release_auprc_mean', 0)
        beats_lr = (v2_auroc > lr_auroc) and (v2_auprc > lr_auprc)
        lr_comparison[ck] = {
            'v2_auroc': v2_auroc, 'lr_auroc': lr_auroc,
            'v2_auprc': v2_auprc, 'lr_auprc': lr_auprc,
            'beats_lr': beats_lr,
        }

    n_safety_pass = len(config_metrics) - len(safety_eliminated)
    n_beats_lr = sum(1 for v in lr_comparison.values() if v['beats_lr'])

    print(f'\nSafety pass: {n_safety_pass}/{len(configs)}')
    print(f'Beats LR: {n_beats_lr}/{n_safety_pass}')
    print(f'Safety eliminated: {len(safety_eliminated)}')

    # ── Write outputs ──
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    integrity = {
        'status': 'PASS' if (not missing_runs and not invalid_runs) else 'HOLD',
        'total_runs_expected': 864,
        'total_configs': 72,
        'missing_runs': len(missing_runs),
        'invalid_runs': len(invalid_runs),
        'source_commit': args.source_commit,
    }
    _atomic_text(staging / 'stage1_integrity_report.json', json.dumps(integrity, indent=2))
    _atomic_text(staging / 'stage1_72config_metrics.json', json.dumps(config_metrics, indent=2))
    _atomic_text(staging / 'stage1_safety_elimination.json', json.dumps(safety_eliminated, indent=2))
    _atomic_text(staging / 'stage1_lr_comparison.json', json.dumps(lr_comparison, indent=2))

    summary = {
        'n_configs': 72,
        'n_safety_pass': n_safety_pass,
        'n_safety_eliminated': len(safety_eliminated),
        'n_beats_lr': n_beats_lr,
        'by_candidate': {},
    }
    for c in ['V2A', 'V2B', 'V2C']:
        c_configs = [ck for ck in config_metrics if ck.startswith(c)]
        c_safe = [ck for ck in c_configs if ck not in safety_eliminated]
        c_lr = [ck for ck in c_safe if lr_comparison.get(ck, {}).get('beats_lr')]
        summary['by_candidate'][c] = {
            'total': len(c_configs),
            'safety_pass': len(c_safe),
            'beats_lr': len(c_lr),
        }
    _atomic_text(staging / 'stage1_summary.json', json.dumps(summary, indent=2))

    _atomic_text(staging / 'source_binding.json', json.dumps({
        'runs_root': str(args.runs_root),
        'lr_baseline_root': str(args.lr_baseline_root),
        'auth_seal': args.auth_seal,
        'source_commit': args.source_commit,
    }, indent=2))

    write_seal(staging)
    os.replace(staging, out)

    print(f'\nAggregation sealed: {out}')
    print(f'Seal: {sha256_file(out / "SHA256SUMS")}')
    for c in ['V2A', 'V2B', 'V2C']:
        s = summary['by_candidate'][c]
        print(f'  {c}: {s["total"]} configs, {s["safety_pass"]} safe, {s["beats_lr"]} beat LR')


if __name__ == '__main__':
    main()
