#!/usr/bin/env python3
"""Aggregate 864 Stage-1 runs into 72 configuration records.

Fail-closed: missing/invalid runs → HOLD.
Seal verification, 72×12 closure, safety elimination, same-split LR, shortlist.
"""
import argparse, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
DEFAULT_RUNS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_RUNS_V1_20260721'
DEFAULT_LR = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_LR_BASELINE_V1_20260721'
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_AGGREGATION_V1_20260721'

CANDIDATES = ['V2A', 'V2B', 'V2C']
W_VALS = [16, 32, 64]; H_VALS = [64, 128]; D_VALS = [0.0, 0.1]; WD_VALS = [1e-5, 1e-4]
OUTER = [0, 1, 2, 3]; INNER = [0, 1, 2]; SEEDS = [42]
SAFETY = {'background_false_emit_rate': 0.10, 'release_overlap_emit_rate': 0.05,
          'unsupported_route_emit_rate': 0.0}


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


def verify_sealed_directory(root):
    s = root / 'SHA256SUMS'; c = root / 'SHA256SUMS.sha256'
    if not s.is_file() or not c.is_file():
        raise ValueError(f'SEAL MISSING: {root}')
    if c.read_text().strip() != f'{sha256_file(s)}  SHA256SUMS':
        raise ValueError(f'SEAL MISMATCH: {root}')
    for l in s.read_text().splitlines():
        d, _, n = l.partition('  '); t = root / n
        if not t.is_file() or sha256_file(t) != d:
            raise ValueError(f'FILE MISMATCH: {root}/{n}')


def make_label(c, W, H, D, WD, o, i, s):
    return f'{c}_W{W}_H{H}_D{D}_WD{WD}_o{o}_i{i}_s{s}'


def make_config_key(c, W, H, D, WD):
    return f'{c}_W{W}_H{H}_D{D}_WD{WD}'


def load_eval_metrics(runs_root, label):
    ef = runs_root / f'eval_{label}.json'
    if not ef.is_file():
        return None
    data = json.loads(ef.read_text())
    keys = list(data.get('per_run', {}).keys())
    return data['per_run'][keys[0]] if keys else None


def check_run(runs_root, label, auth_seal, source_commit):
    """Full integrity check. Returns (ok, issues, metadata)."""
    issues = []
    train_dir = runs_root / label
    pred_dir = runs_root / f'predict_{label}'

    # Training seal
    try:
        verify_sealed_directory(train_dir)
    except Exception as e:
        issues.append(f'TRAIN_SEAL: {e}')
        return False, issues, {}, None

    try:
        verify_sealed_directory(pred_dir)
    except Exception as e:
        issues.append(f'PRED_SEAL: {e}')
        return False, issues, {}, None

    # Metadata
    sb = json.loads((train_dir / 'source_binding.json').read_text())
    ar = json.loads((train_dir / 'authorization_receipt.json').read_text())
    rc = json.loads((train_dir / 'run_config.json').read_text())

    if sb.get('source_commit', '') != source_commit:
        issues.append(f'COMMIT: {sb.get("source_commit", "?")[:8]} != {source_commit[:8]}')
    if ar.get('authorization_seal') != auth_seal:
        issues.append('AUTH_SEAL_MISMATCH')

    meta = {'source_commit': sb.get('source_commit', ''), 'candidate': sb.get('candidate', ''),
            'outer_fold': sb.get('outer_fold'), 'inner_fold': sb.get('inner_fold'),
            'seed': sb.get('seed'), 'W': rc.get('receptive_field'), 'H': rc.get('hidden_dim'),
            'D': rc.get('dropout'), 'WD': rc.get('weight_decay'),
            'param_count': rc.get('parameter_count', 0)}

    # Evaluation
    metrics = load_eval_metrics(runs_root, label)
    if metrics is None:
        issues.append('EVAL_MISSING')

    # Audit
    af = runs_root / f'audit_{label}.json'
    if not af.is_file():
        issues.append('AUDIT_MISSING')
    else:
        audit = json.loads(af.read_text())
        if audit.get('status') != 'PASS':
            issues.append(f'AUDIT_{audit.get("status")}')

    return len(issues) == 0, issues, meta, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--lr-root', type=Path, default=DEFAULT_LR)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--auth-seal', type=str, required=True)
    ap.add_argument('--source-commit', type=str, required=True)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')

    # ── Collect all 864 runs ──
    missing, invalid, complete = [], [], []
    config_runs = defaultdict(dict)
    all_commits = set()

    for c in CANDIDATES:
        for W in W_VALS:
            for H in H_VALS:
                for D in D_VALS:
                    for WD in WD_VALS:
                        for o in OUTER:
                            for i in INNER:
                                for s in SEEDS:
                                    label = make_label(c, W, H, D, WD, o, i, s)
                                    if not (args.runs_root / label).exists():
                                        missing.append(label)
                                        continue
                                    ok, issues, meta, metrics = check_run(
                                        args.runs_root, label, args.auth_seal, args.source_commit)
                                    if not ok:
                                        invalid.append({'label': label, 'issues': issues})
                                    else:
                                        complete.append(label)
                                        all_commits.add(meta.get('source_commit', ''))
                                    config_runs[make_config_key(c, W, H, D, WD)][(o, i)] = {
                                        'label': label, 'ok': ok, 'issues': issues,
                                        'meta': meta, 'metrics': metrics}

    n_total = 864
    n_complete = len(complete)
    n_invalid = len(invalid)
    n_missing = len(missing)
    is_full = (n_complete == 864)

    print(f'Complete: {n_complete}/{n_total}')
    print(f'Missing: {n_missing}  Invalid: {n_invalid}')
    print(f'Commits: {all_commits}')

    if not is_full:
        partial = {'status': 'INCOMPLETE_NOT_FOR_SELECTION', 'complete': n_complete,
                   'missing': n_missing, 'invalid': n_invalid, 'shortlist_valid': False}
        staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
        staging.mkdir(parents=True)
        _atomic_text(staging / 'integrity_report.json', json.dumps(partial, indent=2))
        write_seal(staging); os.replace(staging, out)
        sys.exit(1 if n_missing + n_invalid > 0 else 0)

    # ── Verify 72×12 closure ──
    config_issues = []
    for ck in sorted(config_runs):
        runs = config_runs[ck]
        if len(runs) != 12:
            config_issues.append(f'{ck}: {len(runs)}/12 splits')
        for (o, i), r in runs.items():
            if not r['ok']:
                config_issues.append(f'{ck} o{o}i{i}: {r["issues"]}')

    if config_issues:
        print(f'Config issues: {len(config_issues)}')
        for ci in config_issues[:10]:
            print(f'  {ci}')
        raise SystemExit('HOLD: config closure failed')

    # ── Per-config aggregation ──
    lr_data = json.loads((args.lr_root / 'pooled_metrics.json').read_text())
    lr_splits = json.loads((args.lr_root / 'per_split_metrics.json').read_text())

    config_metrics, safety_elim, lr_comp = {}, {}, {}
    metric_keys = ['release_auroc', 'release_auprc', 'release_short_auprc',
                   'release_recall_05', 'background_false_emit_rate',
                   'release_overlap_emit_rate', 'unsupported_route_emit_rate',
                   'release_first_recall_05', 'release_later_recall_05']

    for ck in sorted(config_runs):
        runs = config_runs[ck]
        mlist = [r['metrics'] for r in runs.values() if r['metrics'] is not None]
        if len(mlist) != 12:
            safety_elim[ck] = {'reason': 'INCOMPLETE_SPLITS', 'n': len(mlist)}
            continue

        agg = {}
        metric_fail = False
        for mk in metric_keys:
            vals = [m[mk] for m in mlist if m.get(mk) is not None]
            if len(vals) != 12:
                if mk in SAFETY:
                    safety_elim[ck] = {'reason': f'MISSING_SAFETY: {mk} has {len(vals)}/12 values'}
                    metric_fail = True
                    break
                else:
                    safety_elim[ck] = {'reason': f'MISSING_METRIC: {mk} has {len(vals)}/12 values'}
                    metric_fail = True
                    break
            agg[f'{mk}_mean'] = mean(vals)
            agg[f'{mk}_stdev'] = stdev(vals) if len(vals) > 1 else 0
            agg[f'{mk}_per_split'] = vals
            if mk in SAFETY:
                agg[f'{mk}_worst'] = max(vals)
        if metric_fail:
            continue

        if ck in safety_elim:
            continue

        agg['n_runs'] = 12
        agg['param_count'] = list(runs.values())[0]['meta'].get('param_count', 0)

        # Safety elimination (worst split)
        safety_fail = []
        for sk, limit in SAFETY.items():
            worst = agg.get(f'{sk}_worst')
            if worst is not None and worst > limit:
                safety_fail.append(f'{sk}={worst:.4f} > {limit}')
        if safety_fail:
            safety_elim[ck] = {'reason': 'SAFETY', 'failures': safety_fail}
            config_metrics[ck] = agg
            continue

        agg['safety_pass'] = True
        config_metrics[ck] = agg

        # Same-split LR comparison
        v2_aurocs = agg.get('release_auroc_per_split', [])
        v2_auprcs = agg.get('release_auprc_per_split', [])
        lr_aurocs = [lr_splits.get(f'o{o}i{i}', {}).get('release_auroc') for (o, i) in runs if lr_splits.get(f'o{o}i{i}', {}).get('release_auroc') is not None]
        lr_auprcs = [lr_splits.get(f'o{o}i{i}', {}).get('release_auprc') for (o, i) in runs if lr_splits.get(f'o{o}i{i}', {}).get('release_auprc') is not None]
        v2_mean_auroc = agg.get('release_auroc_mean', 0)
        v2_mean_auprc = agg.get('release_auprc_mean', 0)
        lr_mean_auroc = mean(lr_aurocs) if lr_aurocs else 0
        lr_mean_auprc = mean(lr_auprcs) if lr_auprcs else 0
        lr_comp[ck] = {'v2_auroc': v2_mean_auroc, 'lr_auroc': lr_mean_auroc,
                       'v2_auprc': v2_mean_auprc, 'lr_auprc': lr_mean_auprc,
                       'v2_auroc_splits': v2_aurocs, 'lr_auroc_splits': lr_aurocs,
                       'beats_lr': v2_mean_auroc > lr_mean_auroc and v2_mean_auprc > lr_mean_auprc}

    n_safe = len(config_metrics) - len(safety_elim)
    n_lr = sum(1 for v in lr_comp.values() if v['beats_lr'])
    print(f'\nSafety pass: {n_safe}/{len(config_runs)}')
    print(f'Beats LR: {n_lr}/{n_safe}')
    print(f'Eliminated: {len(safety_elim)}')

    # ── Lexicographic selection ──
    # Build candidate metrics input for selection tool
    selection_input = {}
    for ck, agg in config_metrics.items():
        if ck in safety_elim:
            continue
        if not lr_comp.get(ck, {}).get('beats_lr'):
            continue
        selection_input[ck] = {
            'release_auprc': agg.get('release_auprc_mean'),
            'release_auprc_per_split': agg.get('release_auprc_per_split'),
            'release_short_auprc': agg.get('release_short_auprc_mean'),
            'release_short_auprc_per_split': agg.get('release_short_auprc_per_split'),
            'first_later_recall_gap': abs((agg.get('release_first_recall_05_mean', 0) or 0) -
                                          (agg.get('release_later_recall_05_mean', 0) or 0)),
            'parameter_count': agg.get('param_count', 0),
            'background_false_emit_rate': agg.get('background_false_emit_rate_worst'),
            'release_overlap_emit_rate': agg.get('release_overlap_emit_rate_worst'),
            'unsupported_route_emit_rate': agg.get('unsupported_route_emit_rate_worst'),
            'release_auroc': agg.get('release_auroc_mean'),
        }

    # Run simple lexicographic: safety→auprc→short→gap→params
    def lexicographic_select(candidates):
        remaining = dict(candidates)
        trace = []
        # Priority 1: Safety (already filtered)
        trace.append({'priority': 'safety', 'n': len(remaining)})
        # Priority 2: Release AUPRC
        best_auprc = max(v['release_auprc'] for v in remaining.values())
        remaining = {k: v for k, v in remaining.items() if v['release_auprc'] >= best_auprc - 0.005}
        trace.append({'priority': 'release_auprc', 'best': best_auprc, 'n': len(remaining)})
        if len(remaining) > 1:
            # Priority 3: Short-event AUPRC
            best_short = max(v['release_short_auprc'] for v in remaining.values())
            remaining = {k: v for k, v in remaining.items() if v['release_short_auprc'] >= best_short - 0.005}
            trace.append({'priority': 'short_auprc', 'best': best_short, 'n': len(remaining)})
        if len(remaining) > 1:
            # Priority 4: First/later gap
            best_gap = min(v['first_later_recall_gap'] for v in remaining.values())
            remaining = {k: v for k, v in remaining.items() if v['first_later_recall_gap'] <= best_gap + 0.02}
            trace.append({'priority': 'gap', 'best': best_gap, 'n': len(remaining)})
        if len(remaining) > 1:
            # Priority 5: Parameter count
            best_params = min(v['parameter_count'] for v in remaining.values())
            remaining = {k: v for k, v in remaining.items() if v['parameter_count'] == best_params}
            trace.append({'priority': 'params', 'best': best_params, 'n': len(remaining)})
        selected = list(remaining.keys())[0] if remaining else None
        return selected, trace

    shortlist_candidates = {k: v for k, v in selection_input.items()}
    selected_config, selection_trace = lexicographic_select(shortlist_candidates)

    print(f'\nSelected: {selected_config}')
    for t in selection_trace:
        print(f'  {t}')

    # ── Write outputs ──
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    _atomic_text(staging / 'stage1_integrity_report.json', json.dumps({
        'status': 'PASS' if is_full else 'HOLD', 'complete': n_complete,
        'total': n_total, 'missing': n_missing, 'invalid': n_invalid,
        'source_commit': args.source_commit, 'commits_found': list(all_commits),
    }, indent=2))

    _atomic_text(staging / 'stage1_72config_metrics.json', json.dumps(config_metrics, indent=2))
    _atomic_text(staging / 'stage1_safety_elimination.json', json.dumps(safety_elim, indent=2))
    _atomic_text(staging / 'stage1_lr_comparison.json', json.dumps(lr_comp, indent=2))

    summary = {'n_configs': 72, 'n_safety_pass': n_safe, 'n_eliminated': len(safety_elim),
               'n_beats_lr': n_lr, 'by_candidate': {}}
    for c in CANDIDATES:
        cc = [ck for ck in config_metrics if ck.startswith(c)]
        cs = [ck for ck in cc if ck not in safety_elim]
        cl = [ck for ck in cs if lr_comp.get(ck, {}).get('beats_lr')]
        summary['by_candidate'][c] = {'total': len(cc), 'safety_pass': len(cs), 'beats_lr': len(cl)}
    shortlist = {
        'selected': selected_config,
        'selection_trace': selection_trace,
        'candidates_considered': len(shortlist_candidates),
        'shortlist': [selected_config] if selected_config else [],
        'shortlist_valid': selected_config is not None,
        'eligible_for_stage2': selected_config is not None,
    }
    _atomic_text(staging / 'stage1_shortlist.json', json.dumps(shortlist, indent=2))
    _atomic_text(staging / 'stage1_selection_trace.json', json.dumps(selection_trace, indent=2))
    _atomic_text(staging / 'stage1_summary.json', json.dumps(summary, indent=2))

    _atomic_text(staging / 'source_binding.json', json.dumps({
        'runs_root': str(args.runs_root), 'lr_root': str(args.lr_root),
        'auth_seal': args.auth_seal, 'source_commit': args.source_commit,
    }, indent=2))

    write_seal(staging); os.replace(staging, out)
    print(f'\nSealed: {out}')
    for c in CANDIDATES:
        s = summary['by_candidate'][c]
        print(f'  {c}: {s["total"]} total, {s["safety_pass"]} safe, {s["beats_lr"]} beat LR')


if __name__ == '__main__':
    main()
