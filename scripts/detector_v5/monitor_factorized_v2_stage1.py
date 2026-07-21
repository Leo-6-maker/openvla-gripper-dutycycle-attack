#!/usr/bin/env python3
"""Read-only Stage-1 progress ledger. Scans output directories, never modifies artifacts.

Reports per-job status: TRAINING | TRAIN_SEALED | PREDICT_SEALED | EVALUATED | AUDIT_PASS | COMPLETE | FAILED
Breaks down by candidate (V2A/V2B/V2C) for accurate progress tracking.
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
DEFAULT_RUNS = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_RUNS_V1_20260721'
DEFAULT_INVENTORY = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_JOB_INVENTORY_V1_20260721'
LOG_DIR = Path('/mnt/sdc/dty_user/openvla_attack/logs/v2_stage1')


def check_job_status(label, output_base):
    """Determine status of one job. Returns (status, details)."""
    train_dir = output_base / label
    pred_dir = output_base / f'predict_{label}'
    eval_file = output_base / f'eval_{label}.json'
    audit_file = output_base / f'audit_{label}.json'

    details = {}

    # Check training
    if not train_dir.exists():
        return 'QUEUED', details

    sha_file = train_dir / 'SHA256SUMS'
    if sha_file.is_file():
        try:
            rc = json.loads((train_dir / 'run_config.json').read_text())
            ar = json.loads((train_dir / 'authorization_receipt.json').read_text())
            details['train_seal'] = 'OK'
            details['source_commit'] = rc.get('source_commit', '?')[:8]
            details['auth_seal'] = ar.get('authorization_seal', '?')[:16]
        except Exception:
            return 'TRAINING', details
    else:
        # Check if staging exists (actively training)
        staging = list(train_dir.parent.glob(f'.{train_dir.name}.*.staging'))
        if staging:
            return 'TRAINING', details
        return 'QUEUED', details

    # Check prediction
    if not pred_dir.exists() or not (pred_dir / 'SHA256SUMS').is_file():
        return 'TRAIN_SEALED', details

    details['pred_seal'] = 'OK'
    try:
        manifest = json.loads((pred_dir / 'prediction_manifest.json').read_text())
        details['pred_steps'] = manifest.get('total_steps', 0)
    except Exception:
        pass

    # Check evaluation
    if not eval_file.is_file():
        return 'PREDICT_SEALED', details

    try:
        eval_data = json.loads(eval_file.read_text())
        if 'per_run' in eval_data:
            metrics = list(eval_data['per_run'].values())[0]
            details['release_auroc'] = round(metrics.get('release_auroc', 0), 4)
            details['bg_emit'] = round(metrics.get('background_false_emit_rate', 0), 4)
    except Exception:
        return 'PREDICT_SEALED', details

    # Check audit
    if not audit_file.is_file():
        return 'EVALUATED', details

    try:
        audit = json.loads(audit_file.read_text())
        if audit.get('status') == 'PASS':
            details['audit'] = 'PASS'
            return 'COMPLETE', details
        else:
            details['audit'] = 'HOLD'
            return 'AUDIT_HOLD', details
    except Exception:
        return 'EVALUATED', details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=DEFAULT_RUNS)
    ap.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=60)
    args = ap.parse_args()

    def generate_ledger():
        inv = json.loads((args.inventory / 'v2_stage1_job_inventory.json').read_text())
        jobs = inv['jobs']

        statuses = defaultdict(lambda: defaultdict(int))
        failed_jobs = []
        complete_jobs = []

        for job in jobs:
            label = job['label']
            candidate = job['candidate']
            status, details = check_job_status(label, args.runs_root)
            statuses[candidate][status] += 1
            statuses['ALL'][status] += 1

            if status == 'FAILED' or status == 'AUDIT_HOLD':
                failed_jobs.append({'label': label, 'status': status, 'details': details})
            elif status == 'COMPLETE':
                complete_jobs.append({'label': label, 'candidate': candidate, 'details': details})

        total = len(jobs)
        complete = statuses['ALL'].get('COMPLETE', 0)
        complete_chain = (statuses['ALL'].get('COMPLETE', 0) +
                          statuses['ALL'].get('AUDIT_HOLD', 0))

        ledger = {
            'total_jobs': total,
            'complete': complete,
            'complete_or_audit_hold': complete_chain,
            'by_status': dict(statuses['ALL']),
            'by_candidate': {c: dict(st) for c, st in statuses.items() if c != 'ALL'},
            'failed': failed_jobs[:20],
            'complete_samples': complete_jobs[:10],
        }
        return ledger

    if args.watch:
        print('Watching grid progress (Ctrl+C to stop)...')
        try:
            while True:
                ledger = generate_ledger()
                complete = ledger['complete']
                total = ledger['total_jobs']
                pct = 100 * complete / max(1, total)
                print(f'\n=== {time.strftime("%H:%M:%S")} | {complete}/{total} ({pct:.1f}%) ===')
                for candidate in ['V2A', 'V2B', 'V2C']:
                    cst = ledger['by_candidate'].get(candidate, {})
                    c_complete = cst.get('COMPLETE', 0)
                    print(f'  {candidate}: {c_complete} complete | {dict(cst)}')
                failed = ledger['failed']
                if failed:
                    print(f'  FAILED: {len(failed)}')
                    for f in failed[:3]:
                        print(f'    {f["label"]}: {f["status"]}')
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    else:
        ledger = generate_ledger()
        if args.output:
            args.output.write_text(json.dumps(ledger, indent=2) + '\n')
        complete = ledger['complete']
        total = ledger['total_jobs']
        print(f'{complete}/{total} complete ({100*complete/max(1,total):.1f}%)')
        for candidate in ['V2A', 'V2B', 'V2C']:
            cst = ledger['by_candidate'].get(candidate, {})
            print(f'  {candidate}: {dict(cst)}')


if __name__ == '__main__':
    main()
