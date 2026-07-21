#!/usr/bin/env python3
"""Build V2 Stage-1 job inventory: exact list of all inner-CV training jobs."""
import argparse, hashlib, json, os, sys, uuid
from pathlib import Path

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1_20260721'

CANDIDATES = ['V2A', 'V2B', 'V2C']
W_VALUES = [16, 32, 64]
HIDDEN_VALUES = [64, 128]
DROPOUT_VALUES = [0.0, 0.1]
WD_VALUES = [1e-5, 1e-4]
OUTER_FOLDS = [0, 1, 2, 3]
INNER_FOLDS = [0, 1, 2]
SEEDS_STAGE1 = [42]
SEEDS_STAGE2 = [42, 123, 456]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--output-base', type=Path,
                    default=OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_RUNS_V1_20260721')
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')

    jobs_stage1 = []
    jobs_stage2 = []

    for candidate in CANDIDATES:
        for W in W_VALUES:
            for hidden_dim in HIDDEN_VALUES:
                for dropout in DROPOUT_VALUES:
                    for wd in WD_VALUES:
                        for outer in OUTER_FOLDS:
                            for inner in INNER_FOLDS:
                                for seed in SEEDS_STAGE1:
                                    label = f'{candidate}_W{W}_H{hidden_dim}_D{dropout}_WD{wd}_o{outer}_i{inner}_s{seed}'
                                    jobs_stage1.append({
                                        'label': label,
                                        'candidate': candidate,
                                        'W': W, 'hidden_dim': hidden_dim,
                                        'dropout': dropout, 'weight_decay': wd,
                                        'outer_fold': outer, 'inner_fold': inner,
                                        'seed': seed,
                                        'output_path': str(args.output_base / label),
                                    })

    # Stage 2: shortlist candidates × all seeds
    for candidate in CANDIDATES:
        for W in W_VALUES:
            for hidden_dim in HIDDEN_VALUES:
                for dropout in DROPOUT_VALUES:
                    for wd in WD_VALUES:
                        for outer in OUTER_FOLDS:
                            for inner in INNER_FOLDS:
                                for seed in SEEDS_STAGE2:
                                    label = f'{candidate}_W{W}_H{hidden_dim}_D{dropout}_WD{wd}_o{outer}_i{inner}_s{seed}'
                                    jobs_stage2.append({
                                        'label': label,
                                        'candidate': candidate,
                                        'W': W, 'hidden_dim': hidden_dim,
                                        'dropout': dropout, 'weight_decay': wd,
                                        'outer_fold': outer, 'inner_fold': inner,
                                        'seed': seed,
                                        'output_path': str(args.output_base / label),
                                    })

    n_stage1 = len(jobs_stage1)
    n_stage2 = len(jobs_stage2)

    print(f'Stage 1: {n_stage1} jobs (seed=42 only)')
    print(f'Stage 2: {n_stage2} jobs (all seeds, worst case)')
    print(f'  Candidates: {CANDIDATES}')
    print(f'  W: {W_VALUES}')
    print(f'  Hidden: {HIDDEN_VALUES}')
    print(f'  Dropout: {DROPOUT_VALUES}')
    print(f'  WD: {WD_VALUES}')
    print(f'  Outer folds: {OUTER_FOLDS}')
    print(f'  Inner folds: {INNER_FOLDS}')
    print(f'  Compute: {n_stage1} × 3 = {n_stage1 * 3} (3 candidates)')

    # Dedup check
    paths = [j['output_path'] for j in jobs_stage1]
    if len(paths) != len(set(paths)):
        dup = len(paths) - len(set(paths))
        raise SystemExit(f'DUPLICATE OUTPUT PATHS: {dup}')

    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    inventory = {
        'schema': 'DETECTOR_V5_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1',
        'stage1_job_count': n_stage1,
        'stage2_job_count': n_stage2,
        'jobs': jobs_stage1,
        'jobs_stage2_template': jobs_stage2,
        'hyperparameter_grid': {
            'candidates': CANDIDATES, 'W': W_VALUES, 'hidden_dim': HIDDEN_VALUES,
            'dropout': DROPOUT_VALUES, 'weight_decay': WD_VALUES,
            'outer_folds': OUTER_FOLDS, 'inner_folds': INNER_FOLDS,
            'seeds_stage1': SEEDS_STAGE1, 'seeds_stage2': SEEDS_STAGE2,
        },
        'compute_estimate': {
            'stage1_total_jobs': n_stage1,
            'estimated_minutes_per_job': 'TBD_after_formal_canary',
            'estimated_gpu_hours': 'TBD',
            'note': 'Re-measure after fixes: train ≈400 eps (2/3 of outer_train)',
        },
    }

    _atomic_text(staging / 'v2_stage1_job_inventory.json', json.dumps(inventory, indent=2))

    # CSV
    csv_lines = ['label,candidate,W,hidden_dim,dropout,weight_decay,outer_fold,inner_fold,seed,output_path']
    for j in jobs_stage1:
        csv_lines.append(f'{j["label"]},{j["candidate"]},{j["W"]},{j["hidden_dim"]},{j["dropout"]},{j["weight_decay"]},{j["outer_fold"]},{j["inner_fold"]},{j["seed"]},{j["output_path"]}')
    _atomic_text(staging / 'v2_stage1_job_inventory.csv', '\n'.join(csv_lines) + '\n')

    _atomic_text(staging / 'source_binding.json', json.dumps({
        'builder_sha': sha256_file(Path(__file__)),
    }, indent=2))

    write_seal(staging)
    os.replace(staging, out)

    print(f'\nSealed: {out}')
    print(f'Seal: {sha256_file(out / "SHA256SUMS")}')


if __name__ == '__main__':
    main()
