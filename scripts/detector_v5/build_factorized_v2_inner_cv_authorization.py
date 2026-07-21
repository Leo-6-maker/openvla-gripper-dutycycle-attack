#!/usr/bin/env python3
"""Build V2 inner-CV authorization: binds all SHAs, protocol, splits, and constraints."""
import argparse, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')


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


def verify_seal(root):
    s = root / 'SHA256SUMS'; c = root / 'SHA256SUMS.sha256'
    if not s.is_file() or not c.is_file():
        raise SystemExit(f'SEAL MISSING: {root}')
    if c.read_text().strip() != f'{sha256_file(s)}  SHA256SUMS':
        raise SystemExit(f'SEAL MISMATCH: {root}')
    for l in s.read_text().splitlines():
        d, _, n = l.partition('  '); t = root / n
        if not t.is_file() or sha256_file(t) != d:
            raise SystemExit(f'FILE MISMATCH: {root}/{n}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--splits-root', type=Path,
                    default=OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721')
    ap.add_argument('--teacher-root', type=Path,
                    default=OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721')
    ap.add_argument('--s1-root', type=Path,
                    default=OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f')
    ap.add_argument('--fold-root', type=Path,
                    default=OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f')
    ap.add_argument('--job-inventory-root', type=Path,
                    default=OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1_20260721')
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')

    # Verify input seals
    splits_seal = sha256_file(args.splits_root / 'SHA256SUMS')
    teacher_seal = sha256_file(args.teacher_root / 'SHA256SUMS')
    s1_seal = sha256_file(args.s1_root / 'SHA256SUMS')
    fold_seal = sha256_file(args.fold_root / 'SHA256SUMS')
    verify_seal(args.splits_root)
    verify_seal(args.teacher_root)
    verify_seal(args.s1_root)
    verify_seal(args.fold_root)

    if not args.job_inventory_root.exists():
        raise SystemExit(f'Job inventory not found: {args.job_inventory_root}')
    verify_seal(args.job_inventory_root)
    inv_seal = sha256_file(args.job_inventory_root / 'SHA256SUMS')
    inv_data = json.loads((args.job_inventory_root / 'v2_stage1_job_inventory.json').read_text())

    # Verify inventory: exactly 864 jobs, all unique, all seeds=42
    jobs = inv_data['jobs']
    if len(jobs) != 864:
        raise SystemExit(f'Expected 864 jobs, got {len(jobs)}')
    labels = [j['label'] for j in jobs]
    paths = [j['output_path'] for j in jobs]
    if len(labels) != len(set(labels)):
        raise SystemExit(f'Duplicate labels in inventory')
    if len(paths) != len(set(paths)):
        raise SystemExit(f'Duplicate output paths in inventory')
    non_42 = [j for j in jobs if j['seed'] != 42]
    if non_42:
        raise SystemExit(f'{len(non_42)} jobs with seed != 42')
    print(f'Inventory verified: {len(jobs)} jobs, all unique, all seed=42')

    # Source SHAs
    src_dir = ROOT / 'src/gripper_attack'
    script_dir = ROOT / 'scripts/detector_v5'

    shas = {
        'protocol_sha': sha256_file(ROOT / 'configs/DETECTOR_V5_FACTORIZED_STUDENT_V2_DEVELOPMENT_PROTOCOL_V1.json'),
        'errata_sha': sha256_file(ROOT / 'configs/DETECTOR_V5_FACTORIZED_STUDENT_V2_DEVELOPMENT_PROTOCOL_V1_ERRATA_1.json'),
        'dataset_v1_sha': sha256_file(src_dir / 'v5_factorized_dataset.py'),
        'model_v2_sha': sha256_file(src_dir / 'v5_factorized_student_v2.py'),
        'loss_v2_sha': sha256_file(src_dir / 'v5_factorized_loss_v2.py'),
        'splits_resolver_sha': sha256_file(src_dir / 'v5_factorized_v2_splits.py'),
        'trainer_sha': sha256_file(script_dir / 'train_factorized_v2_inner_cv.py'),
        'predictor_sha': sha256_file(script_dir / 'predict_factorized_v2_inner_cv.py'),
        'evaluator_sha': sha256_file(script_dir / 'evaluate_factorized_v2_inner_cv.py'),
        'auditor_sha': sha256_file(script_dir / 'audit_factorized_v2_inner_cv_predictions.py'),
        'lr_runner_sha': sha256_file(script_dir / 'run_factorized_v2_lr_baseline.py'),
        'selection_sha': sha256_file(script_dir / 'select_factorized_v2_candidate.py'),
        'launcher_sha': sha256_file(script_dir / 'launch_factorized_v2_inner_cv.py'),
    }

    # Git HEAD
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    if len(head) != 40:
        raise SystemExit(f'Invalid git HEAD: {head}')

    import platform as _platform, torch as _torch, sklearn as _sklearn
    auth = {
        'schema': 'DETECTOR_V5_FACTORIZED_STUDENT_V2_STAGE1_AUTHORIZATION_V1',
        'status': 'V2_STAGE1_INNER_CV_AUTHORIZED',
        'v2_inner_cv_authorized': True,
        'stage2_authorized': False,
        'engineering_oof_authorized': False,
        'full_fit_authorized': False,
        'cal_authorized': False,
        'check_authorized': False,
        'vis_authorized': False,
        'attack_authorized': False,
        'source_commit': head,
        'environment': {
            'python_version': _platform.python_version(),
            'torch_version': _torch.__version__,
            'cuda_version': _torch.version.cuda if _torch.cuda.is_available() else None,
            'sklearn_version': _sklearn.__version__,
            'host': _platform.node(),
        },
        'splits_root': str(args.splits_root),
        'splits_root_seal': splits_seal,
        'teacher_root': str(args.teacher_root),
        'teacher_root_seal': teacher_seal,
        's1_root': str(args.s1_root),
        's1_root_seal': s1_seal,
        'fold_root': str(args.fold_root),
        'fold_root_seal': fold_seal,
        'job_inventory_root': str(args.job_inventory_root),
        'job_inventory_seal': inv_seal,
        **shas,
        'hyperparameter_grid': {
            'candidates': ['V2A', 'V2B', 'V2C'],
            'W': [16, 32, 64],
            'hidden_dim': [64, 128],
            'dropout': [0.0, 0.1],
            'weight_decay': [1e-5, 1e-4],
            'seeds': [42, 123, 456],
            'epochs': 30,
            'batch_size': 8,
            'lr': 0.001,
        },
        'stage1_job_count': 864,
    }

    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)
    _atomic_text(staging / 'authorization.json', json.dumps(auth, indent=2, sort_keys=True) + '\n')
    write_seal(staging)
    os.replace(staging, out)
    print(json.dumps({'status': 'AUTHORIZATION_SEALED', 'commit': head, 'root': str(out)}, indent=2))


if __name__ == '__main__':
    main()
