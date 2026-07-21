#!/usr/bin/env python3
"""Train one V2 candidate on one inner-CV fold×seed.

Loads V2 inner-CV splits, trains on inner train, validates on inner val.
Supports V2A (tcn+step_bce), V2B (tcn+event_loss), V2C (windowed_gru+event_loss).
"""
import argparse, csv, hashlib, json, os, subprocess, sys, uuid, platform
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes, compute_factorized_normalization,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.v5_factorized_student_v2 import FactorizedStudentV2
from gripper_attack.v5_factorized_loss_v2 import FactorizedLossV2A, FactorizedLossV2B
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
FOLD_ROOT = OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f'
POLICY_INTENT = OPS / 'OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_01'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'

DURATION_BUCKETS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 99999)]


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


def compute_identity_weights(eps_list, route):
    """Compute per-episode identity weights so each identity contributes equally."""
    route_eps = [ep for ep in eps_list if ep.mechanism_route == route]
    if not route_eps:
        return {}
    id_to_eps = defaultdict(list)
    for ep in route_eps:
        id_to_eps[ep.canonical_parent_key].append(ep)
    n_ids = len(id_to_eps)
    weights = {}
    for ident, eps in id_to_eps.items():
        w = 1.0 / max(1, len(eps))  # each episode from this identity gets 1/n_eps weight
        for ep in eps:
            weights[id(ep)] = w
    # Normalize so mean weight = 1
    if weights:
        mean_w = sum(weights.values()) / len(weights)
        weights = {k: v / mean_w for k, v in weights.items()}
    return weights


def compute_class_weights(eps_list):
    """Per-route per-head positive/negative class weights (inner-train only)."""
    counts = defaultdict(lambda: defaultdict(lambda: {"pos": 0, "neg": 0}))
    for ep in eps_list:
        r = ep.mechanism_route
        if r not in SUPPORTED_ROUTES:
            continue
        eids = ep.event_id
        for eid in eids.unique().tolist():
            em = eids == eid
            for head, tgt, km in [("grasp", ep.grasp_target, ep.grasp_known_mask),
                                   ("manipulation", ep.manipulation_target, ep.manipulation_known_mask),
                                   ("release", ep.release_target, ep.release_known_mask)]:
                if km[em].any():
                    counts[r][head]["pos"] += int(tgt[em].any())
                    counts[r][head]["neg"] += int(not tgt[em].any() and km[em].any())
    weights = {}
    for r, heads in counts.items():
        weights[r] = {}
        for h, c in heads.items():
            pos, neg = c["pos"], c["neg"]
            w_pos = (pos + neg) / max(1, 2 * pos)
            w_neg = (pos + neg) / max(1, 2 * neg)
            weights[r][h] = {"pos_weight": round(w_pos, 4), "neg_weight": round(w_neg, 4),
                             "pos_count": pos, "neg_count": neg}
    return weights


def build_route_balanced_batches(eps_list, batch_size=8, rng_seed=42):
    """Route-balanced batches with identity resampling (p=0.5)."""
    rng = __import__('random').Random(rng_seed)
    single_eps = [e for e in eps_list if e.mechanism_route == 'single_object_pick_place']
    multi_eps = [e for e in eps_list if e.mechanism_route == 'multi_object_transfer']

    def make_batches(eps, route):
        batches = []
        for i in range(0, len(eps), batch_size):
            batch = eps[i:i+batch_size]
            # Identity resampling: with p=0.5, replace each episode
            resampled = []
            for ep in batch:
                if rng.random() < 0.5:
                    pool = [e for e in (single_eps if route == 'single_object_pick_place' else multi_eps)
                           if e.canonical_parent_key != ep.canonical_parent_key]
                    if pool:
                        resampled.append(rng.choice(pool))
                    else:
                        resampled.append(ep)
                else:
                    resampled.append(ep)
            batches.append((route, resampled))
        return batches

    single_batches = make_batches(single_eps, 'single_object_pick_place')
    multi_batches = make_batches(multi_eps, 'multi_object_transfer')

    N = max(len(single_batches), len(multi_batches))
    if len(single_batches) < N:
        single_batches = single_batches + [single_batches[i % len(single_batches)] for i in range(N - len(single_batches))]
    if len(multi_batches) < N:
        multi_batches = multi_batches + [multi_batches[i % len(multi_batches)] for i in range(N - len(multi_batches))]
    balanced = []
    for i in range(N):
        balanced.append(single_batches[i])
        balanced.append(multi_batches[i])
    rng.shuffle(balanced)
    return balanced


def apply_temporal_jitter(ep, jitter, rng):
    """Past-only temporal jitter: drop last j steps from input, keep target at t."""
    if jitter <= 0:
        return ep
    # Jitter shifts the input window left by j steps (removing j most recent steps)
    # The target labels stay at original positions — model must predict t using info up to t-j
    # We implement this by zeroing out the last j valid steps in the input
    # (the model will use earlier context from the causal window)
    # For simplicity: we don't modify the episode data, the TCN's causal padding
    # combined with shifting the features right by j steps handles this naturally.
    # Implementation: create a shifted version where features[t] = features[t-j]
    T = len(ep.features_25d)
    if jitter >= T:
        return ep
    shifted_25d = torch.cat([ep.features_25d[:1].repeat(jitter, 1), ep.features_25d[:-jitter]], dim=0)
    shifted_valid = torch.cat([torch.zeros(jitter, dtype=torch.bool), ep.valid_mask[:-jitter]], dim=0)
    # Targets unchanged
    from dataclasses import replace
    return replace(ep, features_25d=shifted_25d, valid_mask=shifted_valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', choices=['V2A', 'V2B', 'V2C'], required=True)
    ap.add_argument('--outer-fold', type=int, required=True)
    ap.add_argument('--inner-fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--receptive-field', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--inner-cv-splits-root', type=Path, required=True)
    ap.add_argument('--authorization-root', type=Path, default=None)
    args = ap.parse_args()

    # Validate candidate
    encoder_type = 'tcn' if args.candidate in ['V2A', 'V2B'] else 'windowed_gru'
    use_event_loss = args.candidate in ['V2B', 'V2C']

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load inner-CV splits
    verify_sealed_directory(args.inner_cv_splits_root)
    splits = json.loads((args.inner_cv_splits_root / 'inner_cv_splits.json').read_text())
    fold_key = f'fold_{args.outer_fold}'
    fold_data = splits['splits'][fold_key]

    # Use shared split resolver: inner_val = specified fold, inner_train = other two
    inner_train_ids, inner_val_ids = resolve_inner_train_val_ids(
        splits, args.outer_fold, args.inner_fold)

    # Load registry and filter
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}

    train_rows = [id_to_row[i] for i in inner_train_ids if i in id_to_row]
    val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]

    # Verify source roots
    verify_factorized_source_roots(S1, TEACHER)

    # Load episodes
    train_eps = load_factorized_episodes(S1, TEACHER, train_rows)
    val_eps = load_factorized_episodes(S1, TEACHER, val_rows)

    print(f'Train: {len(train_eps)} episodes, Val: {len(val_eps)} episodes')

    # Normalization (inner train only)
    mean_25d, std_25d = compute_factorized_normalization(train_eps)

    # Class weights (inner train only)
    class_weights = compute_class_weights(train_eps)

    # Build batches
    rng_seed = args.seed + args.outer_fold * 100 + args.inner_fold * 10
    train_batches = build_route_balanced_batches(train_eps, args.batch_size, rng_seed)

    # Validation: filter by route FIRST, then batch
    single_val_eps = [e for e in val_eps if e.mechanism_route == 'single_object_pick_place']
    multi_val_eps = [e for e in val_eps if e.mechanism_route == 'multi_object_transfer']
    val_batches_single = [('single_object_pick_place', single_val_eps[i:i+args.batch_size])
                          for i in range(0, len(single_val_eps), args.batch_size)]
    val_batches_multi = [('multi_object_transfer', multi_val_eps[i:i+args.batch_size])
                         for i in range(0, len(multi_val_eps), args.batch_size)]

    # Model
    model = FactorizedStudentV2(input_dim_25d=25, hidden_dim=args.hidden_dim,
                                 receptive_field=args.receptive_field,
                                 encoder_type=encoder_type, dropout=args.dropout,
                                 use_9d=False).to(device)
    n_params = model.parameter_count()

    if use_event_loss:
        loss_fn = FactorizedLossV2B(consistency_weight=0.1)
    else:
        loss_fn = FactorizedLossV2A(consistency_weight=0.1)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    jitter_rng = __import__('random').Random(args.seed + 999)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [],
               'val_grasp': [], 'val_manipulation': [], 'val_release': []}
    sampling_audit = []

    def batch_to_device(batch_eps, route, training=False):
        B = len(batch_eps)
        max_T = max(len(ep.features_25d) for ep in batch_eps)
        x25 = torch.zeros(B, max_T, 25, device=device)
        mask25 = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        for b, ep in enumerate(batch_eps):
            # Apply temporal jitter in training
            if training:
                j = jitter_rng.randint(0, 4)
                ep_j = apply_temporal_jitter(ep, j, jitter_rng)
            else:
                ep_j = ep
            T = len(ep_j.features_25d)
            x25[b, :T] = ((ep_j.features_25d - mean_25d) / std_25d).to(device)
            mask25[b, :T] = ep_j.valid_mask.to(device)
        return x25, None, mask25, None

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        epoch_audit = {'duration_buckets': {}, 'identity_count': 0}

        for route, batch_eps in train_batches:
            x25, _, mask25, _ = batch_to_device(batch_eps, route, training=True)
            opt.zero_grad()

            # Identity weights
            id_w_map = compute_identity_weights(batch_eps, route)
            id_w = torch.tensor([id_w_map.get(id(ep), 1.0) for ep in batch_eps], device=device)

            logits = model.forward_logits(x25, None, mask25, None, route)
            cw = class_weights.get(route, {})

            if use_event_loss:
                loss, m, audit = loss_fn(logits, batch_eps, mask25, class_weights=cw, identity_weights=id_w)
                for bk, bv in audit.get('duration_buckets', {}).items():
                    epoch_audit['duration_buckets'].setdefault(bk, {'event_count': 0, 'loss_sum': 0.0})
                    epoch_audit['duration_buckets'][bk]['event_count'] += bv['event_count']
                    epoch_audit['duration_buckets'][bk]['loss_sum'] += bv['loss_sum']
            else:
                loss, m = loss_fn(logits, batch_eps, mask25, class_weights=cw, identity_weights=id_w)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        head_metrics = defaultdict(list)
        route_losses = defaultdict(list)
        with torch.no_grad():
            for route, batch_eps in (val_batches_single + val_batches_multi):
                if not batch_eps:
                    continue
                x25, _, mask25, _ = batch_to_device(batch_eps, route, training=False)
                logits = model.forward_logits(x25, None, mask25, None, route)
                cw = class_weights.get(route, {})
                if use_event_loss:
                    loss, m, _ = loss_fn(logits, batch_eps, mask25, class_weights=cw)
                else:
                    loss, m = loss_fn(logits, batch_eps, mask25, class_weights=cw)
                val_losses.append(loss.item())
                route_losses[route].append(loss.item())
                for k in ['grasp', 'manipulation', 'release']:
                    head_metrics[k].append(m.get(k, m[k]) if isinstance(m, dict) else 0)

        avg_train = sum(train_losses) / max(1, len(train_losses))
        single_m = sum(route_losses['single_object_pick_place']) / max(1, len(route_losses['single_object_pick_place']))
        multi_m = sum(route_losses['multi_object_transfer']) / max(1, len(route_losses['multi_object_transfer']))
        avg_val = (single_m + multi_m) / 2

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        for k in ['grasp', 'manipulation', 'release']:
            history[f'val_{k}'].append(sum(head_metrics[k]) / max(1, len(head_metrics[k])))

        if epoch % 5 == 0:
            print(f'  epoch {epoch:2d}: train={avg_train:.4f} val={avg_val:.4f} '
                  f'g={history["val_grasp"][-1]:.4f} m={history["val_manipulation"][-1]:.4f} '
                  f'r={history["val_release"][-1]:.4f}')

        sampling_audit.append(epoch_audit)

    # Save checkpoint
    ckpt = {'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'config': {'candidate': args.candidate, 'encoder_type': encoder_type,
                       'hidden_dim': args.hidden_dim, 'receptive_field': args.receptive_field,
                       'dropout': args.dropout, 'use_9d': False,
                       'outer_fold': args.outer_fold, 'inner_fold': args.inner_fold,
                       'seed': args.seed},
            'epoch': args.epochs}
    torch.save(ckpt, staging / 'checkpoint.pt')

    # Outputs
    env_info = {'python': sys.executable, 'python_version': platform.python_version(),
                'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda, 'host': platform.node()}

    run_config = {'candidate': args.candidate, 'outer_fold': args.outer_fold,
                  'inner_fold': args.inner_fold, 'seed': args.seed,
                  'receptive_field': args.receptive_field, 'hidden_dim': args.hidden_dim,
                  'dropout': args.dropout, 'lr': args.lr, 'weight_decay': args.weight_decay,
                  'epochs': args.epochs, 'batch_size': args.batch_size,
                  'encoder_type': encoder_type, 'use_event_loss': use_event_loss,
                  'parameter_count': n_params}

    _atomic_text(staging / 'run_config.json', json.dumps(run_config, indent=2))
    _atomic_text(staging / 'history.json', json.dumps(history, indent=2))
    _atomic_text(staging / 'normalization.json', json.dumps({
        'mean_25d': mean_25d.tolist(), 'std_25d': std_25d.tolist()}))
    _atomic_text(staging / 'class_weights.json', json.dumps(class_weights, indent=2))
    _atomic_text(staging / 'sampling_audit.json', json.dumps(sampling_audit, indent=2))
    _atomic_text(staging / 'environment.json', json.dumps(env_info, indent=2))
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'candidate': args.candidate, 'outer_fold': args.outer_fold,
        'inner_fold': args.inner_fold, 'seed': args.seed,
        'inner_cv_splits_root': str(args.inner_cv_splits_root),
        'dataset_sha': sha256_file(ROOT / 'src/gripper_attack/v5_factorized_dataset.py'),
        'model_v2_sha': sha256_file(ROOT / 'src/gripper_attack/v5_factorized_student_v2.py'),
        'loss_v2_sha': sha256_file(ROOT / 'src/gripper_attack/v5_factorized_loss_v2.py'),
        'trainer_sha': sha256_file(Path(__file__)),
        'source_commit': source_commit,
    }, indent=2))

    if args.authorization_root:
        auth_root = args.authorization_root.resolve()
        auth_seal = sha256_file(auth_root / 'SHA256SUMS')
        auth = json.loads((auth_root / 'authorization.json').read_text())
        _atomic_text(staging / 'authorization_receipt.json', json.dumps({
            'authorization_root': str(auth_root),
            'authorization_seal': auth_seal,
            'authorized_job_label': f'{args.candidate}_W{args.receptive_field}_H{args.hidden_dim}_D{args.dropout}_WD{args.weight_decay}_o{args.outer_fold}_i{args.inner_fold}_s{args.seed}',
            'inventory_seal': auth.get('job_inventory_seal', ''),
            'source_commit': source_commit,
            'status': 'AUTHORIZED_TRAINING_COMPLETE',
        }, indent=2))

    write_seal(staging)
    os.replace(staging, out)
    print(f'Sealed: {out}')


if __name__ == '__main__':
    main()
