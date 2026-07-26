"""G8: N5 Dual TCN Student — train 3 frozen seeds with G6 frozen parameters.

Architecture: Dual CausalTCN (RF32+RF128) → fusion → 5 ScalarHeads.
Uses G6 normalizer, splits, pos_weights. No candidate_close in pipeline.
"""
import json, os, sys, time, hashlib, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

from n5_dataset import N5Dataset, N5Normalizer, N5_HEAD_NAMES
from n5_student_model import (
    N5MultiHeadStudent, masked_bce_loss, SuiteBalancedSampler,
    compute_schema_sha as n5_schema_sha,
)

# ── Config ──
G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL.json'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g8_n5_training'
BATCH_SIZE = 32
N_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEEDS = [19903, 20903, 21903]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUT_ROOT, exist_ok=True)


# ── Data (same as G7) ──

class N5TensorDataset(torch.utils.data.Dataset):
    def __init__(self, seal, split_key, normalizer):
        self.episodes = []
        self.normalizer = normalizer
        dataset = N5Dataset(IDENTITY_MANIFEST, CS200_ROOT, LABEL_ROOT, split='checkpoint_training')
        split_ids = set(seal['split'][f'{split_key}_identities'])
        for idx, ident in enumerate(dataset.identities):
            if ident not in split_ids:
                continue
            try:
                ep = dataset.get_episode(idx)
                self.episodes.append(ep)
            except Exception as e:
                print(f'  WARN: Failed to load {ident}: {e}')
        print(f'  {split_key}: {len(self.episodes)} episodes')

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        features = self.normalizer.normalize(ep['features'].copy())
        labels = {}
        valid_masks = {}
        for name in N5_HEAD_NAMES:
            labels[name] = torch.tensor(
                (ep['labels'][name] > 0.5).astype(np.float32)
            )
            valid_masks[name] = torch.tensor(ep['valid_masks'][name].astype(bool))
        return {
            'features': torch.tensor(features),
            'labels': labels,
            'valid_masks': valid_masks,
            'identity': ep['identity'],
            'T': ep['T'],
        }


def collate_right_pad(batch):
    max_T = max(item['T'] for item in batch)
    D = batch[0]['features'].shape[-1]
    features = torch.zeros(len(batch), max_T, D)
    timestep_mask = torch.zeros(len(batch), max_T, dtype=torch.bool)
    labels = {name: torch.zeros(len(batch), max_T) for name in N5_HEAD_NAMES}
    valid_masks = {name: torch.zeros(len(batch), max_T, dtype=torch.bool) for name in N5_HEAD_NAMES}
    identities = []
    for i, item in enumerate(batch):
        T = item['T']
        features[i, :T] = item['features']
        timestep_mask[i, :T] = True
        for name in N5_HEAD_NAMES:
            labels[name][i, :T] = item['labels'][name]
            valid_masks[name][i, :T] = item['valid_masks'][name]
        identities.append(item['identity'])
    return {
        'features': features, 'labels': labels, 'valid_masks': valid_masks,
        'timestep_mask': timestep_mask, 'identities': identities,
    }


# ── Training ──

def compute_per_head_metrics(output, labels, valid_masks):
    metrics = {}
    for name in N5_HEAD_NAMES:
        logits = output[name]
        targets = labels[name]
        mask = valid_masks[name]
        n_valid = mask.sum().item()
        if n_valid == 0:
            metrics[name] = {'loss': 0.0, 'n_valid': 0, 'acc': 0.0, 'n_pos': 0, 'n_neg': 0}
            continue
        loss = masked_bce_loss(logits, targets, mask)
        probs = torch.sigmoid(logits[mask])
        preds = (probs > 0.5).float()
        acc = (preds == targets[mask]).float().mean().item()
        n_pos = int(targets[mask].sum().item())
        metrics[name] = {'loss': loss.item(), 'n_valid': n_valid, 'acc': acc,
                         'n_pos': n_pos, 'n_neg': n_valid - n_pos}
    return metrics


def train_epoch(model, dataloader, pos_weights, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        features = batch['features'].to(DEVICE)
        labels = {k: v.to(DEVICE) for k, v in batch['labels'].items()}
        valid_masks = {k: v.to(DEVICE) for k, v in batch['valid_masks'].items()}
        timestep_mask = batch['timestep_mask'].to(DEVICE)

        output = model(features, timestep_mask=timestep_mask)
        batch_loss = 0.0
        active = 0
        for name in N5_HEAD_NAMES:
            mask = valid_masks[name]
            if mask.sum() == 0:
                continue
            pw = pos_weights.get(name)
            loss = masked_bce_loss(output[name], labels[name], mask, pw)
            batch_loss += loss
            active += 1
        if active == 0:
            continue

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(model, dataloader, pos_weights):
    model.eval()
    agg = defaultdict(lambda: {'loss': 0.0, 'n_valid': 0, 'acc': 0.0, 'n_pos': 0, 'n_neg': 0})
    n_batches = 0
    for batch in dataloader:
        features = batch['features'].to(DEVICE)
        labels = {k: v.to(DEVICE) for k, v in batch['labels'].items()}
        valid_masks = {k: v.to(DEVICE) for k, v in batch['valid_masks'].items()}
        timestep_mask = batch['timestep_mask'].to(DEVICE)

        output = model(features, timestep_mask=timestep_mask)
        head_metrics = compute_per_head_metrics(output, labels, valid_masks)
        for name in N5_HEAD_NAMES:
            for k in ('loss', 'n_valid', 'acc', 'n_pos', 'n_neg'):
                agg[name][k] += head_metrics[name][k]
        n_batches += 1

    result = {}
    for name in N5_HEAD_NAMES:
        result[name] = {k: agg[name][k] / max(1, n_batches) for k in ('loss', 'acc')}
        result[name]['total_valid'] = agg[name]['n_valid']
        result[name]['total_pos'] = agg[name]['n_pos']
        result[name]['total_neg'] = agg[name]['n_neg']
    return result


def train_seed(seed, train_loader, val_loader, pos_weights, out_dir):
    """Train one seed of N5MultiHeadStudent."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = N5MultiHeadStudent(input_dim=51, hidden=64, short_rf=32, long_rf=128, dropout=0.1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Seed {seed}: {n_params:,} params')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, pos_weights, optimizer)
        val_metrics = validate(model, val_loader, pos_weights)

        val_loss = sum(val_metrics[name]['loss'] for name in N5_HEAD_NAMES)
        scheduler.step(val_loss)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_metrics': {name: {k: v for k, v in vm.items()}
                                        for name, vm in val_metrics.items()}})
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                'model': model.state_dict(),
                'seed': seed, 'epoch': epoch, 'val_loss': val_loss,
                'n5_schema_sha': n5_schema_sha(),
            }, os.path.join(out_dir, f'n5_seed{seed}_best.pt'))
        else:
            stale += 1

        if epoch % 5 == 0 or is_best:
            crit_acc = val_metrics['physical_criticality']['acc']
            k10_acc = val_metrics['k10_feasible']['acc']
            print(f'  Seed {seed} epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, '
                  f'crit_acc={crit_acc:.3f}, k10_acc={k10_acc:.3f}, best={best_epoch}')

        if stale >= EARLY_STOP_PATIENCE:
            print(f'  Seed {seed}: early stop at epoch {epoch}')
            break

    # Load best
    best_ckpt = torch.load(os.path.join(out_dir, f'n5_seed{seed}_best.pt'),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    final_metrics = validate(model, val_loader, pos_weights)

    print(f'  Seed {seed}: best epoch={best_epoch}, val_loss={best_val_loss:.4f}')
    for name in N5_HEAD_NAMES:
        print(f'    {name}: acc={final_metrics[name]["acc"]:.3f}, '
              f'n_valid={final_metrics[name]["total_valid"]}')

    return {
        'seed': seed, 'n_params': n_params, 'best_epoch': best_epoch,
        'best_val_loss': best_val_loss, 'n_epochs': epoch,
        'final_val_metrics': final_metrics, 'history': history,
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='all',
                       help='Comma-separated seed list or "all"')
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Output: {OUT_ROOT}')
    print()

    # Load G6
    with open(G6_SEAL_PATH) as f:
        seal = json.load(f)
    print(f'G6 Seal SHA: {seal["self_sha256"][:16]}...')
    normalizer = N5Normalizer.load(seal['normalization']['path'])
    pos_weights = seal['pos_weights']
    for name in N5_HEAD_NAMES:
        print(f'  {name}: pos_weight={pos_weights[name]}')
    print()

    # Load data
    print('Loading train data...')
    train_dataset = N5TensorDataset(seal, 'train', normalizer)
    print('Loading val data...')
    val_dataset = N5TensorDataset(seal, 'val', normalizer)

    train_indices = list(range(len(train_dataset)))
    train_suites = [ep['identity'].split('/')[0] for ep in train_dataset.episodes]
    sampler = SuiteBalancedSampler(train_indices, train_suites, BATCH_SIZE, shuffle=True,
                                   seed=seal['sampler']['seed'])
    train_loader = DataLoader(train_dataset, batch_sampler=list(sampler), collate_fn=collate_right_pad)

    val_sampler = SuiteBalancedSampler(list(range(len(val_dataset))),
                                       [ep['identity'].split('/')[0] for ep in val_dataset.episodes],
                                       BATCH_SIZE, shuffle=False, seed=42)
    val_loader = DataLoader(val_dataset, batch_sampler=list(val_sampler), collate_fn=collate_right_pad)

    # Determine seeds
    if args.seeds == 'all':
        seeds_to_run = SEEDS
    else:
        seeds_to_run = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'Seeds: {seeds_to_run}')
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    print(f'N5 Schema SHA: {n5_schema_sha()[:16]}...')
    print()

    all_results = {}
    for seed in seeds_to_run:
        print(f'=== N5 Seed {seed} ===')
        seed_dir = os.path.join(OUT_ROOT, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        result = train_seed(seed, train_loader, val_loader, pos_weights, seed_dir)
        all_results[str(seed)] = result

        with open(os.path.join(seed_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print()

    # Receipt
    receipt = {
        'gate': 'G8_N5_TRAINING',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'device': str(DEVICE),
        'g6_seal_sha': seal['self_sha256'],
        'n5_schema_sha': n5_schema_sha(),
        'seeds': seeds_to_run,
        'results': all_results,
    }
    receipt_path = os.path.join(OUT_ROOT, 'G8_RECEIPT.json')
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print('=' * 60)
    print('G8 N5 Training: complete')
    for seed, r in all_results.items():
        fm = r['final_val_metrics']
        crit_acc = fm['physical_criticality']['acc']
        k10_acc = fm['k10_feasible']['acc']
        print(f'  Seed {seed}: crit_acc={crit_acc:.3f}, k10_acc={k10_acc:.3f}, '
              f'best_epoch={r["best_epoch"]}')
    print(f'Receipt: {receipt_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
