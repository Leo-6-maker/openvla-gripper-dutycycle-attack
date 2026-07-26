"""G7: Train 4 baselines — Prior, MLP, RF32 TCN, RF128 TCN.

All trained on 640 train episodes with G6 frozen normalizer + pos_weights.
Evaluated on 80 val episodes.
"""
import json, os, sys, time, hashlib, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student')

from n5_dataset import N5Dataset, N5Normalizer, N5_HEAD_NAMES
from n5_student_model import (
    N5MultiHeadStudent, CausalTCNEncoder, LastFrameMLP, PriorBaseline,
    FrozenPosWeights, masked_bce_loss, SuiteBalancedSampler,
)

# ── Config ──
G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL.json'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g7_baselines'
BATCH_SIZE = 32
N_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUT_ROOT, exist_ok=True)


# ── Data Loading ──

class N5TensorDataset(torch.utils.data.Dataset):
    """Pre-loads episodes into tensors for fast training."""

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

        print(f'{split_key}: {len(self.episodes)} episodes loaded')

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        features = self.normalizer.normalize(ep['features'].copy())
        labels = {}
        valid_masks = {}
        for name in N5_HEAD_NAMES:
            labels[name] = torch.tensor(
                (ep['labels'][name] > 0.5).astype(np.float32)  # Convert -1/0/1 to 0/1
            )
            valid_masks[name] = torch.tensor(ep['valid_masks'][name].astype(bool))
        features_for_mask = torch.tensor(features)
        timestep_mask = torch.ones(features.shape[0], dtype=torch.bool)
        return {
            'features': features_for_mask,
            'labels': labels,
            'valid_masks': valid_masks,
            'timestep_mask': timestep_mask,
            'identity': ep['identity'],
            'T': ep['T'],
        }


def collate_right_pad(batch):
    """Right-pad batch to max length."""
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
        'features': features,
        'labels': labels,
        'valid_masks': valid_masks,
        'timestep_mask': timestep_mask,
        'identities': identities,
    }


# ── Training ──

def compute_metrics(model_output, labels, valid_masks):
    """Compute per-head metrics."""
    metrics = {}
    for name in N5_HEAD_NAMES:
        logits = model_output[name]
        targets = labels[name]
        mask = valid_masks[name]
        n_valid = mask.sum().item()
        if n_valid == 0:
            metrics[name] = {'loss': 0.0, 'n_valid': 0, 'auroc': 0.0}
            continue

        loss = masked_bce_loss(logits, targets, mask)
        probs = torch.sigmoid(logits[mask])
        targets_masked = targets[mask]

        # AUROC approximation via simple accuracy at threshold 0.5
        preds = (probs > 0.5).float()
        acc = (preds == targets_masked).float().mean().item()

        metrics[name] = {
            'loss': loss.item(),
            'n_valid': n_valid,
            'acc': acc,
            'n_pos': int(targets_masked.sum().item()),
            'n_neg': int(n_valid - targets_masked.sum().item()),
        }
    return metrics


def train_epoch(model, dataloader, pos_weights, optimizer, epoch):
    """Single training epoch."""
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
    """Validation pass."""
    model.eval()
    all_metrics = defaultdict(lambda: {'loss': 0.0, 'n_valid': 0, 'acc': 0.0, 'n_pos': 0, 'n_neg': 0})
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        features = batch['features'].to(DEVICE)
        labels = {k: v.to(DEVICE) for k, v in batch['labels'].items()}
        valid_masks = {k: v.to(DEVICE) for k, v in batch['valid_masks'].items()}
        timestep_mask = batch['timestep_mask'].to(DEVICE)

        output = model(features, timestep_mask=timestep_mask)
        batch_metrics = compute_metrics(output, labels, valid_masks)

        batch_loss = 0.0
        for name in N5_HEAD_NAMES:
            all_metrics[name]['loss'] += batch_metrics[name]['loss']
            all_metrics[name]['n_valid'] += batch_metrics[name]['n_valid']
            all_metrics[name]['acc'] += batch_metrics[name]['acc']
            all_metrics[name]['n_pos'] += batch_metrics[name].get('n_pos', 0)
            all_metrics[name]['n_neg'] += batch_metrics[name].get('n_neg', 0)
            batch_loss += batch_metrics[name]['loss']

        total_loss += batch_loss
        n_batches += 1

    # Average per-head metrics
    result = {}
    for name in N5_HEAD_NAMES:
        m = all_metrics[name]
        n_b = max(1, n_batches)
        result[name] = {
            'avg_loss': m['loss'] / n_b,
            'total_valid': m['n_valid'],
            'avg_acc': m['acc'] / n_b,
            'total_pos': m['n_pos'],
            'total_neg': m['n_neg'],
        }
    result['total_loss'] = total_loss / max(1, n_batches)
    return result


def train_model(model, train_loader, val_loader, pos_weights, name, out_dir):
    """Full training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, pos_weights, optimizer, epoch)
        val_metrics = validate(model, val_loader, pos_weights)
        val_loss = val_metrics['total_loss']

        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_metrics': val_metrics})

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            ckpt_path = os.path.join(out_dir, f'{name}_best.pt')
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, ckpt_path)
        else:
            stale += 1

        if epoch % 5 == 0 or is_best:
            crit_acc = val_metrics['physical_criticality']['avg_acc']
            print(f'  {name} epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, '
                  f'crit_acc={crit_acc:.3f}, best_ep={best_epoch}')

        if stale >= EARLY_STOP_PATIENCE:
            print(f'  {name}: early stop at epoch {epoch}')
            break

    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(out_dir, f'{name}_best.pt'), map_location=DEVICE)
    model.load_state_dict(best_ckpt['model'])

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'n_epochs': epoch,
        'history': history,
    }


# ── Baseline Builders ──

class RF32TCN(nn.Module):
    """Single TCN encoder (RF=32) with 5 heads."""
    def __init__(self, input_dim=51, hidden=64):
        super().__init__()
        self.encoder = CausalTCNEncoder(input_dim, hidden, rf=32)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
            for _ in range(5)
        ])
        self.HEAD_NAMES = N5MultiHeadStudent.HEAD_NAMES
        self.N_HEADS = 5

    def forward(self, x, timestep_mask=None):
        h = self.encoder(x, timestep_mask)
        return {name: head(h).squeeze(-1) for name, head in zip(self.HEAD_NAMES, self.heads)}


class RF128TCN(nn.Module):
    """Single TCN encoder (RF=128) with 5 heads."""
    def __init__(self, input_dim=51, hidden=64):
        super().__init__()
        self.encoder = CausalTCNEncoder(input_dim, hidden, rf=128)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
            for _ in range(5)
        ])
        self.HEAD_NAMES = N5MultiHeadStudent.HEAD_NAMES
        self.N_HEADS = 5

    def forward(self, x, timestep_mask=None):
        h = self.encoder(x, timestep_mask)
        return {name: head(h).squeeze(-1) for name, head in zip(self.HEAD_NAMES, self.heads)}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'prior', 'mlp', 'rf32', 'rf128'])
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    print(f'Output: {OUT_ROOT}')
    print()

    # Load G6 seal
    with open(G6_SEAL_PATH) as f:
        seal = json.load(f)
    print(f'G6 Seal: {G6_SEAL_PATH}')
    print(f'G6 Seal SHA: {seal["self_sha256"][:16]}...')

    # Load normalizer
    norm_path = seal['normalization']['path']
    normalizer = N5Normalizer.load(norm_path)
    print(f'Normalizer loaded: {norm_path}')

    # Pos weights
    pos_weights = seal['pos_weights']
    for name in N5_HEAD_NAMES:
        print(f'  {name}: pos_weight={pos_weights[name]}')
    print()

    # Load data
    print('Loading training data...')
    train_dataset = N5TensorDataset(seal, 'train', normalizer)
    print('Loading validation data...')
    val_dataset = N5TensorDataset(seal, 'val', normalizer)

    # Sampler setup
    train_indices = list(range(len(train_dataset)))
    train_suites = [ep['identity'].split('/')[0] for ep in train_dataset.episodes]
    sampler = SuiteBalancedSampler(train_indices, train_suites, BATCH_SIZE, shuffle=True, seed=seal['sampler']['seed'])

    train_loader = DataLoader(train_dataset, batch_sampler=list(sampler),
                              collate_fn=collate_right_pad)
    val_sampler = SuiteBalancedSampler(list(range(len(val_dataset))),
                                       [ep['identity'].split('/')[0] for ep in val_dataset.episodes],
                                       BATCH_SIZE, shuffle=False, seed=42)
    val_loader = DataLoader(val_dataset, batch_sampler=list(val_sampler),
                            collate_fn=collate_right_pad)

    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    print()

    all_results = {}

    # Baseline 1: Prior
    if args.model in ('all', 'prior'):
        print('=== Baseline 1/4: Prior ===')
        prior_dir = os.path.join(OUT_ROOT, 'prior')
        os.makedirs(prior_dir, exist_ok=True)

        # Compute priors from train data
        priors = {}
        for name in N5_HEAD_NAMES:
            total_pos = seal['pos_neg_counts'][name]['pos']
            total_neg = seal['pos_neg_counts'][name]['neg']
            prior_p = total_pos / max(1, total_pos + total_neg)
            priors[name] = prior_p
            print(f'  {name}: prior={prior_p:.4f}')

        model = PriorBaseline(priors).to(DEVICE)
        val_metrics = validate(model, val_loader, pos_weights)
        print(f'  Val loss: {val_metrics["total_loss"]:.4f}')
        for name in N5_HEAD_NAMES:
            m = val_metrics[name]
            print(f'  {name}: acc={m["avg_acc"]:.3f}, n_valid={m["total_valid"]}')

        prior_result = {
            'model': 'PriorBaseline',
            'priors': priors,
            'val_metrics': val_metrics,
        }
        all_results['prior'] = prior_result
        print()

    # Baseline 2: Last-Frame MLP
    if args.model in ('all', 'mlp'):
        print('=== Baseline 2/4: Last-Frame MLP ===')
        mlp_dir = os.path.join(OUT_ROOT, 'mlp')
        os.makedirs(mlp_dir, exist_ok=True)

        model = LastFrameMLP(input_dim=51, hidden=64).to(DEVICE)
        print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

        mlp_result = train_model(model, train_loader, val_loader, pos_weights, 'mlp', mlp_dir)
        mlp_result['model'] = 'LastFrameMLP'
        mlp_result['n_params'] = sum(p.numel() for p in model.parameters())

        val_metrics = validate(model, val_loader, pos_weights)
        mlp_result['final_val_metrics'] = val_metrics
        for name in N5_HEAD_NAMES:
            print(f'  {name}: acc={val_metrics[name]["avg_acc"]:.3f}')

        all_results['mlp'] = mlp_result
        # Save result
        with open(os.path.join(mlp_dir, 'result.json'), 'w') as f:
            json.dump(mlp_result, f, indent=2, default=str)
        print()

    # Baseline 3: RF32 TCN
    if args.model in ('all', 'rf32'):
        print('=== Baseline 3/4: RF32 TCN ===')
        rf32_dir = os.path.join(OUT_ROOT, 'rf32')
        os.makedirs(rf32_dir, exist_ok=True)

        model = RF32TCN(input_dim=51, hidden=64).to(DEVICE)
        print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

        rf32_result = train_model(model, train_loader, val_loader, pos_weights, 'rf32', rf32_dir)
        rf32_result['model'] = 'RF32_TCN'
        rf32_result['n_params'] = sum(p.numel() for p in model.parameters())

        val_metrics = validate(model, val_loader, pos_weights)
        rf32_result['final_val_metrics'] = val_metrics
        for name in N5_HEAD_NAMES:
            print(f'  {name}: acc={val_metrics[name]["avg_acc"]:.3f}')

        all_results['rf32'] = rf32_result
        with open(os.path.join(rf32_dir, 'result.json'), 'w') as f:
            json.dump(rf32_result, f, indent=2, default=str)
        print()

    # Baseline 4: RF128 TCN
    if args.model in ('all', 'rf128'):
        print('=== Baseline 4/4: RF128 TCN ===')
        rf128_dir = os.path.join(OUT_ROOT, 'rf128')
        os.makedirs(rf128_dir, exist_ok=True)

        model = RF128TCN(input_dim=51, hidden=64).to(DEVICE)
        print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

        rf128_result = train_model(model, train_loader, val_loader, pos_weights, 'rf128', rf128_dir)
        rf128_result['model'] = 'RF128_TCN'
        rf128_result['n_params'] = sum(p.numel() for p in model.parameters())

        val_metrics = validate(model, val_loader, pos_weights)
        rf128_result['final_val_metrics'] = val_metrics
        for name in N5_HEAD_NAMES:
            print(f'  {name}: acc={val_metrics[name]["avg_acc"]:.3f}')

        all_results['rf128'] = rf128_result
        with open(os.path.join(rf128_dir, 'result.json'), 'w') as f:
            json.dump(rf128_result, f, indent=2, default=str)
        print()

    # ── Finalize ──
    receipt = {
        'gate': 'G7_BASELINES',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'device': str(DEVICE),
        'g6_seal_sha': seal['self_sha256'],
        'results': all_results,
    }

    receipt_path = os.path.join(OUT_ROOT, 'G7_RECEIPT.json')
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print('=' * 60)
    print('G7 Baselines: complete')
    print(f'Receipt: {receipt_path}')
    for name, r in all_results.items():
        em = r.get('final_val_metrics', r.get('val_metrics', {}))
        phys_acc = em.get('physical_criticality', {}).get('avg_acc', em.get('physical_criticality', {}).get('acc', 0))
        k10_acc = em.get('k10_feasible', {}).get('avg_acc', em.get('k10_feasible', {}).get('acc', 0))
        print(f'  {name}: crit_acc={phys_acc:.3f}, k10_acc={k10_acc:.3f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
