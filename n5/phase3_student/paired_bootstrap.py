"""Paired bootstrap: delta(N5 - MLP) for criticality accuracy and AUPRC.

Addresses the audit finding: N5 bootstrap CI [0.875, 0.944] includes MLP's 0.881,
so simple CI overlap is insufficient. This computes paired per-episode deltas.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import torch

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)
from n5_dataset import N5Dataset, N5Normalizer, N5_HEAD_NAMES
from n5_student_model import N5MultiHeadStudent, LastFrameMLP

G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
MLP_CKPT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g7_baselines/mlp/mlp_best.pt'
N5_CKPT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g8_n5_training/seed_19903/n5_seed19903_best.pt'
N_BOOTSTRAP = 2000
SEED = 19903
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_model_eval(ckpt_path, model_cls, **kwargs):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = model_cls(**kwargs).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def evaluate_per_episode(model, val_idx, all_episodes, normalizer):
    """Return per-episode metrics: {identity: {crit_acc, k10_acc, crit_auprc, ...}}"""
    per_ep = {}
    for idx in val_idx:
        ep = all_episodes[idx]
        ident = ep['identity']
        feats_n = normalizer.normalize(ep['features'].copy())
        feats_t = torch.tensor(feats_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(feats_t)

        ep_metrics = {}
        for name in ['physical_criticality', 'k10_feasible']:
            logits = output[name][0].cpu().numpy()
            targets = ep['labels'][name]
            masks = ep['valid_masks'][name]
            if masks.sum() == 0:
                ep_metrics[name] = None
                continue

            lf = logits[masks]
            tf = (targets[masks] > 0.5).astype(np.float32)
            pf = sigmoid(lf)
            preds = (pf > 0.5).astype(np.float32)

            ep_metrics[name] = {
                'acc': float((preds == tf).mean()),
                'n_valid': int(masks.sum()),
                'n_pos': int(tf.sum()),
            }
        per_ep[ident] = ep_metrics
    return per_ep


def main():
    print(f'Device: {DEVICE}')

    with open(G6_SEAL_PATH) as f:
        seal = json.load(f)
    normalizer = N5Normalizer.load(seal['normalization']['path'])

    # Load val identities
    val_idents = seal['split']['val_identities']
    print(f'Val identities: {len(val_idents)}')

    # Load all episodes
    dataset = N5Dataset(IDENTITY_MANIFEST, CS200_ROOT, LABEL_ROOT, split='checkpoint_training')
    all_episodes = []
    for i in range(len(dataset)):
        all_episodes.append(dataset.get_episode(i))

    # Build val index lookup
    ident_to_idx = {ep['identity']: i for i, ep in enumerate(all_episodes)}
    val_idx = [ident_to_idx[i] for i in val_idents if i in ident_to_idx]

    # Load models
    print('Loading N5...')
    n5 = load_model_eval(N5_CKPT, N5MultiHeadStudent, input_dim=51, hidden=64, short_rf=32, long_rf=128)
    print('Loading MLP...')
    mlp = load_model_eval(MLP_CKPT, LastFrameMLP, input_dim=51, hidden=64)

    # Evaluate
    print('Evaluating per-episode...')
    n5_per_ep = evaluate_per_episode(n5, val_idx, all_episodes, normalizer)
    mlp_per_ep = evaluate_per_episode(mlp, val_idx, all_episodes, normalizer)

    # Paired bootstrap
    common_idents = sorted(set(n5_per_ep.keys()) & set(mlp_per_ep.keys()))
    print(f'Common identities: {len(common_idents)}')

    rng = np.random.RandomState(SEED)

    for head_name in ['physical_criticality', 'k10_feasible']:
        print(f'\n=== {head_name} ===')

        # Collect per-episode metrics
        deltas_acc = []
        n5_accs = []
        mlp_accs = []
        for ident in common_idents:
            n5m = n5_per_ep[ident].get(head_name)
            mlpm = mlp_per_ep[ident].get(head_name)
            if n5m is not None and mlpm is not None and n5m['n_valid'] > 0:
                deltas_acc.append(n5m['acc'] - mlpm['acc'])
                n5_accs.append(n5m['acc'])
                mlp_accs.append(mlpm['acc'])

        deltas_acc = np.array(deltas_acc)
        n_eps = len(deltas_acc)

        # Bootstrap
        boot_means = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n_eps, n_eps, replace=True)
            boot_means.append(deltas_acc[idx].mean())
        boot_means = np.array(boot_means)

        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        mean_delta = float(deltas_acc.mean())
        p_significant = float((boot_means > 0).mean())

        print(f'  N5 mean acc: {np.mean(n5_accs):.4f}')
        print(f'  MLP mean acc: {np.mean(mlp_accs):.4f}')
        print(f'  Mean delta: {mean_delta:.4f}')
        print(f'  Delta 95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        print(f'  P(delta > 0): {p_significant:.4f}')
        print(f'  N episodes: {n_eps}')

        if p_significant >= 0.95 and ci_low > 0:
            print(f'  VERDICT: N5 significantly better than MLP on {head_name}')
        elif p_significant >= 0.95:
            print(f'  VERDICT: N5 better on average but CI includes small negative values')
        else:
            print(f'  VERDICT: Difference not statistically significant at 95% level')

    # Per-suite AUPRC
    print('\n=== Per-Suite AUPRC (N5) ===')
    from sklearn.metrics import average_precision_score

    suite_data = defaultdict(lambda: {'logits': [], 'targets': []})
    for idx in val_idx:
        ep = all_episodes[idx]
        suite = ep['suite']
        feats_n = normalizer.normalize(ep['features'].copy())
        feats_t = torch.tensor(feats_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = n5(feats_t)
        for name in ['physical_criticality', 'k10_feasible']:
            logits = output[name][0].cpu().numpy()
            masks = ep['valid_masks'][name]
            if masks.sum() > 0:
                suite_data[(suite, name)]['logits'].append(logits[masks])
                suite_data[(suite, name)]['targets'].append(
                    (ep['labels'][name][masks] > 0.5).astype(np.float32))

    for (suite, name), d in sorted(suite_data.items()):
        lf = np.concatenate(d['logits'])
        tf = np.concatenate(d['targets'])
        pf = sigmoid(lf)
        if tf.sum() > 0 and tf.sum() < len(tf):
            auprc = float(average_precision_score(tf, pf))
            print(f'  {suite}/{name}: AUPRC={auprc:.4f} (n={len(tf)}, pos={int(tf.sum())})')

    print('\nDone.')


if __name__ == '__main__':
    main()
