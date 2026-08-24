"""V2 Engineering Trainer — 4 Ablation Variants.

Variants (shared splits, seed, epoch budget, quarantine):
  V2-A:             25D → K10_startability  (no bypass)
  V2-B:             43D → K10_startability  (no bypass)
  V2-Full:          43D → K10_startability  (with raw bypass)
  V2-Phase-Control: 43D → V1 phase labels   (with bypass)

Checkpoint: FIT_DEV episode-level K10 AUPRC (primary).
Normalization: FIT_TRAIN only.
Quarantine: 10 parser-contradiction episodes excluded.
"""
import json, os, sys, hashlib, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, List, Optional

# ── Paths ──
FEAT_ROOT  = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
FIT_TRAIN_LABELS = '/tmp/ft_FIT_TRAIN/labels'
FIT_DEV_LABELS   = '/tmp/ft_FIT_DEV/labels'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline/v2_engineering'

K10_WINDOW = 10
SPLITS_12 = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
             'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
SEED = 42

# ── Imports from V2 scaffold ──
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import (
    CriticalTriggerStudentV2, build_v2_recommended, build_v2_minimal)
from gripper_attack.v6_critical_dataset import (
    load_v2_episodes, CriticalEpisode, CriticalEpisodeDataset, collate_v2_batch)


def auroc(y_true, y_score):
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tpr = np.cumsum(y_sort)/n_pos; fpr = np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr, fpr))

def auprc(y_true, y_score):
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    n_pos = y_true.sum()
    if n_pos == 0: return 0.0
    prec = np.cumsum(y_sort) / np.arange(1, len(y_sort)+1)
    rec = np.cumsum(y_sort) / n_pos
    return float(np.trapz(prec, rec))


def load_fit_data(exclude_parser=True):
    """Load FIT_TRAIN + FIT_DEV episodes."""
    def _load(root, state_range):
        identities = []
        for suite in sorted(os.listdir(root)):
            sp = os.path.join(root, suite)
            if not os.path.isdir(sp): continue
            for task in sorted(os.listdir(sp)):
                tp = os.path.join(sp, task)
                if not os.path.isdir(tp): continue
                for state in sorted(os.listdir(tp)):
                    try:
                        state_num = int(state.replace('state_',''))
                    except: continue
                    if state_num not in state_range: continue
                    eid = '{}/{}/{}'.format(suite, task, state)
                    identities.append(eid)
        sk = 'fit_train' if max(state_range) < 24 else 'fit_dev'
        manifest = {'splits': {sk: {'identities': identities}}}
        return load_v2_episodes(FEAT_ROOT, root, manifest, exclude_parser_contradictions=exclude_parser)

    train_eps = _load(FIT_TRAIN_LABELS, range(0, 20))
    dev_eps   = _load(FIT_DEV_LABELS, range(20, 24))
    return train_eps, dev_eps


def compute_normalization(episodes: List[CriticalEpisode]):
    """Compute per-feature mean/std from FIT_TRAIN only."""
    all_25d = []; all_p9d = []; all_g9d = []
    for ep in episodes:
        all_25d.append(ep.features_25d)
        all_p9d.append(ep.policy_9d)
        all_g9d.append(ep.gripper_9d)
    cat_25d = np.concatenate(all_25d, axis=0)
    cat_p9d = np.concatenate(all_p9d, axis=0)
    cat_g9d = np.concatenate(all_g9d, axis=0)
    return {
        '25d_mean': cat_25d.mean(0).astype(np.float32),
        '25d_std':  cat_25d.std(0).clip(1e-8).astype(np.float32),
        'p9d_mean': cat_p9d.mean(0).astype(np.float32),
        'p9d_std':  cat_p9d.std(0).clip(1e-8).astype(np.float32),
        'g9d_mean': cat_g9d.mean(0).astype(np.float32),
        'g9d_std':  cat_g9d.std(0).clip(1e-8).astype(np.float32),
    }


def normalize_episode(ep, norm):
    """Apply normalization to one episode."""
    ep.features_25d = (ep.features_25d - norm['25d_mean']) / norm['25d_std']
    ep.policy_9d    = (ep.policy_9d - norm['p9d_mean']) / norm['p9d_std']
    ep.gripper_9d   = (ep.gripper_9d - norm['g9d_mean']) / norm['g9d_std']


def build_model(variant: str) -> CriticalTriggerStudentV2:
    """Build model for given variant."""
    head_names = ['k10_startability', 'secure_grasp', 'manipulation_intent']
    if variant == 'V2-A':
        return CriticalTriggerStudentV2(
            head_names=head_names, use_policy_bypass=False, use_gripper_bypass=False)
    elif variant == 'V2-B':
        return CriticalTriggerStudentV2(
            head_names=head_names, use_policy_bypass=False, use_gripper_bypass=False,
            input_dim_25d=43)  # 25+9+9 concatenated as single input (no bypass)
    elif variant == 'V2-Full':
        return build_v2_recommended(head_names=head_names)
    elif variant == 'V2-Phase-Control':
        return build_v2_recommended(head_names=['grasp_established', 'manipulation_active', 'release_or_instability'])
    raise ValueError(f'Unknown variant: {variant}')


def get_targets(ep, variant):
    """Get training targets for a given variant."""
    if variant == 'V2-Phase-Control':
        return {
            'grasp_established': ep.grasp_label.astype(np.float32),
            'manipulation_active': ep.manipulation_label.astype(np.float32),
            'release_or_instability': np.zeros(ep.T, dtype=np.float32),
        }, {
            'grasp_established': ep.grasp_known,
            'manipulation_active': ep.manipulation_known,
            'release_or_instability': np.ones(ep.T, dtype=bool),
        }
    else:
        return {
            'k10_startability': ep.k10_startable.astype(np.float32),
            'secure_grasp': ep.grasp_label.astype(np.float32),
            'manipulation_intent': ep.manipulation_label.astype(np.float32),
        }, {
            'k10_startability': ep.k10_known,
            'secure_grasp': ep.grasp_known,
            'manipulation_intent': ep.manipulation_known,
        }


def train_one_epoch(model, loader, optimizer, variant, device, norm):
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0; n_batches = 0

    # Normalization tensors on device
    n25d_mean = torch.tensor(norm['25d_mean'], device=device)
    n25d_std  = torch.tensor(norm['25d_std'], device=device)
    np9d_mean = torch.tensor(norm['p9d_mean'], device=device)
    np9d_std  = torch.tensor(norm['p9d_std'], device=device)
    ng9d_mean = torch.tensor(norm['g9d_mean'], device=device)
    ng9d_std  = torch.tensor(norm['g9d_std'], device=device)

    for batch in loader:
        x_25d = (batch['x_25d'].to(device) - n25d_mean) / n25d_std
        x_pol = (batch['x_policy'].to(device) - np9d_mean) / np9d_std
        x_grp = (batch['x_gripper'].to(device) - ng9d_mean) / ng9d_std
        T_vals = batch['episode_lengths']

        # Build targets per variant
        if variant == 'V2-Phase-Control':
            targets = {
                'grasp_established': batch['grasp_label'].to(device).float(),
                'manipulation_active': batch['manipulation_label'].to(device).float(),
                'release_or_instability': torch.zeros_like(batch['grasp_label'].to(device)).float(),
            }
            masks = {
                'grasp_established': batch['grasp_known'].to(device).float(),
                'manipulation_active': batch['manipulation_known'].to(device).float(),
                'release_or_instability': torch.ones_like(batch['grasp_known'].to(device)).float(),
            }
        else:
            targets = {
                'k10_startability': batch['k10_startable'].to(device).float(),
                'secure_grasp': batch['grasp_label'].to(device).float(),
                'manipulation_intent': batch['manipulation_label'].to(device).float(),
            }
            masks = {
                'k10_startability': batch['k10_known'].to(device).float(),
                'secure_grasp': batch['grasp_known'].to(device).float(),
                'manipulation_intent': batch['manipulation_known'].to(device).float(),
            }

        # Forward
        if variant == 'V2-A':
            logits = model(x_25d)
        elif variant == 'V2-B':
            x_cat = torch.cat([x_25d, x_pol, x_grp], dim=-1)
            logits = model(x_cat)
        else:
            logits = model(x_25d, x_pol, x_grp)

        # BCE per head, episode-balanced
        loss = 0.0
        for head_name, logit in logits.items():
            if head_name not in targets: continue
            tgt = targets[head_name]; msk = masks[head_name].float()
            # Ensure mask has same shape as logit
            while msk.dim() < logit.dim():
                msk = msk.unsqueeze(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logit, tgt, reduction='none')
            bce = (bce * msk).sum() / msk.sum().clamp(min=1)
            # Head weights: startability=1.0, grasp=0.3, manip=0.3
            w = 1.0 if head_name in ('k10_startability','grasp_established') else 0.3
            if head_name == 'k10_startability': w = 1.0
            elif head_name == 'secure_grasp': w = 0.3
            elif head_name == 'manipulation_intent': w = 0.3
            elif head_name == 'manipulation_active': w = 0.3
            elif head_name == 'release_or_instability': w = 0.1
            loss += w * bce

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_episodes(model, episodes, norm, variant, device):
    """Compute per-episode metrics. Returns dict of arrays."""
    model.eval()
    results = []
    for ep in episodes:
        # Normalize
        f25d = (torch.tensor(ep.features_25d) - torch.tensor(norm['25d_mean'])) / torch.tensor(norm['25d_std'])
        p9d  = (torch.tensor(ep.policy_9d) - torch.tensor(norm['p9d_mean'])) / torch.tensor(norm['p9d_std'])
        g9d  = (torch.tensor(ep.gripper_9d) - torch.tensor(norm['g9d_mean'])) / torch.tensor(norm['g9d_std'])

        f25d = f25d.unsqueeze(0).to(device)
        p9d = p9d.unsqueeze(0).to(device)
        g9d = g9d.unsqueeze(0).to(device)

        if variant == 'V2-A':
            logits = model(f25d)
        elif variant == 'V2-B':
            x_cat = torch.cat([f25d, p9d, g9d], dim=-1)
            logits = model(x_cat)
        else:
            logits = model(f25d, p9d, g9d)

        # Get primary head logits
        if variant == 'V2-Phase-Control':
            k10_logit = logits.get('manipulation_active', list(logits.values())[0])
            k10_prob = torch.sigmoid(k10_logit).squeeze().cpu().numpy()
        else:
            k10_logit = logits.get('k10_startability', list(logits.values())[0])
            k10_prob = torch.sigmoid(k10_logit).squeeze().cpu().numpy()

        T = ep.T; max_t = min(T, T-K10_WINDOW+1)

        # Step-level metrics
        if ep.k10_known[:max_t].sum() > 0:
            step_auc = auroc(ep.k10_startable[:max_t], k10_prob[:max_t])
            step_ap  = auprc(ep.k10_startable[:max_t], k10_prob[:max_t])
        else:
            step_auc = 0.5; step_ap = 0.0

        # Episode-level: max score in valid window
        ep_score = k10_prob[:max_t].max() if max_t > 0 else 0.0

        results.append({
            'eid': ep.eid, 'split': ep.split,
            'has_opp': ep.has_opportunity, 'absence_reason': ep.absence_reason,
            'T': T,
            'step_auc': step_auc, 'step_ap': step_ap,
            'ep_score': float(ep_score),
            'k10_prob_max': float(ep_score),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='V2-Full',
                        choices=['V2-A','V2-B','V2-Full','V2-Phase-Control'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default=None, help='Single split key for smoke test')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'=== V2 ENGINEERING TRAINER: {args.variant} ===')
    print(f'Device: {device}  Epochs: {args.epochs}  LR: {args.lr}')

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # Load data
    print('Loading FIT data...')
    train_eps, dev_eps = load_fit_data(exclude_parser=True)
    print(f'Train: {len(train_eps)} episodes  Dev: {len(dev_eps)} episodes')

    # Compute normalization from FIT_TRAIN only
    # Do NOT normalize in-place — evaluate_episodes handles normalization
    norm = compute_normalization(train_eps)

    # Data loaders
    train_ds = CriticalEpisodeDataset(train_eps)
    dev_ds   = CriticalEpisodeDataset(dev_eps)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                               collate_fn=collate_v2_batch, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=4, shuffle=False,
                               collate_fn=collate_v2_batch, num_workers=0)

    print(f'Training:  opp_rate={train_ds.opportunity_rate:.2%}  absence={train_ds.absence_summary}')
    print(f'Dev:       opp_rate={dev_ds.opportunity_rate:.2%}  absence={dev_ds.absence_summary}')

    # Build model
    model = build_model(args.variant)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.variant}  Params: {n_params:,}  Config: {model.config}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_auprc = -1.0; best_epoch = -1; best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.variant, device, norm)
        scheduler.step()

        # Evaluate on dev set
        dev_results = evaluate_episodes(model, dev_eps, norm, args.variant, device)

        # Episode-level K10 AUPRC (primary metric)
        opp_scores = [r['ep_score'] for r in dev_results if r['has_opp']]
        abs_scores = [r['ep_score'] for r in dev_results if not r['has_opp']]
        all_scores = [r['ep_score'] for r in dev_results]
        all_labels = np.array([1.0 if r['has_opp'] else 0.0 for r in dev_results])

        ep_auprc = auprc(all_labels, np.array(all_scores))
        ep_auroc = auroc(all_labels, np.array(all_scores)) if all_labels.sum() > 0 and (1-all_labels).sum() > 0 else 0.5

        # F3/F4 max scores
        f3_scores = [r['ep_score'] for r in dev_results if r['absence_reason'] == 'F3_NO_MANIPULATION']
        f4_scores = [r['ep_score'] for r in dev_results if r['absence_reason'] == 'F4_NO_STABLE_GRASP']

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'ep_auprc': ep_auprc, 'ep_auroc': ep_auroc})

        if ep_auprc > best_auprc:
            best_auprc = ep_auprc; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch:3d}: loss={train_loss:.4f}  '
              f'ep_auprc={ep_auprc:.4f}  ep_auroc={ep_auroc:.4f}  '
              f'F3_n={len(f3_scores)} F4_n={len(f4_scores)}  '
              f'{"*" if epoch == best_epoch else " "}')

    # Restore best
    model.load_state_dict(best_state)
    print(f'\nBest epoch: {best_epoch}  AUPRC={best_auprc:.4f}')

    # Final evaluation
    dev_results = evaluate_episodes(model, dev_eps, norm, args.variant, device)
    all_scores = np.array([r['ep_score'] for r in dev_results])
    all_labels = np.array([1.0 if r['has_opp'] else 0.0 for r in dev_results])

    # Per-stratum analysis
    for reason in ['OPPORTUNITY_PRESENT','F1_TASK_STRUCTURAL_ZERO','F3_NO_MANIPULATION',
                    'F4_NO_STABLE_GRASP','F6_PARSER_DECODER_ZERO']:
        subset = [r for r in dev_results if r['absence_reason'] == reason]
        if subset:
            scores_sub = np.array([r['ep_score'] for r in subset])
            print(f'  {reason}: n={len(subset)}  max_score: mean={scores_sub.mean():.4f} '
                  f'p50={np.median(scores_sub):.4f} p90={np.percentile(scores_sub,90):.4f}')

    # FS at fixed recall levels
    print('\n=== FS AT FIXED RECALL ===')
    opp_scores = np.array([r['ep_score'] for r in dev_results if r['has_opp']])
    abs_scores = np.array([r['ep_score'] for r in dev_results if not r['has_opp']])
    for rec_target in [0.2, 0.3, 0.5]:
        if len(opp_scores) == 0: continue
        thresh = np.percentile(opp_scores, 100*(1-rec_target))
        fs = (abs_scores > thresh).mean()
        f3_mask = np.array([r['absence_reason']=='F3_NO_MANIPULATION' for r in dev_results if not r['has_opp']])
        f4_mask = np.array([r['absence_reason']=='F4_NO_STABLE_GRASP' for r in dev_results if not r['has_opp']])
        f3_fs = (abs_scores[f3_mask] > thresh).mean() if f3_mask.sum() > 0 else 0
        f4_fs = (abs_scores[f4_mask] > thresh).mean() if f4_mask.sum() > 0 else 0
        print(f'  recall={rec_target:.0%}: threshold={thresh:.4f}  FS={fs:.4f}  F3_FS={f3_fs:.4f}  F4_FS={f4_fs:.4f}')

    # Save checkpoint
    out_dir = os.path.join(OUT_ROOT, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        'state_dict': best_state, 'config': model.config,
        'variant': args.variant, 'best_epoch': best_epoch,
        'best_auprc': best_auprc, 'norm': norm,
        'history': history, 'seed': SEED,
    }
    torch.save(ckpt, os.path.join(out_dir, 'checkpoint.pt'))
    # Seal
    with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f:
        h = hashlib.sha256(open(os.path.join(out_dir,'checkpoint.pt'),'rb').read()).hexdigest()
        f.write(f'{h}  checkpoint.pt\n')

    # Save metrics
    metrics = {
        'variant': args.variant, 'best_epoch': best_epoch,
        'best_ep_auprc': float(best_auprc),
        'best_ep_auroc': float(auroc(all_labels, all_scores)),
        'dev_summary': {
            'n_opp': int(all_labels.sum()), 'n_abs': int((1-all_labels).sum()),
        },
        'per_episode': dev_results,
        'history': history,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nCheckpoint saved: {out_dir}')
    print(f'Done: {args.variant}')


if __name__ == '__main__':
    main()
