"""Autonomous Formal V2-B 12-Split Training Launcher.

Self-contained: reads split manifest, trains 12 splits in parallel on available GPUs,
selects checkpoints by FIT_DEV episode AUPRC, validates reload parity, freezes Student.

Authorized: FIT_TRAIN, FIT_DEV. Forbidden: H1, C2, P2, H2, A, FEC.
"""
import json, os, sys, hashlib, time, random, argparse, subprocess, glob, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ── Frozen paths ──
FEAT_ROOT  = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
FIT_TRAIN_LABELS = '/tmp/ft_FIT_TRAIN/labels'
FIT_DEV_LABELS   = '/tmp/ft_FIT_DEV/labels'
SPLIT_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'
ARCH_RECEIPT_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline/v2_architecture_selection/architecture_selection_receipt.json'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_student_training_v1'

K10 = 10; SEED = 42; EPOCHS = 30; LR = 1e-3; WD = 1e-4
BATCH_SIZE = 4; HIDDEN_DIM = 64; RECEPTIVE_FIELD = 32; DROPOUT = 0.1

SPLIT_NAMES = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
               'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2, CausalTCNEncoder
from gripper_attack.v6_critical_dataset import load_v2_episodes, CriticalEpisodeDataset, collate_v2_batch, CriticalEpisode

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

def auroc(y_true, y_score):
    if len(y_true) < 2: return 0.5
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    tpr = np.cumsum(y_sort)/n_pos; fpr = np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr, fpr))

def auprc(y_true, y_score):
    if len(y_true) < 2: return 0.0
    n_pos = y_true.sum()
    if n_pos == 0: return 0.0
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    prec = np.cumsum(y_sort) / np.arange(1, len(y_sort)+1)
    rec = np.cumsum(y_sort) / n_pos
    return float(np.trapz(prec, rec))


def train_one_split(split_idx, gpu_id):
    """Train one split, return metrics. Called as independent worker."""
    split_name = SPLIT_NAMES[split_idx]
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(OUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load split manifest
    manifest = json.load(open(SPLIT_MANIFEST_PATH))
    outer_idx = split_idx // 3; inner_idx = split_idx % 3
    outer = manifest['splits'][f'fold_{outer_idx}']
    inner = outer['inner_folds'][inner_idx]

    val_ids = set(inner['identities'])
    train_ids = set()
    for j, inf in enumerate(outer['inner_folds']):
        if j != inner_idx:
            train_ids.update(inf['identities'])

    print(f'[{split_name}] Loading data on GPU {gpu_id}...')
    # Build manifest with train/val split and load via load_v2_episodes
    train_manifest = {'splits': {'train': {'identities': list(train_ids)}}}
    dev_manifest = {'splits': {'dev': {'identities': list(val_ids)}}}
    train_eps = load_v2_episodes(FEAT_ROOT, FIT_TRAIN_LABELS, train_manifest, exclude_parser_contradictions=True)
    train_eps += load_v2_episodes(FEAT_ROOT, FIT_DEV_LABELS, train_manifest, exclude_parser_contradictions=True)
    dev_eps = load_v2_episodes(FEAT_ROOT, FIT_TRAIN_LABELS, dev_manifest, exclude_parser_contradictions=True)
    dev_eps += load_v2_episodes(FEAT_ROOT, FIT_DEV_LABELS, dev_manifest, exclude_parser_contradictions=True)

    if len(train_eps) == 0 or len(dev_eps) == 0:
        print(f'[{split_name}] FAIL: No data loaded')
        return {'split': split_name, 'status': 'FAIL_NO_DATA'}

    # Normalization from train
    cat_25d = np.concatenate([ep.features_25d for ep in train_eps], axis=0)
    cat_p9d = np.concatenate([ep.policy_9d for ep in train_eps], axis=0)
    cat_g9d = np.concatenate([ep.gripper_9d for ep in train_eps], axis=0)
    n25d_m = torch.tensor(cat_25d.mean(0), dtype=torch.float32, device=device)
    n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), dtype=torch.float32, device=device)
    np9d_m = torch.tensor(cat_p9d.mean(0), dtype=torch.float32, device=device)
    np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), dtype=torch.float32, device=device)
    ng9d_m = torch.tensor(cat_g9d.mean(0), dtype=torch.float32, device=device)
    ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), dtype=torch.float32, device=device)

    # Build V2-B model
    model = CriticalTriggerStudentV2(
        input_dim_25d=43, hidden_dim=HIDDEN_DIM, receptive_field=RECEPTIVE_FIELD,
        dropout=DROPOUT, use_policy_bypass=False, use_gripper_bypass=False,
        use_instruction_context=False,
        head_names=['k10_startability', 'secure_grasp', 'manipulation_intent'])
    model = model.to(device)

    # Data loaders
    train_ds = CriticalEpisodeDataset(train_eps); dev_ds = CriticalEpisodeDataset(dev_eps)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_v2_batch)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_v2_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_train = len(train_eps); n_dev = len(dev_eps)
    opp_rate = sum(1 for e in train_eps if e.has_opportunity)/max(n_train,1)
    print(f'[{split_name}] Train={n_train} Dev={n_dev} Opp={opp_rate:.1%}')

    # Training loop
    best_auprc = -1.0; best_epoch = -1; best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0; n_batches = 0

        for batch in train_loader:
            # V2-B: concatenate 25D + policy + gripper
            x_cat = torch.cat([
                (batch['x_25d'].to(device) - n25d_m) / n25d_s,
                (batch['x_policy'].to(device) - np9d_m) / np9d_s,
                (batch['x_gripper'].to(device) - ng9d_m) / ng9d_s,
            ], dim=-1)  # [B, T, 43]

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

            logits = model(x_cat)
            loss = 0.0
            weights = {'k10_startability': 1.0, 'secure_grasp': 0.3, 'manipulation_intent': 0.3}
            for head, logit in logits.items():
                if head not in targets: continue
                tgt = targets[head]; msk = masks[head]
                while msk.dim() < logit.dim(): msk = msk.unsqueeze(-1)
                bce = nn.functional.binary_cross_entropy_with_logits(logit, tgt, reduction='none')
                loss += weights.get(head, 0.3) * (bce * msk).sum() / msk.sum().clamp(min=1)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on dev set
        model.eval()
        dev_scores = []; dev_labels = []
        with torch.no_grad():
            for ep in dev_eps:
                f25d = (torch.tensor(ep.features_25d, device=device) - n25d_m) / n25d_s
                p9d  = (torch.tensor(ep.policy_9d, device=device) - np9d_m) / np9d_s
                g9d  = (torch.tensor(ep.gripper_9d, device=device) - ng9d_m) / ng9d_s
                x_cat = torch.cat([f25d, p9d, g9d], dim=-1).unsqueeze(0)
                logits = model(x_cat)
                prob = torch.sigmoid(logits['k10_startability']).squeeze().cpu().numpy()
                max_t = max(1, ep.T - K10 + 1)
                dev_scores.append(float(prob[:max_t].max()))
                dev_labels.append(1.0 if ep.has_opportunity else 0.0)

        dev_labels_a = np.array(dev_labels); dev_scores_a = np.array(dev_scores)
        ep_auprc = auprc(dev_labels_a, dev_scores_a)
        ep_auroc = auroc(dev_labels_a, dev_scores_a)

        history.append({'epoch': epoch, 'loss': avg_loss, 'ep_auprc': ep_auprc, 'ep_auroc': ep_auroc})

        if ep_auprc > best_auprc:
            best_auprc = ep_auprc; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f'[{split_name}] Epoch {epoch:3d}: loss={avg_loss:.4f} ep_auprc={ep_auprc:.4f} ep_auroc={ep_auroc:.4f}')

    # Restore best
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt = {
        'state_dict': best_state, 'config': model.config,
        'split': split_name, 'best_epoch': best_epoch,
        'best_ep_auprc': float(best_auprc), 'best_ep_auroc': float(auroc(dev_labels_a, dev_scores_a)),
        'history': history, 'seed': SEED, 'epochs': EPOCHS,
        'n_train': n_train, 'n_dev': n_dev,
    }
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    torch.save(ckpt, ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)

    # Reload parity test
    model2 = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=HIDDEN_DIM,
        receptive_field=RECEPTIVE_FIELD, dropout=DROPOUT,
        use_policy_bypass=False, use_gripper_bypass=False, use_instruction_context=False,
        head_names=['k10_startability', 'secure_grasp', 'manipulation_intent'])
    model2.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict'])
    model2.to(device); model2.eval()

    # Parity test with fixed input
    test_in = torch.randn(1, 64, 43, device=device)
    with torch.no_grad():
        out1 = model(test_in)['k10_startability']
        out2 = model2(test_in)['k10_startability']
    max_diff = (out1 - out2).abs().max().item()
    reload_ok = max_diff < 1e-5

    # Seal
    seal_path = os.path.join(out_dir, 'SHA256SUMS')
    with open(seal_path, 'w') as f:
        f.write(f'{ckpt_sha}  checkpoint.pt\n')

    result = {
        'split': split_name, 'status': 'PASS' if reload_ok else 'FAIL_RELOAD',
        'n_train': n_train, 'n_dev': n_dev,
        'best_epoch': best_epoch, 'best_ep_auprc': float(best_auprc),
        'best_ep_auroc': float(auroc(dev_labels_a, dev_scores_a)),
        'reload_max_diff': float(max_diff), 'reload_ok': reload_ok,
        'checkpoint_sha256': ckpt_sha, 'checkpoint_path': ckpt_path,
    }
    print(f'[{split_name}] DONE: AUPRC={best_auprc:.4f} reload={reload_ok}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-list', type=str, default='0,1,2,3',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--splits', type=str, default=None,
                        help='Comma-separated split indices (0-11), default=all')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_list.split(',')]
    if args.splits:
        split_indices = [int(x) for x in args.splits.split(',')]
    else:
        split_indices = list(range(12))

    print(f'=== FORMAL V2-B TRAINING ===')
    print(f'GPUs: {gpu_ids}  Splits: {split_indices}')
    print(f'Output: {OUT_ROOT}')
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Launch workers — one per split, round-robin GPU assignment
    results = []
    # Run sequentially for stability (each split is independent)
    for i, split_idx in enumerate(split_indices):
        gpu = gpu_ids[i % len(gpu_ids)]
        r = train_one_split(split_idx, gpu)
        results.append(r)
        if r['status'] != 'PASS':
            s = r['split']; st = r['status']
        print(f'WARNING: {s} status={st}')

    # ── Post-training acceptance ──
    print('\n=== POST-TRAINING ACCEPTANCE ===')
    passed = [r for r in results if r['status'] == 'PASS']
    failed = [r for r in results if r['status'] != 'PASS']

    all_aucs = [r['best_ep_auroc'] for r in passed]
    all_auprcs = [r['best_ep_auprc'] for r in passed]
    pooled_auroc = np.mean(all_aucs) if all_aucs else 0
    pooled_auprc = np.mean(all_auprcs) if all_auprcs else 0

    print(f'Passed: {len(passed)}/12  Failed: {len(failed)}')
    print(f'Pooled AUROC: {pooled_auroc:.4f}  AUPRC: {pooled_auprc:.4f}')
    for r in results:
        flag = ' ✓' if r['status'] == 'PASS' else ' ✗'
        print(f'  {r["split"]}: ep_auroc={r["best_ep_auroc"]:.4f} ep_auprc={r["best_ep_auprc"]:.4f} reload={r["reload_ok"]}{flag}')

    acceptance = pooled_auroc >= 0.90 and pooled_auprc >= 0.90 and len(failed) == 0

    # ── Freeze receipt ──
    freeze = {
        'schema': 'FORMAL_V2_STUDENT_FREEZE_V1',
        'architecture': 'V2B_STARTABILITY_43D_CONCAT_V1',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'status': 'PASS' if acceptance else 'FAIL',
        'pooled_ep_auroc': float(pooled_auroc),
        'pooled_ep_auprc': float(pooled_auprc),
        'acceptance_gates': {
            'pooled_auroc_ge_0.90': pooled_auroc >= 0.90,
            'pooled_auprc_ge_0.90': pooled_auprc >= 0.90,
            'all_12_splits_pass': len(failed) == 0,
            'all_reload_parity_pass': all(r['reload_ok'] for r in results),
        },
        'splits': {r['split']: r for r in results},
        'hyperparameters': {
            'epochs': EPOCHS, 'lr': LR, 'wd': WD, 'batch_size': BATCH_SIZE,
            'hidden_dim': HIDDEN_DIM, 'receptive_field': RECEPTIVE_FIELD,
            'dropout': DROPOUT, 'seed': SEED,
        },
    }
    with open(os.path.join(OUT_ROOT, 'FORMAL_V2_STUDENT_FREEZE_V1.json'), 'w') as f:
        json.dump(freeze, f, indent=2)

    # Global seal
    all_files = []
    for root, dirs, files in os.walk(OUT_ROOT):
        for fn in sorted(files):
            if fn == 'SHA256SUMS': continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, OUT_ROOT)
            all_files.append((rel, sha256_file(fp)))

    with open(os.path.join(OUT_ROOT, 'SHA256SUMS'), 'w') as f:
        for rel, h in sorted(all_files):
            f.write(f'{h}  {rel}\n')

    status_str = 'PASS' if acceptance else 'FAIL'
    c2_str = 'READY' if acceptance else 'BLOCKED'
    print(f'\nFreeze: {OUT_ROOT}')
    print(f'Status: {status_str}')
    print(f'C2: {c2_str}')


if __name__ == '__main__':
    main()
