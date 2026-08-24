"""Formal V2.1 E0 12-split training on DEV 1300.

Recipe: V2-B 43D, step BCE, external K10, W=32, H=64, dropout=0.1.
Checkpoint: FIT_DEV episode AUPRC (primary).
"""
import json, os, sys, hashlib, time, random, argparse, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
DEV_LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']
NEW_LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
DEV_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_DEV_IDENTITY_MANIFEST_V2.json'
SPLIT_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_student_training_v1'

K10=10; SEED=42; EPOCHS=30; LR=1e-3; WD=1e-4; BATCH=4
HIDDEN=64; RF=32; DROPOUT=0.1
SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']

sys.path.insert(0,'/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2

def auroc(yt,ys):
    if len(yt)<2: return 0.5
    n_pos=yt.sum();n_neg=len(yt)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    tpr=np.cumsum(ysort)/n_pos;fpr=np.cumsum(1-ysort)/n_neg
    return float(np.trapz(tpr,fpr))
def auprc(yt,ys):
    if len(yt)<2: return 0.0
    n_pos=yt.sum()
    if n_pos==0: return 0.0
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    prec=np.cumsum(ysort)/np.arange(1,len(ysort)+1)
    rec=np.cumsum(ysort)/n_pos
    return float(np.trapz(prec,rec))

dev_id_set = set(json.load(open(DEV_MANIFEST_PATH))['identities'])
print('DEV identities: {}'.format(len(dev_id_set)))

# ── Load episodes ──
def load_eps():
    eps = {}
    for root in DEV_LABEL_ROOTS + [NEW_LABEL_ROOT]:
        if not os.path.isdir(root): continue
        for suite in sorted(os.listdir(root)):
            sp = os.path.join(root, suite)
            if not os.path.isdir(sp): continue
            for task in sorted(os.listdir(sp)):
                tp = os.path.join(sp, task)
                if not os.path.isdir(tp): continue
                for state in sorted(os.listdir(tp)):
                    eid = '{}/{}/{}'.format(suite, task, state)
                    if eid not in dev_id_set: continue
                    lp = os.path.join(tp, state, 'factorized_teacher_v1.jsonl')
                    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                    if not os.path.isfile(lp) or not os.path.isfile(fp): continue
                    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
                    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
                    labels.sort(key=lambda r:r['step']); T = len(recs); max_t = min(T,T-K10+1)
                    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
                        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
                        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
                        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
                        for r in recs], dtype=np.float32)
                    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    has_opp = bool(k10_s[:max_t].any())
                    eps[eid] = {'eid':eid,'T':T,'f25d':f25d,'p9d':p9d,'g9d':g9d,'k10_s':k10_s,'k10_k':k10_k,'has_opp':has_opp,'max_t':max_t,'suite':suite}
    return eps

all_eps = load_eps()
print('Loaded {} episodes'.format(len(all_eps)))

# ── Load inner-CV splits and filter to DEV ──
split_manifest = json.load(open(SPLIT_MANIFEST_PATH))

def train_one_split(split_name, gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id))
    out_dir = os.path.join(OUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # Get split identities
    outer_idx = SPLITS.index(split_name) // 3
    inner_idx = SPLITS.index(split_name) % 3
    outer = split_manifest['splits'][f'fold_{outer_idx}']
    inner = outer['inner_folds'][inner_idx]
    val_ids = set(inner['identities']) & dev_id_set
    train_ids = set()
    for j, inf in enumerate(outer['inner_folds']):
        if j != inner_idx:
            train_ids.update(inf['identities'])
    train_ids &= dev_id_set

    train_eps = [all_eps[eid] for eid in train_ids if eid in all_eps]
    val_eps = [all_eps[eid] for eid in val_ids if eid in all_eps]
    if len(train_eps) < 10 or len(val_eps) < 3:
        return {'split':split_name,'status':'FAIL_INSUFFICIENT_DATA'}

    # Normalization from train
    cat_25d = np.concatenate([ep['f25d'] for ep in train_eps], axis=0)
    cat_p9d = np.concatenate([ep['p9d'] for ep in train_eps], axis=0)
    cat_g9d = np.concatenate([ep['g9d'] for ep in train_eps], axis=0)
    n25d_m = torch.tensor(cat_25d.mean(0), device=device); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=device)
    np9d_m = torch.tensor(cat_p9d.mean(0), device=device); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=device)
    ng9d_m = torch.tensor(cat_g9d.mean(0), device=device); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=device)

    model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=HIDDEN, receptive_field=RF,
        dropout=DROPOUT, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auprc = -1.0; best_epoch = -1; best_state = None; history = []
    n_train = len(train_eps); n_val = len(val_eps)
    opp_rate = sum(1 for e in train_eps if e['has_opp']) / max(n_train,1)
    print('[{}] Train={} Val={} Opp={:.1%}'.format(split_name, n_train, n_val, opp_rate))

    for epoch in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; n_batches = 0
        random.shuffle(train_eps)
        for i in range(0, n_train, BATCH):
            batch = train_eps[i:i+BATCH]; max_T = max(ep['T'] for ep in batch); B = len(batch)
            x_cat = torch.zeros(B, max_T, 43, device=device); k10_s = torch.zeros(B, max_T, 1, device=device)
            k10_k = torch.zeros(B, max_T, 1, device=device)
            for b, ep in enumerate(batch):
                T = ep['T']
                x_cat[b,:T] = torch.cat([(torch.tensor(ep['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(ep['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(ep['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1)
                k10_s[b,:T,0] = torch.tensor(ep['k10_s'],device=device).float()
                k10_k[b,:T,0] = torch.tensor(ep['k10_k'],device=device).float()
            logits = model(x_cat); logit = logits['k10_startability']
            bce = nn.functional.binary_cross_entropy_with_logits(logit, k10_s, reduction='none')
            loss = (bce * k10_k).sum() / k10_k.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total_loss += loss.item(); n_batches += 1
        sched.step()

        # Eval on validation set
        model.eval(); val_scores = []; val_labels = []
        with torch.no_grad():
            for ep in val_eps:
                x_cat = torch.cat([(torch.tensor(ep['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(ep['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(ep['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1).unsqueeze(0)
                out = model(x_cat); prob = torch.sigmoid(out['k10_startability']).squeeze().cpu().numpy()
                val_scores.append(float(prob[:ep['max_t']].max())); val_labels.append(1.0 if ep['has_opp'] else 0.0)
        val_l = np.array(val_labels); val_s = np.array(val_scores)
        ep_auprc = auprc(val_l, val_s); ep_auc = auroc(val_l, val_s)
        history.append({'epoch':epoch,'loss':total_loss/max(n_batches,1),'ep_auprc':ep_auprc,'ep_auc':ep_auc})

        if ep_auprc > best_auprc:
            best_auprc = ep_auprc; best_epoch = epoch
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print('[{}] Epoch {}: loss={:.4f} ep_auprc={:.4f} ep_auc={:.4f}'.format(split_name, epoch, total_loss/max(n_batches,1), ep_auprc, ep_auc))

    model.load_state_dict(best_state)
    ckpt = {'state_dict':best_state,'config':model.config,'split':split_name,'best_epoch':best_epoch,
            'best_ep_auprc':float(best_auprc),'best_ep_auc':float(auroc(val_l,val_s)),
            'history':history,'seed':SEED,'n_train':n_train,'n_val':n_val}
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    torch.save(ckpt, ckpt_path)
    ckpt_sha = hashlib.sha256(open(ckpt_path,'rb').read()).hexdigest()

    # Reload parity
    model2 = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=HIDDEN, receptive_field=RF,
        dropout=DROPOUT, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability']).to(device)
    model2.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['state_dict'])
    model2.eval()
    with torch.no_grad():
        test_in = torch.randn(1, 64, 43, device=device)
        d = (model(test_in)['k10_startability'] - model2(test_in)['k10_startability']).abs().max().item()
    reload_ok = d < 1e-5

    with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f:
        f.write('{}  checkpoint.pt\n'.format(ckpt_sha))

    return {'split':split_name,'status':'PASS' if reload_ok else 'FAIL_RELOAD',
            'n_train':n_train,'n_val':n_val,'best_epoch':best_epoch,
            'best_ep_auprc':float(best_auprc),'best_ep_auc':float(auroc(val_l,val_s)),
            'reload_max_diff':d,'reload_ok':reload_ok,'checkpoint_sha256':ckpt_sha}

# ── Main ──
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-list', type=str, default='0,3')
parser.add_argument('--splits', type=str, default=None)
args = parser.parse_args()

gpu_ids = [int(x) for x in args.gpu_list.split(',')]
split_indices = [int(x) for x in args.splits.split(',')] if args.splits else list(range(12))

print('=== FORMAL V2.1 E0 TRAINING ===')
print('GPUs: {}  Splits: {}'.format(gpu_ids, split_indices))
os.makedirs(OUT_ROOT, exist_ok=True)

results = []
for i, si in enumerate(split_indices):
    gpu = gpu_ids[i % len(gpu_ids)]
    r = train_one_split(SPLITS[si], gpu)
    results.append(r)
    print('{}: {}'.format(r['split'], r['status']))

# ── Acceptance ──
passed = [r for r in results if r['status']=='PASS']
failed = [r for r in results if r['status']!='PASS']
all_aucs = [r['best_ep_auc'] for r in passed]
all_auprcs = [r['best_ep_auprc'] for r in passed]
pooled_auc = np.mean(all_aucs) if all_aucs else 0; pooled_auprc = np.mean(all_auprcs) if all_auprcs else 0
acc = pooled_auc >= 0.90 and pooled_auprc >= 0.90 and len(failed) == 0

freeze = {
    'schema':'FORMAL_V21_STUDENT_FREEZE_V1','architecture':'V2-B 43D E0','status':'PASS' if acc else 'FAIL',
    'pooled_ep_auroc':float(pooled_auc),'pooled_ep_auprc':float(pooled_auprc),
    'acceptance_gates':{'pooled_auroc_ge_0.90':pooled_auc>=0.90,'pooled_auprc_ge_0.90':pooled_auprc>=0.90,
        'all_splits_pass':len(failed)==0,'all_reload_parity':all(r.get('reload_ok',False) for r in results)},
    'splits':{r['split']:r for r in results},
    'recipe':{'variant':'E0','loss':'step_BCE','epochs':EPOCHS,'lr':LR,'wd':WD,'seed':SEED},
}
with open(os.path.join(OUT_ROOT,'FORMAL_V21_STUDENT_FREEZE_V1.json'),'w') as f: json.dump(freeze, f, indent=2)

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

all_files = []
for root, dirs, fns in os.walk(OUT_ROOT):
    for fn in sorted(fns):
        if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_ROOT)
        all_files.append((rel, sha256_file(fp)))
with open(os.path.join(OUT_ROOT,'SHA256SUMS'),'w') as f:
    for rel, h in sorted(all_files): f.write('{}  {}\n'.format(h, rel))
sh = sha256_file(os.path.join(OUT_ROOT,'SHA256SUMS'))
with open(os.path.join(OUT_ROOT,'SHA256SUMS.sha256'),'w') as f: f.write('{}  SHA256SUMS\n'.format(sh))

print('\nFORMAL_V21_STUDENT_TRAINING = {}'.format('PASS' if acc else 'FAIL'))
print('Pooled AUROC={:.4f} AUPRC={:.4f}'.format(pooled_auc, pooled_auprc))
print('C3 = READY' if acc else 'C3 = BLOCKED')
print('Seal: {}'.format(sh[:16]))
