"""Train one V2.3 N4 split. Called by launcher."""
import json, os, sys, hashlib, random, numpy as np, torch, torch.nn as nn, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--split-name', required=True); ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--feat-root', required=True); ap.add_argument('--label-roots', required=True)
ap.add_argument('--split-manifest', required=True); ap.add_argument('--dev2-manifest', required=True)
ap.add_argument('--out-root', required=True)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-4)
args = ap.parse_args()

split_name = args.split_name; device = torch.device(f'cuda:{args.gpu}')
FEAT_ROOT = args.feat_root; LABEL_ROOTS = args.label_roots.split(',')
OUT_ROOT = args.out_root
K10=10; BATCH=4; HIDDEN=64; DROPOUT=0.1
SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']

sys.path.insert(0,'/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CausalTCNEncoder

class N4Encoder(nn.Module):
    def __init__(self, base_dim=43, proxy_dim=8, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.short_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, long_rf, dropout)
        self.fusion = nn.Linear(hidden*2, hidden)
    def forward(self, x): return self.fusion(torch.cat([self.short_tcn(x), self.long_tcn(x)], dim=-1))

def compute_proxies(f25d, p9d, g9d, T):
    proxies = np.zeros((T, 8), dtype=np.float32)
    cmd = f25d[:,0]; qpos = f25d[:,1]
    proxies[:,0] = cmd - qpos; proxies[:,1] = (cmd < 0).astype(np.float32)
    proxies[1:,2] = np.diff(qpos); proxies[0,2] = 0
    dur = 0; cd = np.zeros(T)
    for t in range(T):
        if cmd[t] < 0: dur += 1
        else: dur = 0
        cd[t] = dur
    proxies[:,3] = cd
    proxies[:,4] = np.sqrt(f25d[:,6]**2 + f25d[:,7]**2 + f25d[:,8]**2)
    for t in range(T):
        w_s = max(0, t-4); w_e = min(T, t+1)
        proxies[t,5] = np.var(qpos[w_s:w_e]) if w_e-w_s > 1 else 0
    proxies[:,6] = g9d[:,0]; proxies[:,7] = g9d[:,7]
    return np.nan_to_num(proxies, 0).astype(np.float32)

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
    prec=np.cumsum(ysort)/np.arange(1,len(ysort)+1);rec=np.cumsum(ysort)/n_pos
    return float(np.trapz(prec,rec))

dev2_ids = set(json.load(open(args.dev2_manifest))['identities'])
split_manifest = json.load(open(args.split_manifest))
outer_idx = SPLITS.index(split_name) // 3; inner_idx = SPLITS.index(split_name) % 3
outer = split_manifest['splits'][f'fold_{outer_idx}']
inner = outer['inner_folds'][inner_idx]
val_ids = set(inner['identities']) & dev2_ids
train_ids = set()
for j, inf in enumerate(outer['inner_folds']):
    if j != inner_idx: train_ids.update(inf['identities'])
train_ids &= dev2_ids

out_dir = os.path.join(OUT_ROOT, split_name); os.makedirs(out_dir, exist_ok=True)

# Load episodes
all_eps = {}
for root in LABEL_ROOTS:
    if not os.path.isdir(root): continue
    for suite in sorted(os.listdir(root)):
        sp = os.path.join(root, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp): continue
            for state in sorted(os.listdir(tp)):
                eid = f'{suite}/{task}/{state}'
                if eid not in dev2_ids: continue
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
                cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                has_opp = bool(k10_s[:max_t].any())
                proxies = compute_proxies(f25d, p9d, g9d, T)
                all_eps[eid] = {'eid':eid,'T':T,'max_t':max_t,'f25d':f25d,'p9d':p9d,'g9d':g9d,
                    'proxies':proxies,'k10_s':k10_s,'k10_k':k10_k,'cc':cc,'has_opp':has_opp}

train_eps = [all_eps[eid] for eid in train_ids if eid in all_eps]
val_eps = [all_eps[eid] for eid in val_ids if eid in all_eps]
if len(train_eps) < 10 or len(val_eps) < 3:
    print(json.dumps({'split':split_name,'status':'FAIL_INSUFFICIENT_DATA'})); sys.exit(1)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

# Normalization
cat_25d = np.concatenate([e['f25d'] for e in train_eps], axis=0)
cat_p9d = np.concatenate([e['p9d'] for e in train_eps], axis=0)
cat_g9d = np.concatenate([e['g9d'] for e in train_eps], axis=0)
n25d_m = torch.tensor(cat_25d.mean(0), device=device); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=device)
np9d_m = torch.tensor(cat_p9d.mean(0), device=device); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=device)
ng9d_m = torch.tensor(cat_g9d.mean(0), device=device); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=device)

encoder = N4Encoder().to(device); head = nn.Linear(HIDDEN, 1).to(device)
opt = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

t_opp = [e for e in train_eps if e['has_opp']]; t_abs = [e for e in train_eps if not e['has_opp']]
n_train = len(train_eps); n_val = len(val_eps)
best_auprc = -1.0; best_epoch = -1; best_state = None; history = []
print(f'[{split_name}] Train={n_train} Val={n_val}')

for epoch in range(1, args.epochs+1):
    encoder.train(); head.train(); total_loss = 0.0; n_batches = 0
    random.shuffle(t_opp); random.shuffle(t_abs)
    n_iter = max(len(t_opp), len(t_abs)) // max(BATCH//2, 1) + 1
    for bi in range(n_iter):
        batch = [t_opp[bi % len(t_opp)], t_abs[bi % len(t_abs)]]
        max_T = max(e['T'] for e in batch); B = len(batch)
        x_cat = torch.zeros(B, max_T, 51, device=device)
        k10_s = torch.zeros(B, max_T, 1, device=device); k10_k = torch.zeros(B, max_T, 1, device=device)
        for b, e in enumerate(batch):
            T = e['T']
            base = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s,
                torch.tensor(e['proxies'],device=device)], dim=-1)
            x_cat[b,:T] = base
            k10_s[b,:T,0] = torch.tensor(e['k10_s'],device=device).float()
            k10_k[b,:T,0] = torch.tensor(e['k10_k'],device=device).float()
        logit = head(encoder(x_cat))
        bce = nn.functional.binary_cross_entropy_with_logits(logit, k10_s, reduction='none')
        loss = (bce * k10_k).sum() / k10_k.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0)
        opt.step(); total_loss += loss.item(); n_batches += 1
    sched.step()

    encoder.eval(); head.eval(); val_scores = []; val_labels = []
    with torch.no_grad():
        for e in val_eps:
            base = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s,
                torch.tensor(e['proxies'],device=device)], dim=-1).unsqueeze(0)
            sc = head(encoder(base)).squeeze().cpu().numpy()
            ep_sc = float(sc[:e['max_t']][e['cc'][:e['max_t']]].max()) if e['cc'][:e['max_t']].any() else float(sc[:e['max_t']].max())
            val_scores.append(ep_sc); val_labels.append(1.0 if e['has_opp'] else 0.0)
    val_l = np.array(val_labels); val_s = np.array(val_scores)
    ep_auprc = auprc(val_l, val_s); ep_auc = auroc(val_l, val_s)
    history.append({'epoch':epoch,'loss':total_loss/max(n_batches,1),'ep_auprc':ep_auprc,'ep_auc':ep_auc})
    if ep_auprc > best_auprc:
        best_auprc = ep_auprc; best_epoch = epoch
        best_state = {'enc':{k:v.cpu().clone() for k,v in encoder.state_dict().items()},
                      'head':{k:v.cpu().clone() for k,v in head.state_dict().items()}}
    if epoch % 10 == 0 or epoch == 1:
        print(f'[{split_name}] Epoch {epoch}: loss={total_loss/max(n_batches,1):.4f} ep_auprc={ep_auprc:.4f} ep_auc={ep_auc:.4f}')

encoder.load_state_dict(best_state['enc']); head.load_state_dict(best_state['head'])
ckpt = {'enc':best_state['enc'],'head':best_state['head'],'split':split_name,
    'best_epoch':best_epoch,'best_ep_auprc':float(best_auprc),'best_ep_auc':float(auroc(val_l,val_s)),
    'history':history,'n_train':n_train,'n_val':n_val,'seed':args.seed}
ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
torch.save(ckpt, ckpt_path)
ckpt_sha = hashlib.sha256(open(ckpt_path,'rb').read()).hexdigest()

# Reload parity
encoder2 = N4Encoder().to(device); head2 = nn.Linear(HIDDEN,1).to(device)
ckpt2 = torch.load(ckpt_path, map_location=device, weights_only=False)
encoder2.load_state_dict(ckpt2['enc']); head2.load_state_dict(ckpt2['head']); encoder2.eval(); head2.eval()
test_in = torch.randn(1, 64, 51, device=device)
d = (head(encoder(test_in)) - head2(encoder2(test_in))).abs().max().item()
reload_ok = d < 1e-5

with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f:
    f.write(f'{ckpt_sha}  checkpoint.pt\n')

print(json.dumps({'split':split_name,'status':'PASS' if reload_ok else 'FAIL_RELOAD',
    'n_train':n_train,'n_val':n_val,'best_epoch':best_epoch,
    'best_ep_auprc':float(best_auprc),'best_ep_auc':float(auroc(val_l,val_s)),
    'reload_max_diff':d,'reload_ok':reload_ok,'checkpoint_sha256':ckpt_sha}))
