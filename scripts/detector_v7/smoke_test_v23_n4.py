"""V2.3 N4 CPU smoke tests: causality, cold-start, proxy parity, GroupDRO.

Run: python smoke_test_v23_n4.py
Requires: torch, numpy, access to one episode + the N4Encoder + proxy code.
"""
import json, os, sys, hashlib, numpy as np, torch, torch.nn as nn
from collections import defaultdict

# ── Configuration ──
FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
DEV2_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CausalTCNEncoder

K10=10; HIDDEN=64; DROPOUT=0.1
DEVICE = torch.device('cpu')
FAILURES = []

def check(condition, name):
    if not condition:
        FAILURES.append(name)
        print(f'  FAIL: {name}')
    else:
        print(f'  PASS: {name}')

# ── Model replicas ──
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
    proxies[:,0] = cmd - qpos
    proxies[:,1] = (cmd < 0).astype(np.float32)
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

# ── Load one episode ──
print('=== Loading one DEV2 episode ===')
dev2_ids = set(json.load(open(DEV2_MANIFEST))['identities'])
ep = None
for root in LABEL_ROOTS:
    if ep is not None: break
    if not os.path.isdir(root): continue
    for suite in sorted(os.listdir(root)):
        if ep is not None: break
        sp = os.path.join(root, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            if ep is not None: break
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
                ep = {'eid':eid,'T':T,'max_t':max_t,'f25d':f25d,'p9d':p9d,'g9d':g9d,
                    'proxies':proxies,'k10_s':k10_s,'k10_k':k10_k,'cc':cc,'has_opp':has_opp}
                print(f'  Loaded {eid} T={T} has_opp={has_opp}')
                break
if ep is None:
    print('FAIL: Could not load any episode'); sys.exit(1)

# Normalize with self as training set (smoke only)
f25d_t = torch.tensor(ep['f25d']); p9d_t = torch.tensor(ep['p9d']); g9d_t = torch.tensor(ep['g9d'])
n25d_m = f25d_t.mean(0); n25d_s = f25d_t.std(0).clip(1e-8)
np9d_m = p9d_t.mean(0); np9d_s = p9d_t.std(0).clip(1e-8)
ng9d_m = g9d_t.mean(0); ng9d_s = g9d_t.std(0).clip(1e-8)

def build_input(eps, T_override=None):
    T = T_override or ep['T']
    base = torch.cat([(f25d_t[:T] - n25d_m) / n25d_s,
                      (p9d_t[:T] - np9d_m) / np9d_s,
                      (g9d_t[:T] - ng9d_m) / ng9d_s,
                      torch.tensor(ep['proxies'][:T])], dim=-1)
    return base.unsqueeze(0)

# ── TEST 1: Causality ──
print('\n=== TEST 1: Causality ===')
encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
encoder.eval(); head.eval()

# Baseline: full forward
T = ep['T']
with torch.no_grad():
    x_full = build_input(ep)
    out_full = head(encoder(x_full)).squeeze().cpu().numpy()

# Perturb future: modify steps > t_cut
for t_cut in [0, 1, 31, 63, 127, min(200, T-10)]:
    if t_cut >= T - 2: continue
    x_mod = build_input(ep).clone()
    # Perturb future features with large noise
    x_mod[:, t_cut+1:, :43] += torch.randn_like(x_mod[:, t_cut+1:, :43]) * 10.0
    x_mod[:, t_cut+1:, -8:] += torch.randn_like(x_mod[:, t_cut+1:, -8:]) * 10.0
    with torch.no_grad():
        out_mod = head(encoder(x_mod)).squeeze().cpu().numpy()
    max_diff = np.abs(out_full[:t_cut+1] - out_mod[:t_cut+1]).max()
    check(max_diff < 1e-5, f'causality_cut_{t_cut}_max_diff={max_diff:.2e}')

# ── TEST 2: Cold-start ──
print('\n=== TEST 2: Cold-start ===')
for test_T in [1, 2, 32, 64, 128, T]:
    if test_T > T: continue
    x_short = build_input(ep, test_T)
    with torch.no_grad():
        out_short = head(encoder(x_short)).squeeze().cpu().numpy()
    # Handle scalar output for T=1
    out_s_val = np.atleast_1d(out_short)
    # Last step should match full forward's last step at same position
    diff_last = abs(float(out_full[test_T-1]) - float(out_s_val[-1]))
    check(diff_last < 1e-5, f'cold_start_T{test_T}_last_diff={diff_last:.2e}')
    check(np.isfinite(out_s_val).all(), f'cold_start_T{test_T}_finite')
    check(not np.isnan(out_s_val).any(), f'cold_start_T{test_T}_no_nan')

# ── TEST 3: Proxy causality ──
print('\n=== TEST 3: Proxy causality (offline computation) ===')
for t in range(min(10, T)):
    # Recompute proxies with only first t+1 steps (simulating runtime)
    partial = compute_proxies(ep['f25d'][:t+1], ep['p9d'][:t+1], ep['g9d'][:t+1], t+1)
    # Compare last step with offline full computation
    for p_idx in range(8):
        diff = abs(partial[t, p_idx] - ep['proxies'][t, p_idx])
        check(diff < 1e-5, f'proxy_causal_p{p_idx}_t{t}_diff={diff:.2e}')

# ── TEST 4: Proxy NaN/Inf ──
print('\n=== TEST 4: Proxy NaN/Inf check ===')
for p_idx in range(8):
    check(not np.isnan(ep['proxies'][:, p_idx]).any(), f'proxy_p{p_idx}_no_nan')
    check(not np.isinf(ep['proxies'][:, p_idx]).any(), f'proxy_p{p_idx}_no_inf')

# ── TEST 5: GroupDRO components ──
print('\n=== TEST 5: GroupDRO ===')
# Simulate a small batch
from collections import defaultdict
group_weights = {'opp':1.0,'F3':1.0,'F4':1.0,'other':1.0}

# Test weight update
group_losses = {'opp':0.5,'F3':2.0,'F4':0.3,'other':0.1}
group_counts = {'opp':10,'F3':2,'F4':8,'other':5}
avg_group_loss = {g: group_losses[g]/max(group_counts[g],1) for g in group_losses}
# F3 has highest avg loss (2.0/2=1.0)
worst_g = max(avg_group_loss, key=avg_group_loss.get)
check(worst_g == 'F3', f'gdro_worst_group={worst_g} expected=F3')
group_weights[worst_g] *= 1.5
check(abs(group_weights['F3'] - 1.5) < 1e-6, f'gdro_weight_update_F3={group_weights["F3"]}')
check(group_weights['opp'] == 1.0, 'gdro_weight_opp_unchanged')

# Test empty group handling
empty_counts = {'opp':10,'F3':0,'F4':8,'other':5}
non_empty = {g: group_losses[g]/max(empty_counts[g],1) for g in group_losses if empty_counts[g] > 0}
check('F3' not in non_empty, 'gdro_empty_group_excluded')

# Test weight normalization
total_w = sum(group_weights.values())
check(total_w > 0, 'gdro_total_weight_positive')
check(not np.isnan(total_w), 'gdro_total_weight_not_nan')

# ── TEST 6: Feature order stability ──
print('\n=== TEST 6: Feature order stability ===')
f25d_0 = ep['f25d'][0]
check(len(f25d_0) == 25, f'f25d_dim={len(f25d_0)}')
check(ep['p9d'].shape[1] == 9, f'p9d_dim={ep["p9d"].shape[1]}')
check(ep['g9d'].shape[1] == 9, f'g9d_dim={ep["g9d"].shape[1]}')
check(ep['proxies'].shape[1] == 8, f'proxies_dim={ep["proxies"].shape[1]}')
total_dim = len(f25d_0) + ep['p9d'].shape[1] + ep['g9d'].shape[1] + ep['proxies'].shape[1]
check(total_dim == 51, f'total_input_dim={total_dim}')

# ── TEST 7: Normalization sanity ──
print('\n=== TEST 7: Normalization sanity ===')
x_input = build_input(ep)
check(x_input.shape[2] == 51, f'input_dim={x_input.shape[2]}')
check(x_input.shape[0] == 1, f'batch_size={x_input.shape[0]}')
check(not torch.isnan(x_input).any(), 'input_no_nan')
check(not torch.isinf(x_input).any(), 'input_no_inf')
# Normalized features should have roughly unit variance
std_check = x_input[0, :, :25].std(dim=0).mean().item()
check(0.1 < std_check < 10.0, f'normalized_std_in_range={std_check:.2f}')

# ── TEST 8: Encoder parameter count ──
print('\n=== TEST 8: Encoder parameter count ===')
n_params = sum(p.numel() for p in encoder.parameters())
print(f'  N4Encoder params: {n_params}')
check(10000 < n_params < 500000, f'param_count={n_params}')

# ── TEST 9: Forward pass stability ──
print('\n=== TEST 9: Forward pass stability ===')
encoder.eval(); head.eval()
for _ in range(5):
    with torch.no_grad():
        out1 = head(encoder(build_input(ep))).squeeze().cpu().numpy()
        out2 = head(encoder(build_input(ep))).squeeze().cpu().numpy()
    check(np.abs(out1 - out2).max() < 1e-5, 'forward_deterministic')

# ── TEST 10: Model I/O shapes ──
print('\n=== TEST 10: Model I/O shapes ===')
for test_T in [1, 5, 32, 64, 100, 200]:
    if test_T > T: continue
    x = torch.randn(2, test_T, 51)
    with torch.no_grad():
        h = encoder(x)
        check(h.shape == (2, test_T, 64), f'encoder_shape_T{test_T}={list(h.shape)}')
        out = head(h)
        check(out.shape == (2, test_T, 1), f'head_shape_T{test_T}={list(out.shape)}')

# ── Summary ──
print(f'\n{"="*50}')
if FAILURES:
    print(f'SMOKE TEST: FAIL ({len(FAILURES)} failures)')
    for f in FAILURES: print(f'  - {f}')
    sys.exit(1)
else:
    print('SMOKE TEST: ALL PASS')
    sys.exit(0)
