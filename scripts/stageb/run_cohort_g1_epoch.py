#!/usr/bin/env python3
"""G1: Cohort source epoch for a parent. Args: parent_name seed"""
import sys, os, json, torch, io, hashlib, csv, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/stageb')
from run_m3_step78_true_pgd_fixed_frame import *
from pathlib import Path

parent = sys.argv[1]  # ketchup_s18
seed = int(sys.argv[2])  # 81 or 82

CONFIGS = {'ketchup_s18': 'configs/cohort_ketchup_s18_v4.yaml'}
INPUTS = {'ketchup_s18': '/data/liuyu/outputs/cohort_g0_ketchup_s18/ketchup_s18_input'}
OUTS = {'ketchup_s18': '/data/liuyu/outputs/cohort_ketchup_v2_epoch_r1'}

cfg = load_config(Path(CONFIGS[parent]))
inp = INPUTS[parent]
out_root = Path(OUTS[parent])
sd = out_root / f'seed{seed}'
sd.mkdir(parents=True, exist_ok=True)

def tsha(t):
    b = io.BytesIO(); torch.save(t.detach().cpu(), b); return hashlib.sha256(b.getvalue()).hexdigest()

raw_image, clean_json = load_frozen_input(Path(inp))
model, processor, device = load_model(cfg['model']['path'], -1)
model_dtype = next(model.parameters()).dtype
action_dim = int(model.get_action_dim(cfg['model']['unnorm_key']))
clean_action = np.asarray(clean_json.get('clean_action', [0]*7), dtype=np.float32)
instruction = str(clean_json['instruction'])
clean_tokens = clean_json.get('clean_tokens', clean_json.get('clean_exact_7_tokens', [31872]*7))
clean_arm = list(clean_tokens[:6])

base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)
clean_pv = base_inputs['pixel_values']; clean_ids = base_inputs['input_ids']
torch.save(clean_pv.detach().cpu(), sd / 'clean_pixel_values.pt')
print(f'{parent} seed{seed}: Clean PV={tsha(clean_pv)[:16]}')

CleanGen = type('CleanGen', (), {})()
CleanGen.sequences = torch.tensor([clean_ids[0].detach().cpu().tolist() + clean_tokens], dtype=torch.long, device=device)
CleanGen.scores = []

def decode_pv(pv_tensor):
    adv = pv_tensor.to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        go = model.generate(input_ids=clean_ids, pixel_values=adv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(clean_ids.shape[1]), expected_new_tokens=action_dim)
    grip = int(tokens[-1]); arm = sum(1 for a,b in zip(list(tokens[:6]), clean_arm) if a==b)
    sr = go.scores[-1][0].detach().float().cpu()
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
    inv = validate_processed_argmax_matches_emitted(sr, grip, tolerance=1e-6)
    target_score = float(sr[31744]); others = sr.clone(); others[31744] = float('-inf')
    margin = target_score - float(others.max())
    linf = (pv_tensor.float() - clean_pv.float()).abs().max().item()
    return {'tokens': [int(t) for t in tokens], 'grip': grip, 'arm': arm, 'margin': margin, 'linf': linf, 'inv': inv['tie_aware_pass']}

# TRUE
print('TRUE_PGD...')
ti, _ = run_true_pgd_condition(name='TRUE_PGD_TRAJECTORY21_SELECTIVE', model=model, processor=processor, cfg=cfg, raw_image=raw_image, instruction=instruction, clean_action=clean_action, clean_gen=CleanGen, device=device, seed=seed, gradient_transform='none')
tt = ti['debug'].get('trajectory_candidate_inputs', [])
feasible = []
for tc in tt:
    pv = tc.get('pixel_values')
    if pv is None or not isinstance(pv, torch.Tensor): continue
    cid = tc.get('candidate_index', 0)
    r = decode_pv(pv); r['cid'] = cid
    if r['grip']==31744 and r['arm']>=5 and r['inv'] and r['linf']<=0.02353: feasible.append(r)
    torch.save(pv.detach().cpu(), sd / f'true_cand{cid}_adv_pv.pt')

if not feasible: print('FATAL: no TRUE feasible'); sys.exit(1)
feasible.sort(key=lambda r: (-r['margin'], r['linf'], r['cid']))
true_sel = feasible[0]
print(f'TRUE: id={true_sel["cid"]} arm={true_sel["arm"]}/6 margin={true_sel["margin"]:.1f}')

# SHUFFLED
print('SHUFFLED...')
si, _ = run_true_pgd_condition(name='SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE', model=model, processor=processor, cfg=cfg, raw_image=raw_image, instruction=instruction, clean_action=clean_action, clean_gen=CleanGen, device=device, seed=seed, gradient_transform=str(cfg['controls']['shuffled_grad_mode']))
st = si['debug'].get('trajectory_candidate_inputs', [])
shuf = []
for tc in st:
    pv = tc.get('pixel_values')
    if pv is None or not isinstance(pv, torch.Tensor): continue
    cid = tc.get('candidate_index', 0)
    r = decode_pv(pv); r['cid'] = cid; shuf.append(r)
    torch.save(pv.detach().cpu(), sd / f'shuffled_cand{cid}_adv_pv.pt')
shuf.sort(key=lambda r: (-r['arm'], -r['margin'], r['linf'], r['cid']))
shuf_sel = shuf[0]
print(f'SHUFFLED: id={shuf_sel["cid"]} arm={shuf_sel["arm"]}/6')

# RAND21
print('RAND21...')
from gripper_attack.m3_controls import rand_seed_schedule, sample_processor_delta, project_and_cast_processor_values
adapter = TokenPrefixPGDAttacker(model, processor, {'attack_optimizer': cfg['attack_optimizer']}, seed=seed, preprocess_kwargs=dict(cfg.get('preprocess', {})), device=device)
x = base_inputs['pixel_values']
seeds_l = rand_seed_schedule(seed + 100000, count=21)
rand = []
for idx, cs in enumerate(seeds_l):
    d = sample_processor_delta(x.shape, epsilon=float(cfg['attack_optimizer']['epsilon']), seed=int(cs), dtype=torch.float32, device=x.device)
    proj, _ = project_and_cast_processor_values(x, d, epsilon=float(cfg['attack_optimizer']['epsilon']), candidate_is_delta=True)
    proj_d = proj.detach()
    r = decode_pv(proj_d); r['cid'] = idx; rand.append(r)
    torch.save(proj_d.cpu(), sd / f'rand_cand{idx}_adv_pv.pt')
rand.sort(key=lambda r: (-r['arm'], -r['margin'], r['linf'], r['cid']))
rand_sel = rand[0]
print(f'RAND: id={rand_sel["cid"]} arm={rand_sel["arm"]}/6')

json.dump({'seed': seed, 'true': true_sel, 'rand': rand_sel, 'shuffled': shuf_sel}, open(sd / 'selections.json', 'w'), indent=2)
print(f'{parent} seed{seed} COMPLETE')
