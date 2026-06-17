#!/usr/bin/env python3
"""H5-V2-B: Atomic source epoch — run TRUE+RAND+SHUFFLED, save all tensors."""
import sys, os, json, torch, io, hashlib, csv, yaml, numpy as np
from pathlib import Path

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/stageb')
from run_m3_step78_true_pgd_fixed_frame import *

def tsha(t):
    buf = io.BytesIO(); torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 81
OUT = Path(f"/data/liuyu/outputs/l3_h5_v2_candidate_epoch_20260617_r1/seed{seed}")
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config(Path('configs/m3_butter_s11_step60_v4.yaml'))

input_dir = f"/data/liuyu/outputs/l3_deepseek_autonomous_20260617_r1/h{'1_v4_butter_s11_step60_seed81' if seed==81 else '2_v4_primary/butter_s11_step0060_seed82'}/input"

raw_image, clean_json = load_frozen_input(Path(input_dir))
model, processor, device = load_model(cfg['model']['path'], -1)
model_dtype = next(model.parameters()).dtype
action_dim = int(model.get_action_dim(cfg['model']['unnorm_key']))
clean_action = np.asarray(clean_json['clean_action'], dtype=np.float32)
instruction = str(clean_json['instruction'])
clean_arm = clean_json['clean_exact_7_tokens'][:6]

base_inputs = preprocess_raw_image(raw_image, processor, instruction, cfg, device, model_dtype)
clean_pv = base_inputs['pixel_values']; clean_ids = base_inputs['input_ids']
torch.save(clean_pv.detach().cpu(), OUT / 'clean_pixel_values.pt')
clean_pv_sha = tsha(clean_pv)
print(f"Seed {seed}: Clean PV={clean_pv_sha[:16]}")

CleanGen = type('CleanGen', (), {})()
CleanGen.sequences = torch.tensor(
    [clean_ids[0].detach().cpu().tolist() + clean_json['clean_exact_7_tokens']],
    dtype=torch.long, device=device)
CleanGen.scores = []

all_candidates = []; selected = {}

def decode_and_classify(pv_tensor, condition, cid, source):
    adv = pv_tensor.to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        go = model.generate(input_ids=clean_ids, pixel_values=adv,
            max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=True)
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(clean_ids.shape[1]),
                                      expected_new_tokens=action_dim)
    grip = int(tokens[-1]); arm = sum(1 for a,b in zip(list(tokens[:6]), clean_arm) if a==b)
    sr = go.scores[-1][0].detach().float().cpu()
    from scripts.stageb.diagnose_m3_true_pgd_fixed_frame import validate_processed_argmax_matches_emitted
    inv = validate_processed_argmax_matches_emitted(sr, grip, tolerance=1e-6)
    delta_fp32 = pv_tensor.detach().float() - clean_pv.detach().float()
    linf = delta_fp32.abs().max().item()
    target_score = float(sr[31744]); others = sr.clone(); others[31744] = float('-inf')
    margin = target_score - float(others.max())
    feasible = grip==31744 and arm>=5 and inv['tie_aware_pass'] and linf<=0.02353
    return {
        'condition': condition, 'candidate_id': cid, 'candidate_source': source,
        'official_gripper_token': grip, 'arm_match': arm, 'arm_den': 6,
        'target_margin': margin, 'linf': linf,
        'score_invariant_pass': inv['tie_aware_pass'], 'feasible': feasible,
    }

# TRUE
print("TRUE_PGD_TRAJECTORY21_SELECTIVE...")
ti, _ = run_true_pgd_condition(
    name='TRUE_PGD_TRAJECTORY21_SELECTIVE', model=model, processor=processor,
    cfg=cfg, raw_image=raw_image, instruction=instruction,
    clean_action=clean_action, clean_gen=CleanGen,
    device=device, seed=seed, gradient_transform='none')
tt = ti['debug'].get('trajectory_candidate_inputs', [])
true_feasible = []
for tc in tt:
    pv = tc.get('pixel_values')
    if pv is None or not isinstance(pv, torch.Tensor): continue
    cid = tc.get('candidate_index', 0); src = tc.get('candidate_source', '')
    row = decode_and_classify(pv, 'TRUE_PGD_TRAJECTORY21_SELECTIVE', cid, src)
    all_candidates.append(row)
    if row['feasible']: true_feasible.append(row)
    torch.save(pv.detach().cpu(), OUT / f'true_cand{cid}_adv_pv.pt')
    torch.save((pv.float()-clean_pv.float()).cpu(), OUT / f'true_cand{cid}_delta.pt')

if not true_feasible:
    print("H5_V2_SOURCE_FAIL: no TRUE feasible"); sys.exit(1)
true_feasible.sort(key=lambda r: (-r['target_margin'], r['linf'], r['candidate_id']))
selected['TRUE'] = true_feasible[0]
print(f"TRUE: id={selected['TRUE']['candidate_id']} arm={selected['TRUE']['arm_match']}/6 margin={selected['TRUE']['target_margin']:.1f}")

# SHUFFLED
print("SHUFFLED_GRAD...")
si, _ = run_true_pgd_condition(
    name='SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE', model=model, processor=processor,
    cfg=cfg, raw_image=raw_image, instruction=instruction,
    clean_action=clean_action, clean_gen=CleanGen,
    device=device, seed=seed, gradient_transform=str(cfg['controls']['shuffled_grad_mode']))
st = si['debug'].get('trajectory_candidate_inputs', [])
shuf_rows = []
for tc in st:
    pv = tc.get('pixel_values')
    if pv is None or not isinstance(pv, torch.Tensor): continue
    cid = tc.get('candidate_index', 0); src = tc.get('candidate_source', '')
    row = decode_and_classify(pv, 'SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE', cid, src)
    all_candidates.append(row); shuf_rows.append(row)
    torch.save(pv.detach().cpu(), OUT / f'shuffled_cand{cid}_adv_pv.pt')
    torch.save((pv.float()-clean_pv.float()).cpu(), OUT / f'shuffled_cand{cid}_delta.pt')
shuf_rows.sort(key=lambda r: (-r['arm_match'], -r['target_margin'], r['linf'], r['candidate_id']))
selected['SHUFFLED'] = shuf_rows[0]
print(f"SHUFFLED: id={selected['SHUFFLED']['candidate_id']} arm={selected['SHUFFLED']['arm_match']}/6")

# RAND21
print("RAND21...")
from gripper_attack.m3_controls import rand_seed_schedule, sample_processor_delta, project_and_cast_processor_values
adapter = TokenPrefixPGDAttacker(
    model, processor, {'attack_optimizer': cfg['attack_optimizer']},
    seed=seed, preprocess_kwargs=dict(cfg.get('preprocess', {})), device=device)
x = base_inputs['pixel_values']
seeds_l = rand_seed_schedule(seed + 100000, count=int(cfg['controls'].get('rand21_count', 21)))
rand_rows = []
for idx, cs in enumerate(seeds_l):
    d = sample_processor_delta(x.shape, epsilon=float(cfg['attack_optimizer']['epsilon']),
                               seed=int(cs), dtype=torch.float32, device=x.device)
    proj, _ = project_and_cast_processor_values(x, d, epsilon=float(cfg['attack_optimizer']['epsilon']), candidate_is_delta=True)
    proj_d = proj.detach()
    row = decode_and_classify(proj_d, 'RAND21_SELECTIVE', idx, 'processor_random')
    all_candidates.append(row); rand_rows.append(row)
    torch.save(proj_d.cpu(), OUT / f'rand_cand{idx}_adv_pv.pt')
    torch.save((proj_d.float()-clean_pv.float()).cpu(), OUT / f'rand_cand{idx}_delta.pt')
rand_rows.sort(key=lambda r: (-r['arm_match'], -r['target_margin'], r['linf'], r['candidate_id']))
selected['RAND'] = rand_rows[0]
print(f"RAND: id={selected['RAND']['candidate_id']} arm={selected['RAND']['arm_match']}/6")

# Save
with open(OUT / 'all_candidates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_candidates[0].keys())); w.writeheader(); w.writerows(all_candidates)
sel_rows = [{'condition': c, **v} for c, v in selected.items()]
with open(OUT / 'selected_candidates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys())); w.writeheader(); w.writerows(sel_rows)

meta = {'seed': seed, 'clean_pv_sha': clean_pv_sha,
    'selected': {c: {'id': v['candidate_id'], 'arm': v['arm_match'],
        'margin': v['target_margin'], 'grip': v['official_gripper_token']} for c, v in selected.items()}}
json.dump(meta, open(OUT / 'source_run_metadata.json', 'w'), indent=2)

hashes = {}
for f in sorted(OUT.iterdir()):
    if f.is_file():
        h = hashlib.sha256(); h.update(f.read_bytes()); hashes[f.name] = h.hexdigest()
with open(OUT / 'recursive_hash_manifest.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['file', 'sha256'])
    for k, v in sorted(hashes.items()): w.writerow([k, v])

n_feas = sum(1 for r in all_candidates if r.get('feasible', False))
print(f"\nSeed{seed} COMPLETE: {len(all_candidates)} candidates, {n_feas} TRUE feasible")
print(f"Selected: TRUE={selected['TRUE']['candidate_id']} RAND={selected['RAND']['candidate_id']} SHUFFLED={selected['SHUFFLED']['candidate_id']}")
