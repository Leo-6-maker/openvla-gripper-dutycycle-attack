#!/usr/bin/env python3
"""Formal SC5-V2 evaluator: trajectory-level metrics on 90 dev trajectories."""
import csv, json, hashlib, math, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

V1_CKPT = REPO / "artifacts/detector/sc5_mlp_s2.pt"
V2_CKPTS = {
    42: REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
    123: REPO / "outputs/sc5_v2_seed123/sc5_mlp_v2.pt",
    456: REPO / "outputs/sc5_v2_seed456/sc5_mlp_v2.pt",
    789: REPO / "outputs/sc5_v2_seed789/sc5_mlp_v2.pt",
    1024: REPO / "outputs/sc5_v2_seed1024/sc5_mlp_v2.pt",
}
DATASET_CSV = REPO / "migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv"
LABELS = {
    'train': REPO / "evidence/m1c/sc5_v2_train_combined_labels.csv",
    'dev': REPO / "evidence/m1c/sc5_v2_dev_combined_labels.csv",
}
TAU_C = 0.3; TAU_R = 0.3; GUARD = 5

def load_runtime(ckpt_path, name):
    rt = SC5DetectorRuntime(str(ckpt_path), tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
    rt.name = name
    return rt

def eval_trajectory(rt, features_list):
    """Run detector on one trajectory. Returns (armed, emitted, emit_step, arm_step)."""
    rt.reset()
    armed = False; emitted = False; emit_step = -1; arm_step = -1
    for step, feats in enumerate(features_list):
        if emitted:
            break
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            continue
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)
        if rt.state == "ARMED" and arm_step < 0:
            arm_step = step
            armed = True
        if dec.get("emitted"):
            emitted = True; emit_step = step
    return armed, emitted, emit_step, arm_step

def main():
    # Load teacher labels
    dev_labels = {}
    for lr in csv.DictReader(open(LABELS['dev'])):
        key = (int(lr['task']), int(lr['state']), lr['source'])
        dev_labels[key] = lr

    train_labels = {}
    for lr in csv.DictReader(open(LABELS['train'])):
        key = (int(lr['task']), int(lr['state']), lr['source'])
        train_labels[key] = lr

    # Load step dataset, group by episode
    print("Loading step dataset...")
    all_rows = list(csv.DictReader(open(DATASET_CSV)))
    print(f"  {len(all_rows)} total steps")

    # Group val rows by episode
    ep_data = defaultdict(list)
    for r in all_rows:
        if r['split'] != 'val':
            continue
        ep_data[r['episode_id']].append(r)

    # Determine slice per episode
    ep_slice = {}
    for eid, rows in ep_data.items():
        task = int(rows[0]['task_idx'])
        state = int(rows[0]['parent_state_id'])
        source = rows[0]['source_pool']
        key = (task, state, source)
        lbl = dev_labels.get(key)
        if lbl is None:
            lbl = train_labels.get(key)
        tv = lbl.get('teacher_valid') == 'True' if lbl else None
        if source == 'primary':
            ep_slice[eid] = 'primary_dev'
        else:
            ep_slice[eid] = 'reserve_dev'

    print(f"  Episodes: {len(ep_data)}  Primary: {sum(1 for v in ep_slice.values() if v=='primary_dev')}  Reserve: {sum(1 for v in ep_slice.values() if v=='reserve_dev')}")

    # Load models
    models = {'V1': load_runtime(V1_CKPT, 'V1')}
    for seed, path in V2_CKPTS.items():
        models[f'V2_s{seed}'] = load_runtime(path, f'V2_s{seed}')

    # Evaluate
    results = {}
    for name, rt in models.items():
        print(f"\nEvaluating {name}...")
        ep_results = {}
        for eid in sorted(ep_data.keys()):
            rows = ep_data[eid]
            features_list = [{fn: float(r[fn]) for fn in SC5_FEATURES if r.get(fn,'') not in ('','nan','NaN')} for r in rows]
            features_list = [f for f in features_list if len(f) == len(SC5_FEATURES)]
            armed, emitted, emit_step, arm_step = eval_trajectory(rt, features_list)
            ep_results[eid] = {'armed': armed, 'emitted': emitted, 'emit_step': emit_step, 'arm_step': arm_step,
                               'slice': ep_slice.get(eid, 'unknown'), 'n_steps': len(features_list)}

        # Per-slice metrics
        for slice_name in ['primary_dev', 'reserve_dev', 'combined_dev']:
            if slice_name == 'combined_dev':
                eps = list(ep_results.values())
            else:
                eps = [v for v in ep_results.values() if v['slice'] == slice_name]

            # Get teacher labels for these episodes
            tv_eps = []; nc_eps = []
            for eid, v in ep_results.items():
                if slice_name != 'combined_dev' and v['slice'] != slice_name:
                    continue
                rows = ep_data[eid]
                task = int(rows[0]['task_idx']); state = int(rows[0]['parent_state_id'])
                source = rows[0]['source_pool']
                key = (task, state, source)
                lbl = dev_labels.get(key) or train_labels.get(key)
                if lbl and lbl.get('teacher_valid') == 'True':
                    tv_eps.append(v)
                else:
                    nc_eps.append(v)

            tv_triggered = sum(1 for v in tv_eps if v['emitted'])
            tv_total = len(tv_eps)
            nc_triggered = sum(1 for v in nc_eps if v['emitted'])
            nc_total = len(nc_eps)

            results.setdefault(slice_name, {})[name] = {
                'tv_recall': tv_triggered / max(tv_total, 1),
                'tv_total': tv_total, 'tv_triggered': tv_triggered,
                'nc_abstain': 1.0 - nc_triggered / max(nc_total, 1),
                'nc_total': nc_total, 'nc_false_trigger': nc_triggered,
                'total_eps': len(eps),
                'armed_count': sum(1 for v in eps if v['armed']),
                'emitted_count': sum(1 for v in eps if v['emitted']),
            }
            r = results[slice_name][name]
            print(f"  {slice_name}: TV={r['tv_recall']:.3f} ({r['tv_triggered']}/{r['tv_total']}) NC_abstain={r['nc_abstain']:.3f} ({r['nc_false_trigger']}/{r['nc_total']})")

    # Save
    out = {
        'gate': 'SC5_V2_FORMAL_EVALUATOR',
        'tau_corridor': TAU_C, 'tau_release': TAU_R, 'guard': GUARD,
        'v1_sha256': hashlib.sha256(open(V1_CKPT,'rb').read()).hexdigest(),
        'dataset_sha256': hashlib.sha256(open(DATASET_CSV,'rb').read()).hexdigest(),
        'results': results,
    }
    out_path = REPO / "evidence/m1c/formal_evaluator_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Checkpoint selection summary
    print("\n=== CHECKPOINT SELECTION ===")
    v1_tv = results['primary_dev']['V1']['tv_recall']
    tv_threshold = max(0.80, v1_tv - 0.05)
    print(f"V1 TV recall: {v1_tv:.3f}  Gate: TV >= {tv_threshold:.3f}")
    for name in sorted(models.keys()):
        if name == 'V1': continue
        r = results['primary_dev'][name]
        tv_ok = r['tv_recall'] >= tv_threshold
        nc_ok = r['nc_abstain'] >= 0.90
        print(f"  {name}: TV={r['tv_recall']:.3f} (ok={tv_ok}) NC_abstain={r['nc_abstain']:.3f} (ok={nc_ok}) episodes={r['total_eps']}")

if __name__ == "__main__":
    main()
