#!/usr/bin/env python3
"""Formal SC5-V2 evaluator v2: per-episode details, fail-closed labels, tie-break metrics."""
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
DEV_LABELS_CSV = REPO / "evidence/m1c/sc5_v2_dev_combined_labels.csv"
TAU_C = 0.3; TAU_R = 0.3; GUARD = 5

def load_runtime(ckpt_path):
    return SC5DetectorRuntime(str(ckpt_path), tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)

def get_teacher_corridor(anchor_str):
    """Parse teacher anchor to get corridor start/end steps."""
    try:
        a = int(anchor_str)
        if a < 0:
            return -1, -1, -1
    except (ValueError, TypeError):
        return -1, -1, -1
    # Teacher anchor is the step where stable_carry begins
    # K=10 means corridor is anchor..anchor+K
    K = 10
    return a, a, a + K  # corridor_start, emit_target, corridor_end

def eval_trajectory(rt, features_list):
    rt.reset()
    armed = False; emitted = False; emit_step = -1; arm_step = -1
    for step, feats in enumerate(features_list):
        if emitted: break
        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)): continue
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)
        if rt.state == "ARMED" and arm_step < 0: arm_step = step; armed = True
        if dec.get("emitted"): emitted = True; emit_step = step
    return armed, emitted, emit_step, arm_step

def main():
    # Load dev labels (fail-closed: no train fallback, no missing)
    dev_labels = {}
    for lr in csv.DictReader(open(DEV_LABELS_CSV)):
        key = (int(lr['task']), int(lr['state']), lr['source'])
        dev_labels[key] = lr
    print(f"Dev labels loaded: {len(dev_labels)} episodes")

    all_rows = list(csv.DictReader(open(DATASET_CSV)))
    ep_data = defaultdict(list)
    for r in all_rows:
        if r['split'] != 'val': continue
        ep_data[r['episode_id']].append(r)

    # Verify all episodes have dev labels (fail-closed)
    missing = 0
    for eid, rows in ep_data.items():
        task = int(rows[0]['task_idx']); state = int(rows[0]['parent_state_id'])
        source = rows[0]['source_pool']
        key = (task, state, source)
        if key not in dev_labels:
            print(f"  MISSING_LABEL: episode={eid} task={task} state={state} source={source}")
            missing += 1
    if missing:
        raise RuntimeError(f"{missing} episodes missing dev labels — fail-closed")
    print(f"  {len(ep_data)} episodes, all labels present")

    # Build per-episode metadata
    ep_meta = {}
    for eid, rows in ep_data.items():
        task = int(rows[0]['task_idx']); state = int(rows[0]['parent_state_id'])
        source = rows[0]['source_pool']
        key = (task, state, source)
        lbl = dev_labels[key]
        tv = lbl.get('teacher_valid') == 'True'
        anchor = lbl.get('teacher_anchor', '-1')
        corr_start, corr_target, corr_end = get_teacher_corridor(anchor)
        ep_meta[eid] = {
            'task': task, 'state': state, 'source': source,
            'teacher_valid': tv, 'teacher_anchor': anchor,
            'corridor_start': corr_start, 'corridor_emit_target': corr_target,
            'corridor_end': corr_end,
            'slice': 'primary_dev' if source == 'primary' else 'reserve_dev',
        }

    # Evaluate all models
    models = {'V1': load_runtime(V1_CKPT)}
    for seed, path in V2_CKPTS.items():
        models[f'V2_s{seed}'] = load_runtime(path)

    all_ep_results = {}
    slice_results = {}
    for name, rt in models.items():
        print(f"\nEvaluating {name}...")
        ep_results = {}
        for eid in sorted(ep_data.keys()):
            rows = ep_data[eid]
            feats_list = [{fn: float(r[fn]) for fn in SC5_FEATURES if r.get(fn,'') not in ('','nan','NaN')} for r in rows]
            feats_list = [f for f in feats_list if len(f) == len(SC5_FEATURES)]
            armed, emitted, emit_step, arm_step = eval_trajectory(rt, feats_list)
            meta = ep_meta[eid]
            emit_before = emit_step >= 0 and emit_step < meta['corridor_start'] if meta['corridor_start'] >= 0 else False
            emit_inside = emit_step >= 0 and meta['corridor_start'] >= 0 and meta['corridor_start'] <= emit_step <= meta['corridor_end']
            emit_after = emit_step >= 0 and emit_step > meta['corridor_end'] if meta['corridor_end'] >= 0 else False
            ep_results[eid] = {
                'task': meta['task'], 'state': meta['state'], 'source': meta['source'],
                'teacher_valid': meta['teacher_valid'], 'teacher_anchor': meta['teacher_anchor'],
                'corridor_start': meta['corridor_start'], 'corridor_end': meta['corridor_end'],
                'armed': armed, 'arm_step': arm_step, 'emitted': emitted, 'emit_step': emit_step,
                'emit_before_corridor': emit_before, 'emit_inside_corridor': emit_inside,
                'emit_after_corridor': emit_after, 'n_steps': len(feats_list),
            }
        all_ep_results[name] = ep_results

        # Per-slice metrics
        for sl in ['primary_dev', 'reserve_dev', 'combined_dev']:
            if sl == 'combined_dev':
                eps = list(ep_results.values())
            else:
                eps = [v for v in ep_results.values() if ep_meta[v['task'] if False else list(ep_results.keys())[0]] is not None]
                eps = [v for v in ep_results.values() if ep_meta[list(ep_results.keys())[list(ep_results.values()).index(v)]]['slice'] == sl]

            # Rebuild: use eid lookup
            tv_eps = [v for v in ep_results.values() if ep_meta[[k for k in ep_results if ep_results[k] is v][0]]['teacher_valid']]
            if sl != 'combined_dev':
                tv_eps = [v for v in tv_eps if ep_meta[[k for k in ep_results if ep_results[k] is v][0]]['slice'] == sl]

        # Simpler: iterate eid
        for sl in ['primary_dev', 'reserve_dev', 'combined_dev']:
            tv_eps = []; nc_eps = []
            for eid, v in ep_results.items():
                meta = ep_meta[eid]
                if sl != 'combined_dev' and meta['slice'] != sl: continue
                if meta['teacher_valid']: tv_eps.append(v)
                else: nc_eps.append(v)

            tv_trig = sum(1 for v in tv_eps if v['emitted'])
            nc_trig = sum(1 for v in nc_eps if v['emitted'])
            emit_inside = sum(1 for v in tv_eps if v['emit_inside_corridor'])
            emit_before = sum(1 for v in tv_eps if v['emit_before_corridor'])

            slice_results.setdefault(sl, {})[name] = {
                'tv_recall': tv_trig / max(len(tv_eps), 1), 'tv_total': len(tv_eps), 'tv_triggered': tv_trig,
                'nc_abstain': 1.0 - nc_trig / max(len(nc_eps), 1), 'nc_total': len(nc_eps), 'nc_false_trigger': nc_trig,
                'emit_inside_corridor': emit_inside, 'emit_before_corridor': emit_before,
                'total_eps': len(tv_eps) + len(nc_eps),
                'armed_count': sum(1 for v in ep_results.values() if v['armed']),
                'emitted_count': sum(1 for v in ep_results.values() if v['emitted']),
            }
            r = slice_results[sl][name]
            print(f"  {sl}: TV={r['tv_recall']:.3f} ({r['tv_triggered']}/{r['tv_total']}) NC_abstain={r['nc_abstain']:.3f} ({r['nc_false_trigger']}/{r['nc_total']}) emit_inside={r['emit_inside_corridor']}/{r['tv_total']}")

    # Tie-break for tied seeds (42, 456, 789)
    print("\n=== TIE-BREAK 42 vs 456 vs 789 ===")
    tied = ['V2_s42', 'V2_s456', 'V2_s789']
    for name in tied:
        r = slice_results['primary_dev'][name]
        ep = all_ep_results[name]
        tv_eps = {eid: v for eid, v in ep.items() if ep_meta[eid]['teacher_valid'] and ep_meta[eid]['slice']=='primary_dev'}
        # Emit inside corridor
        emit_in = sum(1 for v in tv_eps.values() if v['emit_inside_corridor'])
        emit_before = sum(1 for v in tv_eps.values() if v['emit_before_corridor'])
        # Task-level TV recall
        task_tv = defaultdict(lambda: [0, 0])
        for eid, v in tv_eps.items():
            t = ep_meta[eid]['task']
            task_tv[t][1] += 1
            if v['emitted']: task_tv[t][0] += 1
        task_recalls = {t: c[0]/max(c[1],1) for t, c in task_tv.items()}
        worst = min(task_recalls.values()) if task_recalls else 0
        armed_not_emit = r['armed_count'] - r['emitted_count']
        print(f"  {name}: emit_inside={emit_in}/{r['tv_total']} emit_before={emit_before} armed_not_emit={armed_not_emit} worst_task_recall={worst:.3f} task_recalls={dict((t, round(v,3)) for t,v in task_recalls.items())}")

    # Save
    out = {
        'gate': 'SC5_V2_FORMAL_EVALUATOR_V2',
        'tau_corridor': TAU_C, 'tau_release': TAU_R, 'guard': GUARD,
        'v1_sha256': hashlib.sha256(open(V1_CKPT,'rb').read()).hexdigest(),
        'dataset_sha256': hashlib.sha256(open(DATASET_CSV,'rb').read()).hexdigest(),
        'slice_results': {sl: {name: r for name, r in models.items()} for sl, models in slice_results.items()},
    }
    out_path = REPO / "evidence/m1c/formal_evaluator_results.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Selection
    v1_tv = slice_results['primary_dev']['V1']['tv_recall']
    tv_gate = max(0.80, v1_tv - 0.05)
    print(f"\n=== SELECTION (TV >= {tv_gate:.3f}, NC >= 0.90) ===")
    for name in sorted(models.keys()):
        if name == 'V1': continue
        r = slice_results['primary_dev'][name]
        ok = r['tv_recall'] >= tv_gate and r['nc_abstain'] >= 0.90
        print(f"  {name}: TV={r['tv_recall']:.3f} NC={r['nc_abstain']:.3f} {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
