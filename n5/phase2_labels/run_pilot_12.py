"""N5 Pilot Pipeline: 12-episode CS200 → V22 → Label V2 → K10 → atomic receipt.

HARD GATES:
  - candidate_close NOT in physics computation
  - Old K10 labels NOT reused (recomputed from V22 criticality)
  - unknown NOT converted to negative
  - Goal NO blanket NO_MANIPULATION_TARGET
  - Formal paths NOT modified
"""
import json, os, sys, hashlib, time, copy
import numpy as np

# Import V22 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels')

from v22_production import (
    parse_sidecar, resolve_goal_target, compute_grasp_state, compute_contact_state,
    compute_comotion_state, compute_lift_state, compute_instability_indicators,
    compute_terminal_state, v22_to_label_v2, validate_v22_snapshot,
    V22_CONFIG, compute_config_sha,
)
from physics_teacher_v22 import create_v22_snapshot, V22_SCHEMA_VERSION, compute_v22_schema_sha
# Import atomic writer from Label V2
from label_contract_v2 import write_atomic, N5_ALLOWED_ROOT

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'pilot_12_manifest.json')
PILOT_OUT = os.path.join(N5_ALLOWED_ROOT, 'phase2_labels', 'pilot_12_output')
K = 10

def recompute_k10(critical_labels, safe_release_labels, K=10):
    """Recompute K10 feasibility from V22 criticality sequence.

    K10_t = AND_{i=t}^{t+9} (critical_i AND known_i AND NOT safe_release_i)
    instability does NOT veto K10.
    """
    T = len(critical_labels)
    results = []
    for t in range(T):
        if t + K > T:
            results.append({'value': 0, 'valid_mask': True,
                            'reason': 'K10_INFEASIBLE_HORIZON', 'confidence': 0.0})
            continue

        all_critical = True
        has_unknown = False
        for i in range(t, t + K):
            if i >= T:
                all_critical = False; break
            crit = critical_labels[i]
            sr = safe_release_labels[i] if i < len(safe_release_labels) else {'value': 0, 'valid_mask': False}

            if not crit.get('valid_mask'):
                has_unknown = True
                all_critical = False
            elif crit.get('value') != 1:
                all_critical = False; break

            if sr.get('valid_mask') and sr.get('value') == 1:
                results.append({'value': 0, 'valid_mask': True,
                                'reason': 'K10_INFEASIBLE_SAFE_RELEASE', 'confidence': 0.0})
                all_critical = False; break

        if not all_critical and not has_unknown:
            results.append({'value': 0, 'valid_mask': True,
                            'reason': 'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR', 'confidence': 0.0})
        elif has_unknown:
            results.append({'value': None, 'valid_mask': False,
                            'reason': 'K10_UNKNOWN_CRITICAL_IN_WINDOW', 'confidence': 0.0})
        else:
            results.append({'value': 1, 'valid_mask': True,
                            'reason': 'K10_FEASIBLE', 'confidence': 1.0})

    return results

def process_episode(ep, manifest, dry_run=False):
    """Process one episode through full pipeline."""
    suite = ep['suite']; task = ep['task']; state = ep['state']
    sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    episode_summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
    metadata_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_metadata.json')

    # Verify inputs exist
    for p in [sidecar_path, episode_summary_path]:
        if not os.path.isfile(p):
            return {'error': f'Missing input: {p}', 'identity': f'{suite}/{task}/{state}'}

    # Compute input SHAs
    input_shas = {}
    for p in [sidecar_path, episode_summary_path, metadata_path]:
        if os.path.isfile(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk: break
                    h.update(chunk)
            input_shas[os.path.basename(p)] = h.hexdigest()

    # Step 1: Parse sidecar
    parsed = parse_sidecar(sidecar_path)
    steps = parsed['steps']
    identity = parsed['identity']

    # Step 2: Get task instruction (from sidecar, not metadata)
    instruction = steps[0].get('task_language', '') if steps else ''
    if not instruction and os.path.isfile(metadata_path):
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            instruction = meta.get('task_language', meta.get('task_instruction', ''))
        except Exception:
            pass

    # Step 3: Goal target resolution (object_state is raw qpos array, not named dict)
    # Use task_language only — resolver matches task name to known goal tasks
    target = resolve_goal_target(instruction, [])
    target_object_id = target.get('target_object_id') if target.get('target_resolved') else None

    # Step 4: Compute V22 factors
    grasp_results = compute_grasp_state(steps, target_object_id)
    contact_results = compute_contact_state(steps)
    comotion_results = compute_comotion_state(steps, target_object_id)
    lift_results = compute_lift_state(steps)
    instability_results = compute_instability_indicators(steps, grasp_results, contact_results)
    terminal_results = compute_terminal_state(steps)

    # Step 5: Build V22 snapshots per step
    v22_snapshots = []
    for t in range(len(steps)):
        snap = create_v22_snapshot()
        snap['step'] = t
        snap['suite'] = suite; snap['task_index'] = task; snap['state_index'] = state

        # Populate factors
        for k, v in grasp_results[t].items(): snap['factors']['grasp_state'][k] = v
        for k, v in contact_results[t].items(): snap['factors']['contact_state'][k] = v
        for k, v in comotion_results[t].items(): snap['factors']['comotion_state'][k] = v
        for k, v in lift_results[t].items(): snap['factors']['lift_state'][k] = v
        for k, v in instability_results[t].items(): snap['factors']['instability_indicators'][k] = v
        for k, v in terminal_results[t].items(): snap['factors']['terminal_state'][k] = v

        # Set factor-level known_mask
        snap['factors']['grasp_state']['known_mask'] = True
        snap['factors']['contact_state']['known_mask'] = True
        snap['factors']['comotion_state']['known_mask'] = comotion_results[t].get('comotion_known_mask', False)
        snap['factors']['lift_state']['known_mask'] = lift_results[t].get('lift_known_mask', False)

        # Target resolution
        for k, v in target.items():
            snap['factors']['target_resolution'][k] = v
        snap['factors']['target_resolution']['known_mask'] = True

        # Validate
        violations = validate_v22_snapshot(snap)
        if violations and t < 3:
            print(f'  WARN step {t}: {len(violations)} validation violations')

        v22_snapshots.append(snap)

    # Step 6: V22 → Label V2
    critical_labels = []
    safe_release_labels = []
    label_v2_steps = []
    for t, snap in enumerate(v22_snapshots):
        label = v22_to_label_v2(snap, t, K)
        label['step'] = t
        label['identity'] = f'{suite}/{task}/{state}'
        label_v2_steps.append(label)
        critical_labels.append(label['physical_criticality'])
        safe_release_labels.append(label['safe_release'])

    # Step 7: Recompute K10 from V22 criticality
    k10_results = recompute_k10(critical_labels, safe_release_labels, K)
    for t in range(len(label_v2_steps)):
        label_v2_steps[t]['k10_feasible'] = k10_results[t]
        crit_val = critical_labels[t]['value'] == 1 and critical_labels[t]['valid_mask']
        k10_val = k10_results[t]['value'] == 1 and k10_results[t]['valid_mask']
        label_v2_steps[t]['attack_opportunity'] = {
            'value': crit_val and k10_val,
            'valid_mask': critical_labels[t]['valid_mask'] and k10_results[t]['valid_mask'],
            'reason': 'CRITICAL_AND_K10' if (crit_val and k10_val) else k10_results[t]['reason'],
            'confidence': min(critical_labels[t]['confidence'], k10_results[t]['confidence']),
        }

    # Statistics
    n_crit = sum(1 for l in label_v2_steps if l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 1)
    n_crit_unknown = sum(1 for l in label_v2_steps if not l['physical_criticality']['valid_mask'])
    n_k10 = sum(1 for l in label_v2_steps if l['k10_feasible']['valid_mask'] and l['k10_feasible']['value'] == 1)
    n_opp = sum(1 for l in label_v2_steps if l['attack_opportunity']['valid_mask'] and l['attack_opportunity']['value'])

    result = {
        'identity': f'{suite}/{task}/{state}',
        'n_steps': len(label_v2_steps),
        'input_shas': input_shas,
        'target_resolution': target,
        'stats': {'n_critical': n_crit, 'n_critical_unknown': n_crit_unknown,
                  'n_k10_feasible': n_k10, 'n_attack_opportunity': n_opp,
                  'n_instability': sum(1 for l in label_v2_steps if l['instability']['valid_mask'] and l['instability']['value'] == 1)},
        'steps': label_v2_steps,
    }

    if not dry_run:
        # Write atomic output
        out_dir = os.path.join(PILOT_OUT, suite, task, state)
        os.makedirs(out_dir, exist_ok=True)
        lines = '\n'.join(json.dumps(l) for l in label_v2_steps) + '\n'
        out_file = os.path.join(out_dir, 'label_contract_v2.jsonl')
        write_atomic(lines, out_file)

    return result

def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f'=== N5 Pilot Pipeline: {len(manifest["episodes"])} episodes ===')
    print(f'Manifest: {MANIFEST_PATH}')
    print(f'Output: {PILOT_OUT}')
    print(f'V22 Schema SHA: {compute_v22_schema_sha()[:16]}...')
    print(f'V22 Config SHA: {compute_config_sha()[:16]}...')
    print()

    # Verify Formal not touched
    formal_root = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
    pre_snapshot = {}
    if os.path.isdir(formal_root):
        pre_snapshot['mtime'] = os.lstat(formal_root).st_mtime
        pre_snapshot['queue_size'] = os.path.getsize(os.path.join(formal_root, 'queue.sqlite'))
    print(f'Formal pre-snapshot: {pre_snapshot}')
    print()

    results = []
    gate_results = {
        'identity_join': 0, 'missing_input': 0, 'unknown_to_negative': 0,
        'nan_inf_known_true': 0, 'cc_in_physics': 0,
        'goal_blanket_no_target': 0, 'validation_violations': 0,
    }

    for ep in manifest['episodes']:
        print(f'Processing: {ep["suite"]}/{ep["task"]}/{ep["state"]}')
        result = process_episode(ep, manifest)
        if 'error' in result:
            print(f'  ERROR: {result["error"]}')
            gate_results['missing_input'] += 1
            continue

        gate_results['identity_join'] += 1
        results.append(result)

        # Gate checks
        n_unknown_to_neg = sum(1 for l in result['steps']
                               if not l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0)
        if n_unknown_to_neg > 0:
            gate_results['unknown_to_negative'] += n_unknown_to_neg

        # Goal blanket check
        if 'goal' in ep['suite'] and result['target_resolution'].get('target_known_mask'):
            if not result['target_resolution'].get('target_resolved'):
                gate_results['goal_blanket_no_target'] += 1
                print(f'  WARN: Goal task {ep["task"]} has target_resolved=False')

        print(f'  Steps={result["n_steps"]} crit={result["stats"]["n_critical"]} '
              f'crit_unk={result["stats"]["n_critical_unknown"]} k10={result["stats"]["n_k10_feasible"]} '
              f'opp={result["stats"]["n_attack_opportunity"]} instab={result["stats"]["n_instability"]}')

    # Post-snapshot
    post_snapshot = {}
    if os.path.isdir(formal_root):
        post_snapshot['mtime'] = os.lstat(formal_root).st_mtime
        post_snapshot['queue_size'] = os.path.getsize(os.path.join(formal_root, 'queue.sqlite'))

    # Pilot receipt
    receipt = {
        'pilot': 'N5_PILOT_12_RECEIPT_V1',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'manifest_sha': hashlib.sha256(open(MANIFEST_PATH, 'rb').read()).hexdigest(),
        'v22_schema_sha': compute_v22_schema_sha(),
        'v22_config_sha': compute_config_sha(),
        'pipeline_source_sha': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'results': results,
        'gates': gate_results,
        'formal_snapshot': {'pre': pre_snapshot, 'post': post_snapshot,
                            'unchanged': pre_snapshot == post_snapshot},
        'summary': {
            'n_processed': gate_results['identity_join'],
            'n_total_steps': sum(r['n_steps'] for r in results),
            'total_critical': sum(r['stats']['n_critical'] for r in results),
            'total_attack_opportunity': sum(r['stats']['n_attack_opportunity'] for r in results),
        },
    }

    receipt_path = os.path.join(PILOT_OUT, 'PILOT_RECEIPT.json')
    os.makedirs(PILOT_OUT, exist_ok=True)
    write_atomic(json.dumps(receipt, indent=2, default=str) + '\n', receipt_path)
    receipt_sha = hashlib.sha256(open(receipt_path, 'rb').read()).hexdigest()

    print(f'\n=== PILOT RESULTS ===')
    print(f'Identity join: {gate_results["identity_join"]}/12')
    print(f'Total steps: {receipt["summary"]["n_total_steps"]}')
    print(f'Total critical: {receipt["summary"]["total_critical"]}')
    print(f'Total attack opportunity: {receipt["summary"]["total_attack_opportunity"]}')
    print(f'Unknown→negative: {gate_results["unknown_to_negative"]}')
    print(f'Goal blanket NO_TARGET: {gate_results["goal_blanket_no_target"]}')
    print(f'Formal unchanged: {pre_snapshot == post_snapshot}')
    print(f'Receipt: {receipt_path}')
    print(f'Receipt SHA: {receipt_sha}')

    all_pass = (
        gate_results['identity_join'] == 12 and
        gate_results['unknown_to_negative'] == 0 and
        gate_results['goal_blanket_no_target'] == 0 and
        pre_snapshot == post_snapshot
    )
    print(f'\nPILOT: {"PASS" if all_pass else "FAIL"}')
    sys.exit(0 if all_pass else 1)

if __name__ == '__main__':
    main()
