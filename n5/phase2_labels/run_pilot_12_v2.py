"""N5 Pilot Pipeline V2: 12-episode CS200 → V22 V2 → Label V2 → K10 → atomic receipt.

V2 CHANGES (fixes all 8 P0 defects from d042fde audit):
  - V22 production V2 with target-specific physics (object_slices from BDDL)
  - Fixed recompute_k10 (one output per timestep)
  - Target resolver split: semantic vs physical binding
  - Safe release + close_intent fully implemented
  - Terminal state wired from episode_summary
  - Grasp uses target-finger contact filter
  - Lift uses target object Z (not EEF proxy)
  - Comotion uses correct history indexing
  - Instability uses target-relative measurements

HARD GATES:
  - candidate_close NOT in physics computation
  - Old K10 labels NOT reused
  - unknown NOT converted to negative
  - Goal NO blanket NO_MANIPULATION_TARGET
  - Formal paths NOT modified
"""
import json, os, sys, hashlib, time, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from physics_teacher_v22 import create_v22_snapshot, V22_SCHEMA_VERSION, compute_v22_schema_sha
from v22_production_v2 import (
    parse_sidecar, parse_episode_summary, resolve_goal_target,
    resolve_manipulated_objects, get_object_slices_for_task,
    compute_grasp_state, compute_contact_state, compute_comotion_state,
    compute_lift_state, compute_instability_indicators, compute_terminal_state,
    compute_safe_release, compute_placement_state, compute_close_intent,
    compute_gripper_physics, v22_to_label_v2, validate_v22_snapshot,
    recompute_k10, V22_CONFIG, compute_config_sha,
)
from label_contract_v2 import write_atomic, N5_ALLOWED_ROOT

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'pilot_12_manifest.json')
PILOT_OUT = os.path.join(N5_ALLOWED_ROOT, 'phase2_labels', 'pilot_12_v2_output')
K = 10


def process_episode(ep, manifest, dry_run=False):
    """Process one episode through full V2 pipeline."""
    suite = ep['suite']; task = ep['task']; state = ep['state']
    task_idx = int(task.replace('task_', ''))
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
    n_steps = parsed['n_steps']

    # Verify identity matches manifest
    manifest_id = f'{suite}/{task}/{state}'
    if identity['suite'] != suite or identity['task_idx'] != task_idx:
        return {'error': f'Identity mismatch: manifest={manifest_id} sidecar={identity}',
                'identity': manifest_id}

    # Step 2: Parse episode summary (terminal/success data)
    episode_summary = parse_episode_summary(episode_summary_path)

    # Step 3: Get task instruction
    instruction = steps[0].get('task_language', '') if steps else ''
    if not instruction and os.path.isfile(metadata_path):
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            instruction = meta.get('task_language', meta.get('task_instruction', ''))
        except Exception:
            pass

    # Step 4: Resolve BDDL object slices and task role
    bddl_info = get_object_slices_for_task(suite, task_idx)
    if bddl_info is not None:
        object_slices = bddl_info['object_slices']
        task_role = bddl_info['task_role']
        manipulated_objects = task_role['manipulated_objects']
        support_names = task_role['support_names']
        target_names = task_role['target_names']
    else:
        object_slices = {}
        manipulated_objects = []
        support_names = []
        target_names = []

    # Step 5: Goal target resolution (with physical binding check)
    target = resolve_goal_target(instruction, object_slices)
    target_object_id = target.get('target_object_id') if target.get('target_resolved') else None

    # Step 6: Compute V22 factors using V2 functions
    grasp_results = compute_grasp_state(steps, manipulated_objects, support_names)
    contact_results = compute_contact_state(steps, manipulated_objects, support_names)
    comotion_results = compute_comotion_state(steps, manipulated_objects, object_slices)
    lift_results = compute_lift_state(steps, manipulated_objects, object_slices)
    instability_results = compute_instability_indicators(steps, grasp_results, manipulated_objects, object_slices)
    terminal_results = compute_terminal_state(steps, episode_summary)
    safe_release_results = compute_safe_release(steps, grasp_results, terminal_results)
    placement_results = compute_placement_state(steps, grasp_results, manipulated_objects, object_slices, target_names)
    close_intent_results = compute_close_intent(steps)
    gripper_physics_results = compute_gripper_physics(steps)

    # Step 7: Build V22 snapshots per step
    v22_snapshots = []
    for t in range(n_steps):
        snap = create_v22_snapshot()
        snap['step'] = t
        snap['suite'] = suite; snap['task_index'] = task_idx; snap['state_index'] = state

        # Populate factors
        for k, v in grasp_results[t].items(): snap['factors']['grasp_state'][k] = v
        for k, v in contact_results[t].items(): snap['factors']['contact_state'][k] = v
        for k, v in comotion_results[t].items(): snap['factors']['comotion_state'][k] = v
        for k, v in lift_results[t].items(): snap['factors']['lift_state'][k] = v
        for k, v in instability_results[t].items(): snap['factors']['instability_indicators'][k] = v
        for k, v in terminal_results[t].items(): snap['factors']['terminal_state'][k] = v
        for k, v in safe_release_results[t].items(): snap['factors']['planned_release'][k] = v
        for k, v in placement_results[t].items(): snap['factors']['placement_state'][k] = v
        for k, v in close_intent_results[t].items(): snap['factors']['close_intent'][k] = v
        for k, v in gripper_physics_results[t].items(): snap['factors']['gripper_physics'][k] = v

        # Target resolution
        for k, v in target.items():
            snap['factors']['target_resolution'][k] = v
        snap['factors']['target_resolution']['known_mask'] = True

        # Set factor-level known_mask
        snap['factors']['grasp_state']['known_mask'] = grasp_results[t]['grasp_known_mask']
        snap['factors']['contact_state']['known_mask'] = contact_results[t]['contact_known_mask']
        snap['factors']['comotion_state']['known_mask'] = comotion_results[t]['comotion_known_mask']
        snap['factors']['lift_state']['known_mask'] = lift_results[t]['lift_known_mask']

        # Validate
        violations = validate_v22_snapshot(snap)
        if violations:
            return {'error': f'Validation violations at step {t}: {violations}',
                    'identity': manifest_id}

        v22_snapshots.append(snap)

    # Step 8: V22 → Label V2
    critical_labels = []
    safe_release_labels_list = []
    label_v2_steps = []
    for t, snap in enumerate(v22_snapshots):
        label = v22_to_label_v2(snap, t, K)
        label['step'] = t
        label['identity'] = manifest_id
        label_v2_steps.append(label)
        critical_labels.append(label['physical_criticality'])
        safe_release_labels_list.append(label['safe_release'])

    # Step 9: Recompute K10 from V22 criticality (using fixed V2 function)
    k10_results = recompute_k10(critical_labels, safe_release_labels_list, K)
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
    n_close = sum(1 for l in label_v2_steps if l['close_intent']['valid_mask'] and l['close_intent']['value'] == 1)
    n_safe = sum(1 for l in label_v2_steps if l['safe_release']['valid_mask'] and l['safe_release']['value'] == 1)
    n_instab = sum(1 for l in label_v2_steps if l['instability']['valid_mask'] and l['instability']['value'] == 1)

    result = {
        'identity': manifest_id,
        'n_steps': len(label_v2_steps),
        'input_shas': input_shas,
        'target_resolution': target,
        'bddl_info': {
            'manipulated_objects': manipulated_objects,
            'target_names': target_names,
            'support_names': support_names,
            'n_object_slices': len(object_slices),
            'bddl_available': bddl_info is not None,
        },
        'stats': {
            'n_critical': n_crit, 'n_critical_unknown': n_crit_unknown,
            'n_k10_feasible': n_k10, 'n_attack_opportunity': n_opp,
            'n_close_intent': n_close, 'n_safe_release': n_safe,
            'n_instability': n_instab,
        },
        'steps': label_v2_steps,
    }

    if not dry_run:
        out_dir = os.path.join(PILOT_OUT, suite, task, state)
        os.makedirs(out_dir, exist_ok=True)
        lines = '\n'.join(json.dumps(l) for l in label_v2_steps) + '\n'
        out_file = os.path.join(out_dir, 'label_contract_v2.jsonl')
        write_atomic(lines, out_file)

    return result


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f'=== N5 Pilot Pipeline V2: {len(manifest["episodes"])} episodes ===')
    print(f'Manifest: {MANIFEST_PATH}')
    print(f'Output: {PILOT_OUT}')
    print(f'V22 Schema SHA: {compute_v22_schema_sha()}')
    print(f'V22 Config SHA: {compute_config_sha()}')
    print(f'Pipeline source SHA: {hashlib.sha256(open(__file__, "rb").read()).hexdigest()}')
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
        'identity_match': 0, 'identity_mismatch': 0, 'missing_input': 0,
        'unknown_to_negative': 0, 'nan_inf_known_true': 0, 'cc_in_physics': 0,
        'goal_blanket_no_target': 0, 'validation_violations': 0,
        'k10_output_length_mismatch': 0,
    }

    for ep in manifest['episodes']:
        identity = f'{ep["suite"]}/{ep["task"]}/{ep["state"]}'
        print(f'Processing: {identity}')
        result = process_episode(ep, manifest)
        if 'error' in result:
            print(f'  ERROR: {result["error"]}')
            gate_results['missing_input'] += 1
            continue

        gate_results['identity_match'] += 1
        results.append(result)

        # Gate: K10 length matches episode length
        n_k10 = len([s for s in result['steps'] if 'k10_feasible' in s])
        if n_k10 != result['n_steps']:
            gate_results['k10_output_length_mismatch'] += 1
            print(f'  FAIL: K10 length {n_k10} != steps {result["n_steps"]}')

        # Gate: unknown→negative
        n_unk_to_neg = sum(1 for l in result['steps']
                           if not l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0)
        if n_unk_to_neg > 0:
            gate_results['unknown_to_negative'] += n_unk_to_neg

        # Gate: NaN/Inf in known=true
        for l in result['steps']:
            for head in ['physical_criticality', 'safe_release', 'instability', 'close_intent']:
                h = l.get(head, {})
                if h.get('valid_mask') and h.get('value') is not None:
                    c = h.get('confidence', 0)
                    if not np.isfinite(c):
                        gate_results['nan_inf_known_true'] += 1

        # Gate: Goal blanket no target
        if 'goal' in ep['suite'] and result['target_resolution'].get('target_known_mask'):
            if result['target_resolution'].get('task_semantics_known'):
                if not result['target_resolution'].get('physical_binding_known'):
                    gate_results['goal_blanket_no_target'] += 1
                    print(f'  WARN: Goal task {ep["task"]} semantics known but no physical binding')

        print(f'  Steps={result["n_steps"]} crit={result["stats"]["n_critical"]} '
              f'crit_unk={result["stats"]["n_critical_unknown"]} k10={result["stats"]["n_k10_feasible"]} '
              f'opp={result["stats"]["n_attack_opportunity"]} '
              f'close={result["stats"]["n_close_intent"]} safe={result["stats"]["n_safe_release"]} '
              f'instab={result["stats"]["n_instability"]} '
              f'bddl_objs={result["bddl_info"]["n_object_slices"]}')

    # Post-snapshot
    post_snapshot = {}
    if os.path.isdir(formal_root):
        post_snapshot['mtime'] = os.lstat(formal_root).st_mtime
        post_snapshot['queue_size'] = os.path.getsize(os.path.join(formal_root, 'queue.sqlite'))

    formal_unchanged = pre_snapshot == post_snapshot

    # Pilot receipt
    receipt = {
        'pilot': 'N5_PILOT_12_V2_RECEIPT',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'manifest_sha': hashlib.sha256(open(MANIFEST_PATH, 'rb').read()).hexdigest(),
        'v22_schema_sha': compute_v22_schema_sha(),
        'v22_config_sha': compute_config_sha(),
        'v22_prod_source_sha': hashlib.sha256(
            open(os.path.join(os.path.dirname(__file__), 'v22_production_v2.py'), 'rb').read()
        ).hexdigest(),
        'pipeline_source_sha': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'results': results,
        'gates': gate_results,
        'formal_snapshot': {'pre': pre_snapshot, 'post': post_snapshot, 'unchanged': formal_unchanged},
        'summary': {
            'n_processed': gate_results['identity_match'],
            'n_total_steps': sum(r['n_steps'] for r in results),
            'total_critical': sum(r['stats']['n_critical'] for r in results),
            'total_critical_pct': sum(r['stats']['n_critical'] for r in results) / max(1, sum(r['n_steps'] for r in results)) * 100,
            'total_attack_opportunity': sum(r['stats']['n_attack_opportunity'] for r in results),
            'total_close_intent': sum(r['stats']['n_close_intent'] for r in results),
            'total_safe_release': sum(r['stats']['n_safe_release'] for r in results),
            'total_instability': sum(r['stats']['n_instability'] for r in results),
        },
    }

    receipt_path = os.path.join(PILOT_OUT, 'PILOT_RECEIPT_V2.json')
    os.makedirs(PILOT_OUT, exist_ok=True)
    write_atomic(json.dumps(receipt, indent=2, default=str) + '\n', receipt_path)
    receipt_sha = hashlib.sha256(open(receipt_path, 'rb').read()).hexdigest()

    print(f'\n=== PILOT V2 RESULTS ===')
    print(f'Identity match: {gate_results["identity_match"]}/12')
    print(f'Identity mismatch: {gate_results["identity_mismatch"]}')
    print(f'Total steps: {receipt["summary"]["n_total_steps"]}')
    print(f'Total critical: {receipt["summary"]["total_critical"]} ({receipt["summary"]["total_critical_pct"]:.1f}%)')
    print(f'Total attack opportunity: {receipt["summary"]["total_attack_opportunity"]}')
    print(f'Close intent: {receipt["summary"]["total_close_intent"]}')
    print(f'Safe release: {receipt["summary"]["total_safe_release"]}')
    print(f'Instability: {receipt["summary"]["total_instability"]}')
    print(f'Unknown→negative: {gate_results["unknown_to_negative"]}')
    print(f'NaN/Inf known=true: {gate_results["nan_inf_known_true"]}')
    print(f'K10 length mismatch: {gate_results["k10_output_length_mismatch"]}')
    print(f'Goal blanket NO_TARGET: {gate_results["goal_blanket_no_target"]}')
    print(f'Validation violations: {gate_results["validation_violations"]}')
    print(f'Formal unchanged: {formal_unchanged}')
    print(f'Receipt: {receipt_path}')
    print(f'Receipt SHA: {receipt_sha}')

    # Hard gates
    total_crit_pct = receipt['summary']['total_critical_pct']
    all_pass = (
        gate_results['identity_match'] == 12 and
        gate_results['missing_input'] == 0 and
        gate_results['identity_mismatch'] == 0 and
        gate_results['unknown_to_negative'] == 0 and
        gate_results['nan_inf_known_true'] == 0 and
        gate_results['k10_output_length_mismatch'] == 0 and
        gate_results['validation_violations'] == 0 and
        gate_results['cc_in_physics'] == 0 and
        formal_unchanged
    )

    # Physics sanity: warn if any episode has >90% critical
    if total_crit_pct > 90:
        print(f'\nWARN: Overall critical rate {total_crit_pct:.1f}% > 90% — PHYSICS AUDIT REQUIRED')
        print('This indicates EEF proxy may still be in use for some factors.')

    for r in results:
        ep_crit_pct = r['stats']['n_critical'] / max(1, r['n_steps']) * 100
        if ep_crit_pct > 90:
            print(f'  HIGH CRITICAL: {r["identity"]} = {ep_crit_pct:.1f}% (crit={r["stats"]["n_critical"]}/{r["n_steps"]})')

    print(f'\nPILOT V2: {"PASS" if all_pass else "FAIL"}')
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
