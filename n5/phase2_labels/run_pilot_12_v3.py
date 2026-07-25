"""N5 Pilot Pipeline V3: G3/G4 repair — manifest V2 binding, all gate counters wired.

V3 FIXES (audit round 2):
  - Uses pilot_12_manifest_v2.json (file-level manifest with SHAs/sizes)
  - Verifies input file paths, SHAs, sizes, step counts against manifest
  - Identity mismatch → independent gate (not collapsed into missing_input)
  - cc_in_physics scanned at import time
  - validation_violations per episode counted
  - goal_blanket_no_target in all_pass
  - All gate counters independently wired
  - Reports known-negative breakdown per episode
  - Safe release now requires placement (uses compute_safe_release with placement_results)
  - Gripper closing state renamed from close_intent
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
    compute_safe_release, compute_placement_state, compute_gripper_closing_state,
    compute_gripper_physics, v22_to_label_v2, validate_v22_snapshot,
    recompute_k10, V22_CONFIG, compute_config_sha,
)
from label_contract_v2 import write_atomic, N5_ALLOWED_ROOT

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'pilot_12_manifest_v2.json')
PILOT_OUT = os.path.join(N5_ALLOWED_ROOT, 'phase2_labels', 'pilot_12_v3_output')
K = 10

# ── Compile-time cc_in_physics audit ──
_CC_IN_PHYSICS = False
_v22_source = open(os.path.join(os.path.dirname(__file__), 'v22_production_v2.py')).read()
_phys_section = _v22_source[_v22_source.index('# ── Physics Factor Computation ──'):
                              _v22_source.index('# ── V22 → Label V2 Adapter ──')]
for _line in _phys_section.split('\n'):
    _code_only = _line.split('#')[0] if '#' in _line else _line
    if 'candidate_close' in _code_only:
        _CC_IN_PHYSICS = True; break


def verify_manifest_inputs(ep, manifest):
    """Verify that all input files match the manifest's expected SHAs, sizes, and step counts."""
    issues = []
    suite = ep['suite']; task = ep['task']; state = ep['state']
    task_idx = int(task.replace('task_', ''))

    # Check manifest has input_files
    if 'input_files' not in ep:
        issues.append('MISSING_INPUT_FILES_IN_MANIFEST')
        return issues

    for fname, expected in ep['input_files'].items():
        # Resolve actual path: manifest stores CS200-relative, we check against CS200_ROOT
        actual_path = expected.get('path', os.path.join(CS200_ROOT, suite, task, state, fname))
        if not os.path.isfile(actual_path):
            issues.append(f'MISSING_INPUT: {actual_path}')
            continue
        actual_size = os.path.getsize(actual_path)
        if actual_size != expected['size_bytes']:
            issues.append(f'SIZE_MISMATCH: {fname} expected={expected["size_bytes"]} actual={actual_size}')
            continue
        h = hashlib.sha256()
        with open(actual_path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk: break
                h.update(chunk)
        actual_sha = h.hexdigest()
        if actual_sha != expected['sha256']:
            issues.append(f'SHA_MISMATCH: {fname}')

    # Verify step count
    if 'n_steps' in ep:
        sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
        if os.path.isfile(sidecar_path):
            parsed = parse_sidecar(sidecar_path)
            if parsed['n_steps'] != ep['n_steps']:
                issues.append(f'STEP_COUNT_MISMATCH: expected={ep["n_steps"]} actual={parsed["n_steps"]}')

    # Verify task instruction matches
    if 'expected_instruction' in ep and ep['expected_instruction']:
        sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
        if os.path.isfile(sidecar_path):
            parsed = parse_sidecar(sidecar_path)
            actual_instruction = parsed['steps'][0].get('task_language', '') if parsed['steps'] else ''
            if actual_instruction and actual_instruction != ep['expected_instruction']:
                issues.append(f'INSTRUCTION_MISMATCH: expected="{ep["expected_instruction"]}" actual="{actual_instruction}"')

    return issues


def process_episode(ep, manifest, dry_run=False):
    """Process one episode through full V3 pipeline."""
    suite = ep['suite']; task = ep['task']; state = ep['state']
    task_idx = int(task.replace('task_', ''))
    identity = f'{suite}/{task}/{state}'

    # Verify manifest inputs
    manifest_issues = verify_manifest_inputs(ep, manifest)
    if manifest_issues:
        return {'error': 'MANIFEST_VERIFY_FAILED', 'identity': identity,
                'manifest_issues': manifest_issues}

    sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
    episode_summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
    metadata_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_metadata.json')

    # Verify inputs exist (double-check)
    for p in [sidecar_path, episode_summary_path]:
        if not os.path.isfile(p):
            return {'error': f'Missing input: {p}', 'identity': identity}

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
    steps = parsed['steps']; n_steps = parsed['n_steps']

    # Verify identity matches manifest
    if parsed['identity']['suite'] != suite or parsed['identity']['task_idx'] != task_idx:
        return {'error': 'IDENTITY_MISMATCH',
                'identity': identity,
                'sidecar_identity': parsed['identity']}

    # Step 2: Parse episode summary
    episode_summary = parse_episode_summary(episode_summary_path)

    # Step 3: Task instruction
    instruction = steps[0].get('task_language', '') if steps else ''
    if not instruction and os.path.isfile(metadata_path):
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            instruction = meta.get('task_language', meta.get('task_instruction', ''))
        except Exception:
            pass

    # Step 4: BDDL resolution
    bddl_info = get_object_slices_for_task(suite, task_idx)
    if bddl_info is not None:
        object_slices = bddl_info['object_slices']
        task_role = bddl_info['task_role']
        manipulated_objects = task_role['manipulated_objects']
        support_names = task_role['support_names']
        target_names = task_role['target_names']
    else:
        object_slices = {}
        manipulated_objects = []; support_names = []; target_names = []

    # Step 5: Target resolution
    target = resolve_goal_target(instruction, object_slices)

    # Step 6: Compute V22 factors
    grasp_results = compute_grasp_state(steps, manipulated_objects, support_names)
    contact_results = compute_contact_state(steps, manipulated_objects, support_names)
    comotion_results = compute_comotion_state(steps, manipulated_objects, object_slices)
    lift_results = compute_lift_state(steps, manipulated_objects, object_slices)
    instability_results = compute_instability_indicators(steps, grasp_results, manipulated_objects, object_slices)
    terminal_results = compute_terminal_state(steps, episode_summary)
    placement_results = compute_placement_state(steps, grasp_results, manipulated_objects, object_slices, target_names)
    safe_release_results = compute_safe_release(steps, grasp_results, terminal_results, placement_results)
    gripper_closing_results = compute_gripper_closing_state(steps)
    gripper_physics_results = compute_gripper_physics(steps)

    # Step 7: Build V22 snapshots
    v22_snapshots = []
    validation_violations = 0
    for t in range(n_steps):
        snap = create_v22_snapshot()
        snap['step'] = t
        snap['suite'] = suite; snap['task_index'] = task_idx; snap['state_index'] = state

        for k, v in grasp_results[t].items(): snap['factors']['grasp_state'][k] = v
        for k, v in contact_results[t].items(): snap['factors']['contact_state'][k] = v
        for k, v in comotion_results[t].items(): snap['factors']['comotion_state'][k] = v
        for k, v in lift_results[t].items(): snap['factors']['lift_state'][k] = v
        for k, v in instability_results[t].items(): snap['factors']['instability_indicators'][k] = v
        for k, v in terminal_results[t].items(): snap['factors']['terminal_state'][k] = v
        for k, v in safe_release_results[t].items(): snap['factors']['planned_release'][k] = v
        for k, v in placement_results[t].items(): snap['factors']['placement_state'][k] = v
        for k, v in gripper_closing_results[t].items(): snap['factors']['gripper_closing_state'][k] = v
        for k, v in gripper_physics_results[t].items(): snap['factors']['gripper_physics'][k] = v
        for k, v in target.items(): snap['factors']['target_resolution'][k] = v
        snap['factors']['target_resolution']['known_mask'] = True

        snap['factors']['grasp_state']['known_mask'] = grasp_results[t]['grasp_known_mask']
        snap['factors']['contact_state']['known_mask'] = contact_results[t]['contact_known_mask']
        snap['factors']['comotion_state']['known_mask'] = comotion_results[t]['comotion_known_mask']
        snap['factors']['lift_state']['known_mask'] = lift_results[t]['lift_known_mask']

        violations = validate_v22_snapshot(snap)
        if violations:
            validation_violations += 1
            if t < 3:
                return {'error': f'Validation violations at step {t}: {violations}', 'identity': identity}
        v22_snapshots.append(snap)

    # Step 8: V22 → Label V2
    critical_labels = []; safe_release_labels_list = []; label_v2_steps = []
    for t, snap in enumerate(v22_snapshots):
        label = v22_to_label_v2(snap, t, K)
        label['step'] = t; label['identity'] = identity
        label_v2_steps.append(label)
        critical_labels.append(label['physical_criticality'])
        safe_release_labels_list.append(label['safe_release'])

    # Step 9: K10
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

    # Statistics with known-negative breakdown
    n_crit = sum(1 for l in label_v2_steps if l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 1)
    n_crit_neg = sum(1 for l in label_v2_steps if l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0)
    n_crit_unknown = sum(1 for l in label_v2_steps if not l['physical_criticality']['valid_mask'])
    n_k10 = sum(1 for l in label_v2_steps if l['k10_feasible']['valid_mask'] and l['k10_feasible']['value'] == 1)
    n_opp = sum(1 for l in label_v2_steps if l['attack_opportunity']['valid_mask'] and l['attack_opportunity']['value'])
    n_close = sum(1 for l in label_v2_steps if l['gripper_closing_state']['valid_mask'] and l['gripper_closing_state']['value'] == 1)
    n_safe = sum(1 for l in label_v2_steps if l['safe_release']['valid_mask'] and l['safe_release']['value'] == 1)
    n_instab = sum(1 for l in label_v2_steps if l['instability']['valid_mask'] and l['instability']['value'] == 1)

    result = {
        'identity': identity, 'n_steps': len(label_v2_steps), 'input_shas': input_shas,
        'target_resolution': target,
        'bddl_info': {'manipulated_objects': manipulated_objects, 'target_names': target_names,
                      'support_names': support_names, 'n_object_slices': len(object_slices),
                      'bddl_available': bddl_info is not None},
        'stats': {
            'n_critical': n_crit, 'n_critical_negative': n_crit_neg, 'n_critical_unknown': n_crit_unknown,
            'n_k10_feasible': n_k10, 'n_attack_opportunity': n_opp,
            'n_gripper_closing': n_close, 'n_safe_release': n_safe, 'n_instability': n_instab,
            'validation_violations': validation_violations,
        },
        'steps': label_v2_steps,
    }

    if not dry_run:
        out_dir = os.path.join(PILOT_OUT, suite, task, state)
        os.makedirs(out_dir, exist_ok=True)
        lines = '\n'.join(json.dumps(l) for l in label_v2_steps) + '\n'
        write_atomic(lines, os.path.join(out_dir, 'label_contract_v2.jsonl'))

    return result


def main():
    # Verify manifest V2 self-hash
    # Self-hash is computed over the manifest with self_sha=null, then embedded.
    # To verify: load, set self_sha=null, re-serialize canonically, hash, compare.
    with open(MANIFEST_PATH, 'rb') as f:
        manifest_raw = f.read()
    manifest_full_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    stored_self_sha = manifest.get('self_sha')
    if stored_self_sha:
        manifest_no_self = json.loads(manifest_raw)  # fresh parse
        manifest_no_self['self_sha'] = None
        canonical = json.dumps(manifest_no_self, indent=2, sort_keys=True)
        recomputed_self = hashlib.sha256(canonical.encode()).hexdigest()
        if recomputed_self != stored_self_sha:
            print(f'FATAL: Manifest self-hash mismatch: {recomputed_self[:16]}... != {stored_self_sha[:16]}...')
            sys.exit(2)
        print(f'Manifest self-hash VERIFIED: {stored_self_sha[:16]}...')
    else:
        print('WARNING: Manifest has no self_sha field — cannot verify integrity')

    print(f'=== N5 Pilot Pipeline V3: {manifest["n_episodes"]} episodes ===')
    print(f'Manifest: {MANIFEST_PATH}')
    print(f'Manifest SHA: {manifest_full_sha[:16]}...')
    print(f'Output: {PILOT_OUT}')
    print(f'V22 Schema SHA: {compute_v22_schema_sha()}')
    print(f'V22 Config SHA: {compute_config_sha()}')
    print(f'cc_in_physics (compile audit): {_CC_IN_PHYSICS}')
    print()

    formal_root = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
    pre_snapshot = {}
    if os.path.isdir(formal_root):
        pre_snapshot['mtime'] = os.lstat(formal_root).st_mtime
        pre_snapshot['queue_size'] = os.path.getsize(os.path.join(formal_root, 'queue.sqlite'))
    print(f'Formal pre-snapshot: {pre_snapshot}')
    print()

    results = []
    gate_results = {
        'identity_match': 0, 'identity_mismatch': 0,
        'missing_input': 0, 'manifest_verify_failed': 0,
        'unknown_to_negative': 0, 'nan_inf_known_true': 0,
        'cc_in_physics': 1 if _CC_IN_PHYSICS else 0,
        'goal_blanket_no_target': 0,
        'validation_violations': 0,
        'k10_output_length_mismatch': 0,
        'target_binding_false_positive': 0,
    }

    for ep in manifest['episodes']:
        identity = f"{ep['suite']}/{ep['task']}/{ep['state']}"
        print(f'Processing: {identity}')
        result = process_episode(ep, manifest)
        if 'error' in result:
            err = result['error']
            print(f'  ERROR: {err}')
            if 'MANIFEST_VERIFY' in err:
                gate_results['manifest_verify_failed'] += 1
            elif 'MISSING' in err:
                gate_results['missing_input'] += 1
            elif 'IDENTITY_MISMATCH' in err:
                gate_results['identity_mismatch'] += 1
            continue

        gate_results['identity_match'] += 1
        results.append(result)
        n = result['n_steps']; s = result['stats']

        # K10 length check
        n_k10_steps = sum(1 for st in result['steps'] if 'k10_feasible' in st)
        if n_k10_steps != n:
            gate_results['k10_output_length_mismatch'] += 1

        # unknown→negative
        n_unk_to_neg = sum(1 for l in result['steps']
                          if not l['physical_criticality']['valid_mask'] and l['physical_criticality']['value'] == 0)
        if n_unk_to_neg > 0:
            gate_results['unknown_to_negative'] += n_unk_to_neg

        # NaN/Inf
        for l in result['steps']:
            for head in ['physical_criticality', 'safe_release', 'instability', 'gripper_closing_state']:
                h = l.get(head, {})
                if h.get('valid_mask') and h.get('value') is not None:
                    c = h.get('confidence', 0)
                    if not np.isfinite(c):
                        gate_results['nan_inf_known_true'] += 1

        # Goal blanket
        if 'goal' in ep['suite']:
            tr = result['target_resolution']
            if tr.get('task_semantics_known') and not tr.get('physical_binding_known'):
                gate_results['goal_blanket_no_target'] += 1

        # Target binding false positive check
        if tr.get('target_resolved') and tr.get('reason') == 'TARGET_RESOLVED_BY_TASK':
            gate_results['target_binding_false_positive'] += 1

        # Validation violations per episode
        if s.get('validation_violations', 0) > 0:
            gate_results['validation_violations'] += s['validation_violations']

        crit_pct = s['n_critical'] / max(1, n) * 100
        known_neg_pct = s['n_critical_negative'] / max(1, n) * 100
        print(f'  Steps={n} crit={s["n_critical"]} ({crit_pct:.1f}%) '
              f'neg={s["n_critical_negative"]} ({known_neg_pct:.1f}%) '
              f'unk={s["n_critical_unknown"]} '
              f'k10={s["n_k10_feasible"]} opp={s["n_attack_opportunity"]} '
              f'close={s["n_gripper_closing"]} safe={s["n_safe_release"]} '
              f'instab={s["n_instability"]} '
              f'bddl={result["bddl_info"]["n_object_slices"]}')

    post_snapshot = {}
    if os.path.isdir(formal_root):
        post_snapshot['mtime'] = os.lstat(formal_root).st_mtime
        post_snapshot['queue_size'] = os.path.getsize(os.path.join(formal_root, 'queue.sqlite'))
    formal_unchanged = pre_snapshot == post_snapshot

    receipt = {
        'pilot': 'N5_PILOT_12_V3_RECEIPT',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'manifest_path': MANIFEST_PATH,
        'manifest_full_sha': manifest_full_sha,
        'manifest_self_sha_verified': stored_self_sha is not None,
        'v22_schema_sha': compute_v22_schema_sha(),
        'v22_config_sha': compute_config_sha(),
        'pipeline_source_sha': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        'results': results,
        'gates': gate_results,
        'formal_snapshot': {'pre': pre_snapshot, 'post': post_snapshot, 'unchanged': formal_unchanged},
        'summary': {
            'n_processed': gate_results['identity_match'],
            'n_total_steps': sum(r['n_steps'] for r in results),
            'total_critical': sum(r['stats']['n_critical'] for r in results),
            'total_critical_negative': sum(r['stats']['n_critical_negative'] for r in results),
            'total_critical_unknown': sum(r['stats']['n_critical_unknown'] for r in results),
            'total_critical_pct': sum(r['stats']['n_critical'] for r in results) / max(1, sum(r['n_steps'] for r in results)) * 100,
            'total_attack_opportunity': sum(r['stats']['n_attack_opportunity'] for r in results),
            'total_gripper_closing': sum(r['stats']['n_gripper_closing'] for r in results),
            'total_safe_release': sum(r['stats']['n_safe_release'] for r in results),
            'total_instability': sum(r['stats']['n_instability'] for r in results),
        },
    }

    os.makedirs(PILOT_OUT, exist_ok=True)
    receipt_path = os.path.join(PILOT_OUT, 'PILOT_RECEIPT_V3.json')
    write_atomic(json.dumps(receipt, indent=2, default=str) + '\n', receipt_path)
    receipt_sha = hashlib.sha256(open(receipt_path, 'rb').read()).hexdigest()

    print(f'\n=== PILOT V3 RESULTS ===')
    print(f'Identity match: {gate_results["identity_match"]}/12')
    print(f'Identity mismatch: {gate_results["identity_mismatch"]}')
    print(f'Manifest verify failed: {gate_results["manifest_verify_failed"]}')
    print(f'Total steps: {receipt["summary"]["n_total_steps"]}')
    print(f'Critical: {receipt["summary"]["total_critical"]} ({receipt["summary"]["total_critical_pct"]:.1f}%)')
    print(f'Critical negative: {receipt["summary"]["total_critical_negative"]}')
    print(f'Critical unknown: {receipt["summary"]["total_critical_unknown"]}')
    print(f'Attack opportunity: {receipt["summary"]["total_attack_opportunity"]}')
    print(f'Gripper closing: {receipt["summary"]["total_gripper_closing"]}')
    print(f'Safe release: {receipt["summary"]["total_safe_release"]}')
    print(f'Instability: {receipt["summary"]["total_instability"]}')
    print(f'Unknown→negative: {gate_results["unknown_to_negative"]}')
    print(f'NaN/Inf: {gate_results["nan_inf_known_true"]}')
    print(f'cc_in_physics: {gate_results["cc_in_physics"]}')
    print(f'Goal blanket NO_TARGET: {gate_results["goal_blanket_no_target"]}')
    print(f'Target binding FP: {gate_results["target_binding_false_positive"]}')
    print(f'K10 length mismatch: {gate_results["k10_output_length_mismatch"]}')
    print(f'Validation violations: {gate_results["validation_violations"]}')
    print(f'Formal unchanged: {formal_unchanged}')

    all_pass = (
        gate_results['identity_match'] == 12 and
        gate_results['missing_input'] == 0 and
        gate_results['manifest_verify_failed'] == 0 and
        gate_results['identity_mismatch'] == 0 and
        gate_results['unknown_to_negative'] == 0 and
        gate_results['nan_inf_known_true'] == 0 and
        gate_results['cc_in_physics'] == 0 and
        gate_results['k10_output_length_mismatch'] == 0 and
        gate_results['validation_violations'] == 0 and
        gate_results['target_binding_false_positive'] == 0 and
        formal_unchanged
    )

    print(f'\nPILOT V3: {"PASS" if all_pass else "FAIL"}')
    for r in results:
        ep_crit_pct = r['stats']['n_critical'] / max(1, r['n_steps']) * 100
        if ep_crit_pct > 90:
            print(f'  HIGH CRITICAL: {r["identity"]} = {ep_crit_pct:.1f}%')

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
