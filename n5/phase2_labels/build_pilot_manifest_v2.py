"""N5 Pilot Manifest Upgrader: V1 → V2 with file-level SHAs, step counts, target bindings.

G3 requirement: File-level manifest with expected inputs, SHAs, and target semantics.
Must run on Linux server with CS200 and LIBERO access.
"""
import json, os, sys, hashlib

# Paths
MANIFEST_V1_PATH = os.path.join(os.path.dirname(__file__), 'pilot_12_manifest.json')
MANIFEST_V2_PATH = os.path.join(os.path.dirname(__file__), 'pilot_12_manifest_v2.json')
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from physics_teacher_v22 import compute_v22_schema_sha
from v22_production_v2 import (
    parse_sidecar, get_object_slices_for_task, resolve_goal_target,
    GOAL_TASK_TARGETS,
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(MANIFEST_V1_PATH) as f:
        v1 = json.load(f)

    episodes_v2 = []
    issues = []

    for ep in v1['episodes']:
        suite = ep['suite']; task = ep['task']; state = ep['state']
        task_idx = int(task.replace('task_', ''))
        identity = f'{suite}/{task}/{state}'

        # Verify input files exist and compute SHAs
        sidecar_path = os.path.join(CS200_ROOT, suite, task, state, 'privileged_teacher_sidecar.jsonl')
        summary_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_summary.json')
        metadata_path = os.path.join(CS200_ROOT, suite, task, state, 'episode_metadata.json')

        input_files = {}
        for path in [sidecar_path, summary_path, metadata_path]:
            if os.path.isfile(path):
                input_files[os.path.basename(path)] = {
                    'path': path,
                    'size_bytes': os.path.getsize(path),
                    'sha256': sha256_file(path),
                }
            else:
                issues.append(f'MISSING: {path}')
                continue

        # Parse sidecar for step count and instruction
        parsed = parse_sidecar(sidecar_path)
        n_steps = parsed['n_steps']
        instruction = parsed['steps'][0].get('task_language', '') if parsed['steps'] else ''

        # BDDL resolution
        bddl_info = get_object_slices_for_task(suite, task_idx)
        if bddl_info is not None:
            object_names = list(bddl_info['object_slices'].keys())
            manipulated = bddl_info['task_role']['manipulated_objects']
            target_names = bddl_info['task_role']['target_names']
            support_names = bddl_info['task_role']['support_names']
            bddl_status = bddl_info['task_role']['status']
        else:
            object_names = []
            manipulated = []
            target_names = []
            support_names = []
            bddl_status = 'BDDL_UNAVAILABLE'

        # Target semantics
        target_result = resolve_goal_target(instruction, bddl_info['object_slices'] if bddl_info else {})

        ep_v2 = {
            'suite': suite,
            'task': task,
            'state': state,
            'task_idx': task_idx,
            'purpose': ep['purpose'],
            'expected_instruction': instruction,
            'n_steps': n_steps,
            'input_files': input_files,
            'bddl': {
                'available': bddl_info is not None,
                'status': bddl_status,
                'object_names': object_names,
                'manipulated_objects': manipulated,
                'target_names': target_names,
                'support_names': support_names,
                'n_objects': len(object_names),
            },
            'target_semantics': {
                'task_semantics_known': target_result['task_semantics_known'],
                'physical_binding_known': target_result['physical_binding_known'],
                'target_resolved': target_result['target_resolved'],
                'target_object_id': target_result['target_object_id'],
                'reason': target_result['reason'],
            },
        }
        episodes_v2.append(ep_v2)
        print(f'  {identity}: {n_steps} steps, {len(object_names)} objects, '
              f'target={target_result["reason"]}')

    manifest_v2 = {
        'manifest': 'N5_PILOT_12_MANIFEST_V2',
        'upgraded_from': 'N5_PILOT_12_MANIFEST_V1',
        'frozen_at': None,  # filled after build
        'v1_sha': sha256_file(MANIFEST_V1_PATH),
        'v22_schema_sha': compute_v22_schema_sha(),
        'selection_rule': v1['selection_rule'],
        'n_episodes': len(episodes_v2),
        'cs200_root': CS200_ROOT,
        'episodes': episodes_v2,
        'issues': issues,
    }

    import time
    manifest_v2['frozen_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest_v2['self_sha'] = None
    manifest_json = json.dumps(manifest_v2, indent=2)
    manifest_v2['self_sha'] = hashlib.sha256(manifest_json.encode()).hexdigest()
    # Re-serialize with self_sha included
    manifest_json = json.dumps(manifest_v2, indent=2)

    with open(MANIFEST_V2_PATH, 'w') as f:
        f.write(manifest_json + '\n')

    print(f'\nManifest V2 written: {MANIFEST_V2_PATH}')
    print(f'Self SHA: {manifest_v2["self_sha"]}')
    print(f'Episodes: {len(episodes_v2)}')
    print(f'Issues: {len(issues)}')
    for issue in issues:
        print(f'  - {issue}')

    if issues:
        print('\nWARNING: Input files missing — manifest incomplete')
        sys.exit(1)


if __name__ == '__main__':
    main()
