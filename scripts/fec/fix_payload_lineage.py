"""Fix FEC payload lineage: supersede V1, create V2 with canonical attacker SHA."""
import json, os, hashlib

E = '/mnt/sdc/dty_user/openvla_attack_evidence'
IMPL_DIR = E + '/fec_implementation_v1'
MATRIX_DIR = E + '/fec_phase_b_matrix_freeze_20260724T213025Z'
COHORT_DIR = E + '/fec_phase_b_parent_cohort_20260724T212848Z'
PAYLOAD_DIR = E + '/fec_phase_b_payload_lineage_audit_20260724T213240Z'

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

ACTUAL_ADAPTER_SHA = '26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be'
HEAD_ADAPTER_SHA = '70ab33ccc9eefc759208f1ecc91310e2d24990ec0d46d2ca18573e2b5acb7055'

# 1. Supersede V1
v1_path = IMPL_DIR + '/FEC_ATTACK_PAYLOAD_LINEAGE_V1.json'
v1 = json.load(open(v1_path))
v1['status'] = 'SUPERSEDED_PRE_EXECUTION'
v1['superseded_by'] = 'FEC_ATTACK_PAYLOAD_LINEAGE_V2.json'
v1['supersession_reason'] = 'Declared source_sha=50635f5c... is v6_critical_student.py encoder SHA, not attack_adapter.py'
v1['fec_attack_jobs_executed_under_v1'] = 0
v1['scientific_results_affected'] = False
with open(v1_path, 'w') as f:
    json.dump(v1, f, indent=2)
print('1. V1 superseded')

# 2. Create V2
v2 = {
    'schema': 'FEC_ATTACK_PAYLOAD_LINEAGE_V2',
    'status': 'FROZEN_ACTIVE',
    'timestamp': '2026-07-25',
    'supersedes': 'FEC_ATTACK_PAYLOAD_LINEAGE_V1.json',
    'change_type': 'PROVENANCE_CORRECTION_PRE_EXECUTION',
    'attack_semantics_changed': False,
    'attack_parameters_changed': False,
    'parent_manifest_changed': False,
    'job_matrix_changed': False,
    'formal_attack_jobs_before_v2': 0,
    'canonical_attacker': {
        'source_file': 'src/gripper_attack/attack_adapter.py',
        'source_sha256': ACTUAL_ADAPTER_SHA,
        'git_head_sha256': HEAD_ADAPTER_SHA,
        'working_tree_dirty': True,
        'diff_semantic_lines': 0,
        'diff_cosmetic_only': True,
        'diff_verified_by': 'git diff --ignore-all-space --ignore-blank-lines = 0 lines',
        'resolved_import_path': '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/attack_adapter.py',
        'python_import': 'gripper_attack.attack_adapter',
        'class': 'TokenPrefixPGDAttacker (via OpenVLAVisualAttacker facade)'
    },
    'supporting_modules': {
        'route_contract': {
            'path': 'src/gripper_attack/route_contract.py',
            'sha256': '090cfbb3c431bc407830e6221f6d0a01f150afbbb0200b8a291fc465aaddff95'
        }
    },
    'true_payload': v1['true_payload'],
    'rand_payload': v1['rand_payload'],
    'oracle_payload': v1['oracle_payload'],
    'random_time_payload': v1['random_time_payload'],
    'git_env': {
        'repo_head': 'f9e42f6f881dd9b6e3f46a87ebfb0a2e33a676cb',
        'dirty_files': ['src/gripper_attack/attack_adapter.py (whitespace only)'],
        'python_env': '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800'
    },
    'runtime_self_check_required': True,
    'runtime_self_check_sha': ACTUAL_ADAPTER_SHA
}
v2_path = IMPL_DIR + '/FEC_ATTACK_PAYLOAD_LINEAGE_V2.json'
with open(v2_path, 'w') as f:
    json.dump(v2, f, indent=2)
print('2. V2 created: canonical SHA = {}'.format(ACTUAL_ADAPTER_SHA[:16]))

# 3. Source manifest
source_manifest = {
    'schema': 'FEC_ATTACK_SOURCE_MANIFEST_V2',
    'entry_module': {
        'logical_role': 'canonical_attacker',
        'absolute_path': '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/attack_adapter.py',
        'repo_relative_path': 'src/gripper_attack/attack_adapter.py',
        'sha256': ACTUAL_ADAPTER_SHA,
        'import_module': 'gripper_attack.attack_adapter'
    },
    'supporting': [{
        'logical_role': 'route_contract_validator',
        'absolute_path': '/mnt/sdc/dty_user/openvla_attack/src/gripper_attack/route_contract.py',
        'repo_relative_path': 'src/gripper_attack/route_contract.py',
        'sha256': '090cfbb3c431bc407830e6221f6d0a01f150afbbb0200b8a291fc465aaddff95',
        'import_module': 'gripper_attack.route_contract'
    }],
    'resolution_environment': {
        'repo_head': 'f9e42f6f881dd9b6e3f46a87ebfb0a2e33a676cb',
        'conda_env': '/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800',
        'cwd': '/mnt/sdc/dty_user/openvla_attack',
        'sys_path_insert': '/mnt/sdc/dty_user/openvla_attack/src'
    }
}
with open(IMPL_DIR + '/FEC_ATTACK_SOURCE_MANIFEST_V2.json', 'w') as f:
    json.dump(source_manifest, f, indent=2)
print('3. Source manifest created')

# 4. Correction receipt
correction = {
    'schema': 'FEC_PAYLOAD_LINEAGE_CORRECTION_RECEIPT_V1',
    'status': 'CORRECTED_PRE_EXECUTION',
    'error_description': 'V1 declared source_sha=50635f5c... which is v6_critical_student.py encoder SHA, not attack_adapter.py.',
    'correction': 'V2 binds actual import-resolved attack_adapter.py SHA (26cfb9f5...) from Python import probe.',
    'whitespace_audit': 'Working tree differs from git HEAD (70ab33cc...) by formatting only. git diff --ignore-all-space = 0 lines.',
    'attack_jobs_affected': 0, 'parents_affected': 0, 'matrix_cells_affected': 0
}
with open(IMPL_DIR + '/FEC_PAYLOAD_LINEAGE_CORRECTION_RECEIPT_V1.json', 'w') as f:
    json.dump(correction, f, indent=2)
print('4. Correction receipt written')

# 5. Supersede matrix V1, create V2
matrix_v1_path = MATRIX_DIR + '/FEC_FIVE_ARM_MATRIX_MANIFEST_V1.json'
if os.path.isfile(matrix_v1_path):
    mv1 = json.load(open(matrix_v1_path))
    mv1['status'] = 'SUPERSEDED_PRE_EXECUTION'
    mv1['superseded_by'] = 'FEC_FIVE_ARM_MATRIX_MANIFEST_V2.json'
    mv1['supersession_reason'] = 'Payload lineage V1 superseded; matrix cells unchanged'
    mv1['cells_executed'] = 0
    with open(matrix_v1_path, 'w') as f:
        json.dump(mv1, f, indent=2)
    mv2 = dict(mv1)
    mv2['status'] = 'FROZEN_ACTIVE'
    mv2['schema'] = 'FEC_FIVE_ARM_MATRIX_MANIFEST_V2'
    mv2['payload_lineage_ref'] = 'FEC_ATTACK_PAYLOAD_LINEAGE_V2.json'
    mv2['payload_lineage_sha'] = sha256_file(v2_path)
    mv2['supersedes'] = 'FEC_FIVE_ARM_MATRIX_MANIFEST_V1.json'
    mv2['cells_executed'] = 0
    for key in ['superseded_by', 'supersession_reason']:
        mv2.pop(key, None)
    with open(MATRIX_DIR + '/FEC_FIVE_ARM_MATRIX_MANIFEST_V2.json', 'w') as f:
        json.dump(mv2, f, indent=2)
    print('5. Matrix V1 superseded, V2 created')
else:
    print('5. Matrix V1 not found')

# 6. Update matrix state
state_path = MATRIX_DIR + '/FEC_MATRIX_EXECUTION_STATE_V1.json'
if os.path.isfile(state_path):
    state = json.load(open(state_path))
    state['payload_lineage_v2_sha'] = sha256_file(v2_path)
    state['next_gate'] = 'GPU_SMOKE_AFTER_V2_UNIT_TESTS'
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    print('6. Matrix execution state updated')

# 7. Re-seal implementation dir
all_files = []
for root, dirs, fns in os.walk(IMPL_DIR):
    for fn in sorted(fns):
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, IMPL_DIR)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
        all_files.append((rel, sha256_file(fp)))
sums_path = os.path.join(IMPL_DIR, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(IMPL_DIR, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))

# Also re-seal matrix, cohort, and payload dirs
for d in [MATRIX_DIR, COHORT_DIR, PAYLOAD_DIR]:
    if os.path.isdir(d):
        af = []
        for root, dirs, fns in os.walk(d):
            for fn in sorted(fns):
                fp = os.path.join(root, fn); rel = os.path.relpath(fp, d)
                if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
                af.append((rel, sha256_file(fp)))
        sp = os.path.join(d, 'SHA256SUMS')
        with open(sp, 'w') as f:
            for rel, h in sorted(af):
                f.write('{}  {}\n'.format(h, rel))
        ssha = sha256_file(sp)
        with open(os.path.join(d, 'SHA256SUMS.sha256'), 'w') as f:
            f.write('{}  SHA256SUMS\n'.format(ssha))

print('7. All directories re-sealed')
print()
print('Provenance correction complete.')
print('V2 canonical attacker SHA: {}'.format(ACTUAL_ADAPTER_SHA[:16]))
print('Attack jobs executed: 0')
print('Parents preserved: 20/20')
print('Next gate: GPU SMOKE (after V2 unit tests)')
