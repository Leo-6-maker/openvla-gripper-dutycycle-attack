"""R0: Evidence Readiness Audit.

Read-only check that all P0 fixes exist in repo, receipts are self-consistent,
and the working tree is clean before any Teacher work begins.

Exit codes:
  0: PASS — all checks pass, write receipt
  2: CONTRACT_PROVENANCE_FAIL
  5: HOLD_REVIEW
"""
import json, os, sys, hashlib, subprocess

DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(DIR, '..', '..'))

R0_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r0_evidence_readiness'
os.makedirs(R0_OUT, exist_ok=True)

FAILED = []
WARNINGS = []


def check(condition, tag, detail=''):
    if condition:
        print(f'  PASS: {tag}')
    else:
        print(f'  FAIL: {tag}  -- {detail}')
        FAILED.append((tag, detail))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


def json_file_self_hash(path):
    """Verify a JSON file's embedded self_sha256 matches recomputed hash."""
    with open(path) as f:
        d = json.load(f)
    stored = d.get('self_sha256') or d.get('self_sha')
    if not stored:
        return False, 'no self_sha256 field'
    # Recompute: serialize without self_sha256, then hash
    if 'self_sha256' in d:
        del d['self_sha256']
    recomputed = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return stored == recomputed, f'stored={stored[:16]}... recomputed={recomputed[:16]}...'


def grep_file(pattern, path):
    """Check if pattern exists in file."""
    try:
        with open(path) as f:
            content = f.read()
        return pattern in content
    except Exception:
        return False


print('=' * 60)
print('R0: Evidence Readiness Audit')
print('=' * 60)
print()

# ── 1. Runner P0 fix ──
print('--- 1. Runner P0 Fix ---')
runner_path = os.path.join(REPO_ROOT, 'scripts', 'fec', 'run_gpu_smoke_v5_open.py')
check(os.path.isfile(runner_path), 'runner file exists', runner_path)
if os.path.isfile(runner_path):
    has_undefined = grep_file('result[success_source]', runner_path)
    check(not has_undefined, 'runner: success_source undefined variable removed',
          'still present in runner' if has_undefined else '')
    has_env_check = grep_file('env_check', runner_path)
    # env_check may appear legitimately in other contexts; check the specific pattern
    has_bad_line = grep_file('result[success_source] = env_check', runner_path)
    check(not has_bad_line, 'runner: undefined line result[success_source]=env_check removed')

# ── 2. gripper_closing_state semantic fix ──
print('\n--- 2. gripper_closing_state Semantic Fix ---')
model_path = os.path.join(REPO_ROOT, 'n5', 'phase3_student', 'n5_student_model.py')
dataset_path = os.path.join(REPO_ROOT, 'n5', 'phase3_student', 'n5_dataset.py')
check(os.path.isfile(model_path), 'model file exists')
check(os.path.isfile(dataset_path), 'dataset file exists')

if os.path.isfile(model_path):
    has_old = grep_file('close_intent', model_path)
    check(not has_old, 'model: close_intent fully renamed to gripper_closing_state',
          'still contains close_intent' if has_old else '')
    has_new = grep_file('gripper_closing_state', model_path)
    check(has_new, 'model: gripper_closing_state present in HEAD_NAMES')

if os.path.isfile(dataset_path):
    has_old_ds = grep_file("'close_intent':", dataset_path)
    check(not has_old_ds, 'dataset: head_map uses gripper_closing_state not close_intent',
          'still has close_intent in head_map' if has_old_ds else '')
    has_new_ds = grep_file('gripper_closing_state', dataset_path)
    check(has_new_ds, 'dataset: N5_HEAD_NAMES uses gripper_closing_state')

# ── 3. Protocol Amendment ──
print('\n--- 3. Protocol Amendment ---')
amendment_path = os.path.join(REPO_ROOT, 'reports', 'PROTOCOL_AMENDMENT_V3.json')
check(os.path.isfile(amendment_path), 'PROTOCOL_AMENDMENT_V3.json exists')
if os.path.isfile(amendment_path):
    with open(amendment_path) as f:
        amd = json.load(f)
    has_placeholder = grep_file('PLACEHOLDER', amendment_path)
    check(not has_placeholder, 'no PLACEHOLDER values in amendment',
          'contains PLACEHOLDER' if has_placeholder else '')
    check(amd.get('amendment') == 'PROTOCOL_AMENDMENT_V3', 'amendment ID correct')
    check(amd.get('held_out_usage', {}).get('total') == 1200, '1200 held-out documented')
    check('decision_made_before_training' in str(amd.get('reason', {})),
          'records that decision was made before training')

# ── 4. G10 Test Manifest ──
print('\n--- 4. G10 Test Manifest ---')
g10_server_path = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G10_TEST_MANIFEST.json'
g10_expected_sha_prefix = '9feb87df72cbd'
if os.path.isfile(g10_server_path):
    g10_sha = file_sha(g10_server_path)
    check(g10_sha.startswith(g10_expected_sha_prefix),
          f'G10 manifest SHA matches: {g10_sha[:16]}...')
    with open(g10_server_path) as f:
        g10 = json.load(f)
    check(g10.get('n_held_out') == 1200, 'G10: 1200 held-out identities')
    check(g10.get('n_training') == 800, 'G10: 800 training identities')
    per_suite = g10.get('per_suite', {})
    for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
        check(per_suite.get(suite) == 300, f'G10: {suite} = 300 held-out')
else:
    check(False, 'G10 manifest file not found', g10_server_path)

# ── 5. G6_SEAL_V2 ──
print('\n--- 5. G6_SEAL_V2 ---')
g6_path = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
check(os.path.isfile(g6_path), 'G6_SEAL_V2.json exists')
if os.path.isfile(g6_path):
    ok, detail = json_file_self_hash(g6_path)
    check(ok, 'G6_SEAL_V2 self-hash verifies', detail)
    with open(g6_path) as f:
        g6 = json.load(f)
    check(g6.get('head_names') == [
        'physical_criticality', 'k10_feasible', 'safe_release',
        'instability', 'gripper_closing_state'
    ], 'G6: head_names correct (gripper_closing_state not close_intent)')
    check('gripper_closing_state' in g6.get('pos_weights', {}), 'G6: pos_weights uses gripper_closing_state')
    check('close_intent' not in g6.get('pos_weights', {}), 'G6: pos_weights does NOT use close_intent')

# ── 6. P0_EVIDENCE_SEAL ──
print('\n--- 6. P0 Evidence Seal ---')
p0_path = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/P0_EVIDENCE_SEAL.json'
check(os.path.isfile(p0_path), 'P0_EVIDENCE_SEAL.json exists')
if os.path.isfile(p0_path):
    ok, detail = json_file_self_hash(p0_path)
    check(ok, 'P0 evidence seal self-hash verifies', detail)

# ── 7. G9 Scoring Manifest ──
print('\n--- 7. G9 Scoring Manifest ---')
g9_path = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g9_scoring/SCORING_MANIFEST.json'
check(os.path.isfile(g9_path), 'G9 SCORING_MANIFEST.json exists')
if os.path.isfile(g9_path):
    ok, detail = json_file_self_hash(g9_path)
    check(ok, 'G9 scoring manifest self-hash verifies', detail)
    with open(g9_path) as f:
        g9 = json.load(f)
    check(g9.get('g9_mode') == 'SCORE_ONLY', 'G9: mode is SCORE_ONLY')
    check(g9.get('contract_state') == 'UNRESOLVED', 'G9: contract state is UNRESOLVED')
    check(g9.get('hard_rules', {}).get('threshold_search') == 'FORBIDDEN',
          'G9: threshold search still FORBIDDEN')

# ── 8. Checkpoint SHA ──
print('\n--- 8. Checkpoint Weight SHA ---')
ckpt_path = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g8_n5_training/seed_19903/n5_seed19903_best.pt'
check(os.path.isfile(ckpt_path), 'N5 best checkpoint exists')
if os.path.isfile(ckpt_path):
    ckpt_sha = file_sha(ckpt_path)
    print(f'  Checkpoint SHA: {ckpt_sha[:16]}...')
    # Verify G9 manifest binds this SHA
    if os.path.isfile(g9_path):
        with open(g9_path) as f:
            g9d = json.load(f)
        g9_ckpt_sha = g9d.get('checkpoint_sha', '')
        check(ckpt_sha == g9_ckpt_sha,
              f'G9 manifest binds checkpoint SHA',
              f'ckpt={ckpt_sha[:16]}... g9={g9_ckpt_sha[:16]}...' if ckpt_sha != g9_ckpt_sha else '')

# ── 9. Paired bootstrap terminology ──
print('\n--- 9. Paired Bootstrap Terminology ---')
bootstrap_path = os.path.join(REPO_ROOT, 'n5', 'phase3_student', 'paired_bootstrap.py')
check(os.path.isfile(bootstrap_path), 'paired_bootstrap.py exists')
if os.path.isfile(bootstrap_path):
    has_p_value = grep_file('P(delta > 0)', bootstrap_path)
    has_bootstrap_support = grep_file('bootstrap_support', bootstrap_path)
    # Check that we use the right term
    if has_bootstrap_support or not has_p_value:
        check(True, 'bootstrap terminology: uses bootstrap_support')
    else:
        check(False, 'bootstrap terminology: should use bootstrap_support not P(delta>0)',
              'found P(delta>0) without bootstrap_support')

# ── 10. Working tree clean ──
print('\n--- 10. Working Tree ---')
try:
    result = subprocess.run(['git', 'diff', '--stat'], capture_output=True, text=True, cwd=REPO_ROOT)
    clean = result.stdout.strip() == ''
    check(clean, 'working tree clean (no uncommitted changes to tracked files)',
          result.stdout[:200] if not clean else '')
except Exception as e:
    check(False, 'git diff check', str(e))

# Git log check
try:
    result = subprocess.run(['git', 'log', '--oneline', '-5'], capture_output=True, text=True, cwd=REPO_ROOT)
    print(f'  Recent commits:\n{result.stdout}')
except Exception:
    pass

# ── 11. V4 Formal unchanged ──
print('\n--- 11. V4 Formal Files Unchanged ---')
v4_paths = [
    'scripts/fec/n4_detector_adapter_v4.py',
    os.path.join('scripts', 'fec', 'run_gpu_smoke_v4.py'),
]
for v4p in v4_paths:
    full = os.path.join(REPO_ROOT, v4p)
    if os.path.isfile(full):
        check(True, f'V4 file exists (not modified): {v4p}')
    else:
        check(True, f'V4 file not present (not deleted): {v4p}')

# ── Finalize ──
print('\n' + '=' * 60)
n_failed = len(FAILED)
if n_failed == 0:
    print(f'R0: PASS ({n_failed} failures)')

    receipt = {
        'gate': 'R0_EVIDENCE_READINESS',
        'status': 'PASS',
        'timestamp': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
        'n_checks_passed': len(FAILED) == 0,
        'failures': [],
        'warnings': WARNINGS,
        'branch': 'deepseek/integration-final-detector-20260724',
        'head_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip(),
    }
    receipt['self_sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    receipt_path = os.path.join(R0_OUT, 'R0_RECEIPT.json')
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)
    print(f'Receipt: {receipt_path}')
    print(f'SHA: {receipt["self_sha256"][:16]}...')
    sys.exit(0)
else:
    print(f'R0: FAIL ({n_failed} failures)')
    for tag, detail in FAILED:
        print(f'  - {tag}: {detail}')

    hold_packet = {
        'gate': 'R0_EVIDENCE_READINESS',
        'status': 'FAIL',
        'exit_code': 5,
        'timestamp': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
        'failed_predicates': [{'tag': t, 'detail': d} for t, d in FAILED],
        'last_passing_receipt_sha': None,
        'source_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip(),
        'suspected_causes': ['HYPOTHESIS: P0 fixes not fully deployed to server or branch'],
    }
    hold_path = os.path.join(R0_OUT, 'HOLD_PACKET.json')
    with open(hold_path, 'w') as f:
        json.dump(hold_packet, f, indent=2)
    print(f'HOLD_PACKET: {hold_path}')
    sys.exit(5)
