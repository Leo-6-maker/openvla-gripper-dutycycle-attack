"""Build FEC infrastructure audit bundle. Read-only, no rollout."""
import json, os, hashlib

E = '/mnt/sdc/dty_user/openvla_attack_evidence'
AUDIT_DIR = E + '/fec_infra_audit_v1'
os.makedirs(AUDIT_DIR, exist_ok=True)

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

# ── 1. CANONICAL ENTRYPOINT AUDIT ──
entrypoint_audit = {
    'schema': 'FEC_CANONICAL_ENTRYPOINT_AUDIT_V1',
    'canonical_attack_runner': {
        'file': 'scripts/v4_run_eval_openvla.py',
        'function': 'run_real()', 'status': 'ACTIVE_MAINTAINED',
        'description': 'V4 attack rollout loop: env step -> clean decode -> trigger -> budget -> attack -> re-decode -> env step'
    },
    'canonical_attack_engine': {
        'file': 'src/gripper_attack/attack_adapter.py',
        'class': 'TokenPrefixPGDAttacker (via OpenVLAVisualAttacker facade)',
        'status': 'ACTIVE_MAINTAINED'
    },
    'canonical_env_factory': {
        'file': 'src/gripper_attack/libero_v4_env_factory.py',
        'function': 'build_v4_exact_env()', 'status': 'ACTIVE'
    },
    'existing_detector_systems': [
        {'name': 'SC5DetectorRuntime', 'features': '25D', 'status': 'LEGACY_ACTIVE'},
        {'name': 'B3V3StreamingRuntime', 'features': '25D GRU', 'status': 'LEGACY'},
        {'name': 'FactorizedV2SchedulerAdapter', 'heads': '3 (grasp/manip/release)', 'status': 'INCOMPATIBLE'},
        {'name': 'N4 Detector (frozen)', 'heads': '1 (raw logit)', 'status': 'FROZEN_NOT_INTEGRATED'}
    ],
    'key_finding': 'NO existing runner integrates frozen 51D N4 Detector. All runners use 25D SC5 or 3-head factorized detectors.',
    'canonical_attack_protocol': {
        'file': 'src/gripper_attack/b3_v3_attack_protocol.py',
        'status': 'MANIFEST_BUILDER_ONLY (does not execute rollouts)',
        'conditions': ['CLEAN', 'R9Q_DETECTOR_T10', 'RAND_VALID_T10', 'COMMAND_OPEN_ORACLE', 'DETECTOR_SHUFFLED_GRAD_T10', 'R9Q_GRIPPER_ONLY_T10']
    },
    'launcher_scripts': [
        'scripts/run_attack_pipeline.py (CLI -> v4_run_eval_openvla.py)',
        'scripts/run_formal_attack_matrix_one_shot.sh (batch launcher)'
    ]
}
with open(os.path.join(AUDIT_DIR, 'FEC_CANONICAL_ENTRYPOINT_AUDIT_V1.json'), 'w') as f:
    json.dump(entrypoint_audit, f, indent=2)
print('1/9: Entrypoint')

# ── 2. FIVE-ARM SUPPORT MATRIX ──
arm_matrix = {
    'schema': 'FEC_FIVE_ARM_SUPPORT_MATRIX_V1',
    'arms': {
        'CLEAN': {'env': 'SUPPORTED', 'detector': 'PARTIAL', 'k10': 'N/A', 'telemetry': 'SUPPORTED',
                  'gap': 'N4 Detector logging not integrated'},
        'TRUE_T10': {'env': 'SUPPORTED', 'detector': 'MISSING', 'k10': 'PARTIAL', 'payload': 'SUPPORTED',
                     'first_emit': 'MISSING',
                     'gap': 'MUST add N4 online trigger, emit-relative K=10, first-emit binding'},
        'RAND_T10': {'env': 'SUPPORTED', 'detector': 'MISSING', 'k10': 'PARTIAL', 'payload': 'MISSING',
                     'matched_direction': 'PARTIAL',
                     'gap': 'Need matched RAND: same emit step/time/K/epsilon as TRUE, random attack direction only'},
        'COMMAND_OPEN_ORACLE': {'env': 'SUPPORTED', 'detector': 'MISSING', 'k10': 'PARTIAL',
                                'gripper_override': 'PARTIAL', 'arm_preservation': 'UNVERIFIED',
                                'gap': 'Verify action override preserves arm dims for exactly K=10 steps from emit'},
        'RANDOM_TIME_T10': {'env': 'SUPPORTED', 'detector': 'DIAGNOSTIC_ONLY', 'k10': 'PARTIAL',
                            'random_sampler': 'MISSING', 'detector_independent': 'MISSING',
                            'gap': 'LARGEST GAP. Need frozen pre-rollout random time sampler, K10-executable validation, independent of Detector/Teacher/outcome'}
    },
    'summary': {'fully_supported': 0, 'partially_supported': 5,
                'critical_missing': ['N4 Detector integration', 'RANDOM_TIME sampler', 'Matched RAND direction', 'Emit-relative K=10']}
}
with open(os.path.join(AUDIT_DIR, 'FEC_FIVE_ARM_SUPPORT_MATRIX_V1.json'), 'w') as f:
    json.dump(arm_matrix, f, indent=2)
print('2/9: Arm matrix')

# ── 3. DETECTOR INTEGRATION AUDIT ──
detector_audit = {
    'schema': 'FEC_DETECTOR_INTEGRATION_AUDIT_V1',
    'frozen_spec': {'checkpoint': 'o0_i0', 'input_dim': 51, 'platt_a': 0.5190011735319306,
                    'platt_b': 0.812702331013635, 'threshold': 0.855, 'persistence': 6,
                    'candidate_close': 'ON (factorized_teacher_v1.jsonl)', 'one_shot_latch': 'ON'},
    'incompatible_systems': [
        {'name': 'SC5DetectorRuntime', 'reason': '25D features, 3 heads'},
        {'name': 'B3V3StreamingRuntime', 'reason': '25D features, GRU not CausalTCN'},
        {'name': 'FactorizedV2SchedulerAdapter', 'reason': '3-head scheduler vs 1-head N4'}
    ],
    'integration_gaps': [
        'No N4Encoder runtime module in src/gripper_attack/',
        'No N4 trigger class in src/gripper_attack/triggers.py',
        'No 51D feature computer in V4 runner',
        'No Platt calibrator in V4 runner',
        'No candidate_close + threshold + persistence + one-shot scheduler in V4 runner'
    ],
    'recommended_approach': 'Create N4RuntimeTrigger wrapping N4Encoder + Platt + scheduler. Integrate via make_trigger() factory. Compute 51D features online from LIBERO obs + VLA decode.',
    'forbidden': ['Modify checkpoint/Platt/threshold/persistence/candidate_close/one-shot/feature-order/proxy-definition']
}
with open(os.path.join(AUDIT_DIR, 'FEC_DETECTOR_INTEGRATION_AUDIT_V1.json'), 'w') as f:
    json.dump(detector_audit, f, indent=2)
print('3/9: Detector')

# ── 4. GAP REGISTER ──
gaps = {
    'schema': 'FEC_IMPLEMENTATION_GAP_REGISTER_V1',
    'gaps': [
        {'id': 'GAP-01', 'severity': 'BLOCKING', 'component': 'N4 Detector runtime module',
         'desc': 'No runtime module loading o0_i0 ckpt, computing 51D features, applying Platt, running scheduler', 'effort': 'MEDIUM'},
        {'id': 'GAP-02', 'severity': 'BLOCKING', 'component': 'N4 trigger in V4 runner',
         'desc': 'V4 make_trigger() has no N4 trigger class', 'effort': 'MEDIUM'},
        {'id': 'GAP-03', 'severity': 'BLOCKING', 'component': 'Emit-relative K=10 budget',
         'desc': 'V4 uses rho ratio budget, not emit-relative K=10', 'effort': 'MEDIUM'},
        {'id': 'GAP-04', 'severity': 'BLOCKING', 'component': 'RANDOM_TIME sampler',
         'desc': 'No frozen pre-rollout random time sampler with K10-executable validation', 'effort': 'MEDIUM'},
        {'id': 'GAP-05', 'severity': 'BLOCKING', 'component': 'Matched RAND direction',
         'desc': 'FEC RAND must share TRUE emit/K/epsilon, vary only attack direction', 'effort': 'LOW'},
        {'id': 'GAP-06', 'severity': 'HIGH', 'component': 'ORACLE action override',
         'desc': 'Verify gripper-only override with arm preservation for K=10', 'effort': 'LOW'},
        {'id': 'GAP-07', 'severity': 'HIGH', 'component': 'Paired arm execution',
         'desc': 'No mechanism to run 5 arms per parent with shared seed/environment', 'effort': 'MEDIUM'},
        {'id': 'GAP-08', 'severity': 'MEDIUM', 'component': 'N4 telemetry fields',
         'desc': 'Step records lack N4 fields: raw_logit, cal_prob, cc, counter, latch', 'effort': 'LOW'},
        {'id': 'GAP-09', 'severity': 'MEDIUM', 'component': 'Multi-worker concurrency',
         'desc': 'Need process-level parallelism with separate output dirs', 'effort': 'LOW'},
        {'id': 'GAP-10', 'severity': 'LOW', 'component': 'FEC parent pool',
         'desc': 'CS200 pool is for formal attack. FEC pilot needs separate canary pool', 'effort': 'LOW'}
    ],
    'summary': {'total': 10, 'blocking': 5, 'high': 2, 'medium': 2, 'low': 1}
}
with open(os.path.join(AUDIT_DIR, 'FEC_IMPLEMENTATION_GAP_REGISTER_V1.json'), 'w') as f:
    json.dump(gaps, f, indent=2)
print('4/9: Gaps')

# ── 5-9. Remaining audits ──
telemetry = {'schema': 'FEC_TELEMETRY_SCHEMA_AUDIT_V1',
    'existing': 'V4 step_records.jsonl (actions, trigger, budget, grasp, qpos)',
    'missing': ['raw_student_logit','calibrated_probability','candidate_close','persistence_counter',
                'latch_state','detector_emit','attack_frame_index','perturbation_norm','first_emit_step',
                'video_path','telemetry_hash','parent_id','arm','ITT_denominator']}
with open(os.path.join(AUDIT_DIR, 'FEC_TELEMETRY_SCHEMA_AUDIT_V1.json'), 'w') as f:
    json.dump(telemetry, f, indent=2)

multiworker = {'schema': 'FEC_MULTIWORKER_SAFETY_AUDIT_V1',
    'pattern': 'Process-level (separate Python process per GPU)',
    'gpu07_6_workers': 'SUPPORTED', 'env_isolation': 'SUPPORTED',
    'output_isolation': 'SUPPORTED', 'seed_isolation': 'SUPPORTED',
    'gaps': ['Attempt ledger missing', 'Paired arm output isolation needs care']}
with open(os.path.join(AUDIT_DIR, 'FEC_MULTIWORKER_SAFETY_AUDIT_V1.json'), 'w') as f:
    json.dump(multiworker, f, indent=2)

parent_pool = {'schema': 'FEC_PARENT_POOL_INVENTORY_V1',
    'CS200': {'count': 200, 'status': 'FORMAL_ATTACK_POOL', 'overlap_concern': 'Consuming for pilot wastes formal parents'},
    'forbidden_overlap': ['DEV2(1300)', 'C4(200)', 'P4(300)', 'H2(200)'],
    'gap': 'No dedicated FEC canary pool. Need separate 20-40 parent identities for pilot.'}
with open(os.path.join(AUDIT_DIR, 'FEC_PARENT_POOL_INVENTORY_V1.json'), 'w') as f:
    json.dump(parent_pool, f, indent=2)

param_lineage = {'schema': 'FEC_ATTACK_PARAMETER_LINEAGE_V1',
    'payload': {'epsilon': 0.03, 'num_steps': 5, 'objective': 'autoregressive_prefix_gripper_target_token_logratio_arm_v3',
                'target_token_id': 31744, 'k10': 10},
    'detector': {'tau': 0.855, 'persistence': 6, 'platt_a': 0.5190011735319306, 'platt_b': 0.812702331013635},
    'source_sha': {'attack_adapter': '50635f5c65ed3929b102270755ea94bb9cc77187d7d974955a673fa3293b6d84'}}
with open(os.path.join(AUDIT_DIR, 'FEC_ATTACK_PARAMETER_LINEAGE_V1.json'), 'w') as f:
    json.dump(param_lineage, f, indent=2)

print('5-8/9: Telemetry, multiworker, parent pool, param lineage')

# ── SHA256SUMS ──
all_files = []
for root, dirs, fns in os.walk(AUDIT_DIR):
    for fn in sorted(fns):
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, AUDIT_DIR)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
        all_files.append((rel, sha256_file(fp)))
sums_path = os.path.join(AUDIT_DIR, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(AUDIT_DIR, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))

print('9/9: Sealed ({} files, sha={})'.format(len(all_files), sums_sha[:16]))
