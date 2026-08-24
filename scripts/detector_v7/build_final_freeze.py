"""Build FINAL_DETECTOR_V23 qualified freeze bundle."""
import json, os, hashlib

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
BUNDLE = EVIDENCE + '/final_detector_v23_qualified_freeze_v1'
os.makedirs(BUNDLE, exist_ok=True)

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

# ── Evidence SHAs (from collected data) ──
EVIDENCE_SHAS = {
    'recipe': 'd12ff4e28d03fb2503d536a35c7b9fc830824115a3d6105de88d825c98bdcf61',
    'proxy_contract': 'fba74201d014a510adf6d4ecd98f080196bb2a805e36b461b97de8517f74e099',
    'c4_authorization': 'f47d8e29131170a41b5d2ce45b8c54820efc9140cff76e8a5f75ebdbce427378',
    'c4_acceptance_v2': '4583d5ab724fa8e2f5d886c933efc20f6fb95d8d837fea77cc55ce65f55e6566',
    'access_ledger': '71866fe84d4f8db199b54751420c686aa2fac7111e2c99f6f25de77d12308941',
    'student_freeze': 'bc519cc25603970ffa24b9a9346a493aa98b66aec460c8b9783882460c2220bb',
    'checkpoint_manifest': '673d49671bb1ef3bf2784c1367fd82f8ec356c8aed917e82594cdf7385e9964d',
    'access_receipt': 'f8dc1e3de95af76252fce109cfe17f7198b91c2042e57b40e678fe53aa340da0',
    'c4_raw_receipt': 'dc44da055aa40a7a61b1894792acb77bf28ce35e701e2d1af45b04a00215ff6c',
    'c4_localization_corrected': '8ca0962b16b1446ad6e57c50a068a767524fdc6cce7f682be70d7d05dbf08bef',
    'platt_calibrator': '19984f3236e124e50b740f29200ac21e051336d495033dd84a7a239343374833',
    'calibration_receipt': '6c04962bb4ee8c0e7b47c93556dda80d14530ecb48b440e2e22f15b4cca7f0bd',
    'unknown_terminal_audit': 'd6f61202fa9e9637ec2e086c5e061ba3d82ac2a2a33367a0e9aa225d690443c2',
    'p4_composition': 'baafe3845578625e8776be7c5a60062fe59e466f40db97f761da6fc04983d659',
    'p4_scheduler_freeze': '28b274085848e2e2c6d24e36b38f2b363675c6c76ce4b9f77d658b3d39a17b53',
    'p4_search_receipt': '91fd8988c2015757d7c2f531f4bc2dd1fa284e3982c93ff2b3c699503e520333',
    'runtime_parity_receipt': 'b34a489d8d1aab49ed56d76f5149c8c1ee33383b6b3b5058687ca818e26531b6',
    'emit_disposition': '0390af28c78a75138d27d8a35f035cfc80cd191ef1f2aa8a003460b1120c67e9',
    'policy_neighborhood': 'f468ebdb33cb15e7cfe969d0ac4200abfd7ff15ec18f12be09bcf86367508b7d',
    'runtime_contract': '3d8022a35a8d285ecbb903d50dd322e8c70d91e6f1849bd047fd0a5877482c59',
    'h2_blind_bundle': 'a6dafd5e28c4838164cffcd024fe8133cbced5453294f95217e3a8a103ba28f7',
    'h2_eval_receipt': 'f220cf5d4b6b00e418a95ebab6c0016d64f3304073935a76b539c087a9574d26',
    'h2_composition': '9a9f56da5e1e47e59facdabce5612746f1ed0d59f0c885253c12950a043d5d43',
    'h2_ledger': '3f5005d6c572cd2404d15f2d98ce78a76c90404d33ecdcfe5b27cb58e677f083',
    'encoder_source': '50635f5c65ed3929b102270755ea94bb9cc77187d7d974955a673fa3293b6d84',
    'training_script': 'd83726fce83a132a899044fed5822106c90327f459a7b31cfd8cbb148d2b452c',
    'dev2_manifest': '6c27a80a9607805486b821daf56378a55e145ed6be36b831378297a56cde18fe',
    'c4_manifest': 'b86d6345366eaccd0c4687aa768d0b2edecef106b90bf9bd9e2b56c92af8124e',
    'p4_manifest': '89c384baf0cbe8868d6b83812299d8b6b38c0f0f94e7672ea10e484d4274efc6',
    'h2_manifest': '4274e913f2fb1289a4aeffc3857cab76ff6bf2301fac114fe1e324c1efde0482',
}
CHECKPOINT_SHAS = {
    'o0_i0': '685ddadf90ad2ac4ec83bcadbe970d6ad74f07baa4e498a4936c78c0b0695f88',
    'o0_i1': 'd241ddd56648507799a852d718fc690a79d763e82379d9f6bfec9c274420026d',
    'o0_i2': '2d1a7e27d10073ad0d25fede4d9a96a521b815e27d6190c224e03c9989a66d2c',
    'o1_i0': 'beedf0f1674547bba9ae2e5d42e222224958dae085641a8fa5e8a70bd40c4d71',
    'o1_i1': 'b427a0765fd5c12b74d25991bdbc9968d07692deb0068bb61bc7c7c7730390a3',
    'o1_i2': 'a24ed43c80e7842dd7389475aa9cc51e29ca39462eaa1a9c3bd0ea693d756bc2',
    'o2_i0': '9a968d2dbdcaa5adeba7dd4f057b944e1a12502a8cc46f444598d2ce4ad8688f',
    'o2_i1': 'e5f85e0c04ee1172999272291fcc11a7ef0956df874dbe458a4aff5b3b50d616',
    'o2_i2': '1ec325b0b8c1f1ee08060c2c6bf74609b9cca48a8c14e31cf91006de97f69e31',
    'o3_i0': 'c52257f8244b12f71169c1005162400f8a7dcc747628b022cf83c1f12b24903c',
    'o3_i1': '46a34d861d6af4360476e720e42ca05c38dfc1d632382870c6604a9b5e614091',
    'o3_i2': '9f25ed4420e9e5fea32b6de5f4c1567e55c6ba32533a519cc5a181daf72ab1f7',
}
GIT_SHA = 'f9e42f6f881dd9b6e3f46a87ebfb0a2e33a676cb'

# ── 1. QUALIFIED FREEZE ──
freeze = {
    'schema': 'FINAL_DETECTOR_V23_QUALIFIED_FREEZE_V1',
    'timestamp': '2026-07-25',
    'status': 'QUALIFIED_PASS_FROZEN',
    'detector_name': 'OpenVLA Gripper Duty-Cycle Attack Detector V2.3 (Qualified)',

    'performance': {
        'p4_valid_recall': '197/220 = 89.55%',
        'p4_f3_fs': '3/30 = 10.00% (point-estimate PASS)',
        'p4_f4_fs': '3/30 = 10.00% (point-estimate PASS)',
        'p4_all_absent_fs': '6/80 = 7.50%',
        'h2_valid_recall': '159/187 = 85.03%',
        'h2_mistimed_opportunity': '0/187',
        'h2_mistimed_absent': '0/13',
        'h2_f3': 'NOT_ESTIMABLE (n=0)',
        'h2_f4': 'NOT_ESTIMABLE (n=0)',
        'h2_cross_suite_recall': {'libero_10': '85.71%', 'libero_goal': '82.50%',
                                   'libero_object': '79.59%', 'libero_spatial': '91.84%'}
    },

    'limitations': {
        'hard_negative_independent_validation': False,
        'f3_false_start_population_claim': 'NOT_ESTABLISHED',
        'f4_false_start_population_claim': 'NOT_ESTABLISHED',
        'p4_is_scheduler_selection_set': True,
        'h2_has_no_f3_f4': True,
        'c4_has_no_f3': True,
        'only_p4_has_substantial_hard_negatives': True
    },

    'frozen_components': {
        'student_modification_authorized': False,
        'calibrator_modification_authorized': False,
        'scheduler_modification_authorized': False,
        'runtime_modification_authorized': False,
        'formal_attack_authorized': False
    },

    'evidence_bindings': EVIDENCE_SHAS,
    'checkpoint_shas': CHECKPOINT_SHAS,
    'git_sha': GIT_SHA
}

with open(os.path.join(BUNDLE, 'FINAL_DETECTOR_V23_QUALIFIED_FREEZE_V1.json'), 'w') as f:
    json.dump(freeze, f, indent=2)

# ── 2. RUNTIME SPEC ──
runtime_spec = {
    'schema': 'FINAL_DETECTOR_V23_RUNTIME_SPEC_V1',

    'feature_input': {
        'total_dim': 51,
        'components': [
            {'name': 'features_25d', 'dim': 25, 'source': 'step_records.jsonl -> features_25d'},
            {'name': 'clean_policy_intent_9d', 'dim': 9},
            {'name': 'clean_gripper_token_9d', 'dim': 9,
             'fields': ['clean_close_probability_mass','clean_open_probability_mass','clean_top1_is_close',
                       'clean_top1_is_open','clean_top1_probability','clean_best_close_rank_normalized',
                       'clean_best_open_rank_normalized','clean_action_token_entropy_normalized',
                       'clean_open_minus_close_log_mass']},
            {'name': 'response_proxies_8d', 'dim': 8, 'contract': 'V23_RESPONSE_PROXY_CONTRACT.json'}
        ],
        'feature_order_sha': 'V23_RESPONSE_PROXY_CONTRACT.json'
    },

    'normalization': {
        'method': 'z-score per-dimension',
        'source': 'o0_i0 training data (297 episodes)',
        'fitted_on': 'DEV2 split fold_0 inner_folds 1+2 identities',
        'normalized_fields': ['features_25d', 'clean_policy_intent_9d', 'clean_gripper_token_9d'],
        'proxies_not_normalized': True,
        'std_clip': '1e-8'
    },

    'encoder': {
        'class': 'N4Encoder (MultiScaleEncoder)',
        'short_branch': {'receptive_field': 32, 'dilations': [1,2,4,8,16]},
        'long_branch': {'receptive_field': 128, 'dilations': [1,2,4,8,16,32,64]},
        'hidden_dim': 64, 'dropout': 0.1,
        'fusion': 'concat + Linear(128, 64)',
        'output_head': 'Linear(64, 1) -> raw logit',
        'source_sha': EVIDENCE_SHAS['encoder_source'],
        'source_file': 'src/gripper_attack/v6_critical_student.py',
        'causality': 'left-pad zeros, causal dilation stack, trim to T'
    },

    'calibrator': {
        'type': 'pooled_monotonic_Platt',
        'formula': 'calibrated_prob = sigmoid(a * raw_logit + b)',
        'a': 0.5190011735319306,
        'b': 0.812702331013635,
        'constraint': 'a > 0 (monotonic, ranking-preserving)',
        'fitted_on': 'C4 known feasible + known infeasible steps (34056 steps)',
        'sigmoid_implementation': 'numerically stable with clip(-50, 50)',
        'dtype': 'float64 for Platt application, float32 for model inference'
    },

    'scheduler': {
        'threshold': 0.855,
        'persistence': 6,
        'candidate_close': 'ON (frozen from factorized_teacher_v1.jsonl)',
        'one_shot_latch': 'ON',
        'comparison_operator': '>=',
        'semantics': 'Emit on first step t where candidate_close[t] AND calibrated_prob[t] >= 0.855 for 6 consecutive steps. After emit, latch prevents further emits.',
        'counter_reset': 'Counter resets to 0 when condition fails (candidate_close=false or score < threshold)',
        'episode_reset': 'Counter and latch reset at episode boundary',
        'cold_start': 'W128 history accumulates from step 0 with zero-padding; valid score from step 0'
    },

    'dtype_contract': {
        'model_inference': 'float32',
        'platt_application': 'float64',
        'threshold_comparison': 'float64',
        'comparison': '>=',
        'tolerance': 'raw_logit < 1e-5, calibrated_prob < 1e-7'
    }
}

with open(os.path.join(BUNDLE, 'FINAL_DETECTOR_V23_RUNTIME_SPEC_V1.json'), 'w') as f:
    json.dump(runtime_spec, f, indent=2)

# ── 3. CHECKPOINT ROUTING ──
routing = {
    'schema': 'FINAL_DETECTOR_V23_CHECKPOINT_ROUTING_V1',
    'primary_checkpoint': 'o0_i0',
    'routing_rule': 'Pre-registered fallback: identities not in the inner-CV training split manifest use checkpoint o0_i0. This was applied consistently for C4, P4, and H2 evaluation.',
    'routing_applies_to': ['C4 (200 identities)', 'P4 (300 identities)', 'H2 (200 identities)', 'Attack A-pool (all identities)'],
    'ensemble': 'NONE — single checkpoint o0_i0',
    'all_12_checkpoints_available': CHECKPOINT_SHAS,
    'checkpoint_selection_forbidden': True,
    'checkpoints_evaluated': {
        'C4': 'o0_i0', 'P4': 'o0_i0', 'H2': 'o0_i0'
    }
}

with open(os.path.join(BUNDLE, 'FINAL_DETECTOR_V23_CHECKPOINT_ROUTING_V1.json'), 'w') as f:
    json.dump(routing, f, indent=2)

# ── 4. CLAIM BOUNDARY ──
claims = {
    'schema': 'FINAL_DETECTOR_V23_CLAIM_BOUNDARY_V1',

    'claims_supported': [
        'The frozen V2.3 N4 Student + Platt calibrator + threshold/persistence scheduler achieves strong episode-level strict-K10 opportunity ranking on independent holdout data (C4 AUROC 0.984, H2 recall 85.0%)',
        'All four LIBERO suites exceed 79% valid-trigger recall on H2 one-shot evaluation',
        'The detector produces zero mistimed opportunity emissions on H2 (0/187)',
        'Offline batch and runtime streaming implementations produce identical per-step outputs (raw logit max diff 3.81e-06, 300/300 emit parity)',
        'Runtime operates from step 0 with no warm-up blindness (cold-start parity verified at T=1,2,32,64,128)',
        'Response proxies are strictly causal (verified at 80 step positions, no future leakage)',
        'The scheduler achieves 89.5% recall with point-estimate F3/F4 false-start rates of 10.00% on P4'
    ],

    'claims_qualified': [
        'F3 and F4 false-start performance: P4 is the scheduler selection set and the only role with substantial F3/F4 data (30 each). Independent H2 has zero F3/F4 episodes. The 10.00% point-estimate rates on P4 have wide binomial CIs and do not constitute independent post-selection validation.',
        'All-absent false-start rate: H2 has only 13 absent episodes (0 false starts). While the point estimate passes, the 95% CI upper bound exceeds 10%.'
    ],

    'claims_not_supported': [
        'Population F3 false-start rate <= 10%',
        'Population F4 false-start rate <= 10%',
        'Population all-absent false-start rate <= 10%',
        'The detector is a well-calibrated probability estimator (Platt improves NLL/Brier/ECE but calibration was fitted and evaluated on the same C4 data)',
        'Step-level localization: Top-1 corridor hit is 7.4% on C4; the detector identifies opportunities but peak timing is biased toward later feasible steps (median offset +22)'
    ],

    'paper_language': {
        'valid_claim': 'The frozen detector achieved 85.0% valid-trigger recall with zero mistimed opportunity emissions on the one-shot H2 evaluation.',
        'qualified_claim': 'P4 was the only Student-held-out role with substantial F3/F4 coverage, but it was used for scheduler selection and therefore does not constitute an independent post-selection false-start estimate.',
        'not_claimed': 'Independent hard-negative false-start performance was not established.'
    }
}

with open(os.path.join(BUNDLE, 'FINAL_DETECTOR_V23_CLAIM_BOUNDARY_V1.json'), 'w') as f:
    json.dump(claims, f, indent=2)

# ── 5. EVIDENCE MANIFEST ──
manifest = {
    'schema': 'FINAL_DETECTOR_V23_EVIDENCE_MANIFEST_V1',
    'git_sha': GIT_SHA,
    'evidence_root': EVIDENCE,
    'key_artifacts': {
        'recipe': {'path': 'V23_N4_RECIPE.json', 'sha256': EVIDENCE_SHAS['recipe']},
        'proxy_contract': {'path': 'V23_RESPONSE_PROXY_CONTRACT.json', 'sha256': EVIDENCE_SHAS['proxy_contract']},
        'student_freeze': {'path': 'formal_v23_student_training_v1/FORMAL_V23_STUDENT_FREEZE_V1.json', 'sha256': EVIDENCE_SHAS['student_freeze']},
        'checkpoint_manifest': {'path': 'formal_v23_student_training_v1/12_CHECKPOINT_MANIFEST.json', 'sha256': EVIDENCE_SHAS['checkpoint_manifest']},
        'platt_calibrator': {'path': 'c4_calibration_v1/C4_CALIBRATOR_V1.json', 'sha256': EVIDENCE_SHAS['platt_calibrator']},
        'p4_scheduler_freeze': {'path': 'p4_scheduler_v1/P4_SCHEDULER_FREEZE_V1.json', 'sha256': EVIDENCE_SHAS['p4_scheduler_freeze']},
        'runtime_parity': {'path': 'runtime_parity_v1/OFFLINE_RUNTIME_PARITY_RECEIPT_V1.json', 'sha256': EVIDENCE_SHAS['runtime_parity_receipt']},
        'h2_eval': {'path': 'h2_oneshot_v1/H2_ONESHOT_EVALUATION_RECEIPT_V1.json', 'sha256': EVIDENCE_SHAS['h2_eval_receipt']},
        'access_ledger': {'path': 'C4_ACCESS_LEDGER.json', 'sha256': EVIDENCE_SHAS['access_ledger']},
    },
    'all_checkpoints': {sn: {'path': 'formal_v23_student_training_v1/{}/checkpoint.pt'.format(sn), 'sha256': sha}
                        for sn, sha in CHECKPOINT_SHAS.items()},
    'manifests': {
        'dev2': {'sha256': EVIDENCE_SHAS['dev2_manifest']},
        'c4': {'sha256': EVIDENCE_SHAS['c4_manifest']},
        'p4': {'sha256': EVIDENCE_SHAS['p4_manifest']},
        'h2': {'sha256': EVIDENCE_SHAS['h2_manifest']},
    }
}

with open(os.path.join(BUNDLE, 'FINAL_DETECTOR_V23_EVIDENCE_MANIFEST_V1.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

# ── 6. Runtime parity reference (copy key info) ──
rt_ref = {
    'schema': 'RUNTIME_PARITY_REFERENCE',
    'source': 'runtime_parity_v1/OFFLINE_RUNTIME_PARITY_RECEIPT_V1.json',
    'raw_logit_max_diff': 3.81e-06,
    'cal_prob_max_diff': 1.11e-07,
    'emit_parity': '300/300',
    'cold_start_parity': 'PASS',
    'scheduler_parity': '0/300 mismatches'
}
with open(os.path.join(BUNDLE, 'RUNTIME_PARITY_REFERENCE.json'), 'w') as f:
    json.dump(rt_ref, f, indent=2)

# ── 7. H2 reference ──
import shutil
for src, dst_name in [
    ('h2_oneshot_v1/H2_ONESHOT_EVALUATION_RECEIPT_V1.json', 'H2_ONESHOT_EVALUATION_RECEIPT_V1.json'),
    ('h2_oneshot_v1/H2_BLIND_PREDICTION_MANIFEST_V1.json', 'H2_BLIND_PREDICTION_MANIFEST_V1.json'),
]:
    src_path = os.path.join(EVIDENCE, src)
    dst_path = os.path.join(BUNDLE, dst_name)
    if os.path.isfile(src_path):
        with open(src_path) as f_in, open(dst_path, 'w') as f_out:
            f_out.write(f_in.read())

# ── 8. SHA256SUMS ──
all_files = []
for root, dirs, fns in os.walk(BUNDLE):
    for fn in sorted(fns):
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, BUNDLE)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
        all_files.append((rel, sha256_file(fp)))

sums_path = os.path.join(BUNDLE, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(BUNDLE, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))

print('Bundle: {}'.format(BUNDLE))
print('Files: {}'.format(len(all_files)))
print('SHA256SUMS: {}'.format(sums_sha[:16]))
print()
print('FINAL DETECTOR V23 = QUALIFIED_PASS / FROZEN')
print('All modifications forbidden.')
print('Next: FEC pilot authorization required.')
