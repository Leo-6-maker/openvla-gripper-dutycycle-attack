# Official V3 R7.2.2 Closure Replay Audit — PASS

Date: 2026-07-19  
PR: #87, Draft  
Reviewed execution commit: `33f5dc3488546ea9b9bbddb7152998c3fb560566`  
Reviewed handoff commit: `ca2db9f1fea078fb2f769044aa341182870710f4`

## Verdict

R7.2.2 closes the load-bearing R7.2/R7.2.1 failures and is accepted for the limited scientific claim below.

```text
R7_R2_2_OFFICIAL_LOADER             = PASS
R7_R2_2_CHECKPOINT_FORWARD          = PASS
R7_R2_2_POLICY_INTENT_FAIL_CLOSED   = PASS
R7_R2_2_SCHEDULER_REPLAY            = PASS
R7_R2_2_THRESHOLD_DENOMINATORS      = PASS
R7_R2_2_PAIRED_REPRESENTATION       = PASS
R7_R2_2_SOURCE_AND_SEAL_BINDING     = PASS
R7_R2_2_ARTIFACT_AUDIT              = PASS — INTERNAL CONSISTENCY
R7_R2_2_CORE_TRANSFER_CONCLUSION    = PASS
```

Accepted claim:

> On the sealed Fold-0 FIT validation population, the frozen Physics V5-A/V5-B checkpoints transfer poorly to the frozen R7 K10 opportunity-start target, both under the frozen V5 one-shot scheduler and in paired raw-score localization diagnostics.

This is not a claim that proprioception can never solve the target, that the K10 Teacher is attack-vulnerability ground truth, or that the result transfers to FIT-DEV/CAL/CHECK/CS200.

## Load-bearing evidence

Population:

```text
Fold-0 validation identities = 200
K10-feasible episodes        = 26
No-corridor episodes         = 174
Candidate-close parity       = 200/200
Step-count parity            = 200/200
```

Scheduler transfer:

```text
V5-A: 3/26 = 0.1153846 at tau=0.1; 9 total emits; precision 3/9
V5-B: 0/26 at every diagnostic threshold; 2 emits at tau=0.1
```

Paired representation transfer on the same 26 positive episodes:

```text
                    V5-A       V5-B
mean max-in - max-out  -0.1149    -0.0886
delta > 0              2/26       3/26
best rankable step in   2/26       3/26
mean best-feasible rank 30.7       36.2
```

The replay uses the repository `load_v5_episodes`, `CausalMultimodalVulnerabilityRanker`, strict state loading, real sealed policy intent for V5-B, checkpoint normalizations, and `V5OneShotScheduler`. The K10 root contributes only the target `is_feasible_start` plus parity checks.

Server roots:

```text
Replay:
OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_33f5dc3_20260719
SHA256SUMS = 13e8338ed6681dc23fd4f991070ba2caf0dcd1280b6314efe3e740b743f15dab

Audit:
OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_AUDIT_33f5dc3_20260719
SHA256SUMS = 8d038783f398c701a525d2f40af6cf93725d7f69b727ad9af3c475a22a1235c5
Status = PASS
```

## Non-blocking caveats

1. The artifact auditor independently verifies sealing, schemas, population closure and ledger/metric consistency. It does not independently recompute neural-network outputs through a second model implementation.
2. `evaluator_file_blob_sha256` contains the 40-hex Git blob object ID returned by `git hash-object`; it is a valid source identifier but is named imprecisely and is not a SHA-256 digest.
3. The reported even-sample median uses the upper middle order statistic rather than the average of the two central values. Mean delta, sign count and best-step counts are unaffected.
4. Auxiliary fields called `release_or_post_emit`, `outside_rankable_steps` and `k10_containment_rate` should not be reused as formal safety endpoints without clearer definitions. They are not load-bearing for the transfer conclusion or R7.3 authorization.
5. The handoff phrase “negligible signal” should be read as “weak signal on this frozen Fold-0 evaluation,” not as a universal impossibility result.

## Downstream decision

The old Physics-tier checkpoints should be frozen as negative-transfer evidence. Further threshold or scheduler tuning on them is not authorized.

R7.3 is authorized only under `protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md`:

- FIT Fold-0 only;
- two candidates only: linear 25D and causal GRU 25D;
- one fixed seed;
- train-only normalization and train-only OOF threshold selection;
- validation used once at the selected threshold;
- no attacks, simulator, protected split, visual model, or policy-intent candidate.

```text
R7_R3_K10_SPECIFIC_FOLD0_TRAINING = AUTHORIZED
R7_R4_EXACT_PREFIX                = HOLD
R7_R5_ATTACK_CANARY               = HOLD
FIT_DEV / CAL / CHECK             = NOT READ
CS200                              = NOT READ
```
