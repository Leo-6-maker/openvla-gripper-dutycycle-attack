# Supplement: evidence, provenance, and reproducibility

Status: `PAPER_V1_SUPPLEMENT_REPRODUCIBILITY_PASS`

This supplement is a static companion to the Paper V1 draft. It does not
create new scientific evidence. It reads already-sealed handoffs, manifests,
and E3/E4 decomposition artifacts; it performs no OpenVLA inference, simulator
operation, environment step, backward pass, PGD, physical intervention, V_phys
read, Eval160 read, or protected read.

## S1. Evidence hierarchy

| layer | source | role | primary unit | allowed use | not allowed |
|---|---|---|---|---|---|
| physical mechanism | X0 | primary bounded mechanism | source-declared parent/probe | dose, phase, telemetry chain | formal mediation, universal law |
| historical mechanism context | Black Bowl | bounded contextual | raw denominator not identifiable here | fixed-window phase mechanism context | detector/generalization claim |
| held-out timing | VI-B2 | negative scientific | 16 fresh parents; abstains retained | held-out timing/generalization gate | visual attack conclusion |
| detector development | VII | negative scientific | three frozen candidates | cross-suite promotion failure | every-feature failure |
| relative selector | VIII R1 | negative scientific | 56 parents; parent-macro gate | deployment-facing selector not established | physical efficacy |
| model-side utility | IX F0 | primary negative scientific | 1,344 no-environment rows; parent-macro aggregation | model-to-physics factorization gap | physical efficacy |
| scheduler feasibility | E2 | diagnostic only | three bounded Goal successor identities | no-legal-emit diagnostic | strict-method negative |
| selective realizability | E3/E4 | primary bounded model-side | 12 engineering parents | strict structural realizability | physical efficacy/impossibility |
| history | X1/X1R-V1 | invalid/superseded | historical cohorts | governance/provenance lessons | efficacy or negative attack result |

There are no identity-level joins across these rows. The parent is the primary
unit for E3/E4; 72 E3 candidate slots are a within-parent ordered audit and are
diagnostic only.

## S2. Causal architecture and protected firewall

The paper uses the following conceptual separation:

```text
clean history / phase  ->  C_t: timing opportunity
                                  |
                                  | distinct estimand; not assumed causal
                                  v
command-OPEN duration d ->  V_t(d): physical response

clean visual input      ->  E_t: strict selective model-side realization
```

X0 measures a bounded command-OPEN counterfactual and downstream telemetry.
IX is a no-environment model-side audit. E3/E4 stop at structural candidate
materialization and direct-token auditing. None of these layers is promoted
through the firewall into a protected physical efficacy claim.

The paper-wide protected boundary is:

```text
new OpenVLA inference       = 0
new simulator / env.step    = 0
new PGD / backward           = 0
new adversarial image       = 0
physical intervention       = 0
new V_phys read             = 0
Eval160                     = UNREAD
protected evaluation       = UNREAD
R0/R1/R2                   = not entered
```

Reading a sealed aggregate such as the X0 summary or E4 decomposition is a
static paper operation and does not reopen the raw/protected boundary.

## S3. Tokenizer and action authority

The action-token boundary is part of the reproducibility contract.

1. The native generated-token endpoint is authoritative for action semantics.
2. The 31744/31745 boundary is non-bijective under the repository tokenizer
   path; a decoded action must not be silently re-encoded through a surrogate
   token and treated as the generated token.
3. Cached autoregressive behavior is distinct from an optimization surrogate.
   The surrogate may be used only inside the already-frozen authorized method;
   it does not replace the cached/native behavior authority for a paper claim.
4. The E3 strict audit checks direct tokens and native `NATIVE_OPEN` semantics.
   It does not use Student scores to select a time and does not re-encode a
   selected candidate for execution.
5. Exact arm preservation means action coordinates `[0:6]` are unchanged under
   the source-defined strict candidate audit. A candidate with arm drift is not
   a strict-valid candidate even if the gripper token reaches an OPEN class.

These details are reported as engineering provenance. They do not imply that
any candidate has physical efficacy.

## S4. X0 and bounded Black Bowl provenance

### X0

- Handoff: `docs/handoffs/STAGE_X_X0_RESULT_20260817.md`
- Handoff SHA-256: `7626e020ec524ce124be9b93685a3861b85176ad4bfd749bfc495f89482d3c55`
- Source commit/tree: `7c36489a262ee5da4826936da4520608eb30fe46` /
  `f2c9d01d72bcded05fbf887cdf2b4cfe96f18dbb`
- Sealed root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_DUTY_CYCLE_MECHANISM/STAGE_X0_RESULT_20260817T095900Z`
- Summary SHA-256: `ff2e18c905a108cb51dbecf82473d1cd4e301e02a86c5e87bf39aab723fd35af`
- `SHA256SUMS` SHA-256: `fb8da5b1f9ce30bef7884563ec9579c1c9c7d3c2e3fb749efc89c56be3d6fbd1`
- Population: 40 Stage V plus 16 Stage VI-B2 parents; 1,344 probe groups;
  1,126 complete dose rows.
- Consumable rows: T3/T5/T10 = 1,245/1,191/1,126.
- Complete three-dose patterns: 1,126; all monotone `000/001/011/111`.
- Task-failure taxonomy: `NOT_AVAILABLE`; no reconstruction from `V_phys`.

### Black Bowl

- Historical handoff context:
  `docs/handoffs/DETECTOR_V3_D8_HANDOFF_20260731.md`
- Handoff SHA-256:
  `74201a30f86d6d5e5a5750429dc0e7b781e32ba6f67b0d54158c76dc259ffa0b`
- Configuration:
  `configs/paper_black_bowl_attack.yaml`
- Configuration SHA-256:
  `8e1c05da22d00818949a0e36c5708ee374708cba7bd12c061c6f9cde6c187f3a`
- The current checkout does not identify a raw sealed outcome root and
  denominator for this historical context. It is therefore not used as a
  primary quantitative population.

## S5. Timing-selector negative provenance

### VI-B2

- Handoff SHA-256:
  `2c550b1ec8f906212bc1f92223b801bcb0efe07381caf94d1b116e4f5e66d006`
- Source commit/tree: `800b19341e39e71ce75991dc9a13f796bdf5ffdf` /
  `71e5c28a99752960b092ab2a21d732069ed438ab`
- Population: 16 fresh parents, 384 probes, 1,536 planned branches.
- T5: 333 consumable and 51 abstain/censored.
- Overall AUROC/AUPRC/ECE-10: 0.6246432939 / 0.7976720489 /
  0.4606357016.
- Decision: `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`.
- Timing matrix: not authorized after the causal gate failed.

Abstains and censored rows remain separate. They are not converted to negative
labels and are not silently removed from the provenance accounting.

### VII

- Handoff SHA-256:
  `9ee6782530f730e7bf5eaee90ced94134ec6a569d5c9713b7b19f85662a1c33f`
- Source commit/tree: `8ad9859a61a0083948c4e7b73eed72d7bf1d2aad` /
  `d4f6da984041924db3f35eb2be812cd9e8c444fb`
- Candidates: S7-A, S7-B, S7-C.
- All three fail at least one frozen cross-suite generalization/selectivity
  gate; none is promoted.
- Historical initial handoffs are preserved as superseded snapshots and are
  not substituted for the final decision.

### VIII

The direct R1 handoff was absent from the current checkout but was recovered
from immutable Git history before paper assembly:

- R1 handoff path at commit
  `34b8d435264737734f7d4e4ecc9a3343e57d7c1`;
- R1 handoff Git blob SHA-1:
  `0e01f0115eada6dcdf183524438f458c0db03fa3`;
- documentation closure at commit
  `b918104e2e6279364891590fd37a17720dbb6628`;
- closure Git blob SHA-1:
  `75341759abd9cb7859466927661dba67efadee39`;
- runtime source commit/tree:
  `c03cb32b14d38978239e53adc953cc8620b775a1` /
  `6f5cff89855fbab3bbc06b92c170586db9e0091d`;
- R1 sealed root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VIII_RELATIVE_TIMING_SELECTOR/STAGE_VIII_R1_RELATIVE_SELECTOR_20260816T161050Z`;
- R1 summary SHA-256:
  `3ace17e589529c420e59deceeada7d8dbf84efb775d1402999b53702e5f51b06`;
- R1 `ROOT_SEAL.json` SHA-256:
  `cbcc55b350e59cc222e76f944a19f1f430b69e6f64363531823637530593f7a1`.

The R1-A/R1-B parent-macro AUC values are 0.577615/0.658572 and both fail
the frozen gate. This is direct immutable historical evidence, not an inferred
Stage-IX-only status.

## S6. Stage IX factorization-gap provenance

- Handoff: `docs/handoffs/STAGE_IX_F0_RESULT_20260817.md`
- Handoff SHA-256:
  `2c445ea1ae5434661d327a348995c412083b191b382cbd78d485a13064268103`
- Source commit/tree: `98838b91a34c134c24e430c6b660dfcb33ba6137` /
  `daea7d0eabb89027218869c6b5c1b4035b187f31`
- Runner SHA-256:
  `d335a9d41bf4e28572e6f2146f73fc0cb5ed4c1aa526cab9379ce41d9f630862`
- Population: 1,344/1,344 no-environment rows; eight 168-row worker shards.
- Model AUROC E0/E1/E3: 0.870743/0.900510/0.897157.
- Factorized parent-macro AUC E0/E1/E3: 0.483698/0.521112/0.523390.
- Protected counters: zero; Eval160/protected remain unread.

The factorized parent-macro quantity is the relevant aggregation for the
paper. It is not replaced by the stronger-looking row-level model score.

## S7. E2, E3, and E4 provenance

### E2

- Handoff SHA-256:
  `ca3dc00eee3e9873dfcbbf1084330fc5492b5d8669e8e817cc9b222696a8b6c7`
- Source commit/tree: `71d6a7e7c7ac2202a206d872f86ad1f79fea70b9` /
  `76776602598747b8fa90801ceb1400a896937806`
- Root seal SHA-256: `1ea143dbaf866839d0dd3d0f8f304c91b4b701100042e0dd701c712fd5b3c003`.
- Three bounded Goal successor identities had no legal clean Student emit.
- No TRUE PGD probe, attacked environment step, physical intervention, or
  V_phys read occurred.

E2 is a timing/scheduler feasibility hold. It is not a strict visual-method
negative and cannot explain the E3 result.

### E3

- Decision file:
  `reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_DECISION_TABLE_V1.json`
- Decision SHA-256:
  `85a3fcadcda31d3ca31b7a17da049bdbed3823d670947f88ab7182090c617b24`
- E3 root seal SHA-256:
  `20f5be64be1884aded25385e08c69ba3f5982422d546a532a737fd89e1bec16b`
- E3 denominator: 12 engineering-only parents × six ordered candidates = 72
  candidate slots.
- Completeness: 12/12 clean runtimes valid, 12/12 probes available, 12/12
  TRUE invocations reached, 12/12 six-candidate audits complete.
- Strict-valid parents: 2/12; suite split L10 1/3, Goal 0/3, Object 0/3,
  Spatial 1/3.
- No attacked `env.step`, physical intervention, V_phys, Eval160, protected,
  or attack-outcome read.

### E4

- Runtime/source commit/tree: `d642ab5a57d23f44933e1907991c845e2f30e294` /
  `1533bcec345c1b9ee06c46281291700c47527489`.
- Candidate decomposition:
  `reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json`
- Decomposition SHA-256:
  `c4ac212f6998cd0c12b4c427aa2314a0f5a75cfb4cafca118bb78edfd43cd412`
- Synthesis table SHA-256:
  `cda5397fb64d1efcdde91e0e5ca9d2c60165d62ab52a110aa561b17a5603b28d`
- Final E4 claim ledger SHA-256:
  `c67f0035c392a355fe9f51b0769dea069609925cfe0cefa5ac9d368ea1191c25`
- E4 root seal SHA-256:
  `70825f121a87fd048b2d117a08e629bd11397dbd0225dc0fbc329e80c1501e09`
- Parent categories: TARGETABILITY_LIMITED 9, JOINT_LIMITED 1,
  STRICT_REALIZABLE 2, SELECTIVITY_LIMITED 0.
- Candidate diagnostics: 4 exact-arm/native-OPEN, 29 exact-arm/not-native-
  OPEN, 1 native-OPEN/arm-drift, 38 neither.
- `attack_efficacy=false`; mandatory stop after offline decomposition.

## S8. Victim provenance and historical invalidity

Historical X1 used one canonical PGD victim for all rows while the Stage V and
VI-B2 clean/snapshot manifests declared different policy checkpoints for three
suites. The pre-experiment audit records this as a blocking provenance
mismatch, in addition to sequence enumeration and metric naming defects.

X1R-V1 is permanently closed as an incomplete prospective attempt:

- frozen ITT denominator: N=7;
- scientific evaluable population: N=0;
- two runtime-invalid consumed identities;
- five untouched identities;
- no accepted adversarial result, attacked `env.step`, task outcome, or
  `V_phys` read;
- no efficacy estimate;
- no rerun, replacement, top-up, or re-ranking.

These histories are retained to show why provenance and runtime contracts are
part of the scientific evidence boundary. They are not negative attack
experiments.

## S9. No-rerun and no-replacement governance

The paper lock preserves the following rules:

1. A sealed artifact is immutable; corrections are append-only documentation or
   publication metadata, never silent replacement.
2. A missing denominator is not inferred from a nearby stage.
3. An abstain, censoring state, or invalid runtime is not converted to a
   negative scientific label.
4. Candidate slots are not promoted to iid observations.
5. E2 is not upgraded to an attack-method negative.
6. E3/E4 are not upgraded to physical efficacy or impossibility.
7. A manuscript gap is recorded as a limitation; it is not filled by a new
   fixture, threshold, epsilon, step count, objective, or protected read.
8. Any future experimental continuation requires a new explicit scientific
   authority and a new protocol namespace.

## S10. Reproduction commands

The following commands are CPU/static only:

```text
python scripts/paper/audit_paper_v1_authority_map.py
python scripts/paper/build_paper_v1_figures_tables.py
```

The P4 audit command is added by the final paper bundle and checks the draft,
captions, authority map, stage locks, protected boundary, denominators, and
claim ledger. No command in this supplement launches a worker or claims a GPU.

## S11. Final boundary statement

The sealed material supports a bounded mechanism-first/factorization-gap paper.
It does not authorize more attack execution. The final paper package must stop
at `PAPER_V1_MECHANISM_FACTORIZATION_DRAFT_BUNDLE_READY_FOR_PI` and return to
Owner/PI review.
