# Official V3 V5 raw-asset recovery handoff

Date: 2026-07-18  
Local code baseline before this audit: `69c83244700f8849a2e2953873c6bb86ec2f98a0`  
Audit schema: `CLEAN2000_ARTIFACT_RECOVERY_AUDIT_V1`

## Scope

This was a read-only metadata and path inventory. It did not run model
inference, replay a rollout, materialize a Teacher, train a Detector, read
FIT-DEV/CAL/CHECK semantics, or run an attack. No CLEAN artifact, S1 root, or
existing evidence root was modified. The only new server evidence is the
non-overwrite audit root below.

Server audit root:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_CLEAN2000_ARTIFACT_RECOVERY_AUDIT_V1_69c8324_20260718_03
SHA256SUMS = 325827de58fb637ba8da18b96e2fb563ad074f56c55db833c63ac48cd69c5da3
```

The source path-size inventory digests are not content tree hashes. Existing
Official V3 artifact/S1 seals remain the authority for source immutability.

## Source roots inspected

| root | identity coverage | relevant result | disposition |
|---|---:|---|---|
| Official V3 `.../c2g/.../clean` | 2000/2000 | 800 FIT identities have direct policy-intent/top-logit telemetry and privileged sidecars; RGB 0/800 | Official V3-bound source candidate |
| C2F `clean2000_obs_clean_36712cc` | 2000/2000 | 393,513 PNG files, 27,448,562,700 bytes; FIT 800/800 have `rgb_path` alignment PASS | Parallel source, not bound to Official V3 |
| C2F `siglip_full_final` | no identity mapping | 19 derived files, including NPZ/PT outputs | Derived output, not raw episode source |

The Official V3 FIT metadata schema was `OPENVLA_OFFICIAL_CLEAN_EPISODE_V2`
for 800/800 inspected identities. The raw C2F FIT schema was
`C2F_OBS_LANG_CLEAN_EPISODE_V1` for 800/800 and used collector/source commit
`36712cc67ae707e815670e1652d95c9f230e8a07`.

## Official V3 FIT result

The existing Official V3 artifact source already contains the policy-intent
stream needed for a policy-intent ablation. The audit observed, among other
fields:

- `clean_action_token_top_logits`, token IDs and score-head summaries;
- `clean_open_probability_mass`, `clean_close_probability_mass` and
  open-minus-close log mass;
- `clean_policy_intent_9d`, entropy and top-1 action fields;
- `generation_passes_per_step`, `single_generation_parity_pass` and
  `score_adapter_parity_pass`;
- model/checkpoint/processor path and SHA pointers, task language/prompt, and
  worker/provenance pointers;
- `privileged_teacher_sidecar` as a separate source stream.

Therefore the previous PR statement that policy-intent was “not supplied / not
found” is stale. The corrected conclusion is:

```text
OFFICIAL_V3_FIT_POLICY_INTENT_SOURCE = PRESENT 800/800
OFFICIAL_V3_FIT_DIRECT_LOGIT_TELEMETRY = PRESENT 800/800
OFFICIAL_V3_FIT_PRIVILEGED_SIDECAR = PRESENT 800/800
OFFICIAL_V3_FIT_RGB = MISSING 800/800
```

This only establishes source availability. It does not authorize V5-B
training; the policy-intent loader still needs a separate sealed input binding
to the Official V3 S1/registry and the V5 protocol.

## Parallel RGB result and boundary

The C2F raw root contains per-step PNG frames and `rgb_path` pointers. The FIT
sampled content audit found the C2F schema and teacher-like fields such as
`teacher_hazard`, `teacher_phase`, and `teacher_primary_attackable`. It does
not contain the Official V3 artifact provenance contract, and its collector
commit differs from the Official V3 execution line.

Thus this root is useful for source recovery and compatibility investigation,
but is not a V5-C input. No RGB bytes were copied, transformed, replayed, or
joined into the Official V3 corpus.

```text
C2F_RGB_DISCOVERED                 = 2000/2000 identities
C2F_FIT_RGB_PATH_ALIGNMENT         = 800/800 PASS (metadata/path check)
C2F_SOURCE_CAMPAIGN_BINDING        = HOLD
V5_C_CAUSAL_VISUAL_INPUT            = HOLD
```

## Gate and next action

```text
RAW_ASSET_RECOVERY_AUDIT            = PASS_METADATA_ONLY
OFFICIAL_V3_POLICY_INTENT_SOURCE    = PRESENT / BINDING REQUIRED
OFFICIAL_V3_RGB_SOURCE              = MISSING
PARALLEL_C2F_RGB_SOURCE              = DISCOVERED / UNBOUND
V5_A_PROPRIO                         = EXISTING R2 SMOKE HOLD
V5_B_POLICY_INTENT                   = CONDITIONAL AFTER SEALED BINDING
V5_C_CAUSAL_VISUAL                   = HOLD
V5_D_MULTIMODAL                      = HOLD
GPU_TRAINING_THIS_TURN               = NOT STARTED
FIT_DEV / CAL / CHECK                = NOT READ
ATTACK                               = NOT STARTED
```

The shortest safe next step is a small FIT-only V5-B policy-intent binding and
CPU audit, not RGB reuse. V5-C requires an explicit campaign/source mapping
decision for the C2F root; it must not be enabled merely because PNG files
exist.
