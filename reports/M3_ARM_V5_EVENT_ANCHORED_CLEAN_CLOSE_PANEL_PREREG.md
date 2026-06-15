# M3 arm-v5 event-anchored clean-CLOSE panel preregistration

## Decision

```text
PREREG_ONLY
```

V5 replaces the failed absolute-frame denominator with clean-CLOSE event
anchors. This commit freezes the protocol only. It does not authorize GPU clean
capture, PGD, RAND, shuffled-gradient, seed85, seed86, LIBERO rollout, or any
Layer2-triggered attack.

## Base Finding

PR #24 closed the Tomato state0 absolute-frame panel because capture-only
preflight found only `2/8` main frames with clean CLOSE token `31872`; the other
`6/8` were already target token `31744`.

## State Pool

V5 freezes 20 task-state candidates:

```text
10 LIBERO Object tasks x 2 states per task
```

For each task, candidate states are `0..49` except known Layer3 development
states:

```text
bbq_sauce: 0
butter: 2
chocolate_pudding: 2
cream_cheese: 2
tomato_sauce: 0
```

The exclusion source of truth is:

```text
tables/m3_arm_v5_prior_layer3_state_ledger.csv
```

Every row with `used_for_development=1` must be excluded from the state pool.
The helper tests assert that the frozen 20-state pool contains no prior Layer3
development state; this is not a narrative-only exclusion.
The config pool and CSV pool must exactly equal the ledger-derived hash
selection:

```text
ledger
→ used_for_development exclusions
→ enumerate 10 tasks x states 0..49
→ recompute SHA256("M3_ARM_V5_CLOSE_PANEL|task|state")
→ choose first 2 states per task
→ assert equality with YAML and CSV
```

Each task takes the two smallest hashes:

```text
SHA256("M3_ARM_V5_CLOSE_PANEL|task|state")
```

The frozen table is:

```text
tables/m3_arm_v5_preregistered_state_pool.csv
```

No attack result, margin, detector output, teacher probability, task success,
or event count was used to choose the states.

## Clean-Only Event Selector

For each frozen task-state, V5.1 may run one clean trajectory and search for
the earliest clean CLOSE onset satisfying:

```text
min_step <= step <= max_step
records are strictly increasing and unique
current step has adjacent previous row: previous_step == current_step - 1
exact official action tokens == 7
score invariant == PASS
explicit gripper token == tokens[-1]
official_score_argmax_token_id is present and equals emitted token
current gripper token == 31872
previous step gripper token != 31872
```

Frozen limits:

```text
min_step = 0
max_step = 279
```

Each state contributes at most one event: the earliest qualifying event.
Invalid clean artifacts are not skipped in search of a later event. Duplicate
steps, step gaps, invalid previous rows, token-field mismatch, score invariant
failure, or official argmax/emitted mismatch all produce:

```text
V5_CLEAN_EVENT_INFRA_INVALID
```

That state remains invalid for V5.1 and cannot be replaced by a non-frozen
state.

The only accepted argmax evidence field is:

```text
official_score_argmax_token_id
```

Missing argmax evidence is:

```text
V5_CLEAN_EVENT_INFRA_INVALID
reason = missing_official_argmax_evidence
```

The full clean trajectory must be validated before event selection is accepted;
corruption after an apparent first event is still infra-invalid.

Forbidden during event selection:

- choosing the largest or easiest margin;
- choosing based on arm prefix convenience;
- choosing based on attack outcome;
- choosing multiple adjacent frames as independent samples;
- replacing an event after seeing attack results.

## Final Eight Inputs

After clean capture only, sort the 20 frozen task-state candidates by their
precomputed state hash and take the first 8 states that have a qualifying clean
CLOSE event.

If fewer than 8 states have a qualifying event:

```text
V5_CAPTURE_POOL_INSUFFICIENT
STOP
```

Once the eight exact inputs are frozen, no replacement is allowed.

The capture-only runner for this phase is:

```text
scripts/stageb/run_m3_arm_v5_clean_capture.py
```

It is independent from the PGD/RAND/shuffled runners and does not import attack
adapter or control modules. The runner must read the frozen 20-state manifest,
produce at most one clean trajectory per state, preserve invalid/missing states
in its all-state table, and write a hash manifest for capture/preflight
artifacts.

For each selected event, the frozen panel CSV must have non-empty bindings for:

```text
raw_image_path + raw_image_sha256
processed_tensor_path + processed_tensor_sha256
prompt_token_ids + prompt_token_ids_sha256
current and previous official_score_argmax_token_id
model_fingerprint
model_checkpoint_sha256
processor_config_sha256
preprocess_config_sha256
task_state_init_sha256
clean_record_source_path + clean_record_source_sha256
runner_sha256
config_sha256
commit
gpu_query
worktree_status
```

If any selected field is blank, V5.1 must stop with:

```text
V5_EXACT_INPUT_BINDING_INCOMPLETE
```

V5.0C further requires exact capture provenance, not just non-empty fields:

```text
prepared inputs are constructed once before generation
attention_mask is dropped before generation
prompt suffix 29871 is appended before generation when needed
processor_inputs.pt stores the exact input_ids and pixel_values sent to generate
gen.prompt_input_ids must equal the saved input_ids
gen.prompt_len must equal the saved input length
exact new action tokens are sliced with gen.prompt_len
processor uses the official V4 setting use_fast=false
```

Offline selection must be attempt-ledger driven:

```text
state -> final legal attempt -> CAPTURED -> clean_records_path -> SHA256
```

It must not guess filenames or select stale JSON files outside the final
captured attempt.

Model binding uses a deterministic bundle manifest over config, generation
config, tokenizer/preprocessor files, remote-code Python files, and all model
weight shards. `model_checkpoint_sha256` is the canonical manifest SHA, not a
single `config.json` hash.

The exact-input auditor recomputes raw/tensor/source file SHA256 values,
verifies prompt-token SHA, rejects path escape, checks current/previous event
artifacts, requires `worktree_status=CLEAN`, requires a valid `nvidia-smi`
snapshot, and rejects duplicate selected raw/tensor artifacts.

Capture attempt policy is fail-closed:

```text
default: one clean trajectory attempt per frozen state
only retry allowed: FIRST_ACTION_BEFORE_INFRA_FAILURE with first_action_taken=false
maximum attempts per state: 2
all failed attempts preserved in the attempt ledger
```

The capture runner uses phase markers:

```text
ATTEMPT_STARTED
MODEL_READY
ENV_READY
FIRST_ACTION_GENERATED
FIRST_ACTION_TAKEN
CAPTURE_COMPLETED
```

Any crash after `FIRST_ACTION_TAKEN` is a post-action failure and cannot be
automatically retried.

Runtime gates are fail-closed when supplied:

```text
expected commit
expected branch
expected config SHA
expected ledger SHA
expected state-pool CSV SHA
expected CUDA_VISIBLE_DEVICES
expected GPU UUIDs
no existing compute process on target GPU UUIDs
valid nvidia-smi GPU snapshot
clean worktree
new output directory
```

V5.0D makes these runtime gates mandatory for `capture_clean_pool`; omitting any
expected field is `V5_RUNTIME_PROVENANCE_INCOMPLETE`. Branch detection uses
`git rev-parse --abbrev-ref HEAD` for old server Git compatibility and rejects
detached `HEAD`.

Offline event selection must verify the capture-root
`m3_arm_v5_model_bundle_manifest.csv` against the actual model path, recompute
the canonical bundle SHA, and require every selected event row to bind that same
bundle SHA.

An independent auditor entrypoint is required before any V5.1 capture evidence
can be accepted:

```text
scripts/stageb/audit_m3_arm_v5_clean_capture.py
```

The auditor reads existing capture artifacts, validates the 20-state frozen
pool, attempt ledger, phase markers, model bundle, exact selected-input
bindings, and writes `m3_arm_v5_clean_capture_audit.json`. It is a post-producer
artifact audit; it does not run capture or attack code.

The capture retry state machine is fixed at:

```text
max attempts per state = 2
attempt 1 allowed only if attempt 0 fails before FIRST_ACTION_GENERATED
post-generation or post-action failures are terminal
each attempt writes to a distinct state artifact directory
```

## Phase Separation

```text
V5.1:
  clean capture only
  freeze exact 8 inputs
  no PGD/RAND/shuffled

review

V5.2:
  exact 8 frozen inputs
  first attack seed = 428198
  TRUE/RAND/shuffled
  no LIBERO rollout

review

V5.3:
  second seed only after explicit authorization
```

## Attack Method Frozen For Future V5.2

V5.2, if later authorized, must keep arm-v4 unchanged:

| Field | Value |
| --- | --- |
| Target token | `31744` |
| Target class | `CLIP_MEDIATED_OPEN` |
| Epsilon | `6/255` |
| PGD steps | `20` |
| Candidates | `21` per condition |
| Arm gate | actual clean generated arm-prefix match `>=5/6` |
| Controls | `RAND21_SELECTIVE`, `SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE` |
| Selection | official-decode hard-feasible maximum target margin |

No objective, epsilon, candidate budget, arm gate, frame count, or aggregate
threshold tuning is allowed in V5.0.

The first V5.2 attack seed is frozen by:

```text
SHA256("M3_ARM_V5_CLOSE_PANEL|attack_seed|v5.2|seed1")
= f3bfb8e6ca8b7903349099f6803696697ac382dc65611a437388d0592343793e
seed = 428198
```

Seeds `85` and `86` are reserved for arm-v4 and must not be reused.

## Aggregate Gate Frozen For Future V5.2

For the eight frozen clean-CLOSE event inputs:

```text
infra invalid = 0
FRAME_FULL_SELECTIVE_PASS >= 6/8
RAND finite paired frames >= 4
shuffled finite paired frames >= 4
median TRUE-RAND official target margin > 0
median TRUE-shuffled official target margin > 0
```

## Allowed Claim If V5.0 Is Accepted

V5.0 defines an event-anchored clean-CLOSE panel protocol that avoids the
absolute-frame denominator failure found in PR #24.

## Forbidden Claims

Do not claim:

- V5 has eligible events before V5.1 clean capture;
- arm-v4 generalizes across states;
- TRUE_PGD beats random;
- closed-loop Layer3 is established;
- detector-selected Layer3 is established.

## Stop Rule

After this preregistration, stop for review. No V5 capture is authorized by
this commit.
