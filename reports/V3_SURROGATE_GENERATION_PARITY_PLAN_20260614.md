# V3 Surrogate-Generation Parity Diagnostic Plan

Date: 2026-06-14

## Current status

This branch finalizes the replay and four-path diagnostic infrastructure needed
before any GPU parity job is authorized. It does not run replay generation,
does not launch rollouts, and does not implement v3.1.

Frozen scientific state:

- v2 remains `EXECSPEC_V2_FALSIFIED`.
- v3 core engineering smoke passed, but v3 scientific effect is not
  established.
- butter seed811 exposed a no-cache surrogate top-token mismatch versus
  official generation.
- cream seed811 aligned on a boundary token, exposing competition-set
  incompleteness.

## No-overclaim boundaries

- Replay/parity engineering validity is not attack success.
- A surrogate-to-generation mismatch is a method diagnostic, not automatically
  infrastructure corruption.
- An official generation score mismatch is infrastructure invalid.
- v3 remains scientifically unestablished.
- v3.1 remains unimplemented.

## Shared helper consolidation

`src/gripper_attack/v3_generation_parity.py` is now the single source of truth
for:

- exact new-token extraction and length validation;
- surrogate-prefix versus full-AR prefix invariant validation;
- official generation processed-score invariant validation;
- official token decode and execution classification;
- compact score-row summary with top1/top2 token IDs and scores;
- diagnostic token scores for 31744 and 31872 when in vocabulary;
- strongest native OPEN, CLOSE, and boundary token/score;
- bundled surrogate-top score in official score rows;
- safe replay stem construction with sanitized job IDs and a digest;
- replay schema validation with optional on-disk tensor checks;
- stable per-path diagnostic record schema;
- final four-path diagnosis classification.

Runner, standalone diagnostic, and CPU tests call these production helpers
instead of carrying copied logic.

## Replay bundle schema

Replay bundles use schema:

`v3_generation_parity_replay_v2`

Required contents include:

- task, state, attack seed, job ID, step, condition;
- objective and objective tag;
- model path and dtype;
- runner, adapter, semantics, and exec-spec SHA-256 values;
- exact prompt input IDs, shape, and dtype;
- adversarial tensor filename, SHA-256, shape, and dtype;
- surrogate generated six-token prefix;
- official generated seven-token output;
- complete compact `official_generation_score_audit`;
- surrogate and official token-execution dictionaries;
- generation config: `do_sample`, `max_new_tokens`, default/effective
  `use_cache`, EOS ID, pad ID;
- actual clean AR seven-token output and clean six-token arm prefix;
- retokenized clean arm prefix;
- adversarial generated arm prefix;
- adversarial-versus-clean-AR arm match rate;
- transfer classification.

The runner write order is:

1. choose safe filenames;
2. save tensor;
3. compute tensor SHA-256;
4. populate the full bundle;
5. validate complete values and tensor cross-field consistency;
6. write JSON;
7. compute JSON SHA-256;
8. attach JSON/tensor filename and SHA back to the episode record.

Validation rejects missing or empty fields, invalid schema versions, non-hex or
wrong-length hashes, prompt shape mismatch, wrong token lengths, missing tensor
files, tensor SHA mismatch, tensor shape/dtype mismatch, malformed execution
dicts, official score argmax mismatch, and Path A official-output mismatch
during standalone replay.

## Clean AR provenance

`decode_with_scores` now attaches observation-only prompt provenance to the
returned generation object:

- `prompt_input_ids`;
- `prompt_len`.

The runner extracts exactly seven clean generated action tokens using that
prompt length and stores:

- `clean_generated_action_token_ids`;
- `clean_generated_arm_prefix_token_ids`;
- `retokenized_clean_action_arm_prefix`;
- `adv_generated_arm_prefix`;
- `adv_vs_clean_generated_arm_match_rate`.

Retokenized continuous action is no longer treated as the actual clean AR
prefix.

## Score audit completeness

The compact official score audit retains:

- final emitted token;
- processed-score argmax token;
- top-1 token and score;
- top-2 token and score;
- top1-minus-top2 gap;
- scores for tokens 31744 and 31872 when present in the vocabulary;
- best native OPEN/CLOSE/boundary token IDs and scores;
- bundled surrogate-top score in the official score row.

This full dict is written into both the episode record and replay JSON.

## Four-path diagnostic

`scripts/stageb/diagnose_v3_generation_parity.py` consumes one or more
validated replay bundles and compares the same prompt and adversarial tensor
across:

| Path | Method | Required behavior |
| --- | --- | --- |
| A | `generate`, default cache behavior, `output_scores=True` | exactly 7 new tokens; final processed-score argmax equals emitted gripper token; all 7 tokens reproduce bundle official output |
| B | full-sequence forward, `use_cache=False` | raw-logit row summary under surrogate generated prefix |
| C | full-sequence forward, `use_cache=True` | raw-logit row summary under surrogate generated prefix |
| D | `generate(use_cache=False)`, `output_scores=True` | exactly 7 new tokens; final processed-score argmax equals emitted gripper token |

Every path emits the same stable schema. Quantities that do not exist for a
path are explicit `null` with an `unavailable_reason`; raw logits and
generation processed scores are separate fields.

Final diagnosis categories include:

- `CACHE_PATH_MISMATCH_CANDIDATE`;
- `GENERATION_SCORE_PROCESSING_MISMATCH_CANDIDATE`;
- `NEAR_TIE_NUMERICAL_SENSITIVITY_CANDIDATE`;
- `COMPETITION_SET_INCOMPLETENESS_CONFIRMED`;
- `LARGE_UNEXPLAINED_PATH_DIFFERENCE`.

The diagnosis is emitted as a structured object with `class` and `evidence`
fields, including A/B/C/D tokens, score gaps, best native OPEN/CLOSE/boundary
scores, OPEN-minus-CLOSE, boundary-minus-OPEN, and the frozen native-OPEN
margin. A/B path disagreement is classified as cache/path, generation-score
processing, near-tie, or unexplained path difference before any competition-set
claim is considered, with near ties taking precedence over non-cache path
attribution. Competition-set incompleteness is reserved for the path-agreeing
boundary case where A/B/C/D agree when available, native OPEN beats native CLOSE
by at least the frozen margin under hinge semantics, and boundary remains global
top. The script fails loudly on malformed input instead of silently skipping
missing files.

## CPU test coverage

`tests/stageb/test_v3_generation_parity.py` covers:

- exact 7-new-token extraction pass/fail;
- prefix invariant pass/fail;
- official score invariant pass/missing/mismatch;
- native edge, boundary, clipped-low, clipped-high, and mask-false official
  decoding;
- score-row summary fields;
- safe replay stem sanitization and digest;
- replay validation failures for bad hashes, bad token lengths, missing tensor,
  SHA mismatch, shape mismatch, dtype mismatch, prompt mismatch, bad execution
  dict, and bad schema;
- validated replay pass using a temporary tensor file;
- Path A/D exact-token and score-invariant checks using deterministic mocks;
- Path A reproduction mismatch;
- stable four-path output schema;
- deterministic cream-like path-agreeing boundary-over-OPEN diagnosis;
- deterministic butter-like A/B mismatch diagnosis that prioritizes cache/path
  disagreement over competition-set claims;
- near-tie precedence for `B=C!=A` with a tiny processed-score gap;
- margin-equality competition-set confirmation;
- D-path disagreement blocking competition-set confirmation;
- clean AR prefix extraction and match-rate calculation.

`tests/stageb/test_layer3_autoregressive_prefix_v3.py` remains green and keeps
the v3 gradient-equivalence, prefix-length, and no-action-overwrite coverage.

## Authorized next step

Open the draft PR and stop. The next step is independent GPT review of the
pushed code and PR. GPU parity diagnostics, replay generation, B1/B3 screens,
rollouts, and v3.1 work remain unauthorized.
