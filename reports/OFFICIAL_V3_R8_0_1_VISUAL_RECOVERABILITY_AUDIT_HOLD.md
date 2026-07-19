# Official V3 R8.0.1 visual recoverability audit — HOLD pending exhaustive rerun

Date: 2026-07-19  
PR: #87  
Scope: Official V3 FIT states 0–19 only

## Decision

The submitted server result is directionally consistent with the frozen Official V3 source contract:

- 800/800 FIT identity roots were found;
- 7,200 direct files were enumerated, nine per identity;
- no conventional binary/image/video container was reported;
- C2F remains a different rollout and cannot supply Official V3 RGB;
- visual materialization and visual training remain HOLD.

However, the submitted `NO_VISUAL_ASSET` artifact is not yet a formal closure because the committed census implementation did not execute the field-level claim it reported:

```python
for identity in identities[:5]:
    ...
    first = json.loads(fh.readline())
```

It therefore scanned only five identities and only the first row of each JSONL stream. It also enumerated only direct child files rather than recursively checking for nested containers. The statement “zero visual fields across 800 identities and all rows” was not established by that implementation.

```text
R8_0_1_FILE_LEVEL_CENSUS          = PROVISIONAL PASS
R8_0_1_FIELD_LEVEL_CENSUS         = HOLD — sampled 5 identities / first JSONL row
R8_0_1_FORMAL_NO_VISUAL_ASSET     = HOLD
R8_0_1B_EXHAUSTIVE_RERUN          = AUTHORIZED — READ ONLY
R8_ACTION_REPLAY_CANARY           = HOLD
R8_1_VISUAL_MATERIALIZATION       = HOLD
R8_2_VISUAL_TRAINING              = HOLD
```

## GitHub correction

The PR now contains an exhaustive V2 census implementation that:

- recursively enumerates every file under all 800 FIT identity roots;
- requires the exact nine-file artifact set for every identity;
- scans all eight semantic streams for all 800 identities;
- parses every non-empty JSONL row, not only the first row;
- records row counts and per-stream field unions;
- searches both artifact paths and semantic field paths;
- detects image, video, array, checkpoint, archive, and database carriers;
- returns `NO_VISUAL_ASSET` only after full identity, file-set, stream, and field closure;
- otherwise returns `HOLD_INCOMPLETE_CENSUS` or `VISUAL_ASSET_CANDIDATE_PRESENT`.

CPU tests specifically place a visual field in the second row of the second identity and a nested NPZ carrier, ensuring the old sampling/recursion bugs cannot recur.

## Only authorized server action

Run the corrected census once from a clean worktree at the current PR head:

```bash
python scripts/detector_v4/census_r8_official_v3_artifacts.py \
  --clean-root <OFFICIAL_V3_CLEAN_ROOT> \
  --fold-root <SEALED_FOLD_ROOT> \
  --output <NEW_NON_OVERWRITE_R8_0_1B_ROOT>
```

Required acceptance values:

```text
status                              = NO_VISUAL_ASSET
fit_identity_count                  = 800
artifact identity closure           = 800/800
exact expected file set             = 800/800
total files                         = 7,200
binary/image/video/archive carriers = 0
filename visual keyword hits        = 0
field identities scanned            = 800
missing semantic streams            = 0
field visual keyword hits           = 0
protected identity reads            = 0
simulator runs                      = 0
model inference runs                = 0
source mutation                     = 0
```

Post the full 64-character `SHA256SUMS` digest and the per-stream row counts. Preserve the prior R8.0.1 root as superseded evidence; do not overwrite it.

## Stop boundary

Do not start action replay, rendering, OpenVLA inference, embedding extraction, training, validation, exact-prefix work, or attack execution during R8.0.1b. After an exhaustive `NO_VISUAL_ASSET` root is sealed and reviewed, a separately frozen action-replay rerender canary may be considered.
