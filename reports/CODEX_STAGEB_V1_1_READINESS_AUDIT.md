# Codex Stage-B v1.1 Readiness Audit

**Date**: 2026-06-07
**Audited commit**: `1e4f7c5dd9fbe7f43e3eb6c948f1c11c8f478fd1`
**Mode**: CPU-only static audit
**Verdict**: `BLOCKED_FOR_STAGEB_V1_1_BATCH`

## Scope Correction

This audit is explicitly based on the committed tree at `1e4f7c5`, not on local
dirty files. Any local uncommitted v1.1 runner or validator drafts are excluded
from readiness judgment until committed and pushed.

## Summary

Commit `1e4f7c5` successfully freezes the OpenVLA-LIBERO gripper executable
semantics and fixes the `attack_adapter` OPEN token region. That part passes.

Stage-B v1.1 as a runnable/labeling pipeline is not ready. The committed root
runner still does not satisfy the v1.1 contract: it does not import the
executable spec directly, still uses the chat-style prompt, uses raw
`obs['agentview_image']` without an official/legacy marker, lacks the full trace
schema, lacks stable pair identity, and does not record random Linf provenance.

The postprocess and label builder are also still v1-era and can consume
unverified or old intermediates.

## Checklist Verdicts

| Check | Verdict | Notes |
|---|---|---|
| Runner imports executable spec | BLOCKER | `scripts/run_stageb_vis_labeling.py` imports `gripper_semantics`, not `openvla_libero_exec_spec`. |
| Runner still has chat prompt | BLOCKER | Root runner still uses chat-style `USER:/ASSISTANT:` prompt. |
| Image preprocessing marked | BLOCKER | Root runner directly uses `obs['agentview_image']` without official preprocessing or legacy marker. |
| Trace schema complete | BLOCKER | Missing raw action vector, pair/task/window metadata, decoded_open_bool, open_convention, trace/spec/git fields. |
| Pair matching usable | BLOCKER | Runner has no `--pair_id` and no pair_id in trace/summary at committed state. |
| Attack adapter token region | PASS | OPEN tokens are decoded raw `>=0.5` and env `-1`; saturation tokens are classified by physical sign. |
| Postprocess accepts only corrected traces | BLOCKER | It does not require `corrected_stageb_v1_1` and parses metadata from filenames. |
| Label builder avoids old labels | BLOCKER | It does not read old labels directly, but lacks source/trace_version/provenance guards. |
| random_linf metadata | BLOCKER | No seed/norm/perturbation_space output in committed runner. |
| Tests cover readiness | PARTIAL/BLOCKER | Token/open tests exist; full schema and old-label quarantine tests are missing. |

## Blocking Issues

1. **Committed root runner is not v1.1 spec-aligned.**
   It still uses `gripper_semantics`, chat prompt, and raw agentview image. It
   does not record trace/spec/git/image preprocessing fields required for v1.1.

2. **Matched VIS/random pair identity is absent.**
   The committed runner has no `--pair_id` and no `pair_id` field in trace or
   summary. Worker scripts launch VIS and random as separate processes, so
   pair-level matching cannot be verified by `pair_id + task/state/window`.

3. **The legacy paired runner remains unsafe for v1.1.**
   `scripts/stageb/run_paired_vis_random_v1.py` still uses chat prompt, raw
   `obs['agentview_image']`, local open semantics, `patched_stageb_v1`, and
   `GIT_COMMIT='stageb_v1'`.

4. **Postprocess is still v1-era.**
   `scripts/stageb/postprocess_patched_traces_v1.py`:
   - does not require `trace_version == corrected_stageb_v1_1`;
   - parses condition/task from filename;
   - pairs by task/window only;
   - computes shifted qpos using local enumerate index instead of `step+1`.

5. **Label builder lacks provenance hard gates.**
   `scripts/stageb/build_pair_labels_v1.py` consumes the intermediate CSV
   without checking trace_version, pair_id, state_id, corrected source, or
   quarantine status.

6. **Old-label quarantine is not test-covered.**
   No Stage-B test currently proves `patched_stageb_v1`, old overnight labels,
   or 44-row pre-spec outputs are rejected.

## Required Before Stage-B v1.1 Launch

1. Patch and commit the root runner to import `openvla_libero_exec_spec`, use
   official `In:/Out:` prompt, official image preprocessing marker, complete
   trace schema, stable `pair_id`, and random Linf provenance.
2. Patch worker generators to pass the same `pair_id` to VIS and matched random
   jobs and record paired job ids.
3. Rewrite/patch postprocess to:
   - accept only `corrected_stageb_v1_1`;
   - read condition/task/state/window/pair_id from trace or summary fields;
   - pair by `pair_id + task_key + state_id + window_start + window_end`;
   - compute shifted qpos with `step_dict[step+1]`;
   - hard-fail any old/unpatched trace.
4. Harden label builder to reject old/quarantined inputs and require corrected
   provenance.
5. Extend tests for full trace schema, corrected trace_version, pair_id sharing,
   random metadata, and old-label quarantine.

## Audit Artifacts

- Findings table: `tables/codex_stageb_v1_1_readiness_findings.csv`

## Boundary

No GPU batch, VIS rerun, rollout worker, watcher, or server output mutation was
performed.
