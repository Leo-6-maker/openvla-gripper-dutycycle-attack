# Codex Stage-B v1.1 Readiness Audit

**Date**: 2026-06-07
**Base commit**: `1e4f7c5dd9fbe7f43e3eb6c948f1c11c8f478fd1`
**Mode**: CPU-only static audit
**Verdict**: `BLOCKED_FOR_STAGEB_V1_1_BATCH`

## Summary

The root runner `scripts/run_stageb_vis_labeling.py` is largely aligned with the
OpenVLA-LIBERO executable spec from commit `1e4f7c5`: it imports
`openvla_libero_exec_spec`, uses `official_prompt`, defaults to rotated LIBERO
image preprocessing, records raw/env actions, qpos, decoded OPEN, spec version,
git metadata, and random Linf metadata.

However Stage-B v1.1 is not ready to launch or label. The blockers are in
pair identity, postprocess, label-builder provenance, and tests.

## Checklist Verdicts

| Check | Verdict | Notes |
|---|---|---|
| Runner imports executable spec | PASS | Root runner imports `openvla_libero_exec_spec`. Legacy `scripts/stageb/run_paired_vis_random_v1.py` does not. |
| Runner still has chat prompt | PASS for root runner | Root runner uses official `In:/Out:`. Legacy paired runner still uses chat prompt and must not be used. |
| Image preprocessing marked | PARTIAL | Root runner defaults to `official_rot180` and marks legacy mode, but official resize parity is not separately audited. |
| Trace schema complete | PASS for single trace | Root runner records raw action, pair_id, task/window, decoded_open_bool, open_convention, git_commit. |
| Pair matching usable | BLOCKER | Root runner generates a fresh UUID pair_id per process; VIS and random will not share pair_id. |
| Attack adapter token region | PASS | OPEN tokens are decoded raw `>=0.5` and env `-1`; saturation tokens are classified by physical sign. |
| Postprocess accepts only corrected traces | BLOCKER | `postprocess_patched_traces_v1.py` does not require `corrected_stageb_v1_1`. |
| Label builder avoids old labels | BLOCKER | It does not read old labels directly, but lacks source/trace_version/provenance guards. |
| random_linf metadata | PARTIAL | Metadata columns exist, but torch RNG is not explicitly seeded for reproducible random noise. |
| Tests cover readiness | PARTIAL/BLOCKER | Token/open tests pass, but old-label quarantine and full v1.1 schema tests are missing. |

## Blocking Issues

1. **Matched VIS/random pair_id is not stable.**
   `scripts/run_stageb_vis_labeling.py` creates `pair_id = uuid.uuid4()[:8]`
   inside each process. Worker scripts launch VIS and random as separate
   processes, so matched conditions will not share pair identity. This blocks
   pair matching by `pair_id + task/state/window`.

2. **The legacy paired runner remains unsafe for v1.1.**
   `scripts/stageb/run_paired_vis_random_v1.py` still uses chat prompt, raw
   `obs['agentview_image']`, local open semantics, `patched_stageb_v1`, and
   `GIT_COMMIT='stageb_v1'`. It must be quarantined or hard-failed.

3. **Postprocess is still v1-era.**
   `scripts/stageb/postprocess_patched_traces_v1.py`:
   - does not require `trace_version == corrected_stageb_v1_1`;
   - parses condition/task from filename;
   - pairs by task/window only;
   - computes shifted qpos using local enumerate index instead of `step+1`.

4. **Label builder lacks provenance hard gates.**
   `scripts/stageb/build_pair_labels_v1.py` consumes the intermediate CSV
   without checking trace_version, pair_id, state_id, corrected source, or
   quarantine status. It does not directly consume old labels, but it cannot
   prove the input is not old-label-derived.

5. **Old-label quarantine is not test-covered.**
   No Stage-B test currently proves `patched_stageb_v1`, old overnight labels,
   or 44-row pre-spec outputs are rejected.

## Non-Blocking Warnings

- The root runner's image preprocessing is marked and defaults to the official
  180-degree rotation, but this audit did not verify official resize/backend
  parity.
- `random_linf` records `random_seed`, `noise_linf`, `noise_l2`, and
  `perturbation_space`, but `torch.rand_like` is not explicitly tied to the
  recorded seed.
- `scripts/stageb/validate_stageb_outputs_v1.py` is outdated and should be
  upgraded to the full v1.1 trace contract.

## Required Before Stage-B v1.1 Launch

1. Add `--pair_id` to `scripts/run_stageb_vis_labeling.py`, or derive a stable
   deterministic pair id from task/state/window/job pairing.
2. Update worker generators to pass the same `pair_id` to VIS and matched
   random jobs, and record paired job ids.
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
