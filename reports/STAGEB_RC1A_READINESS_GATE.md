# Stage-B RC1a Readiness Gate

**Date**: 2026-06-07
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Commit audited**: `5e2863e2b91f470f18f393cfa62167a8fdcc17f3`
**Mode**: CPU-only source/readiness gate. No GPU, VIS, rollout, watcher, or clean reachability scan was started.

## Gate Verdict

**PASS for clean reachability scan precondition.**

The RC1a code path has no blocking direct OPEN/CLOSE comparisons in current Stage-B v1.1 execution, postprocess, or label-building paths. Legacy and historical hits remain quarantined and must not be used as current Stage-B evidence.

## Direct Open Comparison Grep

Output table:

```text
tables/stageb_rc1a_direct_open_comparison_grep.csv
```

Classification counts:

| Classification | Count |
|---|---:|
| `ALLOWED_IN_SPEC` | 21 |
| `ALLOWED_IN_TEST` | 6 |
| `ALLOWED_LEGACY_QUARANTINE` | 67 |
| `BLOCKER` | 0 |

Gate result:

```text
BLOCKER = 0
```

Search covered Stage-B-relevant direct comparison patterns:

```text
>0.5
<-0.5
raw < 0.5
raw > 0.5
env_grip > 0
env_grip < 0
```

The grep table includes only open/gripper/contextual hits. Generic metric thresholds such as AUROC or probability cutoffs are excluded from the gate denominator unless the line also references gripper/open/close/raw/decoded/env action semantics.

## Attack Adapter Token Region Audit

`src/gripper_attack/attack_adapter.py` now uses official RC1a semantics in `get_gripper_region_by_decoded_action()`:

- OPEN tokens: decoded raw action `> 0.5` and transformed env gripper is physical OPEN (`env < -0.5`).
- CLOSE tokens: decoded raw action `< 0.5` and transformed env gripper is physical CLOSE (`env > +0.5`).
- Boundary tokens: decoded raw action `== 0.5` are tracked as boundary/neutral and excluded from OPEN/CLOSE sets.
- Runtime assertions hard-fail if OPEN/CLOSE token sets overlap, are empty, or classify boundary tokens as executable OPEN/CLOSE.

Regression coverage:

- `tests/stageb/test_attack_open_token_region.py`
- `tests/stageb/test_openvla_libero_exec_spec.py`
- `tests/stageb/test_openvla_full_alignment.py`

## Runner / Postprocess / Label Builder Gate

Current Stage-B v1.1 main path:

| Component | Gate | Evidence |
|---|---|---|
| Runner trace version | PASS | `TRACE_VERSION = corrected_stageb_v1_1` |
| Runner source snapshot | PASS | `SOURCE_SNAPSHOT_ID = f9840cb1` |
| `decoded_open_bool` | PASS | computed by `env_gripper_is_open(env_action_6)` from executable spec |
| qpos source | PASS | trace records `qpos_source = obs_robot0_gripper_qpos` and qpos vector fields |
| summary open counters | PASS | use `env_gripper_is_open(...)` helper |
| postprocess version gate | PASS | accepts only `corrected_stageb_v1_1`; old-format traces hard-fail |
| postprocess qpos | PASS | recomputes shifted qpos from `obs_gripper_qpos_0/1` using `step_dict[s + 1]` |
| label builder version gate | PASS | rejects any non-`corrected_stageb_v1_1` qpos row |
| old labels quarantine | PASS | pre-v1.1 traces/labels remain excluded from current label generation |

## Validation

Local CPU validation on this commit:

```text
PYTHONPATH=src python -m pytest tests/stageb -q
47 passed
```

Prior RC1a isolated server validation copy:

```text
/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
env: /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
pytest tests/stageb -q: 47 passed
SHA table: rows=40, missing=0, py310_fail=0
```

## Remaining Boundary

- The live server reviewed worktree was previously dirty/manual-upload state. Do not launch from it unless it is explicitly synced to `5e2863e` / RC1a `source_snapshot_id = f9840cb1`, or DeepSeek elects to run from the isolated validated copy.
- Legacy detector/diagnostic scripts and old reports still contain direct comparisons or obsolete comments. They are marked `ALLOWED_LEGACY_QUARANTINE` in the grep table and must not feed current Stage-B evidence without rewrite/regeneration.
- This gate authorizes only the next CPU/runner-readiness step: clean reachability scan. It does not validate VIS effectiveness.
