# L3 Campaign Status

**Last updated:** 2026-06-17T04:35:34Z

**Current gate:** H0
**Codex claimed:** H0_IN_PROGRESS
**DeepSeek independent:** H0_INDEPENDENT_AWAITING
**Next authorized:** 

**Codex branch:** exp/l3-vis-handoff-contract-repair-20260617
**Codex head:** 50da442c1b033a780b802c6345c376b23d4833b1
**Watcher:** NO_ACTIVE_WATCHER
**Output root:** /data/liuyu/outputs/l3_vis_codex_results

## Gate Progression

| Gate | Codex | DeepSeek | Time |
|------|-------|----------|------|
| H0 | H0_IN_PROGRESS | H0_INDEPENDENT_AWAITING | 2026-06-17T04:35:34Z |

## Allowed Transitions

- **H0** → H1
- **H1** → H2
- **H2** → H3, H4, H5
- **H3** → H4
- **H4** → H5
- **H5** → H6
- **H6** → TERMINAL

## Stop Conditions

- Scientific FAIL at any gate
- Dirty worktree or unexpected commit drift
- Config drift (lambda, target, epsilon, seeds)
- SHA mismatch or missing artifacts
- Route fallback detected
- GPU mapping change
- Xid/OOM during active phase
- Denominator substitution
