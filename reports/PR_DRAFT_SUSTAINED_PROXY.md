# PR: Add sustained command proxy burst control

**Branch**: `exp/sustained-proxy-burst-control-20260530` → `main`
**Commits**: e7e5bd1, 07e13a0

## Summary

Add `--attack_burst_steps` and `sustained_command_open_proxy` condition. Decouples detector trigger gate from attack burst duration. Clean/oracle/random/inversion conditions unchanged.

## Changed Files

- `scripts/run_official_eval_artifact_rich.py` (+11/-2)
- `tests/v4/test_sustained_proxy_burst.py` (new, 10 tests)

## Tests

```
py_compile: OK
unittest test_success_predicate_regression: 6/6 OK
unittest test_sustained_proxy_burst: 10/10 OK
bash -n scripts/*.sh: OK
```

## Key Experimental Validation

| Gate | Result |
|------|--------|
| Forced micro (4 rollouts) | qpos delta=-0.0308 matches oracle |
| Natural micro (12 rollouts) | robust control survives |
| Validation pilot (48 rollouts) | sus30 selectivity +100% |
| Full10 sus30 (50 rollouts) | High 0/10, Robust 10/10 |

## Caveats

- NOT visual attack (VIS). Command-layer sustained proxy.
- NOT universal attack. Object-suite only.
- Detector is NOT oracle-optimal. Selects candidate windows.
- Effect is task/object-dependent.

## Valid Claims

> `sustained_command_open_proxy_30` selectively causes oracle-level failures on high oracle-sensitive Object tasks while preserving robust controls.
