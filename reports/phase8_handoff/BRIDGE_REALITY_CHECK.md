# Bridge Reality Check — Phase 8 Cross-Suite Generalization

**Generated**: 2026-06-27 16:15 CST
**Auditor**: Claude (experiment lead)
**Server**: pm-364c0001 (via vla-jump → 10.60.2.56:33571)

---

## 1. Git Divergence

| Item | Handoff (`ef7c0bb`) | Server Reality | Match? |
|------|---------------------|----------------|:------:|
| Git HEAD | `ef7c0bb` | `ace18762` | **NO** |
| Branch | `experiments/cross-suite-generalization-v1` | `experiments/cross-suite-generalization-v1` | YES |
| Handoff commit on server | — | **NOT IN HISTORY** | **NO** |
| Working tree | clean | **DIRTY** (4 modified + hundreds untracked) | **NO** |

Server git log (last 10): all `m1c` commits, no `phase8` or `cross-suite` commits visible.
Handoff commit `ef7c0bb` was created locally but **never pushed** to origin.

## 2. Bridge File Integrity

### Primary bridge: `scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py`

| Check | Result |
|-------|--------|
| On-disk SHA256 | `c1af3159c83cb32b2fd0a721f9aed93680c79389d2cee2718e17b6541c5c08d6` |
| In Git HEAD? | **NO** — `fatal: path exists on disk, but not in 'HEAD'` |
| Git tracked? | **NO** — `??` (untracked) |
| Git diff | (none — file is untracked) |
| Matches any local version? | UNKNOWN (local bridge not yet compared) |

### Secondary bridge: `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py`

| Check | Result |
|-------|--------|
| In Git? | YES (tracked, modified) |
| Object-site lookup | **STILL HARDCODED** (lines 138-139) — same Object-specific pattern |
| Attack args patch | Partial — `--attack_objective`, `--arm_lock`, `--libero_preprocess_backend` added |

## 3. Hardcoded Object-Suite Fields in `_telemetry_v2.py`

| Line | Content | Suite-Aware? |
|------|---------|:------------:|
| 33 | `--unnorm_key default="libero_object"` | Partially (overridable via CLI) |
| 34 | `--suite_name default="libero_object"` | Partially (overridable via CLI) |
| 100 | `model.get_action_dim(args.unnorm_key)` | YES (uses arg) |
| 155-156 | `from v4_run_eval_openvla import decode_with_scores...` | N/A (imports) |
| 161 | `suite = bm[args.suite_name]()` | YES (uses arg) |
| **166** | **`build_v4_exact_env(bddl, args.render_gpu, 400, 10)`** | **NO — 400 hardcoded** |
| **171** | **`_obj_key = _task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")`** | **NO — Object-specific parsing** |
| **172** | **`obj_sid = env.sim.model.site_name2id(f"{_obj_key}_1_default_site")`** | **NO — Object site naming** |
| **185** | **`for step in range(400):`** | **NO — 400 hardcoded** |
| 212 | `decode_with_scores(..., args.unnorm_key, ...)` | YES (uses arg) |
| 321 | `attacker.attack(..., unnorm_key=args.unnorm_key)` | YES (uses arg) |
| 339 | `model.get_action_stats(args.unnorm_key)` | YES (uses arg) |

**6 hardcoded fields remain**, of which 3 are CRITICAL blocking (lines 166, 171-172).

## 4. Confirmed Error Signature

GOAL task 0 (`put_the_bowl_on_the_stove`):

```
ValueError: No "site" with name put_the_bowl_on_the_stove_1_default_site exists.
Available "site" names = (..., 'akita_black_bowl_1_default_site', ...)
```

Root cause: `_task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")` is a no-op for GOAL/Spatial task names, returning the full task name as the object key.

## 5. Active Processes (at audit time)

| PID | PPID | GPU | Task | Condition | Status |
|-----|------|-----|------|-----------|:------:|
| 3969276 | 3926004 | 1 | GOAL t2 s2 | PREFIX_NOLOCK | **KILLED** |
| 3969651 | 3926006 | 2 | GOAL t2 s2 | CLEAN | **KILLED** |

Parent dispatchers `bash -s 1 W01` (3926004) and `bash -s 2 W03` (3926006) — both **KILLED**.

All bridge processes confirmed dead. No dispatchers running.

## 6. Output Directory State

| Metric | Count |
|--------|:-----:|
| Total dirs | **91** (up from 67 at handoff) |
| `.done` | **0** |
| `episode_summary.json` | **0** |
| `step_telemetry.csv` | **0** |
| Recent timestamps | 15:58–16:09 (all GOAL, all FAILED_TECHNICAL) |

New 24 dirs (91-67) were created by the now-killed dispatchers. All are GOAL task 0-2.

## 7. GPU State (post-kill)

| GPU | Memory Used | Memory Total | Utilization |
|-----|:----------:|:------------:|:-----------:|
| 0 | 52351 MiB | 81920 MiB | 34% (other users) |
| 1 | 23932 MiB | 81920 MiB | 22% (freed) |
| 2 | 23932 MiB | 81920 MiB | 36% (freed) |
| 3 | 9040 MiB | 81920 MiB | 37% (idle/low) |
| 4 | 46949 MiB | 81920 MiB | 72% (other users) |
| 5 | 42111 MiB | 81920 MiB | 82% (other users) |
| 6 | 9041 MiB | 81920 MiB | 1% (idle) |
| 7 | 39048 MiB | 81920 MiB | 5% (other users) |

GPUs 1, 2, 3, 6 available for Phase 8.

## 8. Model Status

| Suite | Path | Safetensors | Tokenizer | dataset_stats | Status |
|-------|------|:-----------:|:---------:|:-------------:|:------:|
| Spatial | `models/libero-spatial/spatial_c8f03f4_20260620/` | 4/4 | YES | YES | **READY** |
| Goal | `models/libero-goal/` | 4/4 | YES | YES | **READY** |
| LIBERO-10 | `models/libero-10/openvla-7b-finetuned-libero-10/` | **1/4** | **NO** | YES | **INCOMPLETE** |

LIBERO-10: 1.7GB total. Missing: model-00002/03/04-of-00004.safetensors, tokenizer.json, tokenizer_config.json, preprocessor_config.json, special_tokens_map.json, model.safetensors.index.json.

LIBERO-10 `dataset_statistics.json` key: **`"libero_10"`** (confirmed), 7 action dims, mask[6]=false.

## 9. Other Modified Files (non-blocking for bridge fix)

- `v4_run_eval_openvla.py`: Already has `resolve_unnorm_key()` — suite-aware
- `attack_adapter.py`: Already has `_resolve_unnorm_key()` — suite-aware
- `run_sc5_cross_suite_clean.py`: Not inspected (not on critical path)

## 10. Answers to Required Questions

1. **Server bridge matches Git HEAD?** NO — file is untracked, not in any commit
2. **`--unnorm_key` and `--suite_name` exist?** YES (lines 33-34), passed via CLI
3. **Benchmark suite fixed to `libero_object`?** DEFAULT is, but `args.suite_name` overrides at line 161
4. **Decode/action stats use `libero_object`?** NO — use `args.unnorm_key` (parameterized)
5. **Max steps fixed to 400?** YES — lines 166, 185 both hardcoded
6. **Object-site lookup Object-specific?** YES — lines 171-172 hardcoded
7. **Previous 6 bridge processes?** All gone. Replaced by new dispatcher pair (now killed)

## 11. Verdict

**BRIDGE BLOCKING. 91/91 jobs FAILED_TECHNICAL. Same root cause as handoff. No forward progress possible without patch.**

The `--unnorm_key`/`--suite_name` args were correctly forward-patched to the bridge, but the object-site resolver (lines 171-172) and max_steps (lines 166, 185) remain Object-hardcoded.

LIBERO-10 model transfer is incomplete (1/4 shards). Spatial and Goal models are ready.
