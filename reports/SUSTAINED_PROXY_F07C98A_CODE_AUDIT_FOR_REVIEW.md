# Sustained Proxy Code Audit — f07c98a (For External Review)

**Branch**: exp/sustained-proxy-burst-control-20260530  
**Commit**: f07c98a0ba4f7aa8b646331946a86eab984a330c  
**Parent**: c62214f (blob-equivalent to remote audited freeze 0870443)  
**Date**: 2026-05-30

## Summary

**1 file changed, 9 insertions, 2 deletions. All P0 invariants preserved.**

## Complete Patch

```diff
commit f07c98a0ba4f7aa8b646331946a86eab984a330c
Author: Codex <codex@openai.com>
Date:   Sat May 30 14:01:28 2026 +0800

    Add sustained command proxy burst control
    
    Add attack_burst_steps and attack_hold_mode CLI args.
    Add sustained_command_open_proxy condition.
    Decouple detector trigger gate from attack burst duration.
    Clean/oracle/random/inversion conditions unchanged.
    success predicate unchanged (done_any_LIBERO_official).

diff --git a/scripts/run_official_eval_artifact_rich.py b/scripts/run_official_eval_artifact_rich.py
index 78e9b0d..e7f1c5e 100644
--- a/scripts/run_official_eval_artifact_rich.py
+++ b/scripts/run_official_eval_artifact_rich.py
@@ -124,6 +124,7 @@ def attack_action(action, condition, rng):
     a = action.copy()
     if condition == "oracle_open": a[-1] = 1.0
     elif condition == "random_control": a[-1] = 1.0 if rng.random() > 0.5 else -1.0
+    elif condition == "sustained_command_open_proxy": a[-1] = 1.0
     elif condition in ("VIS_targeted", "gripper_inversion_proxy"):
         # NOTE: This is NOT visual PGD. It is a command-layer gripper inversion + noise proxy.
@@ -158,8 +159,12 @@ def parse_args():
     ap.add_argument("--detector_trigger_duration", type=int, default=5)
     ap.add_argument("--detector_cooldown", type=int, default=0)
     ap.add_argument("--force_detector_trigger", action="store_true", ...)
+    ap.add_argument("--attack_burst_steps", type=int, default=0,
+        help="Sustained proxy: attack hold duration in policy steps (0=use trigger_duration)")
+    ap.add_argument("--attack_hold_mode", default="fixed",
+        choices=["fixed"], help="Sustained proxy hold mode (only fixed supported)")
     ap.add_argument("--attack_condition", default="clean",
-        choices=["clean", "oracle_open", "random_control", "gripper_inversion_proxy"])
+        choices=["clean", "oracle_open", "random_control", "gripper_inversion_proxy", "sustained_command_open_proxy"])
     return ap.parse_args()

@@ -448,7 +453,7 @@ def main():
-                           attack_remaining = det_out["trigger_duration"]
+                           burst_steps = args.attack_burst_steps if hasattr(args, "attack_burst_steps") and args.attack_burst_steps > 0 else det_out["trigger_duration"]; attack_remaining = burst_steps
                         if attack_remaining > 0 and args.attack_condition != "clean":

@@ -593,6 +598,8 @@ def main():
+                "attack_burst_steps": args.attack_burst_steps if hasattr(args, "attack_burst_steps") else 0,
+                "attack_hold_mode": args.attack_hold_mode if hasattr(args, "attack_hold_mode") else "none",
```

## P0 Checklist

| # | Invariant | Code Evidence | Status |
|---|-----------|---------------|--------|
| 1 | success_official = done_any | L608 `"success_official": success`, L614 `"done_any": ep_done_any`, L616 `"success_source": "done_any_LIBERO_official"` | ✅ |
| 2 | Clean never attacks | L123 `if condition == "clean": return action`; L455/L457 `attack_condition != "clean"` | ✅ |
| 3 | Clean preserves action | attack_action returns `action` unchanged for clean | ✅ |
| 4 | Oracle unchanged | L125 `a[-1] = 1.0` unchanged | ✅ |
| 5 | Inversion unchanged | L128 `VIS_targeted/gripper_inversion_proxy` path unchanged | ✅ |
| 6 | sustained_command_open_proxy = new | L127, added to choices at L167 | ✅ |
| 7 | attack_burst_steps independent | L162, default=0 → backward compat; only >0 overrides | ✅ |
| 8 | attack_remaining init | L357 `attack_remaining = 0`; L456 uses burst_steps >0 or fallback | ✅ |
| 9 | Manifest logs new fields | L601-602: `attack_burst_steps`, `attack_hold_mode` | ✅ |
| 10 | Step records log fields | L526: `"attack_remaining": int(attack_remaining)` | ✅ |
| 11 | No VIS in proxy path | VIS_targeted only in backward-compat inversion (L128) | ✅ |
| 12 | No large files staged | `git status --short` shows 1 tracked file modified | ✅ |
| 13 | Tests pass | py_compile OK, unittest test_success_predicate_regression 6/6 OK | ✅ |
| 14 | No new attack_condition bypass | L455/L457 guard unchanged: `!= "clean"` | ✅ |

## P1 Warnings

| # | Issue | Detail |
|---|-------|--------|
| 1 | test_sustained_proxy_burst.py | **NOT FOUND** — no dedicated sustained proxy unit test was created on this branch. Existing success predicate tests pass (6/6). Burst integrity was validated experimentally via forced/natural micro (16 rollouts). Recommend writing dedicated unit tests before merge. |

## Keyword Audit

```
L127:  sustained_command_open_proxy  <-- NEW attack_action handler
L128:  VIS_targeted                  <-- legacy invert-only, UNCHANGED
L162:  attack_burst_steps            <-- NEW CLI arg (default 0)
L167:  sustained_command_open_proxy  <-- NEW in choices list
L357:  attack_remaining = 0          <-- unchanged init
L455:  != "clean"                    <-- CLEAN GUARD, unchanged
L456:  burst_steps = attack_burst_steps if >0 else trigger_duration  <-- NEW decouple logic
L457:  != "clean"                    <-- CLEAN GUARD, unchanged
L460:  attack_remaining -= 1         <-- unchanged decrement
L526:  "attack_remaining"            <-- unchanged step record
L532:  ep_done_any = True            <-- unchanged success tracking
L601:  "attack_burst_steps"          <-- NEW manifest field
L608:  "success_official"            <-- unchanged manifest field
L614:  "done_any"                    <-- unchanged manifest field
L616:  "success_source"              <-- unchanged (done_any_LIBERO_official)
```

## Test Results

```
$ python -m py_compile scripts/run_official_eval_artifact_rich.py
OK

$ python -m unittest tests/v4/test_success_predicate_regression.py -v
test_done_true_produces_success ... ok
test_info_success_absent_not_forced_failure ... ok
test_info_success_present_but_done_primary ... ok
test_multiple_predicates_agree_on_success ... ok
test_no_done_no_success ... ok
test_timeout_not_success ... ok
Ran 6 tests in 0.000s OK

$ ls tests/v4/test_sustained_proxy_burst.py
No such file or directory
```

## Conclusion

**P0: PASS. P1: WARN (missing dedicated sustained proxy unit test).**

All 14 core invariants verified. The 9 added lines (`+9/-2`) are correctly scoped, backward-compatible (default `attack_burst_steps=0` preserves old behavior), and do not touch success predicate, clean guard, oracle, or inversion paths. The missing dedicated unit test is noted and should be added before merge to main branch.
