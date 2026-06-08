# Stage-B RC1a Handoff Generation Prompt — 2026-06-08

Use this prompt for DeepSeek to compress the current work window and generate a clean handoff for a new GPT session. Do **not** continue the main experiment while producing this handoff. The handoff is a state-freeze / context-transfer artifact, not a new experiment plan to execute immediately.

---

## Copy/paste prompt for DeepSeek

```text
DeepSeek, pause the mainline experiment and generate a complete but compressed HANDOFF for a new GPT session.

Current branch:
  exp/vis-prefix-margin-repair-20260603

Latest known commit:
  a3aecd7

Do not launch new GPU jobs.
Do not advance Bronze expansion.
Do not start visual-module integration.
Do not train new models beyond summarizing already completed detector readouts.
Do not modify labels, rerun postprocess, or change code while generating the handoff unless explicitly asked.

Your task is to create one compact but complete handoff report and commit it to GitHub.

Expected output file:
  reports/STAGEB_RC1A_SESSION_HANDOFF_20260608.md

Optional supporting table if useful:
  tables/stageb_rc1a_session_handoff_artifact_index.csv

Commit message:
  docs(stageb): add rc1a session handoff

After commit:
  push to origin exp/vis-prefix-margin-repair-20260603
  report final remote tip SHA

============================================================
0. Handoff goal
============================================================

The new GPT session should be able to continue the project without reading the full previous chat.

The handoff must preserve:
1. What problem we are solving.
2. Which old results are quarantined.
3. Which RC1a code/data chain is trusted.
4. Which experimental results are valid.
5. What the detector currently means and does not mean.
6. What the next safe actions are.
7. What must not be done.
8. Where all important artifacts live.

Keep it concise enough to fit in a new chat context, but detailed enough to prevent repeating old mistakes.

============================================================
1. Project one-liner
============================================================

Write this clearly:

We are studying an inference-time VIS PGD attack on OpenVLA-7B in LIBERO Object tasks. The attack targets gripper duty-cycle behavior: identify an online-safe window from clean rollout features, apply a low-budget visual perturbation, induce gripper OPEN commands, and ideally create a physical gripper/qpos response causing task failure. The research highlight should be window selection + low-budget targeted VIS, not brute-force attack.

============================================================
2. Critical semantic correction
============================================================

State explicitly:

OpenVLA-LIBERO gripper execution semantics are frozen as RC1a:
  raw_gripper > 0.5  -> env_action_6 = -1.0 -> physical OPEN
  raw_gripper < 0.5  -> env_action_6 = +1.0 -> physical CLOSE
  raw_gripper == 0.5 -> boundary / neutral, excluded from OPEN/CLOSE regions

Official action chain:
  raw action -> normalize_gripper_action(..., binarize=True) -> invert_gripper_action -> env.step

Trusted spec file:
  src/gripper_attack/openvla_libero_exec_spec.py

VIS token region:
  OPEN tokens decode to raw > 0.5 and env_action_6 < -0.5
  CLOSE tokens decode to raw < 0.5 and env_action_6 > +0.5

Trusted attack file:
  src/gripper_attack/attack_adapter.py

Important: all old labels/results that used env_action_6 > 0.5 as OPEN are invalid/quarantined.

============================================================
3. Quarantined / deprecated results
============================================================

List these as NOT usable for training or claims:

1. Old 44-row patched rerun before RC1a semantic freeze.
   Reason: VIS objective/open convention was inverted or unverified.

2. Old overnight Stage-B labels before corrected_stageb_v1_1.
   Reason: open convention/qpos provenance issues.

3. Any pre-v1.1 trace.
   Reason: missing full env_action/raw_action/qpos/source metadata.

4. Active Probe v0b/v1 no-env results.
   Reason: no-env surrogate does not reliably predict rollout VIS effects. Useful as negative diagnostic only.

5. ProprioNoStep as vulnerable-window detector.
   Reason: it detects contact/opportunity/hazard phases, not pre-grasp VIS-vulnerable windows. Useful only as post-hoc/phase tool.

6. Any detector readout before seed-aware candidate matching and cmd_specific/random_sensitive split.
   Label as DIAGNOSTIC_ONLY_PRE_FIX.

============================================================
4. Trusted RC1a code chain
============================================================

List core trusted files and roles:

- src/gripper_attack/openvla_libero_exec_spec.py
  RC1a official-like executable gripper semantics.

- src/gripper_attack/attack_adapter.py
  Corrected VIS objective token region; OPEN token region excludes boundary.

- scripts/run_stageb_vis_labeling.py
  v1.1 runner, 53-col trace, records raw/env actions, qpos, pair_id, source metadata.

- scripts/stageb/validate_stageb_trace_v1_1.py
  v1.1 validator. Must enforce trace_version, source_snapshot_id, qpos_source, open_convention, decoded_open_bool consistency.

- scripts/stageb/postprocess_traces_v1_1.py
  Trace-level qpos/open recounting; should use trace-level shifted qpos, not old summary qpos.

- scripts/stageb/build_pair_labels_v1_1.py
  v1.1 pair label builder. Pair key includes seed:
    pair_id, task_key, state_id, seed, window_start, window_end.

- scripts/diagnostics/run_detector_v0_fixed.py
  Unified detector readout after fixes. Supports bronze/silver_override/rescue_override and target split.

Mention latest known detector fix:
  a3aecd7 — rescue granularity aggregation fixed; Rescue rows=18, parents=12, 6 parents had 2 repeats; now aggregated with stability logic instead of last-row overwrite.

============================================================
5. Trusted data stages and results
============================================================

Summarize in a table.

A. RC1a Clean Reachability Scan
- 27/27 clean rollouts completed.
- 0 infra fail.
- All trace_version = corrected_stageb_v1_1.
- All source_snapshot_id = f9840cb1.
- All prompt_style = official_in_out.
- All image_preprocess_style = official_rot180_only.
- Candidate generation created 1198 reachable window candidates using actual trace length, not fixed [1,295].

B. Smoke3-B
- 3 windows, 6 jobs, validator PASS.
- Found command-level corrected VIS effect:
  cream_cheese s2 [45,55]: VIS open=8 vs random=0.
- Physical qpos bridge was weak in smoke.

C. Pilot12
- 12 paired windows, 24/24 validator PASS.
- cmd_susceptible = 4/12.
- random_confounded = 0/12.
- vis_specific_physical = 2/12.
- Key positives: butter s0 [70,80], [75,85] produced command + physical response.

D. Bronze Batch
- 96/96 traces validator PASS.
- 45 valid paired windows; 3 unfound/excluded.
- cmd_susceptible = 11/45 = 24.4%.
- random_confounded = 8/45 = 17.8%.
- physical_response_sensitive = 15/45 = 33.3%.
- vis_specific_physical = 7/45 = 15.6%.
- Interpretation: corrected VIS objective produces stable nontrivial command/physical signal in RC1a reachable candidate pool.

E. Silver P1A
- 84/84 jobs.
- 84/84 validator PASS.
- 37 Silver pairs from 23 parent Bronze windows.
- stable_cmd_positive = 9.
- stable_phys_positive = 4.
- stable_rand = 6.
- hard_neg_confirmed = 2.
- unstable = 6.
- positive_stability = 0.64.
- random_confounded_stability = 0.75.
- Interpretation: Bronze positives are not all noise; random-sensitive/general perturbation-sensitive windows are real and stable.

F. P1b
- 36/36 jobs validator PASS.
- 18 pairs.
- cmd_sus = 2/18.
- phys = 3/18.
- rand_conf = 4/18.
- vis_spec = 2/18.
- Interpretation: hard/weak/underrepresented queue adds useful negatives and some positives.

G. Random-confounded Rescue
- Rescue rows = 18.
- Unique parents = 12.
- Multi-repeat parents = 6, each with 2 repeats.
- Latest fix aggregates repeats with stability logic, same as Silver, no last-row overwrite.
- Aggregated rescue_override:
    cmd_specific positives = 11
    vis_specific_physical positives = 3
    random_sensitive positives = 3
- Interpretation: cmd_specific remains task-biased; phys/rand are underpowered in rescue tier.

============================================================
6. Current detector status
============================================================

State this bluntly:

The detector is not yet a strong final online vulnerable-window detector.
It is currently an exploratory multi-head selector.

Targets:
- cmd_any: VIS can induce OPEN, regardless of random confound.
- cmd_specific: cmd_any AND NOT random_sensitive. This is the main VIS-specific command target.
- vis_specific_physical: VIS-specific qpos/physical response.
- random_sensitive: random/general perturbation-sensitive confound; should be used as abstain/avoid head.

Current readout after fixes:

Bronze / Silver / Rescue detector summary:
- cmd_specific:
    Bronze best CleanNoTaskNoTiming P@5=0.40, enrich=1.6x, AUROC=0.46.
    Silver best CleanNoTaskNoTiming P@5=0.40, enrich=0.9x, AUROC=0.34.
    Rescue best TaskOnly P@5=0.60, enrich=2.5x, AUROC=0.68.
    Interpretation: cmd_specific is dominated by task bias, especially butter. Clean features do not robustly add beyond task identity.

- vis_specific_physical:
    Bronze best StratumOnly P@5=0.20, enrich=1.3x, AUROC=0.69.
    Silver best CleanNoTaskNoTiming P@5=0.40, enrich=1.8x, AUROC=0.46.
    Rescue aggregated positives only n=3, underpowered.
    Interpretation: physical bridge has promising but underpowered clean-feature signal. Needs more confirmed physical positives.

- random_sensitive:
    Bronze best StratumOnly P@5=0.60, enrich=3.4x, AUROC=0.70.
    Silver best CleanNoTaskNoTiming P@5=0.60, enrich=1.5x, AUROC=0.77.
    Rescue aggregated positives only n=3, underpowered.
    Interpretation: random-sensitive is a real confounder and should be modeled as abstention, not negative.

High-level detector conclusion:
- cmd_specific detector: not yet effective as a clean-feature online detector; currently task-biased.
- physical bridge selector: promising but underpowered.
- random_sensitive abstain head: useful and necessary.
- Current best story is a multi-head selector:
    final_attack_score = p(physical_bridge or cmd_specific) - lambda * p(random_sensitive)
  but this still needs more balanced data.

============================================================
7. Scientific interpretation / allowed claims
============================================================

Allowed claims:
1. RC1a corrected VIS objective is now semantically aligned and produces repeated command-level effects.
2. A subset of windows show VIS-specific physical qpos response.
3. Random/general perturbation-sensitive windows are real and must be separated from true VIS-specific vulnerable windows.
4. Detector v0 shows weak/exploratory top-K signals, strongest for abstention/random-sensitive and physical bridge, not for command-specific vulnerability.
5. The next detector should be multi-head, not a simple binary classifier.

Forbidden claims:
1. Do not claim final detector solved.
2. Do not claim broad LIBERO generalization.
3. Do not claim real robot readiness.
4. Do not claim official LIBERO SR is the main evidence.
5. Do not treat old 44-row as formal attack baseline.
6. Do not treat Bronze labels as Gold.
7. Do not merge random_sensitive into negative class.

============================================================
8. Immediate next steps, but do not execute until new session review
============================================================

Recommended next direction after handoff, not to be launched now:

Step 1 — Freeze state and audit artifacts
- Ensure a3aecd7 is pushed.
- Confirm detector rescue_override uses aggregated rescue labels, not last-row overwrite.
- Confirm generated handoff paths and output artifacts.

Step 2 — Targeted data expansion, not blind expansion
Goal: improve detector by balancing labels, not just increasing N.

Need more:
- non-butter cmd_specific positives
- confirmed physical positives
- hard negatives
- stable random_sensitive controls
- same-task positive/negative contrasts
- same-episode adjacent windows with different labels

Avoid:
- more butter-only positive-neighborhood sampling
- treating random-sensitive as negative
- blind 48/72-window expansion

Target label counts before stronger detector claims:
- stable_cmd_specific >= 25–30, across multiple tasks
- stable_physical >= 15–20
- stable_random_sensitive >= 25–30
- stable_hard_negative >= 40
- no single task should dominate positives > 25–30%

Step 3 — Visual sidecar pilot
Do not integrate visual into runner yet.
Run offline sidecar extraction from RC1a clean frames:
- window_start / mid / end agentview images
- frozen visual embeddings: CLIP / DINOv2 / OpenVLA vision tower if easy
- compare feature groups:
    ProprioOnly
    VisualOnly
    Proprio+Visual
    TaskOnly
    Task+Visual
    Shuffle
- targets:
    cmd_specific
    vis_specific_physical
    random_sensitive
- same GroupKFold by task_state_seed / overlap clusters.

Purpose:
- See whether visual features reduce task bias.
- Especially test whether visual features improve physical_bridge P@K.

Step 4 — Future method packaging
Reframe detector as:
  Clean-forward multi-head online selector:
    vulnerability / physical bridge score
    random-sensitive abstain score
    optional task/phase prior
  Attack only when vulnerability high and random-sensitive low.

============================================================
9. Must-include artifact index
============================================================

Add an artifact index with paths if available. At minimum include these categories:

Code:
- src/gripper_attack/openvla_libero_exec_spec.py
- src/gripper_attack/attack_adapter.py
- scripts/run_stageb_vis_labeling.py
- scripts/stageb/validate_stageb_trace_v1_1.py
- scripts/stageb/postprocess_traces_v1_1.py
- scripts/stageb/build_pair_labels_v1_1.py
- scripts/diagnostics/run_detector_v0_fixed.py

Data/output roots:
- /data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/
- /data/liuyu/outputs/stageb_v1_1_corrected_smoke3b_rc1a_20260607/
- /data/liuyu/outputs/stageb_v1_1_bronze_batch_rc1a_20260607/
- /data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608/
- /data/liuyu/outputs/stageb_v1_1_silver_p1b_rc1a_20260608/
- /data/liuyu/outputs/stageb_v1_1_random_confounded_rescue_rc1a_20260608/
- /data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/

Reports/tables:
- All Stage-B RC1a readiness / clean reachability / candidate generation / Bronze / Silver / P1b / Rescue / Detector readouts currently present.
- If a file is missing, mark MISSING instead of inventing path.

============================================================
10. Handoff style
============================================================

Write the handoff in Markdown.
Use clear section headers.
Keep it concise but self-contained.
Use tables where helpful.
Do not include raw logs except short critical snippets.
Do not hide uncertainty.
Mark any unverified assumption explicitly.

At the end, include:

## One-paragraph takeaway

A concise summary like:

"The corrected RC1a VIS pipeline is now trustworthy and produces repeatable command/physical attack effects, but the detector is not yet solved. Command-specific prediction is currently task-biased; the more promising route is a multi-head selector that predicts physical bridge susceptibility while abstaining from random-sensitive windows. Next work should focus on targeted balanced data expansion and a visual sidecar pilot, not blind scaling or old-result reuse."

============================================================
11. Commit and push
============================================================

After writing the handoff:

1. git status --short
2. git add only:
   reports/STAGEB_RC1A_SESSION_HANDOFF_20260608.md
   optional tables/stageb_rc1a_session_handoff_artifact_index.csv
3. git commit -m "docs(stageb): add rc1a session handoff"
4. git push origin exp/vis-prefix-margin-repair-20260603
5. Report final remote tip SHA.

Do not stage unrelated dirty files.
```

---

## Short version for the next GPT session

If the new session needs a one-paragraph context seed, use:

```text
We are working on an OpenVLA-LIBERO inference-time visual PGD attack targeting gripper duty-cycle vulnerability. The old Stage-B data were quarantined due to inverted gripper open semantics. RC1a fixed the official action chain: raw_gripper > 0.5 -> env_action_6 = -1.0 -> physical OPEN; raw==0.5 is boundary. The corrected pipeline now has trusted v1.1 traces, seed-aware pairing, and detector scripts. Experiments show corrected VIS reliably induces command-level effects and some physical qpos responses, but the detector is not yet a strong clean-feature vulnerable-window detector. cmd_specific is task-biased, physical_bridge is promising but underpowered, and random_sensitive is a real confounder requiring an abstain head. Next work should freeze current state, avoid old labels, do targeted balanced expansion, and test a frozen visual sidecar rather than blindly scaling or claiming detector success.
```
