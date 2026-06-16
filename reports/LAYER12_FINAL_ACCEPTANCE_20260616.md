# Layer 1/2 Final Acceptance Report

**Date:** 2026-06-16
**Operator:** Yuliu / Claude
**Frozen baseline:** `52bdc33507a584e158581f9eded6dc657ac71029`

## Phase Gate Summary

| Phase | Description | Result |
|-------|-------------|--------|
| Phase 0 | Post-reboot freeze check | ALL GATES PASS |
| Phase 1 | GPU qualification | 6/8 usable, 2 faulty documented |
| Phase 2 | Sidecar OFF/ON smoke | 5/5 identity PASS |
| Phase 3 | Production streaming parity | PARTIAL — scoring parity PASS, live feature NOT TESTED |

## 1. LAYER1_CAPTURE_AND_LABELS

**Status: PASS**

- 120-state balanced capture (D4.4D) — 120/120 BOUND
- Teacher-P labels v2 — 99 labeled + 21 abstain
- Independent Teacher-P auditor — 120/120 PASS, 0 disagreements
- Accepted episode manifest — SHA `fe125a55`
- Labels manifest — SHA `e731c273`

Evidence:
- [d5_teacher_p_independent_audit.csv](../tables/l12_frozen_artifact_manifest.csv)
- [d5_teacher_p_labels_v2.csv](../tables/l12_frozen_artifact_manifest.csv)

## 2. LAYER2_FROZEN_CAUSAL_REPLAY

**Status: PASS**

- D5 candidate ranker trained (MLP [128,64,32], 16 features)
- tau = 0.050 frozen on validation
- Internal: 75/99 in-window (75.8%), 95/99 emit
- Internal test: 11/15 in-window (73.3%), 15/15 emit
- D5 causal replay: 76% (vs First-CLOSE 74%, D1b 53%)
- Frozen config: `d5_frozen_config.json` SHA `d6f6af61`
- Checkpoint: `d5_candidate_best.pt` SHA `7eea609f`

Evidence:
- [d5_evaluation_readout.csv](../tables/l12_frozen_artifact_manifest.csv)
- [d5_evaluation_summary.json](../tables/l12_frozen_artifact_manifest.csv)

## 3. LAYER2_PRODUCTION_STREAMING_PARITY

**Status: PARTIAL — scoring parity only**

### 3a. LAYER2_FROZEN_SCORING_PARITY: PASS

Both paths read the same `detector_candidates.csv` features:
- Path A: frozen D5 replay (`evaluate_d5_frozen.online_detect`)
- Path B: direct normalization + MLP scoring on same CSV features

- Internal 120/120: score diff ≤ 1e-6, emit step match, 0 abstain mismatch
- External 34/34: score diff ≤ 1e-6, emit step match, 0 abstain mismatch
- Negative fail-closed tests: 9/9 PASS

This verifies that normalization and MLP scoring are deterministic.

### 3b. LAYER2_LIVE_FEATURE_EXTRACTION_PARITY: NOT TESTED / KNOWN MISMATCH

Feature extraction code (`critical_close_selector.py`) has evolved through E1.5/E3.1/E4A/E4B.1 since the D4 capture (commit 44bf7b86). Live feature re-computation via `ProductionStreamingDetector._extract_features()` produces different feature values than the frozen `detector_candidates.csv`. The D5 model was trained on capture-time features.

### 3c. LAYER2_TRUE_ONLINE_DEPLOYMENT: BLOCKED ON FROZEN FEATURE ADAPTER

A production deployment cannot read pre-computed `detector_candidates.csv` at inference time. True live streaming parity requires a frozen feature adapter that recovers capture-time feature semantics from step_trace.csv inputs only.

Next step: implement `src/gripper_attack/d5_frozen_feature_adapter_v1.py` and re-run true two-path parity (CSV replay vs live adapter streaming).

Evidence:
- [d5_production_streaming_parity.csv](../tables/d5_production_streaming_parity.csv)
- [d5_production_streaming_negative_tests.csv](../tables/d5_production_streaming_negative_tests.csv)

## 4. LAYER2_EXTERNAL_GENERALIZATION

**Status: MEASURED, NOT SOLVED**

34-state external evaluation:
- 27 labeled + 7 abstain
- 16/27 in-window (59.3%)
- 26/27 emit (96.3%)
- 1 miss (3.7%)
- 10/10 tasks covered

The 59% external in-window rate confirms the detector generalizes above chance but is not robust. Cannot claim "strong generalization."

Evidence:
- [d5_34eval_readout.csv](../tables/l12_frozen_artifact_manifest.csv)

## 5. LAYER2_TIMING_QUALITY

**Status: NOT FINAL**

Early-trigger problem remains significant:
- Internal: 19/99 early (19.2%) — detector fires before Teacher-P window
- External: 9/27 early (33.3%) — one third of labeled states fire too early

33% external early rate is a recognized risk. Future optimization of early rate must use train/validation only and evaluate on a new untouched external set.

## 6. LAYER1/2_TO_LAYER3_ACTIONABILITY

**Status: NOT TESTED / NOT AUTHORIZED**

Layer3 trigger is NOT AUTHORIZED. The current detector is a close-event detector — it identifies when a CLOSE grip is likely occurring. Whether this timing is optimal for VIS attack (Layer3) has not been evaluated. The relationship between "detector-triggered close event" and "VIS-attack-effective frame" is untested.

## Authorized Declarations

"Layer 1/2 engineering chain is operational end-to-end:
clean rollout → Teacher-P labels → D5 training → frozen replay →
live non-invasive sidecar → frozen-candidate scoring parity.

True live feature extraction parity requires a frozen feature adapter
recovering capture-time feature semantics. Blocked on this adapter."

## Prohibited Declarations (NOT made)

- Detector solved
- Robust broad generalization
- Early-trigger problem solved
- VIS-optimal trigger found
- Layer3 ready
- 59% external is strong generalization
- D5 significantly outperforms First-CLOSE (76% vs 74% is a small internal improvement)

## Evidence Index

| Artifact | Path | SHA256 |
|----------|------|--------|
| Frozen config | `/data/liuyu/outputs/d5_training/d5_frozen_config.json` | `d6f6af61...` |
| D5 checkpoint | `/data/liuyu/outputs/d5_training/d5_candidate_best.pt` | `7eea609f...` |
| Teacher-P labels | `/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv` | `e731c273...` |
| Accepted manifest | `/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv` | `fe125a55...` |
| Independent audit | `/data/liuyu/outputs/d5_label_generation/d5_teacher_p_independent_audit.csv` | `fb7a8aac...` |
| D5 eval readout | `/data/liuyu/outputs/d5_training/d5_evaluation_readout.csv` | `76dd307b...` |
| 34-eval readout | `/data/liuyu/outputs/d5_training/d5_34eval_readout.csv` | `0ba7c63b...` |
| Sidecar identity | `tables/l12_sidecar_off_on_identity.csv` | (local) |
| Parity CSV | `tables/d5_production_streaming_parity.csv` | (local) |
| Negative tests | `tables/d5_production_streaming_negative_tests.csv` | (local) |
| GPU qual | `tables/l12_gpu_qual.csv` | (local) |
| Freeze check | `reports/L12_POST_REBOOT_FREEZE_CHECK.md` | (local) |
| GPU qual report | `reports/L12_POST_REBOOT_GPU_QUAL.md` | (local) |

## GPU Status

| Pair | GPUs | Scope |
|------|------|-------|
| (2,6) | GPU2+6 | Full: capture + attack (M3 qual PASS) |
| (1,3) | GPU1+3 | Full: capture + attack |
| (5,7) | GPU5+7 | Capture only (GPU7 forward-only) |
| GPU0, GPU4 | — | FAULT — excluded |
