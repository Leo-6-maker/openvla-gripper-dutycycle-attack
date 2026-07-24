# DeepSeek Handoff — C2f Observation/Language Detector Route

Date: 2026-07-08
Branch: `plan/codex-gated-experiment-v1-c2e0`
Status at handoff: D7B2 still running; D8F closed 25D-only detector route; C2f is next.

## Non-negotiable boundaries

1. Do not modify D7B2 workers, detector, thresholds, or rollout root.
2. Do not use D7B2 partial outcomes for C2f training or threshold selection.
3. C2f is post-D7 and must not enter D7 Table 1.
4. Student model inputs must not include privileged state, attack outcome, manual failure label, or matched attacked-run result.
5. D7B2 completion path remains: D7C audit -> D7D aggregate -> D7E render.

## Why C2f starts now

D8F closed the 25D-only route:

- D8F1 selective abstention: Object/Goal/Spatial nearly solved, L10 abstain collapse.
- D8F2 suite-balanced + L10 positive weighting: L10 rescue failed.
- Conclusion: 25D proprio/action cannot disambiguate L10 primary-vs-distractor events.

C2f adds RGB + task language to answer the missing semantic question: is the current manipulation the task-language-specified primary event?

## Files added for C2f

| File | Purpose |
|---|---|
| `docs/detectors/C2F_OBSERVATION_LANGUAGE_DATA_SPEC.md` | C2f schema, allowed/forbidden inputs, gates. |
| `scripts/stageb/collect_c2f_observation_clean_rollouts.py` | Observation-rich CLEAN rollout artifact writer + adapter boundary. |
| `tools/multisuite_detector/materialize_c2f_frozen_embeddings.py` | Converts C2f clean artifacts to windowed NPZ with RGB/lang embeddings. |
| `tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py` | Trains C2f RGB+language+25D student detector. |

## Immediate DeepSeek tasks

### P0 — Let D7B2 finish

Do not touch D7B2 except liveness checks. After 716/716:

```bash
# example placeholders; use actual D7B2 output root
python scripts/stageb/audit_d7_table1_postrun.py \
  --episode-root <D7B2_OUT> \
  --output-dir <D7C_AUDIT_OUT> \
  --git-commit <CURRENT_COMMIT>

python scripts/stageb/aggregate_d7_table1_four_suite.py \
  --episode-root <D7B2_OUT> \
  --audit-report <D7C_AUDIT_OUT>/audit_report.json \
  --output-dir <D7D_AGG_OUT> \
  --git-commit <CURRENT_COMMIT>
```

D7D is forbidden unless D7C says PASS.

### P1 — Implement the C2f runtime adapter

The collector is schema-stable but adapter-free by design. Implement a module function:

```python
# e.g. scripts/stageb/c2f_libero_openvla_adapter.py

def make_adapter(args):
    return MyAdapter(args)
```

The adapter must implement:

```python
class MyAdapter(RuntimeAdapter):
    def run_clean_episode(self, episode_cfg):
        yield StepRecord(
            step=t,
            rgb_array=rgb_uint8_hwc,
            rgb_path=None,
            features_25d=[...],
            task_language=language,
            teacher_hazard=0_or_1,
            teacher_primary_attackable=0_or_1,
            teacher_release_safe=0_or_1,
            teacher_event_role="primary_attackable|auxiliary_manipulation|distractor_or_setup|unsupported_or_abstain",
            teacher_phase="stable_carry",
        )
```

Use clean privileged state only for labels. Do not put privileged values in student features.

### P2 — Run L10 pilot collection

Start with one worker only while D7B2 is still near completion. After D7 is frozen, scale up.

```bash
CUDA_VISIBLE_DEVICES=<IDLE_GPU> \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nice -n 10 ionice -c2 -n7 \
python scripts/stageb/collect_c2f_observation_clean_rollouts.py \
  --manifest <L10_CLEAN_PARENT_MANIFEST.jsonl> \
  --output-root /mnt/sdc/dty_user/openvla_attack_evidence/c2f/l10_pilot_obs_clean_<RUN_ID> \
  --adapter-module scripts.stageb.c2f_libero_openvla_adapter:make_adapter \
  --suite libero_10 \
  --max-episodes 100 \
  --git-commit <CURRENT_COMMIT> \
  --source-commit <CURRENT_COMMIT>
```

Expected artifact root:

```text
c2f/l10_pilot_obs_clean_<RUN_ID>/
  manifest.json
  SHA256SUMS
  episodes/libero_10/<parent_key>/
    episode_metadata.json
    step_records.jsonl
    rgb/frame_000000.png
```

### P3 — Materialize frozen embeddings

Smoke test first with `--backend stats`. Then use CLIP if dependencies and network/model cache are ready.

```bash
python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
  --c2f-root /mnt/sdc/dty_user/openvla_attack_evidence/c2f/l10_pilot_obs_clean_<RUN_ID> \
  --output-dir /mnt/sdc/dty_user/openvla_attack_evidence/c2f/l10_pilot_embeddings_<RUN_ID> \
  --window 16 \
  --backend stats \
  --device cpu \
  --git-commit <CURRENT_COMMIT>
```

For CLIP:

```bash
CUDA_VISIBLE_DEVICES=<IDLE_GPU> python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
  --c2f-root <C2F_ROOT> \
  --output-dir <EMB_OUT> \
  --window 16 \
  --backend clip \
  --model-name openai/clip-vit-base-patch32 \
  --device cuda \
  --git-commit <CURRENT_COMMIT>
```

### P4 — Train C2f v0

```bash
CUDA_VISIBLE_DEVICES=<IDLE_GPU> \
python tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
  --dataset <EMB_OUT>/c2f_w16_clip_dataset.npz \
  --output-dir /mnt/sdc/dty_user/openvla_attack_evidence/c2f/c2f_train_v0_<RUN_ID> \
  --device cuda \
  --epochs 30 \
  --batch-size 256 \
  --git-commit <CURRENT_COMMIT>
```

Minimum useful report fields:

- overall recall / FP / F1
- macro recall / macro FP
- L10 recall / L10 FP
- per-suite metrics
- threshold sweep `best_c2f_gate`

## C2f gates

C2f is not deployable unless held-out test satisfies:

```text
L10 recall >= 45.6%
L10 FP <= C2e3 L10 FP
Overall FP <= 30%
Object/Goal/Spatial do not collapse
Runtime/artifact provenance PASS
```

If these fail, report C2f as diagnostic only.

## Suggested next commit sequence

1. `feat(C2f): implement LIBERO/OpenVLA clean rollout adapter`
2. `data(C2f): collect L10 observation-rich clean pilot`
3. `feat(C2f): materialize frozen CLIP embeddings`
4. `train(C2f): run RGB-language-temporal detector v0`
5. `audit(C2f): add held-out and per-suite gate report`

## P0 reminder

D7B2 Table 1 remains the priority. C2f must never delay D7C/D7D/D7E once the rollout hits 716/716.
