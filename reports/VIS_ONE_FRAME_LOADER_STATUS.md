# VIS One-Frame Loader Status

Date: 2026-05-31

## Status

Implemented:

```text
scripts/diagnostics/vis_one_frame_loader.py
```

The loader performs a real no-rollout diagnostic path:

```text
saved RGB frame
-> OpenVLA clean decode
-> TokenPrefixPGDAttacker.attack(...)
-> attack_result.debug["adv_inputs"]
-> redecode_openvla_action_from_adv_inputs(...)
-> single-row CSV metrics
```

It never calls `env.step`, never runs a rollout, never uses `action_adv`, and never falls back to zeros.

## Smoke Run

Input frame:

```text
/data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/frames/step_0000.png
```

Command used physical GPU2/6 only:

```bash
CUDA_VISIBLE_DEVICES=2,6 OPENVLA_ATTN_IMPLEMENTATION=eager OPENVLA_CUDA_MAX_MEMORY=9500MiB \
PYTHONPATH=src python scripts/diagnostics/vis_one_frame_loader.py \
  --image_path /data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/frames/step_0000.png \
  --step_records /data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530/runs/libero_object/vis_ketchup_clean_ketchup_s0/step_records.jsonl \
  --step_idx 0 \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
  --unnorm_key libero_object \
  --objective target_action_ce \
  --eps 4/255 \
  --steps 1 \
  --step_size 1/255 \
  --model_gpu_device_id -1 \
  --center_crop \
  --postprocess_gripper \
  --output_csv tables/vis_one_frame_loader_smoke.csv
```

First attempt on single visible GPU2 failed with CUDA OOM. Second attempt with visible GPUs `2,6` passed.

## Result

Output:

```text
tables/vis_one_frame_loader_smoke.csv
```

Key metrics after the TokenPrefixPGD processor-pixel budget fix:

- clean gripper token: `31872`
- adversarial gripper token: `31872`
- token flip: `false`
- clean gripper action: `0.0`
- adversarial gripper action: `0.0`
- gripper delta: `0.0`
- arm L2: `0.184442`
- target CE: `32.0000 -> 15.9500`
- open-bin mass: `5.87e-13 -> 1.52e-07`
- close-bin mass: `0.999996 -> 0.987568`
- perturbation Linf: `0.0078125`
- model dtype: `torch.bfloat16`
- pixel_values dtype: `torch.bfloat16`

## Gate VIS-Loader

Result: PASS.

The loader successfully decoded both clean and adversarial actions from real model execution and `debug["adv_inputs"]`, without fallback zeros and without dtype mismatch.

The perturbation Linf is now within the requested `eps=4/255` under the explicit `processor_pixel_values_linf` semantics.

VIS-1 still fails because the decoded gripper token/action did not change and arm drift is nontrivial.
