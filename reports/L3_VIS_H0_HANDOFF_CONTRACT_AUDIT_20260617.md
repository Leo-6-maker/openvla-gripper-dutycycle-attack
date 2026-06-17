# L3 VIS H0 Handoff Contract Audit

Status: `BLOCKED_PENDING_CLEAN_PACKAGE_REPAIR`

This is a CPU-only handoff contract audit for the Layer3 VIS fixed-frame panel. It did not run GPU inference, PGD, RAND controls, SHUFFLED controls, or LIBERO rollout.

## Inputs

- Branch: `exp/l3-vis-handoff-contract-repair-20260617`
- Base handoff commit: `50da442c1b033a780b802c6345c376b23d4833b1`
- Server audit output: `/data/liuyu/outputs/l3_vis_h0_contract_audit_20260617_r2`
- Frame manifest: `/data/liuyu/outputs/l12_frame_handoff_v2_r1/frame_manifest.json`
- Selected frames: `tables/l3_vis_selected_frames_v1.csv`
- Job plan: `tables/l3_vis_job_plan_v1.csv`

## Gate Result

| Gate | Result |
| --- | --- |
| Selected 10-frame set | PASS |
| V4 job plan shape | PASS |
| Parent action/env identity | 3/3 EXACT_BOUND |
| Full frame inventory | 65/71 PRESENT |
| Raw frame SHA check | 10/10 PASS |
| Saved processor tensor SHA check | 10/10 PASS |
| Canonical attack tensor package | 0/10 PASS |
| Clean generation package | 0/10 PASS |
| Primary clean gripper token 31872 | 0/6 PROVEN |

The current metadata is sufficient to prove the selected frame files exist and the three selected parents match the timing episode on action and env-action hashes. It is not sufficient to launch H1, because the selected frames still lack a clean-generation package and a canonical attack-runner input tensor package.

## Blocking Defects

- `prompt_instruction` is empty for every selected frame.
- No selected frame has `clean_generation.json` with exact 7 clean tokens, clean arm prefix, clean gripper token, prompt token SHA, and model fingerprint.
- No selected frame has a canonical `processor_inputs_attack.pt` package produced by the same attack-runner preprocessing path.
- `obs_hash` is not present in the frame capture manifest, so the current EXACT_BOUND proof covers action/env identity but not obs-hash identity.
- The top-level frame inventory contains 65 captured frames, not the expected 71-frame inventory stated in the handoff contract.

## Allowed Claim

The H0 repair code can now fail-close the Layer3 VIS handoff and independently verify the selected frame set, V4 job plan, raw/proc tensor hashes, and parent action/env identity.

## Forbidden Claim

Do not claim VIS is better than random, official-token attack effect, closed-loop effect, or H1 readiness from this audit. GPU execution remains blocked until 10/10 frame packages include clean-generation and canonical attack tensor evidence.
