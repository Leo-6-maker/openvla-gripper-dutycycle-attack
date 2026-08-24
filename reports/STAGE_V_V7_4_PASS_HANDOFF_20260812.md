# Stage V V7.4 Fresh Qualification Handoff

Date: 2026-08-12  
Status: `PASS_FORMAL_PARENT_QUALIFICATION`  
Protocol: `STAGE_V_V7_FRESH_QUALIFICATION_V1_4`

## Decision

The corrected V7.4 fresh-clean qualification passed under a new root and the
corrected terminal-field contract. The report selected 10 qualified parents in
each suite (40 total); its independent auditor passed with 140/140 queue cells
`DONE_VALID`, no duplicate parent identities, and no worker errors.

The formal parent split is now frozen before any OPEN branch:

```text
TRAIN 24   VAL 8   TEST 8
per suite  6       2      2
```

The split was independently recomputed from the frozen parent manifest using
the registered salt and passed identity, ordering, source, and safety checks.
M4 is authorized to use the matched-clean-action physical-window contract.

## Bound provenance

- Runtime source commit: `17fb76971da28dc9a61aaead52cebea62b653a46`
- Runtime source tree: `6efec29765121710456ae93ffe292965490022bf`
- V7.4 protocol SHA256:
  `c24ecb5eedbb771b977db51cec7c23183a25414dd0f8a53085b3165fba6a2475`
- Qualification root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_V7_FRESH_QUALIFICATION_V1_4_20260812T094000Z`
- Report SHA256:
  `4d5b95ecff63e68228bf5ed859564c1908891dda4287aeb6e2ffd2f3da143887`
- Independent audit SHA256:
  `b08458468404c952287386d96242f2fc8ea3fbb79a913c2c308c523f113b6896`
- Formal parent manifest SHA256:
  `d5076655fff94e04e3b520706835c610b9c544b5b568a4b3fe49a1eed168c10b`
- V7 final receipt SHA256:
  `36af16ea9c79a7a6a6537d5be2de3b18a69fed2cce523b05c368009c99709118`

## Frozen split

- Split root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_V7_FORMAL_PARENT_SPLIT_V1_20260812T110000Z`
- Split artifact SHA256:
  `0811bb0890756a9e5078f8282dd0f049991e17f8e4b3c0b55ee1f19c88952a9f`
- Split salt: `STAGE_V_FORMAL_PARENT_SPLIT_V1_20260812`
- Independent split audit SHA256:
  `af95533119f0d8d1d39095c96dfd7ca1375789b6ea6aa2c7f0090cb846f2705f`

## Safety boundary

Execution used GPUs `0–7`, one project worker per GPU, and the authorized
20-GiB minimum free-memory gate. Foreign processes were not terminated or
interfered with. `eval160_reads=0`, `protected_eval_reads=0`,
`attack_rollouts=0`, and `vis_pgd_attack_rollouts=0`. V1.3.4 and the sealed
V7.3 root remain immutable and non-consumable.

The next scientific stage is M4 only: 40 formal parents × 24 frozen probes ×
`CONTROL/T3/T5/T10`, one execution per branch, with matched canonical clean
non-gripper actions during the primary physical window. No Teacher, Student,
Scheduler, timing, VIS, or protected evaluation is authorized until M4 is
independently audited and passes.
