# M3 GPU15 Stage-2 Preparation Preregistration

This branch is CPU-only preparation for the next Layer3 watcher. It must not
modify `/data/liuyu/repos/m3_gpu15_autonomous_d02db0b`, must not touch the
current `m3_gpu15_autonomous_20260617_r3` output root, and must not launch a
second GPU watcher while the current S3 Tomato watcher is running.

## Entry Gate

Stage-2 may only be planned from:

- `S3_TOMATO_SCREEN/gate_result.json` with `status = PASS`
- a selected lambda whose TRUE row emits token `31744`, passes arm `>=5/6`,
  and beats both `RAND21` and `SHUFFLED`
- a matching selected row in `m3_v3_tomato_results.csv`
- an available Layer1/2 timing handoff CSV

If the Tomato gate fails, Stage-2 planning fails closed.

## S5 Multi-Parent Plan

The CPU planner selects exactly three eligible parents from the Layer1/2 timing
handoff, preferring distinct tasks. For each parent it creates command-ledger
rows for:

- seeds: `81, 82`
- conditions: `CLEAN`, `PGD_DELTA0`, `TRUE_PGD`, `RAND21`, `SHUFFLED`
- target token: `31744`
- selected lambda: copied from the completed Tomato S3 gate
- CUDA mapping: `CUDA_VISIBLE_DEVICES=1,5`
- render physical GPU: `1`

The planner only writes JSON/CSV plans. It does not execute these commands.

## S5 Gate

A parent passes only if both seeds pass the frozen full-selective criteria:

- token `31744`
- arm prefix match `>=5/6`
- TRUE margin greater than `RAND21`
- TRUE margin greater than `SHUFFLED`
- Linf pass
- strict route
- no fallback

S6 is enabled only if at least two of three parents pass.

## S6 Oracle Closed-Loop Plan

S6 is represented as a disabled command-plan stage in this CPU-only branch. It
may only be converted into executable GPU code after S5 has produced a reviewed
`>=2/3` pass result. No LIBERO rollout is launched by this preparation commit.

## Forbidden Claims

- No TRUE_PGD > random claim is made by this preparation.
- No multi-parent effect is established.
- No closed-loop Layer3 result is established.
- No detector-triggered integration is established.
