# Migration Gap Register — A800 Migration 2026-06-20

**Migration branch:** `infra/a800-migration-20260620`
**Base commit:** `141657fdc5d85c5fd564913c955d61e9e6be9ddc` (main)
**Auditor:** DeepSeek
**Status after first-round audit:** BLOCKED at M0

---

## Gate Status Summary

```
A800_CODE_BOOTSTRAP              = ALLOWED (in progress)
A800_READ_ONLY_HOST_AUDIT        = ALLOWED (complete — 2026-06-20)
A800_DEDICATED_ENV_BUILD         = BLOCKED_BY_ROOT        ← M0
OLD_SERVER_METADATA_AUDIT        = ALLOWED_READ_ONLY (complete)
OLD_SERVER_BULK_DATA_TRANSFER    = BLOCKED
A800_FORMAL_ROLLOUTS             = BLOCKED
A800_ATTACK_ROLLOUTS             = FORBIDDEN
OLD_SERVER_DATA_DELETION         = FORBIDDEN
PRIMARY_HOST_CUTOVER             = BLOCKED
```

---

## M0 — Host Safety

### PASS Items

| # | Check | Evidence |
|---|---|---|
| M0.1 | /mnt/sdc writable | `touch /mnt/sdc/dty_user/openvla_attack/tmp/.write_test` → 0 |
| M0.2 | Cache/tmp on /mnt/sdc | All `$TMPDIR`, `$HF_HOME`, `$TORCH_HOME`, etc. point to `/mnt/sdc/dty_user/openvla_attack/` |
| M0.3 | No sudo required | `dty_user` not in sudo group; `sudo -l` would fail |
| M0.4 | GPU lease mechanism | Lock dir exists at `/mnt/sdc/dty_user/openvla_attack/gpu_locks/` |
| M0.5 | No other user impact | Only read-only commands executed |
| M0.6 | Migration branch created | `infra/a800-migration-20260620` pushed to origin |
| M0.7 | Directory skeleton | `/mnt/sdc/dty_user/openvla_attack/` complete |
| M0.8 | Git remotes documented | origin + vla + vla-work mapped |
| M0.9 | Codex branches identified | 5 branches catalogued, none touched |
| M0.10 | Freeze tags catalogued | 7 tags identified |

### FAIL Items

| # | Check | Current | Required | Blocking |
|---|---|---|---|---|
| M0.11 | Root free space | **55 MiB** | ≥ 20 GiB | **YES** |

### M0 Evidence

```
$ df -h /
Filesystem                  Size  Used Avail Use% Mounted on
/dev/mapper/system-lv_root  428G  428G   55M 100% /
```

Root partition contributors:
- `/home` = 343G (ysc2: 177G, sz: 44G, huanzze: 27G, jsq: 16G, dty_user: 16G, others: ~47G)
- `/var/log` = 1.1G
- `/tmp` = 3.7G

dty_user's home (16G) is not the primary consumer. Admin action on other users' homes is needed.

---

## M1 — Environment Ready

**Status: BLOCKED (depends on M0)**

M1 validation script ready but cannot run `conda create` or `pip install` until M0 passes.

Pre-requirements:
- [ ] Admin frees ≥ 20 GiB on root
- [ ] Verify /mnt/sdc still has ≥ 700 GiB
- [ ] Verify candidate GPUs (4, 6) still available

Planned env:
- Path: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- Python: 3.10 (matched to OpenVLA upstream requirements)
- Source: Fresh from OpenVLA upstream README + LIBERO requirements

---

## M2 — Static Parity

**Status: BLOCKED (depends on M1)**

20-frame plan:
- Object: 5 frames
- Spatial: 5 frames
- Goal: 5 frames
- LIBERO-10: 5 frames

Source frames: to be sampled from CLEAN300 frozen dataset.
Scripts: not yet written (depends on M1 env + model availability).

---

## M3 — Closed-Loop Clean Parity

**Status: BLOCKED (depends on M2)**

8-episode canary plan:
- 4 suites × 2 clean episodes each
- Tasks/seeds: TBD from CLEAN300 registry
- Gate: no systematic clean performance degradation

---

## M4 — Transfer Acceptance

**Status: BLOCKED (depends on M3)**

Old server census complete (metadata-only). Tier classification preliminary:
- T0: CLEAN300 dataset, Layer 1/2/3 POC outputs, detector checkpoint
- T1: Train300 outputs, H2 validation packages
- T2: OpenVLA checkpoints (5), conda envs (3)
- T3: Overlay videos, CSVs, plots
- T4: Historical experiments (pre-May)
- TX: Active writers, temporary files

---

## M5 — Primary Cutover

**Status: BLOCKED (depends on M4)**

Far future gate. No action needed now.

---

## Blocker Register

### BLOCKER #1: Root partition full

| Field | Value |
|---|---|
| Severity | CRITICAL |
| Gate | M0 |
| Detail | Root has 55 MiB free. Need ≥ 20 GiB for conda/pip/bootstrap. |
| Root cause | Multi-user home directories consume 343G. dty_user is only 16G of that. |
| Required action | **Admin must clean other users' homes or expand root LV.** |
| Can DeepSeek fix? | **NO** — would require sudo or impacting other users |
| Evidence | `df -h /` → 428G/428G used |
| Request | **APPROVE_GATE_MIG0_ENV_BUILD** after admin confirms root ≥ 20 GiB free |

---

## GitHub Status

| Item | Detail |
|---|---|
| Migration branch | `infra/a800-migration-20260620` |
| Base | `origin/main` @ `141657f` |
| Pushed | Yes |
| PR created | No (pending initial doc commits) |
| Codex branches | Catalogued, zero-contact confirmed |

---

## Completed This Round

1. GitHub full audit (remotes, branches, tags, freeze tags, Codex branches)
2. A800 real-time host audit (disk/GPU/process/memory/env)
3. Migration directory skeleton created and verified on `/mnt/sdc`
4. Cache/tmp paths all redirected to `/mnt/sdc`
5. Migration branch `infra/a800-migration-20260620` created and pushed
6. Old 2080Ti server metadata-only census complete
7. Preliminary T0–TX tier classification
8. M0 gate evidence documented
9. All GPU UUIDs captured
10. Path writability verified

## Next Autonomous Actions (No Approval Needed)

1. Write `A800_MIGRATION_PROTOCOL.md` skeleton
2. Write migration scripts (`audit_compute_host.py`, `capture_runtime_fingerprint.py`, etc.)
3. Get OpenVLA upstream commit SHA and LIBERO upstream SHA (from GitHub)
4. Verify freeze tag SHAs correspond to actual frozen commits
5. Push all docs to `infra/a800-migration-20260620`

## Gate Request

**APPROVE_GATE_MIG0_ENV_BUILD** — pending admin cleanup of root partition.

Cannot proceed beyond read-only audit and script-writing until root ≥ 20 GiB free.
