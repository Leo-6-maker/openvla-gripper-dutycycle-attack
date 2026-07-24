# Evidence Backup Status — 2026-07-02

## Current State

| Item | Status |
|---|---|
| Object evidence (sc5_object_privileged_loto_v1) | dty-server ONLY |
| CLEAN2000 | dty-server ONLY |
| Server code (dirty + clean) | dty-server + vla (partial) + local |
| Local machine evidence copy | NONE (only SCHEMA_CANARY_GATE.json + canary/) |
| **BACKUP_STATUS** | **HOLD_NO_SAFE_SECONDARY_STORAGE** |

## Object Evidence

Single point of failure: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/`

Contains:
- vis_heldout_formal_v1/ (3,495 dirs)
- fold_01–09/ (LOTO detector folds with teacher labels, features, checkpoints)
- Frozen attack results for 14 conditions
- LOTO detector training checkpoints

## Secondary Storage Candidate

| Item | Value |
|---|---|
| Server | vla (10.60.133.4) |
| Path | `/data/liuyu/` |
| Filesystem | /dev/sdb |
| Total capacity | 1.8T |
| Used | 637G (37%) |
| Available | 1.1T |
| Write permission | YES (liuyu user) |
| Network path | dty-server → vla-jump (10.60.133.3) → vla (10.60.133.4) |

## Backup Plan (NOT YET EXECUTED)

```bash
# Estimate Object evidence size
du -sh /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/

# Rsync to vla (via jump host)
rsync -avz --progress \
  -e "ssh -J scene@10.60.133.3" \
  /mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/ \
  liuyu@10.60.133.4:/data/liuyu/openvla_attack_evidence/sc5_object_privileged_loto_v1/

# Verify SHA after copy
diff <(cd src && find . -type f -exec sha256sum {} \; | sort) \
     <(cd dst && find . -type f -exec sha256sum {} \; | sort)
```

## Constraints

1. /mnt/sdc is 95% full — cannot create local tarball
2. Rsync must go through jump host (10.60.133.3)
3. Must preserve directory structure and all SHAs
4. Must not interfere with other users' GPU jobs during transfer
5. CLEAN2000 is ~574MB — trivial to backup
6. Object evidence size unknown — needs measurement before transfer

## What CANNOT be backed up

- GPU-specific binary artifacts tied to physical GPU state
- Running process state (already stopped)
- /mnt/sdc/dty_user/table1_sota_execution_v1/ (execution workspace, not scientific evidence)

## Minimum Viable Backup

Priority order:
1. Object frozen evidence (vis_heldout_formal_v1 + fold_01–09)
2. CLEAN2000_CANONICAL_V1 (~574 MB)
3. Server git diff + dirty file copies
4. table1_sota_execution_v1 logs and manifests
5. Model checkpoints (may already exist elsewhere)

---

NO BACKUP HAS BEEN EXECUTED YET.
THIS IS THE SOLE REMAINING INDEPENDENT HOLD.
