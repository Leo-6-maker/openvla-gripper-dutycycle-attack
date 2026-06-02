# Milestone 1D: MuJoCo 2.3.7 Compat Preflight — BLOCKED

**Timestamp**: 2026-05-26T10:15Z (approx)
**Status**: `blocked_target_ssh_down`

## Preflight Findings (Incomplete)

### What Was Confirmed

| Item | Status | Details |
|------|--------|---------|
| Jump host (10.60.133.3) | UP | scene@scene-SYS-4028GR-TR2 |
| Target server (10.60.133.4) | UP (ping OK) | liuyu@klfy-SYS-4028GR-TR2, 0.3ms from jump |
| Target SSH port 22 | **DOWN** | Initially returned "Too many authentication failures", now timing out completely |
| OpenVLA compat env | EXISTS | /data/aviary/envs/openvla_compat, Python 3.10.18 |
| Main env MuJoCo | 3.8.0 | Confirmed via pip list |
| Main env robosuite | 1.4.1 | Confirmed |
| Main env numpy | 2.2.6 | Confirmed |
| Main env torch | 2.6.0 + CUDA 12.4 | Confirmed |
| Main env transformers | 4.40.1 | Confirmed |
| LIBERO | 0.1.1 | Installed, imports from openvla_sparse env |
| Object checkpoint | EXISTS | /data/aviary/models/openvla/openvla-7b-finetuned-libero-object |
| v4 runner | EXISTS | scripts/v4_run_eval_openvla.py |
| Prior audit tables | EXISTS | /data/liuyu/outputs/milestone_1b_object_repro_audit_20260526/tables/ |
| Primary clean root | EXISTS | /data/liuyu/outputs/libero_full4_clean_official_aligned_eager_10states_20260525 |
| All 8 GPUs | idle, persistence ON | RTX 2080 Ti, driver 530.41.03, temps 22-31C |
| Potential compat env | EXISTS | /data/aviary/envs/openvla_official_libero_20260525 (not yet checked) |

### What Could NOT Be Confirmed

1. Whether `openvla_official_libero_20260525` is already a MuJoCo 2.3.7 compat env
2. Whether mujoco-py is available or installable
3. Whether MuJoCo 2.3.7 can be pip-installed alongside the existing env
4. Whether the v4 runner can run with MuJoCo 2.3.7
5. Exact LIBERO env path for patching MuJoCo version

### SSH Failure Timeline

1. First attempt: `ssh vla` succeeded, got `klfy-SYS-4028GR-TR2`
2. Subsequent attempts: "Too many authentication failures" from target (SSH was running but rate-limiting)
3. User said wait 70s
4. After waiting: SSH on target now timing out completely (even from jump host on same LAN, 0.3ms ping)

## Blocker

**The SSH daemon on 10.60.133.4 (klfy-SYS-4028GR-TR2) appears to have crashed.**
The host is UP (ping responds, port 80 responds with HTTP 302 redirect to /login), but port 22 is not accepting connections.

## Proposed Preflight Script (Ready to Deploy)

When SSH is restored, the following script should be run first to complete preflight:

```bash
#!/bin/bash
# Preflight script for Milestone 1D
OUTDIR=/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526
mkdir -p $OUTDIR/reports $OUTDIR/tables

echo "=== 1. Check compat env ==="
/data/aviary/envs/openvla_official_libero_20260525/bin/python -c "import mujoco; print('mujoco:', mujoco.__version__)" 2>&1
/data/aviary/envs/openvla_official_libero_20260525/bin/pip list 2>/dev/null | grep -iE 'mujoco|robosuite|numpy|torch|transformers|libero'

echo "=== 2. Check mujoco-py ==="
/data/aviary/envs/openvla_compat/bin/pip list 2>/dev/null | grep mujoco-py
find /data/aviary/envs/ -name "mujoco_py*" -type d 2>/dev/null | head -5

echo "=== 3. Check LIBERO env tasks ==="
/data/aviary/envs/openvla_compat/bin/python -c "
import libero
print('libero path:', libero.__file__)
"

echo "=== 4. Check runner help ==="
/data/aviary/envs/openvla_compat/bin/python scripts/v4_run_eval_openvla.py --help 2>&1 | head -60

echo "=== 5. Object task IDs ==="
grep -B1 -A12 'ketchup\|bbq\|cream_cheese\|milk' configs/v4_tasks_libero_full4_20260518.yaml

echo "=== 6. Check if MuJoCo 2.3.7 can be installed ==="
pip install --dry-run mujoco==2.3.7 2>&1 || echo "DRY_RUN_NOT_SUPPORTED"

echo "PREFLIGHT COMPLETE"
```

## Next Action Required from User

**Restart SSH on klfy-SYS-4028GR-TR2 (10.60.133.4).**
The machine is running (ping OK, HTTP on port 80 works) but the SSH daemon has stopped.
