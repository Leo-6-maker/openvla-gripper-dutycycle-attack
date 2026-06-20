# Server Migration Report — pm-364c0001 (dty-server)

**Generated:** 2026-06-20  
**Connection:** `ssh -p 33571 dty_user@10.60.2.56` (alias: `dty-server`)  
**Auth:** SSH key `~/.ssh/id_ed25519_vla` (passwordless, configured 2026-06-20)

---

## 1. Hardware Specs

| Component | Detail |
|---|---|
| **CPU** | 128-core Intel Xeon Platinum 8378A @ 3.00GHz |
| **RAM** | 1.0 TiB (128 GiB used, 858 GiB available) |
| **GPU** | 8× NVIDIA A800-SXM4-80GB (80 GB VRAM each) |
| **GPU Arch** | Compute Capability 8.0 (Ampere) |
| **OS** | Ubuntu 20.04, Kernel 5.15.0-127-generic |
| **Uptime** | 77 days 12 hours (since ~Apr 4 2026) |
| **Load Avg** | 177 / 176 / 175 |
| **Internet** | Yes (pypi.org reachable) |

## 2. GPU Status (Snapshot 2026-06-20 15:24 UTC)

| GPU | Util | VRAM Used | User | Process |
|---|---|---|---|---|
| 0 | 15% | 2.1 GiB | yyn + dty_user | bavit train (`yyn`) + 5× pi0 eval (`dty_user`) |
| 1 | 14% | 25.7 GiB | zkx | CMIN-US ×2, CMIN-CN training |
| 2 | 50% | 17.2 GiB | **dty_user** | mmunlearner cache-build (qwen25vl) |
| 3 | 44% | 17.2 GiB | **dty_user** | mmunlearner cache-build (qwen25vl) |
| 4 | 0% | 17 MiB | — | **IDLE** |
| 5 | 0% | 78.9 GiB | dty_user | lerobot train ACT (PID 4025523, 16h runtime) ⚠️ |
| 6 | 0% | 14 MiB | — | **IDLE** |
| 7 | 99% | 21.5 GiB | dty_user | 5× pi0 eval_policy (dp_attack) |

### GPU Availability

- **Idle GPUs:** 4, 6 (completely free)
- **Lightly used:** 0 (2.1G/80G VRAM, 15% util — could fit more work)
- **You (dty_user) already occupy:** GPU 0 (shared), 2, 3, 5, 7
- **Other users:** yyn (GPU 0, 5), zkx (GPU 1), ysc2 (CPU only)

## 3. Storage & Mount Points

| Mount | Total | Used | Free | Usage | Write? | Notes |
|---|---|---|---|---|---|---|
| `/` (root) | 428G | 428G | **53M** | **100%** | ❌ (full) | CRITICAL — system at risk |
| `/mnt/sdc` | 2.9T | 1.9T | **904G** | 68% | ✅ | **Primary work location** |
| `/llm_jzm` | 2.9T | 2.7T | 57G | 98% | ✅ | Shared (conda envs, cached models) |
| `/boot` | 2.0G | 509M | 1.4G | 27% | ❌ | — |
| `/tmp` | (root) | 3.7G | — | 100% | ✅ | Ephemeral, don't store large files |

### Your Directory Sizes on `/mnt/sdc`

| Directory | Size | Purpose |
|---|---|---|
| `dty_user/pi0/` | 759G | π0 model data & training artifacts |
| `dty_user/Emu3.5/` | 132G | Emu3.5 model |
| `dty_user/cache/` | 88G | HF/modelscope/wandb/big_vision cache |
| `dty_user/RoboTwin_official/` | 85G | RoboTwin benchmark + policies |
| `dty_user/pi0_openpi/` | 53G | OpenPI variant |
| `dty_user/dp_env/` | 5.2G | Python venv for dp_attack |
| `dty_user/dp_attack/` | 60K | Your attack code (lightweight) |
| `dty_user/pi0_attack/` | 4.4M | Attack configs |
| **Total dty_user/** | **~1.2T** | |

### Other Users on `/mnt/sdc`

| Directory | Owner | Size |
|---|---|---|
| `yyn_bavit_new/` | yyn | — |
| `yangyenan/` | yyn | — |
| `zkx/` | zkx | — |
| `taozhen/` | tz | — |
| `vlm-unlearning-models/` | dty_user | — |
| `vlm-unlearning-pydeps/` | dty_user | — |
| `MMUnlearner-Merger/` | dty_user | 279G |
| `lerobot_piper/` | dty_user | 12G |

## 4. Software Environment

### Python & Package Manager

- System Python: 3.8.10
- Conda: available (`/llm_jzm/conda_envs/`)
- No Docker available on this server

### Key Conda Environments

| Env Path | Python | Key Packages | Owner |
|---|---|---|---|
| `/llm_jzm/mt/conda_envs/pi0` | 3.11.14 | torch 2.10.0, transformers 5.9.0, lerobot 0.4.4, jax 0.10.1 | shared (used by dty_user) |
| `/llm_jzm/conda_envs/foundationpose` | — | — | dty_user |
| `/llm_jzm/conda_envs/good` | — | — | dty_user |
| `/home/yyn/.conda/envs/bavit` | 3.11.15 | — | yyn |
| `/llm_jzm/airs/...conda_envs/lerobot-train` | — | lerobot (ACT training) | dty_user |

### Key Libraries in `pi0` env (likely your target)
- PyTorch 2.10.0
- Transformers 5.9.0
- LeRobot 0.4.4
- JAX 0.10.1 + CUDA 12 plugin
- CUDA 12.8 / Driver 550.90.12

## 5. Existing VLA-Related Assets

Your namespace already contains substantial work that is relevant to an OpenVLA attack migration:

```
/mnt/sdc/dty_user/RoboTwin_official/policy/
├── TinyVLA/
├── LLaVA-VLA/
├── pi05/
├── DexVLA/
├── openvla-oft/          ← OpenVLA fine-tuning
├── pi0/
└── pi0_attack/           ← existing attack code

/mnt/sdc/dty_user/dp_attack/      ← your current attack framework
├── config.py
├── deploy_policy.py
├── dp_policy.py
├── deploy_policy.yml
└── run_eval_grid.py

/mnt/sdc/dty_user/pi0_eval_local/  ← local eval shims
/mnt/sdc/dty_user/pi0_attack/      ← attack results/configs
/mnt/sdc/dty_user/cache/huggingface/ ← cached models
```

## 6. Current Workload (dty_user processes)

| PID | GPU(s) | CPU% | Task |
|---|---|---|
| 4025523 | 5 | 0% | lerobot ACT train (200K steps, `master_control_follower`) — possibly stuck |
| 964730 | 2 | 976% | mmunlearner cache-build (qwen25vl, unauthorized_data_collection) |
| 774022 | 3 | 965% | mmunlearner cache-build (qwen25vl, weapon_related_violence) |
| 2163947 | 0,7 | 98% | pi0 eval_policy (move_can_pot, dp_attack, uada) |
| 2262889 | 0,7 | — | pi0 eval_policy |
| 3224176 | 0,7 | — | pi0 eval_policy |
| 3722463 | 0,7 | — | pi0 eval_policy |
| 2009267 | 0,7 | — | pi0 eval_policy |

## 7. Key Issues & Risks

### Critical
1. **Root partition 100% FULL** — 53 MiB remaining. System logs, temp files, and user home dirs will fail. Requires cleanup (1.1G in `/var/log`, 3.7G in `/tmp`). You have no control over this as non-root — ask the admin to clean up.

### Moderate
2. **GPU 5 stuck process** (PID 4025523): 16h runtime, 78.9 GiB VRAM but 0% GPU utilization. Likely hung — consider killing it to free 80GB for migration work.
3. **`/llm_jzm` near full** — 57 GiB remaining. Don't store new large datasets here.
4. **No Docker** — containerized workflows won't work directly; use conda/venv.

### Strategic
5. **Multi-tenant server** — yyn, zkx, ysc2, tz share GPUs. GPU 1 (zkx) and GPU 0 (yyn partially) are in use. Coordinate GPU selection or use GPUs 4, 6 which are consistently idle.
6. **The `pi0` conda env at `/llm_jzm/mt/conda_envs/pi0` already has the right stack** (torch 2.10, lerobot 0.4.4, transformers 5.9.0, CUDA 12.4).

## 8. Migration Recommendations

### For OpenVLA Attack Task

1. **Target GPUs:** 4 and 6 (idle), or share GPU 0 (only 2.1G used). Avoid GPUs 1, 5.

2. **Work directory:** Use `/mnt/sdc/dty_user/openvla_attack/` — 904G free, under your ownership.

3. **Conda env:** Either reuse `/llm_jzm/mt/conda_envs/pi0` (already has torch+lerobot+transformers) or create a new env at `/llm_jzm/conda_envs/openvla_attack/` — but note `/llm_jzm` has only 57G free.

4. **Dataset storage:** `/mnt/sdc/dty_user/` (904G free). Keep an eye on total usage — you're at 1.2T of the 2.9T partition.

5. **Code migration:** Your existing `dp_attack` framework and `RoboTwin_official/policy/openvla-oft` already live here. The new OpenVLA attack code can extend from these.

6. **Model cache:** HuggingFace cache already at `/mnt/sdc/dty_user/cache/huggingface/` — set `HF_HOME` or symlink to reuse.

### Recommended Actions Before Migration

1. Kill GPU 5 stuck process → free 80 GiB VRAM
2. Ask admin to clean root partition
3. Confirm GPU 4 and 6 are consistently available
4. Test the `pi0` conda env: `source activate /llm_jzm/mt/conda_envs/pi0 && python -c "import torch; print(torch.cuda.is_available())"`

---

*Report generated via automated SSH inspection. Share with GPT for review before executing migration.*
