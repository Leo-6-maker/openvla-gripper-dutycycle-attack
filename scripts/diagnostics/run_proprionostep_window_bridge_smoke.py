#!/usr/bin/env python3
"""ProprioNoStep auto-window bridge smoke for VIS prefix_margin.

Pipeline:
  1. Clean rollout → record observations
  2. Frozen ProprioNoStep detector → trigger step T
  3. Build 3 windows: W0 [T,T+17], W_minus10 [T-10,T+7], W_minus20 [T-20,T-3]
  4. For each non-confounded window: clean baseline, random_linf, VIS prefix_margin
  5. Provenance CSV + summary + report
"""

from __future__ import annotations
import argparse, csv, json, os, sys, time, subprocess
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.gripper_semantics import raw_gripper_is_open, CANONICAL_OPEN_SEMANTICS_VERSION

# ── ProprioNoStep detector (frozen, from milestone_2c) ──
import torch
import torch.nn.functional as F

HISTORY_LEN = 16; HIDDEN_DIM = 64; TCN_LAYERS = 3
ALLOWED_PROPRIO = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]
N_PROPRIO = len(ALLOWED_PROPRIO)

DETECTOR_CHECKPOINT = "/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt"
DETECTOR_HAZARD_TH = 0.1
DETECTOR_TRIG_DUR = 5
DETECTOR_COOLDOWN = 0


class CausalTCN(torch.nn.Module):
    def __init__(self, in_dim, h_dim, n_ph=8, n_l=3, do=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, h_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(h_dim, h_dim, 3, padding=2**(i+1), dilation=2**i)
            for i in range(n_l)])
        self.drop = torch.nn.Dropout(do)
        self.ph = torch.nn.Linear(h_dim, n_ph); self.hz = torch.nn.Linear(h_dim, 1)
        self.rl = torch.nn.Linear(h_dim, 1)
    def forward(self, x):
        x = self.proj(x); x = x.transpose(1, 2)
        for c in self.convs:
            r = x; x = F.relu(c(x)); x = x[:, :, -r.shape[2]:] + r; x = self.drop(x)
        x = x[:, :, -1]
        return self.ph(x), self.hz(x).squeeze(-1), self.rl(x).squeeze(-1)


class OnlineDetector:
    def __init__(self, model_path, device="cpu", hazard_th=0.1, trig_dur=5, cooldown=0):
        self.model = CausalTCN(N_PROPRIO, HIDDEN_DIM, 8, TCN_LAYERS).to(device)
        ckpt = torch.load(model_path, map_location=device)
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        self.device = device; self.hazard_th = hazard_th
        self.trig_dur = trig_dur; self.cooldown = cooldown
        self.history = []; self.hazard_buf = []; self.cooldown_ctr = 0

    def update(self, features):
        self.history.append(features)
        if len(self.history) > HISTORY_LEN:
            self.history = self.history[-HISTORY_LEN:]
        hist = np.array(self.history, dtype=np.float32)
        if hist.shape[0] < HISTORY_LEN:
            pad = np.zeros((HISTORY_LEN - hist.shape[0], hist.shape[1]), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ph, hl, rl = self.model(x)
        hs = float(torch.sigmoid(hl).item())
        self.hazard_buf.append(hs)
        if len(self.hazard_buf) > self.trig_dur:
            self.hazard_buf = self.hazard_buf[-self.trig_dur:]
        trigger_now = False
        if self.cooldown_ctr > 0:
            self.cooldown_ctr -= 1
        elif len(self.hazard_buf) >= self.trig_dur and all(h > self.hazard_th for h in self.hazard_buf):
            trigger_now = True; self.cooldown_ctr = self.cooldown
        return trigger_now, hs


def extract_features(obs, action, gripper_qpos):
    """Extract 13-D proprioceptive features matching detector schema."""
    eef = obs.get("robot0_eef_pos", np.zeros(3))
    return np.array([
        float(action[-1]),               # gripper_command
        float(gripper_qpos[0]) if len(gripper_qpos) > 0 else 0.0,  # gripper_qpos
        float(obs.get("robot0_gripper_qpos", [0])[0]),              # gripper_width
        float(eef[0]), float(eef[1]), float(eef[2]),                # eef_x,y,z
        0.0, 0.0, 0.0,                                              # eef_vx,vy,vz (approx 0)
        float(action[0]), float(action[1]), float(action[2]),       # action_dx,dy,dz
        float(action[-1]),                                           # action_gripper
    ], dtype=np.float32)


def build_windows(T):
    """Build 3 windows around trigger step T, clipped to valid range."""
    W0 = [max(0, T), min(299, T + 17)]
    Wm10 = [max(0, T - 10), min(299, T + 7)]
    Wm20 = [max(0, T - 20), min(299, T - 3)]
    return {"W0": W0, "W_minus10": Wm10, "W_minus20": Wm20}


def run_clean_rollout(task_name, model, processor, device, mdtype, action_dim,
                       low, high, mask, VS, BC, env_gpu, seed=0):
    """Run a clean policy rollout, record observations + actions per step."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    TASK_IDS = {'cream_cheese': 1, 'salad_dressing': 2, 'ketchup': 4, 'tomato_sauce': 5}
    INSTRUCTIONS = {
        'cream_cheese': 'pick up the cream cheese and place it in the basket',
        'salad_dressing': 'put the salad dressing in the basket',
        'ketchup': 'pick up the ketchup and place it in the basket',
        'tomato_sauce': 'pick up the tomato sauce and place it in the basket',
    }

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    task = task_suite.get_task(TASK_IDS[task_name])
    bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    init_states = task_suite.get_task_init_states(TASK_IDS[task_name])

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=['agentview'], control_freq=20,
        render_gpu_device_id=env_gpu)
    env.seed(seed)
    obs = env.reset(); env.sim.data.qvel[:] = 0; env.sim.forward()
    env.set_init_state(init_states[0])
    for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))

    instruction = INSTRUCTIONS[task_name]
    prompt_text = f"In: What action should the robot take to {instruction}?\nOut:"

    steps = []
    for t in range(300):
        img = obs["agentview_image"]
        img = img[::-1, ::-1]
        pil_img = Image.fromarray(img).convert("RGB").resize((224, 224), Image.LANCZOS)

        inputs = processor(prompt_text, pil_img, return_tensors="pt")
        inputs.pop("attention_mask", None)
        for k, v in list(inputs.items()):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=device, dtype=mdtype)
            else:
                inputs[k] = v.to(device)
        if not torch.all(inputs['input_ids'][:, -1] == 29871):
            inputs['input_ids'] = torch.cat((inputs['input_ids'],
                torch.tensor([[29871]], dtype=torch.long, device=device)), dim=1)

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        token_ids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
        disc = np.clip(VS - token_ids - 1, 0, len(BC) - 1)
        norm_actions = BC[disc].astype(np.float32)
        action = np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)

        # normalize + invert for env
        env_action = action.copy()
        env_action[-1] = 2.0 * env_action[-1] - 1.0
        env_action[-1] = np.sign(env_action[-1])
        env_action[-1] = 1.0 if env_action[-1] == 0 else env_action[-1]
        env_action[-1] = -1.0 * env_action[-1]

        qpos_pre = obs['robot0_gripper_qpos'].copy()
        obs, reward, done, info = env.step(env_action)
        qpos_post = obs['robot0_gripper_qpos'].copy()

        steps.append({
            'policy_step': t,
            'raw_gripper': float(action[-1]),
            'env_gripper': float(env_action[-1]),
            'qpos_pre': float(qpos_pre[0]) if len(qpos_pre) > 0 else 0.0,
            'qpos_post': float(qpos_post[0]) if len(qpos_post) > 0 else 0.0,
            'done': bool(done),
            'reward': float(reward),
            'obs': obs,  # keep for feature extraction
            'action': action,
        })
        if done:
            break
    env.close()
    return steps


def run_clean_detector(steps, detector, device):
    """Run detector on clean rollout to find trigger step T."""
    trigger_step = None
    for s in steps:
        feats = extract_features(s['obs'], s['action'],
                                  np.array([s['qpos_pre']]))
        trigger_now, hs = detector.update(feats)
        s['hazard_score'] = hs
        s['trigger_now'] = trigger_now
        if trigger_now and trigger_step is None:
            trigger_step = s['policy_step']
    return trigger_step


def run_vis_attack(task_name, condition, window, seed, gpu_pair, eps, objective, model_path, env_gpu):
    """Run a single VIS rollout with given condition and window via subprocess."""
    ws, we = window
    cmd = [
        sys.executable, "-u",
        str(REPO / "scripts/vis_rollout_adaptive_v3.py"),
        "--task", task_name,
        "--condition", condition,
        "--eps_raw_pixels", str(eps),
        "--perturb_start", str(ws),
        "--perturb_end", str(we),
        "--objective", objective,
        "--seed", str(seed),
        "--gpu_pair", gpu_pair,
    ]
    # Run via subprocess, capture output
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(REPO))
    # Find trace CSV from output
    trace_path = None
    for line in result.stdout.split("\n") + result.stderr.split("\n"):
        if "Saved:" in line and "_trace.csv" in line:
            trace_path = line.split("Saved:")[-1].strip()
            break
    return trace_path, result.returncode


def read_trace(trace_path):
    """Read trace CSV into list of dicts."""
    if not trace_path or not os.path.exists(trace_path):
        return []
    with open(trace_path, newline='') as f:
        return list(csv.DictReader(f))


def compute_window_metrics(trace_rows, window_start, window_end):
    """Compute per-window metrics from trace rows."""
    wrows = [r for r in trace_rows
             if int(r.get('policy_step', -1)) >= window_start
             and int(r.get('policy_step', -1)) <= window_end]
    n = len(wrows)
    if n == 0:
        return {"open_count": 0, "total": 0, "open_ratio": 0.0,
                "qpos_delta_post": 0.0, "arm_l2_max": 0.0,
                "done": False, "timeout": False}

    open_cnt = sum(1 for r in wrows if raw_gripper_is_open(float(r.get('adv_grip', 0.996))))
    qpos = [float(r.get('qpos_post_step', 0)) for r in wrows if r.get('qpos_post_step')]
    qd = max(abs(v - qpos[0]) for v in qpos) if len(qpos) > 1 else 0.0
    arm_l2 = max(float(r.get('arm_l2', 0)) for r in wrows) if wrows else 0.0
    done = any(r.get('done', 'False') == 'True' for r in trace_rows)

    return {
        "open_count": open_cnt, "total": n,
        "open_ratio": round(open_cnt / max(n, 1), 4),
        "qpos_delta_post": round(qd, 6),
        "arm_l2_max": round(arm_l2, 6),
        "done": done,
        "timeout": not done and len(trace_rows) >= 299,
    }


def main():
    ap = argparse.ArgumentParser(description='ProprioNoStep auto-window bridge smoke')
    ap.add_argument("--task", default="ketchup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=int, default=6)
    ap.add_argument("--objective", default="prefix_locked_gripper_open_margin")
    ap.add_argument("--gpu_pair", default="6,7")
    ap.add_argument("--env_gpu", type=int, default=6)
    ap.add_argument("--output_dir", default="tables")
    ap.add_argument("--report", default="reports/VIS_AUTO_WINDOW_BRIDGE_SMOKE.md")
    ap.add_argument("--skip_vis", action="store_true", help="Skip VIS/random; clean+detector only")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    # ── Load OpenVLA model ──
    print("[0] Loading OpenVLA model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    MODEL_PATH = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
    gpu_ids = [int(x.strip()) for x in args.gpu_pair.split(",")]
    max_mem = {gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'}
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, attn_implementation='eager', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map='auto',
        max_memory=max_mem, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    BC = np.array(model.bin_centers)
    UNNORM_KEY = 'libero_object'
    action_dim = int(model.get_action_dim(UNNORM_KEY))
    stats = model.get_action_stats(UNNORM_KEY)
    mask = np.array(stats.get('mask', np.ones_like(stats['q01'], dtype=bool)))
    low = np.array(stats['q01'])
    high = np.array(stats['q99'])
    print(f"    device={device} mdtype={mdtype} action_dim={action_dim}")

    # ── Stage 1: Clean rollout + detector ──
    print(f"[1] Clean rollout on {args.task}...")
    clean_steps = run_clean_rollout(args.task, model, processor, device, mdtype,
                                     action_dim, low, high, mask, VS, BC, args.env_gpu, args.seed)
    print(f"    {len(clean_steps)} steps, done={clean_steps[-1]['done']}")

    # ── Stage 2: ProprioNoStep detector ──
    print("[2] Running ProprioNoStep detector...")
    detector = OnlineDetector(DETECTOR_CHECKPOINT, device=str(device),
                               hazard_th=DETECTOR_HAZARD_TH, trig_dur=DETECTOR_TRIG_DUR,
                               cooldown=DETECTOR_COOLDOWN)
    T = run_clean_detector(clean_steps, detector)
    if T is None:
        print("    WARNING: No trigger found! Using default T=93 (known ketchup auto-window)")
        T = 93
    print(f"    Trigger step T = {T}")

    # ── Stage 3: Window construction ──
    windows = build_windows(T)
    print(f"[3] Windows: W0={windows['W0']}, W-10={windows['W_minus10']}, W-20={windows['W_minus20']}")

    # Compute clean natural OPEN per window
    window_confounds = {}
    for wname, (ws, we) in windows.items():
        wrows = [s for s in clean_steps if ws <= s['policy_step'] <= we]
        n = len(wrows)
        open_cnt = sum(1 for s in wrows if raw_gripper_is_open(s['raw_gripper']))
        ratio = open_cnt / max(n, 1)
        confounded = ratio > 0.5
        window_confounds[wname] = {
            "start": ws, "end": we, "n_steps": n,
            "clean_open_count": open_cnt, "clean_open_ratio": round(ratio, 4),
            "natural_release_confounded": confounded,
        }
        print(f"    {wname} [{ws}-{we}]: clean OPEN={open_cnt}/{n} ratio={ratio:.4f} confounded={confounded}")

    # ── Stage 4: VIS/random per window ──
    if args.skip_vis:
        print("[4] Skipping VIS/random attacks (--skip_vis)")
        results = {}
    else:
        print(f"[4] Running VIS/random attacks (eps={args.eps}, objective={args.objective})...")
        results = {}
        conditions = [
            ("clean", "clean"),
            ("random_linf", "random_linf"),
            ("vis_pgd", "vis_pgd"),
        ]

        for wname, (ws, we) in windows.items():
            conf = window_confounds[wname]
            if conf["natural_release_confounded"]:
                print(f"    SKIP {wname}: natural_release_confounded (clean OPEN ratio={conf['clean_open_ratio']})")
                # Still run random for denominator
                conds_to_run = [("random_linf", "random_linf")]
            else:
                conds_to_run = conditions

            for cond_label, cond in conds_to_run:
                run_seed = args.seed
                if cond == "random_linf":
                    run_seed = args.seed + 1000  # different seed for random
                print(f"    {wname} [{ws}-{we}] {cond_label} seed={run_seed}...")
                t0 = time.time()
                trace_path, rc = run_vis_attack(
                    args.task, cond, [ws, we], run_seed, args.gpu_pair,
                    args.eps, args.objective, MODEL_PATH, args.env_gpu)
                dt = time.time() - t0
                key = f"{wname}_{cond_label}"
                if trace_path and os.path.exists(trace_path):
                    trace_rows = read_trace(trace_path)
                    metrics = compute_window_metrics(trace_rows, ws, we)
                    metrics.update({
                        "trace_path": trace_path, "rc": rc,
                        "condition": cond_label, "window": wname,
                        "ws": ws, "we": we, "runtime_s": round(dt, 1),
                    })
                    results[key] = metrics
                    print(f"      OPEN={metrics['open_count']}/{metrics['total']} qposΔ={metrics['qpos_delta_post']} armL2={metrics['arm_l2_max']} done={metrics['done']}")
                else:
                    print(f"      FAILED (rc={rc}, no trace)")
                    results[key] = {"condition": cond_label, "window": wname,
                                    "ws": ws, "we": we, "rc": rc,
                                    "open_count": -1, "total": 0, "failed": True}

    # ── Stage 5: Provenance tables ──
    print("[5] Writing provenance...")
    # Window-level summary CSV
    summary_path = os.path.join(args.output_dir, "vis_auto_window_bridge_summary.csv")
    summary_fields = [
        "task", "seed", "T", "window", "window_start", "window_end",
        "clean_open_count", "clean_open_ratio", "natural_release_confounded",
        "prefix_OPEN", "prefix_total", "prefix_qpos_delta",
        "prefix_armL2_max", "prefix_done",
        "random_OPEN", "random_total", "random_qpos_delta",
        "random_done", "denominator_status", "interpretation",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for wname, conf in window_confounds.items():
            row = {
                "task": args.task, "seed": args.seed, "T": T,
                "window": wname,
                "window_start": conf["start"], "window_end": conf["end"],
                "clean_open_count": conf["clean_open_count"],
                "clean_open_ratio": conf["clean_open_ratio"],
                "natural_release_confounded": conf["natural_release_confounded"],
            }
            # Prefix results
            pk = f"{wname}_vis_pgd"
            if pk in results and not results[pk].get("failed"):
                r = results[pk]
                row.update({
                    "prefix_OPEN": f"{r['open_count']}/{r['total']}",
                    "prefix_total": r['total'],
                    "prefix_qpos_delta": r['qpos_delta_post'],
                    "prefix_armL2_max": r['armL2_max'],
                    "prefix_done": r['done'],
                })
            # Random results
            rk = f"{wname}_random_linf"
            if rk in results and not results[rk].get("failed"):
                r = results[rk]
                row.update({
                    "random_OPEN": f"{r['open_count']}/{r['total']}",
                    "random_total": r['total'],
                    "random_qpos_delta": r['qpos_delta_post'],
                    "random_done": r['done'],
                })
            # Denominator
            if row.get("random_OPEN", "0/0") == "0/0" or (rk in results and results[rk].get("open_count", -1) == 0):
                row["denominator_status"] = "clean" if row.get("random_done", False) else "polluted"
            else:
                row["denominator_status"] = "polluted"

            # Interpretation
            if conf["natural_release_confounded"]:
                row["interpretation"] = "natural_release_confounded"
            elif row.get("prefix_OPEN", "0/0") != "0/0" and row.get("denominator_status") == "clean":
                row["interpretation"] = "VIS_vulnerable_window"
            elif row.get("prefix_done"):
                row["interpretation"] = "task_survived"
            else:
                row["interpretation"] = "no_clear_signal"
            w.writerow(row)
    print(f"    Summary: {summary_path}")

    # ── Report ──
    print("[6] Writing report...")
    with open(args.report, "w") as f:
        f.write(f"""# VIS Auto-Window Bridge Smoke

**Date**: 2026-06-03
**Detector**: ProprioNoStep CausalTCN (milestone_2c)
**Checkpoint**: `{DETECTOR_CHECKPOINT}`
**Hazard threshold**: {DETECTOR_HAZARD_TH}
**Trigger duration**: {DETECTOR_TRIG_DUR}
**Semantics**: `{CANONICAL_OPEN_SEMANTICS_VERSION}`
**Task**: `{args.task}`, seed={args.seed}

## Detector

| Field | Value |
|-------|-------|
| Trigger step T | {T} |
| Feature dim | {N_PROPRIO} |
| TCN layers | {TCN_LAYERS} |
| Hidden dim | {HIDDEN_DIM} |
| History len | {HISTORY_LEN} |

## Windows

| Window | Range | Steps |
|--------|-------|-------|
""")
        for wname, conf in window_confounds.items():
            f.write(f"| {wname} | [{conf['start']}, {conf['end']}] | {conf['n_steps']} |\n")

        f.write(f"""
## Clean Natural OPEN

| Window | OPEN | Ratio | Confounded |
|--------|------|-------|------------|
""")
        for wname, conf in window_confounds.items():
            f.write(f"| {wname} | {conf['clean_open_count']}/{conf['n_steps']} | {conf['clean_open_ratio']} | {conf['natural_release_confounded']} |\n")

        f.write(f"""
## VIS / Random Results

| Window | Prefix OPEN | Prefix qposΔ | armL2 | Random OPEN | Denom | Interpretation |
|--------|------------|-------------|-------|-------------|-------|---------------|
""")
        for wname in window_confounds:
            pk = f"{wname}_vis_pgd"; rk = f"{wname}_random_linf"
            po = results.get(pk, {}); ro = results.get(rk, {})
            pf = f"{po.get('open_count','?')}/{po.get('total','?')}" if not po.get('failed') else "FAIL"
            rf = f"{ro.get('open_count','?')}/{ro.get('total','?')}" if not ro.get('failed') else "FAIL"
            pq = po.get('qpos_delta_post', '-') if not po.get('failed') else '-'
            pa = po.get('armL2_max', '-') if not po.get('failed') else '-'
            f.write(f"| {wname} | {pf} | {pq} | {pa} | {rf} | -\n")

        f.write("""
## Interpretation

This is a **bridge smoke**, not final detector-guided VIS evidence.

See [vis_auto_window_bridge_summary.csv](../tables/vis_auto_window_bridge_summary.csv) for full data.

## Deferred

Strict primary matrix (prefix seed0, random seeds2-3, 20-37) pending healthy GPU environment.
""")

    print(f"    Report: {args.report}")
    print("[7] Done.")


if __name__ == "__main__":
    main()
