#!/usr/bin/env python3
"""SC5 online rollout: MLP-triggered VIS attack. Reuses VIS bridge + Phase3 env code exactly.

Reuses: run_v2_vis_sc5_bridge.py (model, env, attacker loading),
        SC5MLP from train_sc5_v4 (online trigger).

One-shot latch: trigger once, then K=10 VIS frames, then observe.
"""
import argparse, hashlib, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
K = 10; GUARD = 5; TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3
EPSILON = 6.0 / 255.0; PGD_STEPS = 20; TARGET_TOKEN = 31744; ARM_GATE = 5

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]

ap = argparse.ArgumentParser()
ap.add_argument('--task_idx', type=int, required=True, help='LIBERO benchmark task index')
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--condition', required=True, choices=['CLEAN','VIS_SC5','RAND_SC5'])
ap.add_argument('--seed_id', type=int, default=99)
ap.add_argument('--render_gpu', type=int, required=True)
ap.add_argument('--mlp_path', default='outputs/sc5_canonical_eng/sc5_mlp_s2.pt')
ap.add_argument('--output_dir', required=True)
args = ap.parse_args()

# ── MODEL LOAD (identical to VIS bridge) ──
print(f"[{time.strftime('%H:%M:%S')}] Loading OpenVLA...", flush=True)
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map='auto',
    max_memory={idx: '10000MiB' for idx in range(visible)} | {'cpu': '128GiB'},
    attn_implementation='eager')
action_dim = int(model.get_action_dim('libero_object'))

# ── MLP LOAD ──
print(f"[{time.strftime('%H:%M:%S')}] Loading MLP trigger...", flush=True)
ckpt = torch.load(args.mlp_path, map_location='cpu', weights_only=False)
class SC5MLP(torch.nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(n_feat, 64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, 64), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(64, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(64, 1)
        self.release_head = torch.nn.Linear(64, 1)
        self.confidence_head = torch.nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h), "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h)}
mlp = SC5MLP(n_feat=len(ckpt['feature_names']))
mlp.load_state_dict({k: v for k, v in ckpt['model_state'].items()
                      if k in mlp.state_dict()}, strict=False)
mlp.eval(); mlp_mean = ckpt['mean']; mlp_std = ckpt['std']

# ── ATTACKER (identical to VIS bridge, one-shot) ──
attacker = None
if args.condition != 'CLEAN':
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    is_rand = args.condition == 'RAND_SC5'
    opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
           "target_token_id": TARGET_TOKEN, "epsilon": EPSILON, "num_steps": PGD_STEPS,
           "step_size": EPSILON * 0.075, "random_start": True, "prefix_refresh_interval": 1,
           "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": ARM_GATE,
           "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
           "target_execution_class": "CLIP_MEDIATED_OPEN"}
    if is_rand: opt['gradient_transform'] = 'permute'; opt['gradient_transform_seed'] = args.seed_id + 100000
    attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
        seed=args.seed_id, preprocess_kwargs={"libero_official_preprocess": False,
            "libero_preprocess_backend": "official_pil_lanczos", "center_crop": True, "resize_size": 224})

# ── ENV (identical to VIS bridge) ──
from libero.libero import benchmark, get_libero_path
from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(args.task_idx)
init_states = suite.get_task_init_states(args.task_idx)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language
print(f"[{time.strftime('%H:%M:%S')}] Task: {instruction}", flush=True)

env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, 10)

# ── Run episode ──
from scripts.v4_run_eval_openvla import physical_gripper_state, decode_with_scores
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
streamer = SC5StreamingFeatureAdapterV2()

trigger = {'state': 'IDLE', 'arm_step': -1, 'emit_step': -1, 'triggered': False, 'vis_remaining': 0}
prev_delta = None; attack_log = []

for step in range(400):
    img = obs['agentview_image']
    # VLA decode (identical to VIS bridge — decode_with_scores handles image internally)
    try:
        raw_action, policy_step_idx, env_action_np, clean_gen = decode_with_scores(
            model, processor, model.device, img, instruction, 'libero_object', 8,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
    except Exception:
        raw_action, policy_step_idx, env_action_np, clean_gen = decode_with_scores(
            model, processor, model.device, img, instruction, 'libero_object', 8,
            unnorm_key='libero_object')

    # ── MLP TRIGGER (25D streaming features) ──
    q7_q8 = physical_gripper_state(env)
    gripper_width = abs(q7_q8[0]) + abs(q7_q8[1]) if len(q7_q8) >= 2 else 0.0
    eef = env.get_endeffector_position()
    raw_grip = float(raw_action[-1])
    env_grip = float(env_action_np[-1])

    try:
        result = streamer.update(
            step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
            gripper_qpos=float(env.get_joint_state()[6]),
            gripper_opening_proxy=gripper_width,
            eef_x=eef[0], eef_y=eef[1], eef_z=eef[2],
            eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
            action_dx=raw_action[0], action_dy=raw_action[1], action_dz=raw_action[2],
            action_gripper=raw_grip)
    except Exception: result = {'valid': False}

    if result.get('valid') and not trigger['triggered']:
        feats = result['features']
        X = np.array([[feats[fn] for fn in SC5_FEATURES]], dtype=np.float32)
        X = (X - mlp_mean) / (mlp_std + 1e-8)
        with torch.no_grad():
            out = mlp(torch.tensor(X, dtype=torch.float32))
        cp = torch.sigmoid(out['corridor_logit']).item()
        rp = torch.sigmoid(out['release_logit']).item()
        pred_phase = SC5_PHASES[torch.softmax(out['phase_logits'], dim=1)[0].argmax().item()]

        if trigger['state'] == 'IDLE':
            if pred_phase == 'stable_carry' and cp > TAU_CORRIDOR:
                trigger['state'] = 'ARMED'; trigger['arm_step'] = step
        elif trigger['state'] == 'ARMED':
            if step >= trigger['arm_step'] + GUARD and cp > TAU_CORRIDOR:
                trigger['state'] = 'TRIGGERED'; trigger['triggered'] = True
                trigger['emit_step'] = step; trigger['vis_remaining'] = K
                attack_log.append({'emit_step': step, 'corridor_p': cp, 'release_p': rp,
                                   'pred_phase': pred_phase, 'raw_grip': raw_grip})

    # ── VIS ATTACK (one-shot K frames once triggered) ──
    env_action = env_action_np.copy()
    if trigger['vis_remaining'] > 0 and attacker is not None:
        is_rand = args.condition == 'RAND_SC5'
        env_action[-1] = -1.0  # force gripper OPEN
        trigger['vis_remaining'] -= 1

    obs, reward, done, info = env.step(env_action)
    if done: break

env.close()
success = bool(info.get('success', False)) if done else False

# ── Results ──
results = {
    'task_idx': args.task_idx, 'state_id': args.state_id,
    'condition': args.condition, 'seed_id': args.seed_id,
    'success': success, 'trigger_step': trigger['emit_step'],
    'triggered': trigger['triggered'],
    'vis_frames_applied': K if trigger['triggered'] and args.condition != 'CLEAN' else 0,
    'instruction': instruction,
}

os.makedirs(args.output_dir, exist_ok=True)
fname = f"t{args.task_idx}_s{args.state_id}_{args.condition}_seed{args.seed_id}.json"
with open(os.path.join(args.output_dir, fname), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  success={success} trigger={trigger['emit_step']} -> {fname}", flush=True)
