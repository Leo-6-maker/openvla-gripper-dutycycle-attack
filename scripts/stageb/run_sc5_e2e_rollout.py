#!/usr/bin/env python3
"""SC5 end-to-end rollout: MLP online trigger → VIS attack → task failure.

Reuses EXACTLY: run_v2_vis_sc5_bridge.py model/env/attacker loading.
Adds: SC5MLP online trigger state machine (no fixed anchor, no Teacher input).
"""
import argparse, hashlib, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
MLP_PATH = '/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618/outputs/sc5_canonical_eng/sc5_mlp_s2.pt'
K = 10; GUARD = 5; TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3
EPSILON = 6.0 / 255.0; PGD_STEPS = 20; TARGET_TOKEN = 31744; ARM_GATE = 5
SC5_FEATURES = ["gripper_command","gripper_qpos","gripper_opening_proxy","eef_x","eef_y","eef_z",
    "eef_vx","eef_vy","eef_vz","action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed","eef_z_delta_since_close",
    "qpos_delta_1","qpos_delta_3","opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5"]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]

ap = argparse.ArgumentParser()
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--render_gpu', type=int, required=True)
ap.add_argument('--seed_id', type=int, default=1)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--teacher_anchor', type=int, default=-1, help='For audit only, NOT used for trigger')
args = ap.parse_args()

# ── MODEL (identical to VIS bridge) ──
print(f"[{time.strftime('%H:%M:%S')}] Loading OpenVLA...", flush=True)
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto",
    max_memory={idx: "10000MiB" for idx in range(visible)} | {"cpu": "128GiB"}, attn_implementation='eager')
device = "cuda:0"
for v in model.hf_device_map.values():
    if isinstance(v, int): device = f"cuda:{v}"; break

# ── MLP ──
print(f"[{time.strftime('%H:%M:%S')}] Loading MLP...", flush=True)
ckpt = torch.load(MLP_PATH, map_location='cpu', weights_only=False)
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
mlp.load_state_dict({k: v for k, v in ckpt['model_state'].items() if k in mlp.state_dict()}, strict=False)
mlp.eval(); mlp_mean = ckpt['mean']; mlp_std = ckpt['std']

# ── ATTACKER (identical to VIS bridge) ──
from gripper_attack.attack_adapter import OpenVLAVisualAttacker
opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
       "target_token_id": TARGET_TOKEN, "epsilon": EPSILON, "num_steps": PGD_STEPS,
       "step_size": EPSILON * 0.075, "random_start": True, "prefix_refresh_interval": 1,
       "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": ARM_GATE,
       "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
       "target_execution_class": "CLIP_MEDIATED_OPEN"}
attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
    seed=args.seed_id, preprocess_kwargs={"libero_official_preprocess": False,
        "libero_preprocess_backend": "official_pil_lanczos", "center_crop": True, "resize_size": 224})

# ── ENV (identical to VIS bridge) ──
from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path
TASK_IDX = 6
bm = benchmark.get_benchmark_dict(); suite = bm["libero_object"]()
task_obj = suite.get_task(TASK_IDX); init_states = suite.get_task_init_states(TASK_IDX)
bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language
print(f"[{time.strftime('%H:%M:%S')}] Task: {instruction} s{args.state_id}", flush=True)
env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, 10)

# ── Run ──
from scripts.v4_run_eval_openvla import physical_gripper_state
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
streamer = SC5StreamingFeatureAdapterV2()
trigger = {'state': 'IDLE', 'arm_step': -1, 'emit_step': -1, 'triggered': False, 'vis_remaining': 0}
prev_delta = None; attack_log = []
n_steps = 0

for step in range(400):
    img = obs['agentview_image']
    raw_action, policy_step_idx, env_action_np, clean_gen = decode_with_scores(
        model, processor, device, img, instruction, 'libero_object', 8,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)

    # ── MLP ONLINE TRIGGER (no Teacher, no fixed anchor) ──
    if not trigger['triggered']:
        q7_q8 = physical_gripper_state(env)
        gripper_width = abs(q7_q8['qpos'][0]) + abs(q7_q8['qpos'][1]) if len(q7_q8.get('qpos',[])) >= 2 else 0.0
        eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
        raw_grip = float(raw_action[-1]); env_grip = float(env_action_np[-1])
        try:
            result = streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=0.05, gripper_opening_proxy=gripper_width,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
                action_dx=raw_action[0], action_dy=raw_action[1], action_dz=raw_action[2],
                action_gripper=raw_grip)
        except: result = {'valid': False}

        if result.get('valid'):
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
                    attack_log.append({'emit_step': step, 'corridor_p': round(cp,4),
                        'release_p': round(rp,4), 'pred_phase': pred_phase})

    # VIS attack
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(raw_action, dtype=np.float32), enabled=True)
    if trigger['vis_remaining'] > 0:
        env_action_final[-1] = -1.0
        trigger['vis_remaining'] -= 1

    obs, reward, done, info = env.step(env_action_final)
    n_steps += 1
    if done: break

env.close()
success = bool(info.get('success', False)) if done else False

# ── Results ──
result = {
    'state_id': args.state_id, 'instruction': instruction,
    'n_steps': n_steps, 'task_success': success,
    'teacher_anchor': args.teacher_anchor,
    'model_emit_step': trigger['emit_step'],
    'model_triggered': trigger['triggered'],
    'arm_step': trigger['arm_step'],
    'anchor_error': (trigger['emit_step'] - args.teacher_anchor) if (trigger['triggered'] and args.teacher_anchor >= 0) else None,
    'vis_frames': K if trigger['triggered'] else 0,
    'privileged_input_used': False,
    'manual_anchor_used': False,
    'attack_log': attack_log,
}
os.makedirs(args.output_dir, exist_ok=True)
fname = f"e2e_butter_s{args.state_id}_seed{args.seed_id}.json"
with open(os.path.join(args.output_dir, fname), 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"  success={success} emit={trigger['emit_step']} anchor_err={result['anchor_error']} -> {fname}", flush=True)
