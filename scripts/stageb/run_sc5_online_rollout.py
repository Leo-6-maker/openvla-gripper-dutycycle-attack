#!/usr/bin/env python3
"""SC5 online rollout: MLP-triggered VIS attack on LIBERO object tasks.

Reuses: V4 env factory, OpenVLA model loading, frozen attack_adapter, mature SC5MLP.

Measures: task success, trigger coverage, VIS duration, arm delta, condition comparison.
"""
import argparse, csv, json, os, sys, time, hashlib, numpy as np, torch
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
MLP_PATH = 'outputs/sc5_canonical_eng/sc5_mlp_s1.pt'
K = 10; GUARD = 5; TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3
EPSILON = 6.0 / 255.0; PGD_STEPS = 20; TARGET_TOKEN = 31744

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
ap.add_argument('--task', required=True)
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--condition', required=True, choices=['CLEAN','VIS_SC5','RAND_SC5'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--render_gpu', type=int, required=True)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--mlp_path', default=MLP_PATH)
ap.add_argument('--max_steps', type=int, default=300)
args = ap.parse_args()

print(f"[{time.strftime('%H:%M:%S')}] Loading OpenVLA model...", flush=True)
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

# Load MLP
print(f"[{time.strftime('%H:%M:%S')}] Loading MLP trigger...", flush=True)
ckpt = torch.load(args.mlp_path, map_location='cpu', weights_only=False)
class SC5MLPInline(torch.nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(n_feat, 64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, 64), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(64, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(64, 1)
        self.release_head = torch.nn.Linear(64, 1)
    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h), "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h)}
mlp = SC5MLPInline(n_feat=len(ckpt['feature_names']))
mlp.load_state_dict(ckpt['model_state']); mlp.eval()
mlp_mean = ckpt['mean']; mlp_std = ckpt['std']

# Attacker
attacker = None
if args.condition != 'CLEAN':
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    is_rand = args.condition == 'RAND_SC5'
    opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
           "target_token_id": TARGET_TOKEN, "epsilon": EPSILON, "num_steps": PGD_STEPS,
           "step_size": EPSILON * 0.075, "random_start": True, "prefix_refresh_interval": 1,
           "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": 5,
           "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
           "target_execution_class": "CLIP_MEDIATED_OPEN"}
    if is_rand: opt['gradient_transform'] = 'permute'; opt['gradient_transform_seed'] = args.seed + 100000
    attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
        seed=args.seed, preprocess_kwargs={"libero_official_preprocess": False,
            "libero_preprocess_backend": "official_pil_lanczos", "center_crop": True, "resize_size": 224})

# Env
from gripper_attack.libero_v4_env_factory import build_v4_exact_env
from scripts.v4_run_eval_openvla import physical_gripper_state
env = build_v4_exact_env(args.task, args.state_id, args.render_gpu)
env.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
obs = env.reset()

# State
NUM_WAIT = 10
trigger = {'state': 'IDLE', 'arm_step': -1, 'emit_step': -1, 'triggered': False,
           'vis_applied': 0, 'vis_remaining': 0}
results = {'success': False, 'trigger_step': -1, 'vis_steps_applied': 0, 'max_arm_delta': 0.0,
           'condition': args.condition, 'task': args.task, 'state_id': args.state_id}

# Streaming feature adapter for 25D
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
streamer = SC5StreamingFeatureAdapterV2()
prev_clean_action = np.zeros(7)

for step in range(args.max_steps):
    # VLA inference
    img = obs['agentview_image'] if 'agentview_image' in obs else obs.get('image', obs.get('agentview_rgb', None))
    if img is None:
        break
    if trigger['vis_remaining'] > 0 and attacker is not None:
        adv_result = attacker.attack(img, step)
        adv_img = attacker.get_adv_inputs_from_attack_result(adv_result)
    else:
        adv_img = img

    with torch.no_grad():
        inputs = processor(images=adv_img, text=args.task, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    action_tokens = processor.decode(outputs[0], skip_special_tokens=True)
    raw_action = np.array([float(x) for x in action_tokens.strip('[]').split(',')])

    # Postprocess: gripper and arm
    env_action = raw_action.copy()
    env_action[-1] = -1.0 if raw_action[-1] > 0.5 else 1.0
    if trigger['vis_remaining'] > 0:
        env_action[-1] = -1.0  # force OPEN
        trigger['vis_remaining'] -= 1
        results['vis_steps_applied'] += 1
    arm_delta = np.abs(raw_action[:3]).max()
    results['max_arm_delta'] = max(results['max_arm_delta'], arm_delta)
    prev_clean_action = raw_action.copy()

    # MLP trigger: extract 25D proprio features
    if step >= NUM_WAIT:
        q7_q8 = physical_gripper_state(env)
        gripper_width = abs(q7_q8[0]) + abs(q7_q8[1]) if len(q7_q8) >= 2 else 0.0
        eef = env.get_endeffector_position()
        raw_grip = float(raw_action[-1])
        env_grip = float(env_action[-1])

        try:
            feat_result = streamer.update(
                step_id=step - NUM_WAIT, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=float(env.get_joint_state()[6]) if hasattr(env, 'get_joint_state') else 0.05,
                gripper_opening_proxy=gripper_width,
                eef_x=eef[0] if len(eef)>0 else 0.0, eef_y=eef[1] if len(eef)>1 else 0.0,
                eef_z=eef[2] if len(eef)>2 else 0.0,
                eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
                action_dx=raw_action[0], action_dy=raw_action[1], action_dz=raw_action[2],
                action_gripper=raw_grip)
        except Exception:
            feat_result = {'valid': False}

        if feat_result.get('valid'):
            feats = feat_result['features']
            X = np.array([[feats[fn] for fn in SC5_FEATURES]], dtype=np.float32)
            X = (X - mlp_mean) / (mlp_std + 1e-8)
            with torch.no_grad():
                out = mlp(torch.tensor(X, dtype=torch.float32))
            corridor_p = torch.sigmoid(out['corridor_logit']).item()
            release_p = torch.sigmoid(out['release_logit']).item()
            phase_prob = torch.softmax(out['phase_logits'], dim=1)[0]
            pred_phase = SC5_PHASES[phase_prob.argmax().item()]

            # Trigger state machine (one-shot latch)
            if not trigger['triggered']:
                if trigger['state'] == 'IDLE':
                    if pred_phase == 'stable_carry' and corridor_p > TAU_CORRIDOR:
                        trigger['state'] = 'ARMED'; trigger['arm_step'] = step
                elif trigger['state'] == 'ARMED':
                    if step >= trigger['arm_step'] + GUARD and corridor_p > TAU_CORRIDOR:
                        trigger['state'] = 'TRIGGERED'; trigger['triggered'] = True
                        trigger['emit_step'] = step
                        trigger['vis_remaining'] = K
                        results['trigger_step'] = step
                        print(f"  TRIGGER at step {step}", flush=True)

    obs, reward, done, info = env.step(env_action)
    if done:
        results['success'] = info.get('success', info.get('goal_achieved', False))
        break
    if not done and step == args.max_steps - 1:
        results['success'] = False

# Save results
os.makedirs(args.output_dir, exist_ok=True)
fname = f"{args.task}_s{args.state_id}_{args.condition}_seed{args.seed}.json"
with open(os.path.join(args.output_dir, fname), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Result: success={results['success']} trigger_step={results['trigger_step']} "
      f"vis_steps={results['vis_steps_applied']} arm_delta={results['max_arm_delta']:.4f}")
print(f"  Saved: {fname}")
