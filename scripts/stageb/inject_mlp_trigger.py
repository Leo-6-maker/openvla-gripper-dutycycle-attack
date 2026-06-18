#!/usr/bin/env python3
"""Inject MLP online trigger into run_v2_vis_sc5_bridge.py — minimal modification."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else 'scripts/stageb/run_sc5_mlp_bridge.py'

with open(path) as f:
    lines = f.readlines()

# Find apply_dummy_wait line for MLP code insertion
insert_after = None
for i, line in enumerate(lines):
    if 'apply_dummy_wait(env, obs, 10)' in line:
        insert_after = i; break

mlp_block = '''# ── MLP ONLINE TRIGGER (replaces fixed anchor) ──
SC5_FEATURES = ["gripper_command","gripper_qpos","gripper_opening_proxy","eef_x","eef_y","eef_z",
    "eef_vx","eef_vy","eef_vz","action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed","eef_z_delta_since_close",
    "qpos_delta_1","qpos_delta_3","opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5"]
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]
MLP_PATH = "outputs/sc5_canonical_eng/sc5_mlp_s2.pt"
mlp_ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=False)
class SC5MLPTrig(torch.nn.Module):
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
_sc5mlp = SC5MLPTrig(n_feat=len(mlp_ckpt["feature_names"]))
_sc5mlp.load_state_dict({k: v for k, v in mlp_ckpt["model_state"].items() if k in _sc5mlp.state_dict()}, strict=False)
_sc5mlp.eval(); _mlp_mean = mlp_ckpt["mean"]; _mlp_std = mlp_ckpt["std"]
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
_streamer = SC5StreamingFeatureAdapterV2()
_mlp_trigger = {"s": "IDLE", "arm": -1, "emit": -1, "done": False}
K_MLP = 10; GUARD_MLP = 5; TAU_C = 0.3; TAU_R = 0.3
print("[%s] MLP trigger loaded" % time.strftime("%H:%M:%S")); sys.stdout.flush()
'''

if insert_after is not None:
    for ml in reversed(mlp_block.split('\n')):
        lines.insert(insert_after + 1, ml + '\n')

# Replace fixed-step trigger with MLP trigger
start = None; end = None
for i, line in enumerate(lines):
    if 'FIXED-STEP TRIGGER' in line:
        start = i
    if start and 'telemetry.append' in line and i > start:
        end = i; break

new_trigger = '''    # === MLP ONLINE TRIGGER (replaces fixed-step anchor) ===
    attack_this = False; adv_token = None; adv_arm = 0; prev_flag = False
    if not _mlp_trigger["done"]:
        raw_grip_f = float(action[-1]); env_grip_f = -1.0 if raw_grip_f > 0.5 else 1.0
        try:
            _res = _streamer.update(step_id=step, raw_gripper=raw_grip_f, env_gripper=env_grip_f,
                gripper_qpos=0.05, gripper_opening_proxy=0.02,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
                action_dx=float(action[0]), action_dy=float(action[1]), action_dz=float(action[2]),
                action_gripper=raw_grip_f)
        except: _res = {"valid": False}
        if _res.get("valid"):
            _feats = _res["features"]
            _X = np.array([[_feats[fn] for fn in SC5_FEATURES]], dtype=np.float32)
            _X = (_X - _mlp_mean) / (_mlp_std + 1e-8)
            with torch.no_grad():
                _out = _sc5mlp(torch.tensor(_X, dtype=torch.float32))
            _cp = torch.sigmoid(_out["corridor_logit"]).item()
            _rp = torch.sigmoid(_out["release_logit"]).item()
            _pp = SC5_PHASES[torch.softmax(_out["phase_logits"], dim=1)[0].argmax().item()]
            if _mlp_trigger["s"] == "IDLE":
                if _pp == "stable_carry" and _cp > TAU_C:
                    _mlp_trigger["s"] = "ARMED"; _mlp_trigger["arm"] = step
            elif _mlp_trigger["s"] == "ARMED":
                if step >= _mlp_trigger["arm"] + GUARD_MLP and _cp > TAU_C:
                    _mlp_trigger["s"] = "TRIGGERED"; _mlp_trigger["done"] = True
                    _mlp_trigger["emit"] = step
    if IS_ATTACK and _mlp_trigger["done"] and attack_count < ATTACK_FRAMES:
        attack_this = True
        attack_count += 1
'''

if start and end:
    lines[start:end] = [new_trigger + '\n']

# Add MLP emit info to the final print line instead of modifying summary dict
for i, line in enumerate(lines):
    if 'print(f"  {args.condition}' in line or 'print(f"CLEAN' in line:
        lines[i] = line.replace('print(f"', 'print(f"emit={_mlp_trigger.get(\"emit\",-1)} arm={_mlp_trigger.get(\"arm\",-1)} trig={_mlp_trigger.get(\"done\",False)} ')
        break

with open(path, 'w') as f:
    f.writelines(lines)

print(f"Modified {path}")
print(f"  MLP block inserted after line {insert_after}")
print(f"  Trigger replaced at lines {start}-{end}")
