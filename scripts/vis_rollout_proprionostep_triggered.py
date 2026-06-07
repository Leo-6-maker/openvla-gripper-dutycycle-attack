#!/usr/bin/env python3
"""VIS Rollout with ProprioNoStep online CPU phase detector trigger.
Self-contained. No patch required. No import of vis_rollout_adaptive_v3.

Usage:
  CUDA_VISIBLE_DEVICES=2,6 python -u scripts/vis_rollout_proprionostep_triggered.py \
    --task ketchup --state-id 0 --condition vis_pgd \
    --gpu_pair 0,1 --eps_raw_pixels 6 --pgd_steps 40 --pgd_restarts 1 \
    --objective prefix_locked_gripper_open_margin --seed 0 \
    --use_proprionostep_detector --proprionostep_hazard_threshold 0.1
"""
import argparse, csv, json, os, sys, time
from collections import deque
from datetime import datetime
import numpy as np
import torch

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'src'))
from gripper_attack.gripper_semantics import raw_gripper_is_open, env_gripper_is_open, ENV_GRIPPER_OPEN_VALUE
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'
FEATURE_SCHEMA_VERSION = 'proprionostep_v1_13dim_zero_action_delta_20260602'

_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
_GPU_ISOLATED = bool(_VISIBLE)

def log(msg): print('%s %s' % (datetime.now().strftime('%H:%M:%S'), msg))

# ── CLI ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True); ap.add_argument('--state-id', type=int, required=True)
ap.add_argument('--condition', choices=['clean','vis_pgd','random_linf','oracle_open','sustained_command_open_proxy'], required=True)
ap.add_argument('--gpu_pair', required=True)
ap.add_argument('--eps_raw_pixels', type=float, default=6.0)
ap.add_argument('--pgd_steps', type=int, default=40); ap.add_argument('--pgd_restarts', type=int, default=1)
ap.add_argument('--objective', default='prefix_locked_gripper_open_margin')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_steps', type=int, default=400)
# ProprioNoStep
ap.add_argument('--use_proprionostep_detector', action='store_true')
ap.add_argument('--proprionostep_model_path', default='/data/liuyu/outputs/proprionostep_cpu_20260602/proprio_no_step_tcn_cpu.pt')
ap.add_argument('--proprionostep_hazard_threshold', type=float, default=0.1)
ap.add_argument('--proprionostep_trigger_duration', type=int, default=5)
ap.add_argument('--proprionostep_cooldown', type=int, default=20)
ap.add_argument('--proprionostep_burst_len', type=int, default=18)
ap.add_argument('--proprionostep_history_len', type=int, default=32)
ap.add_argument('--proprionostep_trigger_mode', choices=['hazard','phase','hazard_or_phase'], default='hazard')
ap.add_argument('--proprionostep_trigger_phase', type=int, default=-1)
ap.add_argument('--proprionostep_action_delta_mode', choices=['zero','compute'], default='zero')
# Debug
ap.add_argument('--debug_force_trigger_step', type=int, default=-1)
# Output
ap.add_argument('--output_dir', default='/data/liuyu/outputs/proprionostep_triggered_20260607')
args = ap.parse_args()

if _GPU_ISOLATED and args.gpu_pair != '0,1':
    log('FATAL: CUDA_VISIBLE_DEVICES=%s requires --gpu_pair 0,1' % _VISIBLE); sys.exit(1)

gpu_ids = [int(x) for x in args.gpu_pair.split(',')]
# GPU pair ordering: if physical pair is 0,1, swap to use GPU1 as primary (gpu_ids[0]=1)
# to reduce GPU0 hardware fault exposure
_physical = [int(x.strip()) for x in _VISIBLE.split(',') if x.strip().isdigit()] if _VISIBLE else gpu_ids
if _physical and _physical == [0, 1]:
    gpu_ids = [1, 0]  # GPU1 primary, GPU0 secondary
    _render_gpu = 1
    log('GPU pair 0,1 swapped: primary=GPU1 render=GPU1 secondary=GPU0 (GPU0 fault mitigation)')
else:
    _render_gpu = _physical[0] if _physical else gpu_ids[0]
device_str = 'cuda:%d' % gpu_ids[0]
# pgd_restarts guard
if args.condition == 'vis_pgd' and args.pgd_restarts > 1:
    log('FATAL: pgd_restarts > 1 not yet implemented in this runner'); sys.exit(1)
effective_pgd_restarts = max(args.pgd_restarts, 1)

log('GPU: visible=%s logical=%s render=%d pgd_restarts=%d' % (_VISIBLE, args.gpu_pair, _render_gpu, effective_pgd_restarts))
os.makedirs(args.output_dir, exist_ok=True)

# ── Prompt (original In:/Out: format) ─────────────────────────────
def prompt_fn(instruction):
    return "In: What action should the robot take to %s?\nOut:" % instruction

# ── Action transform ──────────────────────────────────────────────
def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize: action[..., -1] = np.sign(action[..., -1]); action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action

def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy(); action[..., -1] = -1.0 * action[..., -1]
    return action

# ── Load OpenVLA ──────────────────────────────────────────────────
from transformers import AutoModelForVision2Seq, AutoProcessor
log('Loading OpenVLA...')
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '8000MiB', 'cpu': '64GiB'},
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval(); model_device = next(model.parameters()).device; model_dtype = next(model.parameters()).dtype
action_dim = int(model.get_action_dim(UNNORM_KEY))
VS = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
BC = np.asarray(model.bin_centers, dtype=np.float32)
_stats = model.get_action_stats(UNNORM_KEY)
low = np.asarray(_stats['q01'], dtype=np.float32); high = np.asarray(_stats['q99'], dtype=np.float32)
mask_np = np.asarray(_stats.get('mask', np.ones_like(low, dtype=bool)), dtype=bool)
log('OpenVLA ready: action_dim=%d device=%s' % (action_dim, model_device))

# ── ProprioNoStep Detector ────────────────────────────────────────
class ProprioNoStepDetectorV2:
    def __init__(self, ckpt_path, device='cpu', history_len=32, hazard_threshold=0.1,
                 trigger_duration=5, cooldown=20, trigger_mode='hazard', trigger_phase=-1,
                 action_delta_mode='zero'):
        self.device = torch.device(device); self.history_len = history_len
        self.hazard_threshold = hazard_threshold; self.trigger_duration = trigger_duration
        self.cooldown = cooldown; self.trigger_mode = trigger_mode
        self.trigger_phase = trigger_phase; self.action_delta_mode = action_delta_mode
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.proj = torch.nn.Linear(13, 64); self.proj.weight.data=sd['proj.weight']; self.proj.bias.data=sd['proj.bias']
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(64,64,3,padding=1) for _ in range(3)])
        for i in range(3): self.convs[i].weight.data=sd[f'convs.{i}.weight']; self.convs[i].bias.data=sd[f'convs.{i}.bias']
        self.phase_head=torch.nn.Linear(64,8); self.phase_head.weight.data=sd['phase_head.weight']; self.phase_head.bias.data=sd['phase_head.bias']
        self.hazard_head=torch.nn.Linear(64,1); self.hazard_head.weight.data=sd['hazard_head.weight']; self.hazard_head.bias.data=sd['hazard_head.bias']
        self.release_head=torch.nn.Linear(64,1); self.release_head.weight.data=sd['release_head.weight']; self.release_head.bias.data=sd['release_head.bias']
        self.to(self.device); self.eval()
        self.history = deque(maxlen=history_len)
        self.trigger_active = False; self.trigger_counter = 0; self.cooldown_counter = 0

    def to(self, d): self.proj.to(d); [c.to(d) for c in self.convs]; self.phase_head.to(d); self.hazard_head.to(d); self.release_head.to(d)
    def eval(self): self.proj.eval(); [c.eval() for c in self.convs]; self.phase_head.eval(); self.hazard_head.eval(); self.release_head.eval()

    def extract_features(self, obs, raw_action, env):
        gripper_command = float(raw_action[-1]) if len(raw_action)>0 else 0.0
        try: qpos=env.sim.data.qpos; gq=float(np.mean([qpos[-2],qpos[-1]])) if len(qpos)>=2 else 0.0
        except: gq=0.0
        gw=gq
        eef=obs.get('robot0_eef_pos',np.zeros(3)); ex,ey,ez=float(eef[0]),float(eef[1]),float(eef[2])
        ev=obs.get('robot0_eef_vel',np.zeros(3)) if isinstance(obs,dict) else np.zeros(3)
        evx,evy,evz=float(ev[0]),float(ev[1]),float(ev[2])
        adx,ady,adz=0.0,0.0,0.0  # training data had zero action deltas
        ag=float(raw_action[-1])
        return np.array([gripper_command,gq,gw,ex,ey,ez,evx,evy,evz,adx,ady,adz,ag], dtype=np.float32)

    def step(self, obs, raw_action, env):
        feats = self.extract_features(obs, raw_action, env); self.history.append(feats)
        result = {'hazard_score':0.0,'release_safe_score':0.0,'phase_idx':-1,'phase_confidence':0.0,'trigger_now':False,'trigger_reason':''}
        if len(self.history) < 8: return result
        x = torch.as_tensor(np.stack(list(self.history),axis=0),dtype=torch.float32,device=self.device).T.unsqueeze(0)
        x = torch.relu(self.proj(x.transpose(1,2))).transpose(1,2)
        for c in self.convs: x = torch.relu(c(x))
        xp = x.mean(dim=-1)
        with torch.no_grad():
            pl=self.phase_head(xp); pp=torch.softmax(pl,dim=-1); pi=int(pp.argmax(dim=-1).item()); pc=float(pp.max(dim=-1).values.item())
            h=float(torch.sigmoid(self.hazard_head(xp)).item()); r=float(torch.sigmoid(self.release_head(xp)).item())
        result['hazard_score']=round(h,6); result['release_safe_score']=round(r,6); result['phase_idx']=pi; result['phase_confidence']=round(pc,6)
        if self.cooldown_counter>0: self.cooldown_counter-=1; return result
        trig = False
        if self.trigger_mode=='hazard': trig = h >= self.hazard_threshold
        elif self.trigger_mode=='phase': trig = (self.trigger_phase<0 and pi not in (4,5)) or (pi==self.trigger_phase)
        elif self.trigger_mode=='hazard_or_phase': trig = (h>=self.hazard_threshold) or (self.trigger_phase<0 and pi not in (4,5))
        if trig: self.trigger_counter+=1
        else: self.trigger_active=False; self.trigger_counter=0
        if self.trigger_counter >= self.trigger_duration and not self.trigger_active:
            self.trigger_active=True; result['trigger_now']=True
            result['trigger_reason']='%s_h%.3f_p%d_c%d'%(self.trigger_mode,h,pi,self.trigger_counter)
        return result

detector = None
if args.use_proprionostep_detector:
    detector = ProprioNoStepDetectorV2(
        ckpt_path=args.proprionostep_model_path, device='cpu',
        history_len=args.proprionostep_history_len, hazard_threshold=args.proprionostep_hazard_threshold,
        trigger_duration=args.proprionostep_trigger_duration, cooldown=args.proprionostep_cooldown,
        trigger_mode=args.proprionostep_trigger_mode, trigger_phase=args.proprionostep_trigger_phase,
        action_delta_mode=args.proprionostep_action_delta_mode)
    log('ProprioNoStep: mode=%s thr=%.3f dur=%d cool=%d burst=%d delta=%s' % (
        args.proprionostep_trigger_mode, args.proprionostep_hazard_threshold,
        args.proprionostep_trigger_duration, args.proprionostep_cooldown,
        args.proprionostep_burst_len, args.proprionostep_action_delta_mode))

# ── Helpers ───────────────────────────────────────────────────────
from PIL import Image
def decode_action(tids):
    t = tids.detach().cpu().numpy() if torch.is_tensor(tids) else np.asarray(tids)
    d = np.clip(VS-t-1,0,len(BC)-1); return np.where(mask_np,0.5*(BC[d].astype(np.float32)+1)*(high-low)+low,BC[d].astype(np.float32)).astype(np.float32)

def clean_decode(img, instruction):
    pil = Image.fromarray(img.astype(np.uint8))
    text = prompt_fn(instruction.lower())
    inputs = processor(text, pil, return_tensors='pt')
    for k,v in list(inputs.items()):
        if torch.is_floating_point(v): inputs[k]=v.to(device=model_device,dtype=model_dtype)
        else: inputs[k]=v.to(model_device)
    if not torch.all(inputs['input_ids'][:,-1]==29871):
        inputs['input_ids']=torch.cat((inputs['input_ids'],torch.tensor([[29871]],dtype=torch.long,device=model_device)),dim=1)
    with torch.inference_mode():
        gen=model.generate(**inputs,max_new_tokens=action_dim,do_sample=False,return_dict_in_generate=True,output_scores=True)
    return decode_action(gen.sequences[0,-action_dim:]), gen.sequences[0,-action_dim:].cpu().numpy(), inputs

# ── Task config ───────────────────────────────────────────────────
TASK_CFG={'alphabet_soup':0,'cream_cheese':1,'salad_dressing':2,'bbq_sauce':3,'ketchup':4,'tomato_sauce':5,'butter':6,'milk':7,'orange_juice':9}
tid = TASK_CFG.get(args.task)
if tid is None: log('FATAL: unknown task'); sys.exit(1)

# ── Env ───────────────────────────────────────────────────────────
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm=benchmark.get_benchmark_dict(); ts=bm['libero_object'](); to=ts.get_task(tid); ist=ts.get_task_init_states(tid)
if args.state_id>=len(ist): log('FATAL: state OOB'); sys.exit(1)
instr = str(to.language) if hasattr(to,'language') and to.language else args.task.replace('_',' ')
bd = os.path.join(get_libero_path('bddl_files'),to.problem_folder,to.bddl_file)
env=OffScreenRenderEnv(bddl_file_name=bd,camera_heights=256,camera_widths=256,has_renderer=False,has_offscreen_renderer=True,
    use_camera_obs=True,camera_names=['agentview'],control_freq=20,render_gpu_device_id=_render_gpu)
env.seed(args.seed); obs=env.reset(); env.sim.data.qvel[:]=0; env.sim.forward(); env.set_init_state(ist[args.state_id])
log('Env: %s s%d [%s]'%(args.task,args.state_id,instr))

# ── Run episode ───────────────────────────────────────────────────
trace_rows = []; step = 0; done = False
trigger_active = False; trigger_step = -1; attack_until = -1; attacks_applied = 0
burst_len = args.proprionostep_burst_len

while step < args.max_steps and not done:
    img = obs['agentview_image']

    # 1. Clean decode
    raw_action, clean_tids, inputs = clean_decode(img, instr)

    # 2. ProprioNoStep inference
    det_result = {}
    if detector is not None:
        det_result = detector.step(obs, raw_action, env)

    # 3. Trigger logic (condition-agnostic)
    force_trigger = (args.debug_force_trigger_step >= 0 and step == args.debug_force_trigger_step)
    if det_result.get('trigger_now') or force_trigger:
        trigger_active = True; trigger_step = step
        attack_until = step + burst_len - 1
        if force_trigger:
            det_result['trigger_now'] = True
            det_result['trigger_reason'] = 'debug_force_step_%d' % step

    # 4. Determine attack_this_step
    attack_this_step = trigger_active and step <= attack_until

    # 5. Apply VIS/random perturbation to raw_action
    adv_grip = float(raw_action[-1]); clean_grip = adv_grip
    pgd_applied = False; attack_dt = 0.0; arm_l2 = 0.0; token_flip = 0
    condition_action_modified = 'none'

    if attack_this_step:
        if args.condition == 'vis_pgd':
            from gripper_attack.attack_adapter import TokenPrefixPGDAttacker
            atk_cfg = {'attack_optimizer': {
                'method':'token_prefix_pgd','objective':args.objective,
                'epsilon':args.eps_raw_pixels/255.0,
                'step_size':args.eps_raw_pixels/255.0/max(args.pgd_steps,1)*1.5,
                'num_steps':args.pgd_steps,'random_start':True,
                'arm_preserve_weight':0.5,'gripper_margin':5.0}}
            attacker_obj = TokenPrefixPGDAttacker(model=model,processor=processor,config=atk_cfg,
                seed=args.seed+step,device=str(model_device),preprocess_kwargs={'postprocess_gripper':True})
            attacker_obj._freeze_model()
            try:
                t0=time.time(); atk_result=attacker_obj.attack(observation=img,instruction=instr,
                    clean_action=raw_action,target_action=raw_action.copy(),clean_model_output=None,unnorm_key=UNNORM_KEY)
                attack_dt=time.time()-t0
                if atk_result.debug and 'adv_inputs' in atk_result.debug:
                    from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs
                    rd=redecode_openvla_action_from_adv_inputs(model=model,processor=processor,
                        adv_inputs=atk_result.debug['adv_inputs'],instruction=instr,unnorm_key=UNNORM_KEY)
                    raw_action=rd.action; adv_grip=float(raw_action[-1])
                    arm_l2=float(np.linalg.norm(raw_action[:6]-np.asarray(decode_action(torch.as_tensor(clean_tids)))[:6]))
                    token_flip=int(clean_tids[-1])!=int(rd.token_ids[-1])
                    pgd_applied=True; attacks_applied+=1; condition_action_modified='vis_pgd'
            except Exception as e: condition_action_modified='vis_pgd_error'

        elif args.condition == 'random_linf':
            eps=args.eps_raw_pixels/255.0
            noise=np.random.uniform(-eps,eps,img.shape).astype(np.float32)
            img_p = np.clip(img.astype(np.float32)/255.0+noise,0,1); img_p=(img_p*255).astype(np.uint8)
            raw_action,_,_ = clean_decode(img_p, instr); condition_action_modified='random_linf'

    # 6. Compute final env_action FROM final raw_action
    env_action = normalize_gripper_action(raw_action.copy(), binarize=True)
    env_action = invert_gripper_action(env_action)
    final_raw_gripper = float(raw_action[-1])
    final_env_gripper = float(env_action[-1])

    # 7. Oracle/proxy override on env_action (post-normalize)
    oracle_open_applied = False
    if attack_this_step:
        if args.condition == 'oracle_open':
            env_action[-1] = ENV_GRIPPER_OPEN_VALUE; oracle_open_applied = True; condition_action_modified = 'oracle_open'
        elif args.condition == 'sustained_command_open_proxy':
            env_action[-1] = ENV_GRIPPER_OPEN_VALUE; oracle_open_applied = True; condition_action_modified = 'sustained_command_open_proxy'
    final_env_gripper = float(env_action[-1])

    # 8. Get qpos before step
    try: qpos=env.sim.data.qpos; qpos_pre=float(np.mean([qpos[-2],qpos[-1]]))
    except: qpos_pre=0.0

    # 9. Step
    obs, reward, done, info = env.step(env_action)

    # 10. Get qpos after step
    try: qpos=env.sim.data.qpos; qpos_post=float(np.mean([qpos[-2],qpos[-1]]))
    except: qpos_post=0.0

    eef = obs.get('robot0_eef_pos', np.zeros(3))
    is_open = env_gripper_is_open(final_env_gripper)

    trace_rows.append({
        'task':args.task,'condition':args.condition,'seed':str(args.seed),'state_id':str(args.state_id),
        'step':step,'in_window':trigger_active,'attack_this_step':attack_this_step,
        'raw_gripper':final_raw_gripper,'env_gripper':final_env_gripper,
        'gripper_qpos':qpos_pre,'qpos_pre':qpos_pre,'qpos_post':qpos_post,
        'clean_grip':clean_grip,'adv_grip':adv_grip,'arm_l2':arm_l2,'token_flip':token_flip,'attack_dt':round(attack_dt,4),
        'pgd_applied':pgd_applied,'attacks_applied':attacks_applied,
        'eef_x':float(eef[0]),'eef_y':float(eef[1]),'eef_z':float(eef[2]),
        'done':bool(done),'timeout':(step>=args.max_steps-1 and not done),
        'condition_action_modified':condition_action_modified,'oracle_open_applied':oracle_open_applied,
        'trigger_active':trigger_active,'trigger_step':trigger_step,'attack_until':attack_until,
        'proprionostep_hazard_score':det_result.get('hazard_score',0.0),
        'proprionostep_release_safe_score':det_result.get('release_safe_score',0.0),
        'proprionostep_phase_idx':det_result.get('phase_idx',-1),
        'proprionostep_phase_confidence':det_result.get('phase_confidence',0.0),
        'proprionostep_trigger_now':int(det_result.get('trigger_now',False)),
        'proprionostep_trigger_reason':det_result.get('trigger_reason',''),
        'qpos_source':'mujoco_finger_joint_mean',
        'prompt_style':'original_in_out',
        'window_source':'proprionostep_triggered' if trigger_active else 'clean_full_episode',
        'render_gpu_device_id':_render_gpu,
        'cuda_visible_devices':_VISIBLE,
    })
    step += 1

env.close()

# ── Summarize ─────────────────────────────────────────────────────
window_rows = [r for r in trace_rows if r['attack_this_step'] or r['in_window']]
all_rows = trace_rows
n_open = sum(1 for r in all_rows if env_gripper_is_open(float(r['env_gripper'])))
streaks=[]; cur=0
for r in all_rows:
    if env_gripper_is_open(float(r['env_gripper'])): cur+=1
    else: streaks.append(cur); cur=0
streaks.append(cur); max_streak=max(streaks) if streaks else 0
attacked=[r for r in all_rows if r['attack_this_step']]
n_flip=sum(1 for r in all_rows if r['token_flip'])
n_trigger=sum(1 for r in all_rows if r['proprionostep_trigger_now'])
hazards=[r['proprionostep_hazard_score'] for r in all_rows if r['proprionostep_hazard_score']]
qpos_window = [r for r in all_rows if r['attack_this_step']] or all_rows
qpos_delta = float(qpos_window[-1]['qpos_post']-qpos_window[0]['qpos_pre']) if len(qpos_window)>1 else 0.0

summary = {
    'task':args.task,'state_id':str(args.state_id),'condition':args.condition,'seed':str(args.seed),
    'success':bool(done),'total_steps':step,
    'trigger_active':trigger_active,'trigger_step':trigger_step,'attack_until':attack_until,
    'attacks_applied':attacks_applied,'token_flips':n_flip,
    'open_count':n_open,'longest_open_streak':max_streak,
    'proprionostep_trigger_count':n_trigger,
    'proprionostep_hazard_mean':round(float(np.mean(hazards)),6) if hazards else 0.0,
    'proprionostep_hazard_max':round(float(np.max(hazards)),6) if hazards else 0.0,
    'qpos_delta':round(qpos_delta,6),'qpos_delta_pre':qpos_window[0]['qpos_pre'] if qpos_window else 0,
    'qpos_delta_post':qpos_window[-1]['qpos_post'] if qpos_window else 0,
    'feature_schema_version':FEATURE_SCHEMA_VERSION,
    'proprionostep_action_delta_mode':args.proprionostep_action_delta_mode,
    'burst_len':burst_len,'effective_pgd_restarts':effective_pgd_restarts,
    'qpos_source':'mujoco_finger_joint_mean','prompt_style':'original_in_out',
    'window_source':'proprionostep_triggered' if trigger_active else 'clean_full_episode',
    'render_gpu_device_id':_render_gpu,'cuda_visible_devices':_VISIBLE,
    # Oracle semantics: official normalize + invert maps raw OPEN to env_action[-1] = -1.0.
    'oracle_open_semantics':'env_action_minus1_equals_minus1_means_OPEN_after_normalize_invert',
}

# ── Save ──────────────────────────────────────────────────────────
ts=datetime.now().strftime('%H%M%S')
rid='vis_%s_s%s_%s_%s'%(args.task,args.state_id,args.condition,ts)
with open(os.path.join(args.output_dir,rid+'_trace.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(trace_rows[0].keys())); w.writeheader(); w.writerows(trace_rows)
with open(os.path.join(args.output_dir,rid+'_summary.json'),'w') as f: json.dump(summary,f,indent=2)
log('Saved: %s (%d steps, trigger=%d hazard=%.4f open=%d streak=%d attacks=%d qpos_delta=%.6f)'%(
    rid,step,n_trigger,summary['proprionostep_hazard_mean'],n_open,max_streak,attacks_applied,qpos_delta))
