"""Phase B: Collect fresh FEC canary parents via clean OpenVLA rollouts."""
import sys, os, json, time, hashlib, argparse, numpy as np, torch
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--suite', required=True, choices=['libero_10','libero_goal','libero_object','libero_spatial'])
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--start-state', type=int, default=100)
ap.add_argument('--max-attempts', type=int, default=50)
ap.add_argument('--target-successes', type=int, default=5)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--task-id', type=int, default=0)
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['MUJOCO_GL'] = 'egl'

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.libero_v4_env_factory import build_v4_exact_env

MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
MAX_STEPS = {'libero_10': 700, 'libero_goal': 400, 'libero_object': 400, 'libero_spatial': 400}

MODEL_PATH = MODEL_PATHS[args.suite]
MAX_STEP = MAX_STEPS[args.suite]
OUT_DIR = Path(args.out_dir) / args.suite
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Suite: {}  Model: {}  MaxSteps: {}'.format(args.suite, MODEL_PATH, MAX_STEP))
print('Target: {} successes, starting state={}'.format(args.target_successes, args.start_state))

# Load OpenVLA model
from transformers import AutoProcessor, AutoModelForVision2Seq
print('Loading model...')
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
                                                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
model.eval()
print('Model loaded.')

def unnorm_action(action, unnorm_key='libero_10'):
    """Default unnorm — per-suite means."""
    means = {'libero_10': 0.0, 'libero_goal': 0.0, 'libero_object': 0.0, 'libero_spatial': 0.0}
    stds = {'libero_10': 1.0, 'libero_goal': 1.0, 'libero_object': 1.0, 'libero_spatial': 1.0}
    return action * stds.get(unnorm_key, 1.0) + means.get(unnorm_key, 0.0)

census = []
successes = []
state_id = args.start_state

while len(successes) < args.target_successes and state_id < args.start_state + args.max_attempts:
    eid = '{}/task_00/state_{}'.format(args.suite, state_id)
    seed = state_id
    print('\n[{}] Trying {} (seed={}) ...'.format(len(successes)+1, eid, seed))

    try:
        env = build_v4_exact_env(task_suite_name=args.suite, task_id=args.task_id, seed=seed, headless=True)
        obs = env.reset()
    except Exception as e:
        print('  Env creation failed: {}'.format(e))
        state_id += 1
        continue

    episode_done = False
    step_count = 0
    task_success = False
    actions = []

    try:
        for step in range(MAX_STEP):
            # OpenVLA forward
            inputs = processor(images=obs['frontview'], text='', return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            action_raw = outputs.logits[0, -1, :7].cpu().numpy()
            action_decoded = unnorm_action(action_raw)
            actions.append(action_decoded.tolist())

            obs, reward, done, info = env.step(action_decoded)
            step_count += 1

            if done:
                task_success = info.get('success', False)
                episode_done = True
                break
    except Exception as e:
        print('  Episode error at step {}: {}'.format(step_count, e))
    finally:
        env.close()

    result = {
        'eid': eid, 'suite': args.suite, 'task_id': args.task_id,
        'state_id': state_id, 'seed': seed,
        'success': task_success, 'steps': step_count,
        'model_path': MODEL_PATH
    }
    census.append(result)

    if task_success:
        successes.append(eid)
        print('  SUCCESS ({} steps) — {}/{} collected'.format(step_count, len(successes), args.target_successes))
        # Save initial state snapshot
        snapshot = {'eid': eid, 'suite': args.suite, 'task_id': args.task_id,
                    'state_id': state_id, 'seed': seed, 'steps': step_count}
        with open(OUT_DIR / 'state_{}.json'.format(state_id), 'w') as f:
            json.dump(snapshot, f, indent=2)
    else:
        print('  FAIL ({} steps)'.format(step_count))

    state_id += 1

# Write census
with open(OUT_DIR / 'census.json', 'w') as f:
    json.dump({'suite': args.suite, 'n_attempts': len(census), 'n_successes': len(successes),
               'successes': successes, 'census': census}, f, indent=2)

print('\n{}: {}/{} successes from {} attempts'.format(args.suite, len(successes), args.target_successes, len(census)))
print('Success IDs:', successes)
print('Census: {}'.format(OUT_DIR / 'census.json'))
