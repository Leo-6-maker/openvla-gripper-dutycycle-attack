import json, os, sys
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels')
from v22_production_v2 import (
    parse_sidecar, get_object_slices_for_task,
    compute_grasp_state, compute_placement_state, compute_safe_release,
    compute_terminal_state, _slice_vector, _dist,
)
CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'

suite, task, state = 'libero_object', 'task_02', 'state_07'
ident = suite + '/' + task + '/' + state
sidecar_path = os.path.join(CS200, suite, task, state, 'privileged_teacher_sidecar.jsonl')
summary_path = os.path.join(CS200, suite, task, state, 'episode_summary.json')
parsed = parse_sidecar(sidecar_path)
steps_data = parsed['steps']
with open(summary_path) as f:
    ep_summary = json.load(f)
task_idx = int(task.replace('task_', ''))
bddl = get_object_slices_for_task(suite, task_idx)
task_role = bddl['task_role']
grasp = compute_grasp_state(steps_data, task_role['manipulated_objects'], task_role['support_names'])
terminal = compute_terminal_state(steps_data, ep_summary)
placement = compute_placement_state(steps_data, grasp, task_role['manipulated_objects'], bddl['object_slices'], task_role['target_names'])
safe_rel = compute_safe_release(steps_data, grasp, terminal, placement)

T = len(steps_data)
n_placed = sum(1 for p in placement if p['object_placed'])
n_sr = sum(1 for s in safe_rel if s.get('planned_release_detected'))
print('Episode:', ident, 'T:', T, 'success:', ep_summary.get('success'))
print('  placed steps:', n_placed, 'SR steps:', n_sr)
print('  grasp_est steps:', sum(1 for g in grasp if g['grasp_established']))
print('  contact_est steps:', sum(1 for g in grasp if g.get('contact_established')))

# Check target fallback
targets = task_role['target_names']
slices = bddl['object_slices']
print('  targets:', targets)
print('  slice keys:', sorted(slices.keys()))
for tname in targets:
    tspec = slices.get(tname)
    if tspec is None:
        found = False
        for fk in sorted(slices.keys()):
            if fk in tname or tname.startswith(fk):
                found = True
                print('  Fallback: ' + tname + ' -> ' + fk)
                break
        if not found:
            print('  NO FALLBACK for: ' + tname)
    else:
        print('  Direct match: ' + tname)

# Last 10 steps: dist + contact state
manip = task_role['manipulated_objects']
print('  Last 10 steps:')
for t in range(max(0, T-10), T):
    for name in manip:
        spec = slices.get(name)
        if spec is None: continue
        obj_pos = _slice_vector(steps_data[t].get('object_state', []), spec, 'pos')
        if obj_pos is None: continue
        for tname in targets:
            tspec = slices.get(tname)
            if tspec is None:
                for fk in sorted(slices.keys()):
                    if fk in tname or tname.startswith(fk):
                        tspec = slices.get(fk)
                        break
            if tspec is None: continue
            tpos = _slice_vector(steps_data[t].get('object_state', []), tspec, 'pos')
            if tpos is None: continue
            d = _dist(obj_pos, tpos)
            ce = grasp[t].get('contact_established', False)
            ge = grasp[t].get('grasp_established', False)
            qpos = steps_data[t].get('robot0_gripper_qpos', [0,0])
            w = abs(qpos[0]) + abs(qpos[1]) if len(qpos) >= 2 else 0
            print('  t=' + str(t) + ': dist=' + str(round(d,3)) + ' ce=' + str(ce) + ' ge=' + str(ge) + ' w=' + str(round(w,3)))
            break
        break
