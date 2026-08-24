"""T2R-C1 Smoke: Verify MuJoCo entity access on 3 representative tasks.

Tests:
  1. Ordinary object target (bowl/plate) — body match
  2. Microwave heating-region — site match
  3. Desk-caddy contain-region — site match

Validates:
  - OffScreenRenderEnv creation
  - sim.model access
  - Exact site existence for region targets
  - STRIP_SUFFIX_BODY is blocked
"""
import json, os, sys, time, hashlib

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import get_object_slices_for_task

SMOKE_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_smoke'
os.makedirs(SMOKE_OUT, exist_ok=True)

# 3 test tasks
SMOKE_TASKS = [
    ('libero_goal', 6, 'ordinary_On_bowl'),       # On(cream_cheese, akita_black_bowl)
    ('libero_10', 9, 'fixture_microwave'),         # In(mug, microwave_heating_region)
    ('libero_10', 5, 'fixture_desk_caddy'),        # In(book, desk_caddy_back_contain_region)
]


def smoke_test_task(suite, task_idx, label):
    """Test environment creation and entity access for one task."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    result = {'suite': suite, 'task_idx': task_idx, 'label': label, 'status': 'STARTING'}

    try:
        # Get task and BDDL
        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)
        result['bddl_path'] = bddl_path

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=224,
            camera_widths=224,
            render_gpu_device_id=-1,
            has_renderer=False,
            has_offscreen_renderer=False,
            horizon=500,
        )
        env.reset()
        result['env_created'] = True

        # Access MuJoCo model
        sim = env.sim
        model = sim.model

        # Collect sites
        sites = {}
        for i in range(model.nsite):
            name = model.site(i).name
            if name:
                sites[name] = {
                    'id': model.site(name).id,
                    'body_id': int(model.site_bodyid[model.site(name).id]),
                    'pos': model.site_pos[model.site(name).id].tolist(),
                    'size': model.site_size[model.site(name).id].tolist(),
                    'type': int(model.site_type[model.site(name).id]),
                }

        # Collect bodies
        bodies = {}
        for i in range(model.nbody):
            name = model.body(i).name
            if name and name != 'world':
                bodies[name] = {
                    'id': model.body(name).id,
                    'parent_id': int(model.body_parentid[model.body(name).id]),
                    'pos': model.body_pos[model.body(name).id].tolist(),
                }

        result['n_sites'] = len(sites)
        result['n_bodies'] = len(bodies)

        # Get BDDL relations
        bddl_info = get_object_slices_for_task(suite, task_idx)
        task_role = bddl_info['task_role'] if bddl_info else {}
        g_rels = task_role.get('goal_relations', [])
        result['goal_relations'] = [(r[0], r[1], r[2]) for r in g_rels]

        # Try to resolve each target
        resolutions = []
        for pred, obj, tgt in g_rels:
            res = {'predicate': pred, 'object': obj, 'target_bddl': tgt}

            # Exact site match
            if tgt in sites:
                res['resolution'] = 'EXACT_SITE'
                res['site_id'] = sites[tgt]['id']
                res['site_size'] = sites[tgt]['size']
                res['parent_body_id'] = sites[tgt]['body_id']
                # Find parent body name
                for bn, bi in bodies.items():
                    if bi['id'] == sites[tgt]['body_id']:
                        res['parent_body_name'] = bn
                        break
            elif tgt in bodies:
                res['resolution'] = 'EXACT_BODY'
                res['body_id'] = bodies[tgt]['id']
            else:
                # Check: does stripping suffix give exact site?
                found = False
                for suffix in ['_contain_region', '_init_region', '_cook_region',
                              '_heating_region', '_top_region', '_front_region',
                              '_back_contain_region']:
                    base = tgt.replace(suffix, '')
                    if base in sites:
                        res['resolution'] = 'STRIP_SUFFIX→SITE'
                        res['site_id'] = sites[base]['id']
                        res['site_name'] = base
                        found = True
                        break
                    if base in bodies:
                        res['resolution'] = 'STRIP_SUFFIX→BODY (BLOCKED)'
                        res['body_id'] = bodies[base]['id']
                        res['body_name'] = base
                        found = True
                        break
                if not found:
                    res['resolution'] = 'UNRESOLVED'
                    res['available_sites'] = sorted(sites.keys())[:15]
                    res['available_bodies'] = sorted(bodies.keys())[:15]

            resolutions.append(res)

        result['resolutions'] = resolutions

        # Check: any STRIP_SUFFIX→BODY resolutions?
        blocked = [r for r in resolutions if 'BLOCKED' in r.get('resolution', '')]
        if blocked:
            result['status'] = 'BLOCKED_BODY_FALLBACK'
        else:
            result['status'] = 'OK'

        env.close()
    except Exception as e:
        result['status'] = f'ERROR: {str(e)[:200]}'
        result['error'] = str(e)

    return result


def main():
    print('=' * 60)
    print('T2R-C1 Smoke: 3-Task Entity Access')
    print('=' * 60)

    all_results = []
    for suite, task_idx, label in SMOKE_TASKS:
        print(f'\n{suite}/task_{task_idx:02d} ({label})...', flush=True)
        r = smoke_test_task(suite, task_idx, label)
        all_results.append(r)

        status = r['status']
        print(f'  Status: {status}')
        print(f'  Env created: {r.get("env_created", False)}')
        print(f'  Sites: {r.get("n_sites", 0)}, Bodies: {r.get("n_bodies", 0)}')
        print(f'  Relations: {r.get("goal_relations", [])}')
        for res in r.get('resolutions', []):
            print(f'    {res["target_bddl"]}: {res["resolution"]}')
            if 'site_size' in res:
                print(f'      site size: {res["site_size"]}')

    # Summary
    ok = all(r['status'] == 'OK' for r in all_results)
    print(f'\n{"PASS" if ok else "FAIL"}: {sum(1 for r in all_results if r["status"]=="OK")}/{len(all_results)} smoke tests OK')

    smoke_report = {
        'gate': 'T2R-C1_SMOKE',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'results': all_results,
    }
    report_path = os.path.join(SMOKE_OUT, 'SMOKE_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(smoke_report, f, indent=2, default=str)
    print(f'Report: {report_path}')

    sys.exit(0 if ok else 5)


if __name__ == '__main__':
    main()
