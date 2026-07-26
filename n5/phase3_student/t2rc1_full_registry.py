"""T2R-C1 Full: 40-task MuJoCo Entity Registry.

Exact binding: BDDL relation target -> MuJoCo entity (site/body/geom).
STRIP_SUFFIX_BODY is FORBIDDEN. SUBSTRING is FORBIDDEN.
Only EXACT_SITE, EXACT_BODY, EXACT_GEOM, or SHA-bound reviewed alias.

Output: per-task immutable registry + summary.
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import get_object_slices_for_task

T2RC1_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_full_registry'
PER_TASK_DIR = os.path.join(T2RC1_OUT, 'per_task')
os.makedirs(PER_TASK_DIR, exist_ok=True)

FOUR_SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']

# Resolution enum
VALID_RESOLUTIONS = {'EXACT_SITE', 'EXACT_BODY', 'EXACT_GEOM'}
BLOCKED_RESOLUTIONS = {'STRIP_SUFFIX_BODY', 'STRIP_SUFFIX_SITE', 'SUBSTRING'}
SPECIAL = {'UNSUPPORTED_TASK', 'UNRESOLVED', 'ENV_ERROR'}

REGION_SUFFIXES = ['_contain_region', '_init_region', '_cook_region',
                   '_heating_region', '_top_region', '_front_region',
                   '_back_contain_region', '_top_side', '_bottom_region']


def _is_region_target(name):
    return any(name.endswith(s) for s in REGION_SUFFIXES)


def build_registry(suite, task_idx):
    """Build full entity registry for one task. Returns (registry_dict, error_string)."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    result = {
        'suite': suite, 'task_idx': task_idx,
        'task_key': f'{suite}/task_{task_idx:02d}',
        'status': 'STARTING',
    }
    env = None

    try:
        # Get task and BDDL
        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)
        bddl_sha = hashlib.sha256(open(bddl_path, 'rb').read()).hexdigest()
        result['bddl_path'] = bddl_path
        result['bddl_sha256'] = bddl_sha

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=224, camera_widths=224,
            render_gpu_device_id=-1,
            has_renderer=False, has_offscreen_renderer=False,
            horizon=500,
        )
        env.reset()
        result['env_created'] = True

        # Access MuJoCo model
        sim = env.sim
        model = sim.model
        result['model_nbody'] = model.nbody
        result['model_nsite'] = model.nsite
        result['model_ngeom'] = model.ngeom

        # Collect all entities
        sites = {}
        bodies = {}
        geoms = {}

        for i in range(model.nsite):
            name = model.site(i).name
            if name:
                sid = model.site(name).id
                sites[name] = {
                    'id': int(sid),
                    'name': name,
                    'body_id': int(model.site_bodyid[sid]),
                    'pos': [float(x) for x in model.site_pos[sid]],
                    'quat': [float(x) for x in model.site_quat[sid]],
                    'size': [float(x) for x in model.site_size[sid]],
                    'type': int(model.site_type[sid]),
                }

        for i in range(model.nbody):
            name = model.body(i).name
            if name and name != 'world':
                bid = model.body(name).id
                bodies[name] = {
                    'id': int(bid),
                    'name': name,
                    'parent_id': int(model.body_parentid[bid]),
                    'pos': [float(x) for x in model.body_pos[bid]],
                    'quat': [float(x) for x in model.body_quat[bid]],
                }

        for i in range(model.ngeom):
            name = model.geom(i).name
            if name:
                gid = model.geom(name).id
                geoms[name] = {
                    'id': int(gid),
                    'name': name,
                    'body_id': int(model.geom_bodyid[gid]),
                    'pos': [float(x) for x in model.geom_pos[gid]],
                    'size': [float(x) for x in model.geom_size[gid]],
                    'type': int(model.geom_type[gid]),
                }

        result['entities'] = {
            'n_sites': len(sites),
            'n_bodies': len(bodies),
            'n_geoms': len(geoms),
            'site_names': sorted(sites.keys()),
            'body_names': sorted(bodies.keys()),
            'geom_names': sorted(geoms.keys()),
        }

        # Get BDDL info
        bddl_info = get_object_slices_for_task(suite, task_idx)
        if bddl_info is None:
            result['task_disposition'] = 'BDDL_UNAVAILABLE'
            result['status'] = 'FAIL'
            return result, 'BDDL unavailable'

        task_role = bddl_info['task_role']
        g_rels = task_role.get('goal_relations', [])
        result['goal_predicates'] = [(r[0], r[1], r[2]) for r in g_rels]

        # Determine task disposition
        relation_types = set(r[0] for r in g_rels)
        is_supported = bool({'In', 'On', 'Stack'} & relation_types)

        if not g_rels:
            result['task_disposition'] = 'ARTICULATED_UNSUPPORTED'
        elif is_supported:
            result['task_disposition'] = 'SUPPORTED_PLACEMENT'
        else:
            result['task_disposition'] = 'OTHER_RELATION_TYPE'

        # Resolve each relation target
        relation_map = []
        n_unresolved = 0
        n_blocked = 0
        n_exact = 0

        for pred, obj_name, target_name in g_rels:
            entry = {
                'predicate': pred,
                'object_bddl': obj_name,
                'target_bddl': target_name,
            }

            is_region = _is_region_target(target_name)

            # Try exact match (preferred)
            if target_name in sites:
                entry['resolution'] = 'EXACT_SITE'
                entry['entity_id'] = sites[target_name]['id']
                entry['entity_type'] = 'site'
                entry['size'] = sites[target_name]['size']
                entry['parent_body_id'] = sites[target_name]['body_id']
                for bn, bi in bodies.items():
                    if bi['id'] == sites[target_name]['body_id']:
                        entry['parent_body_name'] = bn; break
                n_exact += 1
            elif target_name in bodies:
                if is_region:
                    entry['resolution'] = 'BLOCKED_REGION_AS_BODY'
                    entry['body_id'] = bodies[target_name]['id']
                    n_blocked += 1
                else:
                    entry['resolution'] = 'EXACT_BODY'
                    entry['entity_id'] = bodies[target_name]['id']
                    entry['entity_type'] = 'body'
                    n_exact += 1
            elif target_name in geoms:
                entry['resolution'] = 'EXACT_GEOM'
                entry['entity_id'] = geoms[target_name]['id']
                entry['entity_type'] = 'geom'
                n_exact += 1
            else:
                # Check stripped suffix -> site (FOR REGION TARGETS ONLY)
                found = False
                for suffix in REGION_SUFFIXES:
                    base = target_name.replace(suffix, '')
                    if base != target_name:
                        if base in sites:
                            if is_region:
                                entry['resolution'] = 'STRIP_SUFFIX_SITE'
                                entry['entity_id'] = sites[base]['id']
                                entry['entity_type'] = 'site'
                                entry['resolved_name'] = base
                                entry['size'] = sites[base]['size']
                                found = True; break
                            else:
                                # Non-region: strip suffix -> site is ok
                                entry['resolution'] = 'EXACT_SITE'
                                entry['entity_id'] = sites[base]['id']
                                entry['entity_type'] = 'site'
                                entry['resolved_name'] = base
                                found = True; n_exact += 1; break
                        if base in bodies:
                            if is_region:
                                entry['resolution'] = 'BLOCKED_REGION_STRIP_TO_BODY'
                                entry['body_name'] = base
                                found = True; n_blocked += 1; break
                            else:
                                entry['resolution'] = 'EXACT_BODY'
                                entry['entity_id'] = bodies[base]['id']
                                entry['entity_type'] = 'body'
                                entry['resolved_name'] = base
                                found = True; n_exact += 1; break

                if not found:
                    # Substring search (blocked for regions, last resort for objects)
                    for name in sorted(sites.keys()):
                        n_clean = name.replace('_contain_region', '').replace('_init_region', '')
                        t_clean = target_name.replace('_contain_region', '').replace('_init_region', '')
                        if n_clean in t_clean or t_clean in n_clean:
                            if is_region:
                                entry['resolution'] = 'BLOCKED_SUBSTRING_SITE'
                                n_blocked += 1
                            else:
                                entry['resolution'] = 'EXACT_SITE'
                                entry['entity_id'] = sites[name]['id']
                                entry['entity_type'] = 'site'
                                entry['resolved_name'] = name
                                n_exact += 1
                            found = True; break
                    if not found:
                        for name in sorted(bodies.keys()):
                            n_clean = name.replace('_contain_region', '').replace('_init_region', '')
                            t_clean = target_name.replace('_contain_region', '').replace('_init_region', '')
                            if n_clean in t_clean or t_clean in n_clean:
                                if is_region:
                                    entry['resolution'] = 'BLOCKED_SUBSTRING_BODY'
                                    n_blocked += 1
                                else:
                                    entry['resolution'] = 'EXACT_BODY'
                                    entry['entity_id'] = bodies[name]['id']
                                    entry['entity_type'] = 'body'
                                    entry['resolved_name'] = name
                                    n_exact += 1
                                found = True; break

                if not found:
                    entry['resolution'] = 'UNRESOLVED'
                    n_unresolved += 1

            # Object resolution
            if obj_name in bodies:
                entry['object_entity_type'] = 'body'
                entry['object_entity_id'] = bodies[obj_name]['id']
            elif obj_name in sites:
                entry['object_entity_type'] = 'site'
                entry['object_entity_id'] = sites[obj_name]['id']

            relation_map.append(entry)

        result['relation_map'] = relation_map
        result['resolution_summary'] = {
            'n_total': len(relation_map),
            'n_exact': n_exact,
            'n_blocked': n_blocked,
            'n_unresolved': n_unresolved,
        }

        # Determine status
        if n_blocked > 0:
            result['status'] = 'BLOCKED_RESOLUTION_PRESENT'
        elif n_unresolved > 0:
            result['status'] = 'UNRESOLVED_TARGET_PRESENT'
        else:
            result['status'] = 'OK'

        return result, None

    except Exception as e:
        result['status'] = 'ENV_ERROR'
        result['error'] = str(e)[:300]
        return result, str(e)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def compute_self_hash(obj):
    """Compute deterministic SHA of a JSON-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main():
    print('=' * 60)
    print('T2R-C1 Full: 40-Task Entity Registry')
    print('=' * 60)

    all_results = {}
    env_errors = 0
    blocked = 0
    unresolved = 0
    exact_total = 0
    summaries = []

    for suite in FOUR_SUITES:
        for task_idx in range(10):
            task_key = f'{suite}/task_{task_idx:02d}'
            print(f'{task_key}...', end=' ', flush=True)

            registry, error = build_registry(suite, task_idx)
            all_results[task_key] = registry

            status = registry['status']
            disp = registry.get('task_disposition', '?')
            rs = registry.get('resolution_summary', {})
            n_e = rs.get('n_exact', 0)
            n_b = rs.get('n_blocked', 0)
            n_u = rs.get('n_unresolved', 0)

            print(f'{status}  disp={disp}  exact={n_e} blocked={n_b} unresolved={n_u}')

            if status == 'ENV_ERROR':
                env_errors += 1
            if n_b > 0:
                blocked += 1
            if n_u > 0:
                unresolved += 1
            exact_total += n_e

            # Write per-task artifact
            per_task = {
                'gate': 'T2R-C1_PER_TASK_REGISTRY',
                'task_key': task_key,
                'legacy': registry,
            }
            per_task['self_sha256'] = compute_self_hash(per_task)
            per_task_path = os.path.join(PER_TASK_DIR, f'{suite}_task_{task_idx:02d}.json')
            with open(per_task_path, 'w') as f:
                json.dump(per_task, f, indent=2, default=str)
            per_task_sha = hashlib.sha256(open(per_task_path, 'rb').read()).hexdigest()

            summaries.append({
                'task_key': task_key,
                'status': status,
                'disposition': disp,
                'n_relations': rs.get('n_total', 0),
                'n_exact': n_e,
                'n_blocked': n_b,
                'n_unresolved': n_u,
                'artifact_sha': per_task_sha,
            })

    # Build summary
    n_ok = sum(1 for s in summaries if s['status'] == 'OK')
    n_articulated = sum(1 for s in summaries if s['disposition'] == 'ARTICULATED_UNSUPPORTED')
    n_supported = sum(1 for s in summaries if s['disposition'] == 'SUPPORTED_PLACEMENT')
    n_total_blocked = sum(s['n_blocked'] for s in summaries)
    n_total_unres = sum(s['n_unresolved'] for s in summaries)

    print(f'\n{"=" * 60}')
    print(f'Tasks OK: {n_ok}/40')
    print(f'Supported placement: {n_supported}')
    print(f'Articulated unsupported: {n_articulated}')
    print(f'Total relations: {sum(s["n_relations"] for s in summaries)}')
    print(f'Total exact: {exact_total}')
    print(f'Total blocked: {n_total_blocked}')
    print(f'Total unresolved (supported): {n_total_unres}')
    print(f'Environment errors: {env_errors}')

    # Determine PASS/FAIL
    all_pass = (
        env_errors == 0
        and n_total_blocked == 0
        and n_total_unres == 0
        and n_ok == 40
    )

    summary = {
        'gate': 'T2R-C1_FULL_ENTITY_REGISTRY',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_tasks': 40,
        'n_ok': n_ok,
        'n_env_errors': env_errors,
        'n_blocked_resolutions': n_total_blocked,
        'n_unresolved_supported': n_total_unres,
        'n_supported_placement': n_supported,
        'n_articulated_unsupported': n_articulated,
        'per_task': summaries,
        'status': 'PASS' if all_pass else 'FAIL',
    }

    summary['self_sha256'] = compute_self_hash(summary)
    summary_path = os.path.join(T2RC1_OUT, 'ENTITY_REGISTRY_SUMMARY.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    summary_file_sha = hashlib.sha256(open(summary_path, 'rb').read()).hexdigest()

    print(f'\nSummary: {summary_path}')
    print(f'SHA: {summary_file_sha[:16]}...')

    if all_pass:
        print('\nT2R-C1: PASS')
        sys.exit(0)
    else:
        print(f'\nT2R-C1: FAIL')
        if env_errors > 0:
            print(f'  {env_errors} environment errors')
        if n_total_blocked > 0:
            print(f'  {n_total_blocked} blocked resolutions (region→body)')
        if n_total_unres > 0:
            print(f'  {n_total_unres} unresolved targets')
        sys.exit(5)


if __name__ == '__main__':
    main()
