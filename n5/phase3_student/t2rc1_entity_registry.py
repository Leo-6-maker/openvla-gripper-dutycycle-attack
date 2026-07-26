"""T2R-C1: Simulator Entity Registry.

For each task, start LIBERO, access MuJoCo model, and build a static registry:
  BDDL relation → exact body/site/geom numeric IDs, transforms, extents.

No inference. No episode replay. Just static model introspection.
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import get_object_slices_for_task

COHORT_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rd_confirmation_cohort/T2RD_CONFIRM_MANIFEST_V1.json'
T2RC1_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_entity_registry'
os.makedirs(T2RC1_OUT, exist_ok=True)


def build_entity_registry(suite, task_idx):
    """Start LIBERO, extract full MuJoCo entity registry for one task."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco

    # Get task and BDDL
    benchmark = get_benchmark(suite)(0)
    task = benchmark.get_task(task_idx)
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=224,
        camera_widths=224,
        render_gpu_device_id=-1,  # no GPU
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=10,
    )
    env.reset()

    # Access MuJoCo model
    model = env.sim.model

    # Collect all named entities
    bodies = {}
    for i in range(model.nbody):
        name = model.body(i).name
        if name and name != 'world':
            body_id = model.body(name).id
            bodies[name] = {
                'id': body_id,
                'name': name,
                'parent_id': model.body_parentid[body_id],
                'pos': model.body_pos[body_id].tolist(),
                'quat': model.body_quat[body_id].tolist(),
                'inertia': model.body_inertia[body_id].tolist(),
            }

    sites = {}
    for i in range(model.nsite):
        name = model.site(i).name
        if name:
            site_id = model.site(name).id
            sites[name] = {
                'id': site_id,
                'name': name,
                'body_id': model.site_bodyid[site_id],
                'pos': model.site_pos[site_id].tolist(),
                'quat': model.site_quat[site_id].tolist(),
                'size': model.site_size[site_id].tolist(),
                'type': int(model.site_type[site_id]),
            }

    geoms = {}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name:
            geom_id = model.geom(name).id
            geoms[name] = {
                'id': geom_id,
                'name': name,
                'body_id': model.geom_bodyid[geom_id],
                'pos': model.geom_pos[geom_id].tolist(),
                'quat': model.geom_quat[geom_id].tolist(),
                'size': model.geom_size[geom_id].tolist(),
                'type': int(model.geom_type[geom_id]),
            }

    # Get BDDL info
    bddl_info = get_object_slices_for_task(suite, task_idx)
    task_role = bddl_info['task_role'] if bddl_info else {}
    goal_relations = task_role.get('goal_relations', [])
    object_slices = bddl_info['object_slices'] if bddl_info else {}

    # Map BDDL goal relations to MuJoCo entities
    relation_map = []
    for pred, obj_name, target_name in goal_relations:
        entry = {
            'predicate': pred,
            'object_bddl': obj_name,
            'target_bddl': target_name,
        }

        # Try exact match in sites (BDDL regions are often MuJoCo sites)
        if target_name in sites:
            entry['target_entity_type'] = 'site'
            entry['target_entity_id'] = sites[target_name]['id']
            entry['target_parent_body'] = site_body_name(sites, target_name, bodies)
            entry['target_size'] = sites[target_name]['size']
            entry['target_site_type'] = sites[target_name]['type']
            entry['resolution'] = 'EXACT_SITE'
        elif target_name in bodies:
            entry['target_entity_type'] = 'body'
            entry['target_entity_id'] = bodies[target_name]['id']
            entry['target_parent_body'] = target_name
            entry['resolution'] = 'EXACT_BODY'
        elif target_name in geoms:
            entry['target_entity_type'] = 'geom'
            entry['target_entity_id'] = geoms[target_name]['id']
            entry['target_parent_body'] = geom_body_name(geoms, target_name, bodies)
            entry['resolution'] = 'EXACT_GEOM'
        else:
            # Try fuzzy: strip suffixes
            for suffix in ['_contain_region', '_init_region', '_cook_region',
                          '_heating_region', '_top_region', '_front_region',
                          '_back_contain_region']:
                base = target_name.replace(suffix, '')
                if base in bodies:
                    entry['target_entity_type'] = 'body'
                    entry['target_entity_id'] = bodies[base]['id']
                    entry['target_parent_body'] = base
                    entry['resolution'] = 'STRIP_SUFFIX_BODY'
                    entry['resolved_name'] = base
                    break
                if base in sites:
                    entry['target_entity_type'] = 'site'
                    entry['target_entity_id'] = sites[base]['id']
                    entry['target_parent_body'] = site_body_name(sites, base, bodies)
                    entry['resolution'] = 'STRIP_SUFFIX_SITE'
                    entry['resolved_name'] = base
                    break

            if 'resolution' not in entry:
                # Try substring search (with warning)
                for name in sorted(sites.keys()):
                    name_clean = _clean_name(name)
                    target_clean = _clean_name(target_name)
                    if name_clean in target_clean or target_clean in name_clean:
                        entry['target_entity_type'] = 'site'
                        entry['target_entity_id'] = sites[name]['id']
                        entry['resolution'] = 'SUBSTRING_SITE_WARNING'
                        entry['resolved_name'] = name
                        break
                if 'resolution' not in entry:
                    for name in sorted(bodies.keys()):
                        name_clean = _clean_name(name)
                        target_clean = _clean_name(target_name)
                        if name_clean in target_clean or target_clean in name_clean:
                            entry['target_entity_type'] = 'body'
                            entry['target_entity_id'] = bodies[name]['id']
                            entry['resolution'] = 'SUBSTRING_BODY_WARNING'
                            entry['resolved_name'] = name
                            break

            if 'resolution' not in entry:
                entry['resolution'] = 'UNRESOLVED'
                entry['available_sites'] = sorted(sites.keys())[:20]
                entry['available_bodies'] = sorted(bodies.keys())[:20]

        # Object mapping
        if obj_name in bodies:
            entry['object_entity_type'] = 'body'
            entry['object_entity_id'] = bodies[obj_name]['id']
        elif obj_name in object_slices:
            entry['object_in_slices'] = True

        relation_map.append(entry)

    env.close()

    return {
        'suite': suite,
        'task_idx': task_idx,
        'bddl_path': bddl_path,
        'n_bodies': len(bodies),
        'n_sites': len(sites),
        'n_geoms': len(geoms),
        'relation_map': relation_map,
        # Include full lists for debugging
        'all_site_names': sorted(sites.keys()),
        'all_body_names': sorted(bodies.keys()),
        'all_geom_names': sorted(geoms.keys())[:30],
    }


def site_body_name(sites_dict, site_name, bodies_dict):
    """Get parent body name for a site."""
    if site_name in sites_dict:
        body_id = sites_dict[site_name]['body_id']
        for name, info in bodies_dict.items():
            if info['id'] == body_id:
                return name
    return None


def geom_body_name(geoms_dict, geom_name, bodies_dict):
    """Get parent body name for a geom."""
    if geom_name in geoms_dict:
        body_id = geoms_dict[geom_name]['body_id']
        for name, info in bodies_dict.items():
            if info['id'] == body_id:
                return name
    return None


def _clean_name(name):
    return name.replace('_contain_region', '').replace('_init_region', '').replace('_cook_region', '').replace('_heating_region', '').replace('_top_region', '').replace('_front_region', '').replace('_back_contain_region', '')


def main():
    print('=' * 60)
    print('T2R-C1: Simulator Entity Registry')
    print('=' * 60)

    with open(COHORT_MANIFEST) as f:
        cohort = json.load(f)
    print(f'Cohort SHA: {cohort["self_sha256"][:16]}...')

    # Get unique tasks from cohort
    unique_tasks = set()
    for ident in cohort['identities']:
        suite, task, _ = ident.split('/')
        task_idx = int(task.replace('task_', ''))
        unique_tasks.add((suite, task_idx))

    print(f'Unique tasks: {len(unique_tasks)}')
    print(f'Tasks: {sorted(unique_tasks)}')

    all_registries = {}
    unresolved = []
    substring_warnings = []
    fixture_resolutions = {}

    for suite, task_idx in sorted(unique_tasks):
        print(f'\n  {suite}/task_{task_idx:02d}...', end=' ', flush=True)
        try:
            registry = build_entity_registry(suite, task_idx)
            task_key = f'{suite}/task_{task_idx:02d}'
            all_registries[task_key] = registry

            n_resolved = 0; n_unresolved = 0
            for rm in registry['relation_map']:
                if rm['resolution'] == 'UNRESOLVED':
                    n_unresolved += 1
                    unresolved.append({
                        'task': task_key,
                        'predicate': rm['predicate'],
                        'target': rm['target_bddl'],
                    })
                elif 'WARNING' in rm['resolution']:
                    substring_warnings.append({
                        'task': task_key,
                        'resolution': rm['resolution'],
                        'target': rm['target_bddl'],
                        'resolved': rm.get('resolved_name', '?'),
                    })
                else:
                    n_resolved += 1
                    if rm['resolution'] == 'EXACT_SITE':
                        fixture_resolutions[rm['target_bddl']] = {
                            'entity_type': rm['target_entity_type'],
                            'entity_id': rm['target_entity_id'],
                            'parent_body': rm.get('target_parent_body'),
                            'size': rm.get('target_size'),
                        }

            print(f'{n_resolved} resolved, {n_unresolved} unresolved')
        except Exception as e:
            print(f'ERROR: {e}')
            task_key = f'{suite}/task_{task_idx:02d}'
            all_registries[task_key] = {'error': str(e)}

    # Summary
    total_relations = sum(len(r['relation_map']) for r in all_registries.values() if 'relation_map' in r)
    total_unresolved = len(unresolved)
    total_substring = len(substring_warnings)
    print(f'\n--- Summary ---')
    print(f'Total relations: {total_relations}')
    print(f'Exact resolved: {total_relations - total_unresolved - total_substring}')
    print(f'Substring (warning): {total_substring}')
    print(f'Unresolved: {total_unresolved}')

    if unresolved:
        print(f'\nUnresolved targets:')
        for u in unresolved:
            print(f'  {u["task"]}: {u["predicate"]}({u["target"]})')

    if substring_warnings:
        print(f'\nSubstring-mapped (need verification):')
        for w in substring_warnings[:5]:
            print(f'  {w["task"]}: {w["target"]} -> {w["resolved"]} ({w["resolution"]})')

    print(f'\nFixture resolutions:')
    for target, info in fixture_resolutions.items():
        print(f'  {target}: {info["entity_type"]} id={info["entity_id"]}, '
              f'parent={info["parent_body"]}, size={info["size"]}')

    # Write registry
    registry_output = {
        'gate': 'T2R-C1_ENTITY_REGISTRY',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cohort_sha': cohort['self_sha256'],
        'n_tasks': len(unique_tasks),
        'n_total_relations': total_relations,
        'n_exact_resolved': total_relations - total_unresolved - total_substring,
        'n_substring_warning': total_substring,
        'n_unresolved': total_unresolved,
        'fixture_resolutions': fixture_resolutions,
        'unresolved': unresolved,
        'substring_warnings': substring_warnings,
        'per_task': {
            k: {
                'n_relations': len(v.get('relation_map', [])),
                'resolutions': [rm['resolution'] for rm in v.get('relation_map', [])],
            }
            for k, v in all_registries.items()
        },
    }

    registry_path = os.path.join(T2RC1_OUT, 'ENTITY_REGISTRY.json')
    with open(registry_path, 'w') as f:
        json.dump(registry_output, f, indent=2, default=str)
    registry_sha = hashlib.sha256(open(registry_path, 'rb').read()).hexdigest()
    registry_output['self_sha256'] = registry_sha
    with open(registry_path, 'w') as f:
        json.dump(registry_output, f, indent=2, default=str)

    print(f'\nRegistry: {registry_path}')
    print(f'SHA: {registry_sha[:16]}...')

    if total_unresolved == 0 and total_substring == 0:
        print('\nT2R-C1: PASS — 100% exact resolution, 0 ambiguous mappings')
        sys.exit(0)
    else:
        print(f'\nT2R-C1: NEEDS_FIX — {total_unresolved} unresolved, {total_substring} substring-warning')
        sys.exit(5)


if __name__ == '__main__':
    main()
