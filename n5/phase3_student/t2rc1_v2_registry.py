"""T2R-C1-V2: 40-task MuJoCo Entity Registry with Structural Alias Resolution.

Resolution priority (strict, no fallback):
  1. EXACT_SITE       — region target matches site name exactly
  2. EXACT_BODY       — object target matches body name exactly
  3. EXACT_GEOM       — target matches geom name exactly
  4. APPROVED_STRUCTURAL_ALIAS — explicit, unique, structure-verified alias
  5. UNRESOLVED       — everything else

APPROVED alias rules (exhaustive, no automatic discovery):
  R1: {target}_main → body — MuJoCo root body naming convention.
      Verified by: (a) candidate body name is exact {target}_main,
      (b) no other body starts with {target}_ (uniqueness),
      (c) at least one geom with prefix {target}_ has this body as parent
      (hierarchy verification).

FORBIDDEN:
  - STRIP_SUFFIX_BODY, STRIP_SUFFIX_SITE, SUBSTRING
  - region → body fallback
  - first substring match
  - edit-distance acceptance
  - multiple-candidate tie-breaking
  - success-outcome-driven resolution
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'phase2_labels'))
from v22_production_v2 import get_object_slices_for_task

# ── Constants ──
VALID_RESOLUTIONS = {'EXACT_SITE', 'EXACT_BODY', 'EXACT_GEOM', 'APPROVED_STRUCTURAL_ALIAS'}
BLOCKED_RESOLUTIONS = {'STRIP_SUFFIX_BODY', 'STRIP_SUFFIX_SITE', 'SUBSTRING'}
REGION_SUFFIXES = ['_contain_region', '_init_region', '_cook_region',
                   '_heating_region', '_top_region', '_front_region',
                   '_back_contain_region', '_top_side', '_bottom_region']

# Explicit alias rules (the ONLY allowed structural transformations)
APPROVED_ALIAS_RULES = {
    'R1_main_suffix': {
        'description': 'BDDL object name → MuJoCo root body ({name}_main)',
        'transform': lambda name: name + '_main',
        'verification': 'hierarchy',  # must verify geom parent chain
    },
}


def _is_region_target(name):
    return any(name.endswith(s) for s in REGION_SUFFIXES)


def _verify_alias_hierarchy(target_name, alias_body_name, bodies, geoms, target_body_id):
    """Verify that the aliased body has geoms belonging to the target object.

    Checks:
      (a) alias_body_name == target_name + '_main' (exact structural rule)
      (b) No other body starts with target_name + '_' besides the alias
      (c) At least one geom with prefix target_name has this body as parent
    """
    # (a) Structural rule check
    expected = target_name + '_main'
    if alias_body_name != expected:
        return False, f'alias_body_{alias_body_name}_neq_expected_{expected}'

    # (b) Uniqueness: no other body starts with {target_name}_
    prefix = target_name + '_'
    other_bodies = [bn for bn in bodies
                    if bn.startswith(prefix) and bn != alias_body_name]
    if other_bodies:
        return False, f'non_unique_bodies: {other_bodies[:5]}'

    # (c) Hierarchy: at least one geom with prefix {target_name}_ has alias_body as parent
    geom_prefix = target_name + '_g'
    matching_geoms = [gn for gn in geoms if gn.startswith(geom_prefix)]
    if not matching_geoms:
        return False, f'no_geoms_with_prefix_{geom_prefix}'

    for gn in matching_geoms[:3]:  # check first 3
        if geoms[gn]['body_id'] != target_body_id:
            return False, f'geom_{gn}_body_mismatch'

    return True, f'unique_hierarchy_verified_{len(matching_geoms)}_geoms'


def build_registry_v2(suite, task_idx):
    """Build entity registry with structural alias resolution."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    result = {
        'version': 'C1-V2',
        'suite': suite, 'task_idx': task_idx,
        'task_key': f'{suite}/task_{task_idx:02d}',
        'status': 'STARTING',
    }
    env = None

    try:
        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)
        bddl_sha = hashlib.sha256(open(bddl_path, 'rb').read()).hexdigest()
        result['bddl_path'] = bddl_path
        result['bddl_sha256'] = bddl_sha

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=224, camera_widths=224,
            render_gpu_device_id=-1,
            has_renderer=False, has_offscreen_renderer=False,
            horizon=500,
        )
        env.reset()
        result['env_created'] = True

        sim = env.sim; model = sim.model
        result['model_nbody'] = model.nbody
        result['model_nsite'] = model.nsite
        result['model_ngeom'] = model.ngeom

        # Collect entities
        sites, bodies, geoms = {}, {}, {}
        for i in range(model.nsite):
            name = model.site(i).name
            if name:
                sid = model.site(name).id
                sites[name] = {
                    'id': int(sid), 'name': name,
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
                    'id': int(bid), 'name': name,
                    'parent_id': int(model.body_parentid[bid]),
                    'pos': [float(x) for x in model.body_pos[bid]],
                    'quat': [float(x) for x in model.body_quat[bid]],
                }
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name:
                gid = model.geom(name).id
                geoms[name] = {
                    'id': int(gid), 'name': name,
                    'body_id': int(model.geom_bodyid[gid]),
                    'pos': [float(x) for x in model.geom_pos[gid]],
                    'size': [float(x) for x in model.geom_size[gid]],
                    'type': int(model.geom_type[gid]),
                }

        result['entities'] = {
            'n_sites': len(sites), 'n_bodies': len(bodies), 'n_geoms': len(geoms),
            'site_names': sorted(sites.keys()),
            'body_names': sorted(bodies.keys()),
            'geom_names': sorted(geoms.keys()),
        }

        bddl_info = get_object_slices_for_task(suite, task_idx)
        if bddl_info is None:
            result['task_disposition'] = 'BDDL_UNAVAILABLE'
            result['status'] = 'FAIL'
            return result, 'BDDL unavailable'

        task_role = bddl_info['task_role']
        g_rels = task_role.get('goal_relations', [])
        result['goal_predicates'] = [(r[0], r[1], r[2]) for r in g_rels]

        relation_types = set(r[0] for r in g_rels)
        is_supported = bool({'In', 'On', 'Stack'} & relation_types)
        if not g_rels:
            result['task_disposition'] = 'ARTICULATED_UNSUPPORTED'
        elif is_supported:
            result['task_disposition'] = 'SUPPORTED_PLACEMENT'
        else:
            result['task_disposition'] = 'OTHER_RELATION_TYPE'

        # Resolve with C1-V2 priority
        relation_map = []
        alias_ledger = []
        n_exact = 0
        n_alias = 0
        n_blocked = 0
        n_unresolved = 0

        for pred, obj_name, target_name in g_rels:
            entry = {
                'predicate': pred,
                'object_bddl': obj_name,
                'target_bddl': target_name,
            }
            is_region = _is_region_target(target_name)

            # Priority 1: EXACT_SITE
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

            # Priority 2: EXACT_BODY (non-region only)
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

            # Priority 3: EXACT_GEOM
            elif target_name in geoms:
                entry['resolution'] = 'EXACT_GEOM'
                entry['entity_id'] = geoms[target_name]['id']
                entry['entity_type'] = 'geom'
                n_exact += 1

            # Priority 4: APPROVED_STRUCTURAL_ALIAS
            elif not is_region:
                alias_name = target_name + '_main'
                if alias_name in bodies:
                    alias_body = bodies[alias_name]
                    verified, verify_msg = _verify_alias_hierarchy(
                        target_name, alias_name, bodies, geoms, alias_body['id'])
                    if verified:
                        entry['resolution'] = 'APPROVED_STRUCTURAL_ALIAS'
                        entry['alias_rule'] = 'R1_main_suffix'
                        entry['alias_from'] = target_name
                        entry['alias_to'] = alias_name
                        entry['entity_id'] = alias_body['id']
                        entry['entity_type'] = 'body'
                        n_alias += 1
                        alias_ledger.append({
                            'rule': 'R1_main_suffix',
                            'target': target_name,
                            'alias': alias_name,
                            'entity_id': alias_body['id'],
                            'verification': verify_msg,
                        })
                    else:
                        entry['resolution'] = 'UNRESOLVED'
                        entry['alias_attempted'] = True
                        entry['alias_verification_failed'] = verify_msg
                        n_unresolved += 1
                else:
                    entry['resolution'] = 'UNRESOLVED'
                    if is_region:
                        entry['available_sites'] = sorted(sites.keys())[:10]
                    n_unresolved += 1

            # Priority 5: UNRESOLVED
            else:
                entry['resolution'] = 'UNRESOLVED'
                if is_region:
                    entry['available_sites'] = sorted(sites.keys())[:10]
                n_unresolved += 1

            # Object resolution
            if obj_name in bodies:
                entry['object_entity_type'] = 'body'
                entry['object_entity_id'] = bodies[obj_name]['id']
            elif obj_name + '_main' in bodies:
                entry['object_entity_type'] = 'body'
                entry['object_entity_id'] = bodies[obj_name + '_main']['id']
            elif obj_name in sites:
                entry['object_entity_type'] = 'site'
                entry['object_entity_id'] = sites[obj_name]['id']

            relation_map.append(entry)

        result['relation_map'] = relation_map
        result['alias_ledger'] = alias_ledger
        result['resolution_summary'] = {
            'n_total': len(relation_map),
            'n_exact': n_exact,
            'n_alias': n_alias,
            'n_blocked': n_blocked,
            'n_unresolved': n_unresolved,
            'n_ambiguous': 0,
        }

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
            try: env.close()
            except: pass


def compute_self_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, separators=(',', ':'),
                   ensure_ascii=False).encode('utf-8')
    ).hexdigest()


def main():
    print('=' * 60)
    print('T2R-C1-V2: 40-Task Entity Registry with Structural Alias')
    print('=' * 60)

    FOUR_SUITES = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']

    # Output directory — MUST be set before running
    out_dir = os.environ.get('C1_V2_OUT',
              '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_registry')
    per_task_dir = os.path.join(out_dir, 'per_task')
    os.makedirs(per_task_dir, exist_ok=True)

    all_results = {}
    summaries = []
    env_errors = 0
    blocked = 0
    unresolved = 0
    exact_total = 0
    alias_total = 0
    ambiguous_total = 0

    for suite in FOUR_SUITES:
        for task_idx in range(10):
            task_key = f'{suite}/task_{task_idx:02d}'
            print(f'{task_key}...', end=' ', flush=True)

            registry, error = build_registry_v2(suite, task_idx)
            all_results[task_key] = registry

            status = registry['status']
            disp = registry.get('task_disposition', '?')
            rs = registry.get('resolution_summary', {})
            n_e = rs.get('n_exact', 0)
            n_a = rs.get('n_alias', 0)
            n_b = rs.get('n_blocked', 0)
            n_u = rs.get('n_unresolved', 0)

            print(f'{status}  disp={disp}  exact={n_e} alias={n_a} blocked={n_b} unresolved={n_u}')

            if status == 'ENV_ERROR': env_errors += 1
            if n_b > 0: blocked += 1
            if n_u > 0: unresolved += 1
            if n_a > 0: alias_total += 1
            exact_total += n_e

            per_task = {
                'gate': 'T2R-C1-V2_PER_TASK_REGISTRY',
                'task_key': task_key,
                'version': 'C1-V2',
                'legacy': registry,
            }
            per_task['self_sha256'] = compute_self_hash(per_task)
            per_task_path = os.path.join(per_task_dir, f'{suite}_task_{task_idx:02d}.json')
            with open(per_task_path, 'w') as f:
                json.dump(per_task, f, indent=2, default=str)
            per_task_sha = hashlib.sha256(open(per_task_path, 'rb').read()).hexdigest()

            summaries.append({
                'task_key': task_key, 'status': status, 'disposition': disp,
                'n_relations': rs.get('n_total', 0),
                'n_exact': n_e, 'n_alias': n_a, 'n_blocked': n_b,
                'n_unresolved': n_u,
                'artifact_sha': per_task_sha,
            })

    n_ok = sum(1 for s in summaries if s['status'] == 'OK')
    n_supported = sum(1 for s in summaries if s['disposition'] == 'SUPPORTED_PLACEMENT')
    n_articulated = sum(1 for s in summaries if s['disposition'] == 'ARTICULATED_UNSUPPORTED')
    n_total_blocked = sum(s['n_blocked'] for s in summaries)
    n_total_unres = sum(s['n_unresolved'] for s in summaries)
    n_total_alias = sum(s['n_alias'] for s in summaries)

    print(f'\n{"=" * 60}')
    print(f'Tasks OK: {n_ok}/40')
    print(f'Supported placement: {n_supported}')
    print(f'Articulated unsupported: {n_articulated}')
    print(f'Total relations: {sum(s["n_relations"] for s in summaries)}')
    print(f'Total exact: {exact_total}')
    print(f'Total alias: {n_total_alias}')
    print(f'Total blocked: {n_total_blocked}')
    print(f'Total unresolved: {n_total_unres}')
    print(f'Total ambiguous: {ambiguous_total}')
    print(f'Environment errors: {env_errors}')

    all_pass = (
        env_errors == 0 and n_total_blocked == 0
        and n_total_unres == 0 and n_ok == 40
        and ambiguous_total == 0
    )

    summary = {
        'gate': 'T2R-C1-V2_FULL_ENTITY_REGISTRY',
        'version': 'C1-V2',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_tasks': 40, 'n_ok': n_ok, 'n_env_errors': env_errors,
        'n_blocked_resolutions': n_total_blocked,
        'n_unresolved': n_total_unres,
        'n_alias_resolutions': n_total_alias,
        'n_ambiguous': ambiguous_total,
        'n_supported_placement': n_supported,
        'n_articulated_unsupported': n_articulated,
        'resolution_priority': [
            '1. EXACT_SITE', '2. EXACT_BODY', '3. EXACT_GEOM',
            '4. APPROVED_STRUCTURAL_ALIAS', '5. UNRESOLVED'
        ],
        'approved_alias_rules': ['R1_main_suffix'],
        'per_task': summaries,
        'status': 'PASS' if all_pass else 'FAIL',
    }

    summary['self_sha256'] = compute_self_hash(summary)
    summary_path = os.path.join(out_dir, 'ENTITY_REGISTRY_V2_SUMMARY.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    summary_file_sha = hashlib.sha256(open(summary_path, 'rb').read()).hexdigest()

    # Alias ledger (separate artifact)
    all_aliases = []
    for task_key, reg in all_results.items():
        for al in reg.get('alias_ledger', []):
            all_aliases.append({'task_key': task_key, **al})
    alias_ledger_path = os.path.join(out_dir, 'ALIAS_LEDGER.json')
    with open(alias_ledger_path, 'w') as f:
        json.dump({'gate': 'C1-V2_ALIAS_LEDGER', 'n_aliases': len(all_aliases),
                   'aliases': all_aliases}, f, indent=2, default=str)

    print(f'\nSummary: {summary_path}')
    print(f'SHA: {summary_file_sha[:16]}...')
    print(f'Alias ledger: {alias_ledger_path}')

    if all_pass:
        print('\nT2R-C1-V2: PASS')
        sys.exit(0)
    else:
        print(f'\nT2R-C1-V2: FAIL')
        if env_errors > 0: print(f'  {env_errors} environment errors')
        if n_total_blocked > 0: print(f'  {n_total_blocked} blocked')
        if n_total_unres > 0: print(f'  {n_total_unres} unresolved')
        sys.exit(5)


if __name__ == '__main__':
    main()
