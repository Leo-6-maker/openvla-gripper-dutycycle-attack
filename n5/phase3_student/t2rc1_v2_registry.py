"""T2R-C1-V2: 40-task MuJoCo Entity Registry with Structural Alias Resolution.

Resolution priority (strict, no fallback, per-entity):
  1. EXACT_SITE       — name matches site
  2. EXACT_BODY       — name matches body (forbidden if is_region)
  3. EXACT_GEOM       — name matches geom (forbidden if is_region)
  4. APPROVED_STRUCTURAL_ALIAS — explicit, unique, structure-verified alias
  5. UNRESOLVED       — everything else

AMBIGUOUS: name matches multiple entity types, or alias has multiple candidates.
Region/object roles are derived from BDDL declarations, not MuJoCo name availability.

FORBIDDEN:
  - STRIP_SUFFIX_BODY, STRIP_SUFFIX_SITE, SUBSTRING
  - region → body/geom fallback
  - multi-candidate tie-breaking
  - unverified alias acceptance
"""
import json, os, sys, time, hashlib
from collections import defaultdict

# ── Resolution enum ──
VALID_RESOLUTIONS = frozenset({
    'EXACT_SITE', 'EXACT_BODY', 'EXACT_GEOM', 'APPROVED_STRUCTURAL_ALIAS'
})
BLOCKED_RESOLUTIONS = frozenset({
    'STRIP_SUFFIX_BODY', 'STRIP_SUFFIX_SITE', 'SUBSTRING'
})
TERMINAL_RESOLUTIONS = frozenset({'UNRESOLVED', 'AMBIGUOUS', 'ENV_ERROR'})


# ── Core: role-safe structural entity resolver ──

MANIPULATED_OBJECT = 'MANIPULATED_OBJECT'
OBJECT_TARGET = 'OBJECT_TARGET'
REGION_TARGET = 'REGION_TARGET'
ENTITY_ROLES = frozenset({MANIPULATED_OBJECT, OBJECT_TARGET, REGION_TARGET})


def _body_descends_from(body_id, root_body_id, bodies):
    """Return True when body_id is root_body_id or a descendant of it."""
    id_to_parent = {
        int(info['id']): int(info.get('parent_id', -1))
        for info in bodies.values() if 'id' in info
    }
    current = int(body_id)
    root = int(root_body_id)
    visited = set()
    while current not in visited:
        if current == root:
            return True
        visited.add(current)
        if current not in id_to_parent:
            return False
        parent = id_to_parent[current]
        if parent < 0 or parent == current:
            return False
        current = parent
    return False


def _verify_alias_hierarchy_full(target_name, alias_body_name, bodies, geoms,
                                 target_body_id):
    """Verify every matching geom belongs to the accepted root body ancestry."""
    detail = {'alias_body': alias_body_name, 'target': target_name}
    expected = target_name + '_main'
    if alias_body_name != expected:
        detail['error'] = f'body_name_mismatch_expected_{expected}'
        return False, detail

    geom_prefix = target_name + '_g'
    matching_geoms = {
        gn: g for gn, g in geoms.items() if gn.startswith(geom_prefix)
    }
    if not matching_geoms:
        detail['error'] = 'no_geoms'
        return False, detail

    mismatched = []
    descendant = 0
    direct = 0
    for gn, geom in matching_geoms.items():
        body_id = int(geom['body_id'])
        if body_id == int(target_body_id):
            direct += 1
        if _body_descends_from(body_id, target_body_id, bodies):
            descendant += 1
        else:
            mismatched.append(gn)

    detail['total_geoms'] = len(matching_geoms)
    detail['direct_geoms'] = direct
    detail['descendant_geoms'] = descendant
    detail['mismatched_geoms'] = len(mismatched)
    if mismatched:
        detail['error'] = (
            f'geom_outside_alias_ancestry_{len(mismatched)}_of_'
            f'{len(matching_geoms)}'
        )
        detail['mismatched_sample'] = mismatched[:5]
        return False, detail
    return True, detail


def _result(name, role, resolution, **fields):
    result = {'name': name, 'semantic_role': role, 'resolution': resolution}
    result.update(fields)
    return result


def resolve_entity(name, semantic_role, sites, bodies, geoms):
    """Resolve one BDDL operand under an independently supplied semantic role."""
    if semantic_role not in ENTITY_ROLES:
        raise ValueError(f'unknown semantic role: {semantic_role}')

    in_sites = name in sites
    in_bodies = name in bodies
    in_geoms = name in geoms
    exact_types = [
        entity_type for entity_type, present in
        [('site', in_sites), ('body', in_bodies), ('geom', in_geoms)]
        if present
    ]
    if len(exact_types) > 1:
        return _result(
            name, semantic_role, 'AMBIGUOUS',
            error_detail={'reason': 'exact_cross_type_conflict',
                          'entity_types': exact_types},
        )

    if semantic_role == REGION_TARGET:
        if in_sites:
            site = sites[name]
            parent_body_name = None
            for body_name, body in bodies.items():
                if int(body['id']) == int(site['body_id']):
                    parent_body_name = body_name
                    break
            return _result(
                name, semantic_role, 'EXACT_SITE',
                entity_type='site', entity_id=site['id'],
                size=site.get('size'), parent_body_id=site['body_id'],
                parent_body_name=parent_body_name,
            )
        if in_bodies:
            return _result(
                name, semantic_role, 'BLOCKED_REGION_AS_BODY',
                error_detail={'reason': 'declared_region_site_missing'},
            )
        if in_geoms:
            return _result(
                name, semantic_role, 'BLOCKED_REGION_AS_GEOM',
                error_detail={'reason': 'declared_region_site_missing'},
            )
        return _result(
            name, semantic_role, 'UNRESOLVED',
            error_detail={'reason': 'declared_region_site_missing'},
        )

    # Both manipulated objects and object targets are forbidden from sites.
    if in_sites:
        return _result(
            name, semantic_role, 'BLOCKED_OBJECT_AS_SITE',
            error_detail={'reason': 'object_operand_resolved_to_site'},
        )
    if in_bodies:
        body = bodies[name]
        return _result(
            name, semantic_role, 'EXACT_BODY',
            entity_type='body', entity_id=body['id'],
        )
    if in_geoms:
        geom = geoms[name]
        return _result(
            name, semantic_role, 'EXACT_GEOM',
            entity_type='geom', entity_id=geom['id'],
        )

    alias_name = name + '_main'
    if alias_name in sites or alias_name in geoms:
        conflicts = []
        if alias_name in sites:
            conflicts.append('site')
        if alias_name in geoms:
            conflicts.append('geom')
        if alias_name in bodies:
            conflicts.append('body')
        return _result(
            name, semantic_role, 'AMBIGUOUS',
            error_detail={'reason': 'alias_cross_type_conflict',
                          'alias': alias_name, 'entity_types': conflicts},
        )
    if alias_name in bodies:
        alias_body = bodies[alias_name]
        verified, detail = _verify_alias_hierarchy_full(
            name, alias_name, bodies, geoms, alias_body['id'])
        if verified:
            return _result(
                name, semantic_role, 'APPROVED_STRUCTURAL_ALIAS',
                entity_type='body', entity_id=alias_body['id'],
                alias_rule='R1_main_suffix', alias_from=name,
                alias_to=alias_name, alias_verification=detail,
            )
        return _result(
            name, semantic_role, 'UNRESOLVED',
            error_detail={'reason': 'alias_verification_failed',
                          'alias_attempted': alias_name,
                          'verification': detail},
        )

    return _result(name, semantic_role, 'UNRESOLVED',
                   error_detail={'reason': 'no_role_compatible_entity'})


def _target_semantic_role(target_name, bddl_object_names):
    """Derive target role from BDDL declarations, independently of MuJoCo."""
    return OBJECT_TARGET if target_name in bddl_object_names else REGION_TARGET


def resolve_relation(pred, obj_name, target_name, bddl_object_names,
                     sites, bodies, geoms):
    """Resolve a complete relation with BDDL-derived operand roles."""
    object_names = set(bddl_object_names)
    if obj_name not in object_names:
        obj_res = _result(
            obj_name, MANIPULATED_OBJECT, 'UNRESOLVED',
            error_detail={'reason': 'manipulated_object_not_declared_in_bddl'},
        )
    else:
        obj_res = resolve_entity(
            obj_name, MANIPULATED_OBJECT, sites, bodies, geoms)

    target_role = _target_semantic_role(target_name, object_names)
    tgt_res = resolve_entity(target_name, target_role, sites, bodies, geoms)

    obj_ok = obj_res['resolution'] in VALID_RESOLUTIONS
    tgt_ok = tgt_res['resolution'] in VALID_RESOLUTIONS
    obj_ambiguous = obj_res['resolution'] == 'AMBIGUOUS'
    tgt_ambiguous = tgt_res['resolution'] == 'AMBIGUOUS'
    obj_blocked = obj_res['resolution'].startswith('BLOCKED_')
    tgt_blocked = tgt_res['resolution'].startswith('BLOCKED_')

    return {
        'predicate': pred,
        'object_bddl': obj_name,
        'target_bddl': target_name,
        'object_semantic_role': MANIPULATED_OBJECT,
        'target_semantic_role': target_role,
        'target_is_region': target_role == REGION_TARGET,
        'object_resolution': obj_res,
        'target_resolution': tgt_res,
        'relation_ok': obj_ok and tgt_ok,
        'object_blocked': obj_blocked,
        'target_blocked': tgt_blocked,
        'object_ambiguous': obj_ambiguous,
        'target_ambiguous': tgt_ambiguous,
    }


def summarize_relation_resolutions(relations, task_key):
    """Aggregate production counters and alias ledger from relation outputs."""
    counts = {
        'n_relations': len(relations),
        'object_ok': 0, 'object_unresolved': 0,
        'object_ambiguous': 0, 'object_blocked': 0,
        'target_ok': 0, 'target_unresolved': 0,
        'target_ambiguous': 0, 'target_blocked': 0,
    }
    alias_ledger = []
    for rel in relations:
        for role_key, count_prefix, source_name in [
            ('object_resolution', 'object', rel['object_bddl']),
            ('target_resolution', 'target', rel['target_bddl']),
        ]:
            resolved = rel[role_key]
            resolution = resolved['resolution']
            if resolution in VALID_RESOLUTIONS:
                counts[f'{count_prefix}_ok'] += 1
                if resolution == 'APPROVED_STRUCTURAL_ALIAS':
                    alias_ledger.append({
                        'task_key': task_key,
                        'entity_role': count_prefix,
                        'semantic_role': resolved['semantic_role'],
                        'rule': resolved['alias_rule'],
                        'target': source_name,
                        'alias': resolved['alias_to'],
                        'entity_id': resolved['entity_id'],
                        'verification': resolved['alias_verification'],
                    })
            elif resolution.startswith('BLOCKED_'):
                counts[f'{count_prefix}_blocked'] += 1
            elif resolution == 'AMBIGUOUS':
                counts[f'{count_prefix}_ambiguous'] += 1
            else:
                counts[f'{count_prefix}_unresolved'] += 1
    return counts, alias_ledger


# ── Full registry builder (MuJoCo-dependent) ──

def build_registry_v2(suite, task_idx):
    """Build entity registry with structural alias resolution."""
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'phase2_labels'))
    from v22_production_v2 import get_object_slices_for_task

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
        bddl_object_names = set(bddl_info.get('object_slices', {}).keys())
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

        # Resolve all relations with roles derived from BDDL declarations.
        relations = [
            resolve_relation(pred, obj_name, target_name, bddl_object_names,
                             sites, bodies, geoms)
            for pred, obj_name, target_name in g_rels
        ]
        counts, alias_ledger = summarize_relation_resolutions(
            relations, result['task_key'])

        result['relations'] = relations
        result['alias_ledger'] = alias_ledger
        result['resolution_counts'] = counts

        # Determine status
        if counts['object_blocked'] > 0 or counts['target_blocked'] > 0:
            result['status'] = 'BLOCKED_RESOLUTION_PRESENT'
        elif counts['object_unresolved'] > 0 or counts['target_unresolved'] > 0:
            result['status'] = 'UNRESOLVED_TARGET_PRESENT'
        elif counts['object_ambiguous'] > 0 or counts['target_ambiguous'] > 0:
            result['status'] = 'AMBIGUOUS_RESOLUTION_PRESENT'
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
    out_dir = os.environ.get('C1_V2_OUT',
              '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_registry')
    per_task_dir = os.path.join(out_dir, 'per_task')
    os.makedirs(per_task_dir, exist_ok=True)

    all_results = {}
    summaries = []
    totals = {'env_errors': 0, 'blocked': 0,
              'object_unresolved': 0, 'object_ambiguous': 0, 'object_ok': 0,
              'target_unresolved': 0, 'target_ambiguous': 0, 'target_ok': 0}

    for suite in FOUR_SUITES:
        for task_idx in range(10):
            task_key = f'{suite}/task_{task_idx:02d}'
            print(f'{task_key}...', end=' ', flush=True)

            registry, error = build_registry_v2(suite, task_idx)
            all_results[task_key] = registry

            status = registry['status']
            disp = registry.get('task_disposition', '?')
            rc = registry.get('resolution_counts', {})
            print(f'{status}  disp={disp}  '
                  f'o_ok={rc.get("object_ok",0)} o_un={rc.get("object_unresolved",0)} o_amb={rc.get("object_ambiguous",0)} '
                  f't_ok={rc.get("target_ok",0)} t_un={rc.get("target_unresolved",0)} t_amb={rc.get("target_ambiguous",0)}')

            if status == 'ENV_ERROR': totals['env_errors'] += 1
            for k in ['object_ok', 'object_unresolved', 'object_ambiguous',
                       'target_ok', 'target_unresolved', 'target_ambiguous']:
                totals[k] += rc.get(k, 0)
            if rc.get('object_blocked', 0) > 0 or rc.get('target_blocked', 0) > 0:
                totals['blocked'] += 1

            per_task = {
                'gate': 'T2R-C1-V2_PER_TASK_REGISTRY',
                'task_key': task_key, 'version': 'C1-V2',
                'legacy': registry,
            }
            per_task['self_sha256'] = compute_self_hash(per_task)
            per_task_path = os.path.join(per_task_dir, f'{suite}_task_{task_idx:02d}.json')
            with open(per_task_path, 'w') as f:
                json.dump(per_task, f, indent=2, default=str)
            per_task_sha = hashlib.sha256(open(per_task_path, 'rb').read()).hexdigest()

            summaries.append({
                'task_key': task_key, 'status': status, 'disposition': disp,
                'n_relations': rc.get('n_relations', 0),
                'object_ok': rc.get('object_ok', 0),
                'object_unresolved': rc.get('object_unresolved', 0),
                'object_ambiguous': rc.get('object_ambiguous', 0),
                'target_ok': rc.get('target_ok', 0),
                'target_unresolved': rc.get('target_unresolved', 0),
                'target_ambiguous': rc.get('target_ambiguous', 0),
                'artifact_sha': per_task_sha,
            })

    n_ok = sum(1 for s in summaries if s['status'] == 'OK')
    n_supported = sum(1 for s in summaries if s['disposition'] == 'SUPPORTED_PLACEMENT')
    n_articulated = sum(1 for s in summaries if s['disposition'] == 'ARTICULATED_UNSUPPORTED')

    print(f'\n{"=" * 60}')
    print(f'Tasks OK: {n_ok}/40')
    print(f'Supported placement: {n_supported}  Articulated: {n_articulated}')
    print(f'Object:  OK={totals["object_ok"]}  unresolved={totals["object_unresolved"]}  ambiguous={totals["object_ambiguous"]}')
    print(f'Target:  OK={totals["target_ok"]}  unresolved={totals["target_unresolved"]}  ambiguous={totals["target_ambiguous"]}')
    print(f'Blocked tasks: {totals["blocked"]}  Env errors: {totals["env_errors"]}')

    all_pass = (
        totals['env_errors'] == 0 and totals['blocked'] == 0
        and totals['object_unresolved'] == 0 and totals['object_ambiguous'] == 0
        and totals['target_unresolved'] == 0 and totals['target_ambiguous'] == 0
        and n_ok == 40
    )

    summary = {
        'gate': 'T2R-C1-V2_FULL_ENTITY_REGISTRY',
        'version': 'C1-V2',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'n_tasks': 40, 'n_ok': n_ok,
        'n_env_errors': totals['env_errors'],
        'n_blocked': totals['blocked'],
        'object_ok': totals['object_ok'],
        'object_unresolved': totals['object_unresolved'],
        'object_ambiguous': totals['object_ambiguous'],
        'target_ok': totals['target_ok'],
        'target_unresolved': totals['target_unresolved'],
        'target_ambiguous': totals['target_ambiguous'],
        'n_supported_placement': n_supported,
        'n_articulated_unsupported': n_articulated,
        'resolution_priority': [
            '1. EXACT_SITE', '2. EXACT_BODY', '3. EXACT_GEOM',
            '4. APPROVED_STRUCTURAL_ALIAS', '5. UNRESOLVED'
        ],
        'approved_alias_rules': ['R1_main_suffix'],
        'region_detection': 'structural — name in model.site_names, NOT suffix-based',
        'per_task': summaries,
        'status': 'PASS' if all_pass else 'FAIL',
    }

    summary['self_sha256'] = compute_self_hash(summary)
    summary_path = os.path.join(out_dir, 'ENTITY_REGISTRY_V2_SUMMARY.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    alias_ledger_path = os.path.join(out_dir, 'ALIAS_LEDGER.json')
    all_aliases = []
    for task_key, reg in all_results.items():
        for al in reg.get('alias_ledger', []):
            all_aliases.append(al)
    with open(alias_ledger_path, 'w') as f:
        json.dump({'gate': 'C1-V2_ALIAS_LEDGER', 'n_aliases': len(all_aliases),
                   'aliases': all_aliases}, f, indent=2, default=str)

    print(f'\nSummary: {summary_path}')
    print(f'Alias ledger: {alias_ledger_path}')

    if all_pass:
        print('\nT2R-C1-V2: PASS')
        sys.exit(0)
    else:
        print(f'\nT2R-C1-V2: FAIL')
        sys.exit(5)


if __name__ == '__main__':
    main()
