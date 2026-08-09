"""T2R-B: Geometry-Based Relation Resolver.

Implements In/On/Stack checks using object_slices positions and sizes.
No episode_success. No terminal-only shortcuts.
Each relation returns: satisfied (bool), confidence (float), evidence dict.
"""
import numpy as np
from v22_production_v2 import _slice_vector, _dist, _finite_vector


def _estimate_half_size(obj_name):
    """Estimate object half-size from name when sidecar has no size data.

    Sidecar object_state only stores pos+quat, not size/bbox.
    Use name-based heuristics as fallback.
    """
    name_lower = obj_name.lower()
    # Large containers
    if any(k in name_lower for k in ['basket', 'bin', 'box', 'cabinet', 'drawer']):
        return np.array([0.18, 0.18, 0.12])
    # Medium containers/surfaces
    if any(k in name_lower for k in ['plate', 'tray', 'rack']):
        return np.array([0.12, 0.12, 0.02])
    # Bowls
    if 'bowl' in name_lower:
        return np.array([0.08, 0.08, 0.05])
    # Stove/microwave
    if any(k in name_lower for k in ['stove', 'microwave', 'caddy']):
        return np.array([0.15, 0.15, 0.10])
    # Books/boxes
    if any(k in name_lower for k in ['book', 'chocolate']):
        return np.array([0.06, 0.04, 0.02])
    # Mugs/cups
    if any(k in name_lower for k in ['mug', 'cup']):
        return np.array([0.04, 0.04, 0.05])
    # Small food/objects
    return np.array([0.03, 0.03, 0.03])


def _get_object_aabb(steps_data, t, object_slices, obj_name):
    """Get axis-aligned bounding box for an object at step t.
    Returns (center_3d, half_size_3d) or (None, None)."""
    spec = object_slices.get(obj_name)
    if spec is None:
        return None, None
    pos = _slice_vector(steps_data[t].get('object_state', []), spec, 'pos')
    if pos is None:
        return None, None
    # Sidecar has no size data — use name-based estimate
    half = _estimate_half_size(obj_name)
    return np.array(pos), half


def check_On(obj_pos, obj_size_half, support_pos, support_size_half, tolerance=0.02):
    """Check if object is ON a support surface.

    Geometry:
      - Object bottom (Z - half_z) >= support top (Z + half_z) - tolerance
      - Object bottom (Z - half_z) <= support top (Z + half_z) + tolerance (not floating)
      - XY footprint overlap: object XY AABB overlaps support XY AABB
    """
    obj_bottom = obj_pos[2] - obj_size_half[2]
    support_top = support_pos[2] + support_size_half[2]

    # Object bottom must be near support top
    on_surface = abs(obj_bottom - support_top) < tolerance * 2

    # XY overlap
    obj_x_min = obj_pos[0] - obj_size_half[0]
    obj_x_max = obj_pos[0] + obj_size_half[0]
    obj_y_min = obj_pos[1] - obj_size_half[1]
    obj_y_max = obj_pos[1] + obj_size_half[1]

    sup_x_min = support_pos[0] - support_size_half[0]
    sup_x_max = support_pos[0] + support_size_half[0]
    sup_y_min = support_pos[1] - support_size_half[1]
    sup_y_max = support_pos[1] + support_size_half[1]

    xy_overlap = (obj_x_max > sup_x_min and obj_x_min < sup_x_max and
                  obj_y_max > sup_y_min and obj_y_min < sup_y_max)

    satisfied = on_surface and xy_overlap
    conf = 0.75 if (on_surface and xy_overlap) else (0.4 if on_surface else 0.0)

    return satisfied, conf, {
        'obj_bottom_z': float(obj_bottom),
        'support_top_z': float(support_top),
        'z_diff': float(obj_bottom - support_top),
        'on_surface': on_surface,
        'xy_overlap': xy_overlap,
    }


def check_In(obj_pos, obj_size_half, container_pos, container_size_half, tolerance=0.03):
    """Check if object is IN a container.

    Geometry:
      - Object center XY within container XY bounds (expanded by tolerance)
      - Object bottom Z >= container bottom Z - tolerance (not below container)
      - Object bottom Z <= container top Z + tolerance (not above container)
    """
    container_half = container_size_half + tolerance

    xy_contained = (abs(obj_pos[0] - container_pos[0]) < container_half[0] and
                    abs(obj_pos[1] - container_pos[1]) < container_half[1])

    obj_bottom = obj_pos[2] - obj_size_half[2]
    container_bottom = container_pos[2] - container_size_half[2]
    container_top = container_pos[2] + container_size_half[2]

    z_in_range = (obj_bottom >= container_bottom - tolerance and
                  obj_bottom <= container_top + tolerance)

    satisfied = xy_contained and z_in_range
    conf = 0.70 if satisfied else (0.35 if xy_contained else 0.0)

    return satisfied, conf, {
        'xy_contained': xy_contained,
        'z_in_range': z_in_range,
        'obj_bottom': float(obj_bottom),
        'container_bottom': float(container_bottom),
        'container_top': float(container_top),
    }


def check_Stack(obj_pos, obj_size_half, below_pos, below_size_half, tolerance=0.02):
    """Check if object is STACKED on another object.

    Similar to On but for object-on-object:
      - obj bottom near below_obj top
      - XY overlap
      - obj must be above below_obj (not inside)
    """
    obj_bottom = obj_pos[2] - obj_size_half[2]
    below_top = below_pos[2] + below_size_half[2]

    on_top = abs(obj_bottom - below_top) < tolerance * 2

    # XY footprint overlap
    obj_x_min = obj_pos[0] - obj_size_half[0]
    obj_x_max = obj_pos[0] + obj_size_half[0]
    obj_y_min = obj_pos[1] - obj_size_half[1]
    obj_y_max = obj_pos[1] + obj_size_half[1]

    blw_x_min = below_pos[0] - below_size_half[0]
    blw_x_max = below_pos[0] + below_size_half[0]
    blw_y_min = below_pos[1] - below_size_half[1]
    blw_y_max = below_pos[1] + below_size_half[1]

    xy_overlap = (obj_x_max > blw_x_min and obj_x_min < blw_x_max and
                  obj_y_max > blw_y_min and obj_y_min < blw_y_max)

    satisfied = on_top and xy_overlap
    conf = 0.65 if satisfied else 0.0

    return satisfied, conf, {
        'obj_bottom_z': float(obj_bottom),
        'below_top_z': float(below_top),
        'on_top': on_top,
        'xy_overlap': xy_overlap,
    }


def evaluate_goal_relation(steps_data, t, relation, object_slices):
    """Evaluate a single BDDL goal relation at step t.

    Args:
        steps_data: sidecar step list
        t: step index
        relation: (predicate, object_name, target_name) tuple
        object_slices: dict of {name: spec}

    Returns: (satisfied, confidence, evidence_dict)
    """
    pred, obj_name, target_name = relation

    # Resolve names to slice keys (same logic as T2R-A)
    obj_key = target_key = None
    obj_method = tgt_method = 'NONE'

    # Direct match
    if obj_name in object_slices:
        obj_key = obj_name; obj_method = 'DIRECT'
    if target_name in object_slices:
        target_key = target_name; tgt_method = 'DIRECT'

    # Strip suffix for target
    if target_key is None:
        for suffix in ['_contain_region', '_init_region', '_cook_region',
                       '_heating_region', '_top_region', '_front_region']:
            base = target_name.replace(suffix, '')
            if base in object_slices:
                target_key = base; tgt_method = 'STRIP_SUFFIX'
                break

    # Substring fallback for target
    if target_key is None:
        for key in sorted(object_slices.keys()):
            k_clean = key.replace('_contain_region', '').replace('_init_region', '')
            t_clean = target_name.replace('_contain_region', '').replace('_init_region', '')
            if k_clean in t_clean or t_clean in k_clean:
                target_key = key; tgt_method = 'SUBSTRING'
                break

    if obj_key is None or target_key is None:
        return False, 0.0, {'error': 'UNRESOLVED_NAMES',
                            'obj_resolved': obj_key, 'tgt_resolved': target_key}

    obj_pos, obj_half = _get_object_aabb(steps_data, t, object_slices, obj_key)
    tgt_pos, tgt_half = _get_object_aabb(steps_data, t, object_slices, target_key)

    if obj_pos is None or tgt_pos is None:
        return False, 0.0, {'error': 'NO_POSITION_DATA'}

    if pred in ('On', 'OnContainer'):
        satisfied, conf, evidence = check_On(obj_pos, obj_half, tgt_pos, tgt_half)
    elif pred in ('In', 'InContainer', 'Inside'):
        satisfied, conf, evidence = check_In(obj_pos, obj_half, tgt_pos, tgt_half)
    elif pred == 'Stack':
        satisfied, conf, evidence = check_Stack(obj_pos, obj_half, tgt_pos, tgt_half)
    else:
        return False, 0.0, {'error': f'UNKNOWN_PREDICATE:{pred}'}

    evidence['obj_resolved'] = obj_key
    evidence['obj_method'] = obj_method
    evidence['tgt_resolved'] = target_key
    evidence['tgt_method'] = tgt_method
    evidence['predicate'] = pred

    return satisfied, conf, evidence


def evaluate_all_relations(steps_data, t, goal_relations, object_slices):
    """Evaluate all goal relations at step t. Returns list of results."""
    results = []
    for rel in goal_relations:
        satisfied, conf, evidence = evaluate_goal_relation(
            steps_data, t, rel, object_slices)
        results.append({
            'relation': rel,
            'satisfied': satisfied,
            'confidence': conf,
            'evidence': evidence,
        })
    return results
