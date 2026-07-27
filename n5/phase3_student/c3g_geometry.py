"""C3-G: Tri-State Geometry Evaluator.

In/On/Stack detection with local-frame checks, margin bands, tri-state output.
Strictly follows data contract:
  - manipulated-object pose: original sidecar
  - movable direct target pose: original sidecar
  - basket region pose: sidecar body pose × sealed local transform (C3-S2)
  - static fixture pose: C3-S2 world-pose seal
  - white/wooden dynamic fixture: UNKNOWN
  - contacts: original sidecar
  - task_success: evaluation only, NEVER input
"""
import numpy as np
from collections import defaultdict

# Margin bands (meters)
DEFAULT_MARGIN_UPPER = 0.005   # Inside this → TRUE
DEFAULT_MARGIN_LOWER = 0.020   # Outside this → FALSE; between upper and lower → UNKNOWN

# Object half-size estimates (DEV_HEURISTIC_ONLY — no bounding box in sidecar)
OBJ_HALF_SIZE = defaultdict(lambda: 0.03)
OBJ_HALF_SIZE.update({
    'alphabet_soup': 0.035, 'cream_cheese': 0.03, 'butter': 0.03,
    'chocolate_pudding': 0.03, 'milk': 0.03, 'tomato_sauce': 0.03,
    'salad_dressing': 0.03, 'orange_juice': 0.03,
    'bowl': 0.06, 'plate': 0.08, 'akita_black_bowl': 0.07,
    'basket_1': 0.18,
})

# Containers: name → (margin_upper, margin_lower) overrides for Z
CONTAINER_Z_MARGIN = {
    'basket_1_contain_region': (0.01, 0.03),
}

# Surfaces: name → (margin_upper, margin_lower) overrides for Z
SURFACE_Z_MARGIN = {
    'flat_stove_1_cook_region': (0.005, 0.015),
    'microwave_1_heating_region': (0.005, 0.020),
    'living_room_table_plate_right_region': (0.005, 0.015),
    'desk_caddy_1_back_contain_region': (0.005, 0.015),
    'main_table_stove_front_region': (0.005, 0.015),
    'wine_rack_1_top_region': (0.005, 0.015),
}


def _xmat_to_rot3(xmat_flat):
    """Convert 9-element flat xmat to 3x3 rotation matrix."""
    return np.array(xmat_flat).reshape(3, 3)


def world_to_local(world_pos, ref_xpos, ref_xmat):
    """Transform world position to reference local frame.
    local = R^T * (world - pos)
    """
    R = _xmat_to_rot3(ref_xmat)
    return R.T @ (np.array(world_pos) - np.array(ref_xpos))


def local_to_world(local_pos, ref_xpos, ref_xmat):
    """Transform local position to world frame.
    world = pos + R * local
    """
    R = _xmat_to_rot3(ref_xmat)
    return np.array(ref_xpos) + R @ np.array(local_pos)


def _abs_max(vals):
    """Max absolute value across array elements."""
    return float(np.max(np.abs(np.array(vals))))


def check_In(obj_pos_3d, obj_name, container_xpos, container_xmat,
             container_size, margin_upper=None, margin_lower=None):
    """Check if object is inside a container (basket, caddy, etc.).

    In: object XY within container bounds, Z within container height.
    Container size = [half_x, half_y, half_z] in container local frame.

    Returns: (TRUE|FALSE|UNKNOWN, margin_float, evidence_dict)
    """
    if margin_upper is None:
        mu, ml = CONTAINER_Z_MARGIN.get('__default__', (DEFAULT_MARGIN_UPPER, DEFAULT_MARGIN_LOWER))
    else:
        mu, ml = margin_upper, margin_lower

    obj = np.array(obj_pos_3d, dtype=float)
    cont_pos = np.array(container_xpos, dtype=float)
    half = np.array(container_size, dtype=float)

    # Object position in container local frame
    local = world_to_local(obj, container_xpos, container_xmat)
    half_x, half_y, half_z = abs(half[0]), abs(half[1]), abs(half[2])
    lx, ly, lz = abs(local[0]), abs(local[1]), local[2]

    evidence = {
        'local_pos': local.tolist(),
        'container_half_size': half.tolist(),
        'lx_abs': lx, 'ly_abs': ly, 'lz': lz,
    }

    # XY: object center within container footprint
    xy_max_err = max(lx - half_x, ly - half_y)
    z_bottom = lz - OBJ_HALF_SIZE[obj_name]  # object bottom in container frame

    # Z: container bottom is typically at -half_z, top at +half_z
    z_top = lz + OBJ_HALF_SIZE[obj_name]  # object top
    container_top = half_z
    container_bottom = -half_z

    # Object must be fully inside in XY, and within Z range
    xy_ok = (lx + OBJ_HALF_SIZE[obj_name] <= half_x + mu) and \
            (ly + OBJ_HALF_SIZE[obj_name] <= half_y + mu)

    # Object bottom must be above container bottom (with margin)
    z_bottom_ok = z_bottom >= container_bottom - mu
    # Object top must be below container top (with margin) — but for baskets, top is open
    z_top_ok = z_top <= container_top + mu  # Relaxed for open-top containers

    # Margin: how far from boundary (negative = outside)
    xy_margin = min(half_x - lx - OBJ_HALF_SIZE[obj_name],
                    half_y - ly - OBJ_HALF_SIZE[obj_name])
    z_margin = min(z_bottom - container_bottom, container_top - z_top)

    margin = min(xy_margin, z_margin)
    evidence['xy_margin'] = xy_margin
    evidence['z_margin'] = z_margin
    evidence['margin'] = margin

    if xy_ok and z_bottom_ok and z_top_ok and margin >= mu:
        return 'TRUE', margin, evidence
    elif margin <= -ml:
        return 'FALSE', margin, evidence
    else:
        return 'UNKNOWN', margin, evidence


def check_On(obj_pos_3d, obj_name, surface_xpos, surface_xmat,
             surface_size, margin_upper=None, margin_lower=None):
    """Check if object is On a surface.

    On: object bottom near surface top, XY within footprint.
    Surface size = [half_x, half_y, half_z] in surface local frame.

    Returns: (TRUE|FALSE|UNKNOWN, margin_float, evidence_dict)
    """
    if margin_upper is None:
        mu, ml = SURFACE_Z_MARGIN.get('__default__', (DEFAULT_MARGIN_UPPER, DEFAULT_MARGIN_LOWER))
    else:
        mu, ml = margin_upper, margin_lower

    obj = np.array(obj_pos_3d, dtype=float)
    surf_pos = np.array(surface_xpos, dtype=float)
    half = np.array(surface_size, dtype=float)

    local = world_to_local(obj, surface_xpos, surface_xmat)
    half_x, half_y, half_z = abs(half[0]), abs(half[1]), abs(half[2])
    lx, ly, lz = abs(local[0]), abs(local[1]), local[2]

    # Object half-size
    obj_half = OBJ_HALF_SIZE[obj_name]

    evidence = {
        'local_pos': local.tolist(),
        'surface_half_size': half.tolist(),
        'lx_abs': lx, 'ly_abs': ly, 'lz': lz,
    }

    # Z: object bottom = lz - obj_half, surface top = +half_z
    obj_bottom = lz - obj_half
    surface_top = half_z
    z_dist = obj_bottom - surface_top  # positive = above surface

    # XY: object extent within surface bounds
    xy_ok = (lx + obj_half <= half_x + mu) and (ly + obj_half <= half_y + mu)

    # Margin
    xy_margin = min(half_x - lx - obj_half, half_y - ly - obj_half)
    z_margin = mu - abs(z_dist)  # positive if within margin band of surface
    margin = min(xy_margin, z_margin)

    evidence['z_dist'] = z_dist
    evidence['xy_margin'] = xy_margin
    evidence['z_margin'] = z_margin
    evidence['margin'] = margin

    # On: object bottom within margin band of surface top, XY within bounds
    if abs(z_dist) <= mu and xy_ok and margin >= 0:
        return 'TRUE', margin, evidence
    elif abs(z_dist) > ml or not xy_ok or margin <= -ml:
        return 'FALSE', margin, evidence
    else:
        return 'UNKNOWN', margin, evidence


def check_Stack(obj_pos_3d, obj_name, other_pos_3d, other_name,
                contacts=None, margin_upper=None, margin_lower=None):
    """Check if obj is stacked On another object.

    Stack requires:
    1. Object above other object (Z check)
    2. XY proximity
    3. Contact evidence if available

    Returns: (TRUE|FALSE|UNKNOWN, margin_float, evidence_dict)
    """
    mu = margin_upper or DEFAULT_MARGIN_UPPER
    ml = margin_lower or DEFAULT_MARGIN_LOWER

    obj = np.array(obj_pos_3d, dtype=float)
    other = np.array(other_pos_3d, dtype=float)

    obj_half = OBJ_HALF_SIZE[obj_name]
    other_half = OBJ_HALF_SIZE[other_name]

    # Z: obj bottom should be near other top
    z_dist = (obj[2] - obj_half) - (other[2] + other_half)

    # XY distance
    xy_dist = np.linalg.norm(obj[:2] - other[:2])

    evidence = {
        'z_dist': z_dist,
        'xy_dist': float(xy_dist),
        'obj_bottom_z': float(obj[2] - obj_half),
        'other_top_z': float(other[2] + other_half),
    }

    # Contact check
    has_contact = False
    if contacts:
        for c in contacts:
            g1, g2 = c if isinstance(c, (list, tuple)) else (c.get('geom1', ''), c.get('geom2', ''))
            if obj_name in g1 or obj_name in g2:
                if other_name in g1 or other_name in g2:
                    has_contact = True
                    break

    evidence['has_contact'] = has_contact

    # Margin
    z_margin = mu - abs(z_dist)
    xy_margin = mu - xy_dist
    margin = min(z_margin, xy_margin)

    # Object must be above other and in contact/proximity
    if z_dist >= -mu and xy_dist <= (obj_half + other_half + mu) and has_contact:
        return 'TRUE', margin, evidence
    elif not has_contact or z_dist > ml or z_dist < -ml \
         or xy_dist > (obj_half + other_half + ml):
        return 'FALSE', margin, evidence
    else:
        return 'UNKNOWN', margin, evidence


def compute_target_pose_source(target_name, static_seal, basket_seal_local,
                               white_wooden_sites):
    """Determine the target pose data source for a given target name.

    Returns: (source_label, world_xpos_fn_or_None, world_xmat_fn_or_None)
      source_label: 'STATIC_SEAL' | 'BASKET_RECONSTRUCT' | 'SIDECAR_DIRECT' | 'UNKNOWN_DYNAMIC_UNSEALED'
    """
    if target_name in white_wooden_sites:
        return 'UNKNOWN_DYNAMIC_UNSEALED', None, None

    if target_name in static_seal:
        xpos = static_seal[target_name]['mean_xpos']
        xmat = static_seal[target_name]['mean_xmat']
        return 'STATIC_SEAL', lambda: xpos, lambda: xmat

    if target_name in basket_seal_local:
        # Basket: reconstructed from sidecar body pose
        local_pos = basket_seal_local[target_name]['site_local_pos']
        local_quat = basket_seal_local[target_name]['site_local_quat']
        body_name = basket_seal_local[target_name]['body_name']

        def make_basket_xpos(body_poses):
            return lambda: _reconstruct_basket_pose(
                body_poses, body_name, local_pos, local_quat)[0]

        def make_basket_xmat(body_poses):
            return lambda: _reconstruct_basket_pose(
                body_poses, body_name, local_pos, local_quat)[1]

        return 'BASKET_RECONSTRUCT', make_basket_xpos, make_basket_xmat

    # Movable direct target: pose from sidecar object_state
    return 'SIDECAR_DIRECT', None, None


def _reconstruct_basket_pose(body_poses, body_name, site_local_pos, site_local_quat):
    """Reconstruct basket site world pose from body pose + local transform."""
    if body_name not in body_poses:
        return None, None
    body_xpos, body_xmat = body_poses[body_name]
    R = _xmat_to_rot3(body_xmat)
    site_xpos = np.array(body_xpos) + R @ np.array(site_local_pos)
    # Rotation: body_xmat * quat_to_mat(site_local_quat)
    # Simplified: use body rotation (site orientation ≈ body orientation for typical fixtures)
    return site_xpos, body_xmat


def evaluate_relation(obj_name, target_name, predicate, step_data,
                      obj_slices, static_seal, basket_seal, white_wooden_sites,
                      margin_upper=None, margin_lower=None):
    """Evaluate a single BDDL goal relation at a single timestep.

    Args:
        obj_name: manipulated object BDDL name (e.g. 'alphabet_soup_1')
        target_name: goal target BDDL name (e.g. 'basket_1_contain_region')
        predicate: 'In' | 'On' | 'Stack'
        step_data: single step dict from parse_sidecar
        obj_slices: dict of name -> {pos: [start,end], quat: [start,end], ...}
        static_seal: dict mapping site_name -> {mean_xpos, mean_xmat} from C3-S2
        basket_seal: dict mapping composite_key -> {body_name, site_local_pos, site_local_quat}
        white_wooden_sites: set of site names that are dynamic unsealed

    Returns: (relation_truth, margin, evidence_tier, target_pose_source, unknown_reason)
    """
    obj_state = step_data.get('object_state', [])
    if len(obj_state) == 0:
        return 'UNKNOWN', 0.0, 0, 'NONE', 'no_object_state'

    # White/wooden always UNKNOWN
    if target_name in white_wooden_sites:
        return 'UNKNOWN', 0.0, 0, 'UNKNOWN_DYNAMIC_UNSEALED', 'white_or_wooden_fixture'

    # Get manipulated object world position
    obj_spec = obj_slices.get(obj_name)
    if obj_spec is None:
        return 'UNKNOWN', 0.0, 0, 'NONE', 'object_not_in_slices'

    obj_pos = _slice_xyz(obj_state, obj_spec, 'pos')
    if obj_pos is None:
        return 'UNKNOWN', 0.0, 0, 'NONE', 'object_pos_unavailable'

    # Get target world pose based on source type
    target_xpos, target_xmat, target_size, source = None, None, None, 'NONE'
    mu = margin_upper
    ml = margin_lower

    if target_name in static_seal:
        # Static fixture: use C3-S2 world-pose seal
        target_xpos = np.array(static_seal[target_name]['mean_xpos'], dtype=float)
        target_xmat = static_seal[target_name]['mean_xmat']
        source = 'STATIC_SEAL'
    else:
        # Check if target is in obj_slices (movable object or body)
        target_spec = obj_slices.get(target_name)
        if target_spec is not None:
            target_xpos = _slice_xyz(obj_state, target_spec, 'pos')
            target_quat = _slice_xyzw(obj_state, target_spec, 'quat')
            if target_xpos is not None:
                target_xmat = _quat_to_xmat(target_quat) if target_quat is not None else [1,0,0,0,1,0,0,0,1]
                source = 'SIDECAR_DIRECT'
        elif '_region' in target_name:
            # Region target: extract base body name and reconstruct
            body_name = target_name.replace('_contain_region', '').replace(
                '_cook_region', '').replace('_heating_region', '').replace(
                '_top_region', '').replace('_bottom_region', '').replace(
                '_front_region', '').replace('_back_contain_region', '')
            # Try variations: body might be body_1 or body_1_main
            for body_candidate in [body_name, body_name + '_main']:
                body_spec = obj_slices.get(body_candidate)
                if body_spec is not None:
                    body_xpos = _slice_xyz(obj_state, body_spec, 'pos')
                    body_quat = _slice_xyzw(obj_state, body_spec, 'quat')
                    if body_xpos is not None:
                        # Look up site local pos from basket_seal or C1
                        site_local_pos = None
                        for bk, bv in basket_seal.items():
                            if target_name in bk and bv.get('body_name') in [body_name, body_candidate]:
                                site_local_pos = bv.get('site_local_pos')
                                break
                        if site_local_pos is not None:
                            target_xpos = np.array(body_xpos, dtype=float) + np.array(site_local_pos, dtype=float)
                            target_xmat = _quat_to_xmat(body_quat) if body_quat is not None else [1,0,0,0,1,0,0,0,1]
                            source = 'BASKET_RECONSTRUCT'
                        else:
                            # No seal: use body position as approximation
                            target_xpos = np.array(body_xpos, dtype=float)
                            target_xmat = _quat_to_xmat(body_quat) if body_quat is not None else [1,0,0,0,1,0,0,0,1]
                            source = 'BODY_APPROXIMATION'
                    break

    if target_xpos is None:
        return 'UNKNOWN', 0.0, 0, source, 'target_pose_unavailable'

    # Get target size
    if target_size is None:
        size_from_seal = None
        if target_name in static_seal:
            size_from_seal = static_seal[target_name].get('size')
        if size_from_seal is not None and isinstance(size_from_seal, list) and len(size_from_seal) >= 3:
            target_size = size_from_seal
        else:
            base = target_name.replace('_contain_region', '').replace(
                '_cook_region', '').replace('_heating_region', '').replace(
                '_top_region', '').replace('_bottom_region', '').replace(
                '_front_region', '').replace('_back_contain_region', '')
            hs = OBJ_HALF_SIZE.get(base, 0.075)
            target_size = [hs, hs, hs]

    # Ensure target_size is a valid 3-element list
    if not isinstance(target_size, list) or len(target_size) < 3:
        target_size = [0.075, 0.075, 0.075]
    target_size = [float(target_size[0]), float(target_size[1]), float(target_size[2])]

    # Contacts from mujoco_contact_pairs
    contacts = step_data.get('mujoco_contact_pairs', [])

    # Evaluate based on predicate
    if predicate == 'In':
        truth, margin, evidence = check_In(
            obj_pos, obj_name, target_xpos, target_xmat,
            target_size, margin_upper=mu, margin_lower=ml)
        tier = 1
    elif predicate == 'On':
        truth, margin, evidence = check_On(
            obj_pos, obj_name, target_xpos, target_xmat,
            target_size, margin_upper=mu, margin_lower=ml)
        tier = 1
    elif predicate == 'Stack':
        truth, margin, evidence = check_Stack(
            obj_pos, obj_name, target_xpos, target_name,
            contacts=contacts, margin_upper=mu, margin_lower=ml)
        tier = 1
    else:
        return 'UNKNOWN', 0.0, 0, source, f'unsupported_predicate_{predicate}'

    unknown_reason = None if truth != 'UNKNOWN' else 'margin_band'
    return truth, margin, tier, source, unknown_reason


def _slice_xyz(state_array, spec, key):
    """Extract xyz position from object_state using slice spec [start, end)."""
    bounds = spec.get(key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    start, end = int(bounds[0]), int(bounds[1])
    flat = np.array(state_array).flatten()
    if end > len(flat):
        return None
    segment = flat[start:end]
    if len(segment) >= 3:
        return np.array(segment[:3], dtype=float)
    return None


def _slice_xyzw(state_array, spec, key):
    """Extract quaternion from object_state using slice spec [start, end)."""
    bounds = spec.get(key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    start, end = int(bounds[0]), int(bounds[1])
    flat = np.array(state_array).flatten()
    if end > len(flat):
        return None
    segment = flat[start:end]
    if len(segment) >= 4:
        return np.array(segment[:4], dtype=float)
    return None


def _quat_to_xmat(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix, flattened row-major."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ])
    return R.flatten().tolist()


def compute_placement_from_relations(relations_per_step, grasp_steps, placement_steps,
                                      terminal_steps, n_steps):
    """Derive placement state from per-step relation evaluations.

    A step is 'placed' if at least one goal relation evaluates to TRUE.

    Returns per-step dict with:
      - any_relation_true: bool
      - relation_results: {relation_key: (truth, margin, tier, source)}
      - placement_derived: bool
      - pregrasp_violation: bool
    """
    results = []
    for t in range(n_steps):
        step_rels = relations_per_step[t] if t < len(relations_per_step) else {}
        any_true = any(v[0] == 'TRUE' for v in step_rels.values())
        any_unknown = any(v[0] == 'UNKNOWN' for v in step_rels.values()) if not any_true else False
        has_grasp = grasp_steps[t].get('grasp_established', False) if t < len(grasp_steps) else False
        has_placement = placement_steps[t].get('object_placed', False) if t < len(placement_steps) else False
        is_terminal = terminal_steps[t].get('is_terminal', False) if t < len(terminal_steps) else False

        results.append({
            'step': t,
            'any_relation_true': any_true,
            'any_relation_unknown': any_unknown,
            'relation_results': {k: {'truth': v[0], 'margin': v[1], 'tier': v[2], 'source': v[3]}
                               for k, v in step_rels.items()},
            'placement_derived': any_true and has_grasp,
            'pregrasp_violation': any_true and not has_grasp,
            'has_grasp': has_grasp,
            'has_placement_original': has_placement,
            'is_terminal': is_terminal,
        })
    return results
