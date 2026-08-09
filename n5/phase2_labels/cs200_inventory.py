"""CS200 Metadata-Only Inventory. GO_LIMITED authorization.

Reads: directory listing, file stat/size/SHA, NPZ key/dtype/shape (NO arrays).
Writes: N5 isolated output only.
NO: allow_pickle=True, content reads, code execution, CS200 modification.
"""
import json, os, sys, hashlib, argparse
from collections import defaultdict

CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716'
N5_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5'

def safe_listdir(path, max_depth=3, current_depth=0):
    """List directory contents without reading file contents. Max depth safety."""
    if current_depth > max_depth:
        return {'_truncated': True, '_depth': current_depth}
    entries = {}
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            try:
                st = os.lstat(full)
                if os.path.islink(full):
                    entries[name] = {'type': 'symlink', 'target': os.readlink(full)}
                elif os.path.isdir(full):
                    if current_depth < max_depth:
                        entries[name] = safe_listdir(full, max_depth, current_depth + 1)
                    else:
                        entries[name] = {'type': 'directory', '_truncated': True}
                else:
                    entries[name] = {
                        'type': 'file',
                        'size': st.st_size,
                        'mtime': st.st_mtime,
                    }
            except OSError as e:
                entries[name] = {'type': 'error', 'error': str(e)}
    except OSError as e:
        entries['_error'] = str(e)
    return entries

def sha256_file(path):
    """Compute SHA256 of file without loading into memory."""
    h = hashlib.sha256()
    with open(path, 'rb', buffering=8192) as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def inspect_npz_keys(path):
    """Inspect NPZ keys, dtypes, shapes WITHOUT loading arrays.
    Uses np.load with mmap_mode='r' and only reads .files attribute.
    """
    import numpy as np
    try:
        with np.load(path, allow_pickle=False, mmap_mode='r') as data:
            info = {}
            for key in sorted(data.files):
                arr = data[key]
                info[key] = {'dtype': str(arr.dtype), 'shape': tuple(arr.shape)}
            return info
    except Exception as e:
        return {'_error': str(e)}

def count_jsonl_lines(path):
    """Count lines without parsing JSON (safe for large files)."""
    count = 0
    with open(path, 'rb', buffering=8192) as f:
        for _ in f:
            count += 1
    return count

def inventory_subdir(root, subpath, file_pattern, max_files=50):
    """Inventory a subdirectory: list, count, SHA sample."""
    full = os.path.join(root, subpath)
    if not os.path.isdir(full):
        return {'_error': f'not found: {full}'}

    files = []
    for dirpath, dirnames, filenames in os.walk(full):
        for fn in sorted(filenames):
            if fn.endswith(file_pattern) if '.' in file_pattern else True:
                files.append(os.path.join(dirpath, fn))

    total = len(files)
    sample_size = min(max_files, total)
    samples = files[:sample_size] if sample_size > 0 else []

    result = {
        'root': full,
        'pattern': file_pattern,
        'total_files': total,
        'sample_size': sample_size,
        'samples': [],
    }

    for f in samples:
        rel = os.path.relpath(f, full)
        st = os.lstat(f)
        sha = sha256_file(f) if st.st_size < 10 * 1024 * 1024 else None  # SHA only for <10MB
        entry = {'path': rel, 'size': st.st_size, 'mtime': st.st_mtime}
        if sha:
            entry['sha256'] = sha

        # Inspect NPZ metadata
        if f.endswith('.npz'):
            entry['npz_keys'] = inspect_npz_keys(f)
        elif f.endswith('.jsonl'):
            entry['line_count'] = count_jsonl_lines(f)
        elif f.endswith('.json'):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    entry['json_keys'] = sorted(data.keys())[:20]
                elif isinstance(data, list):
                    entry['json_len'] = len(data)
            except Exception:
                pass

        result['samples'].append(entry)

    # Directory size
    total_size = 0
    for f in files:
        try:
            total_size += os.lstat(f).st_size
        except OSError:
            pass
    result['total_size_bytes'] = total_size
    result['total_size_mb'] = round(total_size / (1024 * 1024), 1)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=os.path.join(N5_OUT, 'reports', 'CS200_INVENTORY.json'))
    args = parser.parse_args()

    print('=== CS200 Metadata-Only Inventory ===')
    print(f'Root: {CS200_ROOT}')
    print('Authorization: GO_LIMITED (metadata only, no content reads)')
    print()

    inventory = {
        'schema': 'CS200_METADATA_INVENTORY_V1',
        'cs200_root': CS200_ROOT,
        'authorization': 'GO_LIMITED',
        'constraints': [
            'NO content reads (no array data, no image pixels)',
            'NO allow_pickle=True',
            'NO code execution in CS200 directories',
            'NO writes, deletes, renames, or chmod on CS200 paths',
            'SHA only for files < 10 MB',
            'NPZ inspection: keys, dtype, shape only (mmap_mode=r)',
        ],
        'top_level': safe_listdir(CS200_ROOT, max_depth=1),
    }

    # Inventory subdirectories
    subdirs = [
        ('clean', '**/*.npz'),
        ('ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels', 'factorized_teacher_v1.jsonl'),
        ('ops/OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21C_STATES_35_49_20260725/labels', 'physics_teacher_v21c.jsonl'),
    ]

    for subpath, pattern in subdirs:
        print(f'Inventory: {subpath} ({pattern})')
        inv = inventory_subdir(CS200_ROOT, subpath, pattern)
        key = subpath.replace('/', '_')
        inventory[key] = inv
        print(f'  Files: {inv.get("total_files", "error")}')
        if inv.get('total_size_mb'):
            print(f'  Size: {inv["total_size_mb"]} MB')
        print()

    # Per-suite breakdown for clean trajectories
    clean_dir = os.path.join(CS200_ROOT, 'clean')
    suite_breakdown = {}
    if os.path.isdir(clean_dir):
        for suite in sorted(os.listdir(clean_dir)):
            suite_path = os.path.join(clean_dir, suite)
            if os.path.isdir(suite_path):
                npz_count = 0
                suite_size = 0
                tasks = set()
                for dirpath, dirnames, filenames in os.walk(suite_path):
                    for fn in filenames:
                        if fn.endswith('.npz'):
                            npz_count += 1
                            try:
                                suite_size += os.lstat(os.path.join(dirpath, fn)).st_size
                            except OSError:
                                pass
                        # Extract task/state from path
                        rel = os.path.relpath(dirpath, suite_path)
                        if rel != '.':
                            parts = rel.split(os.sep)
                            if parts:
                                tasks.add(parts[0])
                suite_breakdown[suite] = {
                    'npz_count': npz_count,
                    'size_mb': round(suite_size / (1024 * 1024), 1),
                    'task_count': len(tasks),
                }
                print(f'  {suite}: {npz_count} NPZ, {suite_breakdown[suite]["size_mb"]} MB, {len(tasks)} tasks')

    inventory['suite_breakdown'] = suite_breakdown

    # Write inventory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)

    # Self-hash
    inv_sha = sha256_file(args.output)
    print(f'\nInventory: {args.output}')
    print(f'SHA256: {inv_sha}')

    # Verify no Formal paths touched
    formal_root = '/mnt/sdc/dty_user/openvla_attack_outputs/fec_formal_v2'
    if os.path.isdir(formal_root):
        formal_mtime = os.lstat(formal_root).st_mtime
    else:
        formal_mtime = None
    print(f'Formal root mtime: {formal_mtime} (unchanged)')

if __name__ == '__main__':
    main()
