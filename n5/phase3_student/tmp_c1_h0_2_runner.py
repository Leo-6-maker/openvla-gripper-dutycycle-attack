"""H0.2 C1 Rerun with full provenance recording."""
import json, os, sys, time, hashlib, subprocess

PROV = {
    'producer_agent': 'DeepSeek',
    'gate': 'C1_RERUN_H0_2',
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
os.chdir('/mnt/sdc/dty_user/openvla_attack')
PROV['server_host'] = subprocess.check_output(['hostname'], text=True).strip()
PROV['server_repo_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
PROV['server_repo_dirty'] = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
PROV['python_version'] = sys.version
try:
    import numpy; PROV['numpy_version'] = numpy.__version__
except Exception: PROV['numpy_version'] = 'unknown'
try:
    import mujoco; PROV['mujoco_version'] = mujoco.__version__
except Exception: PROV['mujoco_version'] = 'unknown'

script_path = '/tmp/t2rc1_full_registry_h0_2.py'
PROV['source_script_sha256'] = hashlib.sha256(open(script_path, 'rb').read()).hexdigest()
PROV['claimed_source_commit'] = 'b5c9634'
try:
    subprocess.check_call(['git', 'cat-file', '-e', 'b5c9634^{commit}'])
    PROV['claimed_commit_resolvable'] = True
except Exception:
    PROV['claimed_commit_resolvable'] = False

NEW_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_h0_2_rerun_v2'
os.makedirs(NEW_OUT, exist_ok=True)
PROV['output_dir'] = NEW_OUT

print("=== PROVENANCE ===")
print(json.dumps(PROV, indent=2))

with open(script_path) as f:
    code = f.read()

old_out = "'/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rc1_full_registry'"
new_out = repr(NEW_OUT)
code = code.replace(
    "T2RC1_OUT = " + old_out,
    "T2RC1_OUT = " + new_out
)

# Add phase2_labels to path for v22_production_v2 import
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels')
# Also add the phase3_student dir for sibling imports
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student')

exec_globals = {'__name__': '__main__', '__file__': script_path}
exec(code, exec_globals)
