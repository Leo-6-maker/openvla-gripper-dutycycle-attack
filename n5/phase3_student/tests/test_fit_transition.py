"""[DeepSeek] FIT-INFERENCE Transition Tests (v3 — R2 remediation).
"""
import json, os, sys, hashlib, shutil, tempfile, unittest, re, subprocess
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'phase2_labels'))
from fit_transition import (
    verify_transition, TransitionRejected, sha256_file, full_seal_check,
    compute_model_tree_fingerprint, validate_identity_allowlist, FROZEN_R5E,
)
from build_fit_transition import git_value as _git_value

SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
EXEC_COMMIT = "e" * 40
SCRIPT_SHA = "a" * 64

def _make_pilot():
    recs = []
    for s in ["libero_10","libero_goal","libero_object","libero_spatial"]:
        for t in range(10):
            recs.append({"episode_id":f"{s}/task_{t:02d}/state_0","suite":s,
                          "task_id":t,"state_id":0,"collection_seed":20260717,
                          "initial_state_sha256":"0"*64})
    return json.dumps({"protected_payload_read":False,"no_attack":True,"records":recs})

def _make_allowlist():
    ids = []
    for s in ["libero_10","libero_goal","libero_object","libero_spatial"]:
        for t in range(10):
            ids.append({"episode_id":f"{s}/task_{t:02d}/state_0","suite":s,
                        "task_id":t,"state_id":0,"collection_seed":20260717,
                        "initial_state_sha256":"0"*64})
    return json.dumps({"gate":"FIT-INFERENCE_IDENTITY_ALLOWLIST","n_identities":40,"identities":ids})

def _make_git_repo(commit_msg="test", extra_file=None):
    d = Path(tempfile.mkdtemp(prefix="test_git_"))
    subprocess.run(["git","init","-q"], cwd=str(d), check=True)
    subprocess.run(["git","config","user.email","test@test"], cwd=str(d), check=True)
    subprocess.run(["git","config","user.name","Test"], cwd=str(d), check=True)
    (d / "README.md").write_text(commit_msg)
    subprocess.run(["git","add","."], cwd=str(d), check=True)
    subprocess.run(["git","commit","-q","-m",commit_msg], cwd=str(d), check=True)
    if extra_file:
        (d / extra_file).write_text("dirty")
    return d

def _sealed_transition(manifest_overrides=None, tamper=None, extra_file=None,
                        upstream_root=None, libero_root=None):
    root = Path(tempfile.mkdtemp(prefix="fit_test_t_"))
    model_dir = Path(tempfile.mkdtemp(prefix="test_model_"))
    (model_dir / "config.json").write_text('{"t":1}')
    (model_dir / "preprocessor_config.json").write_text('{"t":1}')
    worker = Path(tempfile.mktemp(suffix=".py"))
    worker.write_text("# test worker")
    pilot = Path(tempfile.mktemp(suffix=".json"))
    pilot.write_text(_make_pilot())
    reg_dir = Path(tempfile.mkdtemp(prefix="test_reg_"))
    (reg_dir / "ENTITY_REGISTRY_V2_SUMMARY.json").write_text('{"status":"PASS"}')
    per_task = reg_dir / "per_task"; per_task.mkdir()
    alias = Path(tempfile.mktemp(suffix=".json"))
    alias.write_text('{"n_aliases":5}')

    if upstream_root is None:
        upstream_root = _make_git_repo("upstream")
    if libero_root is None:
        libero_root = _make_git_repo("libero")

    upstream_commit = _git_value(upstream_root, "rev-parse", "HEAD")
    upstream_tree = _git_value(upstream_root, "rev-parse", "HEAD^{tree}")
    libero_commit = _git_value(libero_root, "rev-parse", "HEAD")
    libero_tree = _git_value(libero_root, "rev-parse", "HEAD^{tree}")

    manifest = {
        "gate":"FIT-INFERENCE_TRANSITION","schema":"FIT_INFERENCE_TRANSITION_V1",
        "status":"FROZEN_BEFORE_EXECUTION","created_at":"2026-07-28T12:00:00Z",
        **FROZEN_R5E,
        "r5f_execution_source_commit":EXEC_COMMIT,"r5f_script_sha256":SCRIPT_SHA,
        "model_tree_sha256":compute_model_tree_fingerprint(model_dir),
        "processor_sha256":sha256_file(model_dir/"preprocessor_config.json"),
        "official_worker_sha256":sha256_file(worker),
        "pilot_manifest_sha256":sha256_file(pilot),
        "registry_summary_sha256":sha256_file(reg_dir/"ENTITY_REGISTRY_V2_SUMMARY.json"),
        "alias_ledger_sha256":sha256_file(alias),
        "upstream_commit":upstream_commit,"upstream_tree":upstream_tree,
        "libero_commit":libero_commit,"libero_tree":libero_tree,
        "identity_allowlist_digest":"", "identity_set_digest":"",
        "authorized_identities":40,"n_pilot_identities":40,
        "allowed_gpus":[6,7],"physical_to_logical_gpu":{"6":0,"7":0},
        "allowed_output_roots":["/tmp/test_output"],
        "openvla_inference_authorized":True,"clean_action_only":True,
        "forward_before_capture":True,"max_episodes":40,"identity_set_frozen":True,
        "teacher_labels_authorized":False,"student_training_authorized":False,
        "detector_load_authorized":False,"attack_authorized":False,
        "protected_payload_read":False,
    }

    # Allowlist
    al_content = _make_allowlist()
    (root / "IDENTITY_ALLOWLIST.json").write_text(al_content)
    manifest["identity_allowlist_digest"] = sha256_file(root/"IDENTITY_ALLOWLIST.json")
    al_data = json.loads(al_content)["identities"]
    manifest["identity_set_digest"] = hashlib.sha256(
        json.dumps(al_data, sort_keys=True).encode()).hexdigest()

    if manifest_overrides:
        manifest.update(manifest_overrides)
    (root / "TRANSITION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    payload = sorted(p for p in root.rglob("*") if p.is_file())
    sums = "\n".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in payload)+"\n"
    (root / "SHA256SUMS").write_text(sums)
    s = sha256_file(root/"SHA256SUMS")
    (root/"SHA256SUMS.sha256").write_text(f"{s}  SHA256SUMS\n")
    # Extra file AFTER seal — will be detected as unsealed
    if extra_file:
        (root / extra_file).write_text("UNSEALED")
    if tamper:
        (root/tamper).write_text((root/tamper).read_text()+"TAMPERED")

    return root, model_dir, worker, pilot, per_task, alias, upstream_root, libero_root


def _verify(root, md, wk, pl, per_task, al, up, lib, **kw):
    return verify_transition(root, kw.get("commit",EXEC_COMMIT),
        kw.get("script",SCRIPT_SHA), md, wk, pl, per_task, al, up, lib,
        kw.get("output","/tmp/test_output"), kw.get("gpu",6),
        kw.get("physical_gpu",6))


class TestSealRejects(unittest.TestCase):
    def test_01_missing(self):
        with self.assertRaises((TransitionRejected,SystemExit,FileNotFoundError)):
            _verify("/nonexistent",*[Path("/tmp/x")]*7)
    def test_02_unsealed(self):
        r=Path(tempfile.mkdtemp());(r/"TRANSITION_MANIFEST.json").write_text("{}")
        with self.assertRaises(TransitionRejected):
            _verify(r,*[Path("/tmp/x")]*7)
        shutil.rmtree(r,ignore_errors=True)
    def test_03_tampered(self):
        a=_sealed_transition(tamper="TRANSITION_MANIFEST.json")
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_04_extra_file(self):
        a=_sealed_transition(extra_file="EXTRA.txt")
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)

class TestFrozenBindings(unittest.TestCase):
    def test_05_c1(self):
        a=_sealed_transition({"c1_canonical_digest":"0"*64})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_06_r5e(self):
        a=_sealed_transition({"r5e_run_a_sha256sums":"0"*64})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_07_comparison_sha(self):
        a=_sealed_transition({"r5e_comparison_sha256":"wrong"})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_08_source_commit(self):
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(*a,commit="0"*40)
        finally: shutil.rmtree(a[0],ignore_errors=True)

class TestModelBinding(unittest.TestCase):
    def test_09_wrong_tree(self):
        a=_sealed_transition({"model_tree_sha256":"0"*64})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_10_wrong_worker(self):
        a=_sealed_transition(); w2=Path(tempfile.mktemp(suffix=".py"));w2.write_text("#wrong")
        try:
            with self.assertRaises(TransitionRejected): _verify(a[0],a[1],w2,*a[3:])
        finally: shutil.rmtree(a[0],ignore_errors=True);os.remove(w2)

class TestRuntimeBinding(unittest.TestCase):
    def test_11_wrong_upstream(self):
        up2=_make_git_repo("different_upstream")
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(a[0],a[1],a[2],a[3],a[4],a[5],up2,a[7])
        finally: shutil.rmtree(a[0],ignore_errors=True);shutil.rmtree(up2,ignore_errors=True)
    def test_12_wrong_libero(self):
        lib2=_make_git_repo("different_libero")
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(a[0],a[1],a[2],a[3],a[4],a[5],a[6],lib2)
        finally: shutil.rmtree(a[0],ignore_errors=True);shutil.rmtree(lib2,ignore_errors=True)
    def test_13_dirty_upstream(self):
        up=_make_git_repo("upstream",extra_file="dirty.txt")
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(a[0],a[1],a[2],a[3],a[4],a[5],up,a[7])
        finally: shutil.rmtree(a[0],ignore_errors=True);shutil.rmtree(up,ignore_errors=True)

class TestPermissions(unittest.TestCase):
    def test_14_teacher(self):
        a=_sealed_transition({"teacher_labels_authorized":True})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_15_attack(self):
        a=_sealed_transition({"attack_authorized":True})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_16_unauthorized_gpu(self):
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(*a,gpu=99)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_17_wrong_output(self):
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(*a,output="/wrong")
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_18_wrong_physical_gpu(self):
        a=_sealed_transition()
        try:
            with self.assertRaises(TransitionRejected): _verify(*a,physical_gpu=99)
        finally: shutil.rmtree(a[0],ignore_errors=True)

class TestIdentityBinding(unittest.TestCase):
    def test_19_wrong_set_digest(self):
        a=_sealed_transition({"identity_set_digest":"0"*64})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_20_wrong_authorized_count(self):
        a=_sealed_transition({"authorized_identities":39})
        try:
            with self.assertRaises(TransitionRejected): _verify(*a)
        finally: shutil.rmtree(a[0],ignore_errors=True)
    def test_21_missing_seed(self):
        pd=json.loads(_make_pilot());del pd["records"][0]["collection_seed"]
        p=Path(tempfile.mktemp(suffix=".json"));p.write_text(json.dumps(pd))
        root=Path(tempfile.mkdtemp())
        with self.assertRaises(TransitionRejected):
            validate_identity_allowlist(root/"x",p)
        shutil.rmtree(root,ignore_errors=True);os.remove(p)

class TestPositive(unittest.TestCase):
    def test_22_valid(self):
        a=_sealed_transition()
        try:
            r=_verify(*a)
            self.assertEqual(r["gate"],"FIT-INFERENCE_TRANSITION")
        finally: shutil.rmtree(a[0],ignore_errors=True)

class TestFingerprint(unittest.TestCase):
    def test_23_different(self):
        d1=Path(tempfile.mkdtemp());(d1/"a.txt").write_text("1")
        d2=Path(tempfile.mkdtemp());(d2/"a.txt").write_text("2")
        self.assertNotEqual(compute_model_tree_fingerprint(d1),compute_model_tree_fingerprint(d2))
        shutil.rmtree(d1,ignore_errors=True);shutil.rmtree(d2,ignore_errors=True)
    def test_24_symlink(self):
        d=Path(tempfile.mkdtemp());(d/"r.txt").write_text("r")
        os.symlink(d/"r.txt",d/"l.txt")
        try:
            with self.assertRaises(TransitionRejected): compute_model_tree_fingerprint(d)
        finally: shutil.rmtree(d,ignore_errors=True)


if __name__=="__main__":
    unittest.main(verbosity=2)
