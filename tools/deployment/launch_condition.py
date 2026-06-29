#!/usr/bin/env python3
"""Launch a Table 1 condition — SHA-bound, GPU-explicit, condition-spec-gated.

Default: dry_run. --execute requires all expected SHAs + --gpus + --condition_spec.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
WORKER = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py"
MIN_FREE_MB, GPU_DENYLIST = 20480, {2}

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gpu_free():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True,timeout=10)
    except: return {}
    d = {}
    for l in out.strip().split("\n"):
        p = l.strip().split(",")
        if len(p)>=2: d[int(p[0].strip())]=int(p[1].strip())
    return d

def check_outputs(jobs):
    issues=[]
    for j in jobs:
        out=j.get("output_dir",""); ep=os.path.join(out,"episode_summary.json")
        if os.path.exists(ep): issues.append(f"COMPLETE:{j.get('job_key','?')}")
        elif os.path.exists(out) and os.listdir(out): issues.append(f"PARTIAL:{j.get('job_key','?')}")
    return issues

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--condition_id",required=True)
    ap.add_argument("--launch_dir",required=True)
    ap.add_argument("--mode",default="formal",choices=["formal","canary"])
    ap.add_argument("--gpus",type=int,nargs="*",help="REQUIRED for --execute")
    ap.add_argument("--condition_spec",help="Condition spec JSON (REQUIRED for --execute)")
    ap.add_argument("--expected_worker_sha"); ap.add_argument("--expected_bridge_sha")
    ap.add_argument("--expected_manifest_sha"); ap.add_argument("--expected_condition_spec_sha")
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()

    jobs=[json.loads(l) for l in open(args.manifest)]
    print(f"Manifest: {len(jobs)} jobs, mode={args.mode}")
    if args.mode=="formal" and len(jobs)!=162: sys.exit(f"Formal needs 162, got {len(jobs)}")
    if not jobs: sys.exit("Empty")
    for j in jobs:
        if j.get("condition_id")!=args.condition_id: sys.exit(f"condition_id mismatch")

    # Pre-execute gates
    if args.execute:
        for name,val in [("--gpus",args.gpus),("--condition_spec",args.condition_spec),
                          ("--expected_worker_sha",args.expected_worker_sha),
                          ("--expected_bridge_sha",args.expected_bridge_sha),
                          ("--expected_manifest_sha",args.expected_manifest_sha),
                          ("--expected_condition_spec_sha",args.expected_condition_spec_sha)]:
            if val is None: sys.exit(f"--execute requires {name}")
        if len(args.gpus)==0: sys.exit("--gpus cannot be empty")

    # Provenance verification
    manifest_sha=sha_file(args.manifest)
    worker_sha=sha_file(WORKER) if os.path.exists(WORKER) else "MISSING"
    bridge_path=os.path.join(os.path.dirname(WORKER),"run_v2_vis_sc5_mlp_bridge.py")
    bridge_sha=sha_file(bridge_path) if os.path.exists(bridge_path) else "MISSING"
    print(f"Worker: {worker_sha[:16]}... Bridge: {bridge_sha[:16]}... Manifest: {manifest_sha[:16]}...")

    if args.expected_worker_sha and worker_sha!=args.expected_worker_sha:
        sys.exit(f"Worker SHA mismatch")
    if args.expected_bridge_sha and bridge_sha!=args.expected_bridge_sha:
        sys.exit(f"Bridge SHA mismatch")
    if args.expected_manifest_sha and manifest_sha!=args.expected_manifest_sha:
        sys.exit(f"Manifest SHA mismatch")

    # Condition spec gate
    if args.condition_spec:
        if not os.path.exists(args.condition_spec): sys.exit(f"Spec missing: {args.condition_spec}")
        spec=json.load(open(args.condition_spec))
        spec_sha=sha_file(args.condition_spec)
        if args.expected_condition_spec_sha and spec_sha!=args.expected_condition_spec_sha:
            sys.exit(f"Spec SHA mismatch: expected {args.expected_condition_spec_sha[:16]}... got {spec_sha[:16]}...")
        if spec.get("execution_status")!="FROZEN":
            sys.exit(f"execution_status={spec.get('execution_status')} (need FROZEN)")
        if spec.get("condition_id")!=args.condition_id:
            sys.exit(f"Spec condition_id={spec.get('condition_id')} != {args.condition_id}")
        print(f"Spec: FROZEN, SHA={spec_sha[:16]}...")

    # Output check
    existing=check_outputs(jobs)
    if existing:
        print(f"ERROR: {len(existing)} jobs have existing output:")
        for e in existing[:10]: print(f"  {e}")
        sys.exit("Reject: existing outputs. Use recovery procedure.")

    # GPU check
    gf=gpu_free(); print(f"GPU free (MB): {gf}")
    for g in (args.gpus or []):
        if g in GPU_DENYLIST: sys.exit(f"GPU {g} denylisted")
        if gf.get(g,0)<MIN_FREE_MB: sys.exit(f"GPU {g}: {gf.get(g,0)}MB < {MIN_FREE_MB}")
    available=sorted(args.gpus) if args.gpus else []
    for g,mb in sorted(gf.items()):
        tag="DENYLIST" if g in GPU_DENYLIST else ("APPROVED" if g in available else "-")
        print(f"  GPU {g}: {mb}MB {tag}")
    if not available: sys.exit("No GPUs")
    if args.execute and not args.gpus: sys.exit("--execute requires --gpus")

    # Split
    nw=min(len(jobs),len(available))
    splits=[[] for _ in range(nw)]
    for i,j in enumerate(jobs): splits[i%nw].append(j)
    if any(len(s)==0 for s in splits):
        sys.exit(f"Empty worker split: n_jobs={len(jobs)} n_workers={nw}")

    os.makedirs(args.launch_dir,exist_ok=True)
    print(f"\nPlan: {nw} workers on {available}, {len(jobs)} jobs")
    for wi,s in enumerate(splits): print(f"  GPU {available[wi]} w{wi}: {len(s)} jobs")
    if not args.execute: print("\nDRY_RUN. Use --execute to launch."); return

    # ── Execute ──
    # Atomic RUNNING claim (after all checks pass, launch_dir exists)
    marker=os.path.join(args.launch_dir,"RUNNING")
    try:
        fd=os.open(marker, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    except FileExistsError:
        try: ri=json.load(open(marker)); sys.exit(f"Already RUNNING since {ri.get('started','?')}")
        except Exception as e: sys.exit(f"Corrupt RUNNING: {e}. Recovery required.")
    running_info={"started":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  "condition_id":args.condition_id,"mode":args.mode,"pid":os.getpid()}
    try:
        with os.fdopen(fd,"w") as mf: json.dump(running_info,mf)
    except Exception:
        os.unlink(marker)
        raise

    plan={"manifest_sha256":manifest_sha,"condition_id":args.condition_id,"mode":args.mode,
          "timestamp":running_info["started"],"gpus":available,"workers":[],
          "provenance":{"worker_sha256":worker_sha,"bridge_sha256":bridge_sha,
                         "manifest_sha256":manifest_sha,"condition_spec_sha256":spec_sha if args.condition_spec else None}}
    for wi,s in enumerate(splits):
        gpu=available[wi]
        mf=os.path.join(args.launch_dir,f"manifest_gpu{gpu}_w{wi}.jsonl")
        lf=os.path.join(args.launch_dir,f"worker_gpu{gpu}_w{wi}.log")
        with open(mf,"w") as f:
            for j in s: f.write(json.dumps(j)+"\n")
        proc=subprocess.Popen([PYTHON,"-u",WORKER,str(gpu),mf],stdout=open(lf,"w"),stderr=subprocess.STDOUT,start_new_session=True)
        plan["workers"].append({"gpu":gpu,"worker_idx":wi,"pid":proc.pid,"n_jobs":len(s),"manifest":mf,"log":lf})
        print(f"  GPU {gpu} w{wi}: PID {proc.pid}, {len(s)} jobs")
    with open(os.path.join(args.launch_dir,"LAUNCH_PLAN.json"),"w") as f: json.dump(plan,f,indent=2)
    print(f"\nLaunched. {args.launch_dir}/LAUNCH_PLAN.json")

if __name__=="__main__": main()
