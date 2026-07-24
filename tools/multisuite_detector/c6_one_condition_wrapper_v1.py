#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

FIELDS = ["parent_id","episode_key","suite","task_id","condition","clean_success_parent","condition_success","contact_quality_failure","contact_quality_success","nad_g","delta_open","qpos_response","width_response","arm_dev","latency","command_open_duty","sustained_open_duty","exact_prefix_shared","clean_success_parent_denominator"]
PRIMARY = {"libero_goal","libero_object","libero_spatial"}
CONDS = {"CLEAN","TRUE_T10","RAND_T10","RANDOM_TIME","EARLY_SHIFT","ORACLE"}
BOOLS = {"clean_success_parent","condition_success","contact_quality_failure","contact_quality_success","exact_prefix_shared","clean_success_parent_denominator"}
FLOATS = {"nad_g","delta_open","qpos_response","width_response","arm_dev","latency","command_open_duty","sustained_open_duty"}
IDS = {"parent_id","episode_key","suite","task_id","condition"}

class E(ValueError): pass

def die(s): raise E(s)
def load(p): return json.loads(Path(p).read_text())
def save(p,o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,sort_keys=True)+"\n")
def get(o,path):
    cur=o
    for x in str(path).split('.'):
        if not isinstance(cur,dict) or x not in cur: die('missing legacy field '+str(path))
        cur=cur[x]
    return cur
def val(o,spec):
    if isinstance(spec,list):
        last=None
        for x in spec:
            try: return get(o,x)
            except E as e: last=e
        raise last
    return get(o,spec)
def b(v,k):
    if isinstance(v,bool): return 'true' if v else 'false'
    if str(v) in {'1','true','TRUE','yes','YES'}: return 'true'
    if str(v) in {'0','false','FALSE','no','NO'}: return 'false'
    die(k+' not bool')
def f(v,k):
    try: x=float(v)
    except Exception: die(k+' not float')
    if x!=x or x in (float('inf'),float('-inf')): die(k+' nonfinite')
    return f'{x:.10g}'
def fmt(t,m): return [str(x).format(**m) for x in t]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runner-binding-json',required=True); p.add_argument('--parent-id',required=True)
    p.add_argument('--episode-key',required=True); p.add_argument('--suite',required=True); p.add_argument('--task-id',required=True)
    p.add_argument('--condition',required=True); p.add_argument('--output-json',required=True); p.add_argument('--work-root')
    p.add_argument('--timeout-seconds',type=int,default=3600)
    a=p.parse_args()
    try:
        if a.suite not in PRIMARY: die('suite not primary')
        if a.condition not in CONDS: die('bad condition')
        cfg=load(a.runner_binding_json)
        fmap=cfg.get('field_map',{}); lits=cfg.get('literal_fields',{})
        miss=sorted((set(FIELDS)-IDS)-set(fmap)-set(lits))
        if miss: die('binding missing '+','.join(miss))
        work=Path(a.work_root or (a.output_json+'.work')); work.mkdir(parents=True,exist_ok=True)
        legacy=work/'legacy_result.json'
        m={'parent_id':a.parent_id,'episode_key':a.episode_key,'suite':a.suite,'task_id':a.task_id,'condition':a.condition,'output_json':a.output_json,'result_json':a.output_json,'work_dir':str(work),'raw_output_dir':str(work),'legacy_result_json':str(legacy)}
        cp=subprocess.run(fmt(cfg['command_template'],m),text=True,capture_output=True,timeout=int(cfg.get('timeout_seconds',a.timeout_seconds)))
        (work/'stdout.txt').write_text(cp.stdout[-20000:]); (work/'stderr.txt').write_text(cp.stderr[-20000:])
        if cp.returncode: die('legacy command failed')
        lp=Path(str(cfg.get('legacy_result_json','{legacy_result_json}')).format(**m))
        if not lp.is_file(): die('legacy result missing')
        src=load(lp); row={'parent_id':a.parent_id,'episode_key':a.episode_key,'suite':a.suite,'task_id':a.task_id,'condition':a.condition}
        for k in FIELDS:
            if k in row: continue
            v=lits[k] if k in lits else val(src,fmap[k])
            row[k]=b(v,k) if k in BOOLS else f(v,k) if k in FLOATS else str(v)
        if row['clean_success_parent']!='true' or row['exact_prefix_shared']!='true' or row['clean_success_parent_denominator']!='true': die('required boundary field false')
        save(a.output_json,row); print(json.dumps(row,sort_keys=True)); return 0
    except Exception as e:
        print(str(e),file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
