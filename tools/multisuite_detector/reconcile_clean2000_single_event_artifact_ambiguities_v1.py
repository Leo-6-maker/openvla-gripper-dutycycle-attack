#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

GATE="D1D0C_CLEAN2000_SINGLE_EVENT_ARTIFACT_AMBIGUITY_RECONCILIATION"
PASS="PASS_SINGLE_EVENT_ARTIFACT_AMBIGUITIES_RECONCILED"
OUT_FILES=["single_event_artifact_reconciliation_report.json","single_event_artifact_reconciled_bindings.csv","single_event_artifact_reconciliation_failures.csv","single_event_artifact_reconciliation_candidate_signatures.csv","checksum_report.json"]
SUITES=["libero_spatial","libero_goal","libero_object","libero_10"]
SINGLE=["libero_spatial","libero_goal","libero_object"]
ID=["parent_id","episode_key","run_id","record_id","id"]
TEXT=["suite","suite_name","benchmark","libero_suite","task_id","task_name","instruction","language_instruction","path","output_root","state_id"]
STATUS=["teacher_label_status","label_status","source_label_status","source_event_status","event_status"]
TASK=re.compile(r"(?:^|[^a-z0-9])task[_\- /]*0*([0-9]+)(?:[^0-9]|$)",re.I)
STATE=[re.compile(r"(?:^|[^a-z0-9])state[_\- /]*0*([0-9]+)(?:[^0-9]|$)",re.I),re.compile(r"(?:^|[^a-z0-9])init[_\- /]*state[_\- /]*0*([0-9]+)(?:[^0-9]|$)",re.I),re.compile(r"(?:^|[^a-z0-9])initial[_\- /]*state[_\- /]*0*([0-9]+)(?:[^0-9]|$)",re.I)]
ALIASES={"libero_spatial":["libero_spatial","libero-spatial","/spatial/","black_bowl"],"libero_goal":["libero_goal","libero-goal","/goal/","drawer"],"libero_object":["libero_object","libero-object","liberoobject","/object/","alphabet_soup","tomato_sauce","cream_cheese","orange_juice","butter","milk"]}
TEMP=["path_step_telemetry.csv","path_step_records.jsonl","path_phase_cues.csv","path_episode_manifest.json"]
STEP=["step","timestep","frame","frame_idx","step_idx","index"]
CMD=["action_gripper","gripper_command","gripper_action","a_gripper"]
OPEN=["gripper_opening_proxy","gripper_qpos","gripper_width","robot0_gripper_qpos","obs_gripper_qpos"]
DIRECT=["teacher_anchor_step","positive_anchor_step","anchor_step","event_step","selected_preplace_step","preplace_step","release_intent_step"]
W0=["teacher_window_start","window_start","positive_window_start","event_window_start"]
W1=["teacher_window_end","window_end","positive_window_end","event_window_end"]

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def r_csv(path:Path)->List[Dict[str,Any]]:
    with path.open(newline='',encoding='utf-8') as f:
        out=[]
        for n,row in enumerate(csv.DictReader(f),start=2):
            row=dict(row); row['__line']=n; out.append(row)
        return out

def w_csv(path:Path,rows:List[Dict[str,Any]],fields:List[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})

def w_json(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def first(row:Dict[str,Any],keys:List[str],default:str='')->str:
    for k in keys:
        if row.get(k) not in (None,''): return str(row[k])
    return default

def rid(row): return first(row,ID,f"row_{row.get('__line','')}")

def alias(text:str)->str:
    low=text.lower()
    if 'libero_10' in low or 'libero-10' in low or 'libero10' in low or 'moka' in low: return 'libero_10'
    for s,als in ALIASES.items():
        if any(a in low for a in als): return s
    return 'UNKNOWN'

def suite(row):
    d=first(row,['suite','suite_name','benchmark','libero_suite','suite_hint'],'').strip()
    if d in SUITES: return d
    return alias(' '.join(str(row.get(k,'') or '') for k in TEXT+ID+['artifact_dir']))

def parse_i(v):
    if v in (None,''): return None
    s=str(v).strip()
    if re.fullmatch(r"0*[0-9]+",s): return int(s)
    try:
        f=float(s); return int(f) if f.is_integer() and f>=0 else None
    except Exception: return None

def parse_ts(text:str)->Tuple[int|None,int|None]:
    tm=TASK.search(text); task=int(tm.group(1)) if tm else None
    st=None
    for p in STATE:
        sm=p.search(text)
        if sm: st=int(sm.group(1)); break
    return task,st

def key_record(row)->Tuple[str,int|None,int|None,str]:
    s=suite(row); task=None; st=None
    for k in ['task_index','task_idx','task_num','task_id']:
        task=parse_i(row.get(k))
        if task is not None: break
    for k in ['state_index','state_idx','state_num','state_id','initial_state_id','init_state_id','state']:
        st=parse_i(row.get(k))
        if st is not None: break
    t2,s2=parse_ts(' '.join(str(row.get(k,'') or '') for k in TEXT+ID))
    task=task if task is not None else t2; st=st if st is not None else s2
    return s,task,st,f"{s}/task_{task:02d}/state_{st:03d}" if task is not None and st is not None else ''

def key_art(row)->Tuple[str,int|None,int|None,str]:
    s=suite(row); task,st=parse_ts(str(row.get('artifact_dir','') or ''))
    return s,task,st,f"{s}/task_{task:02d}/state_{st:03d}" if task is not None and st is not None else ''

def status(row):
    v=first(row,STATUS,'').strip()
    if v: return v
    for k in ['source_positive_anchor_valid','positive_anchor_valid','has_positive_anchor','source_event_valid']:
        if k in row and str(row.get(k,'')).strip() not in {'','0','False','false','NO','no'}: return 'SOURCE_POSITIVE'
    for k in ['source_no_event','no_event','clean_failed']:
        if k in row and str(row.get(k,'')).strip() not in {'','0','False','false','NO','no'}: return 'NO_EVENT'
    return 'UNKNOWN'

def temporal(row):
    for f in TEMP:
        p=str(row.get(f,'') or '').strip()
        if p and Path(p).exists(): return f,p
    d=Path(str(row.get('artifact_dir','') or ''))
    for n in ['step_telemetry.csv','step_records.jsonl','phase_cues.csv','episode_manifest.json']:
        p=d/n
        if p.exists(): return n,str(p)
    return '',''

def load_rows(path:str):
    p=Path(path)
    if not p.exists(): return []
    if p.suffix=='.csv': return r_csv(p)
    if p.suffix in {'.jsonl','.jl'}:
        out=[]
        with p.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj=json.loads(line)
                    if isinstance(obj,dict): out.append(obj)
        return out
    return []

def fl(v):
    try: return float(v)
    except Exception: return math.nan

def series(rows,fields):
    for f in fields:
        vals=[fl(r.get(f)) for r in rows]
        if sum(math.isfinite(x) for x in vals)>=max(3,int(.2*len(rows))): return f,vals
    return '',[math.nan]*len(rows)

def pct(vals,q):
    xs=sorted(x for x in vals if math.isfinite(x))
    if not xs: return math.nan
    if len(xs)==1: return xs[0]
    pos=(len(xs)-1)*q; lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)

def sm(mask,minrun):
    out=[False]*len(mask); start=None
    for i,v in enumerate(list(mask)+[False]):
        if v and start is None: start=i
        if not v and start is not None:
            if i-start>=minrun:
                for j in range(start,i): out[j]=True
            start=None
    return out

def onset(mask,gap):
    out=[]; prev=False; last=-10**9
    for i,v in enumerate(mask):
        if v and not prev and i-last>=gap: out.append(i); last=i
        prev=v
    return out

def direct_int(row,fields):
    for f in fields:
        x=parse_i(row.get(f))
        if x is not None: return x
    return None

def signature(path,args):
    rows=load_rows(path)
    if not rows: return {'status':'NO_ROWS','path':path}
    for r in rows:
        a=direct_int(r,DIRECT)
        if a is not None:
            ws=direct_int(r,W0); we=direct_int(r,W1)
            return {'status':'RESOLVED','method':'direct','field':'','anchor':a,'wstart':ws if ws is not None else a-args.teacher_window_pre,'wend':we if we is not None else a+args.teacher_window_post,'rows':len(rows),'path':path,'file_sha256':sha(Path(path))}
    stepf,stepsv=series(rows,STEP); steps=[int(x) if math.isfinite(x) else i for i,x in enumerate(stepsv)] if stepf else list(range(len(rows)))
    field,vals=series(rows,CMD); method='command_quantile'
    if not field:
        field,op=series(rows,OPEN)
        if not field: return {'status':'NO_GRIPPER_SIGNAL','path':path,'rows':len(rows)}
        vals=[op[i]-(op[i-1] if i>0 and math.isfinite(op[i-1]) else op[i]) for i in range(len(op))]; method='opening_delta_quantile'
    low=pct(vals,args.activity_quantile)
    if not math.isfinite(low): return {'status':'NO_FINITE_SIGNAL','path':path,'rows':len(rows),'field':field}
    active=sm([math.isfinite(v) and v<=low for v in vals],args.min_activity_run)
    ons=onset(active,args.min_segment_gap)
    if not ons: return {'status':'NO_ACTIVITY_ONSET','path':path,'rows':len(rows),'field':field,'method':method}
    ix=ons[-1] if args.primary_policy=='last_segment' else ons[0]
    a=int(steps[min(len(rows)-1,ix+args.anchor_offset)])
    return {'status':'RESOLVED','method':'heuristic_'+method,'field':field,'anchor':a,'wstart':a-args.teacher_window_pre,'wend':a+args.teacher_window_post,'rows':len(rows),'path':path,'file_sha256':sha(Path(path))}

def checksums(out:Path):
    reported={n:sha(out/n) for n in OUT_FILES[:-1] if (out/n).exists()}
    w_json(out/'checksum_report.json',{'algorithm':'sha256','reported_files':reported,'self_referential_checksum_fields':'ABSENT_BY_DESIGN'})
    present=[n for n in OUT_FILES if (out/n).exists()]
    sums=out/'SHA256SUMS'; sums.write_text(''.join(f"{sha(out/n)}  {n}\n" for n in present),encoding='utf-8')
    (out/'SHA256SUMS.sha256').write_text(f"{sha(sums)}  SHA256SUMS\n",encoding='utf-8')

def run(args):
    out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    src=r_csv(Path(args.clean2000_records)); inv=r_csv(Path(args.artifact_inventory)); prior=r_csv(Path(args.partial_bindings))
    byid={r['record_id']:r for r in prior}
    grouped=defaultdict(list); target_s=set(args.target_suite)
    for row in inv:
        s,t,st,k=key_art(row)
        if s not in target_s or t is None or st is None: continue
        kind,path=temporal(row)
        if not kind: continue
        new=dict(row); new.update({'structured_key':k,'temporal_kind':kind,'temporal_path':path})
        grouped[(s,t,st)].append(new)
    reconciled=list(prior); failures=[]; sigrows=[]
    for row in src:
        rid0=rid(row); s,t,st,k=key_record(row); stt=status(row)
        if s not in target_s or rid0 in byid: continue
        cands=grouped.get((s,t,st),[]) if t is not None and st is not None else []
        sigs=[]
        for c in cands:
            sg=signature(c['temporal_path'],args); sg.update({'record_id':rid0,'suite':s,'structured_key':k,'source_status':stt,'artifact_dir':c.get('artifact_dir',''),'temporal_kind':c.get('temporal_kind',''),'temporal_path':c.get('temporal_path','')}); sigs.append(sg); sigrows.append(sg)
        resolved=[x for x in sigs if x.get('status')=='RESOLVED']
        if stt=='NO_EVENT' and args.allow_no_event_without_artifact:
            reconciled.append({'record_id':rid0,'suite':s,'structured_key':k,'source_status':stt,'artifact_dir':'','selection_note':'no_event_without_artifact','temporal_kind':'','temporal_path':'','temporal_columns':'','signal_like_columns':''}); continue
        chosen=None; note=''
        if len(resolved)==1:
            chosen=resolved[0]; note='selected_by_unique_resolvable_candidate'
        elif len(resolved)>1:
            sigkey=lambda x:(x.get('anchor'),x.get('wstart'),x.get('wend'),x.get('method'),x.get('field'))
            keys={sigkey(x) for x in resolved}
            if len(keys)==1:
                chosen=sorted(resolved,key=lambda x:str(x.get('artifact_dir','')))[0]; note='selected_by_equivalent_anchor_signature'
            else:
                failures.append({'record_id':rid0,'suite':s,'structured_key':k,'source_status':stt,'failure_reason':'MULTIPLE_DISTINCT_RESOLVABLE_CANDIDATES','candidate_count':len(cands),'resolved_count':len(resolved)})
        else:
            failures.append({'record_id':rid0,'suite':s,'structured_key':k,'source_status':stt,'failure_reason':'NO_RESOLVABLE_CANDIDATE','candidate_count':len(cands),'resolved_count':0})
        if chosen:
            reconciled.append({'record_id':rid0,'suite':s,'structured_key':k,'source_status':stt,'artifact_dir':chosen.get('artifact_dir',''),'selection_note':note,'temporal_kind':chosen.get('temporal_kind',''),'temporal_path':chosen.get('temporal_path',''),'temporal_columns':'','signal_like_columns':chosen.get('field','')})
    pos_fail=[f for f in failures if f.get('source_status')=='SOURCE_POSITIVE']
    status_out=PASS if len(reconciled)==args.expected_records and not failures else 'HOLD_SINGLE_EVENT_ARTIFACT_AMBIGUITY_RECONCILIATION_INCOMPLETE'
    reason='' if status_out==PASS else f"reconciled={len(reconciled)} expected={args.expected_records} failures={len(failures)} positive_failures={len(pos_fail)}"
    fields=['record_id','suite','structured_key','source_status','artifact_dir','selection_note','temporal_kind','temporal_path','temporal_columns','signal_like_columns']
    w_csv(out/'single_event_artifact_reconciled_bindings.csv',reconciled,fields)
    w_csv(out/'single_event_artifact_reconciliation_failures.csv',failures,['record_id','suite','structured_key','source_status','failure_reason','candidate_count','resolved_count'])
    sigfields=['record_id','suite','structured_key','source_status','artifact_dir','temporal_kind','temporal_path','status','method','field','anchor','wstart','wend','rows','file_sha256']
    w_csv(out/'single_event_artifact_reconciliation_candidate_signatures.csv',sigrows,sigfields)
    rep={'gate':GATE,'status':status_out,'reason':reason,'clean2000_records':args.clean2000_records,'clean2000_records_sha256':sha(Path(args.clean2000_records)),'artifact_inventory':args.artifact_inventory,'artifact_inventory_sha256':sha(Path(args.artifact_inventory)),'partial_bindings':args.partial_bindings,'partial_bindings_sha256':sha(Path(args.partial_bindings)),'expected_records':args.expected_records,'reconciled_count':len(reconciled),'failure_count':len(failures),'positive_failure_count':len(pos_fail),'selection_note_counts':dict(Counter(r.get('selection_note','') for r in reconciled)),'source_status_counts':dict(Counter(r.get('source_status','') for r in reconciled)),'failures_by_reason':dict(Counter(f['failure_reason'] for f in failures)),'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'interpretation':'CPU-only ambiguity reconciliation. Ambiguous artifacts are accepted only when exactly one candidate resolves, or all resolvable candidates share an identical anchor signature.','boundaries':{'CUDA_required':'NOT_REQUIRED','OpenVLA_model':'NOT_LOADED','model_inference':'NOT_PERFORMED','LIBERO_runtime':'NOT_PERFORMED','env_step':'NOT_PERFORMED','rollout':'NOT_PERFORMED','intervention':'NOT_PERFORMED','attack_condition':'NOT_PERFORMED','detector_training':'NOT_PERFORMED'},'git_commit':args.git_commit,'files_changed':args.files_changed,'tests':args.tests}
    w_json(out/'single_event_artifact_reconciliation_report.json',rep); checksums(out); print(json.dumps(rep,sort_keys=True)); return 0 if not status_out.startswith('HOLD_') else 2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean2000-records',required=True); p.add_argument('--artifact-inventory',required=True); p.add_argument('--partial-bindings',required=True); p.add_argument('--target-suite',action='append',default=SINGLE); p.add_argument('--expected-records',type=int,default=1500); p.add_argument('--allow-no-event-without-artifact',action='store_true'); p.add_argument('--primary-policy',choices=['last_segment','first_segment'],default='last_segment'); p.add_argument('--activity-quantile',type=float,default=.10); p.add_argument('--min-activity-run',type=int,default=2); p.add_argument('--min-segment-gap',type=int,default=12); p.add_argument('--anchor-offset',type=int,default=8); p.add_argument('--teacher-window-pre',type=int,default=3); p.add_argument('--teacher-window-post',type=int,default=12); p.add_argument('--output-root',required=True); p.add_argument('--git-commit',required=True); p.add_argument('--files-changed',action='append',default=[]); p.add_argument('--tests',action='append',default=[]); return run(p.parse_args())
if __name__=='__main__': raise SystemExit(main())
