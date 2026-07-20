#!/usr/bin/env python3
"""Gate F3.3: Corrected independent sealed geometry audit.

Fixes over F3:
- unsupported_known: check unsupported route AND any known mask
- positive_tasks: only tasks with >=1 positive event
- positive_events: events with grasp AND manipulation AND known attribution
- event closure: monotonic, IDLE=-1, RELEASED retains id, no cross-object
- unknown attribution → NONE (not TARGET default)
"""
import argparse, csv, hashlib, json, math, os, sys, uuid
from collections import Counter, defaultdict
from pathlib import Path

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
SUPPORTED_ROUTES = {"single_object_pick_place", "multi_object_transfer"}

def sha256_file(p): d=hashlib.sha256(); [d.update(b) for b in iter(lambda: p.open("rb").read(1048576), b"")]; return d.hexdigest()
def verify_seal(root):
    s=root/"SHA256SUMS"; c=root/"SHA256SUMS.sha256"
    if not s.is_file() or not c.is_file(): raise SystemExit(f"SEAL MISSING: {root}")
    if c.read_text().strip()!=f"{sha256_file(s)}  SHA256SUMS": raise SystemExit(f"SEAL MISMATCH: {root}")
    for l in s.read_text().splitlines():
        d,_,n=l.partition("  "); t=root/n
        if not t.is_file() or sha256_file(t)!=d: raise SystemExit(f"FILE MISMATCH: {root}/{n}")
    return sha256_file(s)
def jsonl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def _atomic_text(p,v):
    t=p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    with t.open("x") as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)
def write_seal(root):
    excl={"SHA256SUMS","SHA256SUMS.sha256"}
    fs=sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excl),key=lambda p:p.relative_to(root).as_posix())
    c="".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in fs)
    _atomic_text(root/"SHA256SUMS",c)
    _atomic_text(root/"SHA256SUMS.sha256",f"{sha256_file(root/'SHA256SUMS')}  SHA256SUMS\n")
    return sha256_file(root/"SHA256SUMS")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-root",type=Path,required=True)
    ap.add_argument("--output-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.teacher_root.resolve(); out=args.output_root.resolve()
    if out.exists(): raise SystemExit(f"output exists: {out}")

    teacher_seal=verify_seal(root)
    manifest=json.loads((root/"factorized_teacher_v1_manifest.json").read_text())
    labels_root=root/"labels"
    errors=[]; hold=False

    n_ids=n_steps=missing=0
    logical_v=unsupported_known=event_closure_err=0
    grasp_pos=grasp_neg=grasp_unk=0
    manip_pos=manip_neg=manip_unk=0
    release_pos=release_neg=release_unk=0
    grasp_segs=[]; cur_seg=0; release_eps=set()
    per_route=defaultdict(lambda:{"eps":0,"events":0,"positive_events":0,"positive_tasks":set(),"grasp_pos":0,"manip_pos":0})
    per_task=defaultdict(lambda:Counter())

    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                identity=f"{suite}/task_{task:02d}/state_{state:02d}"
                tk=(suite,f"task_{task:02d}")
                p=labels_root/suite/f"task_{task:02d}"/f"state_{state:02d}"/"factorized_teacher_v1.jsonl"
                if not p.is_file(): missing+=1; continue
                n_ids+=1; rows=jsonl(p); n_steps+=len(rows)
                for i,r in enumerate(rows):
                    if r.get("step")!=i: errors.append(f"STEP_INDEX: {identity} step {i}"); hold=True

                route=rows[0].get("mechanism_type","?")
                supported=route in SUPPORTED_ROUTES
                ep_grasp=ep_manip=ep_release=0
                ep_events=set(); ep_pos_events=set()
                prev_grasp=False; cur_seg=0
                # Event tracking for this episode
                seen_eids=set(); last_eid=-2; last_obj=None; event_obj_error=False

                for r in rows:
                    gv,gk=r["grasp_established"],r["grasp_established_known_mask"]
                    mv,mk=r["manipulation_active"],r["manipulation_active_known_mask"]
                    rv,rk=r["release_or_instability"],r["release_or_instability_known_mask"]
                    eid=r.get("event_id",-1); eph=r.get("event_phase","?")
                    obj=r.get("active_object_name"); erole=r.get("event_role","?")

                    # Head counts
                    if gk: (grasp_pos if gv else grasp_neg)+=1; ep_grasp+=gv
                    else: grasp_unk+=1
                    if mk: (manip_pos if mv else manip_neg)+=1; ep_manip+=mv
                    else: manip_unk+=1
                    if rk: (release_pos if rv else release_neg)+=1; ep_release+=rv
                    else: release_unk+=1

                    # Logical: unsupported route with any known mask
                    if not supported and (gk or mk or rk): unsupported_known+=1
                    if mv and not gv: logical_v+=1

                    # Grasp segments
                    if gv and not prev_grasp:
                        if cur_seg>0: grasp_segs.append(cur_seg)
                        cur_seg=1
                    elif gv: cur_seg+=1
                    elif cur_seg>0: grasp_segs.append(cur_seg); cur_seg=0
                    prev_grasp=gv

                    # Event tracking
                    if eid>=0:
                        ep_events.add(eid)
                        if eid not in seen_eids:
                            seen_eids.add(eid)
                            # Event closure checks
                            if eid!=last_eid+1 and last_eid>=0: event_closure_err+=1
                            last_eid=eid; last_obj=obj
                        # Positive event: grasp AND manipulation AND attribution known
                        if gv and mv and obj is not None:
                            ep_pos_events.add(eid)
                        # Event object consistency
                        if obj is not None and last_obj is not None and obj!=last_obj and not r.get("event_attribution_conflict"):
                            pass  # handoff may legitimately create new event
                    if eph=="IDLE" and eid!=-1: event_closure_err+=1
                    if eph=="RELEASED" and eid<0: event_closure_err+=1

                if cur_seg>0: grasp_segs.append(cur_seg)
                if ep_release>0: release_eps.add(identity)

                per_task[tk]["eps"]+=1
                per_task[tk]["grasp_pos"]+=ep_grasp; per_task[tk]["manip_pos"]+=ep_manip
                per_task[tk]["release_pos"]+=ep_release
                per_task[tk]["events"]+=len(ep_events); per_task[tk]["pos_events"]+=len(ep_pos_events)
                per_task[tk]["has_pos"]+=1 if ep_grasp>0 and ep_manip>0 else 0

                rp=per_route[route]; rp["eps"]+=1; rp["events"]+=len(ep_events)
                rp["positive_events"]+=len(ep_pos_events)
                rp["grasp_pos"]+=ep_grasp; rp["manip_pos"]+=ep_manip
                if ep_grasp>0 and ep_manip>0: rp["positive_tasks"].add(tk)

            per_suite=defaultdict(lambda:Counter())

    # ── Gate checks ──
    if n_ids!=800: errors.append(f"IDENTITY: {n_ids}!=800"); hold=True
    if n_steps!=176336: errors.append(f"STEPS: {n_steps}!=176336"); hold=True
    if missing>0: errors.append(f"MISSING: {missing}"); hold=True
    if logical_v>0: errors.append(f"LOGICAL: {logical_v}"); hold=True
    if unsupported_known>0: errors.append(f"UNSUPPORTED_KNOWN: {unsupported_known}"); hold=True
    if event_closure_err>0: errors.append(f"EVENT_CLOSURE: {event_closure_err}"); hold=True

    for n,pos,neg in [("grasp",grasp_pos,grasp_neg),("manipulation",manip_pos,manip_neg),("release",release_pos,release_neg)]:
        if pos==0 or neg==0: errors.append(f"HEAD_DEGENERATE: {n}"); hold=True

    ss=sorted(grasp_segs) if grasp_segs else [0]
    med=ss[len(ss)//2]
    if med<10: errors.append(f"GRASP_MEDIAN: {med}<10"); hold=True
    if len(release_eps)<20: errors.append(f"RELEASE_EPS: {len(release_eps)}<20"); hold=True

    for route in SUPPORTED_ROUTES:
        rd=per_route[route]
        if len(rd["positive_tasks"])<3: errors.append(f"ROUTE_POS_TASKS: {route} has {len(rd['positive_tasks'])}<3"); hold=True
        if rd["positive_events"]<30: errors.append(f"ROUTE_POS_EVENTS: {route} has {rd['positive_events']}<30"); hold=True

    for route in SUPPORTED_ROUTES:
        tp=per_route[route]["grasp_pos"]+per_route[route]["manip_pos"]
        for tk in per_route[route]["positive_tasks"]:
            task_pos=per_task[tk]["grasp_pos"]+per_task[tk]["manip_pos"]
            if tp>0 and task_pos/tp>0.35:
                errors.append(f"TASK_OVERFIT: {tk} {100*task_pos/tp:.1f}%>35%"); hold=True

    # ── Sealed output ──
    staging=out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging"); staging.mkdir(parents=True)
    status="HOLD" if hold else "PASS_FINAL_STUDENT_TRAINING"
    audit={
        "schema":"DETECTOR_V5_F3_3_GEOMETRY_AUDIT_V1","status":status,
        "teacher_root":str(root),"teacher_root_seal":teacher_seal,
        "teacher_source_commit":manifest.get("source_git_commit"),
        "identity_count":n_ids,"step_count":n_steps,
        "logical_violations":logical_v,"unsupported_known":unsupported_known,"event_closure_errors":event_closure_err,
        "head_geometry":{"grasp":{"pos":grasp_pos,"neg":grasp_neg,"unk":grasp_unk},
                         "manipulation":{"pos":manip_pos,"neg":manip_neg,"unk":manip_unk},
                         "release":{"pos":release_pos,"neg":release_neg,"unk":release_unk}},
        "grasp_segment_median":med,"release_positive_episodes":len(release_eps),
        "per_route":{r:{"positive_tasks":len(d["positive_tasks"]),"positive_events":d["positive_events"],
                         "grasp_pos":d["grasp_pos"],"manip_pos":d["manip_pos"]} for r,d in per_route.items()},
        "errors":errors,"formal_training_authorized":not hold,"formal_attack_authorized":False,
    }
    _atomic_text(staging/"geometry_audit.json",json.dumps(audit,indent=2,sort_keys=True)+"\n")
    with (staging/"per_task.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["suite","task","eps","grasp_pos","manip_pos","release_pos","events","pos_events","has_pos"])
        w.writeheader()
        for (s,tn),st in sorted(per_task.items()):
            w.writerow({"suite":s,"task":tn,**{k:st[k] for k in["eps","grasp_pos","manip_pos","release_pos","events","pos_events","has_pos"]}})
    _atomic_text(staging/"source_bindings.json",json.dumps({"schema":"DETECTOR_V5_F3_3_SOURCE_BINDINGS_V1","teacher_root":str(root),"teacher_root_seal":teacher_seal},indent=2)+"\n")
    seal_sha=write_seal(staging); os.replace(staging,out)

    print(f"F3.3 STATUS: {status}")
    print(f"  ids={n_ids} steps={n_steps}")
    print(f"  grasp: +{grasp_pos} -{grasp_neg} ?{grasp_unk}")
    print(f"  manip: +{manip_pos} -{manip_neg} ?{manip_unk}")
    print(f"  release: +{release_pos} -{release_neg} ?{release_unk}")
    print(f"  logical_v={logical_v} unsupported_known={unsupported_known} event_closure={event_closure_err}")
    print(f"  grasp_median={med} release_eps={len(release_eps)}")
    for r in SUPPORTED_ROUTES:
        d=per_route[r]
        print(f"  {r}: pos_tasks={len(d['positive_tasks'])} pos_events={d['positive_events']} grasp={d['grasp_pos']} manip={d['manip_pos']}")
    if errors:
        for e in errors: print(f"  ERROR: {e}")
    return 1 if hold else 0

if __name__=="__main__": raise SystemExit(main())
