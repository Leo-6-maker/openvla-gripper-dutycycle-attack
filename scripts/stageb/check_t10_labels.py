import json, sys

path = sys.argv[1]
records = [json.loads(l) for l in open(path)]
cid = path.split("/")[-3] + "/" + path.split("/")[-2] + "/" + path.split("/")[-1].replace("_records.jsonl","")

n_t10 = sum(1 for r in records if r.get("retention_continuation_t10") and not r.get("retention_unknown_mask"))
n_active = sum(1 for r in records if r.get("retention_active") and not r.get("retention_unknown_mask"))
n_grasp = sum(1 for r in records if r.get("grasp_support") and not r.get("retention_unknown_mask"))
n_close = sum(1 for r in records if r.get("event_close_onset"))
n_support = sum(1 for r in records if r.get("event_support"))
n_release = sum(1 for r in records if r.get("event_release_onset"))

print("CID:", cid)
print("n_t10=%d n_active=%d n_grasp=%d n_close=%d n_support=%d n_release=%d" % (
    n_t10, n_active, n_grasp, n_close, n_support, n_release))

# Show close events
for r in records:
    if r.get("event_close_onset"):
        onset = r["step"]
        end = r.get("event_end_step", -1)
        print("CLOSE: onset=%d end=%d dur=%d" % (onset, end, end - onset + 1 if end >= onset else 0))
        # Show steps around close event
        for s in range(max(0, onset-3), min(len(records), end+4)):
            rr = records[s]
            print("  step=%d t10=%s active=%s support=%s release=%s grasp=%s close_onset=%s" % (
                rr["step"], rr.get("retention_continuation_t10"),
                rr.get("retention_active"), rr.get("event_support"),
                rr.get("event_release_onset"), rr.get("grasp_support"),
                rr.get("event_close_onset")))
