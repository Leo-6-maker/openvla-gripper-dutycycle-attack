import json, sys
from collections import defaultdict

path = sys.argv[1]
records = [json.loads(l) for l in open(path)]
ep_stats = defaultdict(lambda: {"crit_vals": [], "veto_vals": [], "target_veto": False, "target_crit": False})

for r in records:
    cid = r["identity"]
    ep_stats[cid]["crit_vals"].append(r["criticality_prob"])
    vp = r.get("veto_prob")
    if vp is not None:
        ep_stats[cid]["veto_vals"].append(vp)
    if r["target_veto"]:
        ep_stats[cid]["target_veto"] = True
    if r["target_criticality"]:
        ep_stats[cid]["target_crit"] = True

print("=== HARD-NEGATIVE EPISODES (veto target, no crit target) ===")
for cid, s in sorted(ep_stats.items()):
    if s["target_veto"] and not s["target_crit"]:
        crit_max = max(s["crit_vals"])
        crit_mean = sum(s["crit_vals"]) / len(s["crit_vals"])
        veto_max = max(s["veto_vals"]) if s["veto_vals"] else 0
        veto_mean = sum(s["veto_vals"]) / len(s["veto_vals"]) if s["veto_vals"] else 0
        veto_high = sum(1 for v in s["veto_vals"] if v > 0.5)
        n_emit_crit = sum(1 for v in s["crit_vals"] if v >= 0.5)
        print("  %s" % cid)
        print("    criticality: max=%.4f mean=%.4f  emit_steps=%d/%d" % (
            crit_max, crit_mean, n_emit_crit, len(s["crit_vals"])))
        print("    veto:        max=%.4f mean=%.4f  >0.5=%d/%d" % (
            veto_max, veto_mean, veto_high, len(s["veto_vals"])))

print("\n=== VALID-RETENTION EPISODES (sampled) ===")
n_shown = 0
for cid, s in sorted(ep_stats.items()):
    if s["target_crit"] and not s["target_veto"]:
        crit_max = max(s["crit_vals"])
        veto_max = max(s["veto_vals"]) if s["veto_vals"] else 0
        n_emit_crit = sum(1 for v in s["crit_vals"] if v >= 0.5)
        print("  %s: crit max=%.4f  emit_steps=%d/%d  veto max=%.4f" % (
            cid, crit_max, n_emit_crit, len(s["crit_vals"]), veto_max))
        n_shown += 1
        if n_shown >= 5:
            break
