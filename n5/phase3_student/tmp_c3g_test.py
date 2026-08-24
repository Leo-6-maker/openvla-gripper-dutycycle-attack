"""Quick test of C3-G on one episode."""
import sys, json
sys.path.insert(0, '.')
sys.path.insert(0, '../phase2_labels')
from c3g_evaluate_dev import load_static_seal, evaluate_episode

static_targets, basket_seal, white_wooden = load_static_seal()
print("Static targets:", len(static_targets))
print("Basket seal keys (first 3):", sorted(basket_seal.keys())[:3])
print("White/wooden:", white_wooden)

# Test on a single basket episode
for ident in [
    "libero_object/task_00/state_00",
    "libero_10/task_00/state_00",
    "libero_10/task_06/state_00",
]:
    result, error = evaluate_episode(ident, static_targets, basket_seal, white_wooden)
    if error:
        print(f"{ident}: ERROR: {error}")
    else:
        print(f"{ident}: success={result['success']} T={result['T']} "
              f"placed_steps={result['n_placed_steps']} pregrasp_fp={result['n_pregrasp_fp_steps']} "
              f"relations={result['relation_types']}")
        for i, step in enumerate(result['per_step']):
            if step['any_relation_true']:
                print(f"  First TRUE at step {i}:", json.dumps(step['relation_results'], indent=2)[:200])
                break
        # Show UNKNOWN count
        n_unknown = sum(1 for s in result['per_step'] for r in s.get('relation_results',{}).values() if r['truth']=='UNKNOWN')
        n_false = sum(1 for s in result['per_step'] for r in s.get('relation_results',{}).values() if r['truth']=='FALSE')
        n_true = sum(1 for s in result['per_step'] for r in s.get('relation_results',{}).values() if r['truth']=='TRUE')
        print(f"  TRUE={n_true} FALSE={n_false} UNKNOWN={n_unknown}")
