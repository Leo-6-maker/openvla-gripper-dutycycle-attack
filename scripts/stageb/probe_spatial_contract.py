#!/usr/bin/env python3
"""CPU-only contract probe for LIBERO Spatial suite.

Verifies: suite availability, task inventory, model compatibility,
observation contract, action contract, preprocessing contract.
Outputs: reports + tables for audit.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from gripper_attack.libero_suite_registry import (
    probe_suite_contract, probe_model_contract, list_available_suites,
)


def main():
    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 1. List all available suites
    suites = list_available_suites()
    print(f"Available suites: {suites}")

    # 2. Probe Spatial contract
    spatial_contract = probe_suite_contract("libero_spatial")
    print(f"\n=== Spatial Contract ===")
    print(f"Available: {spatial_contract['available']}")
    print(f"Tasks: {spatial_contract['n_tasks']}")
    print(f"Total init states: {spatial_contract['total_init_states']}")
    if spatial_contract["errors"]:
        print(f"Errors: {spatial_contract['errors']}")
    for td in spatial_contract.get("task_details", []):
        print(f"  {td['name']}: {td['n_states']} states, bddl={td['bddl_exists']}")

    with open("artifacts/libero_spatial_contract_probe.json", "w") as f:
        json.dump(spatial_contract, f, indent=2, default=str)

    # 3. Probe model contract
    model_paths = []
    # Check for suite-matched Spatial checkpoint
    spatial_ckpt = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial"
    object_ckpt = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"

    for mp in [spatial_ckpt, object_ckpt]:
        if os.path.exists(mp):
            mc = probe_model_contract(mp, "libero_spatial")
            print(f"\n=== Model: {mp} ===")
            print(f"  Loadable: {mc['model_loadable']}")
            print(f"  Unnorm keys: {mc['unnorm_keys']}")
            print(f"  Expected: {mc['expected_unnorm_key']}")
            print(f"  Match: {mc['unnorm_available']}")
            if mc["errors"]:
                print(f"  Errors: {mc['errors']}")
            model_paths.append(mc)

    with open("artifacts/libero_spatial_model_contract.json", "w") as f:
        json.dump(model_paths, f, indent=2, default=str)

    # 4. Compare with Object suite for reference
    object_contract = probe_suite_contract("libero_object")
    print(f"\n=== Object Suite (reference) ===")
    print(f"Tasks: {object_contract['n_tasks']}, States: {object_contract['total_init_states']}")

    # 5. Write task inventory CSV
    with open("tables/libero_spatial_task_inventory.csv", "w") as f:
        f.write("suite,task_name,n_init_states,language,bddl_path\n")
        for td in spatial_contract.get("task_details", []):
            f.write(f"libero_spatial,{td['name']},{td['n_states']},{td['language']},{td['bddl_path']}\n")

    # 6. Recommendations
    has_spatial_model = any(m["unnorm_available"] for m in model_paths)
    has_object_model = os.path.exists(object_ckpt)
    print(f"\n=== Recommendations ===")
    if spatial_contract["available"] and has_spatial_model:
        print("SUITE_MATCHED_MODEL_AVAILABLE — run full Spatial baseline")
    elif spatial_contract["available"] and has_object_model:
        print("OBJECT_CHECKPOINT_ONLY — run infrastructure probes, label as CROSS_SUITE_ZERO_SHOT")
    else:
        print("MISSING_DEPENDENCIES — cannot proceed")
        return 1

    # Probe gripper/preprocessing contract
    print("\n=== Gripper Convention ===")
    print("raw > 0.5 → env = -1 → physical OPEN")
    print("raw < 0.5 → env = +1 → physical CLOSE")
    print("decoded_open = 1 if raw > 0.5 else 0")
    print("UNVERIFIED for Spatial — assumes same convention as Object")

    print(f"\nOutput: reports/, tables/, artifacts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
