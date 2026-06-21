#!/usr/bin/env python3
"""A800-F32-S: FP32 single-GPU ablation (MIG2C)."""
import os, json, csv, numpy as np, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from transformers import AutoModelForVision2Seq

MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"
BUNDLE = "/mnt/sdc/dty_user/openvla_attack/migration_audit/parity/cross_host_bundle"
OUT = "/mnt/sdc/dty_user/openvla_attack/migration_audit/parity/a800_fp32_single_results.csv"

m = AutoModelForVision2Seq.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager",
    local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map="cuda:0",
)
DEV = next(m.parameters()).device
print("Devices:", sorted(set(str(p.device) for p in m.parameters())))

with open(BUNDLE + "/cross_host_bundle_manifest.json") as f:
    manifest = json.load(f)

action_dim = m.get_action_dim("libero_spatial")
stats = m.get_action_stats("libero_spatial")
mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
hi = np.array(stats["q99"]); lo = np.array(stats["q01"])

with open(OUT, "w", newline="") as cf:
    w = csv.writer(cf)
    w.writerow(["episode", "step", "frame_file", "run",
                "generated_token_ids", "final_action", "gripper_class",
                "repeat_deterministic"])

    for entry in manifest["frames"]:
        fn = entry["file"]
        data = torch.load(BUNDLE + "/" + fn + ".cpu_tensors.pt", map_location="cpu")
        ids = data["input_ids"]; px = data["pixel_values"].to(dtype=torch.float32, device=DEV)

        actions = []; tokens_all = []
        for r in range(3):
            ids_m = ids.clone()
            if ids_m[0, -1].item() != 29871:
                ids_m = torch.cat((ids_m, torch.tensor([[29871]], dtype=ids_m.dtype)), dim=1)
            ids_m = ids_m.to(device=DEV)

            gen = m.generate(input_ids=ids_m, pixel_values=px,
                max_new_tokens=action_dim, do_sample=False, pad_token_id=m.pad_token_id)
            act_tokens = gen[0, -action_dim:].cpu().numpy()
            tokens_all.append(act_tokens.tolist())

            disc = m.vocab_size - act_tokens
            disc = np.clip(disc - 1, 0, m.bin_centers.shape[0] - 1)
            norm = m.bin_centers[disc]
            act = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
            actions.append(act)

        det = all(np.array_equal(actions[0], a) for a in actions[1:])

        for r in range(3):
            rg = actions[r][6]; ng = (rg * 2) - 1; ig = -(1.0 if ng >= 0 else -1.0)
            cls = "OPEN" if ig < 0 else "CLOSE"
            w.writerow([entry["episode"], entry["step"], fn, r + 1,
                " ".join(str(t) for t in tokens_all[r]),
                " ".join(f"{x:.12f}" for x in actions[r].tolist()),
                cls, det])

        ep_name = entry["episode"]; st = entry["step"]
        print(f"{ep_name}_step{st}: det={det} tokens={tokens_all[0]}")

print("Done:", OUT)
