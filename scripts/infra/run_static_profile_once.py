#!/usr/bin/env python3
"""F1R: Single-process BF16 static matrix with proper VRAM/latency measurement."""
import os, json, csv, time, numpy as np, argparse
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", required=True, choices=["eager", "flash_attention_2"])
    p.add_argument("--cuda_devices", default="6")
    p.add_argument("--bundle_dir", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--num_repeats", type=int, default=3)
    p.add_argument("--warmup_repeats", type=int, default=5)
    p.add_argument("--timing_repeats", type=int, default=10)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    import torch
    from transformers import AutoModelForVision2Seq

    M0_alloc = torch.cuda.memory_allocated()
    M0_resv = torch.cuda.memory_reserved()

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
        device_map="cuda:0", local_files_only=True, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    M1_alloc = torch.cuda.memory_allocated()
    M1_resv = torch.cuda.memory_reserved()
    DEV = next(model.parameters()).device

    actual_attn = getattr(model.config, "_attn_implementation", "unknown")
    action_dim = model.get_action_dim("libero_spatial")
    stats = model.get_action_stats("libero_spatial")
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])

    with open(args.bundle_dir + "/bundle_manifest.json") as f:
        manifest = json.load(f)

    # Warmup (on first frame only)
    entry0 = manifest["frames"][0]
    data0 = torch.load(args.bundle_dir + "/" + entry0["file"] + ".cpu_tensors.pt", map_location="cpu")
    ids0 = data0["input_ids"].to(device=DEV)
    px0 = data0["pixel_values"].to(dtype=torch.bfloat16, device=DEV)
    if ids0[0, -1].item() != 29871:
        ids0 = torch.cat((ids0, torch.tensor([[29871]], dtype=ids0.dtype, device=DEV)), dim=1)

    for _ in range(args.warmup_repeats):
        _ = model.generate(input_ids=ids0.clone(), pixel_values=px0.clone(),
                           max_new_tokens=action_dim, do_sample=False, pad_token_id=model.pad_token_id)

    # Reset peak memory after warmup
    torch.cuda.reset_peak_memory_stats()

    rows = []
    latencies_all = []

    for entry in manifest["frames"]:
        data = torch.load(args.bundle_dir + "/" + entry["file"] + ".cpu_tensors.pt", map_location="cpu")
        ids = data["input_ids"].to(device=DEV)
        px = data["pixel_values"].to(dtype=torch.bfloat16, device=DEV)
        if ids[0, -1].item() != 29871:
            ids = torch.cat((ids, torch.tensor([[29871]], dtype=ids.dtype, device=DEV)), dim=1)

        actions = []
        tokens_list = []
        frame_lats = []

        for r in range(args.num_repeats):
            # Timing with proper CUDA sync
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = model.generate(input_ids=ids.clone(), pixel_values=px.clone(),
                                 max_new_tokens=action_dim, do_sample=False,
                                 pad_token_id=model.pad_token_id)
            torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            if r == 0:
                frame_lats.append(lat)  # record all runs for this frame

            tokens = gen[0, -action_dim:].cpu().numpy()
            disc = model.vocab_size - tokens
            disc = np.clip(disc - 1, 0, model.bin_centers.shape[0] - 1)
            norm = model.bin_centers[disc]
            act = np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)
            actions.append(act)
            tokens_list.append(tokens.tolist())

        det = all(np.array_equal(actions[0], a) for a in actions[1:])
        rg = actions[0][6]; ng = (rg * 2) - 1; ig = -(1.0 if ng >= 0 else -1.0)
        cls = "OPEN" if ig < 0 else "CLOSE"
        rows.append({
            "attn": args.attn, "actual_attn": actual_attn,
            "episode": entry["episode"], "step": entry["step"],
            "fn": entry["file"], "det": det,
            "tokens": " ".join(str(x) for x in tokens_list[0]),
            "action": " ".join("%.12f" % x for x in actions[0].tolist()),
            "gripper_class": cls,
            "latency_median_s": "%.6f" % np.median(frame_lats) if frame_lats else "0",
        })
        latencies_all.extend(frame_lats)

    M2_alloc = torch.cuda.max_memory_allocated()
    M2_resv = torch.cuda.max_memory_reserved()

    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    lat_arr = np.array(latencies_all)
    print("attn=%s actual=%s" % (args.attn, actual_attn))
    print("VRAM: load_delta=%.2fGiB load_reserved=%.2fGiB inference_peak=%.2fGiB inference_reserved=%.2fGiB" % (
        (M1_alloc - M0_alloc) / 1024**3, (M1_resv - M0_resv) / 1024**3,
        (M2_alloc - M1_alloc) / 1024**3, (M2_resv - M1_resv) / 1024**3))
    print("Latency: median=%.4fs p90=%.4fs p95=%.4fs n=%d" % (
        np.median(lat_arr), np.percentile(lat_arr, 90), np.percentile(lat_arr, 95), len(lat_arr)))
    print("Determinism: %d/%d" % (sum(1 for r in rows if r["det"] == "True"), len(rows)))
    print("Output: %s" % args.output_csv)


if __name__ == "__main__":
    main()
