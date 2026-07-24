#!/usr/bin/env python3
"""2080Ti Matched Profile — Spatial Static Parity (MIG2B).
FP32, 4-GPU sharding (10GB cap each), device_map=auto, eager.
Actual execution script that produced 2080ti_M_results.csv on 2026-06-21."""

import os, json, numpy as np, torch, csv, time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from transformers import AutoModelForVision2Seq

MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial'
BUNDLE = '/data/liuyu/outputs/cross_host_bundle'
OUT = '/data/liuyu/outputs/parity_2080ti/2080ti_M_results.csv'

os.makedirs(os.path.dirname(OUT), exist_ok=True)

m = AutoModelForVision2Seq.from_pretrained(
    MODEL, torch_dtype=torch.float32,
    local_files_only=True, trust_remote_code=True,
    device_map='auto',
    max_memory={0: '10000MiB', 1: '10000MiB', 2: '10000MiB', 3: '10000MiB'},
)
print('Devices:', sorted(set(str(p.device) for p in m.parameters())))

with open(BUNDLE + '/cross_host_bundle_manifest.json') as f:
    manifest = json.load(f)

with open(OUT, 'w', newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['episode','task_name','step','frame_file','prompt_token_ids',
                'generated_token_ids','final_action','gripper_class','inference_time_s'])

    for entry in manifest['frames']:
        fn = entry['file']
        prompt_ids_str = ' '.join(str(x) for x in entry['prompt_token_ids'])
        d = torch.load(BUNDLE + '/' + fn + '.cpu_tensors.pt', map_location='cpu')

        t0 = time.time()
        r = m.predict_action(
            input_ids=d['input_ids'],
            pixel_values=d['pixel_values'],
            unnorm_key='libero_spatial',
            do_sample=False,
        )
        dt = time.time() - t0

        vals = np.array(r).flatten()
        rg = vals[6]; ng = (rg * 2) - 1; ig = -(1.0 if ng >= 0 else -1.0)
        cls = 'OPEN' if ig < 0 else 'CLOSE'

        w.writerow([
            entry['episode'], entry['task_name'], entry['step'], fn,
            prompt_ids_str, 'internal_predict_action',
            ' '.join(f'{x:.12f}' for x in vals.tolist()),
            cls, f'{dt:.4f}'
        ])
        print(f'{fn}: {cls} dt={dt:.3f}s')

print('Done. Output:', OUT)
