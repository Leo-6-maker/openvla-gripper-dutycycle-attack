#!/usr/bin/env python3
"""V3 generation parity diagnostic.

Compares four forward paths on a fixed adversarial image + prompt:
  Path A: Official generate(use_cache=True)        — source of truth
  Path B: No-cache full forward (current surrogate) — differentiable
  Path C: Cache full forward (use_cache=True)       — cache parity test
  Path D: Generate(use_cache=False)                 — generation config test

Usage:
  python diagnose_v3_generation_parity.py \\
    --model /path/to/model \\
    --replay-dir /path/to/v3_parity_dumps \\
    --output /path/to/output.json

Reads replay bundles produced by runner with --v3_parity_dump_dir.
"""

import argparse, json, os, sys

import numpy as np
import torch

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_model(model_path):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    visible = torch.cuda.device_count()
    max_memory = {idx: '10000MiB' for idx in range(max(visible, 1))}
    max_memory['cpu'] = '128GiB'
    model = AutoModelForVision2Seq.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='eager', device_map='auto', max_memory=max_memory)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    return model, processor


def run_path_a(model, input_ids, pixel_values, action_dim):
    """Path A: Official generate with cache (source of truth)."""
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=True)
    prompt_len = int(input_ids.shape[1])
    new_ids = gen.sequences[0, prompt_len:]
    scores = gen.scores[-1][0]  # final step
    argmax = int(torch.argmax(scores).cpu())
    top2 = torch.topk(scores, 2)
    return {
        'path': 'A_generate_cache',
        'generated_tokens': [int(t) for t in new_ids.cpu()],
        'score_argmax': argmax,
        'argmax_matches_generated': argmax == int(new_ids[-1]),
        'top1_score': float(top2.values[0].cpu()),
        'top2_score': float(top2.values[1].cpu()),
        'top1_top2_gap': float((top2.values[0] - top2.values[1]).cpu()),
    }


def run_path_b(model, input_ids, prefix_ids, pixel_values):
    """Path B: No-cache full forward (current v3 surrogate)."""
    context = torch.cat([input_ids, prefix_ids.view(1, -1).to(input_ids.device)], dim=1)
    with torch.inference_mode():
        out = model(input_ids=context, pixel_values=pixel_values,
                    use_cache=False, return_dict=True)
    logits = out.logits.float()[0, -1, :]
    argmax = int(torch.argmax(logits).cpu())
    top2 = torch.topk(logits, 2)
    return {
        'path': 'B_nocache_forward',
        'predicted_top_token': argmax,
        'top1_logit': float(top2.values[0].cpu()),
        'top2_logit': float(top2.values[1].cpu()),
        'top1_top2_gap': float((top2.values[0] - top2.values[1]).cpu()),
    }


def run_path_c(model, input_ids, prefix_ids, pixel_values):
    """Path C: Cache forward (use_cache=True)."""
    context = torch.cat([input_ids, prefix_ids.view(1, -1).to(input_ids.device)], dim=1)
    with torch.inference_mode():
        out = model(input_ids=context, pixel_values=pixel_values,
                    use_cache=True, return_dict=True)
    logits = out.logits.float()[0, -1, :]
    argmax = int(torch.argmax(logits).cpu())
    top2 = torch.topk(logits, 2)
    return {
        'path': 'C_cache_forward',
        'predicted_top_token': argmax,
        'top1_logit': float(top2.values[0].cpu()),
        'top2_logit': float(top2.values[1].cpu()),
        'top1_top2_gap': float((top2.values[0] - top2.values[1]).cpu()),
    }


def run_path_d(model, input_ids, pixel_values, action_dim):
    """Path D: Generate with use_cache=False."""
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False,
            use_cache=False, return_dict_in_generate=True,
            output_scores=True)
    prompt_len = int(input_ids.shape[1])
    new_ids = gen.sequences[0, prompt_len:]
    scores = gen.scores[-1][0]
    argmax = int(torch.argmax(scores).cpu())
    top2 = torch.topk(scores, 2)
    return {
        'path': 'D_generate_nocache',
        'generated_tokens': [int(t) for t in new_ids.cpu()],
        'score_argmax': argmax,
        'argmax_matches_generated': argmax == int(new_ids[-1]),
        'top1_score': float(top2.values[0].cpu()),
        'top2_score': float(top2.values[1].cpu()),
        'top1_top2_gap': float((top2.values[0] - top2.values[1]).cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--replay-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    model, processor = load_model(args.model)
    action_dim = int(model.get_action_dim('libero_object'))
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    results = []
    for fn in sorted(os.listdir(args.replay_dir)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(args.replay_dir, fn)) as f:
            bundle = json.load(f)

        pt_fn = fn.replace('.json', '_adv_pv.pt')
        pt_path = os.path.join(args.replay_dir, pt_fn)
        if not os.path.exists(pt_path):
            continue

        pixel_values = torch.load(pt_path).to(device=device, dtype=model_dtype)
        input_ids = torch.tensor(bundle['prompt_input_ids'], device=device, dtype=torch.long)
        prefix_ids = torch.tensor(bundle['generated_arm_prefix'], device=device, dtype=torch.long)

        entry = {'file': fn, 'task': bundle.get('task', ''), 'step': bundle.get('step', '')}

        # Path A: Official generate
        entry['A'] = run_path_a(model, input_ids, pixel_values, action_dim)

        # Path B: No-cache surrogate
        entry['B'] = run_path_b(model, input_ids, prefix_ids, pixel_values)

        # Path C: Cache forward
        entry['C'] = run_path_c(model, input_ids, prefix_ids, pixel_values)

        # Path D: Generate no-cache
        entry['D'] = run_path_d(model, input_ids, pixel_values, action_dim)

        # Key diagnostics
        a_token = entry['A']['generated_tokens'][-1]
        b_token = entry['B']['predicted_top_token']
        c_token = entry['C']['predicted_top_token']
        d_token = entry['D']['generated_tokens'][-1]

        entry['diagnosis'] = {
            'A_official_token': a_token,
            'B_surrogate_token': b_token,
            'C_cache_token': c_token,
            'D_generate_nocache_token': d_token,
            'B_matches_A': b_token == a_token,
            'C_matches_A': c_token == a_token,
            'D_matches_A': d_token == a_token,
            'A_score_argmax_matches': entry['A']['argmax_matches_generated'],
            'D_score_argmax_matches': entry['D']['argmax_matches_generated'],
        }

        results.append(entry)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results: {len(results)} entries -> {args.output}')
    for r in results:
        d = r['diagnosis']
        print(f'  {r["file"]}: A={d["A_official_token"]} B={d["B_surrogate_token"]} '
              f'B==A={d["B_matches_A"]} C==A={d["C_matches_A"]} '
              f'D==A={d["D_matches_A"]} A_argmax_ok={d["A_score_argmax_matches"]}')


if __name__ == '__main__':
    main()
