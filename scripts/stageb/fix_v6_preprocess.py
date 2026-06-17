#!/usr/bin/env python3
"""Fix V6 runner to use V4-aligned preprocessing."""
path = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py'

with open(path) as f:
    src = f.read()

# 1. Add V4 preprocess imports
old = "from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero"
new = "from v4_run_eval_openvla import (\n    decode_with_scores, postprocess_openvla_action_for_libero,\n    prompt, prepare_openvla_image)"
src = src.replace(old, new)
print('1. V4 preprocess imports added')

# 2. Wrap instruction with prompt()
old = "instruction = task_obj.language"
new = "instruction_raw = task_obj.language\ninstruction = prompt(instruction_raw)"
src = src.replace(old, new)
print('2. prompt() wrapping added')

# 3. Fix image preprocessing in the step loop
old = '    img_pil = Image.fromarray(img_uint8)\n\n    # \xe2\x94\x80\xe2\x94\x80 Clean decode (BEFORE any perturbation decision) \xe2\x94\x80\xe2\x94\x80\n    inputs = processor(text=instruction, images=img_pil, return_tensors=\'pt\')\n    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}'

new = '''    # ── Clean decode (BEFORE any perturbation decision) ──
    # V4-aligned preprocessing
    v4_image = prepare_openvla_image(img_uint8, libero_official_preprocess=False, center_crop=False)
    inputs = processor(text=instruction, images=v4_image, return_tensors='pt')
    inputs.pop("attention_mask", None)
    in_ids = inputs.get("input_ids")
    if in_ids is not None and not torch.all(in_ids[:, -1] == 29871):
        inputs["input_ids"] = torch.cat((in_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(in_ids.device)), dim=1)
    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}'''

if old in src:
    src = src.replace(old, new)
    print('3. V4 image preprocessing applied')
else:
    # Try alternative pattern
    if 'Image.fromarray(img_uint8)' in src:
        idx = src.find('Image.fromarray(img_uint8)')
        end = src.find('for k,v in inputs.items()}', idx) + len('for k,v in inputs.items()}')
        src = src[:idx] + new.split('\n')[0] + '\n' + '\n'.join(new.split('\n')[1:]) + src[end:]
        print('3. V4 preprocessing applied (alt pattern)')
    else:
        print('3. WARNING: image preprocessing pattern not found')

# Remove unused PIL import
src = src.replace('from PIL import Image\n', '')

with open(path, 'w') as f:
    f.write(src)
print('V6 runner fixed')
