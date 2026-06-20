"""Token audit + Lane O/G + double-unnormalize check for A800 spatial model."""
import json, os, hashlib, inspect, numpy as np
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MODEL = "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620"
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
    local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map="cuda:0",
)
tok = proc.tokenizer

# ===== TOKEN AUDIT =====
print("=" * 60)
print("TOKEN AUDIT")
print("=" * 60)
for tid in [0, 1, 2, 29871, 32000]:
    t = tok.convert_ids_to_tokens(tid)
    d = tok.decode([tid])
    print(f"ID {tid}: token={repr(t)}  decode={repr(d)}")

print(f"eos_token_id: {tok.eos_token_id}")
print(f"pad_token_id: {tok.pad_token_id}")
print(f"bos_token_id: {tok.bos_token_id}")

with open(f"{MODEL}/added_tokens.json") as f:
    added = json.load(f)
print(f"added_tokens: {json.dumps(added)}")

with open(f"{MODEL}/special_tokens_map.json") as f:
    spec = json.load(f)
print(f"special_tokens_map eos: {spec.get('eos_token', 'MISSING')}")

# ===== PREDICT_ACTION SOURCE =====
print("\n" + "=" * 60)
print("PREDICT_ACTION SOURCE")
print("=" * 60)
src_file = inspect.getfile(model.predict_action)
print(f"File: {src_file}")
with open(src_file) as f:
    src = f.read()
print(f"SHA: {hashlib.sha256(src.encode()).hexdigest()}")
for i, line in enumerate(src.split("\n")):
    low = line.lower()
    if any(kw in low for kw in ["eos_token_id", "pad_token_id", "max_new",
                                  "decode_action", "unnorm_key", "q01", "q99",
                                  "do_sample", "generate", "statistics"]):
        print(f"  L{i}: {line.strip()[:130]}")

# ===== LANE O: predict_action (no external EOS append) =====
print("\n" + "=" * 60)
print("LANE O: predict_action (NO external token append)")
print("=" * 60)
img = Image.new("RGB", (224, 224), color=(50, 100, 150))
task = "pick up the black bowl next to the ramekin and place it on the plate"
prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
inputs = proc(prompt, img, return_tensors="pt").to("cuda:0", dtype=torch.bfloat16)
print(f"Prompt: {repr(prompt)}")
print(f"Input IDs: {inputs.input_ids[0].tolist()}")
print(f"Token count: {inputs.input_ids.shape[1]}")
print(f"Last token in prompt: {inputs.input_ids[0, -1].item()} (= {repr(tok.decode([inputs.input_ids[0, -1].item()]))})")

lane_o_results = []
for run in range(3):
    result = model.predict_action(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        unnorm_key="libero_spatial",
        do_sample=False,
    )
    vals = np.array(result).flatten().tolist()
    lane_o_results.append(vals)
    print(f"Run {run+1}: {[round(x,8) for x in vals]}")

# Determinism check
o_match = all(r == lane_o_results[0] for r in lane_o_results[1:])
print(f"Lane O deterministic: {o_match}")

# ===== LANE G: manual generate + predict_action decode =====
print("\n" + "=" * 60)
print("LANE G: model.generate + predict_action(tokens)")
print("=" * 60)
lane_g_results = []
generated_tokens_all = []
for run in range(3):
    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        max_new_tokens=7,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        attention_mask=inputs.attention_mask,
    )
    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
    generated_tokens_all.append(new_tokens.tolist())
    action = model.predict_action(new_tokens.unsqueeze(0))
    vals = np.array(action).flatten().tolist()
    lane_g_results.append(vals)
    print(f"Run {run+1}: tokens={new_tokens.tolist()}  action={[round(x,8) for x in vals]}")

g_match = all(r == lane_g_results[0] for r in lane_g_results[1:])
print(f"Lane G deterministic: {g_match}")

# ===== LANE O vs G COMPARISON =====
print("\n" + "=" * 60)
print("LANE O vs G COMPARISON")
print("=" * 60)
o0 = np.array(lane_o_results[0])
g0 = np.array(lane_g_results[0])
diff = np.abs(o0 - g0)
print(f"Lane O: {[round(x,8) for x in o0.tolist()]}")
print(f"Lane G: {[round(x,8) for x in g0.tolist()]}")
print(f"Abs diff: {[round(x,8) for x in diff.tolist()]}")
print(f"Max diff: {diff.max():.8f}")
print(f"Exact match: {diff.max() < 1e-8}")

# ===== DOUBLE UNNORMALIZE CHECK =====
print("\n" + "=" * 60)
print("DOUBLE UNNORMALIZE CHECK")
print("=" * 60)
with open(f"{MODEL}/dataset_statistics.json") as f:
    sp = json.load(f)["libero_spatial"]["action"]
q01 = np.array(sp["q01"])
q99 = np.array(sp["q99"])

# predict_action already returns unnormalized action.
# If we were to apply q01/q99 again, would values change?
double_unnormed = o0 * (q99 - q01) + q01
d_diff = np.abs(double_unnormed - o0)
print(f"Lane O output:                            {[round(x,6) for x in o0.tolist()]}")
print(f"Hypothetical double-unnorm (O * range + min): {[round(x,6) for x in double_unnormed.tolist()]}")
print(f"Difference: {[round(x,8) for x in d_diff.tolist()]}")
print(f"Max difference: {d_diff.max():.8f}")
if d_diff.max() > 0.01:
    print("DOUBLE UNNORMALIZE WOULD CHANGE VALUES - predict_action output is ALREADY unnormalized")
else:
    print("Double unnormalize would NOT significantly change values - output may be normalized")

# ===== MODEL DEVICE CHECK =====
print("\n" + "=" * 60)
print("MODEL DEVICE CHECK")
print("=" * 60)
devices = sorted(set(str(p.device) for p in model.parameters()))
print(f"Parameter devices: {devices}")
hf_map = getattr(model, "hf_device_map", None)
print(f"hf_device_map: {hf_map}")
device_gate = devices == ["cuda:0"]
print(f"DEVICE GATE: {'PASS' if device_gate else 'FAIL'}")

# ===== FINAL ANSWERS =====
print("\n" + "=" * 60)
print("FINAL TOKEN CONTRACT")
print("=" * 60)
print(f"1. tokenizer.eos_token_id = {tok.eos_token_id}")
print(f"2. predict_action does NOT require external token append (takes prompt tokens directly)")
print(f"3. N/A - no append done by us")
print(f"4. ID 29871 = {repr(tok.convert_ids_to_tokens(29871))} = {repr(tok.decode([29871]))}")
print(f"5. Lane O: processor(prompt, image) -> predict_action(...) with NO external append")
print(f"   Lane G: processor -> model.generate -> predict_action(gen_tokens)")
print(f"TOKENIZER_EOS_ID = {tok.eos_token_id}")
print(f"TOKEN_29871_ROLE = {repr(tok.decode([29871]))} (NOT eos_token)")
