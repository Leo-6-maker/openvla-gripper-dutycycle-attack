# A800 Dependency Conflict Plan — 2026-06-20

**Environment:** `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
**pip check exit:** 1

---

## Conflict 1: opencv-python vs numpy

### Symptom
```
opencv-python 4.13.0.92 has requirement numpy>=2; python_version >= "3.9",
but you have numpy 1.26.4.
```

### Analysis
- `opencv-python 4.13.0.92` was installed as a dependency of LIBERO
- It requires `numpy>=2` for Python >= 3.9
- `numpy 1.26.4` was pulled in by OpenVLA's `requirements-min.txt` via tensorflow dependencies
- Numpy 2.x is blocked because torch 2.2.0 was compiled against NumPy 1.x

### Root Cause Chain
```
LIBERO → opencv-python>=4.13 → numpy>=2
OpenVLA → tensorflow 2.15 → numpy<2 (via indirect constraint)
PyTorch 2.2.0 → compiled against NumPy 1.x API
```

### Proposed Fix (DO NOT APPLY WITHOUT REVIEW)

**Option A: Downgrade opencv-python** (least risky)
```bash
pip install "opencv-python<4.12"  # versions <4.12 accept numpy 1.x
```
- Impact: LIBERO video/image may use older OpenCV features
- Risk: Low — OpenCV <4.12 still has full functionality

**Option B: Compatibility workaround** (keep both)
```bash
pip install "numpy>=1.26,<2"  # explicit pin
```
- Impact: opencv warning persists but functionality preserved
- Risk: Medium — silent failures possible

**Option C: Upgrade numpy to 2.x** (BLOCKED)
- Cannot upgrade numpy to ≥2.0 — torch 2.2.0 compiled against numpy 1.x API
- Per migration policy: "禁止升级numpy到2.x"

### Recommended: Option A
Keep numpy 1.26.4 stable and downgrade opencv-python.

---

## Conflict 2: protobuf vs tensorflow_metadata (prismatic full import)

### Symptom
```
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

### Analysis
- `protobuf 4.25.9` was installed by OpenVLA's tensorflow dependencies
- `tensorflow-metadata 1.21.0` requires a NEWER protobuf that exports `runtime_version`
- This bug was introduced in protobuf 5.x where `runtime_version` was added
- protobuf 4.25.9 doesn't have `runtime_version`

### Root Cause Chain
```
prismatic → dlimp → tensorflow_datasets → tensorflow_metadata 1.21.0 → needs protobuf >=5.x
OpenVLA setup.py → protobuf >=3.20.3,<6 → resolves to 4.25.9
```

### Impact
- **Inference path is NOT affected**: `AutoProcessor` + `AutoModelForVision2Seq` load and work correctly
- **Training/data loading path is affected**: `import prismatic` triggers TF→TFDS→TF-metadata→protobuf import chain
- For migration, we only need the inference path

### Proposed Fix (DO NOT APPLY WITHOUT REVIEW)

**Option A: Pin compatible protobuf version**
```bash
pip install "protobuf>=5.0,<6"
```
- Impact: May break tensorflow 2.15 which was built against older protobuf
- Risk: HIGH — could cascade through TF ecosystem

**Option B: Cap tensorflow-metadata**
```bash
pip install "tensorflow-metadata<1.15"
```
- Impact: May not be compatible with tensorflow_datasets version
- Risk: Medium

**Option C: Accept limitation** (RECOMMENDED for now)
- Inference path works. prismatic training code not needed for migration.
- Fix this only if parity requires loading datasets through prismatic pipeline.
- Risk: None for inference-only workflow.

---

## Current Status

| Conflict | Severity | Inference Impact | Recommended |
|---|---|---|---|
| opencv vs numpy | LOW | None | Option A (downgrade opencv) |
| protobuf vs TF-metadata | MEDIUM | None | Option C (accept limitation) |

**No changes applied yet. Awaiting user review.**
