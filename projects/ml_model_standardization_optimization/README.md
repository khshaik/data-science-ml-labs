# Model Standardization and Optimization

---

## Overview

Three hands-on labs that walk the full production pipeline:

```
Train → Export to ONNX → Compress → Optimized Runtime → Production
```

| Lab | Topic | Key tools |
|-----|-------|-----------|
| Lab 1 | Keras → ONNX Conversion | `tf2onnx`, `onnx.checker`, ONNX Runtime |
| Lab 2 | Quantization & Pruning | `quantize_dynamic`, TF Lite INT8, filter pruning |
| Lab 3 | Benchmarking Dashboard | p50/p95/p99 latency, throughput sweep, `tracemalloc`, matplotlib |

All labs use **MobileNetV2** (pretrained on ImageNet via `tf.keras.applications`).

---

## Environment setup

### Requirements
- Python 3.9 – 3.11
- ~3 GB disk space (TensorFlow + model downloads)
- CPU-only is fine; no GPU required

### Install

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Optional: `onnxoptimizer` (Lab 1, Step 5)

```bash
pip install onnxoptimizer
```

This step is guarded — the lab runs fine without it and prints a clear
message if the package is not found.

---

## Running the labs

### As Python scripts (`.py`)

Each lab can be run directly from the command line:

```bash
# Lab 1
cd lab1_onnx_conversion
python lab1_onnx_conversion.py

# Lab 2 (run Lab 1 first, or Lab 2 will auto-regenerate the baseline)
cd ../lab2_compression
python lab2_compression.py

# Lab 3 (run Labs 1 & 2 first, or Lab 3 will auto-regenerate all models)
cd ../lab3_benchmarking
python lab3_benchmarking.py
```

### As Jupyter notebooks (`.ipynb`)

```bash
# From the w7_lab root folder
jupyter notebook
```

Open the `.ipynb` file inside each lab folder and run cells top-to-bottom.
Both the `.py` and `.ipynb` files are kept in sync — they are generated
from the same source.

---

## Expected outputs

### Lab 1
```
Baseline Keras latency: ~150 ms
ONNX Runtime latency:   ~6–10 ms
Speedup:                ~15–25x
Output validation:      PASSED (max diff < 1e-4)

Artifacts: lab1_onnx_conversion/artifacts/
  mobilenetv2.onnx           (~13 MB)
  mobilenetv2_optimized.onnx (~13 MB, if onnxoptimizer installed)
```

### Lab 2
```
Variant            Size (MB)  Latency (ms)
FP32 Baseline        ~13.3       ~10
INT8 PTQ (ONNX)       ~3.5       ~30
INT8 TFLite           ~3.8        ~6
Pruned ONNX          ~13.1       ~10

Artifacts: lab2_compression/artifacts/
  mobilenetv2_int8.onnx
  mobilenetv2_int8.tflite
  mobilenetv2_pruned.keras
  mobilenetv2_pruned.onnx
```

### Lab 3
```
Latency: p50 / p95 / p99 per model (200 runs each)
Throughput: images/sec at batch sizes 1, 4, 8, 16
Memory: peak tracemalloc usage per model

Artifacts: lab3_benchmarking/artifacts/
  benchmark_dashboard.png    (4-panel matplotlib figure)
```

> **Note on numbers**: exact latency figures vary by CPU. The relative
> ordering (TFLite fastest, INT8 ONNX smallest, pruned ONNX ~= baseline)
> is consistent across machines.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'tf2onnx'` | `pip install tf2onnx` |
| `ModuleNotFoundError: No module named 'onnxruntime'` | `pip install onnxruntime` |
| `ModuleNotFoundError: No module named 'onnxruntime.quantization'` | Upgrade: `pip install --upgrade onnxruntime` |
| `ModuleNotFoundError: No module named 'onnxoptimizer'` | Optional — skip or `pip install onnxoptimizer` |
| `urllib.error` when loading `weights="imagenet"` | You need internet access for the first run to download ImageNet weights (~14 MB). After the first run they are cached locally. |
| Lab 2 / Lab 3 can't find Lab 1 artifact | Run Lab 1 first, or the lab will auto-regenerate the baseline ONNX. |
| Killed / out of memory | Use `tensorflow-cpu` (already in requirements). Close other applications and retry. |

---

## Learning objectives

| Learning objective | Covered in |
|--------------------|-----------|
| Explain why research-trained models fail in production | Lab 1 intro markdown |
| Identify portability, latency, throughput, and footprint as key production goals | Lab 1 summary + Lab 3 SLO block |
| Export a Keras model to a standard format (ONNX) | Lab 1, Steps 3–4 |
| Validate and optionally optimize an ONNX graph | Lab 1, Steps 4–5 |
| Run a model with an optimized runtime (ONNX Runtime) | Lab 1, Step 6 |
| Apply INT8 dynamic quantization | Lab 2, Step 2 |
| Apply TFLite INT8 quantization with calibration | Lab 2, Step 3 |
| Apply structured filter pruning | Lab 2, Step 4 |
| Compare compression variants on size, latency, and accuracy | Lab 2, Step 7 |
| Benchmark with p50/p95/p99 percentiles, not just mean | Lab 3, Step 2 |
| Profile inference memory usage | Lab 3, Step 4 |
| Make SLO-driven deployment decisions | Lab 3, Step 6 |
