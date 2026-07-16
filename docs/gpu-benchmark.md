# GPU Benchmark Results — RTX 5070 Laptop GPU (Blackwell sm_120)

> **Date**: 2026-06-20  
> **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU  
> **VRAM**: 8151 MiB total  
> **CUDA Runtime**: 12.8 | **PyTorch**: 2.11.0+cu128 | **cuDNN**: 91900  
> **Driver**: 592.27 | **Mode**: WDDM (not TCC — ~5-15% overhead)

## PyTorch Matmul Benchmark (FP16)

| Matrix Size | Time (ms) | TFLOPS |
|------------:|----------:|-------:|
| 256×256 | 0.015 | 2.29 |
| 512×512 | 0.013 | 20.02 |
| 1024×1024 | 0.062 | 34.61 |
| 2048×2048 | 0.521 | 32.95 |
| 3072×3072 | 1.878 | 30.87 |
| 4096×4096 | 6.728 | 20.43 |

**Peak TFLOPS**: 34.61 TFLOPS at 1024×1024 (diminishing at larger sizes due to memory bandwidth saturation)

## YOLO26 Inference Throughput + VRAM

Benchmark: 100 warmup + 100 timed iterations at 640×640 input resolution.

| Variant | Batch | Peak VRAM (MiB) | Fits <=7500? | Latency (ms) | Throughput (img/s) |
|---------|------:|----------------:|:-----------:|-------------:|-------------------:|
| yolo26n | 16 | 697 | ✅ | 49.9 | 320.5 |
| yolo26n | 32 | 1,219 | ✅ | 96.5 | 331.7 |
| yolo26n | 64 | 2,269 | ✅ | 215.4 | 297.2 |
| yolo26s | 16 | 958 | ✅ | 90.2 | 177.5 |
| yolo26s | 32 | 1,706 | ✅ | 189.8 | 168.6 |
| yolo26m | 8 | 841 | ✅ | 96.7 | 82.8 |
| yolo26m | 16 | 1,517 | ✅ | 205.8 | 77.7 |
| yolo26l | 4 | 496 | ✅ | 56.3 | 71.1 |
| yolo26l | 8 | 884 | ✅ | 121.5 | 65.8 |
| yolo26x | 2 | 707 | ✅ | 52.1 | 38.4 |
| yolo26x | 4 | 986 | ✅ | 107.2 | 37.3 |

## Recommended Batch Sizes

Largest batch size that fits within the 7500 MiB target (leaving ~650 MiB headroom for OS + WDDM overhead).

| Variant | Optimal Batch | Peak VRAM (MiB) | Throughput (img/s) |
|---------|--------------:|----------------:|-------------------:|
| yolo26n | 64 | 2,269 | 297.2 |
| yolo26s | 32 | 1,706 | 168.6 |
| yolo26m | 16 | 1,517 | 77.7 |
| yolo26l | 8 | 884 | 65.8 |
| yolo26x | 4 | 986 | 37.3 |

## Training Implications

- **All variants fit comfortably** in 8GB VRAM at recommended batch sizes — no gradient accumulation needed
- **Training VRAM will be higher** than inference due to activations + optimizer states (expect ~2-3× the inference VRAM)
- **Conservative training batch sizes** (reducing by half from optimal inference batch):
  - yolo26n: 32 (from optimal 64)
  - yolo26s: 16 (from optimal 32)
  - yolo26m: 8 (from optimal 16)
  - yolo26l: 4 (from optimal 8)
  - yolo26x: 2 (from optimal 4)

## Notes

- WDDM mode adds ~5-15% GPU kernel launch overhead vs Linux TCC mode
- VRAM readings include WDDM memory manager overhead (~200-500 MiB)
- Actual training throughput will be lower than inference due to backward pass
- Gradient accumulation can be used to simulate larger effective batch sizes if needed
