#!/usr/bin/env python3
"""
GPU Benchmark + VRAM Memory Profiling for YOLO26 model variants.
===============================================================
Measures inference throughput (images/sec) and peak VRAM usage for all 5
YOLO26 variants (n/s/m/l/x) on the RTX 5070 Laptop GPU.

Usage:
    python scripts/gpu_benchmark.py --quick       # quick smoke test
    python scripts/gpu_benchmark.py --vram-only    # VRAM profiling only
    python scripts/gpu_benchmark.py --full         # full benchmark
    python scripts/gpu_benchmark.py --model yolo26m --batch-sizes 8,16

Output: Markdown-friendly tables to stdout and evidence files.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import TypedDict

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VRAM_TOTAL_MIB = 8151  # RTX 5070 Laptop GPU total
VRAM_TARGET_MIB = 7500  # leave ~650 MiB headroom
IMAGE_SIZE = 640
WARMUP_ITERS = 100
TIMED_ITERS = 100


class VariantInfo(TypedDict):
    batch_sizes: list[int]
    weight: str


VARIANT_INFO: dict[str, VariantInfo] = {
    "yolo26n": {"batch_sizes": [16, 32, 64], "weight": "yolo26n.pt"},
    "yolo26s": {"batch_sizes": [16, 32], "weight": "yolo26s.pt"},
    "yolo26m": {"batch_sizes": [8, 16], "weight": "yolo26m.pt"},
    "yolo26l": {"batch_sizes": [4, 8], "weight": "yolo26l.pt"},
    "yolo26x": {"batch_sizes": [2, 4], "weight": "yolo26x.pt"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU Benchmark + VRAM Profiling for YOLO26")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true", help="Reduced benchmark (2 variants, 1 batch size each)"
    )
    mode.add_argument(
        "--vram-only", action="store_true", help="Only profile VRAM usage (skip throughput)"
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Complete benchmark (all variants, multiple batch sizes)",
    )

    parser.add_argument(
        "--model", type=str, default=None, help="Specific model to benchmark (default: all)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Custom batch sizes, comma-separated (overrides defaults)",
    )
    return parser.parse_args()


def resolve_variants(args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    """Return list of (variant_name, [batch_sizes]) to benchmark."""
    if args.model:
        if args.model not in VARIANT_INFO:
            print(f"ERROR: Unknown model '{args.model}'. Choose from: {list(VARIANT_INFO.keys())}")
            sys.exit(1)
        variants = [args.model]
    elif args.quick:
        variants = ["yolo26n", "yolo26m"]
    else:
        variants = sorted(VARIANT_INFO.keys())

    if args.batch_sizes:
        custom = [int(b) for b in args.batch_sizes.split(",")]
        return [(v, custom) for v in variants]
    elif args.quick:
        # 1 batch size each for quick mode
        return [(v, [VARIANT_INFO[v]["batch_sizes"][0]]) for v in variants]
    else:
        return [(v, VARIANT_INFO[v]["batch_sizes"]) for v in variants]


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {props.total_memory / 1024**2:.0f} MiB")
    print(f"CUDA Runtime: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"PyTorch: {torch.__version__}")
    return device


# ---------------------------------------------------------------------------
# Benchmark 1: PyTorch Matmul (matrix size sweep)
# ---------------------------------------------------------------------------


def benchmark_matmul(device: torch.device) -> list[dict]:
    """Sweep matrix sizes 256→4096, measure TFLOPS."""
    print("\n" + "=" * 60)
    print("Benchmark 1: PyTorch Matmul (FP16) — Matrix Size Sweep")
    print("=" * 60)

    results = []
    sizes = [256, 512, 1024, 2048, 3072, 4096]

    for sz in sizes:
        a = torch.randn(sz, sz, dtype=torch.float16, device=device)
        b = torch.randn(sz, sz, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(30):
            _ = a @ b
        torch.cuda.synchronize()

        # Timed
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(100):
            _ = a @ b
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / 100.0  # per iteration

        # FLOPs for square matmul: 2 * N^3
        flops = 2 * sz**3
        tflops = flops / (elapsed_ms / 1000.0) / 1e12

        results.append(
            {
                "size": sz,
                "elapsed_ms": elapsed_ms,
                "tflops": tflops,
            }
        )
        print(f"  {sz:5d}×{sz:<5d}  {elapsed_ms:.3f} ms  {tflops:.2f} TFLOPS")

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: YOLO26 inference throughput + VRAM
# ---------------------------------------------------------------------------


def benchmark_yolo_variant(
    variant: str,
    batch_sizes: list[int],
    device: torch.device,
    vram_only: bool,
) -> list[dict]:
    """Benchmark a single YOLO variant at given batch sizes."""
    weight_path = MODELS_DIR / VARIANT_INFO[variant]["weight"]
    if not weight_path.exists():
        print(f"  WARNING: {weight_path} not found — skipping")
        return []

    print(f"\n{'=' * 60}")
    print(f"Benchmark 2: {variant} — {weight_path.name}")
    print(f"{'=' * 60}")

    # Load model
    model = YOLO(str(weight_path))
    model.to(device)
    model.eval()

    # Warmup model once
    warmup_tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device) / 255.0
    _ = model(warmup_tensor, verbose=False)
    torch.cuda.synchronize()
    print("  Model loaded and warmed up.")

    results = []
    for batch_size in batch_sizes:
        print(f"\n  --- batch_size={batch_size} ---")

        if batch_size > 128:
            print(f"  SKIP: batch_size={batch_size} too large for single forward")
            continue

        dummy = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        dummy_norm = dummy / 255.0  # normalize to [0,1] for YOLO

        # ------------------------------------------------------------------
        # VRAM profiling
        # ------------------------------------------------------------------
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Run a few inference steps to measure peak VRAM
        for _ in range(5):
            _ = model(dummy_norm, verbose=False)
        torch.cuda.synchronize()

        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2
        current_vram = torch.cuda.memory_allocated(device) / 1024**2

        print(f"    Peak VRAM:   {peak_vram:.0f} MiB")
        print(f"    Current VRAM:{current_vram:.0f} MiB")
        fits = peak_vram <= VRAM_TARGET_MIB

        if vram_only:
            results.append(
                {
                    "variant": variant,
                    "batch_size": batch_size,
                    "peak_vram_mib": peak_vram,
                    "current_vram_mib": current_vram,
                    "fits_in_target": fits,
                    "throughput_img_per_sec": None,
                    "latency_ms": None,
                }
            )
            # Cleanup before next batch
            del dummy, dummy_norm
            torch.cuda.empty_cache()
            continue

        # ------------------------------------------------------------------
        # Throughput profiling (skip if VRAM already too high)
        # ------------------------------------------------------------------
        if not fits:
            print(
                f"    SKIP throughput: peak VRAM {peak_vram:.0f} MiB > {VRAM_TARGET_MIB} target (max safe)"
            )
            results.append(
                {
                    "variant": variant,
                    "batch_size": batch_size,
                    "peak_vram_mib": peak_vram,
                    "current_vram_mib": current_vram,
                    "fits_in_target": fits,
                    "throughput_img_per_sec": None,
                    "latency_ms": None,
                }
            )
            # Cleanup before next batch
            del dummy, dummy_norm
            torch.cuda.empty_cache()
            continue

        # Warmup iterations
        for _ in range(WARMUP_ITERS):
            _ = model(dummy_norm, verbose=False)
        torch.cuda.synchronize()

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(TIMED_ITERS):
            _ = model(dummy_norm, verbose=False)
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / TIMED_ITERS
        throughput = (batch_size * 1000.0) / avg_ms  # images/sec

        # Re-check peak after throughput runs
        peak_vram_after = torch.cuda.max_memory_allocated(device) / 1024**2

        print(f"    Avg latency: {avg_ms:.1f} ms / batch")
        print(f"    Throughput:  {throughput:.1f} img/s")
        print(f"    Peak VRAM (post-throughput): {peak_vram_after:.0f} MiB")

        results.append(
            {
                "variant": variant,
                "batch_size": batch_size,
                "peak_vram_mib": peak_vram_after,
                "current_vram_mib": torch.cuda.memory_allocated(device) / 1024**2,
                "fits_in_target": peak_vram_after <= VRAM_TARGET_MIB,
                "throughput_img_per_sec": throughput,
                "latency_ms": avg_ms,
            }
        )

        # Cleanup before next batch
        del dummy, dummy_norm
        torch.cuda.empty_cache()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def print_matmul_table(results: list[dict]) -> str:
    lines = [
        "### PyTorch Matmul Benchmark (FP16)",
        "",
        "| Matrix Size | Time (ms) | TFLOPS |",
        "|------------:|----------:|-------:|",
    ]
    for r in results:
        lines.append(f"| {r['size']}×{r['size']} | {r['elapsed_ms']:.3f} | {r['tflops']:.2f} |")
    return "\n".join(lines) + "\n"


def print_yolo_table(results: list[dict]) -> str:
    lines = [
        "### YOLO26 Inference Throughput + VRAM",
        "",
        "| Variant | Batch | VRAM (MiB) | Fits <=7500? | Latency (ms) | Throughput (img/s) |",
        "|---------|------:|-----------:|:-----------:|-------------:|-------------------:|",
    ]
    for r in results:
        fits_str = "✅" if r["fits_in_target"] else "❌"
        tput = f"{r['throughput_img_per_sec']:.1f}" if r["throughput_img_per_sec"] else "—"
        lat = f"{r['latency_ms']:.1f}" if r["latency_ms"] else "—"
        lines.append(
            f"| {r['variant']} | {r['batch_size']} | {r['peak_vram_mib']:.0f} | {fits_str} | {lat} | {tput} |"
        )
    return "\n".join(lines) + "\n"


def print_optimal_batch_table(results: list[dict]) -> str:
    """Compute optimal batch per variant (largest batch that fits in target)."""
    by_variant: dict[str, list[dict]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)

    lines = [
        "### Recommended Batch Sizes (<=7500 MiB VRAM)",
        "",
        "| Variant | Optimal Batch | VRAM (MiB) | Throughput (img/s) |",
        "|---------|--------------:|-----------:|-------------------:|",
    ]
    for var in sorted(by_variant.keys()):
        candidates = [r for r in by_variant[var] if r["fits_in_target"]]
        if candidates:
            best = candidates[-1]  # largest batch that fits
            tput = (
                f"{best['throughput_img_per_sec']:.1f}" if best["throughput_img_per_sec"] else "—"
            )
            lines.append(f"| {var} | {best['batch_size']} | {best['peak_vram_mib']:.0f} | {tput} |")
        else:
            # Report smallest batch even if it doesn't fit
            smallest = by_variant[var][0]
            tput = (
                f"{smallest['throughput_img_per_sec']:.1f}"
                if smallest["throughput_img_per_sec"]
                else "—"
            )
            lines.append(
                f"| {var} | {smallest['batch_size']} (⚠ exceeds target) | {smallest['peak_vram_mib']:.0f} | {tput} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = get_device()

    # Check models directory
    if not MODELS_DIR.exists():
        print(f"FATAL: Models directory not found: {MODELS_DIR}")
        sys.exit(1)

    variants = resolve_variants(args)
    print(f"\nVariants to benchmark: {[v[0] for v in variants]}")
    print(
        f"VRAM target: {VRAM_TARGET_MIB} MiB (leaving {VRAM_TOTAL_MIB - VRAM_TARGET_MIB} MiB headroom)"
    )

    all_yolo_results: list[dict] = []

    # Benchmark 1: Matmul
    matmul_results = benchmark_matmul(device) if not args.vram_only else []

    # Benchmark 2: YOLO variants
    for variant, batch_sizes in variants:
        variant_results = benchmark_yolo_variant(
            variant, batch_sizes, device, vram_only=args.vram_only
        )
        all_yolo_results.extend(variant_results)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if matmul_results:
        matmul_table = print_matmul_table(matmul_results)
        print("\n" + matmul_table)

    yolo_table = print_yolo_table(all_yolo_results)
    print("\n" + yolo_table)

    optimal_table = print_optimal_batch_table(all_yolo_results)
    print("\n" + optimal_table)

    # ------------------------------------------------------------------
    # Save evidence files
    # ------------------------------------------------------------------
    evidence_dir = Path(__file__).resolve().parent.parent / ".omo" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Benchmark throughput evidence
    bench_lines = [
        f"GPU Benchmark Results — {timestamp}",
        f"Device: {torch.cuda.get_device_name(device)}",
        f"VRAM Total: {VRAM_TOTAL_MIB} MiB",
        f"VRAM Target: {VRAM_TARGET_MIB} MiB",
        f"Input: {IMAGE_SIZE}×{IMAGE_SIZE}",
        f"Warmup: {WARMUP_ITERS} iters | Timed: {TIMED_ITERS} iters",
        "",
    ]
    if matmul_results:
        bench_lines.append(matmul_table)
    bench_lines.append(yolo_table)
    bench_lines.append(optimal_table)

    bench_path = evidence_dir / "task-7-benchmark.txt"
    bench_path.write_text("\n".join(bench_lines), encoding="utf-8")
    print(f"\nEvidence saved: {bench_path}")

    # VRAM-only evidence
    vram_results = [r for r in all_yolo_results]
    vram_lines = [
        f"VRAM Profiling Results — {timestamp}",
        f"Device: {torch.cuda.get_device_name(device)}",
        f"VRAM Total: {VRAM_TOTAL_MIB} MiB",
        f"VRAM Target: {VRAM_TARGET_MIB} MiB",
        "",
        "### VRAM Usage by Variant",
        "",
        "| Variant | Batch | Peak VRAM (MiB) | Current VRAM (MiB) | Fits <=7500? |",
        "|---------|------:|----------------:|-------------------:|:-----------:|",
    ]
    for r in vram_results:
        fits_str = "✅" if r["fits_in_target"] else "❌"
        vram_lines.append(
            f"| {r['variant']} | {r['batch_size']} | {r['peak_vram_mib']:.0f} | {r['current_vram_mib']:.0f} | {fits_str} |"
        )

    torch.cuda.reset_peak_memory_stats()
    vram_lines.extend(
        [
            "",
            "### WDDM Mode Notes",
            "",
            "- This GPU runs in WDDM mode (not TCC).",
            "- WDDM adds 5–15% GPU kernel launch overhead vs Linux TCC.",
            "- VRAM readings may fluctuate due to WDDM memory manager.",
            "- Windows GD paging can cause VRAM values to appear ~200-500 MiB higher.",
        ]
    )

    vram_path = evidence_dir / "task-7-vram.txt"
    vram_path.write_text("\n".join(vram_lines), encoding="utf-8")
    print(f"Evidence saved: {vram_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
