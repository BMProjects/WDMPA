"""Jetson Nano Benchmark Script - Four-Dimensional Performance Measurement.

Measures:
1. Inference Latency (ms) - Batch=1, warmup 100, measure 1000
2. Throughput (FPS) - Frames per second
3. Memory Footprint (MB) - Model size + runtime GPU memory
4. Energy Efficiency (FPS/W) - Using tegrastats

Usage:
    python deploy/jetson_benchmark.py --model wdmpa_fold0.onnx --format onnx
    python deploy/jetson_benchmark.py --model wdmpa_fold0.engine --format tensorrt
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Check if running on Jetson
IS_JETSON = os.path.exists("/etc/nv_tegra_release")


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single model."""

    model_name: str
    format: str

    # Latency (ms)
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float

    # Throughput
    fps: float

    # Memory
    model_size_mb: float
    gpu_memory_mb: float

    # Energy (Jetson only)
    power_watts: float | None = None
    efficiency_fps_per_watt: float | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "format": self.format,
            "latency_ms": f"{self.latency_mean:.2f}±{self.latency_std:.2f}",
            "fps": f"{self.fps:.1f}",
            "model_mb": f"{self.model_size_mb:.1f}",
            "gpu_mb": f"{self.gpu_memory_mb:.1f}",
            "power_w": f"{self.power_watts:.1f}" if self.power_watts else "N/A",
            "eff_fps/w": f"{self.efficiency_fps_per_watt:.2f}" if self.efficiency_fps_per_watt else "N/A",
        }

    def __str__(self) -> str:
        lines = [
            f"\n{'=' * 50}",
            f"Benchmark Results: {self.model_name} ({self.format})",
            f"{'=' * 50}",
            f"Latency:     {self.latency_mean:.2f} ± {self.latency_std:.2f} ms",
            f"             (min: {self.latency_min:.2f}, max: {self.latency_max:.2f})",
            f"Throughput:  {self.fps:.1f} FPS",
            f"Model Size:  {self.model_size_mb:.1f} MB",
            f"GPU Memory:  {self.gpu_memory_mb:.1f} MB",
        ]
        if self.power_watts:
            lines.append(f"Power:       {self.power_watts:.1f} W")
            lines.append(f"Efficiency:  {self.efficiency_fps_per_watt:.2f} FPS/W")
        lines.append("=" * 50)
        return "\n".join(lines)


class TegrastatsMonitor:
    """Monitor GPU power consumption using tegrastats on Jetson."""

    def __init__(self):
        self.process = None
        self.power_readings = []

    def start(self):
        if not IS_JETSON:
            return
        # Start tegrastats in background
        self.power_readings = []
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> float | None:
        if not IS_JETSON or self.process is None:
            return None

        self.process.terminate()
        try:
            stdout, _ = self.process.communicate(timeout=2)
            # Parse power readings from tegrastats output
            for line in stdout.strip().split("\n"):
                # Example: "VDD_IN 3000/3000" or "POM_5V_IN 3000/3000"
                if "VDD_IN" in line or "POM_5V_IN" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "VDD_IN" in p or "POM_5V_IN" in p:
                            try:
                                power_str = parts[i + 1].split("/")[0]
                                self.power_readings.append(int(power_str))
                            except (IndexError, ValueError):
                                pass
        except subprocess.TimeoutExpired:
            self.process.kill()

        if self.power_readings:
            return np.mean(self.power_readings) / 1000.0  # mW to W
        return None


def benchmark_pytorch(
    weights_path: str,
    input_size: tuple = (224, 224),
    warmup: int = 100,
    iterations: int = 1000,
    device: str = "cuda",
) -> BenchmarkResult:
    """Benchmark PyTorch model."""
    import torch

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from Emodels.Emodel_AStarNet_WFD_SESA import Emodel_StarNet

    # Load model
    model = Emodel_StarNet()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Model size
    model_size = Path(weights_path).stat().st_size / (1024 * 1024)

    # Dummy input
    dummy_input = torch.randn(1, 3, *input_size, device=device)

    # GPU memory before
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    # Start power monitoring
    power_monitor = TegrastatsMonitor()
    power_monitor.start()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    # Stop power monitoring
    avg_power = power_monitor.stop()

    # GPU memory
    gpu_memory = 0
    if device == "cuda":
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    latencies = np.array(latencies)
    fps = 1000.0 / latencies.mean()

    return BenchmarkResult(
        model_name=Path(weights_path).stem,
        format="pytorch",
        latency_mean=latencies.mean(),
        latency_std=latencies.std(),
        latency_min=latencies.min(),
        latency_max=latencies.max(),
        fps=fps,
        model_size_mb=model_size,
        gpu_memory_mb=gpu_memory,
        power_watts=avg_power,
        efficiency_fps_per_watt=fps / avg_power if avg_power else None,
    )


def benchmark_onnx(
    onnx_path: str,
    input_size: tuple = (224, 224),
    warmup: int = 100,
    iterations: int = 1000,
) -> BenchmarkResult:
    """Benchmark ONNX model using ONNX Runtime."""
    import onnxruntime as ort

    # Check available providers
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        provider = ["CUDAExecutionProvider"]
    elif "TensorrtExecutionProvider" in providers:
        provider = ["TensorrtExecutionProvider"]
    else:
        provider = ["CPUExecutionProvider"]

    print(f"Using ONNX Runtime provider: {provider}")

    session = ort.InferenceSession(onnx_path, providers=provider)

    # Model size
    model_size = Path(onnx_path).stat().st_size / (1024 * 1024)

    # Dummy input
    dummy_input = np.random.randn(1, 3, *input_size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {"input": dummy_input})

    # Start power monitoring
    power_monitor = TegrastatsMonitor()
    power_monitor.start()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {"input": dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    # Stop power monitoring
    avg_power = power_monitor.stop()

    latencies = np.array(latencies)
    fps = 1000.0 / latencies.mean()

    # Estimate GPU memory (not precise for ONNX Runtime)
    gpu_memory = 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.used / (1024 * 1024)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return BenchmarkResult(
        model_name=Path(onnx_path).stem,
        format="onnx",
        latency_mean=latencies.mean(),
        latency_std=latencies.std(),
        latency_min=latencies.min(),
        latency_max=latencies.max(),
        fps=fps,
        model_size_mb=model_size,
        gpu_memory_mb=gpu_memory,
        power_watts=avg_power,
        efficiency_fps_per_watt=fps / avg_power if avg_power else None,
    )


def benchmark_tensorrt(
    engine_path: str,
    input_size: tuple = (224, 224),
    warmup: int = 100,
    iterations: int = 1000,
) -> BenchmarkResult:
    """Benchmark TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        raise ImportError("TensorRT and PyCUDA are required for TensorRT benchmarking")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, *input_size)
    output_shape = (1, 1, 2, 1, 1)

    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Model size
    model_size = Path(engine_path).stat().st_size / (1024 * 1024)

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Start power monitoring
    power_monitor = TegrastatsMonitor()
    power_monitor.start()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        start = time.perf_counter()
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Stop power monitoring
    avg_power = power_monitor.stop()

    latencies = np.array(latencies)
    fps = 1000.0 / latencies.mean()

    # GPU memory
    gpu_memory = 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.used / (1024 * 1024)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return BenchmarkResult(
        model_name=Path(engine_path).stem,
        format="tensorrt",
        latency_mean=latencies.mean(),
        latency_std=latencies.std(),
        latency_min=latencies.min(),
        latency_max=latencies.max(),
        fps=fps,
        model_size_mb=model_size,
        gpu_memory_mb=gpu_memory,
        power_watts=avg_power,
        efficiency_fps_per_watt=fps / avg_power if avg_power else None,
    )


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save benchmark results to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Benchmark - 4D Performance")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (.pkl, .onnx, or .engine)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "onnx", "tensorrt"],
        default=None,
        help="Model format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input size (H W)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Benchmark iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for PyTorch (cuda/cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect format
    model_path = Path(args.model)
    if args.format is None:
        if model_path.suffix == ".pkl":
            args.format = "pytorch"
        elif model_path.suffix == ".onnx":
            args.format = "onnx"
        elif model_path.suffix in [".engine", ".trt"]:
            args.format = "tensorrt"
        else:
            raise ValueError(f"Unknown model format: {model_path.suffix}")

    print(f"Benchmarking {args.model} ({args.format})")
    print(f"Input size: {args.input_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Running on Jetson: {IS_JETSON}")

    # Run benchmark
    if args.format == "pytorch":
        result = benchmark_pytorch(
            args.model,
            tuple(args.input_size),
            args.warmup,
            args.iterations,
            args.device,
        )
    elif args.format == "onnx":
        result = benchmark_onnx(
            args.model,
            tuple(args.input_size),
            args.warmup,
            args.iterations,
        )
    elif args.format == "tensorrt":
        result = benchmark_tensorrt(
            args.model,
            tuple(args.input_size),
            args.warmup,
            args.iterations,
        )

    print(result)

    # Save results
    if args.output:
        save_results([result], args.output)


if __name__ == "__main__":
    main()
