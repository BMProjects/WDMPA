#!/usr/bin/env python3
"""Jetson Nano benchmark script (Python 3.6 compatible).

Usage:
    python benchmark.py --model wdmpa_fold0.onnx
    python benchmark.py --model wdmpa_fold0_fp16.engine --format tensorrt
"""

from __future__ import print_function
import argparse
import time
import os
import sys

import numpy as np


def get_onnx_session(model_path):
    """Create ONNX Runtime session."""
    import onnxruntime as ort
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def get_tensorrt_context(engine_path):
    """Create TensorRT execution context."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    return engine, context


def benchmark_onnx(session, input_shape, warmup=50, iterations=500):
    """Benchmark ONNX model."""
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    print("Warmup ({} iterations)...".format(warmup))
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    print("Benchmarking ({} iterations)...".format(iterations))
    latencies = []
    for _ in range(iterations):
        start = time.time()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.time() - start) * 1000)  # ms
    
    return latencies


def benchmark_tensorrt(engine, context, input_shape, warmup=50, iterations=500):
    """Benchmark TensorRT engine."""
    import pycuda.driver as cuda
    
    # Allocate buffers
    batch_size = input_shape[0]
    input_size = np.prod(input_shape) * 4  # float32
    output_size = batch_size * 2 * 4  # 2 outputs
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    cuda.memcpy_htod(d_input, dummy_input)
    
    stream = cuda.Stream()
    
    # Warmup
    print("Warmup ({} iterations)...".format(warmup))
    for _ in range(warmup):
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()
    
    # Benchmark
    print("Benchmarking ({} iterations)...".format(iterations))
    latencies = []
    for _ in range(iterations):
        start = time.time()
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()
        latencies.append((time.time() - start) * 1000)
    
    return latencies


def get_memory_usage():
    """Get GPU memory usage on Jetson."""
    try:
        import subprocess
        result = subprocess.check_output(['cat', '/sys/kernel/debug/nvmap/iovmm/total'])
        return int(result.strip()) / (1024 * 1024)  # MB
    except Exception:
        return -1


def print_stats(latencies, model_name):
    """Print benchmark statistics."""
    latencies = np.array(latencies)
    
    print("\n" + "=" * 50)
    print("Results: {}".format(model_name))
    print("=" * 50)
    print("  Mean:   {:.2f} ms".format(np.mean(latencies)))
    print("  Std:    {:.2f} ms".format(np.std(latencies)))
    print("  Min:    {:.2f} ms".format(np.min(latencies)))
    print("  Max:    {:.2f} ms".format(np.max(latencies)))
    print("  P95:    {:.2f} ms".format(np.percentile(latencies, 95)))
    print("  FPS:    {:.1f}".format(1000 / np.mean(latencies)))
    
    mem = get_memory_usage()
    if mem > 0:
        print("  Memory: {:.0f} MB".format(mem))
    
    return {
        'model': model_name,
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'fps': 1000 / np.mean(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description='Jetson Nano Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tensorrt'])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()
    
    input_shape = (args.batch_size, 3, 224, 224)
    model_name = os.path.basename(args.model)
    
    print("Model: {}".format(args.model))
    print("Format: {}".format(args.format))
    print("Input shape: {}".format(input_shape))
    
    if args.format == 'onnx':
        session = get_onnx_session(args.model)
        latencies = benchmark_onnx(session, input_shape, args.warmup, args.iterations)
    else:
        engine, context = get_tensorrt_context(args.model)
        latencies = benchmark_tensorrt(engine, context, input_shape, args.warmup, args.iterations)
    
    stats = print_stats(latencies, model_name)
    
    if args.output:
        import csv
        with open(args.output, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(stats)
        print("\nResults appended to: {}".format(args.output))


if __name__ == '__main__':
    main()
