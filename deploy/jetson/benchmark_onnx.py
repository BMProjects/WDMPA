#!/usr/bin/env python3
"""
ONNX模型推理基准测试 (Jetson优化)
适用于资源受限环境，无需复杂依赖
"""

import time
import argparse
from pathlib import Path
import csv

def benchmark_model(model_path, warmup=50, iterations=500):
    """测试单个ONNX模型的推理延迟"""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip3 install --user numpy onnxruntime-gpu")
        return None
    
    print(f"\n测试模型: {model_path.name}")
    
    # 创建session (优先使用GPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    # 显示使用的provider
    print(f"  Provider: {session.get_providers()[0]}")
    
    # 准备输入 (假设输入shape为 [1, 3, 224, 224])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"  Input shape: {input_shape}")
    
    # 处理动态batch维度
    batch_size = 1 if input_shape[0] in ['batch', None, -1] else input_shape[0]
    actual_shape = [batch_size] + list(input_shape[1:])
    
    dummy_input = np.random.randn(*actual_shape).astype(np.float32)
    
    # Warmup
    print(f"  Warmup: {warmup} iterations...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
    
    # 基准测试
    print(f"  Benchmark: {iterations} iterations...")
    latencies = []
    
    for i in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{iterations} completed...")
    
    # 统计
    import numpy as np
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    fps = 1000 / mean_latency
    
    print(f"  ✅ Results:")
    print(f"    Mean: {mean_latency:.2f} ms ± {std_latency:.2f}")
    print(f"    Range: [{min_latency:.2f}, {max_latency:.2f}] ms")
    print(f"    FPS: {fps:.1f}")
    
    return {
        'model': model_path.stem,
        'mean_ms': f"{mean_latency:.2f}",
        'std_ms': f"{std_latency:.2f}",
        'min_ms': f"{min_latency:.2f}",
        'max_ms': f"{max_latency:.2f}",
        'fps': f"{fps:.1f}",
        'provider': session.get_providers()[0]
    }

def main():
    parser = argparse.ArgumentParser(description='ONNX模型基准测试')
    parser.add_argument('--models', required=True, help='ONNX模型目录或文件')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup迭代次数')
    parser.add_argument('--iterations', type=int, default=500, help='测试迭代次数')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONNX推理基准测试 (Jetson Nano)")
    print("=" * 60)
    
    # 获取模型列表
    models_path = Path(args.models)
    if models_path.is_dir():
        model_files = sorted(models_path.glob('*.onnx'))
    else:
        model_files = [models_path]
    
    if not model_files:
        print(f"❌ 未找到ONNX模型: {args.models}")
        return
    
    print(f"找到 {len(model_files)} 个模型")
    
    # 测试所有模型
    results = []
    for model_file in model_files:
        result = benchmark_model(model_file, args.warmup, args.iterations)
        if result:
            results.append(result)
    
    # 保存结果
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ 结果已保存到: {output_path}")
    else:
        print("\n❌ 没有成功的测试结果")

if __name__ == '__main__':
    main()
