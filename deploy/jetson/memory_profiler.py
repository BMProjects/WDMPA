#!/usr/bin/env python3
"""
内存占用测试 (Jetson优化)
测试每个模型加载后的内存使用情况
"""

import argparse
import csv
from pathlib import Path

def get_memory_usage():
    """获取当前内存使用情况 (MB)"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    except ImportError:
        # 如果psutil不可用，使用系统命令
        import subprocess
        try:
            result = subprocess.run(['ps', '-p', str(os.getpid()), '-o', 'rss='], 
                                  capture_output=True, text=True)
            return float(result.stdout.strip()) / 1024  # KB to MB
        except:
            return 0

def test_model_memory(model_path):
    """测试单个模型的内存占用"""
    import gc
    
    print(f"\n测试模型: {model_path.name}")
    
    # 记录加载前内存
    gc.collect()
    mem_before = get_memory_usage()
    print(f"  加载前内存: {mem_before:.1f} MB")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # 加载模型
        session = ort.InferenceSession(
            str(model_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 记录加载后内存
        mem_after = get_memory_usage()
        mem_model = mem_after - mem_before
        
        print(f"  加载后内存: {mem_after:.1f} MB")
        print(f"  模型占用: {mem_model:.1f} MB")
        
        # 准备输入并运行一次推理
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        batch_size = 1 if input_shape[0] in ['batch', None, -1] else input_shape[0]
        actual_shape = [batch_size] + list(input_shape[1:])
        dummy_input = np.random.randn(*actual_shape).astype(np.float32)
        
        # 运行推理
        _ = session.run(None, {input_name: dummy_input})
        
        # 记录推理后内存
        mem_inference = get_memory_usage()
        mem_runtime = mem_inference - mem_after
        
        print(f"  推理后内存: {mem_inference:.1f} MB")
        print(f"  运行时占用: {mem_runtime:.1f} MB")
        
        result = {
            'model': model_path.stem,
            'model_size_mb': f"{mem_model:.1f}",
            'runtime_mb': f"{mem_runtime:.1f}",
            'total_mb': f"{mem_model + mem_runtime:.1f}",
            'provider': session.get_providers()[0]
        }
        
        # 清理
        del session
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='模型内存占用测试')
    parser.add_argument('--models', required=True, help='ONNX模型目录')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型内存占用测试")
    print("=" * 60)
    
    # 获取模型列表
    models_path = Path(args.models)
    model_files = sorted(models_path.glob('*.onnx'))
    
    if not model_files:
        print(f"❌ 未找到ONNX模型: {args.models}")
        return
    
    print(f"找到 {len(model_files)} 个模型")
    
    # 测试所有模型
    results = []
    for model_file in model_files:
        result = test_model_memory(model_file)
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
    import os
    main()
